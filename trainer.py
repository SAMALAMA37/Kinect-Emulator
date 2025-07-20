## This is the script for finetuning the model, you don't have to run it btw

import os
import warnings
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import h5py
from transformers import AutoModelForDepthEstimation

warnings.filterwarnings("ignore")

@dataclass
class Config:
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    mat_file_path: str = "./nyu_depth_v2_labeled.mat"
    num_frames: int = 3
    image_size: Tuple[int, int] = (518, 518)
    batch_size: int = 2  # Reduced for stability
    num_epochs: int = 10
    learning_rate: float = 5e-6
    weight_decay: float = 1e-4
    num_workers: int = 0  # Set to 0 for debugging
    gradient_clip_val: float = 0.5  # Reduced
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path: str = './best_depth_model.pth'
    checkpoint_path: str = './checkpoint.pth'
    train_split: float = 0.9

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Scale predictions to match target range
        pred_scaled = pred * 10.0  # Depth Anything outputs are typically in [0, 1] range
        return self.l1(pred_scaled, target)

class TemporalDepthModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Load pretrained model
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(config.model_id)
        
        # Freeze backbone
        for name, param in self.depth_model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
        
        # Temporal fusion: simple weighted average
        self.temporal_weights = nn.Parameter(torch.ones(config.num_frames))
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = images.shape
        
        # Process each frame
        depth_maps = []
        for t in range(T):
            frame = images[:, t]  # (B, C, H, W)
            
            # Forward through depth model
            with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision
                output = self.depth_model(frame)
                depth = output.predicted_depth if hasattr(output, 'predicted_depth') else output
            
            # Ensure correct shape
            if depth.dim() == 4:
                depth = depth.squeeze(1)
            
            depth_maps.append(depth)
        
        # Stack and apply temporal fusion
        depth_stack = torch.stack(depth_maps, dim=1)  # (B, T, H, W)
        
        # Normalize weights
        weights = torch.softmax(self.temporal_weights, dim=0)
        weights = weights.view(1, -1, 1, 1)
        
        # Weighted average
        fused_depth = (depth_stack * weights).sum(dim=1)
        
        return fused_depth

class NYUDataset(Dataset):
    def __init__(self, config: Config, split: str = 'train'):
        self.config = config
        self.split = split
        
        # Load dataset info
        with h5py.File(config.mat_file_path, 'r') as f:
            self.total_samples = f['images'].shape[0]
        
        # Split indices
        split_idx = int(self.total_samples * config.train_split)
        if split == 'train':
            self.indices = list(range(split_idx))
        else:
            self.indices = list(range(split_idx, self.total_samples))
        
        # Transforms
        self.transform = T.Compose([
            T.Resize(config.image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.depth_transform = T.Compose([
            T.Resize(config.image_size, antialias=True),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.indices) - self.config.num_frames + 1
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        with h5py.File(self.config.mat_file_path, 'r') as f:
            # Load frames
            frames = []
            for i in range(self.config.num_frames):
                frame_idx = min(start_idx + i, self.total_samples - 1)
                
                # Load image
                img_data = f['images'][frame_idx]
                img_data = np.transpose(img_data, (1, 2, 0))  # to (H, W, C)
                
                # Convert to uint8
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                else:
                    img_data = img_data.astype(np.uint8)
                
                img = Image.fromarray(img_data)
                frames.append(self.transform(img))
            
            # Load depth for middle frame
            middle_idx = start_idx + self.config.num_frames // 2
            depth_data = f['depths'][middle_idx].astype(np.float32)
            
            # Convert to meters if needed
            if depth_data.max() > 100:
                depth_data = depth_data / 1000.0
            
            # Clip to valid range
            depth_data = np.clip(depth_data, 0.1, 10.0)
            
            depth_img = Image.fromarray(depth_data, mode='F')
            depth_tensor = self.depth_transform(depth_img).squeeze(0)
        
        # Stack frames
        frames_tensor = torch.stack(frames)  # (T, C, H, W)
        
        return {
            'frames': frames_tensor,
            'depth': depth_tensor
        }

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = TemporalDepthModel(config).to(config.device)
        
        # Datasets
        self.train_dataset = NYUDataset(config, 'train')
        self.val_dataset = NYUDataset(config, 'val')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.criterion = SimpleLoss()
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        for batch in pbar:
            frames = batch['frames'].to(self.config.device)
            depth_gt = batch['depth'].to(self.config.device)
            
            # Forward pass
            depth_pred = self.model(frames)
            
            # Compute loss
            loss = self.criterion(depth_pred, depth_gt)
            
            # Skip if NaN
            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_val
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            frames = batch['frames'].to(self.config.device)
            depth_gt = batch['depth'].to(self.config.device)
            
            depth_pred = self.model(frames)
            loss = self.criterion(depth_pred, depth_gt)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self):
        print(f"Starting training...")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.best_model_path)
                print(f"Saved best model with val loss: {val_loss:.4f}")

def main():
    config = Config()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
