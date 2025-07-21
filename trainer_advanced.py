import os
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import h5py
from transformers import AutoModelForDepthEstimation, get_scheduler

warnings.filterwarnings("ignore")

@dataclass
class Config:
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    mat_file_path: str = "./nyu_depth_v2_labeled.mat"
    num_frames: int = 3
    image_size: Tuple[int, int] = (518, 518)
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    num_workers: int = 2
    gradient_clip_val: float = 1.0
    warmup_steps: int = 100
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path: str = './best_depth_model.pth'
    checkpoint_path: str = './checkpoint.pth'
    train_split: float = 0.9
    # Performance optimizations
    pin_memory: bool = True
    persistent_workers: bool = True
    compile_model: bool = False  # Disable by default due to Triton dependency
    # PRC settings
    prc_stages: int = 4  # Number of progressive stages
    prc_channels: int = 32  # Base channels for PRC
    prc_probability: float = 0.5  # Probability of applying PRC

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

class ProgressiveRandomConvolution(nn.Module):
    """Progressive Random Convolution module that adds controlled noise to features"""
    def __init__(self, channels: int, stages: int = 4, base_channels: int = 32):
        super().__init__()
        self.stages = stages
        self.current_stage = 0
        self.channels = channels
        
        # Create random convolution layers for each stage
        self.random_convs = nn.ModuleList()
        for i in range(stages):
            # Progressive channel sizes
            stage_channels = base_channels * (i + 1)
            
            # Random conv layer (weights are fixed after initialization)
            conv = nn.Conv2d(channels, stage_channels, 
                           kernel_size=3 + 2*i,  # Increasing kernel sizes: 3, 5, 7, 9
                           padding=(3 + 2*i) // 2,
                           bias=False)
            
            # Initialize with random weights and freeze
            nn.init.normal_(conv.weight, mean=0, std=0.02)
            for param in conv.parameters():
                param.requires_grad = False
                
            self.random_convs.append(conv)
            
        # Fusion layers to combine random features with original
        self.fusion_convs = nn.ModuleList()
        for i in range(stages):
            stage_channels = base_channels * (i + 1)
            fusion = nn.Sequential(
                nn.Conv2d(channels + stage_channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.fusion_convs.append(fusion)
            
        # Stage-wise attention to control contribution
        # Fix: Ensure minimum channels for attention mechanism
        self.stage_attention = nn.ModuleList()
        for i in range(stages):
            # Ensure at least 1 channel for the intermediate layer
            intermediate_channels = max(1, channels // 4)
            attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, intermediate_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(intermediate_channels, 1, 1),
                nn.Sigmoid()
            )
            self.stage_attention.append(attention)
    
    def set_stage(self, epoch: int, total_epochs: int):
        """Update current stage based on training progress"""
        progress = epoch / total_epochs
        self.current_stage = min(int(progress * self.stages), self.stages - 1)
    
    def forward(self, x: torch.Tensor, probability: float = 0.5) -> torch.Tensor:
        # Skip PRC with given probability during training
        if not self.training or torch.rand(1).item() > probability:
            return x
            
        # Apply progressive random convolutions up to current stage
        output = x
        for i in range(self.current_stage + 1):
            # Get random features
            random_features = self.random_convs[i](x)
            
            # Concatenate with original features
            combined = torch.cat([output, random_features], dim=1)
            
            # Fuse features
            fused = self.fusion_convs[i](combined)
            
            # Apply stage-wise attention
            attention = self.stage_attention[i](output)
            
            # Weighted combination
            output = output + attention * fused
            
        return output

class DepthLoss(nn.Module):
    """Optimized combined depth loss"""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure positive predictions
        pred = F.relu(pred) + 1e-3
        
        # Primary losses
        l1_loss = self.l1(pred, target)
        l2_loss = torch.sqrt(self.mse(pred, target) + 1e-6)
        
        # Efficient gradient computation using conv2d
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        pred_grad_x = F.conv2d(pred.unsqueeze(1), sobel_x, padding=1).squeeze(1)
        pred_grad_y = F.conv2d(pred.unsqueeze(1), sobel_y, padding=1).squeeze(1)
        target_grad_x = F.conv2d(target.unsqueeze(1), sobel_x, padding=1).squeeze(1)
        target_grad_y = F.conv2d(target.unsqueeze(1), sobel_y, padding=1).squeeze(1)
        
        grad_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        
        return l1_loss + 0.1 * l2_loss + 0.1 * grad_loss

class OptimizedTemporalAttention(nn.Module):
    """More efficient temporal attention with reduced parameters"""
    def __init__(self, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
        
        # More efficient architecture
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Pool spatial dimensions
        self.attention = nn.Sequential(
            nn.Linear(num_frames, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_frames),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """
        depth_maps: (B, T, H, W)
        Returns: (B, T) attention weights
        """
        B, T = depth_maps.shape[:2]
        
        # Global average pooling for each frame
        pooled = self.global_pool(depth_maps.view(B * T, 1, *depth_maps.shape[2:]))
        pooled = pooled.view(B, T).squeeze(-1).squeeze(-1)  # (B, T)
        
        # Compute attention weights
        weights = self.attention(pooled)
        
        return weights

class TemporalDepthModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Load pretrained model
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(config.model_id)
        
        # More selective freezing - freeze backbone except last 2 layers
        for name, param in self.depth_model.named_parameters():
            if 'backbone' in name and not any(layer in name for layer in ['layer.10', 'layer.11']):
                param.requires_grad = False
                
        # Progressive Random Convolution modules
        # Add PRC after different stages of the model
        self.prc_early = ProgressiveRandomConvolution(
            channels=384,  # Early feature channels (adjust based on model)
            stages=config.prc_stages,
            base_channels=config.prc_channels
        )
        
        self.prc_mid = ProgressiveRandomConvolution(
            channels=192,  # Mid-level feature channels
            stages=config.prc_stages,
            base_channels=config.prc_channels // 2
        )
        
        # Adjusted PRC for single-channel depth maps
        self.prc_late = ProgressiveRandomConvolution(
            channels=1,  # Final depth map
            stages=2,    # Reduce stages for single channel
            base_channels=4  # Reduce base channels for efficiency
        )
                
        # Optimized temporal attention
        self.temporal_attention = OptimizedTemporalAttention(config.num_frames)
        
        # Simplified fusion weights
        self.register_buffer('base_weights', torch.ones(config.num_frames) / config.num_frames)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))  # Learnable mixing coefficient
        
        # Lighter output refinement
        self.output_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def update_prc_stage(self, epoch: int):
        """Update PRC stages based on training progress"""
        self.prc_early.set_stage(epoch, self.config.num_epochs)
        self.prc_mid.set_stage(epoch, self.config.num_epochs)
        self.prc_late.set_stage(epoch, self.config.num_epochs)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = images.shape
        
        # Process frames in batch for efficiency
        images_flat = images.view(B * T, C, H, W)
        
        # Hook to apply PRC at different stages
        def apply_prc_hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                # Apply early PRC to backbone features
                if output.shape[1] == 384:  # Adjust based on your model
                    return self.prc_early(output, self.config.prc_probability)
                elif output.shape[1] == 192:  # Mid-level features
                    return self.prc_mid(output, self.config.prc_probability)
            return output
        
        # Register hooks for PRC during training
        hooks = []
        if self.training:
            for name, module in self.depth_model.named_modules():
                if 'layer.8' in name or 'layer.10' in name:  # Apply to specific layers
                    hook = module.register_forward_hook(apply_prc_hook)
                    hooks.append(hook)
        
        # Single forward pass through depth model
        with torch.cuda.amp.autocast(enabled=False):
            outputs = self.depth_model(images_flat)
            depths = outputs.predicted_depth if hasattr(outputs, 'predicted_depth') else outputs
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Reshape back to temporal dimension
        if depths.dim() == 4:
            depths = depths.squeeze(1)
        depths = depths.view(B, T, H, W)
        
        # Apply PRC to depth predictions
        if self.training:
            depths_prc = []
            for t in range(T):
                depth_t = depths[:, t:t+1, :, :]  # Keep channel dimension
                depth_prc = self.prc_late(depth_t, self.config.prc_probability)
                depths_prc.append(depth_prc.squeeze(1))
            depths = torch.stack(depths_prc, dim=1)
        
        # Compute attention weights
        attention_weights = self.temporal_attention(depths.detach())
        
        # Mix attention weights with base weights
        alpha = torch.sigmoid(self.fusion_alpha)
        combined_weights = alpha * attention_weights + (1 - alpha) * self.base_weights
        
        # Apply weights efficiently
        weighted_depth = torch.sum(depths * combined_weights.unsqueeze(-1).unsqueeze(-1), dim=1)
        
        # Refine output
        refined_depth = self.output_conv(weighted_depth.unsqueeze(1)).squeeze(1)
        
        return refined_depth
        

class NYUDataset(Dataset):
    def __init__(self, config: Config, split: str = 'train'):
        self.config = config
        self.split = split
        
        # Cache dataset info
        with h5py.File(config.mat_file_path, 'r') as f:
            self.total_samples = f['images'].shape[0]
        
        # Split indices
        split_idx = int(self.total_samples * config.train_split)
        self.indices = list(range(split_idx)) if split == 'train' else list(range(split_idx, self.total_samples))
        
        # Pre-compile transforms for efficiency
        base_transforms = [
            T.Resize(config.image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if split == 'train':
            self.transform = T.Compose([T.RandomHorizontalFlip(p=0.5)] + base_transforms)
        else:
            self.transform = T.Compose(base_transforms)
        
        self.depth_transform = T.Compose([
            T.Resize(config.image_size, antialias=True),
            T.ToTensor()
        ])
    
    def __len__(self):
        return max(0, len(self.indices) - self.config.num_frames + 1)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        with h5py.File(self.config.mat_file_path, 'r') as f:
            frames = []
            apply_flip = self.split == 'train' and np.random.rand() > 0.5
            
            for i in range(self.config.num_frames):
                frame_idx = min(start_idx + i, self.total_samples - 1)
                
                # Load and process image
                img_data = f['images'][frame_idx]
                img_data = np.transpose(img_data, (1, 2, 0))
                
                # Efficient conversion to uint8
                img_data = np.clip(img_data * 255 if img_data.max() <= 1.0 else img_data, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_data)
                
                if apply_flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                frames.append(self.transform(img))
            
            # Load depth for middle frame
            middle_idx = start_idx + self.config.num_frames // 2
            depth_data = f['depths'][middle_idx].astype(np.float32)
            
            # Normalize depth
            if depth_data.max() > 100:
                depth_data /= 1000.0
            depth_data = np.clip(depth_data, 0.1, 10.0)
            
            depth_img = Image.fromarray(depth_data, mode='F')
            if apply_flip:
                depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
                
            depth_tensor = self.depth_transform(depth_img).squeeze(0)
        
        return {
            'frames': torch.stack(frames),
            'depth': depth_tensor
        }

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = TemporalDepthModel(config).to(config.device)
        
        # Compile model for better performance (optional)
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, backend="inductor")
                print("Model compiled successfully")
            except Exception as e:
                print(f"Model compilation failed: {e}. Continuing without compilation...")
                # Reset to non-compiled model
                self.model = TemporalDepthModel(config).to(config.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"PRC stages: {config.prc_stages}")
        print(f"PRC probability: {config.prc_probability}")
        
        # Datasets with optimized DataLoader settings
        self.train_dataset = NYUDataset(config, 'train')
        self.val_dataset = NYUDataset(config, 'val')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers
        )
        
        # Optimizer with better defaults
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        self.criterion = DepthLoss()
        self.best_val_loss = float('inf')
        self.current_epoch = 0
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.current_epoch = epoch
        
        # Update PRC stage
        self.model.update_prc_stage(epoch)
        print(f"PRC Stage: {self.model.prc_early.current_stage + 1}/{self.config.prc_stages}")
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch in pbar:
            frames = batch['frames'].to(self.config.device, non_blocking=True)
            depth_gt = batch['depth'].to(self.config.device, non_blocking=True)
            
            # Forward pass
            depth_pred = self.model(frames)
            loss = self.criterion(depth_pred, depth_gt)
            
            # Skip invalid losses
            if not torch.isfinite(loss):
                print("Invalid loss detected, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)  # More efficient
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            frames = batch['frames'].to(self.config.device, non_blocking=True)
            depth_gt = batch['depth'].to(self.config.device, non_blocking=True)
            
            depth_pred = self.model(frames)
            loss = self.criterion(depth_pred, depth_gt)
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self):
        print("Starting training...")
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
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss
            }, self.config.checkpoint_path)
        
        print(f"Training completed! Best val loss: {self.best_val_loss:.4f}")

def main():
    config = Config()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
