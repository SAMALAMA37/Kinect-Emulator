# Webcam Kinect Emulator🔴📷

This is my personal project that emulates live kinect capabilities using only cheap webcam, it mimics fundamental kinect 
capabilties like smooth full body movement and Approximiate 3D Location.
It uses MediaPipe and Depth-Anything-V2 and makes them work simultaneously for both real-time pose estimation and monocular depth prediction to get the approximiate x,y,z data

- Important to note that it only tries to replicate the fundamental kinect functionality using only a webcam, and as a result, it does not achieve 100% accuracy like the original Kinect.

Average webcam speed is 10fps, if you have less than that (like 5), make sure you have CUDA installed, you can check this by running pre-prepared script called "test.py"



## Prerequisites

- **Operating System:** Windows, macOS, or Linux
- **Python:** Version 3.8 or higher (recommended: 3.10+)
- **Webcam:** Any standard webcam connected to your computer
- **(RECOMMENDED) GPU:** NVIDIA GPU with CUDA for faster inference

---

## 1. Clone the Repository

```bash
git clone https://github.com/SAMALAMA37/Kinect-Emulator.git
cd Kinect-Emulator
```

---

## 2. Install Requirements

Install all the required packages:
```
pip install opencv-python torch mediapipe pillow transformers numpy
```
if it gives you error about version downgrade your python to be below 3.13, it's the only version that's not supported

---

## 3. Set up Hugging Face CLI
(You need to log in to download the AI model)

To obtain the Huggin Face API token copy paste it from: "https://huggingface.co/settings/tokens", and then paste it into CLI using


```bash
huggingface-cli login
```

if the command doesn't work, install it using pip
```bash
pip install huggingface_hub
```

---

### 4. Install PyTorch with CUDA Support

For way Better Process Speed (10x) it's recommended to use CUDA along with NVIDIA GPU, here's how to install it

---

Visit this domain https://pytorch.org/get-started/locally/

Choose pip, your software, and CUDA version (should work on both).

Enter the given pip installation command
```bash
// Example:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 5. Run the Kinect Emulator

And finally, run the project from root directory using:

```bash
python kinect_tracker.py
```

---


Usage
---

- A window will appear showing simulated Rygid skeleton with All three-dimensional joints that contain approximate location (x, y, z)
- The default location showed in left up corner is Left Wrist (LW), you can personally change that in the code to contain the full body details on the feed.
- Press **Q** in the window to exit.


---

Troubleshooting
---

- **Cannot open camera:** Ensure your webcam is connected and not being used by another application.
- **CUDA errors:** If you do not have a compatible GPU, the script will fall back to CPU automatically (it will be slower).
- **ModuleNotFoundError:** Double-check that all dependencies are installed.


--- 

