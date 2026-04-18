# AR Mirror Setup Instructions

## Prerequisites

### 1. Install Python 3.8+
- Download from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"
- Verify installation: `python --version`

### 2. Install Node.js 16+
- Download from [nodejs.org](https://nodejs.org/)
- Verify installation: `node --version` and `npm --version`

### 3. Install Git (if not already installed)
- Download from [git-scm.com](https://git-scm.com/)

### 4. GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: Install [CUDA Toolkit 11.2+](https://developer.nvidia.com/cuda-downloads)
- **AMD GPU**: Install [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) (Linux only)
- **CPU Only**: Will work but slower performance

## Setup Instructions

### Step 1: Clone and Navigate to Repository
```bash
git clone [repository-url]
cd "AR Mirror"
```

### Step 2: Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv ar

# Activate virtual environment
# Windows:
ar\Scripts\activate
# macOS/Linux:
source ar/bin/activate
```

### Step 3: Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# IMPORTANT: Install PyTorch with CUDA first (if you have NVIDIA GPU)
# For NVIDIA GPU with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For CPU only (slower):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from requirements file
pip install -r requirements.txt
```

**Note**: The project includes optimized dependencies for CUDA 12.x. If you have an older GPU or different CUDA version, you may need to adjust the PyTorch installation URL.

### Step 4: Set Up Frontend
```bash
# Navigate to frontend directory
cd "frontend/ar-mirror-ui"

# Install Node.js dependencies
npm install

# Return to root directory
cd "../.."
```

### Step 5: Create Required Directories
```bash
# Create necessary directories
mkdir -p models
mkdir -p logs
mkdir -p dataset/train/cloth
mkdir -p dataset/train/image
```

### Step 6: Download Required Models
```bash
# Download MediaPipe models (these will download automatically on first run)
# The app will handle this, but you can pre-download:

# Create a simple test to trigger model downloads
python -c "import mediapipe as mp; mp.solutions.pose.Pose()"
```

### Step 7: Add Sample Garment Images
```bash
# Add some sample garment images to dataset/train/cloth/
# You can use any clothing images (PNG/JPG format)
# Examples: shirt.jpg, dress.png, jacket.jpg, etc.
```

## Running the Application

### Step 1: Start Backend (Terminal 1)
```bash
# Make sure virtual environment is activated
ar\Scripts\activate  # Windows
# source ar/bin/activate  # macOS/Linux

# Start the Python backend
python app.py
```

You should see:
```
✓ GPU optimizations enabled (if GPU available)
✓ AR Mirror initialization complete
✓ Web server started → http://localhost:5050
✓ Camera detected and working
```

### Step 2: Start Frontend (Terminal 2 - New Terminal)
```bash
# Navigate to frontend directory
cd "frontend/ar-mirror-ui"

# Start the React development server
npm start
```

The frontend will open automatically at `http://localhost:3001`

### Step 3: Usage
1. **Allow camera access** when prompted by your browser
2. **Position yourself** in front of the camera (full body visible works best)
3. **Wait for body detection** - you'll see measurements appear
4. **Try different garments** using arrow keys or UI controls
5. **View size recommendations** in the measurements panel

## Controls
- **Arrow Keys**: Change garments
- **'d' Key**: Toggle debug overlay (skeleton view)
- **'o' Key**: Toggle info overlay
- **'q' Key**: Quit application

## Troubleshooting

### Camera Issues
**Problem**: "Camera not found" or "Camera access denied"
**Solutions**:
- Check camera permissions in browser settings
- Close other applications using the camera (Zoom, Skype, etc.)
- Try different browsers (Chrome recommended)
- On Windows: Try running as administrator

### GPU Issues
**Problem**: Low FPS or "CUDA not available"
**Solutions**:
- Verify GPU drivers are up to date
- Install correct CUDA version for your GPU
- CPU-only mode will work (slower but functional)

### Missing Dependencies
**Problem**: Import errors or missing modules
**Solutions**:
```bash
# Reinstall PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then reinstall other dependencies
pip install --upgrade -r requirements.txt

# Check if virtual environment is activated
# Windows: ar\Scripts\activate
# Linux/Mac: source ar/bin/activate
```

### Port Conflicts
**Problem**: "Port already in use"
**Solutions**:
- Kill existing processes using ports 5050/3001
- Change ports in configuration files if needed

### Performance Issues
**Problem**: Low FPS or lag
**Solutions**:
- Close unnecessary applications
- Reduce camera resolution in code if needed
- Ensure good lighting for better detection
- Use GPU acceleration if available

## System Requirements

### Minimum Requirements
- **CPU**: Intel i5 4th gen / AMD Ryzen 3
- **RAM**: 8GB
- **Storage**: 5GB free space
- **Camera**: Any USB/built-in webcam
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements
- **CPU**: Intel i7 8th gen / AMD Ryzen 5
- **RAM**: 16GB
- **GPU**: NVIDIA GTX 1060+ / RTX series
- **Storage**: 10GB free space (SSD preferred)
- **Camera**: HD webcam (1080p)

## Project Structure
```
AR Mirror/
├── app.py                 # Main backend application
├── web_server.py         # Flask web server
├── src/                  # Core modules
│   ├── core/            # Body detection & size recommendation
│   └── app/             # Rendering and processing
├── frontend/            # React UI
│   └── ar-mirror-ui/
├── dataset/             # Garment images
├── models/              # ML models (auto-downloaded)
└── logs/                # Application logs
```

## Need Help?

### Check Logs
- Backend logs: Check terminal output and `logs/` directory
- Frontend logs: Check browser developer console (F12)

### Common Issues
1. **No size recommendations showing**: Ensure good lighting and full body visibility
2. **Garment not fitting well**: Try different poses or camera angles
3. **App crashes**: Check for missing dependencies or insufficient RAM

### Contact
If you encounter issues not covered here, check the project documentation or contact the developer.

---

🚀 **You're all set! Enjoy trying on virtual garments with AR Mirror!**