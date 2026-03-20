# AR Mirror - Copy-Paste Ready Setup

## Step 1: Prerequisites
Download and install:
- Python 3.8+ from python.org
- Node.js 16+ from nodejs.org
- Git from git-scm.com

## Step 2: Clone Repository
```
git clone https://github.com/redwolf261/AR_Mirror.git
```

```
cd AR_Mirror
```

## Step 3: Create Virtual Environment
```
python -m venv ar
```

Windows:
```
ar\Scripts\activate
```

macOS/Linux:
```
source ar/bin/activate
```

## Step 4: Install Python Dependencies

NVIDIA GPU users:
```
pip install --upgrade pip
```
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
```
pip install -r requirements.txt
```

CPU-only users:
```
pip install --upgrade pip
```
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
```
pip install -r requirements.txt
```

## Step 5: Setup Frontend
```
cd frontend/ar-mirror-ui
```
```
npm install
```
```
cd ../..
```

## Step 6: Run Application

Terminal 1 - Backend:
```
ar\Scripts\activate
```
```
python app.py
```

Terminal 2 - Frontend:
```
cd frontend/ar-mirror-ui
```
```
npm start
```

## Step 7: Access Application
Open browser to: http://localhost:3001

## Controls
- Arrow keys: Change garments
- d key: Toggle debug
- q key: Quit

## Quick Troubleshooting

Camera issues:
```
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

Dependencies issues:
```
pip install --upgrade -r requirements.txt
```

Check virtual environment:
```
pip list | findstr torch
```