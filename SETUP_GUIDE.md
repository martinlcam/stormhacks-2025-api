# Signable - Complete Setup Guide (New Team Member)

**Quick reference guide for setting up the entire Signable project from scratch on a new device.**

## 🎯 What is Signable?

Real-time American Sign Language (ASL) recognition system that:
1. Captures webcam video in the browser (Next.js frontend)
2. Streams video to Python backend via WebSocket
3. Uses MediaPipe + TensorFlow to recognize ASL signs
4. Returns predictions in real-time with 99% accuracy

---

## ⏱️ Estimated Setup Time

- **Total**: 30-45 minutes
- **Prerequisites**: 10 minutes
- **Backend**: 15 minutes
- **Frontend**: 5 minutes
- **Testing**: 5 minutes

---

## 📋 Prerequisites Checklist

Before starting, install these tools:

### Required Software

- [ ] **Python 3.12** (or 3.10-3.11)
  - Download: https://www.python.org/downloads/
  - ⚠️ Check "Add to PATH" during installation (Windows)
  
- [ ] **Node.js 20+**
  - Download: https://nodejs.org/ (LTS version)
  - Includes npm
  
- [ ] **pnpm** (package manager)
  ```bash
  npm install -g pnpm
  ```
  
- [ ] **FFmpeg**
  - Windows: https://ffmpeg.org/download.html
    - Extract to `C:\FFmpeg`
    - Add `C:\FFmpeg\bin` to system PATH
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Verify: `ffmpeg -version`
  
- [ ] **Git**
  - Download: https://git-scm.com/downloads
  
- [ ] **Webcam**
  - Built-in or USB webcam
  - Test in browser: https://webcamtests.com/

### Verify Installations

```bash
# Check all tools are installed
python --version    # Should be 3.10, 3.11, or 3.12
node --version      # Should be 20.x or higher
pnpm --version      # Should be 10.x or higher
ffmpeg -version     # Should show FFmpeg version
git --version       # Should show Git version
```

---

## 🔧 Part 1: Backend Setup (Python + TensorFlow)

### Step 1: Clone Repository

```bash
# Clone the backend repository
git clone https://github.com/martinlcam/stormhacks-2025-api.git
cd stormhacks-2025-api
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Python Dependencies

⚠️ **Critical**: Use these exact versions to avoid compatibility issues!

```bash
# Upgrade pip first
pip install --upgrade pip

# Install core dependencies with specific versions
pip install tensorflow==2.16.1
pip install keras==3.0.0
pip install numpy==1.26.4
pip install protobuf==4.25.8
pip install opencv-python==4.9.0.80
pip install mediapipe==0.10.21
pip install websockets==15.0.1

# Install other dependencies
pip install scikit-learn scikit-image pandas matplotlib seaborn
```

**Expected time**: 5-10 minutes (downloads ~2GB)

### Step 4: Download ASL Model

1. Go to: https://www.kaggle.com/datasets/namanmanchanda/american-sign-language-model-99-accuracy
2. Click "Download" (requires Kaggle account - free)
3. Extract the ZIP file
4. Copy `ASL.h5` to `models/` folder:
   ```
   stormhacks-2025-api/
   └── models/
       └── ASL.h5
   ```

### Step 5: Verify Backend Setup

```bash
# Check dependencies
pip check

# Should show no conflicts
```

### Step 6: Test Backend Server

```bash
# Start the server
python asl_pipeline.py
```

**Expected output:**
```
INFO:__main__:ASL model loaded from models/ASL.h5
INFO:__main__:Initializing MediaPipe...
INFO:__main__:Downloading MediaPipe hand landmark model...
INFO:__main__:Using legacy MediaPipe API
INFO:websockets.server:server listening on [::]:4000
INFO:__main__:WebSocket server started on ws://localhost:4000
```

✅ **Backend is ready!** Leave this terminal running.

---

## 🌐 Part 2: Frontend Setup (Next.js + React)

### Step 1: Clone Frontend Repository

Open a **new terminal** (keep backend running):

```bash
# Navigate to parent directory
cd ..

# Clone frontend repository
git clone https://github.com/[your-username]/stormhacks-2025-web.git
cd stormhacks-2025-web
```

### Step 2: Install Frontend Dependencies

```bash
pnpm install
```

**Expected time**: 1-2 minutes

### Step 3: Start Development Server

```bash
pnpm dev
```

**Expected output:**
```
▲ Next.js 15.5.4
- Local:        http://localhost:3000
- Network:      http://192.168.x.x:3000
```

✅ **Frontend is ready!**

---

## 🎬 Part 3: Testing the Complete System

### Step 1: Open Application

1. Open browser to: **http://localhost:3000/stream-preview**
2. You should see the stream preview page

### Step 2: Start Streaming

1. Click **"Start Streaming"** button
2. Allow camera permissions when prompted
3. You should see your webcam feed

### Step 3: View Predictions

1. Open browser console (F12 → Console tab)
2. You should see messages like:
   ```
   WS message: {"type":"asl_batch","predicted_sign":"A","confidence":0.95,...}
   ```

### Step 4: Check Backend Logs

In the backend terminal, you should see:
```
INFO:websockets.server:connection open
INFO:__main__:Client connected: ('::1', 62216, 0, 0) path=/ws/stream/
INFO:__main__:Starting stream handler
INFO:__main__:Predicted: A (confidence: 0.950)
INFO:__main__:Predicted: B (confidence: 0.876)
```

### Step 5: Test ASL Recognition

1. Make ASL hand signs in front of camera
2. Check predictions in console and backend logs
3. Verify high confidence (>0.8) for clear signs

✅ **System is working!**

---

## 🐛 Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**: Virtual environment not activated
```bash
# Windows
.\venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate
```

### Issue: `FileNotFoundError: models/ASL.h5`

**Solution**: Model file missing
1. Download from Kaggle (link above)
2. Place in `models/ASL.h5`

### Issue: `FFmpeg not found`

**Solution**: FFmpeg not in PATH
```bash
# Windows: Add C:\FFmpeg\bin to PATH
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Verify
ffmpeg -version
```

### Issue: `Address already in use` (Port 4000)

**Solution**: Kill existing process
```powershell
# Windows
Get-Process python | Stop-Process -Force

# Mac/Linux
lsof -ti:4000 | xargs kill -9
```

### Issue: Camera not working

**Solution**:
1. Check browser permissions
2. Try Chrome or Firefox
3. Close other apps using camera
4. Use HTTPS or localhost

### Issue: `protobuf version conflict`

**Solution**: Install exact versions
```bash
pip install protobuf==4.25.8 tensorflow==2.16.1
```

---

## 📂 Final Directory Structure

After successful setup:

```
Projects/
├── stormhacks-2025-api/          # Backend
│   ├── venv/                     # Python virtual environment
│   ├── models/
│   │   ├── ASL.h5                # ✅ ASL model file
│   │   └── hand_landmarker.task  # ✅ Auto-downloaded
│   ├── asl_pipeline.py           # Main server
│   └── requirements.txt
│
└── stormhacks-2025-web/          # Frontend
    ├── node_modules/             # Node dependencies
    ├── app/
    │   └── stream-preview/       # Main streaming page
    ├── package.json
    └── next.config.ts
```

---

## ✅ Setup Verification Checklist

- [ ] Python 3.12 installed
- [ ] Node.js 20+ installed
- [ ] pnpm installed
- [ ] FFmpeg installed and in PATH
- [ ] Backend repository cloned
- [ ] Frontend repository cloned
- [ ] Python virtual environment created
- [ ] Python dependencies installed (no conflicts)
- [ ] ASL.h5 model downloaded and placed in `models/`
- [ ] Frontend dependencies installed
- [ ] Backend server starts successfully (port 4000)
- [ ] Frontend server starts successfully (port 3000)
- [ ] WebSocket connection established
- [ ] Camera permissions granted
- [ ] Predictions appearing in console
- [ ] Backend logs showing predictions

---

## 🚀 Quick Start Commands (After Setup)

### Start Backend
```bash
cd stormhacks-2025-api
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Mac/Linux
python asl_pipeline.py
```

### Start Frontend
```bash
cd stormhacks-2025-web
pnpm dev
```

### Access Application
- Frontend: http://localhost:3000
- Stream Page: http://localhost:3000/stream-preview
- Backend: http://localhost:4000 (WebSocket)

---

## 📚 Additional Resources

- **Backend README**: [stormhacks-2025-api/README.md](README.md)
- **Frontend README**: [stormhacks-2025-web/README.md](../stormhacks-2025-web/README.md)
- **Integration Guide**: [INTEGRATION.md](INTEGRATION.md)
- **ASL Dataset**: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **MediaPipe Docs**: https://developers.google.com/mediapipe

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review backend/frontend README files
3. Check terminal/console for error messages
4. Verify all prerequisites are installed correctly
5. Contact team lead or post in team chat

---

## 🎓 Understanding the System

### Data Flow

```
User's Camera
    ↓
MediaRecorder (browser)
    ↓
WebSocket → Backend (Python)
    ↓
FFmpeg (extract frames)
    ↓
MediaPipe (detect hands)
    ↓
ASL Model (classify sign)
    ↓
WebSocket → Browser Console
```

### Key Files

**Backend:**
- `asl_pipeline.py` - Main WebSocket server
- `models/ASL.h5` - Trained ASL recognition model
- `models/hand_landmarker.task` - MediaPipe hand detector

**Frontend:**
- `app/stream-preview/stream-preview.tsx` - Webcam capture component
- `app/stream-preview/page.tsx` - Stream page route

---

**Setup complete! You're ready to contribute to Signable! 🎉**

For questions or issues, reach out to the team lead.

---

**Last updated**: October 2025  
**Maintained by**: StormHacks 2025 Team - Signable
