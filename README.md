# Signable - Real-time ASL Recognition System

Real-time American Sign Language (ASL) recognition system that converts sign language gestures into text using computer vision and deep learning. Built for StormHacks 2025.

## 🎯 Features

- **Real-time ASL Recognition**: Processes live video streams from webcam
- **High Accuracy**: 99% accuracy ASL model recognizing 29 signs (A-Z + space/delete/nothing)
- **WebSocket Streaming**: Low-latency video processing pipeline
- **MediaPipe Integration**: Hand landmark detection for accurate gesture recognition
- **Batch Processing**: Efficient frame batching for stable predictions

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [API Documentation](#api-documentation)

## 💻 System Requirements

### Required Software
- **Python**: 3.10, 3.11, or 3.12 (recommended: 3.12)
- **Node.js**: 20.x or higher
- **pnpm**: 10.x or higher
- **FFmpeg**: Latest version ([Download here](https://ffmpeg.org/download.html))
- **Git**: For cloning the repository

### Operating Systems
- Windows 10/11
- macOS 12+
- Linux (Ubuntu 20.04+)

### Hardware Recommendations
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: Multi-core processor (4+ cores recommended)
- **Webcam**: Required for live ASL recognition
- **GPU**: Optional (NVIDIA GPU with CUDA support for faster processing)

## 🚀 Quick Start

### 1. Clone the Repositories

```bash
# Clone backend
git clone https://github.com/martinlcam/stormhacks-2025-api.git
cd stormhacks-2025-api

# Clone frontend (in a separate directory)
cd ..
git clone https://github.com/[your-username]/stormhacks-2025-web.git
cd stormhacks-2025-web
```

### 2. Install FFmpeg

**Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to `C:\FFmpeg`
3. Add `C:\FFmpeg\bin` to your system PATH
4. Verify installation:
   ```powershell
   ffmpeg -version
   ```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

## 🔧 Backend Setup

### 1. Navigate to Backend Directory
```bash
cd stormhacks-2025-api
```

### 2. Create Virtual Environment

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

### 3. Install Python Dependencies

⚠️ **Important**: Use these specific versions to avoid compatibility issues:

```bash
pip install --upgrade pip
pip install tensorflow==2.16.1 keras==3.0.0 numpy==1.26.4 protobuf==4.25.8
pip install opencv-python==4.9.0.80 mediapipe==0.10.21
pip install websockets==15.0.1
pip install scikit-learn scikit-image pandas matplotlib seaborn
```

Or install from requirements.txt (after ensuring compatible versions):
```bash
pip install -r requirements.txt
```

### 4. Download the ASL Model

**Option A: Pre-trained Model (Recommended)**

1. Download from [Kaggle](https://www.kaggle.com/datasets/namanmanchanda/american-sign-language-model-99-accuracy)
2. Extract `ASL.h5` file
3. Place in `models/` directory:
   ```
   stormhacks-2025-api/
   └── models/
       └── ASL.h5
   ```

**Option B: Train from Scratch** (Not recommended for quick setup)
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. Extract to `data/asl-alphabet/`
3. Run training script: `python alphabet.py`

### 5. Verify MediaPipe Model

The system will automatically download the MediaPipe hand landmark model on first run. Verify it exists:
```
stormhacks-2025-api/
└── models/
    ├── ASL.h5
    └── hand_landmarker.task
```

## 🌐 Frontend Setup

### 1. Navigate to Frontend Directory
```bash
cd stormhacks-2025-web
```

### 2. Install pnpm (if not already installed)

**Windows/macOS/Linux:**
```bash
npm install -g pnpm
```

### 3. Install Node Dependencies
```bash
pnpm install
```

This will install:
- Next.js 15.5.4
- React 19.1.0
- TypeScript 5.x
- Tailwind CSS 4.x
- ESLint

## 🎬 Running the Application

### Start Backend Server

1. Open a terminal in the backend directory
2. Activate virtual environment:
   ```powershell
   # Windows
   .\venv\Scripts\Activate.ps1
   
   # macOS/Linux
   source venv/bin/activate
   ```
3. Start the WebSocket server:
   ```bash
   python asl_pipeline.py
   ```
4. You should see:
   ```
   INFO:__main__:ASL model loaded from models/ASL.h5
   INFO:__main__:Initializing MediaPipe...
   INFO:__main__:Using legacy MediaPipe API
   INFO:websockets.server:server listening on [::]:4000
   INFO:__main__:WebSocket server started on ws://localhost:4000
   ```

### Start Frontend Development Server

1. Open a **new terminal** in the frontend directory
2. Start the Next.js dev server:
   ```bash
   pnpm dev
   ```
3. You should see:
   ```
   ▲ Next.js 15.5.4
   - Local:        http://localhost:3000
   ```

### Access the Application

1. Open your browser to **http://localhost:3000**
2. Navigate to the **Stream Preview** page: **http://localhost:3000/stream-preview**
3. Click **"Start Streaming"** to begin ASL recognition
4. Allow camera permissions when prompted
5. View predictions in the browser console (F12 → Console tab)

## 📁 Project Structure

### Backend (stormhacks-2025-api/)
```
stormhacks-2025-api/
├── models/
│   ├── ASL.h5                    # Trained ASL model (99% accuracy)
│   └── hand_landmarker.task      # MediaPipe hand detection model
├── data/
│   └── asl-alphabet/             # Training dataset (optional)
├── asl_pipeline.py               # Main WebSocket server & ASL pipeline
├── alphabet.py                   # Model training script
├── mediapipe.py                  # MediaPipe utilities
├── test_pipeline.py              # Testing script
├── requirements.txt              # Python dependencies
├── INTEGRATION.md                # Integration documentation
└── README.md                     # This file
```

### Frontend (stormhacks-2025-web/)
```
stormhacks-2025-web/
├── app/
│   ├── components/
│   │   └── Header.tsx            # Navigation header
│   ├── stream-preview/
│   │   ├── page.tsx              # Stream preview page
│   │   └── stream-preview.tsx    # Webcam streaming component
│   ├── layout.tsx                # Root layout
│   ├── page.tsx                  # Home page
│   └── globals.css               # Global styles
├── package.json                  # Node dependencies
├── tsconfig.json                 # TypeScript config
├── next.config.ts                # Next.js config
└── README.md                     # Frontend docs
```

## 🔍 How It Works

### Architecture Overview

```
┌─────────────────┐      WebSocket      ┌──────────────────┐
│                 │    (video/webm)     │                  │
│   Frontend      │ ──────────────────> │   Backend        │
│   (Next.js)     │                     │   (Python)       │
│                 │ <────────────────── │                  │
└─────────────────┘    JSON Response    └──────────────────┘
      │                                         │
      │                                         │
      ▼                                         ▼
  MediaRecorder                           asl_pipeline.py
      │                                         │
      │                                         ▼
      │                                      FFmpeg
      │                                         │
      │                                         ▼
      │                                   Frame Extraction
      │                                         │
      │                                         ▼
      │                                    MediaPipe
      │                                   (Hand Detection)
      │                                         │
      │                                         ▼
      │                                     ASL Model
      │                                   (Classification)
      │                                         │
      │                                         ▼
      │                                   Batch Processing
      │                                    (10 frames avg)
      └────────────────────────────────────────┘
```

### Processing Pipeline

1. **Video Capture**: Frontend captures webcam video using `MediaRecorder`
2. **Streaming**: Sends video chunks (webm format) to backend via WebSocket
3. **Frame Extraction**: Backend uses FFmpeg to extract RGB frames (320×240)
4. **Hand Detection**: MediaPipe detects hand landmarks in each frame
5. **Preprocessing**: Crops hand region, resizes to 64×64, normalizes
6. **Classification**: ASL model predicts sign for each frame
7. **Batching**: Averages predictions over 10 frames for stability
8. **Response**: Sends JSON with predicted sign and confidence to frontend

### Message Format

**Backend Response:**
```json
{
  "type": "asl_batch",
  "predicted_sign": "A",
  "confidence": 0.95,
  "avg_prediction_vector": [0.01, 0.95, 0.02, ...]
}
```

## 🐛 Troubleshooting

### Backend Issues

**Error: `ModuleNotFoundError: No module named 'tensorflow'`**
- Solution: Activate virtual environment before running
  ```bash
  .\venv\Scripts\Activate.ps1  # Windows
  source venv/bin/activate      # Mac/Linux
  ```

**Error: `cannot import name 'keras' from 'tensorflow'`**
- Solution: Install Keras separately (TensorFlow 2.16+ compatibility)
  ```bash
  pip install keras==3.0.0
  ```

**Error: `OSError: [WinError 2] The system cannot find the file specified` (FFmpeg)**
- Solution: Install FFmpeg and add to PATH
- Verify: `ffmpeg -version`

**Error: `FileNotFoundError: models/ASL.h5`**
- Solution: Download and place ASL.h5 model in `models/` directory

**Error: `Address already in use` (Port 4000)**
- Solution: Kill existing process
  ```powershell
  # Windows
  Get-Process python | Stop-Process -Force
  
  # Mac/Linux
  lsof -ti:4000 | xargs kill -9
  ```

**Error: `protobuf version conflict`**
- Solution: Use compatible versions
  ```bash
  pip install protobuf==4.25.8 tensorflow==2.16.1
  ```

### Frontend Issues

**Error: `EADDRINUSE: address already in use :::3000`**
- Solution: Kill process on port 3000 or use different port
  ```bash
  pnpm dev -- -p 3001
  ```

**Camera not working**
- Check browser permissions (Chrome: chrome://settings/content/camera)
- Ensure HTTPS or localhost (some browsers require secure context)
- Try different browser (Chrome/Firefox recommended)

**WebSocket connection fails**
- Ensure backend is running on port 4000
- Check firewall settings
- Verify URL: `ws://localhost:4000/ws/stream/`

**No predictions appearing**
- Open browser console (F12 → Console) to see WebSocket messages
- Check backend terminal for prediction logs
- Ensure good lighting and clear hand gestures

### Performance Issues

**Slow processing / High CPU usage**
- Reduce frame rate: Lower `MediaRecorder.start()` interval in `stream-preview.tsx`
- Increase batch size: Modify `batch_size` in `asl_pipeline.py`
- Use GPU: Install TensorFlow GPU version (requires CUDA)

**Memory leaks**
- Restart backend server periodically
- Check for unclosed WebSocket connections

## 📊 Testing

### Test the Pipeline
```bash
# Backend tests
python test_pipeline.py

# Check dependencies
pip check
```

### Manual Testing
1. Start backend server
2. Start frontend
3. Open browser console
4. Click "Start Streaming"
5. Verify:
   - Backend logs show "Client connected"
   - Backend logs show "Predicted: [letter]"
   - Browser console shows "WS message:" with predictions

## 📚 API Documentation

### WebSocket Endpoint

**URL**: `ws://localhost:4000/ws/stream/`

**Protocol**:
1. Client connects to WebSocket
2. Client sends binary video chunks (webm format)
3. Server processes frames and sends JSON responses
4. Server sends batch predictions every 10 frames

**Response Format**:
```typescript
{
  type: "asl_batch",
  predicted_sign: string,      // A-Z, space, del, nothing
  confidence: number,           // 0.0 to 1.0
  avg_prediction_vector: number[]  // 29-element array
}
```

### Recognized Signs

The model recognizes 29 ASL signs:
- **Letters**: A-Z (26 letters)
- **Special**: space, del (delete), nothing (no hand detected)

## 🔗 Resources & References

- **ASL Dataset**: [Kaggle - ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Pre-trained Model**: [ASL Model 99% Accuracy](https://www.kaggle.com/datasets/namanmanchanda/american-sign-language-model-99-accuracy)
- **MediaPipe**: [Hand Landmark Detection](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- **FFmpeg**: [Official Website](https://ffmpeg.org/)
- **Next.js**: [Documentation](https://nextjs.org/docs)
- **WebSockets**: [Python websockets](https://websockets.readthedocs.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Development Notes

### Version Compatibility

**Critical Dependencies:**
- TensorFlow 2.16.1 (not 2.20+)
- Keras 3.0.0 (separate from TensorFlow)
- NumPy 1.26.4 (not 2.x)
- Protobuf 4.25.8 (MediaPipe compatibility)
- OpenCV 4.9.0.80

**Why these versions?**
- TensorFlow 2.20+ requires protobuf>=5.28
- MediaPipe 0.10.21 requires protobuf<5
- Keras 3.x is separate from TensorFlow
- NumPy 2.x has breaking changes

### Known Limitations

- WebSocket keepalive timeout after ~20 seconds (work in progress)
- Batch predictions only (no real-time per-frame predictions)
- Single hand detection only
- Requires good lighting conditions
- GPU acceleration not yet optimized

## 📄 License

[Add your license here]

## 👥 Team

StormHacks 2025 Team - Signable

---

**Built with ❤️ at StormHacks 2025**

For detailed integration documentation, see [INTEGRATION.md](INTEGRATION.md)