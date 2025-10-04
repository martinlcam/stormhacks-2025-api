# Signable
## Backend for StormHacks 2025 - Signable



Real-time American Sign Language (ASL) recognition system that converts sign language gestures into text and speech. Data: https://www.kaggle.com/datasets/grassknoted/asl-alphabet



## 🚀 Quick Start (New Device Setup)To download the dataset (to recreate this project), run the following commands (for your system):



### 1. Clone and Setup EnvironmentWindows (PS):

```bash```

git clone https://github.com/martinlcam/stormhacks-2025-api.git> kaggle datasets download -d grassknoted/asl-alphabet -p data Dataset URL: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

cd stormhacks-2025-api> Expand-Archive -Path "data/asl-alphabet.zip" -DestinationPath "data/asl-alphabet"

git checkout ASL_Alphabet_Clean```

Mac/Linux:

# Create virtual environment```

python -m venv venv(TBD)

# Activate it```

venv\Scripts\activate  # Windows

source venv/bin/activate  # Mac/Linux
```

### 2. Install Dependencies
```bash
pip install tensorflow keras opencv-python scikit-learn scikit-image pandas numpy matplotlib seaborn mediapipe
```

### 3. Get the Model Files

**Option A: Download Pre-trained Model (Recommended)**
1. Go to [Kaggle Model](https://www.kaggle.com/datasets/namanmanchanda/american-sign-language-model-99-accuracy)
2. Download `american-sign-language-model-99-accuracy.zip`
3. Extract to `models/` directory
4. The `ASL.h5` file should be in `models/ASL.h5`

**Option B: Train from Scratch**
1. Download dataset from [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. Place `asl-alphabet.zip` in `data/` directory  
3. Run: `python alphabet.py` (will take 30-60 minutes)

**Option C: Automated Setup**
```bash
python setup_project.py
```

## 📁 Project Structure
```
├── models/
│   ├── ASL.h5                    # Trained model (4MB)
│   └── *.zip                     # Downloaded model archives
├── data/
│   └── asl-alphabet.zip          # Dataset (1GB+ - not in Git)
├── alphabet.py                   # Model training script
├── setup_project.py              # Automated setup
└── README.md                     # This file
```

## 🧠 Model Details
- **Input**: 64×64×3 RGB images
- **Output**: 29 classes (A-Z + del + nothing + space)
- **Architecture**: CNN with 3 conv layers + 2 dense layers  
- **Accuracy**: ~99% on test set, ~85-95% real-world
- **Size**: 4MB (manageable for Git)

## 🔧 Model Usage
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('models/ASL.h5')

# Predict (input: 64x64x3 normalized image)
image = np.random.random((1, 64, 64, 3))  # Your preprocessed image
prediction = model.predict(image)
class_idx = np.argmax(prediction)

# Class mapping: 0=A, 1=B, ..., 25=Z, 26=del, 27=nothing, 28=space
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
if class_idx < 26:
    predicted_letter = letters[class_idx]
elif class_idx == 26:
    predicted_letter = 'DEL'
elif class_idx == 27:
    predicted_letter = 'NOTHING'
else:
    predicted_letter = 'SPACE'
```

## 🗂️ Data Sources
- **Dataset**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (1GB)
- **Pre-trained Model**: [99% Accuracy ASL Model](https://www.kaggle.com/datasets/namanmanchanda/american-sign-language-model-99-accuracy) (4MB)

## ⚠️ Important Notes
- Large files (models, datasets) are **not tracked in Git**
- Use the setup scripts or manual download for new devices
- Model expects 64×64 input images (not 224×224 like some models)
- Ensure proper image preprocessing: resize to 64×64, normalize to [0,1]

## 🧩 Next Steps
1. Set up MediaPipe hand detection pipeline
2. Create real-time inference system  
3. Build WebSocket API for live predictions
4. Integrate with frontend for complete ASL recognition