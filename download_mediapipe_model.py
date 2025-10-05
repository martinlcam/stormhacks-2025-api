"""
Download MediaPipe Hand Landmarker Model
Run this script to download the hand_landmarker.task file to your models directory
"""

import urllib.request
import os

def download_mediapipe_model():
    """Download the MediaPipe hand landmarker model"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/hand_landmarker.task'
    model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        print(f"✅ Model already exists: {model_path} ({file_size:.1f} MB)")
        return
    
    print("📥 Downloading MediaPipe hand landmarker model...")
    print(f"From: {model_url}")
    print(f"To: {model_path}")
    
    try:
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
        
        urllib.request.urlretrieve(model_url, model_path, reporthook=progress_hook)
        print()  # New line after progress
        
        # Verify download
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Successfully downloaded: {model_path} ({file_size:.1f} MB)")
        
        # File info
        print(f"\n📋 Model Info:")
        print(f"   File: {model_path}")
        print(f"   Size: {file_size:.1f} MB")
        print(f"   Type: MediaPipe Hand Landmarker (float16)")
        print(f"   License: Apache 2.0 (can be included in repo)")
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"\nYou can download manually:")
        print(f"1. Go to: {model_url}")
        print(f"2. Save as: {model_path}")

if __name__ == "__main__":
    download_mediapipe_model()