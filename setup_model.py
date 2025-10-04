#!/usr/bin/env python3
"""
Script to download and set up the pre-trained ASL model
Run this script on new devices to get the model files
"""

import os
import requests
import zipfile
from pathlib import Path

def download_file(url, local_filename):
    """Download a file from URL with progress"""
    print(f"Downloading {local_filename}...")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(local_filename, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end="")
    
    print(f"\n✅ Downloaded {local_filename}")

def setup_asl_model():
    """Download and extract the ASL model"""
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # URLs for the model files (you'll need to update these)
    model_urls = {
        "american-sign-language-model-99-accuracy.zip": "https://www.kaggle.com/datasets/namanmanchanda/american-sign-language-model-99-accuracy/download?datasetVersionNumber=1"
    }
    
    print("🔧 Setting up ASL Model...")
    print("=" * 50)
    
    for filename, url in model_urls.items():
        file_path = models_dir / filename
        
        # Download if doesn't exist
        if not file_path.exists():
            try:
                download_file(url, file_path)
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                print(f"📝 Manual download required from: {url}")
                continue
        else:
            print(f"✅ {filename} already exists")
        
        # Extract if it's a zip file
        if filename.endswith('.zip'):
            print(f"📦 Extracting {filename}...")
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(models_dir)
                print(f"✅ Extracted {filename}")
            except Exception as e:
                print(f"❌ Failed to extract {filename}: {e}")
    
    # Check if ASL.h5 exists
    asl_model = models_dir / "ASL.h5"
    if asl_model.exists():
        print(f"🎉 ASL model ready at: {asl_model}")
        print(f"📏 Model size: {asl_model.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("⚠️  ASL.h5 not found. You may need to:")
        print("   1. Run the training script (alphabet.py)")
        print("   2. Or manually download ASL.h5 from Kaggle")
    
    print("\n🚀 Setup complete!")

if __name__ == "__main__":
    setup_asl_model()