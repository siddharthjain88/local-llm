#!/usr/bin/env python3
"""
Setup script for GPT-OSS-20B model
Downloads the model from HuggingFace and prepares it for local use
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['transformers', 'torch', 'accelerate', 'huggingface_hub']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✓ All required packages are installed")

def download_model():
    """Download GPT-OSS-20B model from HuggingFace"""
    from huggingface_hub import snapshot_download
    
    model_id = "openai/gpt-oss-20b"
    local_dir = Path("./models/gpt-oss-20b")
    
    print(f"Downloading {model_id} to {local_dir}...")
    print("Note: This is a ~13GB download and may take some time")
    
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"✓ Model downloaded successfully to {local_dir}")
        return str(local_dir)
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

def main():
    print("GPT-OSS-20B Setup Script")
    print("=" * 50)
    
    # Check requirements
    check_requirements()
    
    # Download model
    model_path = download_model()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print(f"Model location: {model_path}")
    print("\nTo test the model, run: python test_gpt_oss.py")

if __name__ == "__main__":
    main()