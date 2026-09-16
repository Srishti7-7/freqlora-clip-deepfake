#!/usr/bin/env python3
"""
setup.py - Environment setup (run once)
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*70)
    print("SETUP: Installing dependencies")
    print("="*70)
    
    # Install packages
    print("\n[1/3] Installing packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                   check=True)
    print("✓ Packages installed")
    
    # Download CLIP
    print("\n[2/3] Downloading CLIP model...")
    import clip
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    print("✓ CLIP model ready")
    
    # Create directories
    print("\n[3/3] Creating directories...")
    dirs = [
        'data/processed/images/real',
        'data/processed/images/fake',
        'checkpoints',
        'results'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Download FaceForensics++ c23")
    print("2. Extract frames to data/processed/images/")
    print("3. Run: python experiments/run_main.py")
    print("="*70)

if __name__ == "__main__":
    main()
