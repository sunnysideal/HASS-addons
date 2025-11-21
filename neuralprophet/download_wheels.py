# Script to download all pip dependencies as wheels
# Run this locally before building Docker image

import subprocess
import sys
import os
from pathlib import Path

def download_wheels():
    """Download all pip packages as wheels to ./wheels directory"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    requirements = script_dir / 'requirements.txt'
    wheels_dir = script_dir / 'wheels'
    
    # Create wheels directory if it doesn't exist
    wheels_dir.mkdir(exist_ok=True)
    
    print(f"Downloading wheels from {requirements} to {wheels_dir}...")
    
    subprocess.run([
        sys.executable, '-m', 'pip', 'download',
        '-r', str(requirements),
        '-d', str(wheels_dir),
        '--platform', 'manylinux2014_x86_64',
        '--platform', 'manylinux2014_aarch64', 
        '--only-binary', ':all:',
        '--python-version', '313'
    ], check=True)
    
    print(f"✓ Wheels downloaded to {wheels_dir}")

if __name__ == '__main__':
    download_wheels()
