#!/usr/bin/env python
"""
Setup script for Alice Feedback.

This script:
1. Creates a virtual environment using uv
2. Installs the package with optional CUDA support
3. Installs OpenCV-CUDA from cudawarped GitHub releases if CUDA is enabled

Usage:
  python setup_env.py         # Install with CPU support
  python setup_env.py --cuda  # Install with CUDA support
"""
import os
import platform
import subprocess
import sys
import argparse

def run_command(cmd, desc=None):
    """Run a shell command and display its output."""
    if desc:
        print(f"[+] {desc}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error ({result.returncode}):")
        print(result.stderr)
        sys.exit(1)
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Setup environment for Alice Feedback")
    parser.add_argument("--cuda", action="store_true", help="Install with CUDA support")
    args = parser.parse_args()
    
    use_cuda = args.cuda
    
    # Determine the operating system
    is_windows = platform.system() == "Windows"
    
    # Create virtual environment
    venv_dir = ".venv"
    print(f"[+] Creating virtual environment in {venv_dir} directory...")
    run_command(["uv", "venv", venv_dir])

    
    # Install the package with appropriate option
    extra = "cuda" if use_cuda else "cpu"
    print(f"[+] Installing the package with {extra.upper()} support...")
    run_command(["uv", "sync", "--extra", extra])
    
    if use_cuda:
        # OpenCV-CUDA version 4.11.0.20250124
        
        if is_windows:
            # For Windows, download from GitHub releases
            opencv_cuda_url = f"https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.11.0.20250124/opencv_contrib_python_rolling-4.12.0.86-cp37-abi3-win_amd64.whl"
        else:
            # For Linux, use the corresponding URL
            opencv_cuda_url = f"https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.11.0.20250124/opencv_contrib_python_rolling-4.12.0.20250124-cp37-abi3-linux_x86_64.whl"
        
        # Download and install OpenCV-CUDA
        
        try:
            run_command(["uv", "pip", "install", opencv_cuda_url])
            print("[+] Installed OpenCV with CUDA support")
            print("[!] Note: CUDA 12.8 toolkit is required to use OpenCV with CUDA support.")
            
        except Exception as e:
            print(f"Error installing OpenCV-CUDA: {e}")
    
    print("\n[+] Installation complete!")
    print(f"\n[+] Environment type: {'CUDA-enabled' if use_cuda else 'CPU-only'}")
    print("\nTo run the application:")
    print("    uv run alice-feedback")
    
    if use_cuda:
        print("\n[!] Additional CUDA requirements:")
        print("  - NVIDIA GPU Computing Toolkit v12.8")
        print("  - cuDNN 9.7.0")
        print("\nFor Windows, ensure cuDNN is properly installed and in your PATH or configured in cv2/config.py")

if __name__ == "__main__":
    main()