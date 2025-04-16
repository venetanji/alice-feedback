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
import tempfile
import urllib.request
import argparse

def run_command(cmd, desc=None):
    """Run a shell command and display its output."""
    if desc:
        print(f"[+] {desc}...")
    print(f"Running: {' '.join(cmd)}")
    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        print(f"Error ({result.returncode}):")
        print(result.stderr)
        sys.exit(1)
    return result

def download_file(url, target_path):
    """Download a file from a URL to a target path."""
    print(f"[+] Downloading {url}...")
    urllib.request.urlretrieve(url, target_path)
    print(f"[+] Downloaded to {target_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Setup environment for Alice Feedback")
    parser.add_argument("--cuda", action="store_true", help="Install with CUDA support")
    args = parser.parse_args()
    
    use_cuda = args.cuda
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Determine the operating system
    is_windows = platform.system() == "Windows"
    
    # Create virtual environment
    venv_dir = ".venv"
    print(f"[+] Creating virtual environment in {venv_dir} directory...")
    run_command(["uv", "venv", venv_dir])
    
    # Get path to python and pip in the virtual environment
    if is_windows:
        python_cmd = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_cmd = ["uv", "pip"]
        uv_cmd = "uv"
    else:
        python_cmd = os.path.join(venv_dir, "bin", "python")
        pip_cmd = os.path.join(venv_dir, "bin", "pip")
        uv_cmd = os.path.join(venv_dir, "bin", "uv")
    
    # Install the package with appropriate option
    extra = "cuda" if use_cuda else "cpu"
    print(f"[+] Installing the package with {extra.upper()} support...")
    run_command([uv_cmd, "sync", "--extra", extra])
    
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
            run_command([uv_cmd, "pip", "install", opencv_cuda_url])
            print("[+] Installed OpenCV with CUDA support")
            print("[!] Note: CUDA 12.8 toolkit is required to use OpenCV with CUDA support.")
            
            # Configure cuDNN path (for Windows)
            # if is_windows:
            #     config_path = os.path.join(os.path.dirname(os.path.dirname(pip_cmd)), 
            #                               "Lib", "site-packages", "cv2", "config.py")
                # if os.path.exists(config_path):
                #     # Check if it's already modified 
                #     with open(config_path, "r") as f:
                #         content = f.read()
                    
                #     if "CUDNN" not in content:
                #         cudnn_path = "C:/Program Files/NVIDIA/CUDNN/v9.7/bin/12.8"
                #         config_content = f"""import os
                # BINARIES_PATHS = [
                #     os.path.join('D:/build/opencv/install', 'x64/vc17/bin'),
                #     os.path.join(os.getenv('CUDA_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8'), 'bin'),
                #     os.path.join('{cudnn_path}')
                # ] + BINARIES_PATHS
                # """
                                        
                        # with open(config_path, "w") as f:
                        #     f.write(config_content)
                        # print(f"[+] Updated {config_path} with cuDNN path")
                        # print(f"[!] Note: Make sure you have cuDNN 9.7.0 installed at {cudnn_path}")
        except Exception as e:
            print(f"Error installing OpenCV-CUDA: {e}")

    else:
        # Install regular OpenCV for CPU
        run_command([uv_cmd, "pip", "install", "opencv-python"])
            
    # # Install other dependencies
    # print(f"[+] Installing remaining dependencies...")
    # run_command([uv_cmd, "pip", "install", "-e", "."])
    
    print("\n[+] Installation complete!")
    print(f"\n[+] Environment type: {'CUDA-enabled' if use_cuda else 'CPU-only'}")
    
    print("\nTo activate the virtual environment:")
    if is_windows:
        print("    .venv\\Scripts\\activate")
    else:
        print("    source .venv/bin/activate")
    
    print("\nTo run the application:")
    print("    python main.py")
    
    if use_cuda:
        print("\n[!] Additional CUDA requirements:")
        print("  - NVIDIA GPU Computing Toolkit v12.8")
        print("  - cuDNN 9.7.0")
        print("\nFor Windows, ensure cuDNN is properly installed and in your PATH or configured in cv2/config.py")

if __name__ == "__main__":
    main()