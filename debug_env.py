#!/usr/bin/env python3
"""
Debug script to check Python environment and identify issues.
Run this to understand what's wrong with your Python setup.
"""

import sys
import os
import subprocess
import platform

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def run_command(cmd, description):
    """Run a command and capture its output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"\n{description}:")
        print(f"  Command: {cmd}")
        print(f"  Exit code: {result.returncode}")
        if result.stdout:
            print(f"  Output: {result.stdout.strip()}")
        if result.stderr:
            print(f"  Error: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"\n{description}: ERROR - {e}")
        return False

def check_virtual_env():
    """Check if we're in a virtual environment."""
    venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"\nVirtual Environment Status:")
    print(f"  In virtual environment: {venv_active}")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python prefix: {sys.prefix}")
    if hasattr(sys, 'base_prefix'):
        print(f"  Base prefix: {sys.base_prefix}")
    
    # Check for common virtual env indicators
    if os.environ.get('VIRTUAL_ENV'):
        print(f"  VIRTUAL_ENV: {os.environ['VIRTUAL_ENV']}")
    else:
        print("  VIRTUAL_ENV: Not set")
    
    return venv_active

def check_python_info():
    """Check basic Python information."""
    print(f"\nPython Information:")
    print(f"  Version: {sys.version}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Executable: {sys.executable}")
    print(f"  Path: {sys.path[:3]}...")  # Show first 3 paths

def check_pip():
    """Check pip installation and functionality."""
    try:
        import pip
        print(f"\nPip Information:")
        print(f"  Pip module found: {pip.__file__}")
    except ImportError:
        print(f"\nPip Information:")
        print("  Pip module: NOT FOUND")
    
    # Try to run pip commands
    run_command("python -m pip --version", "Pip version check")
    run_command("pip --version", "Direct pip check")

def check_torch():
    """Check if PyTorch is installed."""
    try:
        import torch
        print(f"\nPyTorch Information:")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  PyTorch location: {torch.__file__}")
        return True
    except ImportError as e:
        print(f"\nPyTorch Information:")
        print(f"  PyTorch: NOT INSTALLED")
        print(f"  Import error: {e}")
        return False

def check_files():
    """Check for important project files."""
    files_to_check = [
        ".venv",
        "requirements.txt", 
        "setup.ps1",
        "setup.bat",
        "activate.bat"
    ]
    
    print(f"\nProject Files:")
    for file in files_to_check:
        exists = os.path.exists(file)
        print(f"  {file}: {'✓' if exists else '✗'}")
        
        if file == ".venv" and exists:
            # Check if .venv has proper structure
            venv_python = os.path.join(".venv", "Scripts", "python.exe")
            venv_activate = os.path.join(".venv", "Scripts", "activate.bat")
            print(f"    .venv/Scripts/python.exe: {'✓' if os.path.exists(venv_python) else '✗'}")
            print(f"    .venv/Scripts/activate.bat: {'✓' if os.path.exists(venv_activate) else '✗'}")

def main():
    print_header("Python Environment Debug Report")
    
    # Basic system info
    check_python_info()
    
    # Virtual environment check
    venv_active = check_virtual_env()
    
    # File structure check
    check_files()
    
    # Pip check
    check_pip()
    
    # PyTorch check
    torch_installed = check_torch()
    
    # Summary and recommendations
    print_header("Summary and Recommendations")
    
    if not venv_active:
        print("❌ ISSUE: Not in virtual environment")
        print("   SOLUTION: Activate virtual environment first")
        print("   Try: .venv\\Scripts\\activate.bat")
        print("   Or: .\\.venv\\Scripts\\Activate.ps1")
    
    if not torch_installed:
        print("❌ ISSUE: PyTorch not installed")
        if venv_active:
            print("   SOLUTION: Install requirements in virtual environment")
            print("   Try: python -m pip install -r requirements.txt")
        else:
            print("   SOLUTION: Activate virtual environment first, then install")
    
    if venv_active and torch_installed:
        print("✅ Environment looks good!")
    
    print("\n" + "="*60)
    print("Debug report complete!")

if __name__ == "__main__":
    main() 