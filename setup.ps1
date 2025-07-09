#!/usr/bin/env pwsh

Write-Host "Setting up no-hyperplanes project environment..." -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion found" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and add it to your PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Creating virtual environment..." -ForegroundColor Yellow

# Remove existing .venv if it exists and is corrupted
if (Test-Path ".venv") {
    Write-Host "Removing existing .venv directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}

# Create virtual environment
python -m venv .venv

Write-Host "Activating virtual environment..." -ForegroundColor Yellow

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..." -ForegroundColor Yellow

# Upgrade pip first
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

Write-Host ""
Write-Host "Testing installation..." -ForegroundColor Yellow

# Test PyTorch installation
try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
    Write-Host "✓ PyTorch version: $torchVersion" -ForegroundColor Green
    
    $torchvisionVersion = python -c "import torchvision; print(torchvision.__version__)" 2>&1
    Write-Host "✓ TorchVision version: $torchvisionVersion" -ForegroundColor Green
    
    $numpyVersion = python -c "import numpy; print(numpy.__version__)" 2>&1
    Write-Host "✓ NumPy version: $numpyVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error testing installation: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment manually:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To test the project:" -ForegroundColor Yellow
Write-Host "  cd src" -ForegroundColor White
Write-Host "  python test_all.py" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan

Read-Host "Press Enter to continue" 