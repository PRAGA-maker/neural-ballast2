@echo off
echo Setting up no-hyperplanes project environment...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo Python found. Creating virtual environment...

REM Create virtual environment
python -m venv .venv

echo Activating virtual environment...

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Installing dependencies...

REM Install requirements
pip install -r requirements.txt

echo.
echo Testing installation...

REM Test PyTorch installation
python -c "import torch; print('✓ PyTorch version:', torch.__version__)"
python -c "import torchvision; print('✓ TorchVision version:', torchvision.__version__)"

echo.
echo ========================================
echo Setup complete! 
echo.
echo To activate the environment manually:
echo   .venv\Scripts\activate.bat
echo.
echo To test the project:
echo   cd src
echo   python test_all.py
echo ========================================

pause 