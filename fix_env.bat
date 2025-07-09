@echo off
echo ================================================================
echo  FIXING Python Environment for no-hyperplanes
echo ================================================================
echo.

echo Step 1: Checking current Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)
echo ✓ Python found
echo.

echo Step 2: Removing old virtual environment...
if exist ".venv" (
    echo Removing existing .venv directory...
    rmdir /s /q .venv
    echo ✓ Old .venv removed
) else (
    echo ✓ No old .venv to remove
)
echo.

echo Step 3: Creating fresh virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

echo Step 4: Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

echo Step 5: Upgrading pip...
python -m pip install --upgrade pip
echo ✓ Pip upgraded
echo.

echo Step 6: Installing project dependencies...
echo Installing: torch, torchvision, numpy, matplotlib...
python -m pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.21.0 matplotlib>=3.5.0
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    echo Trying with requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Both installation methods failed!
        pause
        exit /b 1
    )
)
echo ✓ Dependencies installed
echo.

echo Step 7: Testing installation...
echo Testing PyTorch import...
python -c "import torch; print('✓ PyTorch version:', torch.__version__)"
if errorlevel 1 (
    echo ERROR: PyTorch import failed!
    pause
    exit /b 1
)

python -c "import torchvision; print('✓ TorchVision version:', torchvision.__version__)"
if errorlevel 1 (
    echo ERROR: TorchVision import failed!
    pause
    exit /b 1
)
echo.

echo ================================================================
echo  SUCCESS! Environment is now ready!
echo ================================================================
echo.
echo To use this environment:
echo 1. Run: .venv\Scripts\activate.bat
echo 2. Then: cd src
echo 3. Then: python test_all.py
echo.
echo ================================================================

pause 