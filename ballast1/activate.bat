@echo off
echo Activating no-hyperplanes virtual environment...

if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup.bat first to create the environment.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo You can now run:
echo   cd src
echo   python test_all.py
echo   python diagnostic.py
echo.

cmd /k 