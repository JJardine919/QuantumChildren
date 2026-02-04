@echo off
echo ========================================
echo   QUANTUM CHILDREN - Trading System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Install requirements if needed
pip show numpy >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
)

echo Starting QuantumChildren...
echo.
python quantum_trader.py

pause
