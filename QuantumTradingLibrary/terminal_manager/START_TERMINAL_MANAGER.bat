@echo off
title Quantum Children - Terminal Manager
echo ================================================
echo   QUANTUM CHILDREN - Terminal Manager
echo ================================================
echo.

cd /d "%~dp0"

:: Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

echo Starting Terminal Manager...
echo Open http://localhost:5000 in your browser
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

python app.py

pause
