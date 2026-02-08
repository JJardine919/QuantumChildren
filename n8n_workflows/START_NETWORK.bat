@echo off
echo ============================================
echo Quantum Children Signal Network - Launcher
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Installing Flask...
    pip install flask requests
)

echo.
echo Starting Quantum Children Webhook Server...
echo.
echo Server will run on http://localhost:8889
echo.
echo Endpoints:
echo   GET  http://localhost:8889/              - Health check
echo   POST http://localhost:8889/signal        - Submit signal
echo   GET  http://localhost:8889/signals/latest - Get latest signals
echo.
echo Press Ctrl+C to stop.
echo.

cd /d "%~dp0"
python quantum_webhook_server.py --port 8889 --demo

pause
