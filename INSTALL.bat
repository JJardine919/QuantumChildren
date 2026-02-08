@echo off
title Quantum Children Installer
echo.
echo ============================================================
echo   QUANTUM CHILDREN - INSTALLER
echo ============================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

REM Run the installer
python "%~dp0INSTALL.py"

echo.
pause
