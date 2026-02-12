@echo off
title QUANTUM CHILDREN - MASTER LAUNCH
echo ============================================================
echo   QUANTUM CHILDREN - MASTER ORCHESTRATOR
echo   Multi-Account Trading Automation
echo ============================================================
echo.

cd /d "%~dp0"

REM Check for existing processes/locks
echo Checking for running processes...
.venv312_gpu\Scripts\python.exe process_lock.py --list

echo.
echo Starting MASTER_LAUNCH orchestrator...
echo This will launch all enabled accounts from MASTER_CONFIG.json
echo.
echo Press Ctrl+C to stop the orchestrator (graceful shutdown)
echo.

REM Launch the orchestrator
.venv312_gpu\Scripts\python.exe MASTER_LAUNCH.py

echo.
echo Orchestrator exited.
pause
