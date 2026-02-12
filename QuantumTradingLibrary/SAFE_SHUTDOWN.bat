@echo off
title SAFE SHUTDOWN - QUANTUM CHILDREN
echo ============================================================
echo   SAFE SHUTDOWN - QUANTUM CHILDREN
echo   Gracefully stops all trading processes
echo ============================================================
echo.

cd /d "%~dp0"

REM Check for running processes first
echo Scanning for running processes...
echo.
.venv312_gpu\Scripts\python.exe SAFE_SHUTDOWN.py --dry-run

echo.
echo ============================================================
echo   This will terminate ALL trading processes:
echo   - BRAIN scripts (all accounts)
echo   - Watchdog monitors
echo   - Quantum server
echo   - MASTER_LAUNCH orchestrator
echo.
echo   Open trades will NOT be closed automatically.
echo   Lockfiles will be cleared for safe restart.
echo ============================================================
echo.

REM Run the shutdown script
.venv312_gpu\Scripts\python.exe SAFE_SHUTDOWN.py

echo.
pause
