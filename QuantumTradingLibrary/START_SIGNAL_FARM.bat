@echo off
title Signal Farm Engine - 5 Virtual Accounts
echo ============================================================
echo   SIGNAL FARM ENGINE - Starting...
echo   NO REAL TRADES - Read-Only MT5 Simulation
echo ============================================================
echo.

cd /d "%~dp0"

REM Use .venv311 if available, otherwise try system Python
if exist ".venv311\Scripts\python.exe" (
    echo Using .venv311 Python...
    .venv311\Scripts\python.exe SIGNAL_FARM_ENGINE.py
) else if exist ".venv\Scripts\python.exe" (
    echo Using .venv Python...
    .venv\Scripts\python.exe SIGNAL_FARM_ENGINE.py
) else (
    echo Using system Python...
    python SIGNAL_FARM_ENGINE.py
)

echo.
echo Signal Farm Engine stopped.
pause
