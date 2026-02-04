@echo off
REM ============================================================
REM XAUUSD Grid Trading System - Launcher
REM GetLeveraged Multi-Account Edition
REM ============================================================
REM
REM This script:
REM 1. Deploys the EA to all MT5 terminals
REM 2. Launches the LLM companion for dynamic SL/TP
REM
REM Accounts: 113328, 113326, 107245 @ GetLeveraged-Trade
REM ============================================================

title XAUUSD Grid Trading System

echo ============================================================
echo   XAUUSD GRID TRADING SYSTEM - LAUNCHER
echo   GetLeveraged Multi-Account Edition
echo ============================================================
echo.

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.9+ and add to PATH
    pause
    exit /b 1
)

echo [1/3] Deploying MQL5 files...
python deploy_xauusd_grid.py
if errorlevel 1 (
    echo WARNING: Deployment had issues - check output above
)

echo.
echo [2/3] Starting LLM Companion...
echo.

REM Launch LLM companion in new window
start "XAUUSD LLM Companion" cmd /k python xauusd_llm_companion.py --account 113328

echo LLM Companion launched in separate window
echo.

echo [3/3] Deployment Complete!
echo.
echo ============================================================
echo NEXT STEPS:
echo ============================================================
echo.
echo 1. Open GetLeveraged MT5 Terminal
echo.
echo 2. Log into account (113328, 113326, or 107245)
echo.
echo 3. Open XAUUSD chart (M5 timeframe)
echo.
echo 4. Attach EA: Experts\QuantumChildren\XAUUSD_GridMaster
echo.
echo 5. Load preset from: Presets\XAUUSD_GridMaster\GL_XXXXX.set
echo.
echo 6. Enable Auto-Trading (Algo Trading button)
echo.
echo ============================================================
echo.
echo Press any key to exit...
pause >nul
