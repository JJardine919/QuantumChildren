@echo off
title QUANTUM CHILDREN - Auto Training Loop
color 0B

echo ============================================================
echo   QUANTUM CHILDREN - AUTONOMOUS TRAINING LOOP
echo ============================================================
echo.
echo   RULE #1: DO NOT TOUCH THE LIVE TRADING SYSTEM
echo   The BRAIN self-regulates. We never modify it.
echo.
echo   This loop handles:
echo     - Data collection (BTCUSD, ETHUSD, XAUUSD)
echo     - Signal farm training (genetic evolution)
echo     - LSTM expert retraining
echo     - QNIF prop firm simulations
echo     - TE domestication feeding
echo     - Expert rotation (age out old, promote new)
echo     - System health checks
echo.
echo   Log dir: orchestrator_logs\
echo   Config:  auto_loop_config.json
echo   State:   auto_loop_state.json
echo.
echo   Press Ctrl+C to stop gracefully.
echo ============================================================
echo.

:: Set working directory
cd /d C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary

:: Use the GPU venv
set PYTHON_CMD=C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\.venv312_gpu\Scripts\python.exe

:: Verify Python exists
if not exist "%PYTHON_CMD%" (
    echo [ERROR] Python venv not found at %PYTHON_CMD%
    echo Please set up .venv312_gpu first.
    pause
    exit /b 1
)

:: Start the auto training loop
%PYTHON_CMD% C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\auto_training_loop.py

echo.
echo Auto training loop has stopped.
pause
