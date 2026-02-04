@echo off
title QUANTUM CHILDREN FACTORY - HEAVY TRAINING
color 0A

echo ========================================================
echo  STARTING QUANTUM FACTORY (25 SYMBOLS)
echo ========================================================
echo.
echo  This will train a "Champion" strategy for every symbol.
echo  It uses 5 years of history and evolutionary algorithms.
echo.
echo  Output Log: ETARE_Redux.log
echo  Database:   etare_redux_v2.db
echo.
echo  [IMPORTANT] Make sure MetaTrader 5 is open for data!
echo.

:: Check for GPU Environment
if exist "gpu_env\Scripts\python.exe" (
    echo [INFO] Using GPU Environment - gpu_env found.
    set PYTHON_CMD=gpu_env\Scripts\python.exe
) else (
    echo [WARNING] GPU Environment not found. Falling back to system python...
    set PYTHON_CMD=python
)

:: Run the Heavy Trainer
%PYTHON_CMD% 01_Systems/System_03_ETARE/ETARE_Redux.py

echo.
echo  [DONE] Training complete. Champions are saved in the Database.
pause
