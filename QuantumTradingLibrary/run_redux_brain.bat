@echo off
title ETARE REDUX BRAIN - PERMANENT PROCESS
color 0A

:LOOP
cls
echo ========================================================
echo  ETARE REDUX BRAIN - PERMANENT PROCESS
echo ========================================================
echo.
echo  Starting Redux Signal Generator (LSTM CHAMPIONS)...
echo  Time: %TIME%
echo.

:: Run the Redux Brain using the GPU environment
.\gpu_env\Scripts\python.exe 06_Integration/HybridBridge/etare_signal_generator_redux.py

:: If it crashes/stops, we land here
color 0C
echo.
echo ========================================================
echo  WARNING: BRAIN STOPPED / CRASHED!
echo  Restarting in 5 seconds...
echo ========================================================
timeout /t 5 >nul
color 0A
goto LOOP
