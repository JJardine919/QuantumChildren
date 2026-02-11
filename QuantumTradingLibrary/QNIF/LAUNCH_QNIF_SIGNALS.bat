@echo off
REM ========================================================
REM  QNIF SIGNAL FARM - DUAL MACHINE LAUNCH
REM ========================================================
REM
REM Launches QNIF signal runners on both ATLAS and FTMO.
REM Each runner handles BTC + ETH on its dedicated account.
REM
REM Date: 2026-02-10
REM ========================================================

echo.
echo ========================================================
echo  QNIF SIGNAL FARM - DUAL MACHINE LAUNCH
echo ========================================================
echo.

REM GPU Python environment
set VENV=C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\.venv312_gpu\Scripts\python.exe
set QNIF_DIR=C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\QNIF

REM Verify Python exists
if not exist "%VENV%" (
    echo ERROR: Python venv not found at %VENV%
    echo Please verify the path and try again.
    pause
    exit /b 1
)

REM Verify script exists
if not exist "%QNIF_DIR%\qnif_signal_runner.py" (
    echo ERROR: qnif_signal_runner.py not found in %QNIF_DIR%
    echo Please verify the path and try again.
    pause
    exit /b 1
)

echo [1/2] Starting QNIF Signals on ATLAS (BTC + ETH)...
echo.
start "QNIF-ATLAS" "%VENV%" "%QNIF_DIR%\qnif_signal_runner.py" --account ATLAS --symbols BTCUSD ETHUSD --interval 300

REM Stagger startup to avoid MT5 connection conflicts
timeout /t 5 /nobreak >nul

echo [2/2] Starting QNIF Signals on FTMO (BTC + ETH)...
echo.
start "QNIF-FTMO" "%VENV%" "%QNIF_DIR%\qnif_signal_runner.py" --account FTMO --symbols BTCUSD ETHUSD --interval 300

echo.
echo ========================================================
echo  LAUNCH COMPLETE
echo ========================================================
echo.
echo Both machines are now running:
echo   - ATLAS: BTCUSD + ETHUSD ^(Account 212000584^)
echo   - FTMO:  BTCUSD + ETHUSD ^(Account 1521063483^)
echo.
echo Check the newly opened windows for live status.
echo Signal files will be saved to:
echo   - qnif_signal_BTCUSD.json
echo   - qnif_signal_ETHUSD.json
echo.
echo Logs: %QNIF_DIR%\qnif_signals.log
echo.
echo Press any key to exit this launcher...
pause >nul
