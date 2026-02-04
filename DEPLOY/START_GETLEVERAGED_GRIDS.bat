@echo off
echo ============================================================
echo   GetLeveraged Grid Trading System - Launcher
echo   XAUUSD ^| BTCUSD ^| ETHUSD on 3 Accounts
echo ============================================================
echo.
echo Configuration:
echo   SL: 1.5x ATR (hard-coded)
echo   TP: 3.0x ATR (hard-coded)
echo   Partial TP: 50%%
echo   Entropy Filter: ENABLED (+12 boost)
echo   LLM: Dynamic adjustments via Ollama
echo ============================================================
echo.

:: Check if Ollama is running
echo [1/4] Checking Ollama LLM...
curl -s http://localhost:11434/api/version >nul 2>&1
if %errorlevel% neq 0 (
    echo       Ollama not running. Starting...
    start "" ollama serve
    timeout /t 5 /nobreak >nul
    echo       Ollama started.
) else (
    echo       Ollama is running.
)

:: Pull model if needed
echo [2/4] Ensuring LLM model is available...
ollama list | findstr /i "mistral" >nul
if %errorlevel% neq 0 (
    echo       Pulling mistral model...
    ollama pull mistral
) else (
    echo       Model ready.
)

:: Start LLM optimizer in background
echo [3/4] Starting LLM SL/TP Optimizer...
cd /d "%~dp0"
start "LLM Optimizer" cmd /c "python llm_sltp_optimizer.py 60"
echo       LLM Optimizer running (60s updates)

:: Instructions for MT5
echo.
echo [4/4] MT5 Setup Instructions:
echo ============================================================
echo.
echo Copy these files to your MT5 data folder:
echo   EntropyGridCore.mqh         -> MQL5\Include\
echo   XAUUSD_GridTrader.mq5       -> MQL5\Experts\
echo   BTCUSD_GridTrader.mq5       -> MQL5\Experts\
echo   ETHUSD_GridTrader.mq5       -> MQL5\Experts\
echo   MultiSymbol_Launcher.mq5    -> MQL5\Experts\
echo.
echo Then compile in MetaEditor and attach to charts.
echo.
echo ACCOUNT SETUP:
echo   Account 113328 -> Set InpAccountSelector = 1
echo   Account 113326 -> Set InpAccountSelector = 2
echo   Account 107245 -> Set InpAccountSelector = 3
echo.
echo ============================================================
echo Press any key to open MT5 data folder...
pause >nul

:: Open MT5 data folder
start "" "%APPDATA%\MetaQuotes\Terminal"

echo.
echo System started. Check llm_sltp_optimizer.log for LLM updates.
pause
