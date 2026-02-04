@echo off
title EMERGENCY GUARDIAN (Gemma 12B)
echo ============================================================
echo   EMERGENCY STOP GUARDIAN
echo   Powered by Gemma 12B LLM
echo   Watching all accounts for emergencies
echo ============================================================
echo.

cd /d "%~dp0"
python emergency_stop_guardian.py

pause
