@echo off
title GETLEVERAGED QUANTUM BRAIN
echo ============================================================
echo   GETLEVERAGED QUANTUM BRAIN
echo   Trading BTCUSD
echo   Accounts: 113326, 113328, 107245
echo ============================================================
echo.

cd /d "%~dp0"
python BRAIN_GETLEVERAGED.py

pause
