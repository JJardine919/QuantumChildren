@echo off
title EMERGENCY STOP FIXER
color 0c
echo ==========================================
echo   FIXING BTCUSD STOP LOSSES (ACCOUNT 113326)
echo ==========================================
python fix_getleveraged_stops.py
echo.
echo Done. Review the results above.
pause
