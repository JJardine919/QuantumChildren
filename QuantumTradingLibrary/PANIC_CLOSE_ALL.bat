@echo off
title --- NUCLEAR PANIC BUTTON ---
color 4f
echo ==========================================
echo   WARNING: THIS WILL CLOSE ALL TRADES NOW
echo ==========================================
echo.
pause
python panic_close_all.py
echo.
echo Process complete.
pause
