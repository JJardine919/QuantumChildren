@echo off
title SYSTEM HEALTH GUARDIAN
color 0a
echo Running automated database integrity check and backup...
echo.
python system_health_guardian.py
echo.
echo Check complete. Backup stored in 04_Data/Archive/Backups
pause
