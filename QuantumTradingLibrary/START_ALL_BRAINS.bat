@echo off
echo ============================================================
echo   STARTING ALL QUANTUM BRAINS
echo ============================================================
echo.

cd /d "%~dp0"

echo Starting BlueGuardian Brain...
start "BLUEGUARDIAN BRAIN" cmd /c "python BRAIN_BLUEGUARDIAN.py"

timeout /t 5 /nobreak > nul

echo Starting GetLeveraged Brain...
start "GETLEVERAGED BRAIN" cmd /c "python BRAIN_GETLEVERAGED.py"

timeout /t 5 /nobreak > nul

echo Starting Atlas Brain...
start "ATLAS BRAIN" cmd /c "python BRAIN_ATLAS.py"

echo.
echo All brains started in separate windows.
echo.
pause
