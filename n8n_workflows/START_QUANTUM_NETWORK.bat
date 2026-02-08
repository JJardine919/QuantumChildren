@echo off
:: ============================================================================
:: QUANTUM CHILDREN CASCADE NETWORK - ONE CLICK STARTER
:: ============================================================================
:: Just double-click this file. It handles everything.
:: ============================================================================

title Quantum Children Network Launcher

echo.
echo  ============================================================
echo   QUANTUM CHILDREN CASCADE NETWORK
echo   One-Click Automated Startup
echo  ============================================================
echo.

:: Set working directory to script location
cd /d "%~dp0"

:: Check for PowerShell
where powershell >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell not found!
    pause
    exit /b 1
)

:: Run the automation script with bypassed execution policy
echo [*] Launching automated network startup...
echo.

powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0automation\Start-QuantumNetwork.ps1"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Startup script encountered an error.
    echo         Check the output above for details.
    echo.
    pause
    exit /b 1
)

exit /b 0
