@echo off
REM ============================================================
REM MODEL SYNC - Local PC to VPS
REM ============================================================
REM Transfers trained LSTM models to VPS for hot-reload by BRAIN_ATLAS.
REM IMPORTANT: Models go to staging first, manifest goes LAST.
REM
REM Usage: sync_models.bat [VPS_IP] [VPS_USER]
REM Example: sync_models.bat 45.67.89.101 admin
REM ============================================================

SET VPS_IP=%1
SET VPS_USER=%2
SET LOCAL_MODELS=C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\top_50_experts
SET REMOTE_BASE=C:\QuantumChildren\QuantumTradingLibrary
SET REMOTE_STAGING=%REMOTE_BASE%\top_50_experts_staging
SET REMOTE_MODELS=%REMOTE_BASE%\top_50_experts
SET REMOTE_BACKUP=%REMOTE_BASE%\top_50_experts_backup

IF "%VPS_IP%"=="" (
    echo ERROR: VPS IP required. Usage: sync_models.bat [VPS_IP] [VPS_USER]
    exit /b 1
)
IF "%VPS_USER%"=="" SET VPS_USER=admin

echo ============================================================
echo  MODEL SYNC - Quantum Children
echo  Local -> VPS (%VPS_IP%)
echo ============================================================
echo.

REM Step 1: Create staging directory on VPS
echo [1/5] Creating staging directory on VPS...
ssh %VPS_USER%@%VPS_IP% "if not exist \"%REMOTE_STAGING%\" mkdir \"%REMOTE_STAGING%\""

REM Step 2: Transfer all .pth model files to staging
echo [2/5] Transferring model files to staging...
scp "%LOCAL_MODELS%\*.pth" %VPS_USER%@%VPS_IP%:"%REMOTE_STAGING%\"
IF ERRORLEVEL 1 (
    echo ERROR: Model file transfer failed!
    exit /b 1
)

REM Step 3: Atomic swap on VPS (backup current, promote staging)
echo [3/5] Atomic swap on VPS (backup current, promote staging)...
ssh %VPS_USER%@%VPS_IP% "if exist \"%REMOTE_BACKUP%\" rmdir /s /q \"%REMOTE_BACKUP%\" && if exist \"%REMOTE_MODELS%\" rename \"%REMOTE_MODELS%\" top_50_experts_backup && rename \"%REMOTE_STAGING%\" top_50_experts"

REM Step 4: Transfer manifest LAST (triggers hot-reload)
echo [4/5] Transferring manifest (triggers hot-reload)...
scp "%LOCAL_MODELS%\top_50_manifest.json" %VPS_USER%@%VPS_IP%:"%REMOTE_MODELS%\"
IF ERRORLEVEL 1 (
    echo ERROR: Manifest transfer failed! Models may be inconsistent.
    exit /b 1
)

REM Step 5: Validate on VPS
echo [5/5] Validating models on VPS...
ssh %VPS_USER%@%VPS_IP% "cd \"%REMOTE_BASE%\" && venv\Scripts\python.exe validate_models.py"

echo.
echo ============================================================
echo  SYNC COMPLETE
echo  BRAIN_ATLAS will hot-reload on next cycle (~60 seconds)
echo ============================================================
