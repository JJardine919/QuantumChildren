# ============================================================
# ETARE Redux Brain Deployment Script (PowerShell)
# ============================================================
# Deploys the upgraded LSTM-based signal generator to VPS
#
# Usage: .\deploy_redux_brain.ps1
# ============================================================

$VPS_HOST = "72.62.170.153"
$VPS_USER = "root"
$VPS_PATH = "/root/quantum-brain"
$DOCKER_IMAGE = "quantum-brain-redux"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "ETARE Redux Brain Deployment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "VPS: $VPS_USER@$VPS_HOST"
Write-Host "Path: $VPS_PATH"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if champions exist
if (-not (Test-Path "champions\champions_manifest.json")) {
    Write-Host "ERROR: Champions not found! Run export_champions.py first." -ForegroundColor Red
    Write-Host "  python export_champions.py --output-dir ./champions"
    exit 1
}

Write-Host "[1/6] Verifying champions..." -ForegroundColor Yellow
python export_champions.py --verify
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Champion verification failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/6] Creating deployment package..." -ForegroundColor Yellow

# Create staging directory
$stagingDir = "redux_deploy_staging"
if (Test-Path $stagingDir) { Remove-Item -Recurse -Force $stagingDir }
New-Item -ItemType Directory -Path $stagingDir | Out-Null

# Copy required files
Copy-Item "Dockerfile.redux" "$stagingDir/Dockerfile"
Copy-Item "requirements.txt" "$stagingDir/"
Copy-Item -Recurse "champions" "$stagingDir/"
New-Item -ItemType Directory -Path "$stagingDir/01_Systems/System_03_ETARE" -Force | Out-Null
Copy-Item "01_Systems/System_03_ETARE/ETARE_Redux.py" "$stagingDir/01_Systems/System_03_ETARE/"
New-Item -ItemType Directory -Path "$stagingDir/06_Integration/HybridBridge" -Force | Out-Null
Copy-Item "06_Integration/HybridBridge/etare_signal_generator_redux.py" "$stagingDir/06_Integration/HybridBridge/"

# Also need ETARE_module.py for imports (even though we don't use the MLP)
Copy-Item "01_Systems/System_03_ETARE/ETARE_module.py" "$stagingDir/01_Systems/System_03_ETARE/"

Write-Host ""
Write-Host "[3/6] Uploading files to VPS..." -ForegroundColor Yellow
Write-Host "Creating remote directory..."
ssh "$VPS_USER@$VPS_HOST" "mkdir -p $VPS_PATH"

Write-Host "Uploading Dockerfile..."
scp "$stagingDir/Dockerfile" "${VPS_USER}@${VPS_HOST}:${VPS_PATH}/"

Write-Host "Uploading requirements.txt..."
scp "$stagingDir/requirements.txt" "${VPS_USER}@${VPS_HOST}:${VPS_PATH}/"

Write-Host "Uploading champions..."
scp -r "$stagingDir/champions" "${VPS_USER}@${VPS_HOST}:${VPS_PATH}/"

Write-Host "Uploading code..."
ssh "$VPS_USER@$VPS_HOST" "mkdir -p $VPS_PATH/01_Systems/System_03_ETARE $VPS_PATH/06_Integration/HybridBridge"
scp "$stagingDir/01_Systems/System_03_ETARE/ETARE_Redux.py" "${VPS_USER}@${VPS_HOST}:${VPS_PATH}/01_Systems/System_03_ETARE/"
scp "$stagingDir/01_Systems/System_03_ETARE/ETARE_module.py" "${VPS_USER}@${VPS_HOST}:${VPS_PATH}/01_Systems/System_03_ETARE/"
scp "$stagingDir/06_Integration/HybridBridge/etare_signal_generator_redux.py" "${VPS_USER}@${VPS_HOST}:${VPS_PATH}/06_Integration/HybridBridge/"

Write-Host ""
Write-Host "[4/6] Building Docker image on VPS..." -ForegroundColor Yellow
ssh "$VPS_USER@$VPS_HOST" "cd $VPS_PATH && docker build -t $DOCKER_IMAGE ."

Write-Host ""
Write-Host "[5/6] Stopping old container..." -ForegroundColor Yellow
ssh "$VPS_USER@$VPS_HOST" "docker stop quantum-brain 2>/dev/null || true; docker rm quantum-brain 2>/dev/null || true"

Write-Host ""
Write-Host "[6/6] Starting Redux container..." -ForegroundColor Yellow
$MT5_FILES = "/root/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files"
ssh "$VPS_USER@$VPS_HOST" "docker run -d --name quantum-brain --restart unless-stopped -v '${MT5_FILES}:${MT5_FILES}' $DOCKER_IMAGE"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To check status:" -ForegroundColor Cyan
Write-Host "  ssh $VPS_USER@$VPS_HOST `"docker logs -f quantum-brain`""
Write-Host ""
Write-Host "To view signals:" -ForegroundColor Cyan
Write-Host "  ssh $VPS_USER@$VPS_HOST `"cat '${MT5_FILES}/etare_signals.json'`""
Write-Host ""

# Cleanup
Remove-Item -Recurse -Force $stagingDir

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
