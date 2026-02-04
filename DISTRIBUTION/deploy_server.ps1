# QuantumChildren Server Deployment Script
# Run this to deploy the collection server to your VPS

$VPS_IP = "203.161.61.61"
$VPS_USER = "root"
$REMOTE_DIR = "/opt/quantumchildren"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  QUANTUM CHILDREN - Server Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ServerDir = Join-Path $ScriptDir "SERVER"

Write-Host "[1/4] Creating remote directory..." -ForegroundColor Yellow
ssh ${VPS_USER}@${VPS_IP} "mkdir -p ${REMOTE_DIR}"

Write-Host "[2/4] Uploading server files..." -ForegroundColor Yellow
scp "${ServerDir}\collection_server.py" "${VPS_USER}@${VPS_IP}:${REMOTE_DIR}/"
scp "${ServerDir}\requirements.txt" "${VPS_USER}@${VPS_IP}:${REMOTE_DIR}/"
scp "${ServerDir}\start_server.sh" "${VPS_USER}@${VPS_IP}:${REMOTE_DIR}/"

Write-Host "[3/4] Installing dependencies..." -ForegroundColor Yellow
ssh ${VPS_USER}@${VPS_IP} "cd ${REMOTE_DIR} && pip3 install -r requirements.txt"

Write-Host "[4/4] Starting server..." -ForegroundColor Yellow
ssh ${VPS_USER}@${VPS_IP} "cd ${REMOTE_DIR} && chmod +x start_server.sh && nohup python3 collection_server.py > server.log 2>&1 &"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Server running at: http://${VPS_IP}:8888" -ForegroundColor White
Write-Host "Check stats: http://${VPS_IP}:8888/stats" -ForegroundColor White
Write-Host ""
Write-Host "To check if it's running:" -ForegroundColor Gray
Write-Host "  ssh ${VPS_USER}@${VPS_IP} 'ps aux | grep collection_server'" -ForegroundColor Gray
