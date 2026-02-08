# ============================================================================
# QUANTUM CHILDREN - STOP NETWORK
# ============================================================================
# Cleanly stops all Quantum Children network services
# ============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Red
Write-Host "  QUANTUM CHILDREN NETWORK - SHUTDOWN" -ForegroundColor Red
Write-Host "============================================================" -ForegroundColor Red
Write-Host ""

# Find and stop n8n processes
Write-Host "[*] Stopping n8n processes..." -ForegroundColor Yellow
$n8nProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -like "*n8n*" -or $_.CommandLine -like "*n8n*" }

if ($n8nProcesses) {
    $n8nProcesses | Stop-Process -Force
    Write-Host "[OK] n8n processes stopped" -ForegroundColor Green
} else {
    Write-Host "[*] No n8n processes found" -ForegroundColor Gray
}

# Find and stop signal emitter
Write-Host "[*] Stopping Signal Emitter processes..." -ForegroundColor Yellow
$emitterProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -like "*Signal Emitter*" -or $_.CommandLine -like "*signal_emitter*" }

if ($emitterProcesses) {
    $emitterProcesses | Stop-Process -Force
    Write-Host "[OK] Signal Emitter stopped" -ForegroundColor Green
} else {
    Write-Host "[*] No Signal Emitter processes found" -ForegroundColor Gray
}

# Also try to stop any node processes on port 5678
$portProcess = Get-NetTCPConnection -LocalPort 5678 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess |
    ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }

if ($portProcess) {
    Write-Host "[*] Stopping process on port 5678..." -ForegroundColor Yellow
    $portProcess | Stop-Process -Force
    Write-Host "[OK] Port 5678 freed" -ForegroundColor Green
}

Write-Host ""
Write-Host "[OK] Quantum Children Network stopped" -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
