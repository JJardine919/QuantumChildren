# ============================================================================
# QUANTUM CHILDREN - SIMPLE N8N STARTER
# ============================================================================
# This is a simpler version that just starts n8n and opens the browser.
# Use this if the full automation has issues.
# ============================================================================

$N8N_PORT = 5678
$N8N_URL = "http://localhost:$N8N_PORT"
$WORKFLOW_PATH = "$PSScriptRoot\..\QuantumChildren_LOCAL_Dashboard.json"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  QUANTUM CHILDREN - SIMPLE N8N STARTER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if n8n is already running
$portInUse = Get-NetTCPConnection -LocalPort $N8N_PORT -ErrorAction SilentlyContinue

if ($portInUse) {
    Write-Host "[OK] n8n is already running!" -ForegroundColor Green
} else {
    Write-Host "[*] Starting n8n..." -ForegroundColor Yellow

    # Set environment
    $env:N8N_BASIC_AUTH_ACTIVE = "false"
    $env:NODE_ID = "QC-MASTER"

    # Start n8n
    $n8nPath = "$env:APPDATA\npm\n8n.cmd"
    if (Test-Path $n8nPath) {
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "title n8n Server && `"$n8nPath`" start" -WindowStyle Normal
    } else {
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "title n8n Server && n8n start" -WindowStyle Normal
    }

    Write-Host "[*] Waiting for n8n to start..." -ForegroundColor Yellow

    # Wait for n8n
    $waited = 0
    while ($waited -lt 90) {
        Start-Sleep -Seconds 3
        $waited += 3
        Write-Host "." -NoNewline

        try {
            $response = Invoke-WebRequest -Uri "$N8N_URL/healthz" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host ""
                Write-Host "[OK] n8n is ready!" -ForegroundColor Green
                break
            }
        } catch {}
    }
}

# Wait a moment then open browser
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "[*] Opening n8n in browser..." -ForegroundColor Yellow
Start-Process "$N8N_URL"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  NEXT STEPS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. In n8n, click the menu (three lines) -> Import from file"
Write-Host ""
Write-Host "  2. Select this file:"
Write-Host "     $WORKFLOW_PATH" -ForegroundColor Yellow
Write-Host ""
Write-Host "  3. Click the toggle to ACTIVATE the workflow"
Write-Host ""
Write-Host "  4. Your endpoints will be:"
Write-Host "     - Send Signal:    $N8N_URL/webhook/quantum-signal" -ForegroundColor Green
Write-Host "     - Get Status:     $N8N_URL/webhook/signals" -ForegroundColor Green
Write-Host "     - Cascade:        $N8N_URL/webhook/cascade-signal" -ForegroundColor Green
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Copy workflow path to clipboard
$WORKFLOW_PATH | Set-Clipboard
Write-Host "[*] Workflow path copied to clipboard!" -ForegroundColor Magenta
Write-Host ""
Write-Host "Press any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
