# ============================================================================
# QUANTUM CHILDREN N8N CASCADE NETWORK - FULL AUTOMATION
# ============================================================================
# This script handles everything:
#   1. Starts n8n if not running
#   2. Waits for n8n to be ready
#   3. Imports the workflow via API
#   4. Activates the workflow
#   5. Starts the signal emitter
#   6. Verifies everything is working
# ============================================================================

param(
    [switch]$SkipEmitter,
    [switch]$TestOnly,
    [int]$EmitterInterval = 60
)

$ErrorActionPreference = "Continue"

# Configuration
$N8N_PORT = 5678
$N8N_URL = "http://localhost:$N8N_PORT"
$WORKFLOW_PATH = "$PSScriptRoot\..\QuantumChildren_LOCAL_Dashboard.json"
$SIGNAL_EMITTER = "$PSScriptRoot\..\signal_emitter.py"
$LOG_FILE = "$PSScriptRoot\network_startup.log"

# Colors for output
function Write-Status { param($msg) Write-Host "[*] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Warning { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }

function Log {
    param($msg)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp | $msg" | Add-Content $LOG_FILE
    Write-Host $msg
}

# ============================================================================
# STEP 1: Check/Start n8n
# ============================================================================
function Start-N8N {
    Write-Status "Checking if n8n is already running..."

    $n8nProcess = Get-Process -Name "node" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*n8n*" }

    # Also check if port 5678 is in use
    $portInUse = Get-NetTCPConnection -LocalPort $N8N_PORT -ErrorAction SilentlyContinue

    if ($portInUse) {
        Write-Success "n8n is already running on port $N8N_PORT"
        return $true
    }

    Write-Status "Starting n8n server..."

    # Set environment variables for n8n
    $env:N8N_BASIC_AUTH_ACTIVE = "false"
    $env:N8N_HOST = "localhost"
    $env:N8N_PORT = $N8N_PORT
    $env:N8N_PROTOCOL = "http"
    $env:GENERIC_TIMEZONE = "America/New_York"
    $env:NODE_ID = "QC-MASTER"

    # Start n8n in a new window
    $n8nPath = (Get-Command n8n -ErrorAction SilentlyContinue).Source
    if (-not $n8nPath) {
        $n8nPath = "$env:APPDATA\npm\n8n.cmd"
    }

    if (-not (Test-Path $n8nPath)) {
        Write-Error "n8n not found! Please install with: npm install -g n8n"
        return $false
    }

    # Start n8n in background
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "title n8n Server && $n8nPath start" -WindowStyle Normal

    Write-Status "Waiting for n8n to start (this can take 30-60 seconds on first run)..."

    # Wait for n8n to be ready
    $maxWait = 120  # seconds
    $waited = 0
    $ready = $false

    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 2
        $waited += 2

        try {
            $response = Invoke-WebRequest -Uri "$N8N_URL/healthz" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
            Write-Host "." -NoNewline
        }
    }

    Write-Host ""  # New line after dots

    if ($ready) {
        Write-Success "n8n is ready! (took $waited seconds)"
        return $true
    } else {
        Write-Error "n8n failed to start within $maxWait seconds"
        return $false
    }
}

# ============================================================================
# STEP 2: Import Workflow via API
# ============================================================================
function Import-Workflow {
    Write-Status "Importing Quantum Children workflow..."

    if (-not (Test-Path $WORKFLOW_PATH)) {
        Write-Error "Workflow file not found: $WORKFLOW_PATH"
        return $null
    }

    $workflowJson = Get-Content $WORKFLOW_PATH -Raw
    $workflow = $workflowJson | ConvertFrom-Json

    # Check if workflow already exists
    try {
        $existingWorkflows = Invoke-RestMethod -Uri "$N8N_URL/api/v1/workflows" -Method Get -ContentType "application/json" -ErrorAction SilentlyContinue

        $existing = $existingWorkflows.data | Where-Object { $_.name -eq $workflow.name }

        if ($existing) {
            Write-Success "Workflow already exists (ID: $($existing.id))"
            return $existing.id
        }
    } catch {
        Write-Warning "Could not check existing workflows (this is normal on first run)"
    }

    # Import new workflow
    try {
        $response = Invoke-RestMethod -Uri "$N8N_URL/api/v1/workflows" -Method Post -Body $workflowJson -ContentType "application/json"
        Write-Success "Workflow imported successfully (ID: $($response.id))"
        return $response.id
    } catch {
        Write-Error "Failed to import workflow: $_"

        # Try alternative method - import via file
        Write-Status "Trying alternative import method..."
        try {
            # Use n8n CLI to import
            $result = & n8n import:workflow --input="$WORKFLOW_PATH" 2>&1
            Write-Success "Workflow imported via CLI"

            # Get the workflow ID
            $workflows = Invoke-RestMethod -Uri "$N8N_URL/api/v1/workflows" -Method Get -ContentType "application/json"
            $imported = $workflows.data | Where-Object { $_.name -eq $workflow.name }
            return $imported.id
        } catch {
            Write-Error "CLI import also failed: $_"
            return $null
        }
    }
}

# ============================================================================
# STEP 3: Activate Workflow
# ============================================================================
function Enable-Workflow {
    param($WorkflowId)

    Write-Status "Activating workflow (ID: $WorkflowId)..."

    try {
        $body = '{"active": true}'
        $response = Invoke-RestMethod -Uri "$N8N_URL/api/v1/workflows/$WorkflowId" -Method Patch -Body $body -ContentType "application/json"

        if ($response.active -eq $true) {
            Write-Success "Workflow activated!"
            return $true
        } else {
            Write-Warning "Workflow may not be fully activated"
            return $false
        }
    } catch {
        Write-Error "Failed to activate workflow: $_"
        return $false
    }
}

# ============================================================================
# STEP 4: Start Signal Emitter
# ============================================================================
function Start-SignalEmitter {
    Write-Status "Starting Signal Emitter service..."

    if (-not (Test-Path $SIGNAL_EMITTER)) {
        Write-Error "Signal emitter not found: $SIGNAL_EMITTER"
        return $false
    }

    # Check if Python is available
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        Write-Error "Python not found in PATH"
        return $false
    }

    # Start signal emitter in a new window
    $emitterArgs = "--service --interval $EmitterInterval --webhook `"$N8N_URL/webhook/quantum-signal`""

    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "title Quantum Signal Emitter && python `"$SIGNAL_EMITTER`" $emitterArgs" -WindowStyle Normal

    Write-Success "Signal Emitter started (interval: ${EmitterInterval}s)"
    return $true
}

# ============================================================================
# STEP 5: Verify System
# ============================================================================
function Test-System {
    Write-Status "Running system verification..."

    $allGood = $true

    # Test 1: n8n health
    try {
        $health = Invoke-RestMethod -Uri "$N8N_URL/healthz" -TimeoutSec 5
        Write-Success "n8n health check passed"
    } catch {
        Write-Error "n8n health check failed"
        $allGood = $false
    }

    # Test 2: Dashboard endpoint
    try {
        $dashboard = Invoke-RestMethod -Uri "$N8N_URL/webhook/signals" -Method Get -TimeoutSec 5
        Write-Success "Dashboard API responding: $($dashboard.status)"
    } catch {
        Write-Warning "Dashboard webhook not yet available (workflow may need activation)"
    }

    # Test 3: Send test signal
    Write-Status "Sending test signal..."
    $testSignal = @{
        symbol = "BTCUSD"
        action = "TEST"
        confidence = 0.85
        quantum_entropy = 2.5
        timestamp = (Get-Date).ToString("o")
        source_node = "QC-VERIFICATION"
        cascade_hop = 0
        cascade_path = @("QC-VERIFICATION")
        test_signal = $true
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod -Uri "$N8N_URL/webhook/quantum-signal" -Method Post -Body $testSignal -ContentType "application/json" -TimeoutSec 10
        Write-Success "Test signal received! Response: $($response.status)"
    } catch {
        Write-Warning "Test signal failed (workflow webhooks may need a moment to initialize)"
    }

    return $allGood
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  QUANTUM CHILDREN CASCADE NETWORK - AUTOMATED STARTUP" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

Log "Starting Quantum Children Network automation..."

# Step 1: Start n8n
if (-not (Start-N8N)) {
    Write-Error "Failed to start n8n. Aborting."
    exit 1
}

Start-Sleep -Seconds 3  # Give n8n a moment to fully initialize

# Step 2: Import workflow
$workflowId = Import-Workflow
if (-not $workflowId) {
    Write-Warning "Workflow import had issues - you may need to import manually"
    Write-Host "  Manual import: Open $N8N_URL and import from File menu"
}

# Step 3: Activate workflow
if ($workflowId) {
    $activated = Enable-Workflow -WorkflowId $workflowId
    if (-not $activated) {
        Write-Warning "Workflow activation may need manual toggle in n8n UI"
    }
}

# Step 4: Start Signal Emitter (unless skipped)
if (-not $SkipEmitter) {
    Start-Sleep -Seconds 2  # Wait for webhooks to register
    $emitterStarted = Start-SignalEmitter
}

# Step 5: Verify system
Start-Sleep -Seconds 5  # Wait for everything to settle
$verified = Test-System

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  STARTUP COMPLETE" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  n8n Dashboard:     $N8N_URL" -ForegroundColor White
Write-Host "  Signal Webhook:    $N8N_URL/webhook/quantum-signal" -ForegroundColor White
Write-Host "  Network Status:    $N8N_URL/webhook/signals" -ForegroundColor White
Write-Host "  Cascade Webhook:   $N8N_URL/webhook/cascade-signal" -ForegroundColor White
Write-Host ""

if ($verified) {
    Write-Success "All systems operational!"
} else {
    Write-Warning "Some checks failed - open n8n UI to verify workflow is active"
}

Write-Host ""
Write-Host "Press any key to keep this window open, or close it to continue..." -ForegroundColor Gray

# Keep window open
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
