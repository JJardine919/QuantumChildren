# ============================================================================
# QUANTUM CHILDREN - NETWORK TEST SCRIPT
# ============================================================================
# Sends test signals and verifies the network is working
# ============================================================================

$N8N_URL = "http://localhost:5678"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  QUANTUM CHILDREN NETWORK - TEST SUITE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$passed = 0
$failed = 0

# Test 1: n8n Health
Write-Host "[TEST 1] n8n Health Check..." -NoNewline
try {
    $health = Invoke-RestMethod -Uri "$N8N_URL/healthz" -TimeoutSec 5
    Write-Host " PASSED" -ForegroundColor Green
    $passed++
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "         n8n is not running or not responding"
    $failed++
}

# Test 2: Dashboard API
Write-Host "[TEST 2] Dashboard API..." -NoNewline
try {
    $dashboard = Invoke-RestMethod -Uri "$N8N_URL/webhook/signals" -Method Get -TimeoutSec 5
    Write-Host " PASSED" -ForegroundColor Green
    Write-Host "         Status: $($dashboard.status), Node: $($dashboard.node_id)"
    $passed++
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "         Dashboard webhook not responding (workflow may not be active)"
    $failed++
}

# Test 3: Signal Reception
Write-Host "[TEST 3] Signal Reception..." -NoNewline
$testSignal = @{
    symbol = "BTCUSD"
    action = "BUY"
    confidence = 0.75
    quantum_entropy = 2.8
    dominant_state = 0.08
    catboost_prob = 0.72
    llm_adjustment = 0.03
    timestamp = (Get-Date).ToString("o")
    source_node = "QC-TEST"
    cascade_hop = 0
    cascade_path = @("QC-TEST")
    test_signal = $true
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$N8N_URL/webhook/quantum-signal" -Method Post -Body $testSignal -ContentType "application/json" -TimeoutSec 10
    Write-Host " PASSED" -ForegroundColor Green
    Write-Host "         Signal received and processed"
    $passed++
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "         Signal webhook not responding"
    $failed++
}

# Test 4: Cascade Endpoint
Write-Host "[TEST 4] Cascade Relay..." -NoNewline
$cascadeSignal = @{
    symbol = "ETHUSD"
    action = "SELL"
    confidence = 0.68
    quantum_entropy = 3.2
    timestamp = (Get-Date).ToString("o")
    source_node = "QC-REMOTE"
    cascade_hop = 1
    cascade_path = @("QC-REMOTE", "QC-TEST")
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$N8N_URL/webhook/cascade-signal" -Method Post -Body $cascadeSignal -ContentType "application/json" -TimeoutSec 10
    Write-Host " PASSED" -ForegroundColor Green
    Write-Host "         Cascade hop: $($response.hop)"
    $passed++
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "         Cascade webhook not responding"
    $failed++
}

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  TEST RESULTS: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if ($failed -eq 0) {
    Write-Host "  All systems operational!" -ForegroundColor Green
    Write-Host "  The Quantum Children network is ready for signals." -ForegroundColor Green
} else {
    Write-Host "  Some tests failed. Troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "  1. Make sure n8n is running (check for n8n window)"
    Write-Host "  2. Open $N8N_URL in browser"
    Write-Host "  3. Make sure the workflow is ACTIVE (toggle switch is ON)"
    Write-Host "  4. Try importing the workflow again if needed"
}

Write-Host ""
Write-Host "Press any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
