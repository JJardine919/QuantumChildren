# ============================================================================
# DEPLOY_AND_COMPILE_V2.ps1
# Full deployment + compilation for all 6 MT5 terminals
# ============================================================================

$ErrorActionPreference = "Continue"

$PROJECT = "C:\Users\jimjj\Music\QuantumChildren"
$QTL = "$PROJECT\QuantumTradingLibrary"
$TERMINAL_BASE = "C:\Users\jimjj\AppData\Roaming\MetaQuotes\Terminal"

# Terminal map
$terminalMap = @{
    "4613C16E0E09DDABD5134D36D83110E5" = "Blue Guardian Challenge"
    "59C07D676775FCCF79E223EC24AB0D86" = "Blue Guardian Instant"
    "81A933A9AFC5DE3C23B15CAB19C63850" = "FTMO"
    "99B22614732346ADC2C7FD4EA1646D37" = "GetLeveraged"
    "D0E8209F77C8CF37AD8BF550E51FF075" = "Default MT5"
    "F6E5FFA163BE6F3F89ECBCA1BA487B55" = "Atlas Funded"
}

$editorMap = @{
    "4613C16E0E09DDABD5134D36D83110E5" = "C:\Program Files\Blue Guardian MT5 Terminal 2\MetaEditor64.exe"
    "59C07D676775FCCF79E223EC24AB0D86" = "C:\Program Files\Blue Guardian MT5 Terminal\MetaEditor64.exe"
    "81A933A9AFC5DE3C23B15CAB19C63850" = "C:\Program Files\FTMO Global Markets MT5 Terminal\MetaEditor64.exe"
    "99B22614732346ADC2C7FD4EA1646D37" = "C:\Program Files\GetLeveraged MT5 Terminal\MetaEditor64.exe"
    "D0E8209F77C8CF37AD8BF550E51FF075" = "C:\Program Files\MetaTrader 5\MetaEditor64.exe"
    "F6E5FFA163BE6F3F89ECBCA1BA487B55" = "C:\Program Files\Atlas Funded MT5 Terminal\MetaEditor64.exe"
}

# Build copy list as hashtable with source -> relative destination
$copies = @{}

# Standalone (BlueGuardian family)
$copies["$PROJECT\DEPLOY\BG_AtlasGrid_JardinesGate.mq5"]     = "Standalone\BG_AtlasGrid_JardinesGate.mq5"
$copies["$PROJECT\DEPLOY\BG_AtlasGrid_Original.mq5"]          = "Standalone\BG_AtlasGrid_Original.mq5"
$copies["$PROJECT\DEPLOY\BlueGuardian_Elite.mq5"]              = "Standalone\BlueGuardian_Elite.mq5"
$copies["$PROJECT\DEPLOY\BlueGuardian_Dynamic.mq5"]            = "Standalone\BlueGuardian_Dynamic.mq5"
$copies["$PROJECT\DEPLOY\BG_AtlasGrid.mq5"]                   = "Standalone\BG_AtlasGrid.mq5"
$copies["$QTL\BlueGuardian_Deploy\BG_SimpleGrid.mq5"]          = "Standalone\BG_SimpleGrid.mq5"
$copies["$QTL\BlueGuardian_Deploy\BG_AggressiveCompetition.mq5"] = "Standalone\BG_AggressiveCompetition.mq5"
$copies["$QTL\BlueGuardian_Deploy\BG_AtlasStyle.mq5"]          = "Standalone\BG_AtlasStyle.mq5"
$copies["$QTL\BlueGuardian_Quantum.mq5"]                       = "Standalone\BlueGuardian_Quantum.mq5"
$copies["$QTL\BG_Executor.mq5"]                                = "Standalone\BG_Executor.mq5"
$copies["$QTL\DataExporter.mq5"]                                = "Standalone\DataExporter.mq5"
$copies["$QTL\BlueGuardian_Deploy\BG_DataExporter.mq5"]         = "Standalone\BG_DataExporter.mq5"
$copies["$QTL\BlueGuardian_Deploy\BG_Diagnostic.mq5"]           = "Standalone\BG_Diagnostic.mq5"
$copies["$QTL\BlueGuardian_Deploy\BG_ForceTrade.mq5"]           = "Standalone\BG_ForceTrade.mq5"
$copies["$QTL\BlueGuardian_Deploy\BG_MultiExecutor.mq5"]        = "Standalone\BG_MultiExecutor.mq5"
$copies["$QTL\Include\JardinesGate.mqh"]                        = "Standalone\JardinesGate.mqh"
$copies["$QTL\Include\QuantumEdgeFilter.mqh"]                   = "Standalone\QuantumEdgeFilter.mqh"

# EntropyGrid
$copies["$PROJECT\DEPLOY\BTCUSD_GridTrader.mq5"]              = "EntropyGrid\BTCUSD_GridTrader.mq5"
$copies["$PROJECT\DEPLOY\ETHUSD_GridTrader.mq5"]              = "EntropyGrid\ETHUSD_GridTrader.mq5"
$copies["$PROJECT\DEPLOY\XAUUSD_GridTrader.mq5"]              = "EntropyGrid\XAUUSD_GridTrader.mq5"
$copies["$PROJECT\DEPLOY\MultiSymbol_Launcher.mq5"]            = "EntropyGrid\MultiSymbol_Launcher.mq5"
$copies["$PROJECT\DEPLOY\EntropyGridCore.mqh"]                 = "EntropyGrid\EntropyGridCore.mqh"

# Also copy EntropyGrid from GetLeveraged_Grid sources (latest)
$copies["$QTL\GetLeveraged_Grid\BTCUSD_GridTrader.mq5"]       = "EntropyGrid\BTCUSD_GridTrader.mq5"
$copies["$QTL\GetLeveraged_Grid\ETHUSD_GridTrader.mq5"]       = "EntropyGrid\ETHUSD_GridTrader.mq5"
$copies["$QTL\GetLeveraged_Grid\XAUUSD_GridTrader.mq5"]       = "EntropyGrid\XAUUSD_GridTrader.mq5"
$copies["$QTL\GetLeveraged_Grid\MultiSymbol_Launcher.mq5"]     = "EntropyGrid\MultiSymbol_Launcher.mq5"
$copies["$QTL\GetLeveraged_Grid\EntropyGridCore.mqh"]          = "EntropyGrid\EntropyGridCore.mqh"

# GridMaster300K
$copies["$QTL\GridMaster300K\GridMaster_Orchestrator.mq5"]     = "GridMaster300K\GridMaster_Orchestrator.mq5"
$copies["$QTL\GridMaster300K\GridExpert_Bullish.mq5"]          = "GridMaster300K\GridExpert_Bullish.mq5"
$copies["$QTL\GridMaster300K\GridExpert_Bearish.mq5"]          = "GridMaster300K\GridExpert_Bearish.mq5"
$copies["$QTL\GridMaster300K\GridExpert_Neutral.mq5"]          = "GridMaster300K\GridExpert_Neutral.mq5"
$copies["$QTL\GridMaster300K\GridCore.mqh"]                    = "GridMaster300K\GridCore.mqh"

# XAUUSD GridMaster
$copies["$QTL\XAUUSD_GridMaster\XAUUSD_GridMaster.mq5"]       = "XAUUSD_GridMaster\XAUUSD_GridMaster.mq5"
$copies["$QTL\XAUUSD_GridMaster\XAUUSD_GridCore.mqh"]         = "XAUUSD_GridMaster\XAUUSD_GridCore.mqh"

# StrikeBoss
$copies["$PROJECT\my-trading-work\StrikeBot_AI3.mq5"]          = "StrikeBoss\StrikeBot_AI3.mq5"
$copies["$PROJECT\StrikeBOSS\FinMaster_Fixed.mq5"]             = "StrikeBoss\FinMaster_Fixed.mq5"
$copies["$PROJECT\StrikeBOSS\OnnxHandler.mqh"]                 = "StrikeBoss\OnnxHandler.mqh"
$copies["$PROJECT\StrikeBOSS\QuantumAI_FIXED.mq5"]             = "StrikeBoss\QuantumAI_FIXED.mq5"
$copies["$PROJECT\StrikeBOSS\StrikeBOSS_Reality_Test.mq5"]     = "StrikeBoss\StrikeBOSS_Reality_Test.mq5"

# NexaAI
$copies["$PROJECT\my-trading-work\Nexa_AI_Minimal_Relay_MarketReady_FINAL_v2.07.mq5"] = "NexaAI\Nexa_AI_Minimal_Relay_MarketReady_FINAL_v2.07.mq5"
$copies["$PROJECT\my-trading-work\Nexa_AI_Autonomous.mq5"]                             = "NexaAI\Nexa_AI_Autonomous.mq5"
$copies["$PROJECT\my-trading-work\Nexa_AI_Market_ValidationReady_v1_05.mq5"]            = "NexaAI\Nexa_AI_Market_ValidationReady_v1_05.mq5"
$copies["$PROJECT\my-trading-work\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED.mq5"]          = "NexaAI\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED.mq5"
$copies["$PROJECT\my-trading-work\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED_CLEANED.mq5"]  = "NexaAI\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED_CLEANED.mq5"
$copies["$PROJECT\my-trading-work\Nexa_AI_Only_FINAL_RELAYFIXED.mq5"]                   = "NexaAI\Nexa_AI_Only_FINAL_RELAYFIXED.mq5"
$copies["$PROJECT\my-trading-work\nexa_ai_validation_optimized.mq5"]                     = "NexaAI\nexa_ai_validation_optimized.mq5"
$copies["$PROJECT\my-trading-work\SUSTAI_AI_Bot (1).mq5"]                                = "NexaAI\SUSTAI_AI_Bot_v1.mq5"
$copies["$PROJECT\my-trading-work\SUSTAI_AI_Bot (2).mq5"]                                = "NexaAI\SUSTAI_AI_Bot_v2.mq5"

# MyTradingWork
$copies["$PROJECT\my-trading-work\SUSTAI_AI_Bot_Pro.mq5"]                  = "MyTradingWork\SUSTAI_AI_Bot_Pro.mq5"
$copies["$PROJECT\my-trading-work\OnnxHandler.mqh"]                         = "MyTradingWork\OnnxHandler.mqh"
$copies["$PROJECT\my-trading-work\NeuralNetwork.mqh"]                       = "MyTradingWork\NeuralNetwork.mqh"
$copies["$PROJECT\my-trading-work\Adaptive_Bitcoin_Master_Bot_Final.mq5"]   = "MyTradingWork\Adaptive_Bitcoin_Master_Bot_Final.mq5"
$copies["$PROJECT\my-trading-work\Scalping-EA-MT5.mq5"]                    = "MyTradingWork\Scalping-EA-MT5.mq5"
$copies["$PROJECT\my-trading-work\DPO_Val.mq5"]                            = "MyTradingWork\DPO_Val.mq5"
$copies["$PROJECT\my-trading-work\DPO_Zero_Crossover.mq5"]                 = "MyTradingWork\DPO_Zero_Crossover.mq5"
$copies["$PROJECT\my-trading-work\Neural_Networks_Propagation_EA.mq5"]      = "MyTradingWork\Neural_Networks_Propagation_EA.mq5"
$copies["$PROJECT\my-trading-work\neuropro.mq5"]                            = "MyTradingWork\neuropro.mq5"
$copies["$PROJECT\my-trading-work\ExpertAdvisor.mq5"]                       = "MyTradingWork\ExpertAdvisor.mq5"

# WizardTeks
$copies["$PROJECT\WIZARD TEKS(PART 82)\WZ_82.mq5"]       = "WizardTeks\WZ_82.mq5"

# SignalWZ_82.mqh, SRI\PipeLine.mqh, and ONNX resources go to Include paths (handled in Phase 1 loop)
$signalSrc = "$PROJECT\WIZARD TEKS(PART 82)\SignalWZ_82.mqh"
$pipelineSrc = "$PROJECT\WIZARD TEKS(PART 82)\SRI\PipeLine.mqh"
$onnxResources = @(
    "$PROJECT\WIZARD TEKS(PART 82)\Python\82_1_0.onnx",
    "$PROJECT\WIZARD TEKS(PART 82)\Python\82_4_0.onnx",
    "$PROJECT\WIZARD TEKS(PART 82)\Python\82_5_0.onnx"
)

$subfolders = @("Standalone", "EntropyGrid", "GridMaster300K", "XAUUSD_GridMaster", "StrikeBoss", "NexaAI", "MyTradingWork", "WizardTeks")

$totalCopied = 0
$totalMissing = 0
$totalCompiled = 0
$totalCompileFail = 0
$failedFiles = @()

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  QUANTUM CHILDREN - DEPLOY + COMPILE V2" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# PHASE 1: COPY FILES TO ALL TERMINALS
# ============================================================================
Write-Host "PHASE 1: FILE DEPLOYMENT" -ForegroundColor Yellow
Write-Host "========================" -ForegroundColor Yellow

foreach ($hash in $terminalMap.Keys | Sort-Object) {
    $name = $terminalMap[$hash]
    $qcPath = "$TERMINAL_BASE\$hash\MQL5\Experts\QuantumChildren"
    $mql5Path = "$TERMINAL_BASE\$hash\MQL5"

    Write-Host ""
    Write-Host "--- $name ---" -ForegroundColor Green

    # Create subfolders
    foreach ($sub in $subfolders) {
        $dir = "$qcPath\$sub"
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }

    # Create Include signal path
    $signalDir = "$mql5Path\Include\Expert\Signal\My"
    if (-not (Test-Path $signalDir)) {
        New-Item -ItemType Directory -Path $signalDir -Force | Out-Null
    }

    $termCopied = 0
    $termMissing = 0

    # Copy all files
    foreach ($src in $copies.Keys) {
        $relDest = $copies[$src]
        $dst = "$qcPath\$relDest"
        $fname = Split-Path $src -Leaf

        if (Test-Path $src) {
            $dstDir = Split-Path $dst -Parent
            if (-not (Test-Path $dstDir)) {
                New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
            }
            Copy-Item -Path $src -Destination $dst -Force
            $termCopied++
            $totalCopied++
        } else {
            $termMissing++
            $totalMissing++
        }
    }

    # Copy SignalWZ_82.mqh to Include path
    if (Test-Path $signalSrc) {
        Copy-Item -Path $signalSrc -Destination "$signalDir\SignalWZ_82.mqh" -Force
        $termCopied++
        $totalCopied++
    }

    # Copy SRI\PipeLine.mqh to Include path
    $sriDir = "$mql5Path\Include\SRI"
    if (-not (Test-Path $sriDir)) {
        New-Item -ItemType Directory -Path $sriDir -Force | Out-Null
    }
    if (Test-Path $pipelineSrc) {
        Copy-Item -Path $pipelineSrc -Destination "$sriDir\PipeLine.mqh" -Force
        $termCopied++
        $totalCopied++
    }

    # Copy ONNX resource files to Python/ next to SignalWZ_82.mqh
    $pythonDir = "$signalDir\Python"
    if (-not (Test-Path $pythonDir)) {
        New-Item -ItemType Directory -Path $pythonDir -Force | Out-Null
    }
    foreach ($onnxFile in $onnxResources) {
        if (Test-Path $onnxFile) {
            $onnxName = Split-Path $onnxFile -Leaf
            Copy-Item -Path $onnxFile -Destination "$pythonDir\$onnxName" -Force
            $termCopied++
            $totalCopied++
        }
    }

    Write-Host "  Copied: $termCopied files" -ForegroundColor White
    if ($termMissing -gt 0) {
        Write-Host "  Missing sources: $termMissing" -ForegroundColor DarkYellow
    }
}

Write-Host ""
Write-Host "PHASE 1 TOTAL: $totalCopied files copied across 6 terminals" -ForegroundColor Yellow
if ($totalMissing -gt 0) {
    Write-Host "  ($totalMissing source files not found - likely from my-trading-work)" -ForegroundColor DarkYellow
}

# ============================================================================
# PHASE 2: COMPILE ALL .mq5 FILES
# ============================================================================
Write-Host ""
Write-Host "PHASE 2: COMPILATION" -ForegroundColor Yellow
Write-Host "====================" -ForegroundColor Yellow

foreach ($hash in $terminalMap.Keys | Sort-Object) {
    $name = $terminalMap[$hash]
    $qcPath = "$TERMINAL_BASE\$hash\MQL5\Experts\QuantumChildren"
    $mql5Path = "$TERMINAL_BASE\$hash\MQL5"
    $editor = $editorMap[$hash]

    Write-Host ""
    Write-Host "--- Compiling: $name ---" -ForegroundColor Green

    if (-not (Test-Path $editor)) {
        Write-Host "  [SKIP] MetaEditor not found: $editor" -ForegroundColor Red
        continue
    }

    $mq5Files = Get-ChildItem -Path $qcPath -Filter "*.mq5" -Recurse -ErrorAction SilentlyContinue

    if ($null -eq $mq5Files -or $mq5Files.Count -eq 0) {
        Write-Host "  [SKIP] No .mq5 files" -ForegroundColor DarkYellow
        continue
    }

    Write-Host "  $($mq5Files.Count) files to compile..."

    foreach ($mq5 in $mq5Files) {
        $fpath = $mq5.FullName
        $fname = $mq5.Name
        $logFile = [System.IO.Path]::ChangeExtension($fpath, ".log")

        if (Test-Path $logFile) { Remove-Item $logFile -Force -ErrorAction SilentlyContinue }

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $editor
        $psi.Arguments = "/compile:`"$fpath`" /log:`"$logFile`" /inc:`"$mql5Path`""
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true

        $proc = [System.Diagnostics.Process]::Start($psi)
        $proc.WaitForExit(30000) | Out-Null

        $success = $false
        $errorDetail = ""

        if (Test-Path $logFile) {
            $logContent = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
            if ($logContent -match "0 error") {
                $success = $true
            } else {
                $errorLines = Get-Content $logFile -ErrorAction SilentlyContinue | Where-Object { $_ -match "error" } | Select-Object -First 2
                $errorDetail = ($errorLines -join " | ")
            }
        }

        $ex5Path = [System.IO.Path]::ChangeExtension($fpath, ".ex5")
        if (Test-Path $ex5Path) { $success = $true }

        if ($success) {
            Write-Host "  [ OK ] $fname" -ForegroundColor White
            $totalCompiled++
        } else {
            Write-Host "  [FAIL] $fname" -ForegroundColor Red
            if ($errorDetail) {
                Write-Host "         $errorDetail" -ForegroundColor DarkRed
            }
            $totalCompileFail++
            $failedFiles += "$name | $fname | $errorDetail"
        }
    }
}

# ============================================================================
# FINAL REPORT
# ============================================================================
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  FINAL REPORT" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Files copied:        $totalCopied (across 6 terminals)" -ForegroundColor White
Write-Host "  Compiled OK:         $totalCompiled" -ForegroundColor Green
Write-Host "  Compile failures:    $totalCompileFail" -ForegroundColor $(if ($totalCompileFail -gt 0) { "Red" } else { "Green" })

if ($failedFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "  FAILURES:" -ForegroundColor Red
    foreach ($f in $failedFiles) {
        Write-Host "    $f" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "  Terminals:" -ForegroundColor White
Write-Host "    4613...E5  Blue Guardian Challenge" -ForegroundColor White
Write-Host "    59C0...86  Blue Guardian Instant" -ForegroundColor White
Write-Host "    81A9...50  FTMO" -ForegroundColor White
Write-Host "    99B2...37  GetLeveraged" -ForegroundColor White
Write-Host "    D0E8...75  Default MT5" -ForegroundColor White
Write-Host "    F6E5...55  Atlas Funded" -ForegroundColor White
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
