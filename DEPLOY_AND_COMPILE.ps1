# ============================================================================
# DEPLOY_AND_COMPILE.ps1
# Phase 1: Copy all EA source files to all 6 MT5 terminals
# Phase 2: Compile all .mq5 files using MetaEditor64
# ============================================================================

$ErrorActionPreference = "Continue"

$PROJECT = "C:\Users\jimjj\Music\QuantumChildren"
$DEPLOY = "$PROJECT\DEPLOY"
$QTL = "$PROJECT\QuantumTradingLibrary"
$BG_DEPLOY = "$QTL\BlueGuardian_Deploy"
$BG_EA = "$QTL\BlueGuardian_EA"
$GM300K = "$QTL\GridMaster300K"
$XAUUSD_GM = "$QTL\XAUUSD_GridMaster"
$GL_GRID = "$QTL\GetLeveraged_Grid"
$QTL_INCLUDE = "$QTL\Include"
$MTW = "$PROJECT\my-trading-work"
$STRIKEBOSS = "$PROJECT\StrikeBOSS"
$WIZARD = "$PROJECT\WIZARD TEKS(PART 82)"

$TERMINAL_BASE = "C:\Users\jimjj\AppData\Roaming\MetaQuotes\Terminal"

# Map terminal hashes to names and MetaEditor paths
$terminals = @{
    "4613C16E0E09DDABD5134D36D83110E5" = @{ Name = "Blue Guardian Challenge"; Editor = "C:\Program Files\Blue Guardian MT5 Terminal 2\MetaEditor64.exe" }
    "59C07D676775FCCF79E223EC24AB0D86" = @{ Name = "Blue Guardian Instant";   Editor = "C:\Program Files\Blue Guardian MT5 Terminal\MetaEditor64.exe" }
    "81A933A9AFC5DE3C23B15CAB19C63850" = @{ Name = "FTMO";                     Editor = "C:\Program Files\FTMO Global Markets MT5 Terminal\MetaEditor64.exe" }
    "99B22614732346ADC2C7FD4EA1646D37" = @{ Name = "GetLeveraged";             Editor = "C:\Program Files\GetLeveraged MT5 Terminal\MetaEditor64.exe" }
    "D0E8209F77C8CF37AD8BF550E51FF075" = @{ Name = "Default MT5";              Editor = "C:\Program Files\MetaTrader 5\MetaEditor64.exe" }
    "F6E5FFA163BE6F3F89ECBCA1BA487B55" = @{ Name = "Atlas Funded";             Editor = "C:\Program Files\Atlas Funded MT5 Terminal\MetaEditor64.exe" }
}

# Define file copy operations: [source, relative_dest_in_QC_folder]
$copyOps = @(
    # --- Standalone (BlueGuardian family) ---
    @("$DEPLOY\BG_AtlasGrid_JardinesGate.mq5",     "Standalone\BG_AtlasGrid_JardinesGate.mq5")
    @("$DEPLOY\BG_AtlasGrid_Original.mq5",          "Standalone\BG_AtlasGrid_Original.mq5")
    @("$DEPLOY\BlueGuardian_Elite.mq5",              "Standalone\BlueGuardian_Elite.mq5")
    @("$DEPLOY\BlueGuardian_Dynamic.mq5",            "Standalone\BlueGuardian_Dynamic.mq5")
    @("$BG_DEPLOY\BG_SimpleGrid.mq5",                "Standalone\BG_SimpleGrid.mq5")
    @("$BG_DEPLOY\BG_AggressiveCompetition.mq5",     "Standalone\BG_AggressiveCompetition.mq5")
    @("$BG_DEPLOY\BG_AtlasStyle.mq5",                "Standalone\BG_AtlasStyle.mq5")
    @("$QTL\BlueGuardian_Quantum.mq5",               "Standalone\BlueGuardian_Quantum.mq5")
    @("$QTL_INCLUDE\JardinesGate.mqh",               "Standalone\JardinesGate.mqh")
    @("$QTL_INCLUDE\QuantumEdgeFilter.mqh",          "Standalone\QuantumEdgeFilter.mqh")

    # --- EntropyGrid (4 EAs + 1 include) ---
    @("$DEPLOY\BTCUSD_GridTrader.mq5",               "EntropyGrid\BTCUSD_GridTrader.mq5")
    @("$DEPLOY\ETHUSD_GridTrader.mq5",               "EntropyGrid\ETHUSD_GridTrader.mq5")
    @("$DEPLOY\XAUUSD_GridTrader.mq5",               "EntropyGrid\XAUUSD_GridTrader.mq5")
    @("$DEPLOY\MultiSymbol_Launcher.mq5",             "EntropyGrid\MultiSymbol_Launcher.mq5")
    @("$DEPLOY\EntropyGridCore.mqh",                  "EntropyGrid\EntropyGridCore.mqh")

    # --- GridMaster300K (4 EAs + 1 include) ---
    @("$GM300K\GridMaster_Orchestrator.mq5",          "GridMaster300K\GridMaster_Orchestrator.mq5")
    @("$GM300K\GridExpert_Bullish.mq5",               "GridMaster300K\GridExpert_Bullish.mq5")
    @("$GM300K\GridExpert_Bearish.mq5",               "GridMaster300K\GridExpert_Bearish.mq5")
    @("$GM300K\GridExpert_Neutral.mq5",               "GridMaster300K\GridExpert_Neutral.mq5")
    @("$GM300K\GridCore.mqh",                          "GridMaster300K\GridCore.mqh")

    # --- XAUUSD GridMaster (1 EA + 1 include) ---
    @("$XAUUSD_GM\XAUUSD_GridMaster.mq5",            "XAUUSD_GridMaster\XAUUSD_GridMaster.mq5")
    @("$XAUUSD_GM\XAUUSD_GridCore.mqh",              "XAUUSD_GridMaster\XAUUSD_GridCore.mqh")

    # --- StrikeBoss (3 EAs) ---
    @("$MTW\StrikeBot_AI3.mq5",                       "StrikeBoss\StrikeBot_AI3.mq5")
    @("$STRIKEBOSS\FinMaster_Fixed.mq5",              "StrikeBoss\FinMaster_Fixed.mq5")
    @("$STRIKEBOSS\QuantumAI_FIXED.mq5",              "StrikeBoss\QuantumAI_FIXED.mq5")

    # --- NexaAI (9 EAs) ---
    @("$MTW\Nexa_AI_Minimal_Relay_MarketReady_FINAL_v2.07.mq5",       "NexaAI\Nexa_AI_Minimal_Relay_MarketReady_FINAL_v2.07.mq5")
    @("$MTW\Nexa_AI_Autonomous.mq5",                                   "NexaAI\Nexa_AI_Autonomous.mq5")
    @("$MTW\Nexa_AI_Market_ValidationReady_v1_05.mq5",                 "NexaAI\Nexa_AI_Market_ValidationReady_v1_05.mq5")
    @("$MTW\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED.mq5",               "NexaAI\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED.mq5")
    @("$MTW\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED_CLEANED.mq5",       "NexaAI\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED_CLEANED.mq5")
    @("$MTW\Nexa_AI_Only_FINAL_RELAYFIXED.mq5",                        "NexaAI\Nexa_AI_Only_FINAL_RELAYFIXED.mq5")
    @("$MTW\nexa_ai_validation_optimized.mq5",                          "NexaAI\nexa_ai_validation_optimized.mq5")
    @("$MTW\SUSTAI_AI_Bot (1).mq5",                                     "NexaAI\SUSTAI_AI_Bot_v1.mq5")
    @("$MTW\SUSTAI_AI_Bot (2).mq5",                                     "NexaAI\SUSTAI_AI_Bot_v2.mq5")

    # --- MyTradingWork (8 EAs) ---
    @("$MTW\SUSTAI_AI_Bot_Pro.mq5",                   "MyTradingWork\SUSTAI_AI_Bot_Pro.mq5")
    @("$MTW\Adaptive_Bitcoin_Master_Bot_Final.mq5",    "MyTradingWork\Adaptive_Bitcoin_Master_Bot_Final.mq5")
    @("$MTW\Scalping-EA-MT5.mq5",                     "MyTradingWork\Scalping-EA-MT5.mq5")
    @("$MTW\DPO_Val.mq5",                             "MyTradingWork\DPO_Val.mq5")
    @("$MTW\DPO_Zero_Crossover.mq5",                  "MyTradingWork\DPO_Zero_Crossover.mq5")
    @("$MTW\Neural_Networks_Propagation_EA.mq5",       "MyTradingWork\Neural_Networks_Propagation_EA.mq5")
    @("$MTW\neuropro.mq5",                             "MyTradingWork\neuropro.mq5")
    @("$MTW\ExpertAdvisor.mq5",                        "MyTradingWork\ExpertAdvisor.mq5")

    # --- WizardTeks (1 EA + 3 ONNX) ---
    @("$WIZARD\WZ_82.mq5",                            "WizardTeks\WZ_82.mq5")
    @("$WIZARD\82_1_0.onnx",                           "WizardTeks\82_1_0.onnx")
    @("$WIZARD\82_4_0.onnx",                           "WizardTeks\82_4_0.onnx")
    @("$WIZARD\82_5_0.onnx",                           "WizardTeks\82_5_0.onnx")
)

# SignalWZ_82.mqh goes to a different location (Include path)
$signalInclude = @("$WIZARD\SignalWZ_82.mqh", "Include\Expert\Signal\My\SignalWZ_82.mqh")

# Also deploy: BG_Executor, DataExporter, and other standalone EAs from QTL root
$rootEAs = @(
    @("$QTL\BG_Executor.mq5",                         "Standalone\BG_Executor.mq5")
    @("$QTL\DataExporter.mq5",                         "Standalone\DataExporter.mq5")
)

# Additional EAs from DEPLOY folder and BG_DEPLOY
$additionalEAs = @(
    @("$BG_DEPLOY\BG_DataExporter.mq5",               "Standalone\BG_DataExporter.mq5")
    @("$BG_DEPLOY\BG_Diagnostic.mq5",                 "Standalone\BG_Diagnostic.mq5")
    @("$BG_DEPLOY\BG_ForceTrade.mq5",                 "Standalone\BG_ForceTrade.mq5")
    @("$BG_DEPLOY\BG_MultiExecutor.mq5",              "Standalone\BG_MultiExecutor.mq5")
    @("$DEPLOY\BG_AtlasGrid.mq5",                     "Standalone\BG_AtlasGrid.mq5")
    @("$STRIKEBOSS\StrikeBOSS_Reality_Test.mq5",      "StrikeBoss\StrikeBOSS_Reality_Test.mq5")
)

$allCopyOps = $copyOps + $rootEAs + $additionalEAs

# Counters
$totalCopied = 0
$totalFailed = 0
$totalCompiled = 0
$totalCompileFail = 0

$subfolders = @("Standalone", "EntropyGrid", "GridMaster300K", "XAUUSD_GridMaster", "StrikeBoss", "NexaAI", "MyTradingWork", "WizardTeks")

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  QUANTUM CHILDREN - DEPLOY + COMPILE TO ALL MT5 TERMINALS" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# PHASE 1: COPY FILES
# ============================================================================
Write-Host "PHASE 1: DEPLOYING FILES TO ALL TERMINALS" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
Write-Host ""

foreach ($hash in $terminals.Keys) {
    $info = $terminals[$hash]
    $termPath = "$TERMINAL_BASE\$hash"
    $qcPath = "$termPath\MQL5\Experts\QuantumChildren"
    $mql5Path = "$termPath\MQL5"

    Write-Host "--- $($info.Name) ($hash) ---" -ForegroundColor Green

    # Create subfolders
    foreach ($sub in $subfolders) {
        $dir = "$qcPath\$sub"
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "  [MKDIR] $sub"
        }
    }

    # Create Include signal path
    $signalDir = "$mql5Path\Include\Expert\Signal\My"
    if (-not (Test-Path $signalDir)) {
        New-Item -ItemType Directory -Path $signalDir -Force | Out-Null
        Write-Host "  [MKDIR] Include\Expert\Signal\My"
    }

    # Copy all files
    foreach ($op in $allCopyOps) {
        $src = $op[0]
        $relDest = $op[1]
        $dst = "$qcPath\$relDest"
        $fname = Split-Path $src -Leaf

        if (Test-Path $src) {
            try {
                Copy-Item -Path $src -Destination $dst -Force
                Write-Host "  [ OK ] $fname"
                $totalCopied++
            } catch {
                Write-Host "  [FAIL] $fname - copy error" -ForegroundColor Red
                $totalFailed++
            }
        } else {
            Write-Host "  [MISS] $fname - source not found" -ForegroundColor DarkYellow
            $totalFailed++
        }
    }

    # Copy SignalWZ_82.mqh to Include path
    $src = $signalInclude[0]
    $dst = "$mql5Path\$($signalInclude[1])"
    $fname = Split-Path $src -Leaf
    if (Test-Path $src) {
        try {
            Copy-Item -Path $src -Destination $dst -Force
            Write-Host "  [ OK ] $fname -> Include\Expert\Signal\My\"
            $totalCopied++
        } catch {
            Write-Host "  [FAIL] $fname" -ForegroundColor Red
            $totalFailed++
        }
    } else {
        Write-Host "  [MISS] $fname" -ForegroundColor DarkYellow
        $totalFailed++
    }

    Write-Host ""
}

Write-Host ""
Write-Host "PHASE 1 COMPLETE: $totalCopied files copied, $totalFailed failed/missing" -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# PHASE 2: COMPILE ALL .mq5 FILES
# ============================================================================
Write-Host "PHASE 2: COMPILING ALL EAs" -ForegroundColor Yellow
Write-Host "==========================" -ForegroundColor Yellow
Write-Host ""

$compileResults = @()

foreach ($hash in $terminals.Keys) {
    $info = $terminals[$hash]
    $termPath = "$TERMINAL_BASE\$hash"
    $qcPath = "$termPath\MQL5\Experts\QuantumChildren"
    $editor = $info.Editor

    Write-Host "--- Compiling for: $($info.Name) ---" -ForegroundColor Green

    if (-not (Test-Path $editor)) {
        Write-Host "  [SKIP] MetaEditor not found at: $editor" -ForegroundColor Red
        continue
    }

    # Find all .mq5 files under QuantumChildren
    $mq5Files = Get-ChildItem -Path $qcPath -Filter "*.mq5" -Recurse -ErrorAction SilentlyContinue

    if ($mq5Files.Count -eq 0) {
        Write-Host "  [SKIP] No .mq5 files found" -ForegroundColor DarkYellow
        continue
    }

    Write-Host "  Found $($mq5Files.Count) .mq5 files to compile"

    foreach ($mq5 in $mq5Files) {
        $fname = $mq5.Name
        $fpath = $mq5.FullName
        $logFile = [System.IO.Path]::ChangeExtension($fpath, ".log")

        # Remove old log file
        if (Test-Path $logFile) { Remove-Item $logFile -Force }

        # Compile using MetaEditor64
        $proc = Start-Process -FilePath $editor -ArgumentList "/compile:`"$fpath`" /log:`"$logFile`" /inc:`"$termPath\MQL5`"" -Wait -PassThru -NoNewWindow -ErrorAction SilentlyContinue

        # Check compilation result
        $success = $false
        $errorMsg = ""

        if (Test-Path $logFile) {
            $logContent = Get-Content $logFile -Raw -ErrorAction SilentlyContinue
            if ($logContent -match "0 error") {
                $success = $true
            } else {
                # Extract error lines
                $errorLines = Get-Content $logFile -ErrorAction SilentlyContinue | Where-Object { $_ -match "error|Error" } | Select-Object -First 3
                $errorMsg = ($errorLines -join " | ")
            }
        }

        # Also check if .ex5 was produced
        $ex5File = [System.IO.Path]::ChangeExtension($fpath, ".ex5")
        $ex5Exists = Test-Path $ex5File

        if ($success -or $ex5Exists) {
            Write-Host "  [ OK ] $fname" -ForegroundColor White
            $totalCompiled++
            $compileResults += [PSCustomObject]@{ Terminal = $info.Name; File = $fname; Status = "OK"; Error = "" }
        } else {
            Write-Host "  [FAIL] $fname" -ForegroundColor Red
            if ($errorMsg) {
                Write-Host "         $errorMsg" -ForegroundColor DarkRed
            }
            $totalCompileFail++
            $compileResults += [PSCustomObject]@{ Terminal = $info.Name; File = $fname; Status = "FAIL"; Error = $errorMsg }
        }
    }
    Write-Host ""
}

# ============================================================================
# FINAL REPORT
# ============================================================================
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT + COMPILATION COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  PHASE 1 - File Copy:" -ForegroundColor White
Write-Host "    Files copied:       $totalCopied" -ForegroundColor White
Write-Host "    Copy failures:      $totalFailed" -ForegroundColor $(if ($totalFailed -gt 0) { "Red" } else { "White" })
Write-Host ""
Write-Host "  PHASE 2 - Compilation:" -ForegroundColor White
Write-Host "    Compiled OK:        $totalCompiled" -ForegroundColor White
Write-Host "    Compile failures:   $totalCompileFail" -ForegroundColor $(if ($totalCompileFail -gt 0) { "Red" } else { "White" })
Write-Host ""

if ($totalCompileFail -gt 0) {
    Write-Host "  FAILED COMPILATIONS:" -ForegroundColor Red
    $failedItems = $compileResults | Where-Object { $_.Status -eq "FAIL" }

    # Group by filename to show unique failures
    $grouped = $failedItems | Group-Object -Property File
    foreach ($g in $grouped) {
        $err = ($g.Group | Select-Object -First 1).Error
        $termNames = ($g.Group | ForEach-Object { $_.Terminal }) -join ", "
        Write-Host "    $($g.Name)" -ForegroundColor Red
        if ($err) { Write-Host "      Error: $err" -ForegroundColor DarkRed }
    }
    Write-Host ""
}

Write-Host "  Terminals: 6 (BG Challenge, BG Instant, FTMO, GetLeveraged, Default MT5, Atlas)" -ForegroundColor White
Write-Host ""
Write-Host "  NEXT: Right-click Navigator in each MT5 terminal > Refresh" -ForegroundColor Yellow
Write-Host "============================================================================" -ForegroundColor Cyan
