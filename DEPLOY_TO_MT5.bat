@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: DEPLOY_TO_MT5.bat
:: Deploys ALL Quantum Children EAs to every detected MT5 terminal
::
:: INVENTORY (organized by subfolder under Experts\QuantumChildren\):
::   Standalone/         -  8 EAs  (BlueGuardian family)
::   EntropyGrid/        -  4 EAs  + EntropyGridCore.mqh
::   GridMaster300K/     -  4 EAs  + GridCore.mqh
::   XAUUSD_GridMaster/  -  1 EA   + XAUUSD_GridCore.mqh
::   StrikeBoss/         -  3 EAs  (StrikeBot_AI3, FinMaster_Fixed, QuantumAI_FIXED)
::   NexaAI/             -  9 EAs  (all Nexa variants, latest of duplicates)
::   MyTradingWork/      -  8 EAs  (SUSTAI, Bitcoin, Scalping, DPO, Neural, neuropro, EA)
::   WizardTeks/         -  1 EA   + SignalWZ_82.mqh + 3 ONNX models
::
:: Source priority: DEPLOY/ and QuantumTradingLibrary/ are newer than
::                  QuantumChildren-github/ -- only latest versions are copied.
:: ============================================================================

title Quantum Children - FULL MT5 Deployment Tool

echo.
echo ============================================================================
echo   QUANTUM CHILDREN - FULL MT5 DEPLOYMENT TOOL
echo   Deploying ALL EAs + includes to all detected MT5 terminals
echo ============================================================================
echo.

:: ---------------------------------------------------------------------------
:: Source paths (preference: DEPLOY > QTL > QuantumChildren-github)
:: ---------------------------------------------------------------------------
set "PROJECT=C:\Users\jimjj\Music\QuantumChildren"
set "DEPLOY=%PROJECT%\DEPLOY"
set "QTL=%PROJECT%\QuantumTradingLibrary"
set "BG_DEPLOY=%QTL%\BlueGuardian_Deploy"
set "BG_EA=%QTL%\BlueGuardian_EA"
set "GM300K=%QTL%\GridMaster300K"
set "XAUUSD_GM=%QTL%\XAUUSD_GridMaster"
set "GL_GRID=%QTL%\GetLeveraged_Grid"
set "QTL_INCLUDE=%QTL%\Include"
set "MTW=%PROJECT%\my-trading-work"
set "STRIKEBOSS=%PROJECT%\StrikeBOSS"
set "WIZARD=%PROJECT%\WIZARD TEKS(PART 82)"

:: Counters
set /a TOTAL_SUCCESS=0
set /a TOTAL_FAIL=0
set /a TOTAL_FILES=0
set /a TERMINALS_FOUND=0

:: ---------------------------------------------------------------------------
:: Auto-detect MT5 terminal data folders
:: ---------------------------------------------------------------------------
set "TERMINAL_BASE=C:\Users\jimjj\AppData\Roaming\MetaQuotes\Terminal"

if not exist "%TERMINAL_BASE%" (
    echo [ERROR] MetaQuotes Terminal folder not found at:
    echo         %TERMINAL_BASE%
    echo.
    echo         Make sure MetaTrader 5 has been run at least once.
    goto :END
)

echo [SCAN] Searching for MT5 terminals in:
echo        %TERMINAL_BASE%
echo.

:: Build list of valid terminal folders (ones that contain MQL5\Experts)
for /d %%T in ("%TERMINAL_BASE%\*") do (
    if exist "%%T\MQL5\Experts" (
        set /a TERMINALS_FOUND+=1
        echo        Found: %%~nxT
    )
)

echo.

if !TERMINALS_FOUND! equ 0 (
    echo [ERROR] No MT5 terminal data folders found containing MQL5\Experts.
    echo         Looked in: %TERMINAL_BASE%
    goto :END
)

echo [INFO] Found !TERMINALS_FOUND! MT5 terminal(s).
echo.

:: ---------------------------------------------------------------------------
:: Deploy to each terminal
:: ---------------------------------------------------------------------------
for /d %%T in ("%TERMINAL_BASE%\*") do (
    if exist "%%T\MQL5\Experts" (
        echo ============================================================================
        echo   DEPLOYING TO: %%~nxT
        echo ============================================================================
        echo.

        set "QC=%%T\MQL5\Experts\QuantumChildren"
        set "INC_SIGNAL=%%T\MQL5\Include\Expert\Signal\My"

        :: =============================================================
        :: Create folder structure
        :: =============================================================
        echo [MKDIR] Creating folder structure...

        for %%D in (
            Standalone
            EntropyGrid
            GridMaster300K
            XAUUSD_GridMaster
            StrikeBoss
            NexaAI
            MyTradingWork
            WizardTeks
        ) do (
            if not exist "!QC!\%%D" (
                mkdir "!QC!\%%D" 2>nul
                echo         Created: QuantumChildren\%%D
            ) else (
                echo         Exists:  QuantumChildren\%%D
            )
        )

        :: WZ_82 signal include goes into MQL5\Include\Expert\Signal\My
        if not exist "!INC_SIGNAL!" (
            mkdir "!INC_SIGNAL!" 2>nul
            echo         Created: Include\Expert\Signal\My
        ) else (
            echo         Exists:  Include\Expert\Signal\My
        )

        echo.

        :: =============================================================
        :: 1. STANDALONE (8 EAs) - BlueGuardian family
        ::    Sources: DEPLOY/ for AtlasGrid, BG_DEPLOY/ for others,
        ::             QTL/ for BlueGuardian_Quantum
        :: =============================================================
        echo [COPY] Standalone - BlueGuardian family ^(8 EAs^)...

        call :COPY_FILE "%DEPLOY%\BG_AtlasGrid_JardinesGate.mq5"      "!QC!\Standalone\BG_AtlasGrid_JardinesGate.mq5"
        call :COPY_FILE "%DEPLOY%\BG_AtlasGrid_Original.mq5"          "!QC!\Standalone\BG_AtlasGrid_Original.mq5"
        call :COPY_FILE "%DEPLOY%\BlueGuardian_Elite.mq5"              "!QC!\Standalone\BlueGuardian_Elite.mq5"
        call :COPY_FILE "%DEPLOY%\BlueGuardian_Dynamic.mq5"            "!QC!\Standalone\BlueGuardian_Dynamic.mq5"
        call :COPY_FILE "%BG_DEPLOY%\BG_SimpleGrid.mq5"                "!QC!\Standalone\BG_SimpleGrid.mq5"
        call :COPY_FILE "%BG_DEPLOY%\BG_AggressiveCompetition.mq5"     "!QC!\Standalone\BG_AggressiveCompetition.mq5"
        call :COPY_FILE "%BG_DEPLOY%\BG_AtlasStyle.mq5"                "!QC!\Standalone\BG_AtlasStyle.mq5"
        call :COPY_FILE "%QTL%\BlueGuardian_Quantum.mq5"               "!QC!\Standalone\BlueGuardian_Quantum.mq5"

        :: Include files needed by AtlasGrid EAs
        call :COPY_FILE "%QTL_INCLUDE%\JardinesGate.mqh"               "!QC!\Standalone\JardinesGate.mqh"
        call :COPY_FILE "%QTL_INCLUDE%\QuantumEdgeFilter.mqh"          "!QC!\Standalone\QuantumEdgeFilter.mqh"

        echo.

        :: =============================================================
        :: 2. ENTROPY GRID (4 EAs + 1 include)
        ::    Source: DEPLOY/ (latest) over QTL/GetLeveraged_Grid
        :: =============================================================
        echo [COPY] EntropyGrid ^(4 EAs + EntropyGridCore.mqh^)...

        call :COPY_FILE "%DEPLOY%\BTCUSD_GridTrader.mq5"               "!QC!\EntropyGrid\BTCUSD_GridTrader.mq5"
        call :COPY_FILE "%DEPLOY%\ETHUSD_GridTrader.mq5"               "!QC!\EntropyGrid\ETHUSD_GridTrader.mq5"
        call :COPY_FILE "%DEPLOY%\XAUUSD_GridTrader.mq5"               "!QC!\EntropyGrid\XAUUSD_GridTrader.mq5"
        call :COPY_FILE "%DEPLOY%\MultiSymbol_Launcher.mq5"            "!QC!\EntropyGrid\MultiSymbol_Launcher.mq5"
        call :COPY_FILE "%DEPLOY%\EntropyGridCore.mqh"                 "!QC!\EntropyGrid\EntropyGridCore.mqh"

        echo.

        :: =============================================================
        :: 3. GRIDMASTER 300K (4 EAs + 1 include)
        ::    Source: QTL/GridMaster300K (latest)
        :: =============================================================
        echo [COPY] GridMaster300K ^(4 EAs + GridCore.mqh^)...

        call :COPY_FILE "%GM300K%\GridMaster_Orchestrator.mq5"          "!QC!\GridMaster300K\GridMaster_Orchestrator.mq5"
        call :COPY_FILE "%GM300K%\GridExpert_Bullish.mq5"               "!QC!\GridMaster300K\GridExpert_Bullish.mq5"
        call :COPY_FILE "%GM300K%\GridExpert_Bearish.mq5"               "!QC!\GridMaster300K\GridExpert_Bearish.mq5"
        call :COPY_FILE "%GM300K%\GridExpert_Neutral.mq5"               "!QC!\GridMaster300K\GridExpert_Neutral.mq5"
        call :COPY_FILE "%GM300K%\GridCore.mqh"                         "!QC!\GridMaster300K\GridCore.mqh"

        echo.

        :: =============================================================
        :: 4. XAUUSD GRIDMASTER (1 EA + 1 include)
        ::    Source: QTL/XAUUSD_GridMaster (latest)
        :: =============================================================
        echo [COPY] XAUUSD_GridMaster ^(1 EA + XAUUSD_GridCore.mqh^)...

        call :COPY_FILE "%XAUUSD_GM%\XAUUSD_GridMaster.mq5"            "!QC!\XAUUSD_GridMaster\XAUUSD_GridMaster.mq5"
        call :COPY_FILE "%XAUUSD_GM%\XAUUSD_GridCore.mqh"              "!QC!\XAUUSD_GridMaster\XAUUSD_GridCore.mqh"

        echo.

        :: =============================================================
        :: 5. STRIKEBOSS (3 EAs)
        ::    StrikeBot_AI3 from my-trading-work (latest version)
        ::    FinMaster_Fixed + QuantumAI_FIXED from StrikeBOSS/
        :: =============================================================
        echo [COPY] StrikeBoss ^(3 EAs^)...

        call :COPY_FILE "%MTW%\StrikeBot_AI3.mq5"                      "!QC!\StrikeBoss\StrikeBot_AI3.mq5"
        call :COPY_FILE "%STRIKEBOSS%\FinMaster_Fixed.mq5"             "!QC!\StrikeBoss\FinMaster_Fixed.mq5"
        call :COPY_FILE "%STRIKEBOSS%\QuantumAI_FIXED.mq5"             "!QC!\StrikeBoss\QuantumAI_FIXED.mq5"

        echo.

        :: =============================================================
        :: 6. NEXA AI (9 EAs - all unique variants, latest of duplicates)
        ::    Source: my-trading-work/ (skip (1) dupes)
        :: =============================================================
        echo [COPY] NexaAI ^(9 EAs - all unique Nexa variants^)...

        call :COPY_FILE "%MTW%\Nexa_AI_Minimal_Relay_MarketReady_FINAL_v2.07.mq5"       "!QC!\NexaAI\Nexa_AI_Minimal_Relay_MarketReady_FINAL_v2.07.mq5"
        call :COPY_FILE "%MTW%\Nexa_AI_Autonomous.mq5"                                  "!QC!\NexaAI\Nexa_AI_Autonomous.mq5"
        call :COPY_FILE "%MTW%\Nexa_AI_Market_ValidationReady_v1_05.mq5"                 "!QC!\NexaAI\Nexa_AI_Market_ValidationReady_v1_05.mq5"
        call :COPY_FILE "%MTW%\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED.mq5"               "!QC!\NexaAI\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED.mq5"
        call :COPY_FILE "%MTW%\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED_CLEANED.mq5"       "!QC!\NexaAI\Nexa_AI_Minimal_Relay_v2_04_CHAT_FIXED_CLEANED.mq5"
        call :COPY_FILE "%MTW%\Nexa_AI_Only_FINAL_RELAYFIXED.mq5"                        "!QC!\NexaAI\Nexa_AI_Only_FINAL_RELAYFIXED.mq5"
        call :COPY_FILE "%MTW%\nexa_ai_validation_optimized.mq5"                          "!QC!\NexaAI\nexa_ai_validation_optimized.mq5"
        call :COPY_FILE "%MTW%\SUSTAI_AI_Bot (1).mq5"                                    "!QC!\NexaAI\SUSTAI_AI_Bot_v1.mq5"
        call :COPY_FILE "%MTW%\SUSTAI_AI_Bot (2).mq5"                                    "!QC!\NexaAI\SUSTAI_AI_Bot_v2.mq5"

        echo.

        :: =============================================================
        :: 7. MY TRADING WORK (8+ EAs - individual strategies)
        ::    Source: my-trading-work/ (latest, skip (1)(2) dupes)
        :: =============================================================
        echo [COPY] MyTradingWork ^(8 EAs - individual strategies^)...

        call :COPY_FILE "%MTW%\SUSTAI_AI_Bot_Pro.mq5"                   "!QC!\MyTradingWork\SUSTAI_AI_Bot_Pro.mq5"
        call :COPY_FILE "%MTW%\Adaptive_Bitcoin_Master_Bot_Final.mq5"   "!QC!\MyTradingWork\Adaptive_Bitcoin_Master_Bot_Final.mq5"
        call :COPY_FILE "%MTW%\Scalping-EA-MT5.mq5"                    "!QC!\MyTradingWork\Scalping-EA-MT5.mq5"
        call :COPY_FILE "%MTW%\DPO_Val.mq5"                            "!QC!\MyTradingWork\DPO_Val.mq5"
        call :COPY_FILE "%MTW%\DPO_Zero_Crossover.mq5"                 "!QC!\MyTradingWork\DPO_Zero_Crossover.mq5"
        call :COPY_FILE "%MTW%\Neural_Networks_Propagation_EA.mq5"      "!QC!\MyTradingWork\Neural_Networks_Propagation_EA.mq5"
        call :COPY_FILE "%MTW%\neuropro.mq5"                            "!QC!\MyTradingWork\neuropro.mq5"
        call :COPY_FILE "%MTW%\ExpertAdvisor.mq5"                       "!QC!\MyTradingWork\ExpertAdvisor.mq5"

        echo.

        :: =============================================================
        :: 8. WIZARD TEKS (1 EA + SignalWZ_82.mqh + 3 ONNX models)
        ::    Source: WIZARD TEKS(PART 82)/ (root project, latest)
        ::    SignalWZ_82.mqh -> MQL5\Include\Expert\Signal\My\
        ::    ONNX files -> alongside EA in WizardTeks/
        :: =============================================================
        echo [COPY] WizardTeks ^(1 EA + signal include + 3 ONNX models^)...

        call :COPY_FILE "%WIZARD%\WZ_82.mq5"                           "!QC!\WizardTeks\WZ_82.mq5"
        call :COPY_FILE "%WIZARD%\SignalWZ_82.mqh"                     "!INC_SIGNAL!\SignalWZ_82.mqh"
        call :COPY_FILE "%WIZARD%\82_1_0.onnx"                         "!QC!\WizardTeks\82_1_0.onnx"
        call :COPY_FILE "%WIZARD%\82_4_0.onnx"                         "!QC!\WizardTeks\82_4_0.onnx"
        call :COPY_FILE "%WIZARD%\82_5_0.onnx"                         "!QC!\WizardTeks\82_5_0.onnx"

        echo.
    )
)

:: ---------------------------------------------------------------------------
:: Final Report
:: ---------------------------------------------------------------------------
echo ============================================================================
echo   DEPLOYMENT COMPLETE
echo ============================================================================
echo.
echo   Terminals deployed to:  !TERMINALS_FOUND!
echo   Files copied:           !TOTAL_SUCCESS! / !TOTAL_FILES!
echo   Failures:               !TOTAL_FAIL!
echo.

if !TOTAL_FAIL! equ 0 (
    echo   STATUS: ALL FILES DEPLOYED SUCCESSFULLY
) else (
    echo   STATUS: SOME FILES FAILED - check output above for [FAIL] lines
)

echo.
echo   Folder structure in each terminal:
echo     MQL5\Experts\QuantumChildren\
echo       Standalone\           ^(  8 EAs  ^) BG_AtlasGrid_JardinesGate, BG_AtlasGrid_Original,
echo                                         BlueGuardian_Elite, BlueGuardian_Dynamic,
echo                                         BG_SimpleGrid, BG_AggressiveCompetition,
echo                                         BG_AtlasStyle, BlueGuardian_Quantum
echo                                         + JardinesGate.mqh, QuantumEdgeFilter.mqh
echo       EntropyGrid\          ^(  4 EAs  ^) BTCUSD/ETHUSD/XAUUSD_GridTrader, MultiSymbol_Launcher
echo                                         + EntropyGridCore.mqh
echo       GridMaster300K\       ^(  4 EAs  ^) GridMaster_Orchestrator, GridExpert_Bullish/Bearish/Neutral
echo                                         + GridCore.mqh
echo       XAUUSD_GridMaster\    ^(  1 EA   ^) XAUUSD_GridMaster + XAUUSD_GridCore.mqh
echo       StrikeBoss\           ^(  3 EAs  ^) StrikeBot_AI3, FinMaster_Fixed, QuantumAI_FIXED
echo       NexaAI\               ^(  9 EAs  ^) All Nexa AI variants + SUSTAI v1/v2
echo       MyTradingWork\        ^(  8 EAs  ^) SUSTAI_Pro, Bitcoin, Scalping, DPO x2,
echo                                         Neural_Networks, neuropro, ExpertAdvisor
echo       WizardTeks\           ^(  1 EA   ^) WZ_82 + 3 ONNX models
echo.
echo     MQL5\Include\Expert\Signal\My\
echo       SignalWZ_82.mqh                    ^(required by WZ_82 EA^)
echo.
echo   TOTAL: 38 EAs  ^|  7 include/mqh files  ^|  3 ONNX models  ^|  48 files per terminal
echo.
echo   NOTE: Duplicate files with ^(1^), ^(2^) suffixes were skipped.
echo         Only the canonical latest version of each file is deployed.
echo         QuantumChildren-github/ was skipped ^(older mirror^).
echo.
echo   NEXT STEP: Open MetaTrader 5 ^> right-click Navigator ^> Refresh
echo              EAs appear under Expert Advisors\QuantumChildren\
echo ============================================================================

goto :END

:: ---------------------------------------------------------------------------
:: Subroutine: COPY_FILE
:: Usage: call :COPY_FILE "source" "destination"
:: ---------------------------------------------------------------------------
:COPY_FILE
set /a TOTAL_FILES+=1
set "SRC=%~1"
set "DST=%~2"
set "FNAME=%~nx1"

if not exist "%SRC%" (
    echo         [FAIL] %FNAME% - source not found
    echo                %SRC%
    set /a TOTAL_FAIL+=1
    goto :eof
)

copy /y "%SRC%" "%DST%" >nul 2>&1
if errorlevel 1 (
    echo         [FAIL] %FNAME% - copy failed
    set /a TOTAL_FAIL+=1
) else (
    echo         [ OK ] %FNAME%
    set /a TOTAL_SUCCESS+=1
)
goto :eof

:: ---------------------------------------------------------------------------
:END
echo.
pause
endlocal
