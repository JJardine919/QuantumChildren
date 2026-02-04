================================================================================
                    XAUUSD GRID TRADING EXPERT ADVISOR
                    GetLeveraged Multi-Account Edition
================================================================================

                        DooDoo - Quantum Trading Library


OVERVIEW
--------
This is a comprehensive XAUUSD Grid Trading system designed to run on 3
GetLeveraged accounts simultaneously. It features neural network integration,
entropy filtering, and LLM-based dynamic SL/TP adjustments.

ACCOUNTS CONFIGURED
-------------------
1. Account 113328 @ GetLeveraged-Trade
2. Account 113326 @ GetLeveraged-Trade
3. Account 107245 @ GetLeveraged-Trade


================================================================================
                              KEY FEATURES
================================================================================

ATR-BASED DYNAMIC SL/TP (HARD-CODED)
------------------------------------
- Stop Loss:     1.5x ATR (minimum, adjusted by LLM)
- Take Profit:   3.0x ATR (minimum, adjusted by LLM)
- ATR Period:    14 (standard)

These values are HARD-CODED in XAUUSD_GridCore.mqh and cannot be changed
via input parameters. The LLM can ONLY adjust these values within safe bounds.


PARTIAL TAKE PROFIT
-------------------
- 50% of position closed at first TP target
- Remaining 50% runs to full TP with trailing stop


BREAK-EVEN SYSTEM
-----------------
- Activation: 30% of TP distance
- SL moves to entry + small buffer (50 points)


TRAILING STOP
-------------
- Activation: 50% of TP distance
- Trail Distance: 1x ATR


ENTROPY FILTERING
-----------------
The system only trades when market is PREDICTABLE (low entropy).
- Shannon entropy calculated from recent price changes
- Default threshold: 2.0 (lower = more selective)
- This filters out choppy, unpredictable market conditions


COMPRESSION BOOST (+12)
-----------------------
- Adds +12% to confidence threshold
- Ensures trades only occur in high-confidence environments
- Based on QuantumChildren compression research showing 14% edge improvement


HIDDEN SL/TP
------------
- All stop-loss and take-profit levels are HIDDEN from broker
- Managed internally by the EA
- Protects against stop hunting and broker manipulation


GRID TRADING LOGIC
------------------
Three specialized managers coordinate based on market regime:

1. BULLISH Mode:
   - Opens BUY positions on pullbacks
   - Grid spacing based on ATR
   - Maximum 5 positions per manager

2. BEARISH Mode:
   - Opens SELL positions on rallies
   - Grid spacing based on ATR
   - Maximum 5 positions per manager

3. NEUTRAL Mode:
   - Trades range-bound markets
   - BUYs near range lows, SELLs near range highs
   - Maximum 5 positions (split between buys/sells)


LLM INTEGRATION (Ollama)
------------------------
The Python companion script provides:
- Real-time volatility regime detection (LOW/NORMAL/HIGH/EXTREME)
- Dynamic SL/TP multiplier adjustments based on market conditions
- Confidence scoring for trade signals

Supported models:
- gemma3:12b (recommended)
- gemma2:2b (faster, lighter)
- llama3:8b
- mistral:7b


================================================================================
                              FILE STRUCTURE
================================================================================

XAUUSD_GridMaster/
|
+-- XAUUSD_GridMaster.mq5       Main EA file (attach to XAUUSD chart)
+-- XAUUSD_GridCore.mqh         Core grid logic and ATR calculations
|
+-- xauusd_llm_companion.py     LLM integration for dynamic adjustments
|
+-- getleveraged_accounts.json  Account configuration
+-- deploy_xauusd_grid.py       Deployment automation script
|
+-- LAUNCH_XAUUSD_GRID.bat      Windows batch launcher
+-- Launch-XAUUSDGrid.ps1       PowerShell launcher (advanced)
|
+-- README.txt                  This file


================================================================================
                           INSTALLATION GUIDE
================================================================================

PREREQUISITES
-------------
1. MetaTrader 5 (GetLeveraged Terminal)
2. Python 3.9+ with packages:
   - MetaTrader5: pip install MetaTrader5
   - ollama: pip install ollama
   - numpy, pandas

3. Ollama installed with a model (optional but recommended):
   - Install: https://ollama.ai
   - Pull model: ollama pull gemma3:12b


STEP 1: DEPLOY MQL5 FILES
-------------------------
Option A (Automatic):
    Double-click LAUNCH_XAUUSD_GRID.bat

Option B (PowerShell):
    .\Launch-XAUUSDGrid.ps1 -Deploy

Option C (Manual):
    1. Copy XAUUSD_GridMaster.mq5 to:
       [MT5 Data Folder]\MQL5\Experts\QuantumChildren\XAUUSD_GridMaster\

    2. Copy XAUUSD_GridCore.mqh to same folder

    3. Compile in MetaEditor (press F7)


STEP 2: LAUNCH LLM COMPANION
----------------------------
Option A (Automatic with deployment):
    The batch launcher starts this automatically

Option B (Separate):
    python xauusd_llm_companion.py --account 113328

Option C (PowerShell):
    .\Launch-XAUUSDGrid.ps1 -LaunchLLM


STEP 3: CONFIGURE MT5 TERMINAL
------------------------------
1. Open GetLeveraged MT5 Terminal
2. Log into your account (113328, 113326, or 107245)
3. Open XAUUSD chart with M5 timeframe
4. Navigate to: Navigator > Expert Advisors > QuantumChildren > XAUUSD_GridMaster
5. Drag EA onto chart
6. In settings dialog:
   - Load preset from Presets\XAUUSD_GridMaster\GL_XXXXX.set
   - Or configure manually using account-specific magic numbers
7. Click OK
8. Enable Auto Trading (green button in toolbar)


STEP 4: VERIFY OPERATION
------------------------
Check the "Experts" tab in MT5 for log messages:
- "XAUUSD GRID MANAGER INITIALIZED"
- "ATR SL Multiplier: 1.5x (HARD-CODED)"
- "Compression Boost: +12"

Check LLM companion output:
- Should update signal file every 30 seconds
- Signal file location: %APPDATA%\MetaQuotes\Terminal\Common\Files\


================================================================================
                          ACCOUNT-SPECIFIC SETTINGS
================================================================================

ACCOUNT 113328 (Primary)
------------------------
Magic Numbers:
  - Bullish: 113001
  - Bearish: 113002
  - Neutral: 113003

Preset File: GL_113328.set


ACCOUNT 113326 (Secondary)
--------------------------
Magic Numbers:
  - Bullish: 113101
  - Bearish: 113102
  - Neutral: 113103

Preset File: GL_113326.set


ACCOUNT 107245 (Tertiary)
-------------------------
Magic Numbers:
  - Bullish: 107001
  - Bearish: 107002
  - Neutral: 107003

Preset File: GL_107245.set


================================================================================
                             RISK MANAGEMENT
================================================================================

Default Settings:
- Risk per grid level: 0.25%
- Maximum orders per expert: 5
- Maximum total orders: 15
- Daily drawdown limit: 5%
- Maximum drawdown limit: 10%

IMPORTANT:
The EA will STOP trading if drawdown limits are exceeded.
This protects your account from catastrophic losses.


================================================================================
                              TROUBLESHOOTING
================================================================================

PROBLEM: EA not trading
-----------------------
1. Check if Auto Trading is enabled (green button)
2. Verify in Experts tab there are no errors
3. Check entropy - if above threshold, market is too choppy
4. Verify LLM companion is running and signal file exists

PROBLEM: LLM companion not starting
-----------------------------------
1. Ensure Python is in PATH: python --version
2. Install dependencies: pip install MetaTrader5 ollama numpy pandas
3. Check if Ollama is running: ollama list

PROBLEM: Signal file not updating
---------------------------------
1. Check LLM companion console for errors
2. Verify MT5 is connected and providing data
3. Check file permissions in Common folder

PROBLEM: Orders rejected
------------------------
1. Check trade permissions in MT5
2. Verify account has sufficient margin
3. Check if symbol is available for trading
4. Review GetLeveraged restrictions


================================================================================
                                DISCLAIMER
================================================================================

This trading system is provided for educational purposes only.

Trading involves substantial risk of loss and is not suitable for all
investors. Past performance is not indicative of future results.

NEVER trade with money you cannot afford to lose.

The authors are not responsible for any financial losses incurred through
the use of this system.


================================================================================
                               CHANGE LOG
================================================================================

Version 1.0.0 (2026-02-03)
- Initial release
- XAUUSD Grid Trading with 3 regime modes
- ATR-based dynamic SL/TP (hard-coded multipliers)
- Entropy filtering for predictability
- +12 compression boost
- LLM integration for dynamic adjustments
- Hidden SL/TP management
- Partial take profit (50%)
- Trailing stop and break-even
- Multi-account configuration


================================================================================
                            Contact & Support
================================================================================

Author: DooDoo
Project: Quantum Trading Library

For issues and updates, check the QuantumChildren repository.


================================================================================
