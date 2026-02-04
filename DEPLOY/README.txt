================================================================================
                    QUANTUM CHILDREN - DEPLOY FOLDER
================================================================================

LOCATION: C:\Users\jimjj\Music\QuantumChildren\DEPLOY

This is your ONE-STOP location for all trading systems.

================================================================================
HOW TO START
================================================================================

Just double-click: START.bat

================================================================================
WHAT'S HERE
================================================================================

MQL5 FILES (copy to MT5):
  BlueGuardian_Dynamic.mq5   - Dynamic SL/TP with LLM (ATR_MULT=0.0438)
  BlueGuardian_Elite.mq5     - +12 Compression Expert (89% win rate)
  EntropyGridCore.mqh        - Shared grid library
  XAUUSD_GridTrader.mq5      - Gold grid trader
  BTCUSD_GridTrader.mq5      - Bitcoin grid trader
  ETHUSD_GridTrader.mq5      - Ethereum grid trader
  MultiSymbol_Launcher.mq5   - All-in-one grid EA

PYTHON FILES:
  llm_monitor.py             - LLM that monitors BlueGuardian positions
  llm_sltp_optimizer.py      - LLM that optimizes SL/TP for grid traders

BATCH FILES:
  START.bat                  - Master launcher (USE THIS)
  START_GETLEVERAGED_GRIDS.bat - Old launcher (use START.bat instead)

================================================================================
QUICK REFERENCE
================================================================================

BLUEGUARDIAN DYNAMIC:
  ATR_MULT:    0.0438 (hard-coded)
  TP:          3x
  SL:          50 points x 1.5
  DYN_TP:      30%
  ROLLING_SL:  TRUE

GRID TRADERS:
  SL:          1.5x ATR
  TP:          3.0x ATR
  Partial:     50%
  Entropy:     +12 boost

ACCOUNTS:
  GetLeveraged 1: 113328
  GetLeveraged 2: 113326
  GetLeveraged 3: 107245

================================================================================
