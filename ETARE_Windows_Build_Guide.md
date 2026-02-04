# ETARE Trading System - Windows Build & Installation Guide

## System Overview

The ETARE (Evolutionary Trading Algorithm for Real-time Execution) system is a hybrid algorithmic trading platform that combines genetic algorithms, neural networks, and grid trading strategies with MetaTrader 5 execution.

---

## 1. System Architecture

```
+============================================================================+
|                        ETARE TRADING SYSTEM ARCHITECTURE                    |
+============================================================================+

+-------------------------+     JSON Signal File      +------------------------+
|                         |    (etare_signals.json)   |                        |
|   ETARE_module.py       |-------------------------->|   BG_Executor.mq5      |
|   (Python Engine)       |                           |   (MT5 Expert Advisor) |
|                         |                           |                        |
| +---------------------+ |                           | +--------------------+ |
| | Genetic Algorithm   | |                           | | Signal Reader      | |
| | Population Manager  | |                           | | (10s interval)     | |
| +---------------------+ |                           | +--------------------+ |
|           |             |                           |           |            |
| +---------------------+ |                           | +--------------------+ |
| | Neural Network      | |                           | | Risk Management    | |
| | (128->64->6 nodes)  | |                           | | (Dynamic SL/TP)    | |
| +---------------------+ |                           | +--------------------+ |
|           |             |                           |           |            |
| +---------------------+ |                           | +--------------------+ |
| | Technical Analysis  | |                           | | Position Manager   | |
| | (RSI,MACD,BB,EMA)   | |                           | | (Partial TP/Trail) | |
| +---------------------+ |                           | +--------------------+ |
|           |             |                           |           |            |
| +---------------------+ |                           | +--------------------+ |
| | Grid Trading Logic  | |                           | | Order Execution    | |
| | (Limit Orders)      | |                           | | (Market Orders)    | |
| +---------------------+ |                           | +--------------------+ |
|           |             |                           |           |            |
|           v             |                           |           v            |
| +---------------------+ |                           | +--------------------+ |
| | SQLite Database     | |                           | | MT5 Trade Server   | |
| | (trading_history.db)| |                           | | (Broker)           | |
| +---------------------+ |                           | +--------------------+ |
+-------------------------+                           +------------------------+
        |                                                       |
        |  MetaTrader5 Python API                              |
        +-------------------+-----------------------------------+
                            |
                            v
              +---------------------------+
              |     MetaTrader 5          |
              |     Terminal              |
              |  +---------------------+  |
              |  | Market Data Feed   |  |
              |  | Account Management |  |
              |  | Order Routing      |  |
              |  +---------------------+  |
              +---------------------------+
```

---

## 2. Data Flow Explanation

### Signal Generation Flow (ETARE Python Engine)

```
Market Data (MT5 API)
        |
        v
+------------------+
| Data Collection  |  <-- 5-minute candles, 100 bar history
+------------------+
        |
        v
+------------------+
| Feature Prep     |  <-- RSI, MACD, Bollinger, EMAs, Stochastic, CCI
+------------------+
        |
        v
+------------------+
| Neural Network   |  <-- Input(N) -> Hidden(128) -> Hidden(64) -> Output(6)
| Prediction       |      Actions: OPEN_BUY, OPEN_SELL, CLOSE_*
+------------------+
        |
        v
+------------------+
| Genetic          |  <-- Tournament selection, crossover, mutation
| Evolution        |      Extinction events every 10 generations
+------------------+
        |
        v
+------------------+
| Signal Output    |  --> etare_signals.json
+------------------+
```

### Trade Execution Flow (BG_Executor EA)

```
etare_signals.json
        |
        v (every 10 seconds)
+------------------+
| Signal Parser    |  <-- Reads action + confidence from JSON
+------------------+
        |
        v
+------------------+
| Risk Calculator  |  <-- BaseSL * ScaleMultiplier * AccountScale
+------------------+      TP1 = SL * 1.5 (partial close)
        |                 TP2 = SL * 3.0 (full target)
        v
+------------------+
| Order Execution  |  <-- Market order with SL/TP
+------------------+
        |
        v (every tick)
+------------------+
| Position Manager |  <-- TP1 hit: 50% close, move SL to breakeven
+------------------+      Trailing stop: locks in profit
        |
        v
+------------------+
| Dynamic TP       |  <-- Remove TP2 after TP1, let trailing handle exit
+------------------+
```

---

## 3. Component Inventory

| Component | File | Purpose |
|-----------|------|---------|
| **Signal Generator** | `ETARE_module.py` | Python engine with genetic algorithms and neural networks for signal generation |
| **Trade Executor** | `BG_Executor.mq5` | MQL5 Expert Advisor for executing trades with risk management |
| **Signal File** | `etare_signals.json` | JSON file for inter-process communication (Python -> MT5) |
| **Database** | `trading_history.db` | SQLite database storing population state and trade history |

### ETARE_module.py - Key Classes

| Class | Purpose |
|-------|---------|
| `Action` | Enum defining 6 trading actions (OPEN_BUY, OPEN_SELL, CLOSE_*) |
| `GeneticWeights` | Dataclass holding neural network weight matrices |
| `RLMemory` | Replay buffer for reinforcement learning (10,000 capacity) |
| `TradingIndividual` | Neural network trader with learning capabilities |
| `GridParameters` | Evolutionary grid trading parameters |
| `GridTrader` | Combined neural network + grid trading individual |
| `HybridGridTrader` | Population manager with evolutionary operations |

### BG_Executor.mq5 - Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `SignalFile` | `etare_signals.json` | Path to signal file |
| `MagicNumber` | `365060` | Unique EA identifier |
| `BaseSL_Dollars` | `50.0` | Base stop loss in USD |
| `RR_Ratio` | `3.0` | Risk:Reward for full TP |
| `ScaleMultiplier` | `1.5` | SL/TP distance multiplier |
| `TP1_Percent` | `50.0` | Partial close percentage at TP1 |
| `TP1_RR_Ratio` | `1.5` | R:R ratio for TP1 level |
| `TrailDistance` | `50.0` | Trailing stop distance in USD |
| `LotSize` | `2.5` | Fixed lot size |
| `BaseAccountSize` | `100000.0` | Reference account for scaling |

---

## 4. Step-by-Step Windows Installation

### Prerequisites

- Windows 10/11 (64-bit)
- Administrator access
- Stable internet connection
- Broker account with MT5 support

### Step 4.1: Install MetaTrader 5

1. **Download MT5 from your broker** or from [MetaQuotes](https://www.metatrader5.com/en/download)

2. **Run the installer**:
   ```
   mt5setup.exe
   ```

3. **Complete installation** with default settings

4. **Launch MT5 and log in** to your broker account

5. **Note the installation path** (typically):
   ```
   C:\Program Files\MetaTrader 5\
   ```

6. **Locate the data folder** (File -> Open Data Folder):
   ```
   C:\Users\<YourUsername>\AppData\Roaming\MetaQuotes\Terminal\<TERMINAL_ID>\
   ```

### Step 4.2: Install Python Environment

1. **Download Python 3.9+** from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation

2. **Open Command Prompt as Administrator**

3. **Create a virtual environment** (recommended):
   ```cmd
   cd C:\Trading
   python -m venv etare_env
   etare_env\Scripts\activate
   ```

4. **Install required packages**:
   ```cmd
   pip install numpy pandas MetaTrader5
   ```

5. **Verify MetaTrader5 package**:
   ```cmd
   python -c "import MetaTrader5 as mt5; print(mt5.__version__)"
   ```

### Step 4.3: Deploy BG_Executor Expert Advisor

1. **Copy `BG_Executor.mq5`** to the Experts folder:
   ```
   C:\Users\<YourUsername>\AppData\Roaming\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Experts\
   ```

2. **Open MetaEditor** (F4 in MT5 or Tools -> MetaQuotes Language Editor)

3. **Open `BG_Executor.mq5`** in MetaEditor

4. **Compile the EA** (F7 or Compile button)
   - Ensure "0 errors, 0 warnings" in output

5. **Restart MetaTrader 5**

6. **Verify EA appears** in Navigator panel under Expert Advisors

### Step 4.4: Deploy ETARE Python Module

1. **Create project directory**:
   ```cmd
   mkdir C:\Trading\ETARE
   cd C:\Trading\ETARE
   ```

2. **Copy `ETARE_module.py`** to this directory

3. **Create the signal output directory** (same as MT5 Files folder):
   ```cmd
   mkdir "C:\Users\<YourUsername>\AppData\Roaming\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Files"
   ```

4. **Modify ETARE_module.py** to output signals to correct path:
   ```python
   SIGNAL_FILE_PATH = r"C:\Users\<YourUsername>\AppData\Roaming\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Files\etare_signals.json"
   ```

### Step 4.5: Configure MetaTrader 5 Terminal

1. **Enable Algo Trading**:
   - Tools -> Options -> Expert Advisors
   - Check "Allow algorithmic trading"
   - Check "Allow DLL imports"

2. **Enable Auto Trading** on toolbar (green button should be active)

3. **Open the chart** for your trading symbol (e.g., BTCUSD)

4. **Set chart timeframe** to M5 (5-minute)

5. **Drag BG_Executor** from Navigator to the chart

6. **Configure EA inputs** in the popup dialog:
   ```
   SignalFile       = etare_signals.json
   MagicNumber      = 365060
   TradeEnabled     = true  (or false for paper trading)
   BaseSL_Dollars   = 50.0
   RR_Ratio         = 3.0
   ScaleMultiplier  = 1.5
   TP1_Percent      = 50.0
   LotSize          = 2.5   (adjust for your risk tolerance)
   ```

7. **Click OK** to attach the EA

8. **Verify EA is running** (smiley face icon on chart)

---

## 5. Configuration Reference

### 5.1 BG_Executor.mq5 Configuration

Edit input parameters before compiling or when attaching to chart:

```mql5
//--- Signal Configuration
input string SignalFile       = "etare_signals.json";   // Signal file path
input int    MagicNumber      = 365060;                 // Unique EA identifier
input int    Slippage         = 50;                     // Max slippage (points)
input bool   TradeEnabled     = true;                   // Enable live execution
input int    CheckIntervalSec = 10;                     // Signal check interval

//--- Risk Management
input double BaseSL_Dollars   = 50.0;    // Base SL in USD
input double RR_Ratio         = 3.0;     // Risk:Reward ratio
input double ScaleMultiplier  = 1.5;     // SL/TP multiplier

//--- Partial Take Profit
input double TP1_Percent      = 50.0;    // % to close at TP1
input double TP1_RR_Ratio     = 1.5;     // R:R for TP1 level

//--- Trailing Stop
input double TrailDistance    = 50.0;    // Trail distance (USD)
input bool   TrailAfterTP1    = true;    // Start trailing after TP1

//--- Position Sizing
input double LotSize          = 2.5;     // Fixed lot size
input bool   UseFixedLots     = true;    // Use fixed lots
input double BaseAccountSize  = 100000.0; // Reference account size
```

### 5.2 ETARE_module.py Configuration

Edit at the top of the file or via constants:

```python
# Trading Symbols (adjust for your broker)
SYMBOLS = [
    "EURUSD.ecn",
    "GBPUSD.ecn",
    "BTCUSD",
    # Add your symbols here
]

# Population Settings
POPULATION_SIZE = 50
TOURNAMENT_SIZE = 3
ELITE_SIZE = 5
EXTINCTION_RATE = 0.3
EXTINCTION_INTERVAL = 10

# Neural Network
INPUT_SIZE = 100  # Auto-detected from features
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 64
OUTPUT_SIZE = 6  # Number of actions

# Grid Trading
GRID_STEP_MIN = 0.00005
GRID_STEP_MAX = 0.0002
ORDERS_COUNT_MIN = 3
ORDERS_COUNT_MAX = 10
BASE_VOLUME_MIN = 0.01
BASE_VOLUME_MAX = 0.1

# Learning Parameters
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
EPSILON = 0.1  # Exploration rate
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.1

# Timing
CYCLE_INTERVAL = 300  # 5 minutes between cycles
```

### 5.3 Signal File Format (etare_signals.json)

The EA expects signals in this JSON format:

```json
{
    "symbol": "BTCUSD",
    "action": "BUY",
    "confidence": 0.85,
    "timestamp": "2026-02-01T12:00:00"
}
```

| Field | Values | Description |
|-------|--------|-------------|
| `symbol` | Trading symbol | Must match chart symbol |
| `action` | `BUY`, `SELL`, `HOLD` | Trade direction |
| `confidence` | 0.0 - 1.0 | Model confidence level |
| `timestamp` | ISO 8601 | Signal generation time |

---

## 6. Testing and Verification

### 6.1 Test MetaTrader 5 Connection

Open Python and run:

```python
import MetaTrader5 as mt5

# Initialize
if not mt5.initialize():
    print(f"Initialize failed: {mt5.last_error()}")
else:
    print("MT5 initialized successfully")

    # Account info
    account = mt5.account_info()
    print(f"Account: {account.login}")
    print(f"Balance: ${account.balance:.2f}")
    print(f"Server: {account.server}")

    # Symbol info
    symbol = "BTCUSD"  # Adjust for your broker
    info = mt5.symbol_info(symbol)
    if info:
        print(f"Symbol: {symbol}")
        print(f"Bid: {info.bid}, Ask: {info.ask}")

    mt5.shutdown()
```

### 6.2 Test Signal File Writing

```python
import json
import os

# Adjust path to your MT5 Files folder
signal_path = r"C:\Users\<YourUsername>\AppData\Roaming\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Files\etare_signals.json"

signal = {
    "symbol": "BTCUSD",
    "action": "BUY",
    "confidence": 0.75,
    "timestamp": "2026-02-01T12:00:00"
}

with open(signal_path, 'w') as f:
    json.dump(signal, f)

print(f"Signal written to: {signal_path}")
print(f"File exists: {os.path.exists(signal_path)}")
```

### 6.3 Test EA Signal Reading (Dry Run)

1. Set `TradeEnabled = false` in EA inputs
2. Attach EA to chart
3. Run the signal writing test above
4. Check Experts tab in MT5 for:
   ```
   DRY RUN >> Signal: BUY (75.0%) | Scaled SL: $75.00 | TP1: $112.50 | TP2: $225.00
   ```

### 6.4 Full System Test (Paper Trading)

1. **Start ETARE Python module**:
   ```cmd
   cd C:\Trading\ETARE
   etare_env\Scripts\activate
   python ETARE_module.py
   ```

2. **Verify in MT5 Experts tab**:
   - Signal reading messages
   - DRY RUN trade logs (if TradeEnabled=false)

3. **Check database creation**:
   ```
   C:\Trading\ETARE\trading_history.db
   ```

### 6.5 Live Trading Verification

1. Set `TradeEnabled = true` in EA
2. Monitor for actual order execution
3. Verify in MT5:
   - Trade tab shows open positions
   - History tab shows completed trades
   - Journal tab shows execution logs

---

## 7. Troubleshooting

### 7.1 MetaTrader 5 Issues

| Problem | Solution |
|---------|----------|
| EA not appearing in Navigator | Recompile in MetaEditor, restart MT5 |
| "Algorithm trading disabled" | Enable in Tools -> Options -> Expert Advisors |
| EA shows frowning face | Check Experts tab for errors, verify symbol selected |
| Signal file not found | Verify path matches MT5 Files folder |

### 7.2 Python Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: MetaTrader5` | `pip install MetaTrader5` |
| MT5 initialization failed | Ensure MT5 is running and logged in |
| Permission denied on signal file | Run Python as administrator or check file permissions |
| Database locked | Close any SQLite viewers, ensure single instance |

### 7.3 Signal Communication Issues

| Problem | Solution |
|---------|----------|
| EA not reading signals | Check file path, verify JSON format is valid |
| Signals not generating | Check Python console for errors, verify MT5 data connection |
| Symbol mismatch | Ensure signal symbol exactly matches chart symbol |

### 7.4 Trade Execution Issues

| Problem | Solution |
|---------|----------|
| Orders rejected | Check account permissions, verify lot size within limits |
| Slippage errors | Increase `Slippage` parameter |
| "Invalid stops" | Verify SL/TP distances meet broker minimums |
| Position not closing | Check MagicNumber matches, verify position exists |

### 7.5 Log Locations

- **MT5 EA logs**: Experts tab in Terminal window
- **MT5 Journal**: Journal tab in Terminal window
- **Python logs**: Console output (configure logging level in ETARE_module.py)
- **Database**: `trading_history.db` (view with SQLite browser)

### 7.6 Common Error Codes

| MT5 Retcode | Meaning | Action |
|-------------|---------|--------|
| 10004 | Requote | Retry with updated price |
| 10006 | Request rejected | Check account/symbol permissions |
| 10016 | Invalid stops | Increase SL/TP distance |
| 10019 | Not enough money | Reduce lot size |
| 10025 | Invalid price | Use current market price |

---

## 8. Directory Structure Reference

```
C:\Trading\
    ETARE\
        ETARE_module.py          # Python signal generator
        trading_history.db       # SQLite database (auto-created)
        etare_env\               # Python virtual environment
            Scripts\
            Lib\

C:\Users\<Username>\AppData\Roaming\MetaQuotes\Terminal\<TERMINAL_ID>\
    MQL5\
        Experts\
            BG_Executor.mq5      # EA source code
            BG_Executor.ex5      # Compiled EA (auto-generated)
        Files\
            etare_signals.json   # Signal file (inter-process communication)
        Logs\
            *.log                # EA execution logs
```

---

## 9. Quick Start Checklist

- [ ] MetaTrader 5 installed and logged into broker account
- [ ] Python 3.9+ installed with MetaTrader5 package
- [ ] BG_Executor.mq5 compiled and in Experts folder
- [ ] ETARE_module.py configured with correct signal path
- [ ] Algo trading enabled in MT5 options
- [ ] Auto Trading enabled (green button)
- [ ] EA attached to correct chart/symbol
- [ ] Signal file path verified (both Python and EA match)
- [ ] Dry run test completed successfully
- [ ] Database created and accessible

---

## 10. Support Resources

- **MetaTrader 5 Documentation**: [https://www.mql5.com/en/docs](https://www.mql5.com/en/docs)
- **MetaTrader5 Python Package**: [https://pypi.org/project/MetaTrader5/](https://pypi.org/project/MetaTrader5/)
- **MQL5 Community**: [https://www.mql5.com/en/forum](https://www.mql5.com/en/forum)

---

*Document Version: 1.0*
*Last Updated: February 2026*
*System: ETARE Trading System v2.0*
