# Quantum Trading System - Windows Rebuild Guide

## Instructions for Claude

Pass this entire document to Claude with this prompt:
> "Rebuild this trading system on my Windows computer using this guide."

---

## What This System Does

Automated crypto trading (BTCUSD, ETHUSD) on prop firm accounts using LSTM neural networks with quantum regime detection.

**Prop Firm Accounts:**
- BlueGuardian: 366604 ($5K), 365060 ($100K)
- GetLeveraged: 113326, 113328, 107245
- Atlas: 212000584

---

## Step 1: Install Required Software on Windows

### Install Python 3.10 or 3.11
1. Download from https://python.org/downloads/
2. During install, CHECK "Add Python to PATH"
3. Verify: Open Command Prompt, type `python --version`

### Install MetaTrader 5
1. Download MT5 from each broker's website
2. Install separate terminals for each broker
3. Login to each account using credentials

### Install Python Packages
Open Command Prompt and run:
```cmd
pip install torch numpy pandas MetaTrader5 pywt scipy scikit-learn
```

---

## Step 2: Create Project Folder

```cmd
mkdir C:\QuantumTrading
cd C:\QuantumTrading
```

All files will go in this folder.

---

## Step 3: Trading Strategy Parameters

**RISK MANAGEMENT (DO NOT CHANGE):**
```
Stop Loss:
  - ETH: $5 price move (~$0.50 loss at 0.1 lot)
  - BTC: $50 price move (~$0.50 loss at 0.01 lot)

Take Profit:
  - ETH: $15 price move (~$1.50 profit)
  - BTC: $150 price move (~$1.50 profit)
  - Ratio: 3:1 reward to risk

Partial Close:
  - At 50% of TP: Close half the position
  - After partial: Move SL to breakeven
  - Let remaining half run to full TP

Lot Sizes:
  - ETH: 0.1 (scale by balance)
  - BTC: 0.01 (scale by balance)
```

**REGIME DETECTION:**
```
CLEAN (fidelity >= 0.95):    TRADE
VOLATILE (0.85 - 0.95):      HOLD
CHOPPY (< 0.85):             HOLD
```

**LSTM SETTINGS:**
```
Input Size:     8 features
Hidden Size:    128
Layers:         2
Sequence:       30 bars
Confidence:     0.55 minimum
Features:       rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr
Normalization:  Global Z-score (NOT rolling window)
```

---

## Step 4: Core Code to Create

### File 1: BRAIN_BLUEGUARDIAN.py

Account configuration:
```python
import MetaTrader5 as mt5
import torch
import numpy as np
import pandas as pd

ACCOUNTS = {
    'BG_5K_INSTANT': {
        'account': 366604,
        'terminal_path': r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
        'symbols': ['BTCUSD', 'ETHUSD'],
        'magic_number': 366001,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
    },
    'BG_100K_CHALLENGE': {
        'account': 365060,
        'password': ')8xaE(gAuU',
        'symbols': ['BTCUSD', 'ETHUSD'],
        'magic_number': 365001,
        'locked': True,
    },
}

# Stop Loss and Take Profit calculation
def get_sl_tp(symbol, direction, entry_price):
    if 'ETH' in symbol:
        sl_distance = 5.0    # ~$0.50 loss at 0.1 lot
        tp_distance = 15.0   # ~$1.50 profit (3:1)
    elif 'BTC' in symbol:
        sl_distance = 50.0   # ~$0.50 loss at 0.01 lot
        tp_distance = 150.0  # ~$1.50 profit (3:1)

    if direction == 'BUY':
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance

    return sl, tp

# Feature columns - MUST match training exactly
FEATURE_COLS = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
```

### File 2: lstm_model.py

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=3, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
```

### File 3: regime_detector.py

```python
import numpy as np
import zlib

def analyze_regime(prices):
    """Compression-based regime detection"""
    data_bytes = prices.astype(np.float32).tobytes()
    compressed = zlib.compress(data_bytes, level=9)
    ratio = len(data_bytes) / len(compressed)

    if ratio >= 3.5:
        return "CLEAN", 0.96    # Safe to trade
    elif ratio >= 2.5:
        return "VOLATILE", 0.88  # Hold
    else:
        return "CHOPPY", 0.75    # Hold
```

### File 4: partial_close.py

```python
import MetaTrader5 as mt5

def manage_positions():
    """50% partial close at halfway to TP, then breakeven SL"""
    positions = mt5.positions_get()
    if not positions:
        return

    for pos in positions:
        tp_distance = abs(pos.tp - pos.price_open)

        if pos.type == mt5.ORDER_TYPE_BUY:
            current_profit_distance = pos.price_current - pos.price_open
        else:
            current_profit_distance = pos.price_open - pos.price_current

        # If at 50% of TP
        if current_profit_distance >= (tp_distance * 0.5):
            # Close half
            half_volume = round(pos.volume / 2, 2)
            if half_volume >= mt5.symbol_info(pos.symbol).volume_min:
                close_partial(pos, half_volume)
                # Move SL to breakeven
                modify_sl_to_breakeven(pos)
```

---

## Step 5: Expert Models

The trained LSTM models should be in: `top_50_experts/`
- Manifest: `top_50_experts/top_50_manifest.json`
- Model files: `*.pth` files

**If models are missing:**
1. Copy from the original machine, OR
2. Retrain using the ETARE training system

---

## Step 6: Batch Files to Create

### START_BLUEGUARDIAN_BRAIN.bat
```batch
@echo off
cd /d C:\QuantumTrading
python BRAIN_BLUEGUARDIAN.py
pause
```

### START_GETLEVERAGED_BRAIN.bat
```batch
@echo off
cd /d C:\QuantumTrading
python BRAIN_GETLEVERAGED.py
pause
```

### START_ALL_BRAINS.bat
```batch
@echo off
start "" cmd /k "cd /d C:\QuantumTrading && python BRAIN_BLUEGUARDIAN.py"
start "" cmd /k "cd /d C:\QuantumTrading && python BRAIN_GETLEVERAGED.py"
start "" cmd /k "cd /d C:\QuantumTrading && python BRAIN_ATLAS.py"
```

### PANIC_CLOSE_ALL.bat
```batch
@echo off
cd /d C:\QuantumTrading
python panic_close_all.py
pause
```

---

## Step 7: Commands to Run

Start a brain:
```cmd
cd C:\QuantumTrading
python BRAIN_BLUEGUARDIAN.py
```

Emergency close all:
```cmd
python panic_close_all.py
```

Check account status:
```cmd
python check_mt5_status.py
```

---

## Critical Bugs to Avoid

1. **Feature Mismatch**: Use EXACTLY these 8 features: `rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr`. Training and trading MUST match.

2. **Lot Size**: Check `symbol_info.volume_min` for each symbol. ETH is usually 0.1 min, BTC is 0.01 min.

3. **Order Filling**: Use `mt5.ORDER_FILLING_FOK` (not `SYMBOL_FILLING_FOK`).

4. **Normalization**: Use GLOBAL Z-score normalization across entire dataset, NOT rolling windows.

5. **MT5 Terminal Paths**: Update paths in config to match where you installed MT5.

---

## Prop Firm Rules

- 5% daily loss limit
- 10% maximum drawdown
- 10% profit target
- Trade only crypto: BTCUSD, ETHUSD

---

## Quick Start Prompt for Claude

Copy and paste this to Claude to get started:

```
I need to rebuild my dad's quantum trading system on Windows.

Requirements:
1. Trade BTCUSD and ETHUSD on prop firm accounts
2. BlueGuardian accounts: 366604, 365060
3. Stop loss: ~$0.50 per trade with 3:1 reward ratio
4. 50% partial close at halfway to TP, then move SL to breakeven
5. Only trade in CLEAN regime (compression fidelity >= 0.95)
6. LSTM experts with 8 features: rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr
7. Global Z-score normalization (not rolling)

Please create the complete trading brain scripts for Windows.
```

---

## Folder Structure

```
C:\QuantumTrading\
├── BRAIN_BLUEGUARDIAN.py
├── BRAIN_GETLEVERAGED.py
├── BRAIN_ATLAS.py
├── lstm_model.py
├── regime_detector.py
├── partial_close.py
├── panic_close_all.py
├── check_mt5_status.py
├── START_BLUEGUARDIAN_BRAIN.bat
├── START_GETLEVERAGED_BRAIN.bat
├── START_ALL_BRAINS.bat
├── PANIC_CLOSE_ALL.bat
└── top_50_experts\
    ├── top_50_manifest.json
    └── *.pth (model files)
```

---

*Built by Dad + Claude | January 2026*
*For Windows 10/11*
