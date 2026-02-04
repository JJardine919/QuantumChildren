# Quantum Trading System - Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          QUANTUM TRADING SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ BlueGuardian │    │ GetLeveraged │    │    Atlas     │                   │
│  │    BRAIN     │    │    BRAIN     │    │    BRAIN     │                   │
│  │              │    │              │    │              │                   │
│  │  Account:    │    │  Accounts:   │    │  Account:    │                   │
│  │  366604      │    │  113326      │    │  212000584   │                   │
│  │  365060      │    │  113328      │    │              │                   │
│  │              │    │  107245      │    │              │                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         └───────────────────┼───────────────────┘                            │
│                             │                                                │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      SHARED COMPONENTS                                │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │    LSTM     │  │   Regime    │  │   Partial   │  │    Risk     │  │   │
│  │  │   Models    │  │  Detector   │  │    Close    │  │   Manager   │  │   │
│  │  │             │  │             │  │             │  │             │  │   │
│  │  │ 8 features  │  │ Compression │  │ 50% at TP/2 │  │ 5% daily    │  │   │
│  │  │ 128 hidden  │  │   based     │  │ Breakeven   │  │ 10% max DD  │  │   │
│  │  │ 2 layers    │  │   CLEAN/    │  │    SL       │  │             │  │   │
│  │  │             │  │   VOLATILE/ │  │             │  │             │  │   │
│  │  │ BUY/SELL/   │  │   CHOPPY    │  │             │  │             │  │   │
│  │  │ HOLD        │  │             │  │             │  │             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                             │                                                │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      METATRADER 5 TERMINALS                           │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ BlueGuardian│  │ GetLeveraged│  │   Atlas     │                   │   │
│  │  │    MT5      │  │    MT5      │  │    MT5      │                   │   │
│  │  │  Terminal   │  │  Terminal   │  │  Terminal   │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │     MARKETS     │                                       │
│                    │                 │                                       │
│                    │   BTCUSD        │                                       │
│                    │   ETHUSD        │                                       │
│                    └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
┌───────────────────────────────────────────────────────────────────────┐
│                           TRADING CYCLE                                │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. FETCH DATA                                                         │
│     MT5 Terminal → Get last 100 bars (1-minute) for BTCUSD/ETHUSD     │
│                                                                        │
│  2. CALCULATE FEATURES                                                 │
│     Raw Bars → Calculate 8 indicators:                                 │
│                                                                        │
│     ┌─────────────────────────────────────────────────────────────┐   │
│     │  RSI (14)        → Relative Strength Index                  │   │
│     │  MACD            → Moving Average Convergence Divergence    │   │
│     │  MACD Signal     → MACD Signal Line                         │   │
│     │  BB Upper        → Bollinger Band Upper (20, 2)             │   │
│     │  BB Lower        → Bollinger Band Lower (20, 2)             │   │
│     │  Momentum (10)   → Price momentum                           │   │
│     │  ROC (10)        → Rate of Change                           │   │
│     │  ATR (14)        → Average True Range (volatility)          │   │
│     └─────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  3. NORMALIZE                                                          │
│     Features → Global Z-score normalization (mean=0, std=1)           │
│                                                                        │
│  4. CHECK REGIME                                                       │
│     Price Data → Compression Analysis:                                 │
│                                                                        │
│     ┌─────────────────────────────────────────────────────────────┐   │
│     │  Compress prices with zlib                                  │   │
│     │  ratio = original_size / compressed_size                    │   │
│     │                                                             │   │
│     │  ratio >= 3.5  →  CLEAN (trade)                             │   │
│     │  ratio >= 2.5  →  VOLATILE (hold)                           │   │
│     │  ratio < 2.5   →  CHOPPY (hold)                             │   │
│     └─────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  5. GET PREDICTION (only if CLEAN regime)                              │
│     Normalized Features → LSTM Model → Prediction:                     │
│                                                                        │
│     ┌─────────────────────────────────────────────────────────────┐   │
│     │  Class 0 = BUY   (confidence > 0.55)                        │   │
│     │  Class 1 = SELL  (confidence > 0.55)                        │   │
│     │  Class 2 = HOLD  (or if confidence <= 0.55)                 │   │
│     └─────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  6. EXECUTE TRADE                                                      │
│     Prediction → Calculate SL/TP → Send Order to MT5                  │
│                                                                        │
│  7. MANAGE POSITIONS                                                   │
│     Monitor open positions → 50% partial close → Breakeven SL         │
│                                                                        │
│  8. REPEAT                                                             │
│     Sleep 60 seconds → Go to step 1                                   │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Trading Brains

Each broker has its own "brain" script that runs independently:

| Brain                  | File                    | Accounts          |
|------------------------|-------------------------|-------------------|
| BlueGuardian           | BRAIN_BLUEGUARDIAN.py   | 366604, 365060    |
| GetLeveraged           | BRAIN_GETLEVERAGED.py   | 113326, 113328, 107245 |
| Atlas                  | BRAIN_ATLAS.py          | 212000584         |

Each brain:
- Connects to its MT5 terminal
- Runs on a 60-second loop
- Trades BTCUSD and ETHUSD
- Has its own magic numbers for order identification

---

### 2. LSTM Neural Network

```
┌─────────────────────────────────────────────────────────────────┐
│                     LSTM MODEL ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT LAYER                                                     │
│  └── 8 features × 30 time steps = 240 inputs                    │
│                                                                  │
│  LSTM LAYER 1                                                    │
│  └── 128 hidden units                                           │
│                                                                  │
│  LSTM LAYER 2                                                    │
│  └── 128 hidden units                                           │
│                                                                  │
│  DROPOUT                                                         │
│  └── 40% dropout rate                                           │
│                                                                  │
│  OUTPUT LAYER                                                    │
│  └── 3 classes (BUY, SELL, HOLD)                                │
│                                                                  │
│  ACTIVATION                                                      │
│  └── Softmax → probabilities for each class                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Model Files Location:** `top_50_experts/`

Each `.pth` file is a trained expert for a specific symbol.

---

### 3. Regime Detection

The "quantum" part - uses data compression to detect market regime:

```
┌─────────────────────────────────────────────────────────────────┐
│                     REGIME DETECTION LOGIC                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WHY COMPRESSION?                                                │
│  ├── Random data compresses poorly (low ratio)                  │
│  ├── Structured/trending data compresses well (high ratio)      │
│  └── High compression ratio = price pattern is predictable      │
│                                                                  │
│  HOW IT WORKS:                                                   │
│  1. Take last 100 price closes                                  │
│  2. Convert to bytes                                            │
│  3. Compress with zlib (level 9)                                │
│  4. Calculate ratio = original / compressed                     │
│                                                                  │
│  INTERPRETATION:                                                 │
│  ├── ratio >= 3.5 → CLEAN    → Market is structured, TRADE      │
│  ├── ratio >= 2.5 → VOLATILE → Moderate structure, HOLD         │
│  └── ratio <  2.5 → CHOPPY   → Random/noisy, HOLD               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4. Risk Management

```
┌─────────────────────────────────────────────────────────────────┐
│                     RISK MANAGEMENT RULES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PER-TRADE RISK:                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Symbol    Lot      SL Distance    Loss       TP Distance │  │
│  │  ─────────────────────────────────────────────────────── │  │
│  │  ETHUSD    0.1      $5             ~$0.50     $15         │  │
│  │  BTCUSD    0.01     $50            ~$0.50     $150        │  │
│  │                                                           │  │
│  │  Risk:Reward = 1:3                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  PARTIAL CLOSE:                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. When price reaches 50% of TP:                         │  │
│  │     → Close half the position (lock in ~$0.75 profit)     │  │
│  │                                                           │  │
│  │  2. After partial close:                                  │  │
│  │     → Move SL to entry price (breakeven)                  │  │
│  │     → Let remaining half run to full TP                   │  │
│  │                                                           │  │
│  │  Result: Guaranteed profit + free runner                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ACCOUNT LIMITS (PROP FIRM RULES):                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Daily Loss Limit:     5% of account                      │  │
│  │  Maximum Drawdown:     10% of account                     │  │
│  │  Profit Target:        10% of account                     │  │
│  │                                                           │  │
│  │  If limits approached → Brain stops trading               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5. Feature Engineering

The 8 features fed to the LSTM:

| Feature      | Calculation                          | Purpose                    |
|--------------|--------------------------------------|----------------------------|
| RSI(14)      | Relative Strength Index              | Overbought/oversold        |
| MACD         | EMA(12) - EMA(26)                    | Trend momentum             |
| MACD Signal  | EMA(9) of MACD                       | Trend change signals       |
| BB Upper     | SMA(20) + 2*StdDev                   | Upper volatility band      |
| BB Lower     | SMA(20) - 2*StdDev                   | Lower volatility band      |
| Momentum(10) | Close - Close[10]                    | Price momentum             |
| ROC(10)      | (Close - Close[10]) / Close[10]      | Rate of change %           |
| ATR(14)      | Average True Range                   | Volatility measure         |

**Normalization:** Global Z-score (subtract mean, divide by std of entire dataset)

---

## File Structure

```
C:\QuantumTrading\
│
├── BRAIN_BLUEGUARDIAN.py      # Main trading brain for BlueGuardian
├── BRAIN_GETLEVERAGED.py      # Main trading brain for GetLeveraged
├── BRAIN_ATLAS.py             # Main trading brain for Atlas
│
├── lstm_model.py              # LSTM neural network definition
├── regime_detector.py         # Compression-based regime detection
├── partial_close.py           # 50% close + breakeven logic
│
├── panic_close_all.py         # Emergency: close all positions
├── check_mt5_status.py        # Check account/connection status
│
├── START_BLUEGUARDIAN_BRAIN.bat
├── START_GETLEVERAGED_BRAIN.bat
├── START_ALL_BRAINS.bat
├── PANIC_CLOSE_ALL.bat
│
└── top_50_experts\            # Trained LSTM models
    ├── top_50_manifest.json   # Model metadata
    ├── btcusd_expert_1.pth    # Model weights
    ├── ethusd_expert_1.pth
    └── ...
```

---

## Safety Systems

### Emergency Guardian
- Monitors all accounts continuously
- Triggers panic close if drawdown limit approached
- Can be started with: `python emergency_stop_guardian.py`

### Panic Close
- Immediately closes ALL open positions across ALL accounts
- Run with: `python panic_close_all.py` or double-click `PANIC_CLOSE_ALL.bat`

### Magic Numbers
- Each account uses unique magic numbers for order tracking
- Prevents brains from interfering with each other's orders
- BlueGuardian: 366001, 365001
- GetLeveraged: 113001, 113002, 107001
- Atlas: 212001

---

## Trading Logic Summary

```
EVERY 60 SECONDS:
│
├── 1. Check account health (drawdown, daily loss)
│   └── If limits exceeded → Stop trading
│
├── 2. Get market data (100 bars)
│
├── 3. Check regime
│   └── If not CLEAN → Hold (no new trades)
│
├── 4. Calculate features (8 indicators)
│
├── 5. Normalize features (Z-score)
│
├── 6. Get LSTM prediction
│   └── If confidence < 0.55 → Hold
│
├── 7. Execute trade if BUY or SELL signal
│   ├── Set SL (5 for ETH, 50 for BTC)
│   └── Set TP (15 for ETH, 150 for BTC)
│
├── 8. Manage existing positions
│   ├── Check if at 50% TP → Close half
│   └── After partial → Move SL to breakeven
│
└── 9. Sleep 60 seconds → Repeat
```

---

*System designed by Dad + Claude | January 2026*
