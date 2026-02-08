# QUANTUM CHILDREN - GETTING STARTED GUIDE

**AI-Driven Trading System | Version 2.0**

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [First Run](#4-first-run)
5. [Understanding the System](#5-understanding-the-system)
6. [Trading Modes](#6-trading-modes)
7. [Troubleshooting](#7-troubleshooting)
8. [FAQ](#8-faq)

---

## 1. System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11 (64-bit) |
| **Python** | 3.9 or newer |
| **RAM** | 8 GB minimum |
| **Storage** | 2 GB free space |
| **Internet** | Stable connection |

### Required Software

- **MetaTrader 5** - Download from your broker or [metatrader5.com](https://www.metatrader5.com)
- **Python 3.9+** - Download from [python.org](https://www.python.org)

### Supported Brokers

Any broker with MT5 support works. Tested with:
- BlueGuardian
- Atlas Funded
- GetLeveraged
- FTMO

---

## 2. Installation

### Option A: Automated (Recommended)

1. **Download** the Quantum Children folder
2. **Run** the installer:
   ```
   Double-click: INSTALL.bat
   ```
   Or:
   ```bash
   python INSTALL.py
   ```
3. **Follow** the prompts to:
   - Accept terms
   - Install dependencies
   - Configure credentials

### Option B: Manual

1. **Install dependencies**:
   ```bash
   pip install numpy pandas MetaTrader5 python-dotenv requests
   ```

2. **Optional packages** (for full features):
   ```bash
   pip install torch catboost scikit-learn qiskit
   ```

3. **Configure credentials**:
   ```bash
   cd QuantumTradingLibrary
   copy .env.example .env
   # Edit .env with your passwords
   ```

---

## 3. Configuration

### 3.1 Credentials Setup

Your MT5 account passwords are stored in `.env` (never in git):

**Location**: `QuantumTradingLibrary/.env`

```
BG_INSTANT_PASSWORD=your_password
BG_CHALLENGE_PASSWORD=your_password
ATLAS_PASSWORD=your_password
GL_1_PASSWORD=your_password
GL_2_PASSWORD=your_password
GL_3_PASSWORD=your_password
```

### 3.2 Trading Settings

All trading parameters are in `MASTER_CONFIG.json`:

**Location**: `QuantumTradingLibrary/MASTER_CONFIG.json`

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_LOSS_DOLLARS` | 1.00 | Maximum loss per trade |
| `INITIAL_SL_DOLLARS` | 0.60 | Starting stop loss |
| `TP_MULTIPLIER` | 3.0 | Take profit = SL x this |
| `CONFIDENCE_THRESHOLD` | 0.22 | Minimum signal confidence |
| `CHECK_INTERVAL_SECONDS` | 30 | How often to check positions |

### 3.3 Account Configuration

Accounts are defined in `MASTER_CONFIG.json` under `ACCOUNTS`:

```json
"ATLAS": {
    "account": 212000584,
    "server": "AtlasFunded-Server",
    "symbols": ["BTCUSD", "ETHUSD", "XAUUSD"],
    "enabled": true
}
```

---

## 4. First Run

### Step 1: Verify Installation

```bash
cd QuantumTradingLibrary
python config_loader.py
```

You should see your settings displayed.

### Step 2: Check Credentials

```bash
python credential_manager.py
```

All accounts should show `[OK]`.

### Step 3: Test MT5 Connection

1. Open your MT5 terminal
2. Login to your account
3. Run:
   ```bash
   python -c "import MetaTrader5 as mt5; print('Connected:', mt5.initialize())"
   ```

### Step 4: Paper Trading (Recommended)

Before live trading:
1. Use a **demo account**
2. Set `enabled: false` in config
3. Monitor signals for 1-2 weeks
4. Verify the system behaves as expected

### Step 5: Go Live

When ready:
```bash
python BRAIN_ATLAS.py      # For Atlas account
python BRAIN_BG_INSTANT.py # For BlueGuardian
```

---

## 5. Understanding the System

### How It Works

```
Market Data (MT5)
      |
      v
+------------------+
| Compression      |  <-- Quantum-inspired entropy analysis
| Analysis         |
+------------------+
      |
      v
+------------------+
| Regime Detection |  <-- CLEAN (trade) / VOLATILE (wait)
+------------------+
      |
      v
+------------------+
| Signal Generator |  <-- Neural network predictions
+------------------+
      |
      v
+------------------+
| Trade Execution  |  <-- With fixed-dollar stop loss
+------------------+
```

### Key Concepts

**Compression Ratio**: Measures market "noise". Lower = cleaner patterns.

**Market Regime**:
- **CLEAN**: Low noise, predictable patterns - TRADE
- **VOLATILE**: High noise, unpredictable - WAIT

**Confidence Score**: 0.0 to 1.0. Trades only execute above threshold.

**Rolling Stop Loss**: SL moves with price to lock in profits.

---

## 6. Trading Modes

### BlueGuardian Mode

For BlueGuardian prop firm accounts:
```bash
python BRAIN_BG_INSTANT.py   # $5K Instant
python BRAIN_BG_CHALLENGE.py # $100K Challenge
```

### Atlas Mode

For Atlas Funded accounts:
```bash
python BRAIN_ATLAS.py
```

### GetLeveraged Mode

For GetLeveraged accounts:
```bash
python BRAIN_GETLEVERAGED.py
```

### Grid Trading Mode

For automated grid trading:
```bash
# In MetaTrader 5:
# Load: XAUUSD_GridTrader.ex5
# Or:   BTCUSD_GridTrader.ex5
```

---

## 7. Troubleshooting

### MT5 Won't Connect

**Symptoms**: `mt5.initialize()` returns `False`

**Solutions**:
1. Make sure MT5 terminal is running
2. Login to your account in MT5 first
3. Enable "Algo Trading" in MT5 settings
4. Check firewall isn't blocking Python

### Credentials Not Loading

**Symptoms**: `CredentialError: Password not found`

**Solutions**:
1. Check `.env` file exists in `QuantumTradingLibrary/`
2. Verify variable names match exactly (case-sensitive)
3. Install python-dotenv: `pip install python-dotenv`
4. Restart Python after editing `.env`

### Trades Not Executing

**Symptoms**: Signals generated but no trades

**Possible Causes**:
1. Confidence below threshold (check logs)
2. Market regime is VOLATILE (system is waiting)
3. Max positions reached
4. Account not enabled in config

### Import Errors

**Symptoms**: `ModuleNotFoundError`

**Solution**:
```bash
pip install -r requirements.txt
```

Or install individual packages:
```bash
pip install numpy pandas MetaTrader5 python-dotenv
```

---

## 8. FAQ

### General

**Q: Is this really free?**
A: Yes. The system collects anonymized trading signals to improve the models. Better data = better predictions for everyone.

**Q: Can I modify the code?**
A: Yes. Keep the entropy collector running so the network improves.

**Q: What data is collected?**
A: Trading signals, entropy values, and outcomes. NO passwords, account numbers, or personal information. See `PRIVACY_POLICY.md`.

### Trading

**Q: What's the expected win rate?**
A: Backtests show 70-89% depending on market conditions. Past performance doesn't guarantee future results.

**Q: How much can I lose?**
A: Maximum loss per trade is controlled by `MAX_LOSS_DOLLARS` (default $1.00). Total exposure depends on number of positions.

**Q: Which symbols work best?**
A: Optimized for BTCUSD, XAUUSD, ETHUSD. Other pairs may work but haven't been extensively tested.

### Technical

**Q: Why one script per account?**
A: MT5 Python API is singleton. Running multiple accounts in one script causes conflicts and can close trades unexpectedly.

**Q: Can I run on VPS?**
A: Yes. See the `docker_vps/` folder for deployment scripts.

**Q: What about Mac/Linux?**
A: MT5 is Windows-only. Use Wine on Linux or a Windows VPS.

---

## Quick Reference

### Important Files

| File | Purpose |
|------|---------|
| `INSTALL.py` | Automated installer |
| `QuantumTradingLibrary/MASTER_CONFIG.json` | All settings |
| `QuantumTradingLibrary/.env` | Credentials (secret) |
| `QuantumTradingLibrary/config_loader.py` | Settings loader |
| `QuantumTradingLibrary/credential_manager.py` | Secure credential access |
| `QuantumTradingLibrary/BRAIN_*.py` | Trading scripts |

### Important Commands

```bash
# Check settings
python config_loader.py

# Check credentials
python credential_manager.py

# Test MT5 connection
python -c "import MetaTrader5 as mt5; print(mt5.initialize())"

# Run trading (example)
python BRAIN_ATLAS.py
```

### Support

- Documentation: This file
- Risk Warning: `RISK_DISCLAIMER.md`
- Terms: `TERMS_OF_SERVICE.md`
- Privacy: `PRIVACY_POLICY.md`
- Website: quantum-children.com

---

**Happy Trading!**

*Remember: Start with paper trading. Never risk more than you can afford to lose.*
