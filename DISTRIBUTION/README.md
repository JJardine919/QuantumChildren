# ⚡ QUANTUM CHILDREN - FREE TRADING SYSTEM

## The Deal

This is a **free**, fully-functional trading system. No catch, no subscription, no BS.

**What you get:**
- Quantum-enhanced market analysis
- Compression-based regime detection
- LSTM neural network predictions
- Works with any MT5 broker

**What we get:**
- Your trading signals (anonymized) help improve the models
- More data = better predictions for everyone
- You benefit from a system that gets smarter over time

That's it. You trade, the system learns, everyone wins.

---

## Quick Start (5 minutes)

### 1. Install Python Requirements

```bash
pip install numpy pandas MetaTrader5 torch requests
```

### 2. Download This Folder

Put it anywhere. We recommend:
```
C:\Trading\QuantumChildren\
```

### 3. Open MT5 and Login

Login to any MT5 broker account. Demo or live, doesn't matter.

### 4. Run It

```bash
cd C:\Trading\QuantumChildren
python quantum_trader.py
```

That's it. It's running.

---

## What It Does

1. **Reads market data** from MT5 (BTCUSD, XAUUSD, ETHUSD by default)
2. **Compresses the data** to find patterns (entropy analysis)
3. **Detects market regime** (CLEAN = tradeable, VOLATILE = wait)
4. **Generates signals** when confidence is high enough
5. **Sends signals** to the QuantumChildren network (this improves the models)
6. **Executes trades** based on your settings

---

## Configuration

Edit `config.json`:

```json
{
    "symbols": ["BTCUSD", "XAUUSD"],
    "lot_size": 0.01,
    "confidence_threshold": 0.55,
    "max_positions": 3,
    "enable_trading": false
}
```

**Start with `enable_trading: false`** to watch it before going live.

---

## Files

| File | Purpose |
|------|---------|
| `quantum_trader.py` | Main trading system |
| `entropy_collector.py` | Sends data to network (don't delete) |
| `config.json` | Your settings |
| `quantum_data/` | Local data backup |

---

## FAQ

**Is this really free?**
Yes. The value is in the aggregated data, not individual sales.

**Can I modify it?**
Yes. Just don't remove the entropy_collector - that's how the network improves.

**Is my account info collected?**
No. Only signals, entropy values, and trade outcomes. No passwords, no account numbers.

**What if I lose money?**
This is trading. There's risk. Start with demo accounts or small positions.

**How do I know it's getting better?**
Check the network stats: http://203.161.61.61:8888/stats

---

## Support

This is free software. No paid support.

But the community helps each other:
- GitHub Issues (coming soon)
- Discord (coming soon)

---

## License

MIT License. Do whatever you want.

Just keep the entropy collector running - that's how we all win.

---

*Built by QuantumChildren*
*The secret is in the compression.*
