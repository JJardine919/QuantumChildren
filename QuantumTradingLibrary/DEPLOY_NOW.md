# 🚀 ETARE Deployment Instructions (Optimized)

**Status:** System is optimized for **BTCUSD** ($150k account target).
**Trainer Result:** BTCUSD survived Chaos Mode in Batch 2 with +$258k profit.

---

## 1. Start the Brain (Python)

Open a terminal in `QuantumTradingLibrary` and run:

```powershell
python 06_Integration/HybridBridge/etare_signal_generator.py
```

*This will start generating signals based on the latest evolutionary data.*

---

## 2. Start the Body (MT5)

1.  Open **MetaTrader 5**.
2.  Open a **BTCUSD** chart (M5 timeframe).
3.  Drag `ETARE_Executor.ex5` onto the chart.
4.  **Inputs:**
    *   `SignalFile`: `etare_signals.json` (default is fine)
    *   `MagicNumber`: `85000`
    *   `TradeEnabled`: `true`

---

## 3. Future Optimization (Forex Pairs)

The trainer skipped Forex pairs (EURUSD, etc.) because history was missing.
**To fix this for next time:**
1.  Open EURUSD, GBPUSD, etc. charts in MT5.
2.  Press `Home` key to scroll back 1-2 years to force data download.
3.  Run the trainer again:
    ```powershell
    python 06_Integration/HybridBridge/ETARE_WalkForward_Trainer.py
    ```

---

**Good luck! The system is ready.** 
*Generated: Jan 20, 2026*
