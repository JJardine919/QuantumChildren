# QNIF 3-Week Simulation Guide

Built by Biskits - 20 years of building complicated systems taught me this: test it before you deploy it.

## What This Does

`qnif_3week_sim.py` runs a 3-week backtest of the QNIF pipeline against PropFarmAccount prop firm rules. It:

1. Fetches 21 days of M5 data (from MT5 or CSV)
2. Runs QNIF_Engine.process_pulse() on rolling 256-bar windows
3. Feeds signals into multiple PropFarmAccount instances (55 parameter sets)
4. Evaluates prop firm challenge pass/fail (daily DD, total DD, profit target)
5. Reports win rate, profit factor, best performing accounts
6. Saves detailed JSON results to `sim_results/`

This is **not** a toy. This is what you run before risking capital.

---

## Usage

### Basic Run (MT5 Data)
```bash
python qnif_3week_sim.py --account ATLAS
```

### CSV Mode (Offline)
```bash
python qnif_3week_sim.py --csv
```

### Custom Symbols and Accounts
```bash
python qnif_3week_sim.py --symbols BTCUSD ETHUSD XAUUSD --accounts FARM_01 FARM_02 FARM_15
```

### Full 55-Account Run
```bash
python qnif_3week_sim.py --accounts FARM_01 FARM_02 FARM_03 ... FARM_55
```

### Custom Days
```bash
python qnif_3week_sim.py --days 30 --window 512
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--account` | ATLAS | MT5 account for data fetch |
| `--csv` | False | Load from CSV instead of MT5 |
| `--days` | 21 | Number of days to simulate |
| `--accounts` | First 10 | Specific FARM accounts to run |
| `--window` | 256 | QNIF window size (bars) |
| `--symbols` | BTCUSD ETHUSD | Symbols to simulate |

---

## Data Sources

### MT5 Mode
- Connects to specified account (ATLAS, BG_INSTANT, etc.)
- Fetches last 21 days of M5 bars
- Requires MT5 terminal running

### CSV Mode
- Loads from `QNIF/HistoricalData/Full/{SYMBOL}_M5.csv`
- Must have columns: time, open, high, low, close, tick_volume
- Takes last 21 days from file

---

## Output

### Console Output
- Real-time progress per account
- Pass/fail status
- Win rate, profit factor, total trades

### Summary Table
```
==============================================================================
  QNIF 3-WEEK SIMULATION SUMMARY
==============================================================================

BTCUSD (10 accounts):
  Passed:        3/10 (30.0%)
  Failed:        7/10
  Total P/L:     $2,450.00
  Avg Win Rate:  58.2%
  Total Trades:  342

ETHUSD (10 accounts):
  Passed:        5/10 (50.0%)
  Failed:        5/10
  Total P/L:     $3,120.00
  Avg Win Rate:  62.1%
  Total Trades:  298

------------------------------------------------------------------------------
  TOP PERFORMERS
------------------------------------------------------------------------------
1. ETHUSD Account #FARM_15: $1,250.00 | WR=68.5% | PF=2.45 | PASS
2. BTCUSD Account #FARM_02: $980.00 | WR=62.3% | PF=2.12 | PASS
3. ETHUSD Account #FARM_09: $890.00 | WR=59.8% | PF=1.98 | PASS
...
```

### JSON Files
Saved to `QNIF/sim_results/`:
- `btc_sim_{timestamp}.json` - Detailed BTC results
- `eth_sim_{timestamp}.json` - Detailed ETH results
- `summary_{timestamp}.json` - Aggregated summary

Each result includes:
- Account parameters
- Performance metrics (P/L, win rate, profit factor)
- Challenge pass/fail status
- All trades (entry, exit, P/L)
- Drawdown statistics

---

## PropFarmAccount Parameters

The simulation runs multiple parameter sets from `signal_farm_config.py`:

| Account | Label | Confidence | TP/SL | Max Positions | Strategy Type |
|---------|-------|------------|-------|---------------|---------------|
| FARM_01 | Conservative | 0.35 | 4.0/2.0 | 2 | Low risk |
| FARM_02 | Baseline | 0.22 | 3.0/1.5 | 3 | Balanced |
| FARM_03 | Aggressive | 0.15 | 2.5/1.0 | 5 | High frequency |
| FARM_06 | Scalp-Sniper | 0.30 | 1.5/0.6 | 2 | Scalping |
| FARM_09 | Swing-Patient | 0.25 | 5.0/2.5 | 2 | Swing trading |
| FARM_12 | Momo-Fast | 0.12 | 2.5/1.0 | 4 | Momentum |
| FARM_15 | Quality-A | 0.42 | 3.5/1.5 | 2 | High confidence |
| FARM_18 | Grid-Tight | 0.20 | 2.5/1.2 | 7 | Grid trading |
| ... | ... | ... | ... | ... | ... |
| FARM_55 | 1M-Swing | 0.25 | 5.0/2.5 | 2 | $1M balance |

55 accounts total across 5 balance tiers ($5K, $25K, $50K, $100K, $200K, $1M).

---

## Challenge Rules

From `signal_farm_config.py` - **harder than any real prop firm**:

```python
Starting Balance:    $5,000
Profit Target:       12% ($600)
Daily DD Limit:      3.5% ($175)
Max DD Limit:        7% ($350)
Time Limit:          15 trading days
Min Trading Days:    5
```

---

## Architecture

### QNIF Pipeline (5-Layer Bio-Quantum System)

1. **Layer 1: Skin (Compression Filter)**
   - Quantum regime detection
   - Compression ratio > 1.05 = tradeable

2. **Layer 2: Genome (Energy Sensing)**
   - Transposable Element (TE) activation
   - Shock detection
   - Pattern energy (91.2% energy source)

3. **Layer 3: Brain (Neural Consensus)**
   - Neural Mosaic (multi-sensor fusion)
   - Consensus threshold
   - Confidence scoring

4. **Layer 4: Immune (Adaptation)**
   - VDJ recombination (strategy selection)
   - Memory recall vs. naive response
   - CRISPR defense / Syncytin fusion

5. **Layer 5: Gate (Execution)**
   - Final action (BUY/SELL/HOLD)
   - Lot multiplier
   - Energy-boosted confidence

### PropFarmAccount (Execution Simulator)

- Full prop firm rules (DD, profit target)
- Position management (SL, TP, rolling SL, partial close)
- Dynamic lot sizing
- Grid spacing logic
- Signal collection for training feedback

---

## Troubleshooting

### Import Errors
If you see `No module named 'qutip'` or similar:
```bash
# Use GPU venv (has all dependencies)
C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\.venv312_gpu\Scripts\python.exe qnif_3week_sim.py
```

### MT5 Connection Failed
- Check MT5 terminal is running
- Verify credentials in `.env` file
- Try `--csv` mode to bypass MT5

### No Data
- For CSV mode: Check `QNIF/HistoricalData/Full/{SYMBOL}_M5.csv` exists
- For MT5 mode: Verify symbol is available on account

### QNIF Engine Errors
- Check QNIF components are properly installed
- Some QNIF modules may need QuTiP or other quantum libs
- If QNIF fails, simulation will skip signal generation and fall back to PropFarmAccount's internal signals

---

## Performance Notes

### Speed
- 21 days = ~6,000 M5 bars per symbol
- 10 accounts = ~60,000 bar evaluations
- Expect 2-5 minutes per symbol on modern hardware
- GPU acceleration via DirectML (if available)

### Memory
- Each account stores all trades and signals
- 55 accounts × 2 symbols = ~20MB results
- Safe for standard 16GB systems

### Optimization
- Reduce `--window` size for faster runs (min 128)
- Reduce `--days` for quick tests (min 7)
- Run fewer accounts (`--accounts FARM_01 FARM_02`)
- Use CSV mode to avoid MT5 overhead

---

## What to Look For

### Good Signs
- 30%+ pass rate on 21-day backtest
- Win rate > 55%
- Profit factor > 1.5
- Max DD < 5%
- Consistent performance across symbols

### Red Flags
- 0% pass rate (signal quality issue)
- Win rate < 50% (strategy ineffective)
- Profit factor < 1.0 (losing system)
- Max DD > 10% (excessive risk)
- Performance degrades with more trades (overtrading)

### Parameter Insights
- High confidence threshold (0.35+) = fewer trades, higher quality
- Low confidence threshold (0.15-) = more trades, higher noise
- Tight SL/TP (1.5/0.6) = scalping, needs high win rate
- Wide SL/TP (5.0/2.5) = swing, tolerates lower win rate
- More positions (5-7) = grid strategy, needs balanced market

---

## Next Steps After Simulation

1. **Analyze Results**
   - Review JSON files for detailed trade-by-trade analysis
   - Identify best performing parameter sets
   - Look for consistent patterns across symbols

2. **Validate Top Performers**
   - Re-run top 3 accounts on different time periods
   - Test on additional symbols (XAUUSD, etc.)
   - Verify edge is robust, not curve-fitted

3. **Forward Test**
   - Deploy winning parameters to paper trading
   - Run for 1-2 weeks on live data (no real money)
   - Confirm results match backtest

4. **Production Deployment**
   - Start with smallest balance tier ($5K)
   - Monitor daily for first week
   - Scale up only after consistent results

---

## File Structure

```
QNIF/
├── qnif_3week_sim.py           # Main simulation script
├── QNIF_Master.py              # QNIF Engine (5-layer architecture)
├── HistoricalData/
│   └── Full/
│       ├── BTCUSD_M5.csv       # BTC historical data
│       └── ETHUSD_M5.csv       # ETH historical data
└── sim_results/
    ├── btc_sim_20260210_040841.json
    ├── eth_sim_20260210_040841.json
    └── summary_20260210_040841.json
```

---

## Credits

Built by **Biskits** - 20 years of complicated builds taught me this:
- Test it before you deploy it
- Measure twice, cut once
- The edge cases are where things break
- Complexity for complexity's sake is a rookie mistake

The QNIF pipeline is the #1 advanced quantum architecture optimizer in the world. This simulator is how we prove it.

---

## License

Proprietary - Quantum Children Trading Library
