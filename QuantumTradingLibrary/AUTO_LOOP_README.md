# Quantum Children - Autonomous Training Loop

## RULE #1: DO NOT TOUCH THE LIVE TRADING SYSTEM

The BRAIN self-regulates. We never modify it. Only shutdown if runaway.
This is non-negotiable.

---

## What Is This?

A self-contained automated pipeline that runs 24/7 on the VPS alongside
the live BRAIN trading scripts. It handles all training, data collection,
expert evolution, and domestication learning -- without human intervention.

The BRAIN scripts handle live trading. This loop handles everything that
makes the BRAIN smarter over time.

## Architecture

```
VPS (Windows, 24/7)
|
+-- LIVE TRADING (do not touch)
|   +-- BRAIN_ATLAS.py          (trades ETHUSD)
|   +-- BRAIN_BG_INSTANT.py     (trades BTCUSD)
|   +-- BRAIN_BG_CHALLENGE.py   (trades BTCUSD)
|   +-- BRAIN_FTMO.py           (trades BTC/ETH/XAU)
|   +-- STOPLOSS_WATCHDOG.py    (safety net)
|   +-- quantum_server.py       (port 8000)
|
+-- AUTO TRAINING LOOP (this system)
    +-- auto_training_loop.py   (orchestrator -- single process)
    |   |
    |   +-- Stage 1: Health Check           (every 15 min)
    |   +-- Stage 2: Data Collection        (every 4 hours)
    |   +-- Stage 3: TE Feeding             (every 30 min)
    |   +-- Stage 4: Signal Farm Training   (every 6 hours)
    |   +-- Stage 5: LSTM Retraining        (every 12 hours)
    |   +-- Stage 6: QNIF Simulation        (every 24 hours)
    |   +-- Stage 7: Expert Rotation        (every 24 hours)
    |
    +-- auto_data_collector.py  (fetches OHLCV from MT5)
    +-- auto_te_feeder.py       (feeds trade outcomes to domestication)
    +-- auto_loop_config.json   (all scheduling/safety config)
    +-- auto_loop_state.json    (persistent state -- last run times)
    +-- orchestrator_logs/      (daily log files)
```

## Pipeline Stages

### Stage 1: Health Check (every 15 minutes)
- Verifies BRAIN scripts are running (process detection via psutil)
- Checks expert freshness (age in hours)
- Checks TE domestication database stats
- Checks disk space
- Checks signal farm champion count
- READ-ONLY: never modifies anything

### Stage 2: Data Collection (every 4 hours)
- Fetches OHLCV bars from MT5 for BTCUSD, ETHUSD, XAUUSD
- Timeframes: M5 (30K bars), M15 (15K bars), H1 (5K bars), H4 (2.5K bars)
- Saves to `quantum_data/` as CSV files
- Collects entropy snapshots for pattern analysis
- Uses `auto_data_collector.py`

### Stage 3: TE Domestication Feeding (every 30 minutes)
- Polls MT5 for recently closed trades
- Matches trades to TEQA signals via signal history DB
- Feeds win/loss outcomes to `teqa_domestication.db`
- This is the closed-loop learning mechanism
- Domesticated patterns get boost factors (confidence multiplier)
- Uses `auto_te_feeder.py`

### Stage 4: Signal Farm Training (every 6 hours)
- Runs the 34-round genetic training protocol
- 17 timeframes x 2 cycles per symbol
- GPU-accelerated via DirectML (AMD RX 6800 XT)
- Evolves trading individuals through tournament selection
- Saves champions to `signal_farm_output/`
- Uses `signal_farm_trainer.py`

### Stage 5: LSTM Retraining (every 12 hours)
- Retrains LSTM experts with walk-forward validation
- IMPORTANT: Runs on CPU (DirectML cannot backprop LSTM)
- 80 epochs with early stopping
- Class-weighted loss + label smoothing + confidence thresholding
- Saves best model to `top_50_experts/expert_RETRAINED_best.pth`
- Uses `lstm_retrain_fast.py`

### Stage 6: QNIF Simulation (every 24 hours)
- Runs prop firm comparison simulations
- Baseline (no immune) vs Full QNIF (with CRISPR/VDJ immune filtering)
- Validates current expert performance on recent data
- Uses `QNIF/qnif_1month_comparison.py`

### Stage 7: Expert Rotation (every 24 hours)
- Checks expert ages against LSTM_MAX_AGE_DAYS (7 days)
- Archives expired experts (never deletes, moves to `expert_archive/`)
- Updates `top_50_manifest.json`
- SAFETY: Will not archive if fewer than 5 fresh experts remain
- Prevents the BRAIN from running with no experts

## GPU Resource Sharing

The BRAIN uses GPU for fast inference (small, brief operations).
Training uses GPU for larger batch tensor operations.

Coexistence strategy:
- Auto loop process runs at **BELOW_NORMAL** priority
- Windows scheduler gives BRAIN scripts CPU/GPU priority
- Training yields between symbols (configurable delay)
- GPU pool manager controls concurrent access
- LSTM training stays on CPU entirely (DirectML limitation)

## Files Created

| File | Purpose |
|------|---------|
| `auto_training_loop.py` | Main orchestrator (run this) |
| `auto_data_collector.py` | Market data fetcher |
| `auto_te_feeder.py` | TE domestication feeder |
| `auto_loop_config.json` | All configuration |
| `auto_loop_state.json` | Persistent state (auto-created) |
| `install_auto_loop.py` | Windows Task Scheduler setup |
| `START_AUTO_LOOP.bat` | Manual start (double-click) |

## Quick Start

### 1. Manual start (for testing)

```cmd
cd C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary
.venv312_gpu\Scripts\python.exe auto_training_loop.py --dry-run
.venv312_gpu\Scripts\python.exe auto_training_loop.py --once
.venv312_gpu\Scripts\python.exe auto_training_loop.py
```

Or double-click `START_AUTO_LOOP.bat`.

### 2. Install as Windows service (runs at boot)

```cmd
.venv312_gpu\Scripts\python.exe install_auto_loop.py --install
```

Check status:
```cmd
.venv312_gpu\Scripts\python.exe install_auto_loop.py --status
```

### 3. Monitor

Check last run status:
```cmd
.venv312_gpu\Scripts\python.exe auto_training_loop.py --status
```

View logs:
```cmd
type orchestrator_logs\auto_loop_20260212.log
```

Check TE domestication stats:
```cmd
.venv312_gpu\Scripts\python.exe auto_te_feeder.py --stats
```

## Configuration

Edit `auto_loop_config.json` to change intervals, safety settings, or GPU throttle.

Key settings:
```json
{
    "SCHEDULE": {
        "DATA_COLLECTION_INTERVAL_HOURS": 4,
        "SIGNAL_FARM_INTERVAL_HOURS": 6,
        "LSTM_RETRAIN_INTERVAL_HOURS": 12,
        "EXPERT_ROTATION_INTERVAL_HOURS": 24,
        "TE_FEEDING_INTERVAL_MINUTES": 30,
        "HEALTH_CHECK_INTERVAL_MINUTES": 15
    },
    "GPU_THROTTLE": {
        "TRAINING_PROCESS_PRIORITY": "BELOW_NORMAL",
        "MAX_GPU_MEMORY_PCT": 70,
        "YIELD_TO_BRAIN_SECONDS": 2
    },
    "SAFETY": {
        "NEVER_MODIFY_BRAIN_SCRIPTS": true,
        "NEVER_MODIFY_MASTER_CONFIG": true,
        "NEVER_CLOSE_POSITIONS": true,
        "MAX_CONSECUTIVE_FAILURES": 10
    }
}
```

## Safety Guarantees

1. The orchestrator runs at BELOW_NORMAL priority -- BRAIN always gets CPU first
2. Each task runs in an isolated subprocess -- orchestrator cannot crash from task errors
3. Retry logic with exponential backoff -- transient failures are handled
4. Lockfile prevents duplicate orchestrator instances
5. Expert rotation will NOT archive if too few fresh experts remain
6. Graceful shutdown on Ctrl+C (SIGINT) -- current task completes, then exits
7. All state is persisted to disk -- survives restarts
8. Daily log rotation -- disk space is managed
9. The orchestrator NEVER modifies MASTER_CONFIG.json
10. The orchestrator NEVER places, modifies, or closes trades

## Troubleshooting

### "Another auto_training_loop is already running"
Delete the stale lockfile:
```cmd
del .locks\auto_training_loop.lock
```

### MT5 initialization fails
- Ensure at least one MT5 terminal is running
- The data collector will retry (3 attempts with backoff)

### LSTM retrain takes too long
- Expected: 30-60 minutes for 80 epochs
- Timeout is set to 120 minutes
- Early stopping kicks in after 15 stale epochs

### Expert rotation blocked
- Means fewer than 5 fresh experts exist
- Run LSTM retrain first: `.venv312_gpu\Scripts\python.exe lstm_retrain_fast.py`
- Check status: `.venv312_gpu\Scripts\python.exe auto_training_loop.py --status`

### No TE patterns being fed
- Requires trades to close (SL/TP hit)
- Requires TEQA signals in `teqa_signal_history.db`
- Run stats: `.venv312_gpu\Scripts\python.exe auto_te_feeder.py --stats`
