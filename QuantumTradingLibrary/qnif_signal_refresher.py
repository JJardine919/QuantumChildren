"""
QNIF SIGNAL REFRESHER -- Generates Fresh QNIF Signals from Collected Data
===========================================================================

Reads OHLCV data from quantum_data/ (already collected by auto_data_collector.py)
and runs it through the QNIF 5-layer pipeline to produce fresh signal JSON files.

This script does NOT need MT5. It uses data already on disk.
Solves the MT5 singleton problem: BRAIN_ATLAS owns MT5, we read files.

Usage:
    python qnif_signal_refresher.py                    # One-shot refresh
    python qnif_signal_refresher.py --loop --interval 300  # Refresh every 5 min
    python qnif_signal_refresher.py --symbols BTCUSD ETHUSD  # Specific symbols

Output:
    qnif_signal_BTCUSD.json
    qnif_signal_ETHUSD.json
    qnif_signal_XAUUSD.json
    qnif_signal.json  (copy of most recent)

Authors: Claude (Opus 4.6)
Date:    2026-02-13
"""

import os
import sys
import json
import time
import glob
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "QNIF"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][QNIF_REFRESH] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("QNIF_REFRESH")

# Where auto_data_collector saves CSV files
DATA_DIR = SCRIPT_DIR / "quantum_data"

# Where to write signal files (same dir as BRAIN reads from)
SIGNAL_DIR = SCRIPT_DIR

# Symbols to process
DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD", "XAUUSD"]

# Preferred timeframe for QNIF processing (M15 gives good balance)
PREFERRED_TF = "M15"
FALLBACK_TFS = ["H1", "M5", "H4"]

# Minimum bars needed for QNIF to function
MIN_BARS = 300


def find_latest_csv(symbol: str, timeframe: str = PREFERRED_TF) -> Path:
    """
    Find the most recent CSV file for a symbol/timeframe.
    Looks for pattern: quantum_data/{SYMBOL}_{TF}_{DATE}.csv
    """
    pattern = str(DATA_DIR / f"{symbol}_{timeframe}_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return Path(max(files, key=os.path.getmtime))


def load_bars_from_csv(csv_path: Path) -> np.ndarray:
    """
    Load CSV into the numpy bars format that QNIF_Engine.process_pulse() expects.
    Returns array with columns: [time, open, high, low, close, tick_volume, spread, real_volume]
    """
    df = pd.read_csv(csv_path)

    # Convert time to epoch seconds if it's a datetime string
    if "time" in df.columns and df["time"].dtype == object:
        df["time"] = pd.to_datetime(df["time"]).astype(np.int64) // 10**9

    # QNIF_Master expects bars with at least close prices in column 3
    # Standard MT5 format: time, open, high, low, close, tick_volume, spread, real_volume
    numeric_cols = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    available_cols = [c for c in numeric_cols if c in df.columns]

    if not available_cols or "close" not in available_cols:
        raise ValueError(f"CSV missing required columns. Found: {list(df.columns)}")

    # Build bars array: QNIF_Master.process_pulse accesses column 3 (close)
    # So we need: [open, high, low, close, ...]
    bars = df[available_cols].values.astype(np.float64)
    return bars


def generate_qnif_signal(symbol: str, bars: np.ndarray) -> dict:
    """
    Run bars through the QNIF 5-layer pipeline and produce a signal dict.
    """
    try:
        from QNIF.QNIF_Master import QNIF_Engine
    except ImportError as e:
        log.error(f"Cannot import QNIF_Engine: {e}")
        return None

    try:
        engine = QNIF_Engine(
            memory_db=str(SCRIPT_DIR / "QNIF" / "vdj_memory_cells.db")
        )
    except Exception as e:
        log.error(f"Failed to initialize QNIF_Engine: {e}")
        return None

    try:
        state = engine.process_pulse(symbol, bars[-MIN_BARS:])
    except Exception as e:
        log.error(f"QNIF process_pulse failed for {symbol}: {e}")
        return None

    # Build the signal JSON matching what qnif_bridge.py expects
    signal = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "action": state.final_action,
        "confidence": float(state.final_confidence),
        "lot_multiplier": float(state.lot_multiplier),
        "compression": {
            "ratio": float(state.compression_ratio),
            "regime": state.regime,
            "tradeable": state.is_tradeable,
        },
        "teqa": {
            "active_tes": state.active_tes,
            "shock_level": float(state.shock_level),
            "shock_label": state.shock_label,
            "consensus": float(state.neural_consensus),
        },
        "vdj": {
            "antibody_id": state.selected_antibody.get("antibody_id", ""),
            "generation": state.generation,
            "is_memory": state.is_memory_recall,
        },
    }

    return signal


def write_signal(symbol: str, signal: dict):
    """Write signal to JSON file."""
    # Per-symbol file
    filepath = SIGNAL_DIR / f"qnif_signal_{symbol}.json"
    filepath.write_text(json.dumps(signal, indent=4))
    log.info(f"  Wrote {filepath.name}: {signal['action']} "
             f"conf={signal['confidence']:.4f} regime={signal['compression']['regime']}")

    # Also write generic file (last symbol processed)
    generic = SIGNAL_DIR / "qnif_signal.json"
    generic.write_text(json.dumps(signal, indent=4))


def refresh_all(symbols: list):
    """Refresh QNIF signals for all symbols."""
    log.info(f"Refreshing QNIF signals for: {', '.join(symbols)}")

    for symbol in symbols:
        # Find the best available data file
        csv_path = None
        used_tf = None
        for tf in [PREFERRED_TF] + FALLBACK_TFS:
            csv_path = find_latest_csv(symbol, tf)
            if csv_path is not None:
                used_tf = tf
                break

        if csv_path is None:
            log.warning(f"  {symbol}: No data files found in {DATA_DIR}")
            continue

        # Check freshness of data file
        file_age_hours = (time.time() - csv_path.stat().st_mtime) / 3600
        log.info(f"  {symbol}: Using {csv_path.name} ({used_tf}, {file_age_hours:.1f}h old)")

        if file_age_hours > 24:
            log.warning(f"  {symbol}: Data is {file_age_hours:.0f}h old -- signals may be less accurate")

        # Load bars
        try:
            bars = load_bars_from_csv(csv_path)
        except Exception as e:
            log.error(f"  {symbol}: Failed to load CSV: {e}")
            continue

        if len(bars) < MIN_BARS:
            log.warning(f"  {symbol}: Only {len(bars)} bars (need {MIN_BARS}), using all available")
            if len(bars) < 50:
                log.error(f"  {symbol}: Not enough data ({len(bars)} bars), skipping")
                continue

        # Generate QNIF signal
        signal = generate_qnif_signal(symbol, bars)
        if signal is None:
            log.error(f"  {symbol}: QNIF engine returned no signal")
            continue

        # Write signal file
        write_signal(symbol, signal)

    log.info("QNIF refresh complete.")


def main():
    parser = argparse.ArgumentParser(description="QNIF Signal Refresher")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to refresh")
    parser.add_argument("--loop", action="store_true",
                        help="Run in continuous loop mode")
    parser.add_argument("--interval", type=int, default=300,
                        help="Loop interval in seconds (default: 300 = 5 min)")
    args = parser.parse_args()

    if args.loop:
        log.info(f"Starting QNIF refresh loop (every {args.interval}s)")
        while True:
            try:
                refresh_all(args.symbols)
            except Exception as e:
                log.error(f"Refresh cycle failed: {e}")
            time.sleep(args.interval)
    else:
        refresh_all(args.symbols)


if __name__ == "__main__":
    main()
