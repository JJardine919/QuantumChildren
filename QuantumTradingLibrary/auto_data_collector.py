"""
AUTO DATA COLLECTOR - Automated Market Data Fetching
=====================================================
Part of the auto training loop. Fetches latest OHLCV data from MT5
for all configured symbols and saves to quantum_data/ for downstream
training and simulation.

This script ONLY reads data. It NEVER places or modifies trades.

Usage:
    python auto_data_collector.py --symbols BTCUSD ETHUSD XAUUSD
    python auto_data_collector.py  (uses defaults from MASTER_CONFIG)

Authors: DooDoo + Claude
Date: 2026-02-12
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][DATA_COLLECT] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("DATA_COLLECT")


# MT5 timeframes to collect
COLLECTION_TIMEFRAMES = {
    "M5":  {"bars": 30000, "desc": "5-minute bars (3.5 months)"},
    "M15": {"bars": 15000, "desc": "15-minute bars (5 months)"},
    "H1":  {"bars": 5000,  "desc": "1-hour bars (7 months)"},
    "H4":  {"bars": 2500,  "desc": "4-hour bars (16 months)"},
}

DATA_DIR = SCRIPT_DIR / "quantum_data"
DATA_DIR.mkdir(exist_ok=True)


def collect_for_symbol(mt5, symbol: str) -> dict:
    """
    Fetch data for one symbol across all configured timeframes.

    Returns dict of {timeframe: DataFrame} or empty dict on failure.
    """
    results = {}
    tf_map = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
    }

    for tf_name, spec in COLLECTION_TIMEFRAMES.items():
        tf_const = tf_map.get(tf_name)
        if tf_const is None:
            continue

        try:
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, spec["bars"])
            if rates is None or len(rates) == 0:
                log.warning(f"  {symbol} {tf_name}: No data returned")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")

            # Save to CSV
            filename = f"{symbol}_{tf_name}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = DATA_DIR / filename
            df.to_csv(str(filepath), index=False)

            results[tf_name] = {
                "bars": len(df),
                "start": str(df["time"].iloc[0]),
                "end": str(df["time"].iloc[-1]),
                "file": str(filepath),
            }

            log.info(f"  {symbol} {tf_name}: {len(df):,} bars ({df['time'].iloc[0]} -> {df['time'].iloc[-1]})")

        except Exception as e:
            log.error(f"  {symbol} {tf_name}: Error: {e}")

    return results


def collect_entropy_snapshots(mt5, symbols: list):
    """Collect entropy data locally for training use."""
    try:
        from entropy_collector import collect_entropy_snapshot
    except ImportError:
        log.debug("entropy_collector not available, skipping entropy snapshots")
        return

    for symbol in symbols:
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 256)
            if rates is None or len(rates) < 100:
                continue

            df = pd.DataFrame(rates)
            closes = df["close"].values

            # Simple entropy calculation
            returns = np.diff(np.log(closes + 1e-10))
            hist, _ = np.histogram(returns, bins=20, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            dominant = float(np.max(hist))
            significant = int(np.sum(hist > 0.01))
            variance = float(np.var(returns))

            collect_entropy_snapshot(
                symbol=symbol,
                timeframe="M5",
                entropy=entropy,
                dominant=dominant,
                significant=significant,
                variance=variance,
                price=float(closes[-1]),
            )
            log.info(f"  {symbol} entropy snapshot collected (entropy={entropy:.4f})")

        except Exception as e:
            log.debug(f"  {symbol} entropy snapshot error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Auto Data Collector")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to collect (default: from MASTER_CONFIG)")
    args = parser.parse_args()

    # Get symbols
    symbols = args.symbols
    if symbols is None:
        try:
            with open(SCRIPT_DIR / "MASTER_CONFIG.json") as f:
                mc = json.load(f)
            symbols = list(mc.get("ASSET_CLASSES", {}).keys())[:3]
        except Exception:
            symbols = ["BTCUSD", "ETHUSD", "XAUUSD"]

    log.info(f"Collecting data for: {symbols}")

    # Initialize MT5
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.error("MetaTrader5 package not installed!")
        sys.exit(1)

    if not mt5.initialize():
        log.error("MT5 initialization failed. Is MetaTrader 5 running?")
        sys.exit(1)

    try:
        all_results = {}
        for symbol in symbols:
            log.info(f"--- {symbol} ---")
            results = collect_for_symbol(mt5, symbol)
            all_results[symbol] = results

        # Collect entropy snapshots
        collect_entropy_snapshots(mt5, symbols)

        # Save collection summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "results": all_results,
        }
        summary_path = DATA_DIR / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(str(summary_path), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Count total bars collected
        total_bars = sum(
            info["bars"]
            for sym_results in all_results.values()
            for info in sym_results.values()
        )
        log.info(f"Collection complete: {total_bars:,} total bars across {len(symbols)} symbols")

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
