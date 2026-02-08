"""
AOI OBSERVER - Live Market Data Observation & Recording
========================================================
Connects to MT5, pulls live bars for all configured symbols,
runs the full Artificial Organism Intelligence pipeline,
and records every observation to JSONL.

READ-ONLY MODE: No trades are executed. This is for testing
and recording the biological algorithm outputs on live data.

Run:
    .venv312_gpu\\Scripts\\python.exe AOI_OBSERVER.py
    .venv312_gpu\\Scripts\\python.exe AOI_OBSERVER.py --account ATLAS
    .venv312_gpu\\Scripts\\python.exe AOI_OBSERVER.py --interval 15
    .venv312_gpu\\Scripts\\python.exe AOI_OBSERVER.py --cycles 100

Author: James Jardine / Quantum Children
Date: 2026-02-08
"""

import sys
import os
import io
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import MetaTrader5 as mt5

# AOI Pipeline
from aoi_pipeline import AOIPipeline

# TEQA core for TE activations
try:
    from teqa_v3_neural_te import TEActivationEngine, TEDomesticationTracker
    TEQA_AVAILABLE = True
except ImportError:
    TEQA_AVAILABLE = False

# Config
from config_loader import CONFIDENCE_THRESHOLD, CHECK_INTERVAL_SECONDS

# Credentials
from credential_manager import get_credentials, CredentialError

# Configure logging
LOG_DIR = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][AOI] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / 'aoi_observer.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# JSONL output file
OBSERVATION_FILE = LOG_DIR / "aoi_observations.jsonl"

# Account configs (from MASTER_CONFIG.json)
ACCOUNT_SYMBOLS = {
    "ATLAS":        {"account": 212000584, "symbols": ["BTCUSD", "ETHUSD"]},
    "BG_INSTANT":   {"account": 366604,    "symbols": ["BTCUSD"]},
    "BG_CHALLENGE":  {"account": 365060,    "symbols": ["BTCUSD"]},
    "GL_1":         {"account": 113326,    "symbols": ["BTCUSD", "ETHUSD"]},
    "GL_2":         {"account": 113328,    "symbols": ["BTCUSD", "ETHUSD"]},
    "GL_3":         {"account": 107245,    "symbols": ["BTCUSD", "ETHUSD"]},
    "FTMO":         {"account": 1521063483, "symbols": ["BTCUSD", "XAUUSD", "ETHUSD"]},
}

BARS_FOR_ANALYSIS = 256


def connect_mt5(account_key: str) -> bool:
    """Connect to an MT5 account."""
    try:
        creds = get_credentials(account_key)
    except CredentialError as e:
        log.error(f"No credentials for {account_key}: {e}")
        return False

    terminal_path = creds.get("terminal_path")
    if terminal_path:
        ok = mt5.initialize(path=terminal_path)
    else:
        ok = mt5.initialize()

    if not ok:
        log.error(f"MT5 init failed for {account_key}: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        log.error(f"Could not get account info for {account_key}")
        return False

    expected = ACCOUNT_SYMBOLS.get(account_key, {}).get("account", 0)
    if info.login != expected:
        log.warning(f"{account_key}: Expected account {expected}, got {info.login}")

    log.info(f"Connected: {account_key} (#{info.login}) Balance: ${info.balance:,.2f}")
    return True


def get_bars(symbol: str, n_bars: int = BARS_FOR_ANALYSIS) -> Optional[np.ndarray]:
    """Pull M1 bars from MT5 as numpy array [open, high, low, close, volume]."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n_bars)
    if rates is None or len(rates) < 50:
        log.warning(f"Insufficient bars for {symbol}: got {len(rates) if rates is not None else 0}")
        return None

    # Convert to numpy: [open, high, low, close, tick_volume]
    bars = np.column_stack([
        [r[1] for r in rates],  # open
        [r[2] for r in rates],  # high
        [r[3] for r in rates],  # low
        [r[4] for r in rates],  # close
        [r[5] for r in rates],  # tick_volume
    ])
    return bars


def get_price_info(symbol: str) -> Dict:
    """Get current price info for a symbol."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"bid": 0, "ask": 0, "spread": 0}
    info = mt5.symbol_info(symbol)
    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "spread": info.spread if info else 0,
    }


def write_observation(obs: dict):
    """Append an observation to the JSONL file."""
    with open(OBSERVATION_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obs, default=str) + '\n')


def print_dashboard(cycle: int, account: str, results: Dict[str, dict]):
    """Print a concise dashboard to console."""
    print(f"\n{'=' * 70}")
    print(f"  AOI OBSERVER | {account} | Cycle #{cycle} | {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 70}")

    for symbol, data in results.items():
        aoi = data.get("aoi", {})
        price = data.get("price", {})

        dir_str = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(
            aoi.get("final_direction", 0), "?"
        )
        orig_dir_str = {1: "LONG", -1: "SHORT", 0: "NEUTRAL"}.get(
            aoi.get("original_direction", 0), "?"
        )

        gate_str = "PASS" if aoi.get("final_gate_pass", True) else "BLOCKED"
        n_ran = aoi.get("n_algorithms_ran", 0)
        n_avail = aoi.get("n_algorithms_available", 0)
        elapsed = aoi.get("total_elapsed_ms", 0)

        print(f"\n  {symbol} | Bid: {price.get('bid', 0):.2f} | Spread: {price.get('spread', 0)}")
        print(f"  Direction: {orig_dir_str} -> {dir_str} | "
              f"Confidence: {aoi.get('original_confidence', 0):.3f} -> "
              f"{aoi.get('final_confidence', 0):.3f}")
        print(f"  Gate: {gate_str} | Active TEs: {len(aoi.get('active_tes', []))} | "
              f"Boost: {aoi.get('domestication_boost', 1.0):.2f}")
        print(f"  Algorithms: {n_ran}/{n_avail} ran | {elapsed:.1f}ms")

        # Algorithm details
        algos = aoi.get("algorithms", {})
        for name, a in algos.items():
            if not a.get("ran", False):
                status = "  OFF"
            elif a.get("error"):
                status = f"  ERR: {a['error'][:40]}"
            elif not a.get("gate_pass", True):
                status = "  BLOCKED"
            else:
                mod = a.get("confidence_modifier", 1.0)
                status = f"  x{mod:.3f}"
            print(f"    [{name:>22}] {status}")


def simulate_direction(bars: np.ndarray) -> tuple:
    """
    Simple momentum-based direction for observation purposes.
    Uses 8-period vs 21-period EMA crossover on close prices.
    Returns (direction, confidence).
    """
    close = bars[:, 3]

    # EMA-8 vs EMA-21
    def ema(data, period):
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    ema8 = ema(close, 8)
    ema21 = ema(close, 21)

    diff = ema8[-1] - ema21[-1]
    norm_diff = diff / (ema21[-1] + 1e-10)

    # Direction
    if norm_diff > 0.0001:
        direction = 1
    elif norm_diff < -0.0001:
        direction = -1
    else:
        direction = 0

    # Confidence from magnitude
    confidence = min(1.0, abs(norm_diff) * 100)

    return direction, confidence


def run_observer(account_key: str, interval: int, max_cycles: int = 0):
    """Main observation loop."""

    # Connect to MT5
    if not connect_mt5(account_key):
        log.error(f"Cannot connect to {account_key}. Make sure MT5 terminal is running.")
        sys.exit(1)

    symbols = ACCOUNT_SYMBOLS.get(account_key, {}).get("symbols", ["BTCUSD"])

    # Initialize AOI pipeline
    log.info("Initializing AOI Pipeline...")
    pipeline = AOIPipeline(symbols=symbols)
    status = pipeline.get_status()
    log.info(f"Pipeline status: {status['n_available']}/8 algorithms available")
    for name, avail in status['algorithms'].items():
        log.info(f"  {name}: {'OK' if avail else 'MISSING'}")

    # Initialize TE activation engine
    te_engine = None
    domestication_tracker = None
    if TEQA_AVAILABLE:
        te_engine = TEActivationEngine()
        domestication_tracker = TEDomesticationTracker()
        log.info("TEQA TE Activation Engine: OK (33 qubits)")
    else:
        log.warning("TEQA TE Activation Engine: NOT AVAILABLE")

    cycle = 0
    log.info(f"Starting observation loop: {account_key} | Symbols: {symbols} | Interval: {interval}s")
    log.info(f"Recording to: {OBSERVATION_FILE}")

    try:
        while True:
            cycle += 1
            if max_cycles > 0 and cycle > max_cycles:
                log.info(f"Reached max cycles ({max_cycles}), stopping.")
                break

            cycle_results = {}

            for symbol in symbols:
                # Pull bars
                bars = get_bars(symbol)
                if bars is None:
                    cycle_results[symbol] = {"error": "no bars"}
                    continue

                price_info = get_price_info(symbol)

                # Get TE activations
                active_tes = []
                boost = 1.0
                te_activations = []

                if te_engine is not None:
                    te_activations = te_engine.compute_all_activations(bars)
                    active_tes = [a["te"] for a in te_activations if a["strength"] > 0.5]
                    if domestication_tracker and active_tes:
                        boost = domestication_tracker.get_boost(active_tes)

                # Get a directional signal (momentum-based for observation)
                direction, confidence = simulate_direction(bars)

                # Run full AOI pipeline
                aoi_result = pipeline.process(
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    bars=bars,
                    active_tes=active_tes,
                    domestication_boost=boost,
                    strategy_id=f"{account_key}_{symbol}",
                    timeframe="M1",
                )

                aoi_dict = aoi_result.to_dict()

                # Build observation record
                observation = {
                    "cycle": cycle,
                    "account": account_key,
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "price": price_info,
                    "n_bars": len(bars),
                    "n_active_tes": len(active_tes),
                    "active_tes": active_tes[:10],  # Cap for log size
                    "domestication_boost": boost,
                    "momentum_direction": direction,
                    "momentum_confidence": round(confidence, 4),
                    "aoi": aoi_dict,
                    "te_activations_summary": {
                        "total": len(te_activations),
                        "active": len(active_tes),
                        "top_5": sorted(
                            [{"te": a["te"], "strength": round(a["strength"], 3),
                              "direction": a["direction"]}
                             for a in te_activations if a["strength"] > 0.3],
                            key=lambda x: x["strength"],
                            reverse=True,
                        )[:5],
                    },
                }

                # Write to JSONL
                write_observation(observation)

                cycle_results[symbol] = {
                    "aoi": aoi_dict,
                    "price": price_info,
                }

            # Print dashboard
            print_dashboard(cycle, account_key, cycle_results)

            # Show JSONL file size
            if OBSERVATION_FILE.exists():
                size_kb = OBSERVATION_FILE.stat().st_size / 1024
                print(f"\n  Observations: {OBSERVATION_FILE.name} ({size_kb:.1f} KB)")

            # Wait
            if max_cycles == 0 or cycle < max_cycles:
                print(f"\n  Next cycle in {interval}s... (Ctrl+C to stop)")
                time.sleep(interval)

    except KeyboardInterrupt:
        log.info("Observer stopped by user.")
    finally:
        mt5.shutdown()
        log.info(f"Completed {cycle} observation cycles.")
        if OBSERVATION_FILE.exists():
            size_kb = OBSERVATION_FILE.stat().st_size / 1024
            log.info(f"Recorded to: {OBSERVATION_FILE} ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="AOI Observer - Live Market Data Observation")
    parser.add_argument("--account", type=str, default="ATLAS",
                        choices=list(ACCOUNT_SYMBOLS.keys()),
                        help="Account to observe (default: ATLAS)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between observation cycles (default: 30)")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Max cycles (0 = infinite, default: 0)")

    args = parser.parse_args()

    print("=" * 70)
    print("  ARTIFICIAL ORGANISM INTELLIGENCE - OBSERVER")
    print("  Quantum Children AOI Pipeline v1.0")
    print("  READ-ONLY MODE: No trades will be executed")
    print("=" * 70)
    print(f"  Account:  {args.account}")
    print(f"  Interval: {args.interval}s")
    print(f"  Output:   {OBSERVATION_FILE}")
    print("=" * 70)

    run_observer(args.account, args.interval, args.cycles)


if __name__ == "__main__":
    main()
