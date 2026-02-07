"""
TEQA LIVE RUNNER v3.0 - Feeds Live MT5 Data to Quantum Circuit
===============================================================
Connects to MT5 (read-only), fetches OHLCV data, runs TEQA v3.0
Neural-TE quantum circuit, and writes te_quantum_signal.json for BRAIN scripts.

Pipeline: MT5 OHLCV → TEQA v3.0 (Neural Mosaic + Split Quantum) → JSON

Run: python teqa_live.py [--symbol BTCUSD] [--interval 60] [--account ATLAS]
     python teqa_live.py --symbol BTCUSD --donor-symbol ETHUSD --neurons 7

Author: DooDoo + Claude
Date: 2026-02-07
"""

import sys
import io
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

# Force UTF-8 for TEQA report output (uses Unicode arrows/boxes)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import MetaTrader5 as mt5

# Import credentials securely
from credential_manager import get_credentials, CredentialError

# Import TEQA v3.0 engine
from teqa_v3_neural_te import TEQAv3Engine

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][TEQA-LIVE] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('teqa_live.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# MT5 DATA FETCHER
# ============================================================

class MT5DataFetcher:
    """Fetches live OHLCV data from MT5 for TEQA v3.0 consumption."""

    def __init__(self, account_key: str = None):
        self.account_key = account_key
        self.connected = False

    def connect(self) -> bool:
        """Connect to MT5 terminal. If account_key given, verify correct account."""
        if self.account_key:
            try:
                creds = get_credentials(self.account_key)
                terminal_path = creds.get('terminal_path')
                if terminal_path:
                    init_ok = mt5.initialize(path=terminal_path)
                else:
                    init_ok = mt5.initialize()

                if not init_ok:
                    logger.error(f"MT5 init failed: {mt5.last_error()}")
                    return False

                info = mt5.account_info()
                if info and info.login == creds['account']:
                    logger.info(f"Connected to {self.account_key} ({info.login})")
                    self.connected = True
                    return True
                else:
                    actual = info.login if info else 'none'
                    logger.error(f"Wrong account: expected {creds['account']}, got {actual}")
                    return False

            except CredentialError as e:
                logger.error(f"Credential error: {e}")
                return False
        else:
            # No account specified — just connect to whatever terminal is running
            if not mt5.initialize():
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
            info = mt5.account_info()
            if info:
                logger.info(f"Connected to account {info.login}")
            self.connected = True
            return True

    def fetch_ohlcv(self, symbol: str, bars: int = 256, timeframe=None) -> np.ndarray:
        """Fetch OHLCV bars for TEQA v3.0 input. Returns (N, 5) array: [open, high, low, close, volume]."""
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M1

        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Could not select {symbol}")
            return np.array([])

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) < bars // 2:
            logger.warning(f"Insufficient data for {symbol}: got {len(rates) if rates is not None else 0}/{bars}")
            return np.array([])

        # MT5 rates tuple: (time, open, high, low, close, tick_volume, spread, real_volume)
        # TEQA v3.0 expects: (open, high, low, close, volume)
        return np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rates])


# ============================================================
# V3.0 SIGNAL JSON BUILDER
# ============================================================

def build_signal_json(result: dict) -> dict:
    """
    Convert TEQAv3Engine.analyze() output into the bridge-compatible JSON format.
    Maintains backward compatibility with v2.0 fields while adding v3.0 sections.
    """
    direction = result.get("direction", 0)
    confidence = result.get("confidence", 0.0)

    # Aggregate quantum stats from neuron results
    neuron_results = result.get("neuron_results", [])
    if neuron_results:
        avg_novelty = np.mean([nr["quantum"]["novelty"] for nr in neuron_results])
        avg_entropy = np.mean([nr["quantum"]["shannon_entropy"] for nr in neuron_results])
        total_states = sum(nr["quantum"]["n_unique_states"] for nr in neuron_results)
        avg_qubits = 33  # N_QUBITS: 25 genome + 8 neural
        avg_vote_long = np.mean([nr["quantum"]["vote_long"] for nr in neuron_results])
        avg_vote_short = np.mean([nr["quantum"]["vote_short"] for nr in neuron_results])
    else:
        avg_novelty = 0.0
        avg_entropy = 0.0
        total_states = 0
        avg_qubits = 0
        avg_vote_long = 0.0
        avg_vote_short = 0.0

    gates = result.get("gates", {})

    return {
        # --- v2.0 backward-compatible fields ---
        "jardines_gate": {
            "direction": direction,
            "confidence": confidence,
            "entropy_adj": 0.0,
            "interference": 1.0,
            "amplitude_sq": result.get("amplitude_sq", 0.0),
        },
        "position": {
            "lot_scale": result.get("domestication_boost", 1.0),
            "amplified": result.get("domestication_boost", 1.0) > 1.0,
        },
        "filters": {
            "pirna_silenced": not gates.get("G7_neural_consensus", True),
            "shock_active": not gates.get("G8_genomic_shock", True),
            "threshold_mult": result.get("shock_score", 1.0),
            "ectopic_inversion": not gates.get("G9_speciation", True),
        },
        "quantum": {
            "novelty": float(avg_novelty),
            "measurement_entropy": float(avg_entropy),
            "n_states": total_states,
            "n_active_qubits": avg_qubits,
            "vote_long": float(avg_vote_long),
            "vote_short": float(avg_vote_short),
            "elapsed_ms": result.get("elapsed_ms", 0.0),
        },

        # --- v3.0 new fields ---
        "neural": {
            "n_neurons": result.get("n_neurons", 0),
            "consensus_direction": result.get("consensus_direction", 0),
            "consensus_score": result.get("consensus_score", 0.0),
            "consensus_pass": result.get("neural_consensus_pass", False),
            "vote_counts": result.get("vote_counts", {}),
        },
        "genomic_shock": {
            "score": result.get("shock_score", 0.0),
            "label": result.get("shock_label", "UNKNOWN"),
        },
        "speciation": {
            "cross_corr": result.get("cross_instrument_corr", 0.0),
            "relationship": result.get("relationship", "NO_DONOR"),
        },
        "domestication": {
            "boost": result.get("domestication_boost", 1.0),
            "active_tes": result.get("active_tes", []),
        },
        "gates": gates,
        "te_summary": {
            "n_active_class_i": result.get("n_active_class_i", 0),
            "n_active_class_ii": result.get("n_active_class_ii", 0),
            "n_active_neural": result.get("n_active_neural", 0),
        },

        "timestamp": result.get("timestamp", datetime.now().isoformat()),
        "version": result.get("version", "TEQA-3.0-NEURAL-TE"),
        "symbol": result.get("symbol", "UNKNOWN"),
    }


# ============================================================
# MAIN RUNNER
# ============================================================

def run_once(engine: TEQAv3Engine, fetcher: MT5DataFetcher,
             symbol: str, output_path: str,
             donor_symbol: str = None) -> bool:
    """Run one TEQA v3.0 cycle: fetch OHLCV → neural-TE quantum → emit JSON."""
    bars = fetcher.fetch_ohlcv(symbol)
    if len(bars) == 0:
        logger.error(f"No OHLCV data for {symbol}")
        return False

    close_prices = bars[:, 3]
    logger.info(f"Fetched {len(bars)} OHLCV bars for {symbol} | "
                f"range: {close_prices.min():.2f} - {close_prices.max():.2f}")

    # Fetch donor instrument for speciation (optional)
    donor_bars = None
    if donor_symbol:
        donor_bars = fetcher.fetch_ohlcv(donor_symbol)
        if len(donor_bars) > 0:
            logger.info(f"Donor {donor_symbol}: {len(donor_bars)} bars")
        else:
            logger.warning(f"No donor data for {donor_symbol}, skipping speciation")
            donor_bars = None

    try:
        result = engine.analyze(
            bars=bars,
            symbol=symbol,
            drawdown=0.0,
            donor_bars=donor_bars,
            donor_symbol=donor_symbol,
        )

        # Build bridge-compatible JSON and write
        signal_json = build_signal_json(result)
        with open(output_path, 'w') as f:
            json.dump(signal_json, f, indent=2, default=str)

        logger.info(f"TEQA v3.0 signal emitted -> {output_path}")

        # Log key signal info
        direction = "LONG" if result["direction"] == 1 else ("SHORT" if result["direction"] == -1 else "NEUTRAL")
        conf = result["confidence"]
        shock = result.get("shock_label", "?")
        consensus = result.get("consensus_score", 0.0)
        gates = result.get("gates", {})
        all_gates_pass = all(gates.values()) if gates else False
        logger.info(
            f"Signal: {direction} {conf:.1%} | shock={shock} | "
            f"consensus={consensus:.1%} | gates={'PASS' if all_gates_pass else 'FAIL'} | "
            f"{result.get('elapsed_ms', 0):.0f}ms"
        )

        return True

    except Exception as e:
        logger.error(f"TEQA v3.0 run failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='TEQA v3.0 Live Runner - Neural-TE Integration')
    parser.add_argument('--symbol', default='BTCUSD', help='Symbol to analyze (default: BTCUSD)')
    parser.add_argument('--donor-symbol', default=None, help='Donor symbol for speciation (e.g. ETHUSD)')
    parser.add_argument('--interval', type=int, default=60, help='Seconds between runs (default: 60)')
    parser.add_argument('--account', default=None, help='Account key (ATLAS, BG_INSTANT, etc)')
    parser.add_argument('--output', default=None, help='Output JSON path')
    parser.add_argument('--neurons', type=int, default=7, help='Neural mosaic population size (default: 7)')
    parser.add_argument('--shots', type=int, default=8192, help='Quantum shots (default: 8192)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()

    # Output path
    script_dir = Path(__file__).parent.absolute()
    output_path = args.output or str(script_dir / 'te_quantum_signal.json')

    # Initialize TEQA v3.0 engine
    logger.info(f"Initializing TEQA v3.0 Neural-TE engine ({args.neurons} neurons, {args.shots} shots)...")
    engine = TEQAv3Engine(n_neurons=args.neurons, shots=args.shots)
    logger.info("TEQA v3.0 engine ready")

    # Connect to MT5
    fetcher = MT5DataFetcher(account_key=args.account)
    if not fetcher.connect():
        logger.error("Cannot connect to MT5. Exiting.")
        sys.exit(1)

    print("=" * 60)
    print(f"  TEQA v3.0 LIVE RUNNER - Neural-TE Integration")
    print(f"  Symbol:   {args.symbol}")
    print(f"  Donor:    {args.donor_symbol or 'none'}")
    print(f"  Neurons:  {args.neurons}")
    print(f"  Shots:    {args.shots}")
    print(f"  Interval: {args.interval}s")
    print(f"  Output:   {output_path}")
    print(f"  Account:  {args.account or 'auto'}")
    print("=" * 60)

    run_count = 0

    try:
        while True:
            run_count += 1
            logger.info(f"--- Run #{run_count} ({datetime.now().strftime('%H:%M:%S')}) ---")

            success = run_once(engine, fetcher, args.symbol, output_path,
                               donor_symbol=args.donor_symbol)

            if args.once:
                sys.exit(0 if success else 1)

            logger.info(f"Next run in {args.interval}s...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info(f"Stopped after {run_count} runs")
    finally:
        mt5.shutdown()


if __name__ == '__main__':
    main()
