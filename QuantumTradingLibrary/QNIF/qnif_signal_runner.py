"""
QNIF SIGNAL RUNNER v1.0
========================
Multi-symbol signal orchestrator for QNIF Engine.

Spawns separate processes for each symbol (BTCUSD, ETHUSD).
Each process maintains its own MT5 connection and generates independent signals.

Run: python qnif_signal_runner.py --account ATLAS --symbols BTCUSD ETHUSD --interval 300

Architecture:
- ONE MT5 connection per runner instance (respects ONE SCRIPT PER ACCOUNT rule)
- Each symbol runs in an isolated process
- Separate signal files: qnif_signal_BTCUSD.json, qnif_signal_ETHUSD.json
- Automatic crash recovery and restart
- Unified logging with per-symbol context

Date: 2026-02-10
"""

import os
import sys
import time
import json
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import MetaTrader5 as mt5

# Add root to path
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

from QNIF.QNIF_Master import QNIF_Engine, BioQuantumState
from credential_manager import get_credentials, CredentialError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(current_dir / 'qnif_signals.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("QNIF-RUNNER")


class SymbolWorker:
    """
    Individual worker process for a single symbol.
    Each worker maintains its own MT5 connection and QNIF engine.
    """

    def __init__(self, symbol: str, account: str, interval: int = 300):
        self.symbol = symbol
        self.account = account
        self.interval = interval
        self.engine = None
        self.logger = logging.getLogger(f"QNIF-{symbol}")
        self.running = False

    def connect_mt5(self) -> bool:
        """Establish MT5 connection for this worker."""
        try:
            creds = get_credentials(self.account)

            # Initialize MT5
            if not mt5.initialize(path=creds.get('terminal_path')):
                self.logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False

            # Login to account
            if not mt5.login(creds['account'], password=creds['password'], server=creds['server']):
                self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

            account_info = mt5.account_info()
            self.logger.info(f"Connected to {self.account} (Account: {account_info.login}, Balance: ${account_info.balance:.2f})")
            return True

        except CredentialError as e:
            self.logger.error(f"Credential error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False

    def fetch_bars(self, count: int = 300) -> np.ndarray:
        """Fetch historical bars for the symbol."""
        try:
            bars = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, count)

            if bars is None or len(bars) == 0:
                self.logger.error(f"Failed to fetch bars for {self.symbol}: {mt5.last_error()}")
                return None

            # Convert to numpy array [open, high, low, close, tick_volume]
            result = np.zeros((len(bars), 5))
            for i, bar in enumerate(bars):
                result[i] = [bar['open'], bar['high'], bar['low'], bar['close'], bar['tick_volume']]

            return result

        except Exception as e:
            self.logger.error(f"Error fetching bars: {e}")
            return None

    def save_signal(self, state: BioQuantumState):
        """Save signal to symbol-specific JSON file."""
        try:
            # Symbol-specific signal file
            signal_file = root_dir / f"qnif_signal_{self.symbol}.json"

            data = {
                "symbol": state.symbol,
                "account": self.account,
                "timestamp": datetime.now().isoformat(),
                "action": state.final_action,
                "confidence": float(state.final_confidence),
                "lot_multiplier": float(state.lot_multiplier),
                "compression": {
                    "ratio": float(state.compression_ratio),
                    "regime": state.regime,
                    "tradeable": state.is_tradeable,
                    "source_tier": int(state.source_tier)
                },
                "teqa": {
                    "active_tes": state.active_tes,
                    "shock_level": float(state.shock_level),
                    "shock_label": state.shock_label,
                    "consensus": float(state.neural_consensus),
                    "pattern_energy": float(state.pattern_energy)
                },
                "vdj": {
                    "antibody_id": state.selected_antibody.get("antibody_id", "NONE"),
                    "generation": int(state.generation),
                    "is_memory": state.is_memory_recall,
                    "crispr_blocked": state.crispr_blocked,
                    "fusion_active": state.fusion_active
                },
                "meta": {
                    "worker_pid": os.getpid(),
                    "interval": self.interval
                }
            }

            with open(signal_file, 'w') as f:
                json.dump(data, f, indent=4)

            self.logger.debug(f"Signal saved: {signal_file.name}")

        except Exception as e:
            self.logger.error(f"Error saving signal: {e}")

    def run_pulse(self):
        """Execute one QNIF pulse."""
        self.logger.info(f"--- PULSE START ---")

        bars = self.fetch_bars()
        if bars is None:
            self.logger.warning("No bars fetched, skipping pulse")
            return

        try:
            state = self.engine.process_pulse(self.symbol, bars)
            self.save_signal(state)

            self.logger.info(
                f"Action: {state.final_action} | "
                f"Confidence: {state.final_confidence:.3f} | "
                f"Compression: {state.compression_ratio:.2f} | "
                f"Regime: {state.regime}"
            )

        except Exception as e:
            self.logger.error(f"Error processing pulse: {e}", exc_info=True)

    def start(self):
        """Main worker loop."""
        self.logger.info(f"Worker starting for {self.symbol} on {self.account} (Interval: {self.interval}s, PID: {os.getpid()})")

        # Connect to MT5
        if not self.connect_mt5():
            self.logger.error("MT5 connection failed. Worker exiting.")
            return

        # Initialize QNIF Engine
        try:
            self.engine = QNIF_Engine()
            self.logger.info("QNIF Engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize QNIF Engine: {e}")
            mt5.shutdown()
            return

        # Main loop
        self.running = True
        self.logger.info(f"Entering main loop (interval: {self.interval}s)")

        try:
            while self.running:
                self.run_pulse()
                time.sleep(self.interval)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            mt5.shutdown()
            self.logger.info("Worker stopped")


def worker_process(symbol: str, account: str, interval: int):
    """Entry point for worker process."""
    worker = SymbolWorker(symbol, account, interval)
    worker.start()


class SignalOrchestrator:
    """
    Orchestrates multiple symbol workers.
    Monitors worker health and restarts crashed processes.
    """

    def __init__(self, account: str, symbols: List[str], interval: int = 300):
        self.account = account
        self.symbols = symbols
        self.interval = interval
        self.processes = {}
        self.running = False

    def start_worker(self, symbol: str):
        """Start a worker process for a symbol."""
        logger.info(f"Starting worker for {symbol}...")

        process = mp.Process(
            target=worker_process,
            args=(symbol, self.account, self.interval),
            name=f"QNIF-{symbol}"
        )
        process.start()
        self.processes[symbol] = process

        logger.info(f"Worker started for {symbol} (PID: {process.pid})")

    def monitor_workers(self):
        """Monitor worker health and restart if crashed."""
        for symbol, process in list(self.processes.items()):
            if not process.is_alive():
                exit_code = process.exitcode
                logger.warning(f"Worker {symbol} died (exit code: {exit_code}). Restarting...")

                # Remove dead process
                del self.processes[symbol]

                # Wait before restart
                time.sleep(5)

                # Restart worker
                self.start_worker(symbol)

    def start(self):
        """Start all workers and monitoring loop."""
        logger.info("="*60)
        logger.info(f"QNIF SIGNAL ORCHESTRATOR - {self.account}")
        logger.info("="*60)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Interval: {self.interval}s")
        logger.info(f"PID: {os.getpid()}")
        logger.info("="*60)

        # Start all workers
        for symbol in self.symbols:
            self.start_worker(symbol)
            time.sleep(2)  # Stagger startup

        self.running = True
        logger.info("All workers started. Entering monitoring loop...")

        try:
            while self.running:
                time.sleep(30)  # Check every 30 seconds
                self.monitor_workers()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down all workers...")
            self.stop()

    def stop(self):
        """Stop all workers."""
        self.running = False

        for symbol, process in self.processes.items():
            if process.is_alive():
                logger.info(f"Terminating worker {symbol} (PID: {process.pid})...")
                process.terminate()
                process.join(timeout=10)

                if process.is_alive():
                    logger.warning(f"Worker {symbol} did not terminate gracefully, killing...")
                    process.kill()

        logger.info("All workers stopped")


def main():
    parser = argparse.ArgumentParser(
        description="QNIF Multi-Symbol Signal Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qnif_signal_runner.py --account ATLAS --symbols BTCUSD ETHUSD
  python qnif_signal_runner.py --account FTMO --symbols BTCUSD ETHUSD --interval 300
        """
    )

    parser.add_argument(
        "--account",
        type=str,
        required=True,
        choices=["ATLAS", "FTMO"],
        help="Trading account to connect to"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        nargs='+',
        default=["BTCUSD", "ETHUSD"],
        help="Symbols to trade (space-separated)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Pulse interval in seconds (default: 300)"
    )

    args = parser.parse_args()

    # Create orchestrator and start
    orchestrator = SignalOrchestrator(
        account=args.account,
        symbols=args.symbols,
        interval=args.interval
    )

    orchestrator.start()


if __name__ == "__main__":
    # Set multiprocessing start method for Windows
    mp.set_start_method('spawn', force=True)
    main()
