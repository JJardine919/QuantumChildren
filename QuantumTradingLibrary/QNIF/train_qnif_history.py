"""
QNIF TRAINING ORCHESTRATOR
==========================
Accelerated "Bone Marrow" Training for the Bio-Quantum System.

Instead of waiting for live ticks, this script:
1. Fetches deep history (1 Year M5 data) from MT5.
2. Spins up the QNIF Engine + VDJ Immune System.
3. "Feeds" the organism the historical data at max speed.
4. Forces VDJ Recombination on every historical "Loss" or "Shock".
5. Populates 'vdj_memory_cells.db' with thousands of battle-tested antibodies.

Usage:
    python train_qnif_history.py --symbol BTCUSD --days 365
"""

import sys
import logging
import argparse
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# Add root to path
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

from QNIF.QNIF_Master import QNIF_Engine, BioQuantumState
from credential_manager import get_credentials

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][QNIF-TRAIN] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("QNIF-TRAIN")

class QNIF_Trainer:
    def __init__(self, symbol: str, days: int, account: str):
        self.symbol = symbol
        self.days = days
        self.account = account
        self.engine = QNIF_Engine()
        self.history = None

    def connect_and_fetch(self):
        """Fetch historical data for training."""
        logger.info(f"Connecting to {self.account} to fetch {self.days} days of {self.symbol}...")
        try:
            creds = get_credentials(self.account)
            if not mt5.initialize(path=creds.get('terminal_path')):
                logger.error("MT5 init failed")
                return False
            
            # Fetch data
            # Use copy_rates_from_pos for simplicity and robustness
            count = self.days * 288
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, count)

            mt5.shutdown()

            if rates is None:
                logger.error("No data fetched.")
                return False

            self.history = pd.DataFrame(rates)
            self.history['time'] = pd.to_datetime(self.history['time'], unit='s')
            logger.info(f"Fetched {len(self.history)} bars. Range: {self.history['time'].iloc[0]} -> {self.history['time'].iloc[-1]}")
            return True

        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return False

    def train(self):
        """Run the training loop."""
        if self.history is None:
            return

        logger.info("BEGINNING BONE MARROW ACCELERATION...")
        logger.info("Injecting historical data into QNIF Engine...")

        # Convert to numpy for speed
        # [open, high, low, close, tick_volume]
        data_np = self.history[['open', 'high', 'low', 'close', 'tick_volume']].to_numpy()
        
        # Sliding window of 256 bars
        window_size = 256
        total_steps = len(data_np) - window_size
        
        # Performance tracking
        antibodies_generated = 0
        start_time = time.time()

        # Step through history
        step_stride = 5 
        
        for i in range(0, total_steps, step_stride):
            window = data_np[i : i + window_size]
            
            # Pulse the engine
            # We suppress detailed logging inside the engine for speed
            # by manipulating the logger level momentarily if needed, 
            # but QNIF_Master logs to INFO.
            
            state = self.engine.process_pulse(self.symbol, window)
            
            if state.selected_antibody.get('source') == 'BONE_MARROW':
                antibodies_generated += 1
            
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"\rProgress: {i}/{total_steps} ({i/total_steps*100:.1f}%) | Antibodies: {antibodies_generated} | Speed: {rate:.1f} bars/s | Regime: {state.regime}", end="")

        print("\nTRAINING COMPLETE.")
        print(f"Total Antibodies Generated: {antibodies_generated}")
        print("Memory Database populated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSD")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--account", type=str, default="ATLAS")
    args = parser.parse_args()

    trainer = QNIF_Trainer(args.symbol, args.days, args.account)
    if trainer.connect_and_fetch():
        trainer.train()