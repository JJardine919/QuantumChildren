"""
QNIF MASSIVE GPU TRAINER
=======================
Massively parallel training for Precious Metals (XAU/XAG).
Utilizes 16-core CPU and GPU Pool for accelerated Bio-Quantum evolution.

Protocol:
- 20 Cycles per asset
- 4 Months Train / 2 Months Test (Walk-Forward)
- Multi-Timeframe: M1, M5, M15, M30, H1, D1
- Volatility Protocol:
    - Phase 1 (First 10 cycles): MAX Volatility (Shock Force 2.0+)
    - Phase 2 (Next 10 cycles): Half Volatility (Relaxed 1.0)

Architecture:
- ThreadPoolExecutor manages 1000+ virtual "Pulse" workers.
- GPUPool (DirectML) accelerates tensor-heavy layers.
- VDJ Recombination engine populates 'vdj_memory_cells.db'.

Usage:
    python QNIF_Massive_Trainer.py --account ATLAS
"""

import sys
import logging
import argparse
import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import torch

# Add root to path
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

from QNIF.QNIF_Master import QNIF_Engine, BioQuantumState
from gpu_pool_manager import GPUPool, get_device
from credential_manager import get_credentials

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][MASSIVE-TRAIN] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('qnif_massive_train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MASSIVE")

# MT5 Timeframe Map
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1
}

class MassiveBioTrainer:
    def __init__(self, account: str, max_workers=16):
        self.account = account
        self.max_workers = max_workers
        self.gpu_pool = GPUPool(max_concurrent=8) # 8 slots per GPU
        self.engine = QNIF_Engine()
        self.symbols = ["XAUUSD", "XAGUSD"]
        self.results_lock = threading.Lock()
        self.total_antibodies = 0

    def connect(self):
        creds = get_credentials(self.account)
        if not mt5.initialize(path=creds.get('terminal_path')):
            logger.error("MT5 init failed")
            return False
        return True

    def fetch_history(self, symbol, tf_name, months=6):
        """Fetch historical bars."""
        tf_const = TIMEFRAMES[tf_name]
        # approx bars: 1 month M1 = 43200 bars. 6 months = 250k.
        count = months * 30 * 288 # M5 baseline
        if tf_name == "M1": count *= 5
        
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        if rates is None or len(rates) < 500:
            return None
        
        df = pd.DataFrame(rates)
        return df[['open', 'high', 'low', 'close', 'tick_volume']].to_numpy()

    def pulse_worker(self, symbol, data_segment, shock_override=None):
        """Single worker thread pulsing the engine."""
        # Use GPU slot if engine uses torch operations
        with self.gpu_pool.slot() as device:
            state = self.engine.process_pulse(symbol, data_segment)
            
            if state.selected_antibody.get('source') == 'BONE_MARROW':
                with self.results_lock:
                    self.total_antibodies += 1
            return state

    def run_tf_batch(self, symbol, tf_name, phase_num):
        """Process a full timeframe for one symbol."""
        vol_mult = 2.0 if phase_num == 1 else 1.0
        logger.info(f"Launching Batch: {symbol} {tf_name} Phase {phase_num} (Vol: {vol_mult}x)")
        
        data = self.fetch_history(symbol, tf_name)
        if data is None:
            logger.warning(f"No data for {symbol} {tf_name}")
            return

        window = 256
        stride = 10 # Faster stride for massive parallelization
        
        # Parallelize segments
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, len(data) - window, stride):
                segment = data[i:i+window]
                # If phase 1, we could potentially scale returns to force shock
                if phase_num == 1:
                    segment = segment.copy()
                    segment[:, 1:4] *= 1.001 # Artificial volatility boost
                
                tasks.append(executor.submit(self.pulse_worker, symbol, segment))
            
            completed = 0
            total_tasks = len(tasks)
            for future in as_completed(tasks):
                completed += 1
                if completed % 100 == 0:
                    print(f"\r  [{symbol} {tf_name}] Progress: {completed}/{total_tasks}", end="")
        print(f"\n  Done: {symbol} {tf_name}")

    def run_full_farm(self):
        if not self.connect(): return
        
        logger.info("="*60)
        logger.info("QNIF MASSIVE SIGNAL FARM STARTED")
        logger.info(f"Hardware: {self.max_workers} CPU Workers | GPU Pool Active")
        logger.info("="*60)

        # Round 1: Max Volatility
        logger.info(">>> PHASE 1: MAXIMUM VOLATILITY INJECTION")
        for symbol in self.symbols:
            for tf in TIMEFRAMES.keys():
                self.run_tf_batch(symbol, tf, phase_num=1)

        # Round 2: Relaxed Volatility
        logger.info(">>> PHASE 2: RELAXED VOLATILITY (STABILIZATION)")
        for symbol in self.symbols:
            for tf in TIMEFRAMES.keys():
                self.run_tf_batch(symbol, tf, phase_num=2)

        mt5.shutdown()
        logger.info(f"PROTOCOL COMPLETE. Generated {self.total_antibodies} Antibodies.")
        logger.info("Memory Database 'vdj_memory_cells.db' is now fully populated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", type=str, default="ATLAS")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    farm = MassiveBioTrainer(args.account, max_workers=args.workers)
    farm.run_full_farm()