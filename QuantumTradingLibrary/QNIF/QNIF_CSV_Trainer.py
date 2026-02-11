"""
QNIF CSV MASSIVE TRAINER (XAU/XAG)
==================================
Uses local CSV data to bypass MT5 download limits.
Accelerated evolution for Precious Metals.

Architecture:
- Loads Binance-format CSVs (XAUUSDT, XAGUSDT).
- Pulsing segments into QNIF Engine.
- Multi-threaded segment processing via GPU Pool.
- Populates 'vdj_memory_cells.db' with surviving antibodies.

Usage:
    python QNIF_CSV_Trainer.py
"""

import sys
import logging
import json
import threading
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add root to path
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

from QNIF.QNIF_Master import QNIF_Engine, BioQuantumState
from gpu_pool_manager import GPUPool

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][CSV-TRAIN] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("CSV-TRAIN")

CSV_DIR = current_dir / "HistoricalData" / "Full"

class CSVMassiveTrainer:
    def __init__(self, max_workers=12):
        self.max_workers = max_workers
        self.gpu_pool = GPUPool(max_concurrent=8)
        self.engine = QNIF_Engine()
        self.results_lock = threading.Lock()
        self.total_antibodies = 0

    def load_csv(self, file_path):
        """Loads and formats Binance CSV."""
        try:
            # Binance columns: OpenTime, Open, High, Low, Close, Volume, ...
            df = pd.read_csv(file_path)
            # Map columns to QNIF format [open, high, low, close, volume]
            if 'Open' in df.columns:
                data = df[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
            else:
                data = df.iloc[:, 1:6].to_numpy()
            return data.astype(float)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None

    def pulse_worker(self, symbol, segment):
        """Worker thread pulsing the engine."""
        with self.gpu_pool.slot() as device:
            state = self.engine.process_pulse(symbol, segment)
            if state.selected_antibody.get('source') == 'BONE_MARROW' or state.is_memory_recall:
                with self.results_lock:
                    self.total_antibodies += 1
            return state

    def process_file(self, file_path):
        symbol = file_path.stem.split('_')[0].replace('USDT', 'USD')
        tf = file_path.stem.split('_')[1]
        
        logger.info(f">>> Processing Genome: {symbol} {tf}")
        data = self.load_csv(file_path)
        if data is None: return

        window = 256
        stride = 20 # Fast stride for massive history
        
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, len(data) - window, stride):
                segment = data[i:i+window]
                tasks.append(executor.submit(self.pulse_worker, symbol, segment))
            
            completed = 0
            total = len(tasks)
            for future in as_completed(tasks):
                completed += 1
                if completed % 500 == 0:
                    print(f"\r  [{symbol} {tf}] Progress: {completed}/{total} | Antibodies: {self.total_antibodies}", end="")
        print(f"\n  Finished {symbol} {tf}")

    def run_all(self):
        logger.info("="*60)
        logger.info("QNIF CSV SIGNAL FARM STARTED")
        logger.info(f"Searching: {CSV_DIR}")
        logger.info("="*60)

        csv_files = list(CSV_DIR.glob("*.csv"))
        # Prioritize 5m and 1h
        csv_files.sort(key=lambda x: "5m" in x.name or "1h" in x.name, reverse=True)

        for f in csv_files:
            self.process_file(f)

        logger.info(f"TRAINING COMPLETE. Memory Bank Populated with {self.total_antibodies} strategic snapshots.")

if __name__ == "__main__":
    trainer = CSVMassiveTrainer(max_workers=12)
    trainer.run_all()