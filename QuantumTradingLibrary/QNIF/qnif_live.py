"""
QNIF LIVE RUNNER v1.0
====================
The unified runner for Quantum Neural-Immune Fusion.
Integrates Layer 1 (Compression), Layer 2/3 (TEQA), and Layer 4 (VDJ).

Run: python qnif_live.py --symbol BTCUSD --account ATLAS
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import MetaTrader5 as mt5

# Add root to path
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

from QNIF.QNIF_Master import QNIF_Engine, BioQuantumState
from credential_manager import get_credentials, CredentialError

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][QNIF-LIVE] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('qnif_live.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QNIF-LIVE")

class QNIF_LiveRunner:
    def __init__(self, symbol: str, account: str, interval: int = 300):
        self.symbol = symbol
        self.account = account
        self.interval = interval
        self.engine = QNIF_Engine()
        self.running = False

    def connect_mt5(self):
        try:
            creds = get_credentials(self.account)
            if not mt5.initialize(path=creds.get('terminal_path')):
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
            
            if not mt5.login(creds['account'], password=creds['password'], server=creds['server']):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
                
            logger.info(f"Connected to MT5 Account {self.account}")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def fetch_bars(self, count: int = 300):
        bars = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, count)
        if bars is None or len(bars) == 0:
            logger.error(f"Failed to fetch bars for {self.symbol}")
            return None
        
        # Convert to numpy [open, high, low, close, tick_volume]
        res = np.zeros((len(bars), 5))
        for i, b in enumerate(bars):
            res[i] = [b['open'], b['high'], b['low'], b['close'], b['tick_volume']]
        return res

    def save_signal(self, state: BioQuantumState):
        signal_file = root_dir / f"qnif_signal_{self.symbol}.json"
        data = {
            "symbol": state.symbol,
            "timestamp": datetime.now().isoformat(),
            "action": state.final_action,
            "confidence": float(state.final_confidence),
            "lot_multiplier": float(state.lot_multiplier),
            "compression": {
                "ratio": float(state.compression_ratio),
                "regime": state.regime,
                "tradeable": state.is_tradeable
            },
            "teqa": {
                "active_tes": state.active_tes,
                "shock_level": float(state.shock_level),
                "shock_label": state.shock_label,
                "consensus": float(state.neural_consensus)
            },
            "vdj": {
                "antibody_id": state.selected_antibody.get("antibody_id", ""),
                "generation": int(state.generation),
                "is_memory": state.is_memory_recall
            }
        }
        with open(signal_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        # Legacy link
        shutil_file = root_dir / "qnif_signal.json"
        with open(shutil_file, 'w') as f:
            json.dump(data, f, indent=4)

    def run_once(self):
        logger.info(f"--- QNIF PULSE: {self.symbol} ---")
        bars = self.fetch_bars()
        if bars is not None:
            state = self.engine.process_pulse(self.symbol, bars)
            self.save_signal(state)
            logger.info(f"Action: {state.final_action} | Conf: {state.final_confidence:.2f} | Ratio: {state.compression_ratio:.2f}")

    def start(self):
        if not self.connect_mt5():
            return
            
        self.running = True
        logger.info(f"Starting QNIF Live Loop for {self.symbol} (Interval: {self.interval}s)")
        
        try:
            while self.running:
                self.run_once()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("Stopping QNIF Live Runner...")
        finally:
            mt5.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSD")
    parser.add_argument("--account", type=str, default="ATLAS")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()

    runner = QNIF_LiveRunner(args.symbol, args.account, args.interval)
    runner.start()
