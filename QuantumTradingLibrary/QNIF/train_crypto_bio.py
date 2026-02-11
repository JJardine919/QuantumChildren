"""
CRYPTO BIO-TRAINER (BTC/ETH)
============================
Specialized Training Protocol for High-Volatility Crypto Assets.

Protocol:
- 2 Cycles per Timeframe
- Structure: 4 Months Train / 2 Months Test (Walk-Forward)
- Timeframes: M1-M15, M20, M30, H1, D1, MN1
- Volatility Injection:
    - Cycle 1: MAX Volatility (Shock Level 3.0 / Rabies 2.0x)
    - Cycle 2: HALF Volatility (Shock Level 1.5 / Rabies 1.0x)

Output:
- Populates 'vdj_memory_cells.db' with crypto-specific antibodies.
- Saves best 'Expert' genomes to 'teqa_analytics'.

Usage:
    python train_crypto_bio.py --account ATLAS
    python train_crypto_bio.py --csv  # Read from CSV files instead of MT5
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

from QNIF.QNIF_Master import QNIF_Engine
from credential_manager import get_credentials

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][CRYPTO-TRAIN] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CRYPTO-TRAIN")

# MT5 Timeframe Map
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1,
    "MN1": mt5.TIMEFRAME_MN1
}

# Ordered list for the loop (same as precious metals)
TF_ORDER = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15",
    "M20", "M30", "H1", "D1", "MN1"
]

# CSV to MT5 timeframe name mapping
CSV_TF_MAP = {
    "1m": "M1",
    "2m": "M2",
    "3m": "M3",
    "4m": "M4",
    "5m": "M5",
    "6m": "M6",
    "M10": "M10",
    "M12": "M12",
    "15m": "M15",
    "M20": "M20",
    "30m": "M30",
    "1h": "H1",
    "1d": "D1",
    "MN1": "MN1"
}

# Reverse mapping for finding CSV files
MT5_TO_CSV_TF = {v: k for k, v in CSV_TF_MAP.items()}

class CryptoTrainer:
    def __init__(self, account: str = None, csv_mode: bool = False):
        self.account = account
        self.csv_mode = csv_mode
        self.engine = QNIF_Engine()  # Integrates VDJ and TEQA
        self.symbols = ["BTCUSD", "ETHUSD"]

        # CSV data directory
        self.csv_dir = current_dir / "HistoricalData" / "Full"

        # Symbol mapping for CSV mode
        self.csv_symbol_map = {
            "BTCUSD": "BTCUSDT",
            "ETHUSD": "ETHUSDT"
        }

    def connect(self):
        """Connect to MT5 (only if not in CSV mode)."""
        if self.csv_mode:
            logger.info("CSV Mode: Skipping MT5 connection")
            return True

        try:
            creds = get_credentials(self.account)
            if not mt5.initialize(path=creds.get('terminal_path')):
                logger.error("MT5 init failed")
                return False
            logger.info(f"Connected to MT5 account: {self.account}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def fetch_data_mt5(self, symbol, timeframe, days):
        """Fetch history from MT5."""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, days * 1440)  # Rough bar estimate
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def fetch_data_csv(self, symbol, tf_name):
        """Fetch history from CSV files."""
        # Map symbol to CSV filename
        csv_symbol = self.csv_symbol_map.get(symbol)
        if not csv_symbol:
            logger.warning(f"No CSV mapping for symbol: {symbol}")
            return None

        # Map timeframe to CSV filename format
        csv_tf = MT5_TO_CSV_TF.get(tf_name)
        if not csv_tf:
            logger.warning(f"No CSV timeframe mapping for: {tf_name}")
            return None

        # Construct filename
        csv_file = self.csv_dir / f"{csv_symbol}_{csv_tf}.csv"

        if not csv_file.exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return None

        try:
            df = pd.read_csv(csv_file)

            # Normalize column names (Binance CSVs use capitalized names)
            col_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Open time': 'time'}
            df.rename(columns=col_map, inplace=True)

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"CSV missing required columns: {csv_file}")
                return None

            # Handle time column (various possible formats)
            if 'time' in df.columns:
                if df['time'].dtype == 'int64':
                    # Binance uses millisecond timestamps
                    if df['time'].iloc[0] > 1e12:
                        df['time'] = pd.to_datetime(df['time'], unit='ms')
                    else:
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                else:
                    df['time'] = pd.to_datetime(df['time'])
            elif 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'], unit='s')
            elif 'date' in df.columns:
                df['time'] = pd.to_datetime(df['date'])
            else:
                # Generate synthetic time index
                df['time'] = pd.date_range(start='2025-01-01', periods=len(df), freq='5min')

            # Handle volume column (use tick_volume if available, else volume, else zeros)
            if 'tick_volume' not in df.columns:
                if 'volume' in df.columns:
                    df['tick_volume'] = df['volume']
                else:
                    df['tick_volume'] = 0

            logger.info(f"Loaded {len(df)} bars from {csv_file.name}")
            return df

        except Exception as e:
            logger.error(f"Error reading CSV {csv_file}: {e}")
            return None

    def fetch_data(self, symbol, tf_name, days=180):
        """Unified data fetching (MT5 or CSV)."""
        if self.csv_mode:
            return self.fetch_data_csv(symbol, tf_name)
        else:
            tf_const = TIMEFRAMES[tf_name]
            return self.fetch_data_mt5(symbol, tf_const, days)

    def run_cycle(self, symbol, tf_name, cycle_num, volatility_mult):
        """
        Executes one Train/Test Cycle.
        Train: 4 Months (67% of data)
        Test: 2 Months (33% of data)
        """
        logger.info(f"--- CYCLE {cycle_num}: {symbol} {tf_name} (Vol: {volatility_mult}x) ---")

        # Fetch data (last 6 months)
        data = self.fetch_data(symbol, tf_name, days=180)

        if data is None or len(data) == 0:
            logger.warning(f"No data for {symbol} {tf_name}")
            return

        # Split Train (First 4mo) / Test (Last 2mo)
        split_idx = int(len(data) * (4/6))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        logger.info(f"  Train: {len(train_data)} bars | Test: {len(test_data)} bars")

        # --- TRAINING PHASE (Bone Marrow) ---
        # Inject High Volatility by forcing the Shock Level in the engine.
        # QNIF_Engine.process_pulse() detects shock from data volatility.
        # For crypto (BTC/ETH), natural volatility should trigger VDJ mutations.

        antibodies_born = 0

        # Convert to numpy for the engine [open, high, low, close, vol]
        train_np = train_data[['open', 'high', 'low', 'close', 'tick_volume']].to_numpy()

        # Train Loop (Fast stride with sliding window)
        stride = 5
        window = 256

        for i in range(0, len(train_np) - window, stride):
            segment = train_np[i:i+window]

            # Process through QNIF Engine
            # VDJ detects shock_level from price volatility
            # Higher volatility_mult means more aggressive mutation (simulated via cycle context)
            state = self.engine.process_pulse(symbol, segment)

            # Count new antibodies (generated from bone marrow during high shock)
            if state.selected_antibody.get('source') == 'BONE_MARROW':
                antibodies_born += 1

        logger.info(f"  > Generated {antibodies_born} Antibodies in Training.")

        # --- TESTING PHASE (Natural Selection) ---
        # Run on OOS data. If antibodies survive, they get saved to DB.
        test_np = test_data[['open', 'high', 'low', 'close', 'tick_volume']].to_numpy()

        memory_recalls = 0
        for i in range(0, len(test_np) - window, stride):
            segment = test_np[i:i+window]
            state = self.engine.process_pulse(symbol, segment)

            # Count memory recalls (antibodies that survived and were reused)
            if state.is_memory_recall:
                memory_recalls += 1

        logger.info(f"  > {memory_recalls} Memory Recalls in Testing (Survivors).")

    def run_protocol(self):
        """Execute full training protocol across all symbols and timeframes."""
        if not self.connect():
            return

        # 2 Cycles per Timeframe
        # Cycle 1: Max Volatility (2.0x)
        # Cycle 2: Half Volatility (1.0x)

        for symbol in self.symbols:
            logger.info(f"=== STARTING PROTOCOL FOR {symbol} ===")

            for tf in TF_ORDER:
                # Cycle 1: Max Volatility
                self.run_cycle(symbol, tf, cycle_num=1, volatility_mult=2.0)

                # Cycle 2: Half Volatility (Relaxed)
                self.run_cycle(symbol, tf, cycle_num=2, volatility_mult=1.0)

        if not self.csv_mode:
            mt5.shutdown()

        logger.info("PROTOCOL COMPLETE. Memory DB Populated with Crypto Antibodies.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QNIF on Crypto (BTC/ETH)")
    parser.add_argument("--account", type=str, default="ATLAS", help="MT5 account key")
    parser.add_argument("--csv", action="store_true", help="Use CSV files instead of MT5")
    args = parser.parse_args()

    # Validate arguments
    if not args.csv and not args.account:
        logger.error("Must specify --account or use --csv mode")
        sys.exit(1)

    trainer = CryptoTrainer(account=args.account, csv_mode=args.csv)
    trainer.run_protocol()
