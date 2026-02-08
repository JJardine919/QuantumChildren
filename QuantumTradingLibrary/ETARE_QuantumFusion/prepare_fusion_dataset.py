
import os
import json
import glob
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import sys

# Import prepare_features from System_03_ETARE
sys.path.append(os.path.abspath('01_Systems/System_03_ETARE'))
from ETARE_module import prepare_features

SIGNAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'quantum_data')


def extract_signals_from_jsonl(signal_dir):
    """Read all locally collected signals from JSONL files."""
    data_list = []
    pattern = os.path.join(signal_dir, "signals_*.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No signal files found in {signal_dir}")
        return pd.DataFrame()

    for filepath in files:
        count = 0
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    signal = json.loads(line)
                    data_list.append({
                        'timestamp': signal.get('timestamp', ''),
                        'symbol': signal.get('symbol', ''),
                        'price': signal.get('price', 0),
                        'direction': signal.get('direction', 'HOLD'),
                        'confidence': signal.get('confidence', 0),
                        'quantum_entropy': signal.get('quantum_entropy', 0),
                        'dominant_state': signal.get('dominant_state', 0),
                        'regime': signal.get('regime', ''),
                        'source': signal.get('source', ''),
                    })
                    count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {count} signals from {os.path.basename(filepath)}")

    return pd.DataFrame(data_list)


# Legacy support: read from .dqcp.npz archives if they exist
def extract_features_from_archives(archive_dir):
    data_list = []
    if not os.path.exists(archive_dir):
        return pd.DataFrame()
    for filename in os.listdir(archive_dir):
        if filename.endswith('.dqcp.npz'):
            path = os.path.join(archive_dir, filename)
            try:
                archive = np.load(path, allow_pickle=True)
                data_list.append({
                    'timestamp': archive['timestamp'].item() if hasattr(archive['timestamp'], 'item') else archive['timestamp'],
                    'ratio': archive['ratio'].item() if hasattr(archive['ratio'], 'item') else archive['ratio'],
                    'symbol': str(archive['symbol']),
                    'file': filename
                })
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return pd.DataFrame(data_list)


def get_market_features_from_signals(df_signals):
    """Fetch MT5 data for each signal and generate ETARE features."""
    # Try known terminal paths
    terminal_paths = [
        r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
        r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe",
    ]
    initialized = False
    for path in terminal_paths:
        if os.path.exists(path):
            if mt5.initialize(path=path):
                initialized = True
                print(f"  MT5 connected via {os.path.basename(os.path.dirname(path))}")
                break
    if not initialized:
        if not mt5.initialize():
            print("MT5 initialization failed - ensure a terminal is running")
            return None

    all_data = []
    total = len(df_signals)
    skipped = 0

    for idx, row in df_signals.iterrows():
        symbol = row['symbol']
        ts_str = row['timestamp']

        try:
            # Parse ISO timestamp
            if 'T' in str(ts_str):
                dt = pd.Timestamp(ts_str).to_pydatetime()
            elif '_' in str(ts_str):
                dt = datetime.strptime(str(ts_str), "%Y%m%d_%H%M%S")
            else:
                dt = pd.Timestamp(ts_str).to_pydatetime()
        except Exception:
            skipped += 1
            continue

        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            skipped += 1
            continue

        # Fetch 150 M5 bars ending at signal timestamp
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, dt, 150)
        if rates is None or len(rates) < 100:
            # Fallback: try fetching recent data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 150)

        if rates is not None and len(rates) >= 100:
            df_rates = pd.DataFrame(rates)
            df_rates['time'] = pd.to_datetime(df_rates['time'], unit='s')

            # Prepare ETARE features
            features = prepare_features(df_rates)

            if len(features) > 0:
                # Get the last row (at the timestamp)
                last_features = features.iloc[-1].to_dict()

                # Add signal metadata (real values, not mocked)
                last_features['timestamp'] = ts_str
                last_features['symbol'] = symbol
                last_features['price'] = row['price']
                last_features['direction'] = row['direction']
                last_features['confidence'] = row['confidence']
                last_features['quantum_entropy'] = row['quantum_entropy']
                last_features['dominant_state'] = row['dominant_state']
                last_features['regime'] = row['regime']

                # Fusion score from real data
                rsi_val = last_features.get('rsi', 0)
                last_features['fusion_score'] = (
                    row['confidence'] * 0.4 +
                    (0.3 if row['quantum_entropy'] > 0.5 else 0.0) +
                    (0.3 if abs(rsi_val) < 1.0 else 0.1)
                )

                all_data.append(last_features)
        else:
            skipped += 1

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{total} signals ({len(all_data)} valid, {skipped} skipped)")

    mt5.shutdown()
    print(f"  Final: {len(all_data)} valid samples from {total} signals ({skipped} skipped)")
    return pd.DataFrame(all_data)


if __name__ == "__main__":
    # Primary source: local JSONL signals
    print(f"Loading signals from {SIGNAL_DIR}...")
    df_signals = extract_signals_from_jsonl(SIGNAL_DIR)
    print(f"Found {len(df_signals)} collected signals.")

    # Fallback: check for legacy archives
    archive_dir = "04_Data/Archive"
    if len(df_signals) == 0 and os.path.exists(archive_dir):
        print(f"No JSONL signals found, trying archives in {archive_dir}...")
        df_signals = extract_features_from_archives(archive_dir)
        print(f"Found {len(df_signals)} archived states.")

    if len(df_signals) > 0:
        print(f"\nGenerating ETARE features from MT5 data...")
        df_fusion = get_market_features_from_signals(df_signals)
        if df_fusion is not None and len(df_fusion) > 0:
            output_path = "ETARE_QuantumFusion/data/fusion_training_set.csv"
            df_fusion.to_csv(output_path, index=False)
            print(f"\nSuccessfully created fusion dataset: {output_path}")
            print(f"Dataset shape: {df_fusion.shape}")
            print(f"Columns: {df_fusion.columns.tolist()}")
            print(f"Symbols: {df_fusion['symbol'].value_counts().to_dict()}")
            print(f"Directions: {df_fusion['direction'].value_counts().to_dict()}")
        else:
            print("Failed to generate features - check MT5 connection")
    else:
        print("No signal data found. Run BRAIN scripts to collect signals first.")
