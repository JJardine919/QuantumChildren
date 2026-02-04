
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import sys

# Import prepare_features from System_03_ETARE
sys.path.append(os.path.abspath('01_Systems/System_03_ETARE'))
from ETARE_module import prepare_features

def extract_features_from_archives(archive_dir):
    data_list = []
    for filename in os.listdir(archive_dir):
        if filename.endswith('.dqcp.npz'):
            path = os.path.join(archive_dir, filename)
            try:
                archive = np.load(path, allow_pickle=True)
                # Structure: ['state', 'params', 'ratio', 'symbol', 'timestamp']
                data_list.append({
                    'timestamp': archive['timestamp'].item() if hasattr(archive['timestamp'], 'item') else archive['timestamp'],
                    'ratio': archive['ratio'].item() if hasattr(archive['ratio'], 'item') else archive['ratio'],
                    'symbol': str(archive['symbol']),
                    'file': filename
                })
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return pd.DataFrame(data_list)

def get_market_features(df_archives):
    if not mt5.initialize():
        print("MT5 initialization failed")
        return None

    all_data = []
    
    for symbol in df_archives['symbol'].unique():
        symbol_archives = df_archives[df_archives['symbol'] == symbol]
        print(f"Processing {symbol}...")
        
        # Get historical data around the timestamps
        # We need at least 100 bars for ETARE features
        for _, row in symbol_archives.iterrows():
            ts = row['timestamp']
            if isinstance(ts, str):
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            else:
                dt = ts
                
            # Fetch 150 bars leading up to the timestamp to ensure enough data for rolling features
            rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, dt, 1) # Just one bar at the timestamp
            if rates is None or len(rates) == 0:
                # Try fetching by position if timestamp is too recent/exact
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 150)
            else:
                # Fetch 150 bars ending at this timestamp
                rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, dt, 150)
            
            if rates is not None and len(rates) >= 100:
                df_rates = pd.DataFrame(rates)
                df_rates['time'] = pd.to_datetime(df_rates['time'], unit='s')
                
                # Prepare ETARE features
                features = prepare_features(df_rates)
                
                # Get the last row (at the timestamp)
                last_features = features.iloc[-1].to_dict()
                last_features['ratio'] = row['ratio']
                last_features['timestamp'] = ts
                last_features['symbol'] = symbol
                
                # Add mock entropy for now (can be replaced with real Quantum LSTM output later)
                last_features['quantum_entropy'] = 1.5 + np.random.normal(0, 0.5) 
                last_features['fusion_score'] = (1.0 - row['ratio']) * 0.5 + (0.5 if last_features['rsi'] > 50 else 0.0)
                
                all_data.append(last_features)
            else:
                print(f"Skipping {ts} - insufficient MT5 data")

    mt5.shutdown()
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    archive_dir = "04_Data/Archive"
    print(f"Extracting features from {archive_dir}...")
    df_archives = extract_features_from_archives(archive_dir)
    print(f"Found {len(df_archives)} archived states.")
    
    if len(df_archives) > 0:
        df_fusion = get_market_features(df_archives)
        if df_fusion is not None:
            output_path = "ETARE_QuantumFusion/data/fusion_training_set.csv"
            df_fusion.to_csv(output_path, index=False)
            print(f"Successfully created fusion dataset: {output_path}")
            print(f"Dataset shape: {df_fusion.shape}")
            print("Columns:", df_fusion.columns.tolist())
