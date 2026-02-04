import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def price_to_quantum_state(df, feature='close', vector_length=256):
    """
    Convert price series to normalized quantum state vector.
    """
    prices = df[feature].values
    
    # Ensure exact length: truncate or pad
    if len(prices) > vector_length:
        prices = prices[-vector_length:]
    elif len(prices) < vector_length:
        prices = np.pad(prices, (0, vector_length - len(prices)), mode='constant')
    
    # Normalize to [0, 1]
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    # Use real amplitudes (complex for quantum compatibility)
    state_vector = normalized.astype(complex)
    
    # Normalize to unit norm (required for quantum state)
    norm = np.linalg.norm(state_vector)
    if norm > 0:
        state_vector /= norm
    
    return state_vector

import os

# List of files and labels
files = [
    ('../../04_Data/MarketData/btc_uptrend_256bars.csv', 'uptrend'),
    ('../../04_Data/MarketData/btc_choppy_256bars.csv', 'choppy'),
    ('../../04_Data/MarketData/btc_pullback_256bars.csv', 'pullback')
]

output_dir = '../../04_Data/QuantumStates'
os.makedirs(output_dir, exist_ok=True)

for csv_path, label in files:
    try:
        df = pd.read_csv(csv_path)
        if 'close' not in df.columns:
            print(f"Error: 'close' column missing in {csv_path}")
            continue
        
        state = price_to_quantum_state(df)
        
        output_file = os.path.join(output_dir, f'btc_{label}_state.npy')
        np.save(output_file, state)
        print(f"Generated: {output_file} (shape: {state.shape}, norm: {np.linalg.norm(state):.6f})")
        
        # Quick regime check
        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        print(f"  {label.capitalize()} regime net change: {price_change:.2f}%")
        
    except Exception as e:
        print(f"Failed processing {csv_file}: {e}")

print("\nAll conversions complete. Proceed to compression testing.")