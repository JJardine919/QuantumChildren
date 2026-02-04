"""
Create Quantum States from Market Data for Compression Testing
Purpose: Convert BTCUSD price movements to quantum state vectors for testing compression theory
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def normalize_to_angles(data, min_val=-np.pi, max_val=np.pi):
    """Normalize data to angle range for quantum encoding"""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data)
    normalized = (data - data_min) / (data_max - data_min)
    return min_val + normalized * (max_val - min_val)

def price_to_quantum_state(price_data, num_qubits=8):
    """
    Convert 256 bars of price data to 256-dimensional quantum state vector
    Using Direct Amplitude Encoding (Real-valued)
    
    Args:
        price_data: Array of 256 price values
        num_qubits: Number of qubits (8 qubits = 2^8 = 256 states)

    Returns:
        Complex-valued quantum state vector (256 elements)
    """
    assert len(price_data) == 256, f"Need exactly 256 prices, got {len(price_data)}"

    # Direct Amplitude Encoding
    # The state vector is simply the normalized price series.
    # Trending markets (smooth curves) are low-frequency and compress well.
    # Choppy markets (high frequency noise) are high-frequency and compress poorly.
    
    # Shift to positive (if needed) and normalize L2 norm to 1
    data = np.array(price_data) - np.min(price_data) + 0.01 # Ensure positive
    norm = np.linalg.norm(data)
    state_vector = data / norm
    
    # Convert to complex type (required for Qutip/Qiskit)
    state_vector = state_vector.astype(complex)

    return state_vector

def get_market_data(symbol, timeframe, bars, start_date=None):
    """Get historical data from MT5"""
    if not mt5.initialize():
        print("MT5 initialization failed")
        return None

    if start_date is None:
        # Get most recent data
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    else:
        # Get data from specific date
        rates = mt5.copy_rates_from(symbol, timeframe, start_date, bars)

    mt5.shutdown()

    if rates is None or len(rates) < bars:
        print(f"Failed to get {bars} bars, got {len(rates) if rates is not None else 0}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def identify_market_regimes():
    """
    Identify different market regimes for testing
    Returns dates for trending and choppy periods
    """
    print("\n" + "="*60)
    print("MARKET REGIME IDENTIFICATION")
    print("="*60)

    # We'll use recent data and manually inspect
    # In production, you'd use algorithmic detection (ADX, volatility, etc.)

    regimes = {
        'uptrend': {
            'description': 'Strong uptrend - January 2026',
            'start_date': datetime(2026, 1, 1),
            'expected_compression': '< 0.6 (high compressibility)'
        },
        'choppy': {
            'description': 'Choppy/ranging - December 2025',
            'start_date': datetime(2025, 12, 15),
            'expected_compression': '> 0.8 (low compressibility)'
        },
        'downtrend': {
            'description': 'Pullback/correction - Late December 2025',
            'start_date': datetime(2025, 12, 28),
            'expected_compression': '< 0.6 (high compressibility)'
        }
    }

    return regimes

def create_test_states():
    """Create quantum states for different market regimes"""

    print("\n" + "="*60)
    print("CREATING QUANTUM STATES FOR COMPRESSION TESTING")
    print("="*60)

    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_M5
    bars = 256

    regimes = identify_market_regimes()

    output_dir = "C:/Users/jjj10/QuantumTradingLibrary/04_Data/QuantumStates"
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for regime_name, regime_info in regimes.items():
        print(f"\n{'='*60}")
        print(f"REGIME: {regime_name.upper()}")
        print(f"Description: {regime_info['description']}")
        print(f"Expected compression: {regime_info['expected_compression']}")
        print(f"{'='*60}")

        # Get market data
        print(f"Fetching {bars} bars of {symbol} M5 from {regime_info['start_date'].strftime('%Y-%m-%d')}...")
        df = get_market_data(symbol, timeframe, bars, regime_info['start_date'])

        if df is None:
            print(f"❌ Failed to get data for {regime_name}")
            continue

        # Extract close prices
        close_prices = df['close'].values

        # ============================================================
        # STEP 1.5: DENOISE THE DATA (DISABLED FOR VALIDATION)
        # We want the quantum autoencoder to detect the noise/complexity itself.
        # Pre-denoising makes choppy markets look trending, defeating the metric.
        # ============================================================
        # print("\nDenoising data using 'Midas-style' wavelet filter...")
        # from utils.signal_processing import NoiseReducer
        # reducer = NoiseReducer()
        # denoised_prices = reducer.midas_style_denoise(close_prices)
        
        # Use RAW prices for validation
        denoised_prices = close_prices

        # Calculate basic statistics
        returns = np.diff(denoised_prices) / denoised_prices[:-1]
        volatility = np.std(returns)
        trend = (denoised_prices[-1] - denoised_prices[0]) / denoised_prices[0]

        print(f"\nMarket Statistics (Denoised):")
        print(f"  Price range: ${denoised_prices.min():.2f} - ${denoised_prices.max():.2f}")
        print(f"  Total return: {trend*100:.2f}%")
        print(f"  Volatility: {volatility*100:.3f}%")
        print(f"  Avg price: ${denoised_prices.mean():.2f}")

        # Convert to quantum state
        print(f"\nConverting to quantum state vector...")
        quantum_state = price_to_quantum_state(denoised_prices)

        # Save to .npy file
        output_file = f"{output_dir}/btc_{regime_name}_state.npy"
        np.save(output_file, quantum_state)

        print(f"✅ Saved quantum state to: {output_file}")
        print(f"   State vector size: {len(quantum_state)} complex numbers")
        print(f"   Norm (should be ~1.0): {np.linalg.norm(quantum_state):.6f}")

        results.append({
            'regime': regime_name,
            'file': output_file,
            'volatility': volatility,
            'trend': trend,
            'expected_compression': regime_info['expected_compression']
        })

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - FILES READY FOR COMPRESSION TESTING")
    print("="*60)

    for result in results:
        print(f"\n{result['regime'].upper()}:")
        print(f"  File: {result['file']}")
        print(f"  Expected compression: {result['expected_compression']}")

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Run deep_quantum_compress_pro.py GUI:")
    print("   python C:/Users/jjj10/QuantumTradingLibrary/01_Systems/QuantumCompression/deep_quantum_compress_pro.py")
    print("\n2. Load each .npy file and compress")
    print("\n3. Record compression ratios:")
    print("   - Trending (up/down): Should be < 0.6")
    print("   - Choppy: Should be > 0.8")
    print("\n4. If ratios match expectations → Theory validated ✅")
    print("   If ratios don't separate → Need to rethink approach ❌")
    print("="*60)

    return results

if __name__ == "__main__":
    create_test_states()
