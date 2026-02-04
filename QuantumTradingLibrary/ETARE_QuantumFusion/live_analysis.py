
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from modules.compression_layer import CompressionLayer
import sys
import os

def main():
    print("ETARE QUANTUM FUSION >> LIVE ENTROPY ANALYSIS")
    
    # 1. Connect to MT5
    if not mt5.initialize():
        print(f"FAILED to init MT5: {mt5.last_error()}")
        return

    symbol = "BTCUSD"
    mt5.symbol_select(symbol, True)
    
    # 2. Fetch 256 bars (M5)
    print(f"Fetching 256 bars for {symbol}...")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 256)
    mt5.shutdown()
    
    if rates is None or len(rates) < 256:
        print("Insufficient data.")
        return
        
    df = pd.DataFrame(rates)
    prices = df['close'].values
    
    # 3. Run Compression
    print("Analyzing Information Entropy...")
    engine = CompressionLayer()
    result = engine.process(prices)
    
    # 4. REPORT
    ratio = result['ratio']
    regime = result['regime']
    
    print("\n" + "="*40)
    print(f" LIVE MARKET STATE: {symbol}")
    print("="*40)
    print(f" Compression Ratio: {ratio:.4f}")
    print(f" Market Regime:     {regime}")
    print(f" Tradeable?         {result['tradeable']}")
    print("="*40)
    
    if ratio < 0.5:
        print(">>> TRICK DETECTED: Market is LYING. High predictability hidden in noise.")
    elif ratio > 0.8:
        print(">>> AVOID: Market is HONEST. This is pure random noise.")
    else:
        print(">>> TRANSITION: Waiting for coherence.")

if __name__ == "__main__":
    # Ensure we can find the modules folder
    sys.path.append(os.path.join(os.getcwd(), "ETARE_QuantumFusion"))
    main()
