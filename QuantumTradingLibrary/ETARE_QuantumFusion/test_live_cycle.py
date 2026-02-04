
import sys
import os
import pandas as pd
import MetaTrader5 as mt5

# Add paths
sys.path.append(os.path.abspath('ETARE_QuantumFusion'))
sys.path.append(os.path.abspath('ETARE_QuantumFusion/modules'))

from etare_fusion_trader import ETAREQuantumFusion

def test_single_cycle():
    print("\n" + "="*50)
    print("ETARE QUANTUM FUSION: LIVE CYCLE TEST")
    print("="*50)
    
    # We'll test with BTCUSD as it's highly volatile/active
    symbols = ["BTCUSD", "EURUSD", "GBPUSD"]
    
    trader = ETAREQuantumFusion(
        symbols=symbols,
        model_path="ETARE_QuantumFusion/models/fusion_champion_v1.pth"
    )
    
    print(f"MT5 Init: {mt5.initialize()}")
    
    print("\nExecuting analysis pipeline...")
    trader.analyze_and_trade()
    
    print("\n" + "="*50)
    print("TEST RESULTS:")
    print(f"Total Trades Attempted: {trader.stats['total_trades']}")
    print(f"Kill-Switch Activations: {trader.stats['kill_switch_activations']}")
    print(f"High-Confidence Boosts: {trader.stats['highly_compressible_boosts']}")
    print("="*50)
    
    mt5.shutdown()

if __name__ == "__main__":
    test_single_cycle()
