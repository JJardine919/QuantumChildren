import os
import yaml
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

# Import adapters
from qpe_adapter import QPEAdapter
from volatility_adapter import VolatilityAdapter
from lstm_adapter import LSTMAdapter
from bars_3d_adapter import Bars3DAdapter
from compression_layer import QuantumCompressionLayer

class SignalFusionEngine:
    """
    The Core 'Brain' of the Strike Boss Fusion Engine.
    Aggregates signals from multiple quantum systems based on configured weights.
    """
    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)
        self.compression_layer = QuantumCompressionLayer(
            fid_threshold=self.config['compression']['fid_threshold']
        )
        
        # Initialize adapters
        self.adapters = {
            "qpe": QPEAdapter(),
            "volatility": VolatilityAdapter(),
            "lstm": LSTMAdapter(),
            "bars_3d": Bars3DAdapter()
        }
        
        self.weights = self.config['fusion']['weights']

    def _load_config(self, path):
        # If path is already absolute, use it
        if os.path.isabs(path):
            full_path = path
        else:
            # Resolve relative to the project root (C:\Users\jjj10\QuantumTradingLibrary)
            # or relative to the script location if not found
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir) # ETARE_QuantumFusion
            repo_root = os.path.dirname(project_root) # QuantumTradingLibrary
            
            full_path = os.path.join(repo_root, path)
            if not os.path.exists(full_path):
                # Fallback to local project path
                full_path = os.path.join(project_root, os.path.basename(os.path.dirname(path)), os.path.basename(path))

        with open(full_path, 'r') as f:
            return yaml.safe_load(f)

    def get_fused_signal(self, symbol, timeframe):
        """
        Gathers signals from all systems and produces a weighted master signal.
        """
        print(f"\n[FUSION] Analyzing {symbol} on {timeframe}...")
        
        results = []
        weighted_sum = 0
        total_confidence = 0
        
        # 1. Check Regime via Compression
        if not mt5.initialize():
            return {"error": "MT5 Init Failed"}
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 256)
        if rates is not None:
            prices = pd.DataFrame(rates)['close'].values
            # Normalize for quantum layer
            norm_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-10)
            regime = self.compression_layer.analyze_regime(norm_prices.astype(complex))
            print(f"  [REGIME] {regime['regime']} (Ratio: {regime['ratio']:.2f})")
        else:
            regime = {"regime": "UNKNOWN", "ratio": 1.0}

        # 2. Collect Signals
        for key, adapter in self.adapters.items():
            try:
                # Map config weight names to adapter keys
                weight_key = self._map_key_to_weight(key)
                weight = self.weights.get(weight_key, 0.0)
                
                res = adapter.get_signal(symbol, timeframe)
                if "error" in res:
                    print(f"  [!] {res['name']} Error: {res['error']}")
                    continue
                
                # Signal: -1 to 1. Confidence: 0 to 1.
                # Weighted contribution = signal * confidence * weight
                contribution = res['signal'] * res['confidence'] * weight
                weighted_sum += contribution
                total_confidence += res['confidence'] * weight
                
                results.append(res)
                print(f"  [SIGNAL] {res['name']}: {res['signal']} (Conf: {res['confidence']:.2f}, Weight: {weight})")
            except Exception as e:
                print(f"  [!] Error in {key} adapter: {e}")

        # 3. Final Decision
        # Normalize weighted sum by the sum of weights actually used
        final_signal = weighted_sum / sum(self.weights.values())
        
        # Adjust behavior based on regime
        if regime['regime'] == "CHOPPY":
            print("  [ADVICE] Market is CHOPPY. Reducing confidence threshold.")
            final_signal *= 0.8 # De-risk in choppy markets
            
        decision = "NEUTRAL"
        if final_signal > 0.15: decision = "BUY"
        elif final_signal < -0.15: decision = "SELL"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "decision": decision,
            "composite_score": final_signal,
            "regime": regime,
            "individual_signals": results
        }

    def _map_key_to_weight(self, key):
        mapping = {
            "qpe": "qpe_analysis",
            "volatility": "volatility_predictor",
            "lstm": "quantum_lstm",
            "bars_3d": "quantum_3d"
        }
        return mapping.get(key, key)

if __name__ == "__main__":
    import numpy as np
    engine = SignalFusionEngine()
    # Test on EURUSD
    final = engine.get_fused_signal("EURUSD", mt5.TIMEFRAME_H1)
    print("\n" + "="*40)
    print(f"FINAL FUSION DECISION: {final['decision']}")
    print(f"COMPOSITE SCORE: {final['composite_score']:.4f}")
    print("="*40)
    mt5.shutdown()