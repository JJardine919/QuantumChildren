import sys
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not installed. 3D Bars Adapter will be disabled.")

from sklearn.preprocessing import MinMaxScaler

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_adapter import BaseAdapter

class Bars3D:
    """Simplified 3D Bars logic from quantum_3d_system.py"""
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(3, 9))
        
    def create_features(self, df):
        d = df.copy()
        d['typical_price'] = (d['high'] + d['low'] + d['close']) / 3
        d['price_return'] = d['typical_price'].pct_change().fillna(0)
        d['volatility'] = d['price_return'].rolling(20).std().fillna(0)
        d['volume_change'] = d['tick_volume'].pct_change().fillna(0)
        
        # 3D Metrics
        d['bar3d_yellow_cluster'] = ((d['volatility'] > d['volatility'].quantile(0.7)) & 
                                    (d['volume_change'].abs() > d['volume_change'].abs().quantile(0.7))).astype(float)
        return d

class Bars3DAdapter(BaseAdapter):
    """
    Adapter for the 3D Bars system.
    Uses CatBoost to classify market direction based on multidimensional features.
    """
    
    def __init__(self, model_path="models/catboost_quantum_3d.cbm"):
        super().__init__("Quantum_3D_Bars")
        if CATBOOST_AVAILABLE:
            self.model = CatBoostClassifier()
        else:
            self.model = None
            
        self.bars3d = Bars3D()
        self.is_trained = False
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.full_model_path = os.path.join(project_root, model_path)
        
        if CATBOOST_AVAILABLE and os.path.exists(self.full_model_path):
            self.model.load_model(self.full_model_path)
            self.is_trained = True

    def train_if_needed(self, symbol, timeframe):
        """Trains a quick CatBoost model if none exists."""
        if not CATBOOST_AVAILABLE or self.is_trained: return
        
        print(f"Training 3D Bars model for {symbol}...")
        if not mt5.initialize(): return
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1000)
        if rates is None: return
        
        df = pd.DataFrame(rates)
        df_3d = self.bars3d.create_features(df)
        
        # Features for CatBoost
        features = df_3d[['price_return', 'volatility', 'volume_change', 'bar3d_yellow_cluster']].fillna(0)
        target = (df['close'].shift(-1) > df['close']).astype(int).iloc[:-1]
        X = features.iloc[:-1]
        
        self.model.fit(X, target, verbose=False)
        self.model.save_model(self.full_model_path)
        self.is_trained = True

    def get_signal(self, symbol, timeframe, lookback=100):
        if not CATBOOST_AVAILABLE:
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "CatBoost Not Installed"}
            
        self.train_if_needed(symbol, timeframe)
        
        if not mt5.initialize():
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "MT5 Error"}

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
        if rates is None:
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "No Data"}

        df = pd.DataFrame(rates)
        df_3d = self.bars3d.create_features(df)
        X = df_3d[['price_return', 'volatility', 'volume_change', 'bar3d_yellow_cluster']].tail(1)
        
        proba = self.model.predict_proba(X)[0]
        prob_up = proba[1]
        
        signal = 1.0 if prob_up > 0.5 else -1.0
        confidence = prob_up if signal == 1.0 else (1.0 - prob_up)
        
        return {
            "name": self.name,
            "signal": signal,
            "confidence": confidence,
            "metadata": {
                "yellow_cluster": float(df_3d['bar3d_yellow_cluster'].iloc[-1]),
                "prob_up": prob_up
            }
        }

if __name__ == "__main__":
    adapter = Bars3DAdapter()
    print("Testing 3D Bars Adapter...")
    res = adapter.get_signal("EURUSD", mt5.TIMEFRAME_H1)
    print(res)
    mt5.shutdown()
