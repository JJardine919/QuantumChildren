import sys
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to allow imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_adapter import BaseAdapter

class VolatilityAdapter(BaseAdapter):
    """
    Adapter for the Volatility Predictor system.
    Predicts if extreme volatility is expected in the near future (forward_window).
    """
    
    def __init__(self, forward_window=12, threshold=0.7):
        super().__init__("Volatility_Predictor")
        self.forward_window = forward_window
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def _calculate_features(self, df):
        d = df.copy()
        d["returns"] = d["close"].pct_change().fillna(0)
        d["true_range"] = np.maximum(
            d["high"] - d["low"],
            np.maximum(
                abs(d["high"] - d["close"].shift(1).fillna(d["high"])),
                abs(d["low"] - d["close"].shift(1).fillna(d["low"])),
            ),
        )
        
        # Lookback periods from original VolPredictor
        for period in [5, 10, 20]:
            d[f"atr_{period}"] = d["true_range"].rolling(window=period).mean().fillna(0)
            d[f"volatility_{period}"] = d["returns"].rolling(window=period).std().fillna(0)
            d[f"vol_change_{period}"] = d[f"volatility_{period}"] / d[f"volatility_{period}"].shift(1).replace(0, np.nan)
        
        d["parkinson_vol"] = np.sqrt(1 / (4 * np.log(2)) * np.power(np.log(d["high"] / d["low"]), 2))
        
        # Time features
        time = pd.to_datetime(d["time"], unit="s")
        d["hour_sin"] = np.sin(2 * np.pi * time.dt.hour / 24)
        d["hour_cos"] = np.cos(2 * np.pi * time.dt.hour / 24)
        
        feature_cols = [col for col in d.columns if any(x in col for x in ["atr_", "volatility_", "vol_change_", "parkinson", "hour_"])]
        features = d[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        return features

    def train_model(self, symbol, timeframe, n_candles=2000):
        """Trains the XGBoost model on the latest data."""
        if not mt5.initialize():
            return False
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
        if rates is None:
            return False
            
        df = pd.DataFrame(rates)
        features = self._calculate_features(df)
        
        # Create target: 1 if future vol > 75th percentile
        future_vol = df["close"].pct_change().rolling(window=self.forward_window).std().shift(-self.forward_window).fillna(0)
        vol_threshold = np.percentile(future_vol, 75)
        target = (future_vol > vol_threshold).astype(int)
        
        # Scale and Train
        X = self.scaler.fit_transform(features)
        self.model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        self.model.fit(X, target)
        self.is_trained = True
        return True

    def get_signal(self, symbol, timeframe, lookback=100):
        if not self.is_trained:
            print(f"Training Volatility model for {symbol}...")
            self.train_model(symbol, timeframe)
            
        if not mt5.initialize():
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "MT5 Error"}

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
        if rates is None:
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "No Data"}

        df = pd.DataFrame(rates)
        features = self._calculate_features(df)
        X = self.scaler.transform(features.tail(1))
        
        prob = self.model.predict_proba(X)[0][1]
        
        # Signal: 1.0 (High Vol Expected), -1.0 (Stable Expected)
        signal = 1.0 if prob > 0.5 else -1.0
        
        return {
            "name": self.name,
            "signal": signal,
            "confidence": prob if signal == 1.0 else (1.0 - prob),
            "metadata": {
                "high_vol_probability": prob,
                "threshold_alert": prob > self.threshold
            }
        }

if __name__ == "__main__":
    adapter = VolatilityAdapter()
    print("Testing Volatility Adapter...")
    res = adapter.get_signal("EURUSD", mt5.TIMEFRAME_H1)
    print(res)
    mt5.shutdown()
