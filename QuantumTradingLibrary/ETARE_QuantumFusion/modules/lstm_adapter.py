import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_adapter import BaseAdapter

# --- Copy of QuantumLSTM Architecture from quantum_lstm_system.py ---
class QuantumLSTM(nn.Module):
    def __init__(self, input_size=5, quantum_feature_size=7, hidden_size=128, num_layers=3, dropout=0.3):
        super(QuantumLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.quantum_processor = nn.Sequential(
            nn.Linear(quantum_feature_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, price_seq, quantum_features):
        lstm_out, _ = self.lstm(price_seq)
        lstm_last = lstm_out[:, -1, :]
        quantum_processed = self.quantum_processor(quantum_features)
        combined = torch.cat([lstm_last, quantum_processed], dim=1)
        output = self.fusion(combined)
        return output

class LSTMAdapter(BaseAdapter):
    """
    Adapter for the Bidirectional Quantum LSTM system.
    Uses pre-trained weights to predict next-bar direction.
    """
    
    def __init__(self, model_path="models/quantum_lstm_best.pth"):
        super().__init__("Quantum_LSTM")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = QuantumLSTM().to(self.device)
        
        # Resolve absolute path for model
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_model_path = os.path.join(project_root, model_path)
        
        if os.path.exists(full_model_path):
            self.model.load_state_dict(torch.load(full_model_path, map_location=self.device))
            self.model.eval()
            self.ready = True
        else:
            print(f"⚠️ LSTM Model not found at {full_model_path}")
            self.ready = False

        self.simulator = AerSimulator()

    def _extract_quantum_features(self, price_data):
        """Extracts 7 quantum features (logic from QuantumFeatureExtractor)"""
        # Simplified version for the adapter
        returns = np.diff(price_data) / (price_data[:-1] + 1e-10)
        feats = np.array([np.mean(returns), np.std(returns), np.max(returns) - np.min(returns)])
        feats = np.tanh(feats)
        
        qc = QuantumCircuit(3, 3)
        for i in range(3):
            qc.ry(np.clip(np.pi * feats[i % len(feats)], -2*np.pi, 2*np.pi), i)
        for i in range(2):
            qc.cx(i, i + 1)
        qc.measure(range(3), range(3))
        
        job = self.simulator.run(transpile(qc, self.simulator), shots=1000)
        counts = job.result().get_counts()
        
        # Probabilities
        probs = {state: count/1000 for state, count in counts.items()}
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs.values())
        dom_prob = max(probs.values())
        sig_states = sum(1 for p in probs.values() if p > 0.05)
        
        # Return 7 features as expected by the model
        return np.array([entropy, dom_prob, sig_states/8.0, 0.5, 0.5, 0.005, float(sig_states)])

    def get_signal(self, symbol, timeframe, lookback=100):
        if not self.ready:
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "Model Not Loaded"}

        if not mt5.initialize():
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "MT5 Error"}

        # Need 50 bars for sequence + extra for features
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
        if rates is None or len(rates) < 50:
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "Insufficient Data"}

        df = pd.DataFrame(rates)
        
        # 1. Price Features (5 inputs: returns, log_returns, high_low, close_open, tick_volume)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low'] = (df['high'] - df['low']) / df['close']
        df['close_open'] = (df['close'] - df['open']) / df['open']
        df = df.dropna()
        
        price_features = df[['returns', 'log_returns', 'high_low', 'close_open', 'tick_volume']].tail(50).values
        # Standardize (using simple local standardization as in the original live code)
        mean = price_features.mean(axis=0)
        std = price_features.std(axis=0)
        price_data = (price_features - mean) / (std + 1e-8)
        
        # 2. Quantum Features
        q_feats = self._extract_quantum_features(df['close'].tail(50).values)
        
        # Predict
        with torch.no_grad():
            p_tensor = torch.FloatTensor(price_data).unsqueeze(0).to(self.device)
            q_tensor = torch.FloatTensor(q_feats).unsqueeze(0).to(self.device)
            output = self.model(p_tensor, q_tensor)
            prob = torch.sigmoid(output).item()
            
        signal = 1.0 if prob > 0.5 else -1.0
        confidence = prob if signal == 1.0 else (1.0 - prob)
        
        return {
            "name": self.name,
            "signal": signal,
            "confidence": confidence,
            "metadata": {
                "probability": prob,
                "quantum_entropy": q_feats[0]
            }
        }

if __name__ == "__main__":
    adapter = LSTMAdapter()
    if adapter.ready:
        print("Testing LSTM Adapter...")
        res = adapter.get_signal("EURUSD", mt5.TIMEFRAME_H1)
        print(res)
    mt5.shutdown()
