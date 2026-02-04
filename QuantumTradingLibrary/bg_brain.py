"""
ETARE Redux Signal Generator (Blue Guardian Edition)
====================================================
Dedicated Brain for Account 365060.
Operating Mode: LINUX DOCKER (File-Based Data)

Logic:
1. Read market_data.json (Written by MT5 Service)
2. Predict using LSTM
3. Write etare_signals.json (Read by MT5 Executor)
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
DATA_FILE = Path("/mt5files/market_data.json")  # Mapped Volume path
SIGNAL_FILE = Path("/mt5files/etare_signals.json")
SYMBOLS = ["BTCUSD"]
SEQ_LENGTH = 30
DEVICE = torch.device("cpu") # Stability

# ============================================================================
# LSTM MODEL
# ============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# ============================================================================
# DATA HANDLING
# ============================================================================
def load_market_data(symbol, lookback=100):
    """Read JSON data exported from MT5"""
    if not DATA_FILE.exists():
        return None
    
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        if symbol not in data: return None
        
        # Convert to DataFrame
        raw = data[symbol][-lookback:]
        df = pd.DataFrame(raw)
        
        # Ensure numeric
        cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for c in cols: df[c] = pd.to_numeric(df[c])
        
        return df
    except Exception as e:
        print(f"[ERROR] Reading data: {e}")
        return None

def prepare_features(df):
    """Extract Technical Indicators"""
    df = df.copy()
    
    # 1. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 2. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 3. Bollinger
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2*df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2*df['bb_std']

    # 4. Momentum/ROC
    df['momentum'] = df['close'] / df['close'].shift(10)
    df['roc'] = df['close'].pct_change(10) * 100

    # 5. ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    df = df.dropna()
    if df.empty: return None

    # Normalize
    features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
    for c in features:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)
        
    return df[features]

def get_prediction(model, df):
    if len(df) < SEQ_LENGTH: return "HOLD", 0.0
    
    seq = df.iloc[-SEQ_LENGTH:].values
    tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        
        # Aggression Tuner (Reduce HOLD)
        probs[0][0] *= 0.4 
        probs = probs / probs.sum()
        
        action_idx = torch.argmax(probs).item()
        conf = probs[0][action_idx].item()
        
    actions = {0: "HOLD", 1: "BUY", 2: "SELL"}
    return actions[action_idx], conf

# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    print("="*60)
    print("ETARE REDUX BRAIN - BLUE GUARDIAN EDITION")
    print("Mode: REAL DATA (File-Based)")
    print("="*60)

    # Load Model
    model_path = Path("/app/champions/champion_BTCUSD.pth")
    if not model_path.exists():
        print(f"[FATAL] Champion model not found at {model_path}")
        sys.exit(1)
        
    state = torch.load(model_path, map_location=DEVICE)
    model = LSTMModel(input_size=8, hidden_size=state['lstm.weight_ih_l0'].shape[0]//4)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("[OK] Champion BTCUSD Loaded")

    while True:
        try:
            # 1. Read Data
            df = load_market_data("BTCUSD")
            
            if df is None:
                print(f"[WAIT] Waiting for market_data.json update...")
                time.sleep(5)
                continue
                
            # 2. Predict
            feat = prepare_features(df)
            if feat is not None:
                action, conf = get_prediction(model, feat)
                print(f"[SIGNAL] BTCUSD: {action} ({conf:.1%})")
                
                # 3. Export
                output = {
                    "BTCUSD": {
                        "action": action,
                        "confidence": conf,
                        "timestamp": datetime.now().isoformat()
                    },
                    "_meta": {"engine": "BlueGuardian_Redux", "status": "ACTIVE"}
                }
                
                with open(SIGNAL_FILE, 'w') as f:
                    json.dump(output, f)
            
            time.sleep(10) # 10s cycle
            
        except Exception as e:
            print(f"[ERROR] Cycle failed: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
