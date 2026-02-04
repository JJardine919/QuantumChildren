print("--- STARTING SIMULATION SCRIPT ---", flush=True)

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================

ACCOUNT_SIZE = 150000.0
DAILY_LOSS_LIMIT = 3750.0
MAX_DRAWDOWN_LIMIT = 6000.0
SPREAD_COST = 10.0
COMMISSION = 3.50
LOT_SIZE = 5.0  # 5 BTC
DEVICE = torch.device("cpu")
SEQ_LENGTH = 30

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

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Simple feature extraction for brevity and to avoid errors
    # (Assuming the original function was correct, but simplifying just in case)
    for c in ['open', 'high', 'low', 'close', 'tick_volume']:
        df[c] = df[c].astype(float)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # BB
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # Momentum/ROC
    df['momentum'] = df['close'] / df['close'].shift(10)
    df['roc'] = df['close'].pct_change(10) * 100

    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    df = df.dropna()
    if df.empty: return None

    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df[col] = (df[col] - mean) / std

    return df[feature_cols]

def create_sequence(df: pd.DataFrame) -> torch.Tensor:
    if df is None or len(df) < SEQ_LENGTH: return None
    seq = df.iloc[-SEQ_LENGTH:].values
    return torch.FloatTensor(seq).unsqueeze(0)

# ============================================================================
# MAIN
# ============================================================================

def run_simulation():
    print("=" * 60, flush=True)
    print("TRADEIFY SIMULATION", flush=True)
    print("=" * 60, flush=True)

    if not mt5.initialize():
        print("MT5 Init Failed", flush=True)
        return

    symbol = "BTCUSD"
    if not mt5.symbol_select(symbol, True):
        # Fallback search
        for s in mt5.symbols_get():
            if "BTC" in s.name and "USD" in s.name:
                symbol = s.name
                break
    
    print(f"Symbol: {symbol}", flush=True)

    # Load 1000 bars
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1000)
    if rates is None:
        print("No Data", flush=True)
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Loaded {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}", flush=True)

    # Load Model
    model_path = "champions/champion_BTCUSD.pth"
    if not os.path.exists(model_path):
        print("Model not found", flush=True)
        return

    state_dict = torch.load(model_path, map_location=DEVICE)
    input_size = state_dict['lstm.weight_ih_l0'].shape[1]
    hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
    model = LSTMModel(input_size, hidden_size)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("Model Loaded", flush=True)

    # Simulation Variables
    balance = ACCOUNT_SIZE
    equity = ACCOUNT_SIZE
    current_day_start_equity = ACCOUNT_SIZE
    current_day = df['time'].iloc[0].date()
    
    position = 0
    entry_price = 0.0
    trades = []

    full_features = prepare_features(df)
    start_idx = len(df) - len(full_features)

    print("Running Loop...", flush=True)

    for i in range(SEQ_LENGTH, len(full_features)):
        bar_idx = start_idx + i
        current_time = df['time'].iloc[bar_idx]
        current_price = df['close'].iloc[bar_idx]

        # Reset Daily Loss
        if current_time.date() > current_day:
            current_day = current_time.date()
            current_day_start_equity = equity

        # Check Limits
        if (current_day_start_equity - equity) > DAILY_LOSS_LIMIT:
            print(f"FAIL: Daily Loss at {current_time}", flush=True)
            break
        if (ACCOUNT_SIZE - equity) > MAX_DRAWDOWN_LIMIT:
            print(f"FAIL: Max DD at {current_time}", flush=True)
            break

        # Inference
        seq_data = full_features.iloc[i-SEQ_LENGTH:i]
        seq = create_sequence(seq_data).to(DEVICE)
        with torch.no_grad():
            output = model(seq)
            probs = torch.softmax(output, dim=1)
            probs[0][0] *= 0.4
            action = torch.argmax(probs).item()

        # Trade Execution
        # 1=BUY, 2=SELL
        if action == 1:
            if position == -1: # Close Short
                pnl = (entry_price - current_price) * LOT_SIZE
                pnl -= (SPREAD_COST + COMMISSION*2) * LOT_SIZE
                balance += pnl
                trades.append(pnl)
                position = 0
            if position == 0: # Open Long
                entry_price = current_price
                position = 1
        elif action == 2:
            if position == 1: # Close Long
                pnl = (current_price - entry_price) * LOT_SIZE
                pnl -= (SPREAD_COST + COMMISSION*2) * LOT_SIZE
                balance += pnl
                trades.append(pnl)
                position = 0
            if position == 0: # Open Short
                entry_price = current_price
                position = -1

        # Update Equity
        floating = 0
        if position == 1: floating = (current_price - entry_price) * LOT_SIZE
        elif position == -1: floating = (entry_price - current_price) * LOT_SIZE
        equity = balance + floating

    # Final Report
    total_pnl = equity - ACCOUNT_SIZE
    print(f"\nRESULTS:", flush=True)
    print(f"Start: ${ACCOUNT_SIZE:,.2f}", flush=True)
    print(f"End:   ${equity:,.2f}", flush=True)
    print(f"PnL:   ${total_pnl:,.2f}", flush=True)
    print(f"Trades: {len(trades)}", flush=True)
    
    if total_pnl > 0:
        print("PROFITABLE", flush=True)
    else:
        print("LOSS", flush=True)

    mt5.shutdown()

if __name__ == "__main__":
    run_simulation()
