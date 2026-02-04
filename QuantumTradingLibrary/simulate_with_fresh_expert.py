"""
Simulate BlueGuardian with Fresh LSTM Expert
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# LSTM Model (same architecture)
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load fresh expert
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleLSTM().to(device)
model.load_state_dict(torch.load('fresh_btc_expert.pt', map_location=device))
model.eval()

print("="*60)
print("BLUEGUARDIAN + FRESH LSTM EXPERT SIMULATION")
print("="*60)

# Initialize MT5
if not mt5.initialize():
    print("MT5 init failed")
    exit()

# Settings
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOT_SIZE = 0.06
MAX_LOSS_DOLLARS = 1.00
TP_MULTIPLIER = 3.0
CONFIDENCE_THRESHOLD = 0.60  # Higher = fewer but better trades
SEQ_LEN = 50

# Get data
rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1500)  # 25 hours
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Feature engineering
for c in ['open', 'high', 'low', 'close', 'tick_volume']:
    df[c] = df[c].astype(float)

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / (loss + 1e-8)
df['rsi'] = 100 - (100 / (1 + rs))

# MACD
exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = exp1 - exp2
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# Bollinger
df['bb_middle'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

# Momentum
df['momentum'] = df['close'] / df['close'].shift(10)
df['roc'] = df['close'].pct_change(10) * 100

# ATR
df['tr'] = np.maximum(df['high'] - df['low'],
    np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
df['atr'] = df['tr'].rolling(14).mean()

feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
df = df.dropna()

# Normalize
for col in feature_cols:
    col_mean = df[col].mean()
    col_std = df[col].std() + 1e-8
    df[col] = (df[col] - col_mean) / col_std

symbol_info = mt5.symbol_info(SYMBOL)
tick_value = symbol_info.trade_tick_value
tick_size = symbol_info.trade_tick_size
sl_ticks = MAX_LOSS_DOLLARS / (tick_value * LOT_SIZE)
sl_distance = sl_ticks * tick_size

print(f"Symbol: {SYMBOL}")
print(f"Max Loss: ${MAX_LOSS_DOLLARS} | TP: ${MAX_LOSS_DOLLARS * TP_MULTIPLIER}")
print(f"SL Distance: ${sl_distance:.2f}")
print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
print(f"Simulating {len(df)} bars...")
print("="*60)

mt5.shutdown()

# Simulation
trades = []
position = None
balance = 5000.00
starting_balance = balance

for i in range(SEQ_LEN, len(df) - 10):  # Leave room for future price check
    row = df.iloc[i]
    price = row['close']
    high = row['high']
    low = row['low']

    # Get LSTM prediction
    seq = df[feature_cols].iloc[i-SEQ_LEN:i].values
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(seq_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0, pred].item()

    # Map prediction: 0=BUY, 1=HOLD, 2=SELL
    if pred == 0:
        signal = 'BUY'
    elif pred == 2:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    # Manage existing position
    if position:
        closed = False
        pnl = 0
        reason = ""

        if position['type'] == 'BUY':
            if low <= position['sl']:
                pnl = -MAX_LOSS_DOLLARS
                closed = True
                reason = "SL HIT"
            elif high >= position['tp']:
                pnl = MAX_LOSS_DOLLARS * TP_MULTIPLIER
                closed = True
                reason = "TP HIT"
            else:
                # Trailing at 80%
                progress = (price - position['entry']) / (position['tp'] - position['entry'])
                if progress >= 0.8:
                    trail_sl = price - (sl_distance * 0.3)
                    if trail_sl > position['sl']:
                        position['sl'] = trail_sl
        else:  # SELL
            if high >= position['sl']:
                pnl = -MAX_LOSS_DOLLARS
                closed = True
                reason = "SL HIT"
            elif low <= position['tp']:
                pnl = MAX_LOSS_DOLLARS * TP_MULTIPLIER
                closed = True
                reason = "TP HIT"
            else:
                progress = (position['entry'] - price) / (position['entry'] - position['tp'])
                if progress >= 0.8:
                    trail_sl = price + (sl_distance * 0.3)
                    if trail_sl < position['sl']:
                        position['sl'] = trail_sl

        if closed:
            balance += pnl
            trades.append({
                'time': row['time'],
                'type': position['type'],
                'entry': position['entry'],
                'pnl': pnl,
                'reason': reason,
                'balance': balance,
                'confidence': position['confidence']
            })
            position = None

    # Open new position if confident signal and no position
    if position is None and confidence >= CONFIDENCE_THRESHOLD and signal != 'HOLD':
        if signal == 'BUY':
            position = {
                'type': 'BUY',
                'entry': price,
                'sl': price - sl_distance,
                'tp': price + (sl_distance * TP_MULTIPLIER),
                'confidence': confidence
            }
        elif signal == 'SELL':
            position = {
                'type': 'SELL',
                'entry': price,
                'sl': price + sl_distance,
                'tp': price - (sl_distance * TP_MULTIPLIER),
                'confidence': confidence
            }

# Results
print("\n" + "="*60)
print("SIMULATION RESULTS")
print("="*60)

if trades:
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]

    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / len(trades) * 100
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0

    print(f"Total Trades: {len(trades)}")
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
    print(f"")
    print(f"Starting Balance: ${starting_balance:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Return: {(total_pnl/starting_balance)*100:.2f}%")

    # Drawdown
    peak = starting_balance
    max_dd = 0
    for t in trades:
        if t['balance'] > peak:
            peak = t['balance']
        dd = (peak - t['balance']) / starting_balance * 100
        if dd > max_dd:
            max_dd = dd

    print(f"\n" + "="*60)
    print("PROP FIRM CHECK")
    print("="*60)
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Daily Loss (5%): {'PASS' if max_dd < 5 else 'FAIL'}")
    print(f"Total DD (10%): {'PASS' if max_dd < 10 else 'FAIL'}")

    profit_pct = total_pnl / starting_balance * 100
    print(f"Profit: {profit_pct:.2f}%")
    if profit_pct >= 10:
        print("PROFIT TARGET: PASS")
    else:
        need = starting_balance * 0.10 - total_pnl
        print(f"PROFIT TARGET: Need ${need:.2f} more")

    # Expected value
    ev = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
    print(f"\nExpected Value per trade: ${ev:.2f}")

    if ev > 0:
        print("\n*** STRATEGY IS PROFITABLE - READY TO DEPLOY ***")
    else:
        print("\n*** STRATEGY NEEDS MORE WORK ***")

    # Recent trades
    print(f"\n" + "="*60)
    print("RECENT TRADES (last 10)")
    print("="*60)
    for t in trades[-10:]:
        emoji = "+" if t['pnl'] > 0 else ""
        print(f"{t['time'].strftime('%H:%M')} | {t['type']:4} | Conf: {t['confidence']:.2f} | {emoji}${t['pnl']:.2f} | {t['reason']}")

else:
    print("No trades executed")

print("\n" + "="*60)
