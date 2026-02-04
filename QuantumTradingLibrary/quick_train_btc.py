"""
Quick Train - Pull recent BTC data and train a fresh LSTM expert
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import os

print("="*60)
print("QUICK TRAIN - FRESH BTC EXPERT")
print("="*60)

# Initialize MT5
if not mt5.initialize():
    print("MT5 init failed")
    exit()

# Get recent data - last 3 days of M1 data (4320 bars)
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
BARS = 4320  # 3 days of M1

print(f"\nPulling {BARS} bars of {SYMBOL} M1 data...")

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
if rates is None or len(rates) < 1000:
    print(f"Failed to get data: {mt5.last_error()}")
    mt5.shutdown()
    exit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
print(f"Got {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

mt5.shutdown()

# Feature Engineering
print("\nEngineering features...")

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

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

# Momentum & ROC
df['momentum'] = df['close'] / df['close'].shift(10)
df['roc'] = df['close'].pct_change(10) * 100

# ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr'] = df['tr'].rolling(14).mean()

# Labels - predict if price goes up/down/flat in next 10 bars
df['future_return'] = df['close'].shift(-10) / df['close'] - 1
df['label'] = 1  # HOLD
df.loc[df['future_return'] > 0.001, 'label'] = 0  # BUY (price goes up)
df.loc[df['future_return'] < -0.001, 'label'] = 2  # SELL (price goes down)

# Clean data
feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
df = df.dropna()
df = df[:-10]  # Remove last 10 rows (no labels)

print(f"Training samples: {len(df)}")
print(f"Label distribution: BUY={sum(df['label']==0)}, HOLD={sum(df['label']==1)}, SELL={sum(df['label']==2)}")

# Normalize features
for col in feature_cols:
    col_mean = df[col].mean()
    col_std = df[col].std() + 1e-8
    df[col] = (df[col] - col_mean) / col_std

# Create sequences
SEQ_LEN = 50
X, y = [], []

for i in range(SEQ_LEN, len(df)):
    seq = df[feature_cols].iloc[i-SEQ_LEN:i].values
    label = df['label'].iloc[i]
    X.append(seq)
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print(f"Sequences created: {len(X)}")

# Split train/val
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Create DataLoaders
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# LSTM Model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining on: {device}")

model = SimpleLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 20
print(f"\nTraining for {EPOCHS} epochs...")

best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    val_acc = correct / total * 100
    if val_acc > best_val_acc:
        best_val_acc = val_acc

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.1f}%")

print(f"\nBest Validation Accuracy: {best_val_acc:.1f}%")

# Save model
save_path = os.path.join(os.path.dirname(__file__), 'fresh_btc_expert.pt')
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to: {save_path}")

# Quick backtest on validation data
print("\n" + "="*60)
print("QUICK BACKTEST ON VALIDATION DATA")
print("="*60)

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(predicted)):
            pred = predicted[i].item()
            actual = batch_y[i].item()
            conf = probs[i, pred].item()

            if conf >= 0.48:  # Confidence threshold
                predictions.append(pred)
                actuals.append(actual)

if predictions:
    # Simulate trades
    balance = 5000
    wins = 0
    losses = 0

    for pred, actual in zip(predictions, actuals):
        if pred == 1:  # HOLD - no trade
            continue

        # BUY prediction
        if pred == 0:
            if actual == 0:  # Correct - price went up
                balance += 3  # TP hit
                wins += 1
            else:
                balance -= 1  # SL hit
                losses += 1

        # SELL prediction
        elif pred == 2:
            if actual == 2:  # Correct - price went down
                balance += 3
                wins += 1
            else:
                balance -= 1
                losses += 1

    total_trades = wins + losses
    if total_trades > 0:
        win_rate = wins / total_trades * 100
        pnl = balance - 5000

        print(f"Total Trades: {total_trades}")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"P&L: ${pnl:.2f}")
        print(f"Final Balance: ${balance:.2f}")

        # Expected outcome
        expected = (win_rate/100 * 3) - ((100-win_rate)/100 * 1)
        print(f"\nExpected $ per trade: ${expected:.2f}")
        if expected > 0:
            print("STRATEGY IS PROFITABLE")
        else:
            print("STRATEGY NEEDS IMPROVEMENT")
    else:
        print("No trades taken (all HOLD)")
else:
    print("No confident predictions made")

print("\n" + "="*60)
