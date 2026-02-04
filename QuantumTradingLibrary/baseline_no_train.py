"""
ETARE 50 Experts - Baseline WIN RATE (NO TRAINING)
Test the raw build from context
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import MetaTrader5 as mt5

try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"GPU: {torch_directml.device_name(0)}")
except:
    DEVICE = torch.device("cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x_cpu = x.cpu() if x.device.type != 'cpu' else x
        out, _ = self.lstm(x_cpu)
        out = self.dropout(out[:, -1, :])
        out = out.to(x.device) if x.device.type != 'cpu' else out
        return self.fc(out)

def prepare_data(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-10)))
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2*df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2*df['bb_std']
    df['momentum'] = df['close'] / df['close'].shift(10)
    df['roc'] = df['close'].pct_change(10) * 100
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df = df.dropna()

    cols = ['rsi','macd','macd_signal','bb_upper','bb_lower','momentum','roc','atr']
    for c in cols:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

    future = df['close'].shift(-5)
    df['target'] = 0
    df.loc[future > df['close']*1.001, 'target'] = 1
    df.loc[future < df['close']*0.999, 'target'] = 2
    df = df.dropna()
    return df[cols + ['target', 'close']]

print("="*60)
print("ETARE 50 EXPERTS - BASELINE (NO TRAINING)")
print("="*60)

# Init MT5
mt5.initialize()
rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M5, 0, 5000)
df = pd.DataFrame(rates)
df = prepare_data(df)
print(f"Data: {len(df)} bars")

# Create sequences
SEQ = 30
features = df[['rsi','macd','macd_signal','bb_upper','bb_lower','momentum','roc','atr']].values
targets = df['target'].values
prices = df['close'].values

X, y, p = [], [], []
for i in range(len(features) - SEQ):
    X.append(features[i:i+SEQ])
    y.append(targets[i+SEQ])
    p.append(prices[i+SEQ])

X = torch.FloatTensor(np.array(X))
y = torch.LongTensor(np.array(y))
p = np.array(p)

# Use last 20% for testing
split = int(len(X) * 0.8)
X_test = X[split:]
y_test = y[split:]
p_test = p[split:]

print(f"Test set: {len(X_test)} samples")
print(f"\nCreating 50 experts (NO TRAINING)...")

all_wr = []
all_acc = []

for i in range(50):
    # Create fresh model - random weights, NO training
    model = LSTMModel(8).to(DEVICE)
    model.lstm.to('cpu')
    model.dropout.to('cpu')
    model.eval()

    # Test win rate
    with torch.no_grad():
        out = model(X_test.to(DEVICE))
        pred = torch.argmax(out, dim=1).cpu().numpy()

    # Calculate accuracy
    correct = (pred == y_test.numpy()).sum()
    acc = correct / len(y_test) * 100

    # Calculate win rate (trading simulation)
    wins, losses = 0, 0
    for j in range(len(pred) - 5):
        if pred[j] == 0: continue  # Skip HOLD
        future_p = p_test[j+5]
        curr_p = p_test[j]
        if pred[j] == 1:  # BUY
            win = future_p > curr_p
        else:  # SELL
            win = future_p < curr_p
        if win: wins += 1
        else: losses += 1

    total_trades = wins + losses
    wr = wins / total_trades * 100 if total_trades > 0 else 0

    all_wr.append(wr)
    all_acc.append(acc)

    if (i+1) % 10 == 0:
        print(f"Expert {i+1}/50 - Acc: {acc:.1f}% - WR: {wr:.1f}% - Trades: {total_trades}")

print(f"\n{'='*60}")
print(f"BASELINE RESULTS (50 EXPERTS - NO TRAINING)")
print(f"{'='*60}")
print(f"Average Accuracy: {np.mean(all_acc):.1f}%")
print(f"Average Win Rate: {np.mean(all_wr):.1f}%")
print(f"Best Accuracy: {np.max(all_acc):.1f}%")
print(f"Best Win Rate: {np.max(all_wr):.1f}%")
print(f"{'='*60}")
print(f"TARGET: 63-65%")
print(f"{'='*60}")

mt5.shutdown()
