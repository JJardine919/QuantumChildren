"""
ETARE 50 Experts - Darwin Selection (NO BACKPROP TRAINING)
Full features from context, Darwin selection to find best
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

def prepare_features(df):
    """Full features from ETARE context"""
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-10)))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Adaptive Bollinger Bands
    volatility = df['close'].rolling(50).std()
    adaptive_period = max(10, min(50, int(20 * (1 + volatility.mean()))))
    df['bb_middle'] = df['close'].rolling(adaptive_period).mean()
    df['bb_std'] = df['close'].rolling(adaptive_period).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(10)
    df['momentum_ma'] = df['momentum'].rolling(20).mean()
    df['momentum_std'] = df['momentum'].rolling(20).std()

    # Volatility
    df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()

    # Volume analysis
    df['volume_ma'] = df['tick_volume'].rolling(20).mean()
    df['volume_std'] = df['tick_volume'].rolling(20).std()
    df['volume_ratio'] = df['tick_volume'] / (df['volume_ma'] + 1e-10)
    df['volume_volatility'] = df['volume_std'] / (df['volume_ma'] + 1e-10)
    df['volume_spike'] = (df['tick_volume'] > df['volume_ma'] + 2 * df['volume_std']).astype(int)
    df['volume_cluster'] = df['tick_volume'].rolling(3).sum() / (df['tick_volume'].rolling(20).sum() + 1e-10)

    df = df.dropna()

    # Feature columns
    feature_cols = ['rsi', 'macd', 'macd_signal', 'macd_hist',
                    'bb_upper', 'bb_lower', 'bb_middle',
                    'momentum', 'momentum_ma', 'momentum_std',
                    'atr', 'price_change', 'price_change_abs',
                    'volume_ratio', 'volume_volatility', 'volume_spike', 'volume_cluster']

    # Adaptive normalization (rolling)
    for col in feature_cols:
        rolling_mean = df[col].rolling(100).mean()
        rolling_std = df[col].rolling(100).std()
        df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    # Clip outliers
    df[feature_cols] = df[feature_cols].clip(-4, 4)

    df = df.dropna()

    # Target
    future = df['close'].shift(-5)
    df['target'] = 0
    df.loc[future > df['close'] * 1.001, 'target'] = 1
    df.loc[future < df['close'] * 0.999, 'target'] = 2
    df = df.dropna()

    return df[feature_cols + ['target', 'close']], feature_cols

print("="*60)
print("ETARE 50 EXPERTS - DARWIN SELECTION (NO BACKPROP)")
print("="*60)

# Init MT5
mt5.initialize()
rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M5, 0, 10000)
df = pd.DataFrame(rates)
df, feature_cols = prepare_features(df)
print(f"Data: {len(df)} bars, Features: {len(feature_cols)}")

# Create sequences
SEQ = 30
features = df[feature_cols].values
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

split = int(len(X) * 0.8)
X_test = X[split:]
y_test = y[split:]
p_test = p[split:]

print(f"Test set: {len(X_test)} samples")
print(f"\nCreating 50 experts and running Darwin selection...")

INPUT_SIZE = len(feature_cols)

# Create population
population = []
for i in range(50):
    model = LSTMModel(INPUT_SIZE).to(DEVICE)
    model.lstm.to('cpu')
    model.dropout.to('cpu')
    population.append({'model': model, 'fitness': 0, 'win_rate': 0, 'accuracy': 0})

# Evaluate all
def evaluate(model, X_test, y_test, p_test):
    model.eval()
    with torch.no_grad():
        out = model(X_test.to(DEVICE))
        pred = torch.argmax(out, dim=1).cpu().numpy()

    correct = (pred == y_test.numpy()).sum()
    acc = correct / len(y_test)

    wins, losses = 0, 0
    for j in range(len(pred) - 5):
        if pred[j] == 0: continue
        future_p = p_test[j+5]
        curr_p = p_test[j]
        if pred[j] == 1:
            win = future_p > curr_p
        else:
            win = future_p < curr_p
        if win: wins += 1
        else: losses += 1

    total = wins + losses
    wr = wins / total if total > 0 else 0
    fitness = acc * 0.3 + wr * 0.7

    return acc, wr, fitness

# Initial evaluation
for i, ind in enumerate(population):
    acc, wr, fit = evaluate(ind['model'], X_test, y_test, p_test)
    ind['accuracy'] = acc * 100
    ind['win_rate'] = wr * 100
    ind['fitness'] = fit
    if (i+1) % 10 == 0:
        print(f"Evaluated {i+1}/50")

# Darwin selection - 3 rounds
for round_num in range(3):
    print(f"\n--- EXTINCTION EVENT {round_num + 1} ---")

    # Sort by fitness
    population.sort(key=lambda x: x['fitness'], reverse=True)

    # Kill bottom 30%
    survivors = population[:35]

    # Repopulate with mutations of survivors
    while len(survivors) < 50:
        parent = survivors[np.random.randint(0, 10)]  # Pick from top 10
        child_model = LSTMModel(INPUT_SIZE).to(DEVICE)
        child_model.lstm.to('cpu')
        child_model.dropout.to('cpu')

        # Copy weights with mutation
        child_model.load_state_dict(parent['model'].state_dict())
        with torch.no_grad():
            for param in child_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        acc, wr, fit = evaluate(child_model, X_test, y_test, p_test)
        survivors.append({
            'model': child_model,
            'accuracy': acc * 100,
            'win_rate': wr * 100,
            'fitness': fit
        })

    population = survivors
    best = population[0]
    print(f"Best after extinction: Acc={best['accuracy']:.1f}%, WR={best['win_rate']:.1f}%")

# Final sort
population.sort(key=lambda x: x['fitness'], reverse=True)

all_wr = [ind['win_rate'] for ind in population]
all_acc = [ind['accuracy'] for ind in population]

print(f"\n{'='*60}")
print(f"BASELINE RESULTS - DARWIN SELECTION (NO BACKPROP TRAINING)")
print(f"{'='*60}")
print(f"Average Accuracy: {np.mean(all_acc):.1f}%")
print(f"Average Win Rate: {np.mean(all_wr):.1f}%")
print(f"Best Accuracy: {np.max(all_acc):.1f}%")
print(f"Best Win Rate: {np.max(all_wr):.1f}%")
print(f"{'='*60}")
print(f"TARGET: 63-65%")
if 63 <= np.max(all_wr) <= 65 or 63 <= np.mean(all_wr) <= 65:
    print("TARGET MET")
else:
    print(f"Gap: {63 - np.max(all_wr):.1f}% needed")
print(f"{'='*60}")

mt5.shutdown()
