"""
LSTM EXPERT RETRAINER - FAST VERSION
=====================================
Optimized: Top 10 experts only, 5000 bars (matches ETARE_50_Darwin.py),
faster evolution cycles.

Target: 88% win rate
"""
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from pathlib import Path
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
SCRIPT_DIR = Path(__file__).parent.absolute()

SEQ_LENGTH = 30
PREDICTION_HORIZON = 5
POP_SIZE = 10
ELITE = 3
EPOCHS = 30
BATCH_SIZE = 64


class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_cpu = x.cpu() if x.device != torch.device('cpu') else x
        out, _ = self.lstm(x_cpu)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def get_data(symbol='BTCUSD', bars=5000):
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
            mt5.shutdown()
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                print(f"  Got {len(df):,} bars from MT5 ({symbol})")
                return df
    except:
        pass
    raise RuntimeError(f"Could not get data for {symbol}")


def prepare_features(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std

    df['momentum'] = df['close'] / df['close'].shift(10)
    df['roc'] = df['close'].pct_change(10) * 100

    tr = np.maximum(df['high'] - df['low'],
                    np.maximum(abs(df['high'] - df['close'].shift(1)),
                               abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    df = df.dropna()

    cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
    for c in cols:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

    future = df['close'].shift(-PREDICTION_HORIZON)
    df['target'] = 0
    df.loc[future > df['close'] * 1.001, 'target'] = 1
    df.loc[future < df['close'] * 0.999, 'target'] = 2
    df = df.dropna()
    return df, cols


def create_sequences(df, cols):
    features = df[cols].values
    targets = df['target'].values
    prices = df['close'].values
    X, y, p = [], [], []
    for i in range(len(features) - SEQ_LENGTH):
        X.append(features[i:i + SEQ_LENGTH])
        y.append(targets[i + SEQ_LENGTH])
        p.append(prices[i + SEQ_LENGTH])
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y)), np.array(p)


def evaluate(model, X, y, prices):
    model.eval()
    with torch.no_grad():
        out = model(X)
        _, pred = torch.max(out.data, 1)
        pred = pred.cpu()

        correct = (pred == y).sum().item()
        accuracy = correct / y.size(0)

        wins, losses = 0, 0
        for i in range(len(pred) - 5):
            p = pred[i].item()
            if p == 0:
                continue
            fp = prices[i + 5]
            cp = prices[i]
            if p == 1:
                profit = (fp - cp) / cp
            else:
                profit = (cp - fp) / cp
            if profit > 0:
                wins += 1
            else:
                losses += 1

        total = wins + losses
        wr = wins / total if total > 0 else 0
        hold_pct = 1.0 - (total / len(pred)) if len(pred) > 0 else 0

    return accuracy, wr, wins, losses, total, hold_pct


def train_one(model, optimizer, criterion, X_train, y_train):
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    total_loss = 0
    n = 0
    for bx, by in loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n if n > 0 else 0


if __name__ == '__main__':
    print()
    print("#" * 60)
    print("#  LSTM RETRAIN - FAST (Top 10, 5000 bars)")
    print(f"#  Target: 88% | Epochs: {EPOCHS}")
    print("#" * 60)

    print("\n[DATA]")
    df = get_data('BTCUSD', 5000)
    df, cols = prepare_features(df)
    X, y, prices = create_sequences(df, cols)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    p_test = prices[split:]

    # Target distribution
    hold_n = (y_train == 0).sum().item()
    buy_n = (y_train == 1).sum().item()
    sell_n = (y_train == 2).sum().item()
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Targets: HOLD={hold_n} ({hold_n/len(y_train)*100:.0f}%), "
          f"BUY={buy_n} ({buy_n/len(y_train)*100:.0f}%), "
          f"SELL={sell_n} ({sell_n/len(y_train)*100:.0f}%)")

    # Load top 10 best experts as starting population
    print("\n[LOADING TOP 10 EXPERTS]")
    experts_dir = SCRIPT_DIR / 'top_50_experts_v2' / 'top_50_experts'

    # Pick diverse experts
    picks = [
        'expert_BTCUSD_special.pth',
        'expert_rank01_AUDNZD.pth',
        'expert_rank05_AUDNZD.pth',
        'expert_rank11_XAUUSD.pth',
        'expert_rank15_XAUUSD.pth',
        'expert_rank21_ETHUSD.pth',
        'expert_rank23_ETHUSD.pth',
        'expert_rank25_ETHUSD.pth',
        'expert_rank31_BTCUSD.pth',
        'expert_rank35_BTCUSD.pth',
    ]

    population = []
    for name in picks:
        path = experts_dir / name
        if not path.exists():
            continue
        model = LSTMModel()
        state = torch.load(str(path), map_location='cpu', weights_only=False)
        model.load_state_dict(state)
        population.append({
            'model': model,
            'opt': optim.Adam(model.parameters(), lr=0.001),
            'crit': nn.CrossEntropyLoss(),
            'name': name.replace('.pth', ''),
        })

    # Fill to POP_SIZE with fresh
    while len(population) < POP_SIZE:
        model = LSTMModel()
        population.append({
            'model': model,
            'opt': optim.Adam(model.parameters(), lr=0.001),
            'crit': nn.CrossEntropyLoss(),
            'name': f'fresh_{len(population)}',
        })

    print(f"  Population: {len(population)}")

    # Initial eval
    for ind in population:
        acc, wr, w, l, t, h = evaluate(ind['model'], X_test, y_test, p_test)
        ind['wr'] = wr
        print(f"  {ind['name']:<35} WR={wr*100:.1f}% Acc={acc*100:.1f}% Hold={h*100:.0f}%")

    # TRAINING
    print(f"\n[TRAINING {EPOCHS} EPOCHS]")
    best_ever = 0
    best_model = None

    for epoch in range(EPOCHS):
        # Train each individual
        for ind in population:
            loss = train_one(ind['model'], ind['opt'], ind['crit'], X_train, y_train)

        # Evaluate
        for ind in population:
            acc, wr, w, l, t, h = evaluate(ind['model'], X_test, y_test, p_test)
            ind['wr'] = wr
            ind['acc'] = acc
            ind['trades'] = t
            ind['hold'] = h

        population.sort(key=lambda x: x['wr'], reverse=True)
        best = population[0]
        avg_wr = np.mean([p['wr'] for p in population])

        if best['wr'] > best_ever:
            best_ever = best['wr']
            best_model = deepcopy(best['model'].state_dict())

        print(f"  E{epoch+1:>2}/{EPOCHS} | Best={best['wr']*100:.1f}% ({best['name']}) "
              f"| Avg={avg_wr*100:.1f}% | Hold={best['hold']*100:.0f}% | Trades={best['trades']}")

        # Extinction every 3 epochs
        if (epoch + 1) % 3 == 0 and epoch < EPOCHS - 1:
            kill = int(POP_SIZE * 0.3)
            survivors = population[:POP_SIZE - kill]
            while len(survivors) < POP_SIZE:
                parent = random.choice(population[:ELITE])
                child = LSTMModel()
                child.load_state_dict(deepcopy(parent['model'].state_dict()))
                with torch.no_grad():
                    for p in child.parameters():
                        if random.random() < 0.15:
                            p.add_(torch.randn_like(p) * 0.05)
                survivors.append({
                    'model': child,
                    'opt': optim.Adam(child.parameters(), lr=0.001),
                    'crit': nn.CrossEntropyLoss(),
                    'name': f'mut_e{epoch+1}_{len(survivors)}',
                    'wr': 0,
                })
            population = survivors
            print(f"         EXTINCTION: -{kill} +{kill}")

        if best['wr'] >= 0.88:
            print(f"\n  88% TARGET HIT!")
            break

    # Final
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for i, ind in enumerate(population):
        acc, wr, w, l, t, h = evaluate(ind['model'], X_test, y_test, p_test)
        print(f"  #{i+1} {ind['name']:<30} WR={wr*100:.1f}% ({w}W/{l}L={t}T) Hold={h*100:.0f}%")

    print(f"\n  Best ever: {best_ever*100:.1f}%")
    print(f"{'='*60}")

    if best_model:
        path = SCRIPT_DIR / 'top_50_experts' / 'expert_RETRAINED_best.pth'
        torch.save(best_model, str(path))
        print(f"  Saved: {path.name}")
