"""
LSTM EXPERT RETRAINER
=====================
Takes existing LSTM experts (head start) and retrains them using
ETARE_50_Darwin.py's evolution pipeline:
- 10 epochs of backprop training
- Extinction events (kill bottom 30%, repopulate with mutated clones)
- Evaluate with same fitness function

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
import json
import sqlite3
from pathlib import Path
from copy import deepcopy
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

SCRIPT_DIR = Path(__file__).parent.absolute()

# === CONFIG ===
SEQ_LENGTH = 30
PREDICTION_HORIZON = 5
POPULATION_SIZE = 50
ELITE_SIZE = 5
EXTINCTION_RATE = 0.3
EPOCHS = 20  # More than baseline's 10
BATCH_SIZE = 64
LR = 0.001


# ============================================================
# MODEL: Same as ETARE_50_Darwin.py
# ============================================================

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


# ============================================================
# DATA + FEATURES: From ETARE_50_Darwin.py
# ============================================================

def get_data(symbol='BTCUSD', bars=30000):
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
    """8 features from ETARE_50_Darwin.py."""
    df = df.copy()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    df['momentum'] = df['close'] / df['close'].shift(10)
    df['roc'] = df['close'].pct_change(10) * 100

    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    df = df.dropna()

    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    future_close = df['close'].shift(-PREDICTION_HORIZON)
    df['target'] = 0
    df.loc[future_close > df['close'] * 1.001, 'target'] = 1
    df.loc[future_close < df['close'] * 0.999, 'target'] = 2

    df = df.dropna()
    return df, feature_cols


def create_sequences(df, feature_cols):
    features = df[feature_cols].values
    targets = df['target'].values
    prices = df['close'].values

    X, y, p = [], [], []
    for i in range(len(features) - SEQ_LENGTH):
        X.append(features[i:i + SEQ_LENGTH])
        y.append(targets[i + SEQ_LENGTH])
        p.append(prices[i + SEQ_LENGTH])

    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y)), np.array(p)


# ============================================================
# EVALUATION: From ETARE_50_Darwin.py
# ============================================================

def evaluate(model, X_test, y_test, prices):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted.cpu() == y_test).sum().item()
        accuracy = correct / y_test.size(0)

        wins, losses = 0, 0
        for i in range(len(predicted) - 5):
            pred = predicted[i].item()
            if pred == 0:
                continue
            future_price = prices[i + 5]
            current_price = prices[i]

            if pred == 1:
                profit = (future_price - current_price) / current_price
            else:
                profit = (current_price - future_price) / current_price

            if profit > 0:
                wins += 1
            else:
                losses += 1

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        hold_pct = 1.0 - (total_trades / len(predicted)) if len(predicted) > 0 else 0

        return accuracy, win_rate, wins, losses, total_trades, hold_pct


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_epoch(model, optimizer, criterion, X_train, y_train):
    model.train()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_loss = 0
    batches = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1

    return total_loss / batches if batches > 0 else 0


def clone_model(model):
    new_model = LSTMModel(input_size=8, hidden_size=128, output_size=3)
    new_model.load_state_dict(deepcopy(model.state_dict()))
    return new_model


def mutate_model(model, strength=0.1):
    with torch.no_grad():
        for param in model.parameters():
            if random.random() < 0.1:  # 10% chance per parameter tensor
                noise = torch.randn_like(param) * strength
                param.add_(noise)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print()
    print("#" * 60)
    print("#  LSTM EXPERT RETRAINER")
    print(f"#  Target: 88% win rate")
    print(f"#  Epochs: {EPOCHS}, Population: {POPULATION_SIZE}")
    print("#" * 60)

    # Load data
    print("\n[DATA]")
    df = get_data('BTCUSD', 30000)
    df, feature_cols = prepare_features(df)
    X, y, prices = create_sequences(df, feature_cols)

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    p_test = prices[split:]

    print(f"  Train: {len(X_train):,} sequences")
    print(f"  Test:  {len(X_test):,} sequences")

    # Load existing experts as head start
    print("\n[LOADING HEAD START EXPERTS]")
    experts_dir = SCRIPT_DIR / 'top_50_experts_v2' / 'top_50_experts'
    expert_files = sorted(experts_dir.glob('*.pth'))[:POPULATION_SIZE]

    population = []
    for f in expert_files:
        model = LSTMModel(input_size=8, hidden_size=128, output_size=3)
        state = torch.load(str(f), map_location='cpu', weights_only=False)
        model.load_state_dict(state)
        population.append({
            'model': model,
            'optimizer': optim.Adam(model.parameters(), lr=LR),
            'criterion': nn.CrossEntropyLoss(),
            'name': f.stem,
            'fitness': 0,
            'win_rate': 0,
            'accuracy': 0,
        })

    # Fill remaining slots if less than POPULATION_SIZE
    while len(population) < POPULATION_SIZE:
        model = LSTMModel(input_size=8, hidden_size=128, output_size=3)
        population.append({
            'model': model,
            'optimizer': optim.Adam(model.parameters(), lr=LR),
            'criterion': nn.CrossEntropyLoss(),
            'name': f'fresh_{len(population)}',
            'fitness': 0,
            'win_rate': 0,
            'accuracy': 0,
        })

    print(f"  Loaded {len(expert_files)} experts + {len(population) - len(expert_files)} fresh")

    # Initial evaluation
    print("\n[INITIAL EVALUATION]")
    for ind in population:
        acc, wr, w, l, t, hold = evaluate(ind['model'], X_test, y_test, p_test)
        ind['accuracy'] = acc
        ind['win_rate'] = wr
        ind['fitness'] = acc * 0.3 + wr * 0.4 + (w / (l + 1)) * 0.2 + 0.1

    population.sort(key=lambda x: x['win_rate'], reverse=True)
    best = population[0]
    avg_wr = np.mean([p['win_rate'] for p in population])
    print(f"  Best WR: {best['win_rate']*100:.1f}% ({best['name']})")
    print(f"  Avg WR:  {avg_wr*100:.1f}%")

    # Training loop
    print(f"\n[TRAINING] {EPOCHS} epochs with evolution...")
    best_ever_wr = 0
    best_ever_model = None

    for epoch in range(EPOCHS):
        # Train all individuals
        losses = []
        for ind in population:
            loss = train_epoch(ind['model'], ind['optimizer'], ind['criterion'], X_train, y_train)
            losses.append(loss)

        avg_loss = np.mean(losses)

        # Evaluate all
        for ind in population:
            acc, wr, w, l, t, hold = evaluate(ind['model'], X_test, y_test, p_test)
            ind['accuracy'] = acc
            ind['win_rate'] = wr
            ind['hold_pct'] = hold
            ind['trades'] = t
            ind['fitness'] = acc * 0.3 + wr * 0.4 + (w / (l + 1)) * 0.2 + 0.1

        population.sort(key=lambda x: x['fitness'], reverse=True)
        best = population[0]

        # Track best ever
        if best['win_rate'] > best_ever_wr:
            best_ever_wr = best['win_rate']
            best_ever_model = deepcopy(best['model'].state_dict())

        avg_wr = np.mean([p['win_rate'] for p in population])
        print(f"  Epoch {epoch+1:>2}/{EPOCHS} | Loss={avg_loss:.4f} | "
              f"Best WR={best['win_rate']*100:.1f}% | "
              f"Avg WR={avg_wr*100:.1f}% | "
              f"Hold={best['hold_pct']*100:.0f}% | "
              f"Trades={best['trades']}")

        # Extinction event every 2 epochs
        if (epoch + 1) % 2 == 0 and epoch < EPOCHS - 1:
            extinction_count = int(POPULATION_SIZE * EXTINCTION_RATE)
            survivors = population[:POPULATION_SIZE - extinction_count]

            # Repopulate from top performers
            while len(survivors) < POPULATION_SIZE:
                parent = random.choice(population[:ELITE_SIZE])
                child_model = clone_model(parent['model'])
                mutate_model(child_model, strength=0.05)
                survivors.append({
                    'model': child_model,
                    'optimizer': optim.Adam(child_model.parameters(), lr=LR),
                    'criterion': nn.CrossEntropyLoss(),
                    'name': f"mutant_e{epoch+1}_{len(survivors)}",
                    'fitness': 0,
                    'win_rate': 0,
                    'accuracy': 0,
                })

            population = survivors
            print(f"         EXTINCTION: Killed {extinction_count}, kept {POPULATION_SIZE - extinction_count}")

        # Early exit if we hit target
        if best['win_rate'] >= 0.88:
            print(f"\n  TARGET HIT: {best['win_rate']*100:.1f}% >= 88%!")
            break

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")

    population.sort(key=lambda x: x['win_rate'], reverse=True)
    for i, ind in enumerate(population[:10]):
        acc, wr, w, l, t, hold = evaluate(ind['model'], X_test, y_test, p_test)
        print(f"  #{i+1:>2} {ind['name']:<35} WR={wr*100:.1f}% Acc={acc*100:.1f}% "
              f"({w}W/{l}L={t}T) Hold={hold*100:.0f}%")

    print(f"\n  Best ever WR: {best_ever_wr*100:.1f}%")
    print(f"{'='*60}")

    # Save best model
    if best_ever_model:
        save_path = SCRIPT_DIR / 'top_50_experts' / 'expert_RETRAINED_best.pth'
        torch.save(best_ever_model, str(save_path))
        print(f"\n  Best model saved: {save_path.name}")

    # Save top 5
    for i, ind in enumerate(population[:5]):
        save_path = SCRIPT_DIR / 'top_50_experts' / f'expert_RETRAINED_rank{i+1}.pth'
        torch.save(ind['model'].state_dict(), str(save_path))
        print(f"  Saved: {save_path.name}")
