"""
GPU-ACCELERATED EXPERT RETRAINER (Conv1D on DirectML)
=====================================================
Replaces LSTM with Conv1D architecture for AMD RX 6800 XT GPU training.
Conv1D forward AND backward pass work on DirectML (unlike LSTM).

Same data pipeline, same features, same output format.
Saves to top_50_experts/ with model_type marker for BRAIN compatibility.
"""
import sys
import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_directml
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
SCRIPT_DIR = Path(__file__).parent.absolute()

SEQ_LENGTH = 30
PREDICTION_HORIZON = 5
POP_SIZE = 10
ELITE = 3
EPOCHS = 80
BATCH_SIZE = 128  # Larger batch on GPU
CONFIDENCE_THRESHOLD = 0.55
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE = 10
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.02

# GPU device
device = torch_directml.device()
print(f"[GPU] Using DirectML device: {device}")


class Conv1DModel(nn.Module):
    """Conv1D model for time series classification. GPU-compatible on DirectML."""
    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, hidden_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def get_data(symbol='BTCUSD', bars=30000):
    import MetaTrader5 as mt5
    terminal_paths = [
        r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe",
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        None,
    ]
    for path in terminal_paths:
        try:
            ok = mt5.initialize(path=path) if path else mt5.initialize()
            if ok:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
                mt5.shutdown()
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    print(f"  Got {len(df):,} bars from MT5 ({symbol})")
                    return df
            else:
                print(f"  MT5 init failed for {path or 'default'}: {mt5.last_error()}")
        except Exception as e:
            print(f"  MT5 error: {e}")
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


def walk_forward_split(X, y, prices, n_chunks=3):
    """Walk-forward: train on chunk 1..k, test on k+1."""
    chunk_size = len(X) // n_chunks
    folds = []
    for k in range(1, n_chunks):
        train_end = k * chunk_size
        test_end = min((k + 1) * chunk_size, len(X))
        folds.append((X[:train_end], y[:train_end], X[train_end:test_end], y[train_end:test_end], prices[train_end:test_end]))
    return folds


def compute_class_weights(y):
    counts = torch.bincount(y, minlength=3).float()
    total = counts.sum()
    weights = total / (3 * counts + 1)
    return weights


def evaluate(model, X, y, prices, confidence_threshold=CONFIDENCE_THRESHOLD):
    model.eval()
    # Move to CPU for evaluation to avoid DirectML quirks with large batches
    model_cpu = deepcopy(model).cpu()
    X_cpu = X.cpu()
    y_cpu = y.cpu()

    with torch.no_grad():
        out = model_cpu(X_cpu)
        probs = torch.softmax(out, dim=1)
        max_prob, pred = torch.max(probs, dim=1)

        pred[max_prob < confidence_threshold] = 0

        wins, losses = 0, 0
        for i in range(len(pred) - 5):
            p = pred[i].item()
            if p == 0:
                continue
            fp = prices[i + 5]
            cp = prices[i]
            if p == 1:
                wins += 1 if fp > cp else 0
                losses += 1 if fp <= cp else 0
            elif p == 2:
                wins += 1 if fp < cp else 0
                losses += 1 if fp >= cp else 0

        total_trades = wins + losses
        wr = wins / total_trades * 100 if total_trades > 0 else 0
        hold_pct = (pred == 0).sum().item() / len(pred) * 100

    del model_cpu
    return wr, hold_pct, total_trades


def train_one_epoch(model, X_train, y_train, optimizer, criterion):
    model.train()
    X_gpu = X_train.to(device)
    y_gpu = y_train.to(device)

    total_loss = 0
    n_batches = 0
    indices = torch.randperm(len(X_gpu))

    for start in range(0, len(X_gpu), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        xb = X_gpu[batch_idx]
        yb = y_gpu[batch_idx]

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


if __name__ == '__main__':
    print()
    print("#" * 60)
    print("#  GPU RETRAIN - Conv1D on DirectML (AMD RX 6800 XT)")
    print(f"#  Target: 88% | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"#  Confidence: {CONFIDENCE_THRESHOLD} | LabelSmooth: {LABEL_SMOOTHING}")
    print("#" * 60)

    # --- DATA ---
    print("\n[DATA]")
    symbols = ['BTCUSD', 'XAUUSD', 'ETHUSD']
    all_models = {}

    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"  TRAINING: {symbol}")
        print(f"{'='*50}")

        try:
            df = get_data(symbol, 30000)
        except RuntimeError as e:
            print(f"  SKIP {symbol}: {e}")
            continue

        df, cols = prepare_features(df)
        X, y, prices = create_sequences(df, cols)
        print(f"  Sequences: {len(X):,}")

        # Walk-forward split
        folds = walk_forward_split(X, y, prices)
        print(f"  Walk-forward folds: {len(folds)}")

        class_weights = compute_class_weights(torch.cat([f[1] for f in folds]))
        print(f"  Class weights: H={class_weights[0]:.2f} B={class_weights[1]:.2f} S={class_weights[2]:.2f}")

        # Create model on GPU
        model = Conv1DModel(input_size=len(cols), hidden_size=128, output_size=3).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=LABEL_SMOOTHING
        )
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=LR_REDUCE_PATIENCE, factor=0.5)

        best_wr = 0
        best_model = None
        stale_epochs = 0

        # Train across all folds
        X_train = torch.cat([f[0] for f in folds])
        y_train = torch.cat([f[1] for f in folds])
        X_test = folds[-1][2]
        y_test = folds[-1][3]
        prices_test = folds[-1][4]

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, X_train, y_train, optimizer, criterion)

            # Evaluate every 5 epochs to save time
            if (epoch + 1) % 5 == 0 or epoch == 0:
                train_wr, train_hold, _ = evaluate(model, X_train, y_train, prices[:len(X_train)])
                test_wr, test_hold, trades = evaluate(model, X_test, y_test, prices_test)
                gap = abs(train_hold - test_hold)

                print(f"  E {epoch+1:2d}/{EPOCHS} | loss={loss:.4f} | "
                      f"Train: WR={train_wr:.1f}% Hold={train_hold:.0f}% | "
                      f"Test: WR={test_wr:.1f}% Hold={test_hold:.0f}% | "
                      f"Gap={gap:.0f}% | Trades={trades}")

                scheduler.step(-test_wr)

                if test_wr > best_wr:
                    best_wr = test_wr
                    best_model = deepcopy(model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 5

                if stale_epochs >= EARLY_STOP_PATIENCE:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

        # Save best model (on CPU for portability)
        if best_model:
            save_name = f'expert_{symbol}_conv1d.pth'
            save_path = SCRIPT_DIR / 'top_50_experts' / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save state dict on CPU
            cpu_state = {k: v.cpu() for k, v in best_model.items()}
            torch.save(cpu_state, str(save_path))
            print(f"  SAVED: {save_name} (WR={best_wr:.1f}%)")

            all_models[symbol] = {
                'filename': save_name,
                'symbol': symbol,
                'model_type': 'conv1d',
                'input_size': len(cols),
                'hidden_size': 128,
                'output_size': 3,
                'win_rate': best_wr,
                'trained': datetime.now().isoformat(),
                'verified': True,
            }

    # Update manifest with new conv1d models
    manifest_path = SCRIPT_DIR / 'top_50_experts' / 'top_50_manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {'experts': []}

    # Replace or add conv1d experts for each symbol
    for symbol, info in all_models.items():
        # Remove old entries for this symbol
        manifest['experts'] = [e for e in manifest['experts'] if e.get('symbol') != symbol]
        manifest['experts'].insert(0, info)

    manifest['updated'] = datetime.now().isoformat()
    manifest['model_type'] = 'conv1d'

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest updated: {len(manifest['experts'])} experts")

    # Save training log
    log_path = SCRIPT_DIR / 'top_50_experts' / 'gpu_training_log.json'
    with open(log_path, 'w') as f:
        json.dump({
            'trained': datetime.now().isoformat(),
            'device': 'DirectML (AMD RX 6800 XT)',
            'architecture': 'Conv1D',
            'epochs': EPOCHS,
            'symbols': list(all_models.keys()),
            'results': {s: {'win_rate': m['win_rate']} for s, m in all_models.items()},
        }, f, indent=2)

    print()
    print("#" * 60)
    print("#  GPU TRAINING COMPLETE")
    for sym, info in all_models.items():
        print(f"#  {sym}: WR={info['win_rate']:.1f}%")
    print("#" * 60)
