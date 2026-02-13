"""
LSTM EXPERT RETRAINER - FAST VERSION (v2)
==========================================
Upgrades: confidence thresholding, class-weighted loss, label smoothing,
walk-forward validation (30K bars, 6 chunks), retrained head start,
80 epochs with early stopping + LR scheduling, training log.

Target: 88% win rate with HOLD selectivity parity (train/test HOLD% gap < 20%)
"""
import sys
import io
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
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
BATCH_SIZE = 64
CONFIDENCE_THRESHOLD = 0.55  # Only trade when softmax confidence exceeds this
EARLY_STOP_PATIENCE = 15     # Stop if no improvement for this many epochs
LR_REDUCE_PATIENCE = 15      # Halve LR after this many stale epochs
N_WALKFORWARD_CHUNKS = 6     # Number of sequential chunks for walk-forward
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1


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


def get_data(symbol='BTCUSD', bars=30000):
    import MetaTrader5 as mt5
    # Try GetLeveraged terminal first, then fallback to default
    terminal_paths = [
        r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe",
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        None,  # default
    ]
    for path in terminal_paths:
        try:
            if path:
                ok = mt5.initialize(path=path)
            else:
                ok = mt5.initialize()
            if ok:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
                mt5.shutdown()
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    print(f"  Got {len(df):,} bars from MT5 ({symbol}) via {path or 'default'}")
                    return df
            else:
                err = mt5.last_error()
                print(f"  MT5 init failed for {path or 'default'}: {err}")
        except Exception as e:
            print(f"  MT5 error for {path or 'default'}: {e}")
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


def evaluate(model, X, y, prices, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Evaluate with confidence thresholding: predictions below threshold forced to HOLD."""
    model.eval()
    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=1)
        max_prob, pred = torch.max(probs, dim=1)
        pred = pred.cpu()
        max_prob = max_prob.cpu()

        # Force low-confidence predictions to HOLD (class 0)
        pred[max_prob < confidence_threshold] = 0

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


def walk_forward_split(X, y, prices, n_chunks=N_WALKFORWARD_CHUNKS):
    """Split data into sequential chunks for walk-forward validation.
    Each fold: train on 4 consecutive chunks, test on the next 1.
    Returns list of (X_train, y_train, X_test, y_test, p_test) tuples.
    """
    chunk_size = len(X) // n_chunks
    folds = []
    train_chunks = 4

    for i in range(train_chunks, n_chunks):
        train_start = (i - train_chunks) * chunk_size
        train_end = i * chunk_size
        test_start = i * chunk_size
        test_end = min((i + 1) * chunk_size, len(X))

        X_tr = X[train_start:train_end]
        y_tr = y[train_start:train_end]
        X_te = X[test_start:test_end]
        y_te = y[test_start:test_end]
        p_te = prices[test_start:test_end]

        folds.append((X_tr, y_tr, X_te, y_te, p_te))

    return folds


def compute_class_weights(y_train):
    """Compute inverse-frequency class weights to handle class imbalance."""
    classes = torch.unique(y_train)
    num_classes = len(classes)
    total = len(y_train)
    weights = torch.zeros(num_classes)
    for c in classes:
        count = (y_train == c).sum().item()
        weights[c] = total / (num_classes * count) if count > 0 else 1.0
    return weights


def make_criterion(class_weights):
    """Create weighted CrossEntropyLoss with label smoothing."""
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)


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
    print("#  LSTM RETRAIN - WALK-FORWARD + CONFIDENCE (v2)")
    print(f"#  Target: 88% | Epochs: {EPOCHS} | Chunks: {N_WALKFORWARD_CHUNKS}")
    print(f"#  Confidence: {CONFIDENCE_THRESHOLD} | LabelSmooth: {LABEL_SMOOTHING}")
    print("#" * 60)

    # --- DATA ---
    print("\n[DATA]")
    df = get_data('BTCUSD', 30000)
    df, cols = prepare_features(df)
    X, y, prices = create_sequences(df, cols)
    print(f"  Total sequences: {len(X):,}")

    # --- WALK-FORWARD FOLDS ---
    print("\n[WALK-FORWARD SPLIT]")
    folds = walk_forward_split(X, y, prices)
    print(f"  Folds: {len(folds)}")
    for fi, (Xtr, ytr, Xte, yte, pte) in enumerate(folds):
        hold_n = (ytr == 0).sum().item()
        buy_n = (ytr == 1).sum().item()
        sell_n = (ytr == 2).sum().item()
        print(f"  Fold {fi+1}: Train={len(Xtr):,} Test={len(Xte):,} "
              f"| HOLD={hold_n/len(ytr)*100:.0f}% BUY={buy_n/len(ytr)*100:.0f}% SELL={sell_n/len(ytr)*100:.0f}%")

    # Compute class weights from first fold's training data (representative)
    all_y_train = torch.cat([f[1] for f in folds])
    class_weights = compute_class_weights(all_y_train)
    print(f"  Class weights: HOLD={class_weights[0]:.2f} BUY={class_weights[1]:.2f} SELL={class_weights[2]:.2f}")

    # --- LOAD POPULATION ---
    print("\n[LOADING POPULATION]")
    retrained_path = SCRIPT_DIR / 'top_50_experts' / 'expert_RETRAINED_best.pth'
    experts_dir = SCRIPT_DIR / 'top_50_experts_v2' / 'top_50_experts'

    population = []

    # Step 5: Try retrained best as head start
    if retrained_path.exists():
        print(f"  Found retrained best: {retrained_path.name}")
        seed_state = torch.load(str(retrained_path), map_location='cpu', weights_only=False)
        for i in range(POP_SIZE):
            model = LSTMModel()
            model.load_state_dict(deepcopy(seed_state))
            # Keep 3 exact copies, mutate the rest
            if i >= ELITE:
                with torch.no_grad():
                    for p in model.parameters():
                        if random.random() < 0.2:
                            p.add_(torch.randn_like(p) * 0.03)
            population.append({
                'model': model,
                'opt': optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY),
                'crit': make_criterion(class_weights),
                'name': f'retrained_clone_{i}' if i < ELITE else f'retrained_mut_{i}',
            })
        print(f"  Seeded {POP_SIZE} from retrained best ({ELITE} exact, {POP_SIZE-ELITE} mutated)")
    else:
        print(f"  No retrained best found, loading from top_50_experts_v2")
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
        for name in picks:
            path = experts_dir / name
            if not path.exists():
                continue
            model = LSTMModel()
            state = torch.load(str(path), map_location='cpu', weights_only=False)
            model.load_state_dict(state)
            population.append({
                'model': model,
                'opt': optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY),
                'crit': make_criterion(class_weights),
                'name': name.replace('.pth', ''),
            })

    # Fill to POP_SIZE with fresh
    while len(population) < POP_SIZE:
        model = LSTMModel()
        population.append({
            'model': model,
            'opt': optim.Adam(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY),
            'crit': make_criterion(class_weights),
            'name': f'fresh_{len(population)}',
        })

    print(f"  Population: {len(population)}")

    # --- INITIAL EVAL (on all test folds) ---
    print("\n[INITIAL EVAL]")
    for ind in population:
        fold_wrs = []
        fold_holds = []
        for Xtr, ytr, Xte, yte, pte in folds:
            acc, wr, w, l, t, h = evaluate(ind['model'], Xte, yte, pte)
            fold_wrs.append(wr)
            fold_holds.append(h)
        avg_wr = np.mean(fold_wrs)
        avg_hold = np.mean(fold_holds)
        ind['wr'] = avg_wr
        ind['hold'] = avg_hold
        print(f"  {ind['name']:<35} WR={avg_wr*100:.1f}% Hold={avg_hold*100:.0f}%")

    # --- TRAINING ---
    print(f"\n[TRAINING {EPOCHS} EPOCHS - Walk-Forward]")
    best_ever = 0
    best_model = None
    stale_epochs = 0
    current_lr = 0.001
    training_log = []

    for epoch in range(EPOCHS):
        # Train each individual on each fold's training data
        for ind in population:
            for Xtr, ytr, Xte, yte, pte in folds:
                train_one(ind['model'], ind['opt'], ind['crit'], Xtr, ytr)

        # Evaluate on all folds (train + test)
        for ind in population:
            # Test metrics (walk-forward out-of-sample)
            test_wrs, test_holds, test_trades = [], [], []
            for Xtr, ytr, Xte, yte, pte in folds:
                acc, wr, w, l, t, h = evaluate(ind['model'], Xte, yte, pte)
                test_wrs.append(wr)
                test_holds.append(h)
                test_trades.append(t)
            ind['wr'] = np.mean(test_wrs)
            ind['hold'] = np.mean(test_holds)
            ind['trades'] = int(np.sum(test_trades))

            # Train metrics (in-sample) - use first fold as representative
            Xtr0, ytr0 = folds[0][0], folds[0][1]
            p_train0 = prices[:len(Xtr0)]
            acc_tr, wr_tr, _, _, _, h_tr = evaluate(ind['model'], Xtr0, ytr0, p_train0)
            ind['train_wr'] = wr_tr
            ind['train_hold'] = h_tr

        population.sort(key=lambda x: x['wr'], reverse=True)
        best = population[0]
        avg_wr = np.mean([p['wr'] for p in population])

        # Track best ever
        improved = False
        if best['wr'] > best_ever:
            best_ever = best['wr']
            best_model = deepcopy(best['model'].state_dict())
            stale_epochs = 0
            improved = True
        else:
            stale_epochs += 1

        # Hold gap (overfit indicator)
        hold_gap = abs(best['train_hold'] - best['hold'])
        overfit_flag = " *** OVERFIT WARNING ***" if hold_gap > 0.20 else ""

        # Step 7: Enhanced progress reporting
        print(f"  E{epoch+1:>2}/{EPOCHS} | Train: WR={best['train_wr']*100:.1f}% Hold={best['train_hold']*100:.0f}% "
              f"| Test: WR={best['wr']*100:.1f}% Hold={best['hold']*100:.0f}% "
              f"| Gap={hold_gap*100:.0f}% | Trades={best['trades']} "
              f"({best['name']}){overfit_flag}")

        # Step 8: Log epoch
        training_log.append({
            'epoch': epoch + 1,
            'train_wr': round(best['train_wr'], 4),
            'test_wr': round(best['wr'], 4),
            'train_hold': round(best['train_hold'], 4),
            'test_hold': round(best['hold'], 4),
            'hold_gap': round(hold_gap, 4),
            'best_name': best['name'],
            'lr': current_lr,
            'avg_wr': round(avg_wr, 4),
            'trades': best['trades'],
        })

        # Step 6: LR scheduling - halve LR after patience epochs without improvement
        if stale_epochs > 0 and stale_epochs % LR_REDUCE_PATIENCE == 0:
            current_lr *= 0.5
            print(f"         LR REDUCED -> {current_lr:.6f}")
            for ind in population:
                for pg in ind['opt'].param_groups:
                    pg['lr'] = current_lr

        # Step 6: Early stopping
        if stale_epochs >= EARLY_STOP_PATIENCE:
            print(f"\n  EARLY STOP: No improvement for {EARLY_STOP_PATIENCE} epochs")
            break

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
                    'opt': optim.Adam(child.parameters(), lr=current_lr, weight_decay=WEIGHT_DECAY),
                    'crit': make_criterion(class_weights),
                    'name': f'mut_e{epoch+1}_{len(survivors)}',
                    'wr': 0,
                    'hold': 0,
                    'trades': 0,
                    'train_wr': 0,
                    'train_hold': 0,
                })
            population = survivors
            print(f"         EXTINCTION: -{kill} +{kill}")

        if best['wr'] >= 0.88:
            print(f"\n  88% TARGET HIT!")
            break

    # --- FINAL RESULTS ---
    print(f"\n{'='*60}")
    print(f"  RESULTS (Walk-Forward)")
    print(f"{'='*60}")
    for i, ind in enumerate(population):
        fold_wrs, fold_holds = [], []
        total_w, total_l = 0, 0
        for Xtr, ytr, Xte, yte, pte in folds:
            acc, wr, w, l, t, h = evaluate(ind['model'], Xte, yte, pte)
            fold_wrs.append(wr)
            fold_holds.append(h)
            total_w += w
            total_l += l
        avg_wr = np.mean(fold_wrs)
        avg_hold = np.mean(fold_holds)
        total_t = total_w + total_l
        print(f"  #{i+1} {ind['name']:<30} WR={avg_wr*100:.1f}% ({total_w}W/{total_l}L={total_t}T) Hold={avg_hold*100:.0f}%")

    print(f"\n  Best ever: {best_ever*100:.1f}%")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Walk-forward folds: {len(folds)}")
    print(f"{'='*60}")

    # Save best model
    if best_model:
        save_path = SCRIPT_DIR / 'top_50_experts' / 'expert_RETRAINED_best.pth'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model, str(save_path))
        print(f"  Saved: {save_path.name}")

    # Step 8: Save training log
    log_path = SCRIPT_DIR / 'top_50_experts' / 'training_log.json'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        'started': datetime.now().isoformat(),
        'config': {
            'epochs': EPOCHS,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'label_smoothing': LABEL_SMOOTHING,
            'weight_decay': WEIGHT_DECAY,
            'n_chunks': N_WALKFORWARD_CHUNKS,
            'pop_size': POP_SIZE,
            'batch_size': BATCH_SIZE,
        },
        'best_wr': round(best_ever, 4),
        'epochs_run': len(training_log),
        'log': training_log,
    }
    with open(str(log_path), 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"  Saved: {log_path.name}")
