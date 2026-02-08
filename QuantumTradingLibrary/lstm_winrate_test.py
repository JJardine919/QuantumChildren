"""
LSTM WIN RATE TEST - Using ETARE_50_Darwin.py logic EXACTLY
============================================================
Tests LSTM experts with:
- Correct 8 features (rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr)
- 30-bar sequences (SEQ_LENGTH=30) so LSTM memory is engaged
- Same evaluation as ETARE_50_Darwin.py evaluate_fitness()
- Same target thresholds (1.001 / 0.999)
"""
import sys
import io
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).parent.absolute()

SEQ_LENGTH = 30
PREDICTION_HORIZON = 5


# ============================================================
# MODEL: Same LSTMModel from ETARE_50_Darwin.py
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_cpu = x.cpu() if x.device != torch.device('cpu') else x
        out, _ = self.lstm(x_cpu)
        out = self.dropout(out[:, -1, :])
        out = out.to(x.device) if x.device != torch.device('cpu') else out
        out = self.fc(out)
        return out


# ============================================================
# DATA: MT5 with Binance fallback
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

    # Binance fallback
    import requests
    from datetime import datetime, timedelta
    binance_sym = symbol.replace('USD', 'USDT')
    print(f"  Fetching {binance_sym} from Binance...")
    url = "https://api.binance.com/api/v3/klines"
    end = datetime.now()
    start = end - timedelta(days=90)
    all_data = []
    current = start
    chunk = timedelta(minutes=5 * 1000)
    while current < end:
        chunk_end = min(current + chunk, end)
        params = {'symbol': binance_sym, 'interval': '5m',
                  'startTime': int(current.timestamp() * 1000),
                  'endTime': int(chunk_end.timestamp() * 1000), 'limit': 1000}
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                all_data.extend(r.json())
        except:
            pass
        current = chunk_end

    df = pd.DataFrame(all_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    print(f"  Got {len(df):,} bars from Binance")
    return df


# ============================================================
# FEATURES: Exact match to ETARE_50_Darwin.py prepare_features
# ============================================================

def prepare_features(df):
    """8 features matching ETARE_50_Darwin.py EXACTLY."""
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (adjust=False)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # Momentum + ROC
    df['momentum'] = df['close'] / df['close'].shift(10)
    df['roc'] = df['close'].pct_change(10) * 100

    # True ATR
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    df = df.dropna()

    # Z-score normalize
    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    # Targets (same thresholds as ETARE_50_Darwin.py)
    future_close = df['close'].shift(-PREDICTION_HORIZON)
    df['target'] = 0  # HOLD
    df.loc[future_close > df['close'] * 1.001, 'target'] = 1  # BUY (>0.1% up)
    df.loc[future_close < df['close'] * 0.999, 'target'] = 2  # SELL (>0.1% down)

    df = df.dropna()
    return df, feature_cols


# ============================================================
# SEQUENCES: 30-bar windows for LSTM
# ============================================================

def create_sequences(df, feature_cols):
    """Create 30-bar sequences exactly like ETARE_50_Darwin.py."""
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
# EVALUATION: Exact copy of ETARE_50_Darwin.py evaluate_fitness
# ============================================================

def evaluate_expert(model, X_test, y_test, prices):
    """Same logic as ETARE_50_Darwin.py evaluate_fitness."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted.cpu() == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total

        # Trading simulation — same as ETARE_50_Darwin.py
        balance = 10000
        wins, losses = 0, 0

        for i in range(len(predicted) - 5):
            pred = predicted[i].item()
            if pred == 0:
                continue

            future_price = prices[i + 5]
            current_price = prices[i]

            if pred == 1:  # BUY
                profit = (future_price - current_price) / current_price
            else:  # SELL
                profit = (current_price - future_price) / current_price

            if profit > 0:
                wins += 1
                balance *= (1 + abs(profit) * 0.1)
            else:
                losses += 1
                balance *= (1 - abs(profit) * 0.1)

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        profit_factor = wins / (losses + 1) if losses > 0 else wins
        final_profit = balance - 10000

        fitness = (
            accuracy * 0.3 +
            win_rate * 0.4 +
            profit_factor * 0.2 +
            1.0 * 0.1  # drawdown_resistance = 1/(1+0) = 1
        )

        return {
            'accuracy': accuracy * 100,
            'win_rate': win_rate * 100,
            'wins': wins,
            'losses': losses,
            'total_trades': total_trades,
            'fitness': fitness,
            'profit': final_profit,
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print()
    print("#" * 60)
    print("#  LSTM EXPERT WIN RATE TEST")
    print("#  Using ETARE_50_Darwin.py logic (30-bar sequences)")
    print("#" * 60)

    # Get data
    print("\n[DATA]")
    df = get_data('BTCUSD', 30000)
    df, feature_cols = prepare_features(df)
    print(f"  Prepared: {len(df):,} bars, {len(feature_cols)} features")

    # Create sequences
    X, y, prices = create_sequences(df, feature_cols)
    print(f"  Sequences: {len(X):,} (SEQ_LENGTH={SEQ_LENGTH})")

    # Test experts
    experts_dir = SCRIPT_DIR / 'top_50_experts_v2' / 'top_50_experts'
    expert_files = sorted(experts_dir.glob('*.pth'))
    print(f"  Found {len(expert_files)} experts")

    # Test top experts
    test_list = [
        'expert_BTCUSD_special.pth',
        'expert_rank01_AUDNZD.pth',
        'expert_rank11_XAUUSD.pth',
        'expert_rank21_ETHUSD.pth',
        'expert_rank31_BTCUSD.pth',
        'expert_rank41_BTCUSD.pth',
    ]

    results = []
    for name in test_list:
        path = experts_dir / name
        if not path.exists():
            print(f"  SKIP: {name} not found")
            continue

        print(f"\n[{name}]")
        model = LSTMModel(input_size=8, hidden_size=128, output_size=3)
        state = torch.load(str(path), map_location='cpu', weights_only=False)
        model.load_state_dict(state)

        r = evaluate_expert(model, X, y, prices)
        results.append({'name': name, **r})
        print(f"  Accuracy: {r['accuracy']:.1f}%")
        print(f"  Win Rate: {r['win_rate']:.1f}%  ({r['wins']}W / {r['losses']}L = {r['total_trades']} trades)")
        print(f"  Fitness:  {r['fitness']:.4f}")
        print(f"  Profit:   ${r['profit']:.2f}")

    # Summary
    print()
    print("=" * 70)
    print(f"  {'Expert':<35} {'Acc':>6} {'WR':>6} {'Trades':>7} {'Fitness':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*7} {'-'*8}")
    for r in results:
        name = r['name'].replace('.pth', '')
        print(f"  {name:<35} {r['accuracy']:>5.1f}% {r['win_rate']:>5.1f}% {r['total_trades']:>7} {r['fitness']:>8.4f}")
    print("=" * 70)
