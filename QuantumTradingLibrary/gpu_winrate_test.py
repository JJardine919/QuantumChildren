"""
GPU-accelerated win rate test for ETARE champions
Tests actual win/loss ratio on held-out data
"""
import torch
import torch.nn as nn
import sqlite3
import io
import sys
import MetaTrader5 as mt5

# Force unbuffered
def print_flush(*args, **kwargs):
    sys.stdout.write(' '.join(map(str, args)) + '\n')
    sys.stdout.flush()
print = print_flush

# GPU Setup
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"GPU: {torch_directml.device_name(0)}")
except:
    DEVICE = torch.device("cpu")
    print("Using CPU")

class TradingLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # Run LSTM on CPU (DirectML doesn't support it well)
        self.lstm = self.lstm.cpu()
        x_cpu = x.cpu()
        lstm_out, _ = self.lstm(x_cpu)
        # FC layer can use GPU
        self.fc = self.fc.to(DEVICE)
        return self.fc(lstm_out[:, -1, :].to(DEVICE))

def get_mt5_data(symbol, timeframe, bars=50000):
    """Get data from MT5"""
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return None

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"No data for {symbol}")
        return None

    import pandas as pd
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def prepare_features(df):
    """8 features for LSTM"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain/(loss+1e-10)))

    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["macd"] = exp1 - exp2

    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()

    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["momentum"] = df["close"] / df["close"].shift(10)
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["price_change"] = df["close"].pct_change()

    df = df.ffill().bfill()

    cols = ['rsi', 'macd', 'bb_middle', 'bb_std', 'ema_10', 'momentum', 'atr', 'price_change']
    for c in cols:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

    return df.dropna(), cols

def evaluate_winrate(model, features, prices, hold_bars=5):
    """Calculate win rate"""
    model.eval()

    # Process in batches for GPU efficiency
    batch_size = 1000
    all_actions = []

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size]
            output = model(batch)
            actions = torch.argmax(output, dim=1).cpu().numpy()
            all_actions.extend(actions)

    actions = all_actions
    wins, losses = 0, 0
    position = None
    entry = 0
    entry_idx = 0

    for i in range(len(actions) - hold_bars):
        a = actions[i]
        curr = prices[i]

        if position is None:
            if a == 0:  # BUY
                position, entry, entry_idx = 'buy', curr, i
            elif a == 1:  # SELL
                position, entry, entry_idx = 'sell', curr, i
        else:
            held = (i - entry_idx) >= hold_bars
            opposite = (position == 'buy' and a == 1) or (position == 'sell' and a == 0)

            if held or opposite:
                exit_price = curr
                if position == 'buy':
                    if exit_price > entry: wins += 1
                    else: losses += 1
                else:
                    if exit_price < entry: wins += 1
                    else: losses += 1
                position = None

                if opposite:
                    if a == 0: position, entry, entry_idx = 'buy', curr, i
                    elif a == 1: position, entry, entry_idx = 'sell', curr, i

    total = wins + losses
    return (wins / total * 100) if total > 0 else 0, wins, losses, total

# Load champions from database
print("="*60)
print("GPU WIN RATE TEST - ETARE CHAMPIONS")
print("="*60)

conn = sqlite3.connect('etare_redux_v2.db')
cursor = conn.cursor()

# Get unique symbols that completed training
cursor.execute('''
    SELECT DISTINCT symbol FROM training_log
    WHERE batch = 10
''')
symbols = [row[0] for row in cursor.fetchall()]
print(f"Symbols with complete training: {symbols}")

# Test each symbol
results = []
for symbol in symbols[:5]:  # Test top 5
    print(f"\n--- Testing {symbol} ---")

    # Get champion weights from population_state (best fitness)
    cursor.execute('''
        SELECT weights, fitness FROM population_state
        WHERE symbol = ?
        ORDER BY fitness DESC
        LIMIT 1
    ''', (symbol,))

    row = cursor.fetchone()
    if not row:
        print(f"  No weights found for {symbol}")
        continue

    # Load model
    try:
        weights_blob = row[0]
        weights_buffer = io.BytesIO(weights_blob)
        state_dict = torch.load(weights_buffer, map_location='cpu', weights_only=False)

        model = TradingLSTM(input_size=8, hidden_size=128, num_layers=2, output_size=3)
        model.load_state_dict(state_dict)
        print(f"  Model loaded")
    except Exception as e:
        print(f"  Error loading model: {e}")
        continue

    # Get fresh test data from MT5
    tf = mt5.TIMEFRAME_M5
    df = get_mt5_data(symbol, tf, 30000)
    if df is None:
        continue

    df, cols = prepare_features(df)
    print(f"  Test data: {len(df)} bars")

    features = torch.FloatTensor(df[cols].values).to(DEVICE)
    prices = df['close'].values

    # Test win rate
    wr, wins, losses, total = evaluate_winrate(model, features, prices)
    results.append({'symbol': symbol, 'wr': wr, 'wins': wins, 'losses': losses, 'total': total})
    print(f"  WIN RATE: {wr:.1f}% ({wins}W/{losses}L = {total} trades)")

conn.close()
mt5.shutdown()

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if results:
    results.sort(key=lambda x: x['wr'], reverse=True)
    for r in results:
        status = "*** KEEPER ***" if r['wr'] >= 65 else ("*** TARGET ***" if r['wr'] >= 70 else "")
        print(f"  {r['symbol']}: {r['wr']:.1f}% {status}")

    best = results[0]
    print(f"\nBEST: {best['symbol']} @ {best['wr']:.1f}%")

    if best['wr'] >= 70:
        print("\n*** TARGET MET! Ready for entropy remover! ***")
    elif best['wr'] >= 65:
        print(f"\n*** KEEPER FOUND! Gap to 70%: {70-best['wr']:.1f}% ***")
    else:
        print(f"\nGap to 70%: {70-best['wr']:.1f}%")
print("="*60)
