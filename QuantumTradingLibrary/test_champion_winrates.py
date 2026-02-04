"""
Quick test of current champion win rates on recent data
Champions are LSTM models
"""
import torch
import torch.nn as nn
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

print = lambda *args, **kwargs: (sys.stdout.write(' '.join(map(str, args)) + '\n'), sys.stdout.flush())

# Force CPU for LSTM compatibility
DEVICE = torch.device("cpu")
print("Using CPU for LSTM evaluation")

class TradingLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features) or (batch, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def fetch_binance(symbol, interval, months):
    """Fetch recent data from Binance"""
    print(f"Fetching {symbol} {interval} ({months} months)...")

    url = "https://api.binance.com/api/v3/klines"
    end_time = datetime.now()
    start_time = end_time - timedelta(days=months * 30)

    all_data = []
    current = start_time
    interval_mins = {'1m': 1, '5m': 5, '15m': 15}[interval]
    chunk = timedelta(minutes=interval_mins * 1000)

    while current < end_time:
        chunk_end = min(current + chunk, end_time)
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': int(current.timestamp() * 1000),
            'endTime': int(chunk_end.timestamp() * 1000),
            'limit': 1000
        }

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
        'taker_buy_quote', 'ignore'
    ])

    df['time'] = pd.to_datetime(df['time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    print(f"  Got {len(df):,} bars")
    return df

def prepare_features(df):
    """Prepare 8 features matching the LSTM input size"""
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

    # 8 features to match input_size=8
    cols = ['rsi', 'macd', 'bb_middle', 'bb_std', 'ema_10', 'momentum', 'atr', 'price_change']

    for c in cols:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

    return df.dropna(), cols

def evaluate_winrate(model, features, prices):
    """Calculate actual win rate - 3 actions: BUY=0, SELL=1, HOLD=2"""
    model.eval()
    with torch.no_grad():
        output = model(features)
        actions = torch.argmax(output, dim=1).cpu().numpy()

    wins, losses = 0, 0
    position = None
    entry = 0
    entry_idx = 0
    hold_period = 5  # Bars to hold before checking profit

    for i in range(len(actions) - hold_period):
        a = actions[i]
        curr = prices[i]

        if position is None:
            if a == 0:  # BUY
                position, entry, entry_idx = 'buy', curr, i
            elif a == 1:  # SELL
                position, entry, entry_idx = 'sell', curr, i
        else:
            # Check if we've held long enough or got opposite signal
            held_long_enough = (i - entry_idx) >= hold_period
            opposite_signal = (position == 'buy' and a == 1) or (position == 'sell' and a == 0)

            if held_long_enough or opposite_signal:
                # Close position
                exit_price = curr
                if position == 'buy':
                    if exit_price > entry: wins += 1
                    else: losses += 1
                else:  # sell
                    if exit_price < entry: wins += 1
                    else: losses += 1
                position = None

                # Open new position if opposite signal
                if opposite_signal:
                    if a == 0:
                        position, entry, entry_idx = 'buy', curr, i
                    elif a == 1:
                        position, entry, entry_idx = 'sell', curr, i

    total = wins + losses
    return (wins / total * 100) if total > 0 else 0, wins, losses, total

# Test on recent 3 months of data
print("="*60)
print("CHAMPION WIN RATE TEST (LSTM Models)")
print("="*60)

# Get test data
test_df = fetch_binance('BTCUSDT', '5m', 3)
test_df, cols = prepare_features(test_df)
test_features = torch.FloatTensor(test_df[cols].values).to(DEVICE)
test_prices = test_df['close'].values

print(f"\nTest data: {len(test_df):,} bars (3 months M5)")
print("-"*60)

# Load and test each champion
champion_dir = "quantu/champions"
results = []

for filename in os.listdir(champion_dir):
    if filename.endswith('.pth'):
        symbol = filename.replace('champion_', '').replace('.pth', '')
        filepath = os.path.join(champion_dir, filename)

        try:
            # Create model with matching architecture (3 outputs: BUY, SELL, HOLD)
            model = TradingLSTM(input_size=8, hidden_size=128, num_layers=2, output_size=3)

            # Load weights
            state = torch.load(filepath, map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            model = model.to(DEVICE)

            wr, wins, losses, total = evaluate_winrate(model, test_features, test_prices)
            results.append({'symbol': symbol, 'wr': wr, 'wins': wins, 'losses': losses, 'total': total})
            print(f"  {symbol}: {wr:.1f}% ({wins}W/{losses}L = {total} trades)")

        except Exception as e:
            print(f"  {symbol}: Error - {e}")

print("-"*60)
if results:
    results.sort(key=lambda x: x['wr'], reverse=True)
    best = results[0]
    avg_wr = sum(r['wr'] for r in results) / len(results)

    print(f"\nBEST: {best['symbol']} @ {best['wr']:.1f}%")
    print(f"AVG:  {avg_wr:.1f}%")

    above_65 = [r for r in results if r['wr'] >= 65]
    above_70 = [r for r in results if r['wr'] >= 70]

    print(f"\n>= 65%: {len(above_65)} champions")
    print(f">= 70%: {len(above_70)} champions")

    if best['wr'] >= 70:
        print("\n*** TARGET MET: Ready for entropy remover! ***")
    else:
        print(f"\nGap to 70%: {70 - best['wr']:.1f}%")

print("="*60)
