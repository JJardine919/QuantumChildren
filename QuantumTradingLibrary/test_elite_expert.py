import json
import numpy as np
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Load elite expert
with open('C:/Users/jjj10/ETARE_WalkForward/elite_experts_70plus/expert_C7_E49_WR73.json', 'r') as f:
    data = json.load(f)

input_weights = np.array(data['input_weights'])
hidden_weights = np.array(data['hidden_weights'])
output_weights = np.array(data['output_weights'])
hidden_bias = np.array(data.get('hidden_bias', np.zeros(64)))
output_bias = np.array(data.get('output_bias', np.zeros(3)))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def predict(features):
    hidden = relu(np.dot(features, input_weights) + hidden_bias)
    hidden2 = relu(np.dot(hidden, hidden_weights))
    output = np.dot(hidden2, output_weights) + output_bias
    probs = softmax(output)
    return np.argmax(probs), probs[np.argmax(probs)]

def calc_indicators(df):
    df = df.copy()
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()

    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
    df['bb_lower'] = df['bb_mid'] - (bb_std * 2)

    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    df['momentum'] = df['close'] - df['close'].shift(10)

    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
    df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

    df['volume_ma'] = df['tick_volume'].rolling(20).mean()
    df['volume_std'] = df['tick_volume'].rolling(20).std()
    df['price_change'] = df['close'].diff()

    return df.dropna()

def extract_features(row):
    features = [
        (row['close'] - row['ema_5']) / row['close'],
        (row['close'] - row['ema_10']) / row['close'],
        (row['close'] - row['ema_20']) / row['close'],
        (row['close'] - row['ema_50']) / row['close'],
        row['macd'] / row['close'],
        row['macd_signal'] / row['close'],
        row['macd_hist'] / row['close'],
        row['rsi'] / 100.0,
        row['stoch_k'] / 100.0,
        row['stoch_d'] / 100.0,
        np.clip(row['cci'] / 100.0, -1, 1),
        (row['williams_r'] + 100) / 100.0,
        row['roc'] / 100.0,
        (row['close'] - row['bb_mid']) / row['bb_mid'],
        (row['bb_upper'] - row['bb_lower']) / row['bb_mid'],
        row['atr'] / row['close'],
        row['momentum'] / row['close'],
        row['price_change'] / row['close'],
        (row['tick_volume'] - row['volume_ma']) / row['volume_ma'] if row['volume_ma'] > 0 else 0,
        row['volume_std'] / row['volume_ma'] if row['volume_ma'] > 0 else 0,
    ]
    return np.array(features, dtype=np.float32)

def compression_ratio(prices):
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    volatility = np.std(returns)
    trend_strength = abs(np.mean(returns)) / (volatility + 1e-8)
    return 1.0 + trend_strength * 2

print('Testing 73% Elite Expert with Compression Filter...')

mt5.initialize()
end = datetime.now()
start = end - timedelta(days=30)
rates = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M5, start, end)
mt5.shutdown()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = calc_indicators(df)

print(f'Data: {len(df)} bars')

wins_trending = 0
losses_trending = 0
wins_choppy = 0
losses_choppy = 0
skipped = 0

for i in range(256, len(df) - 12, 12):
    row = df.iloc[i]
    features = extract_features(row)
    features = np.nan_to_num(features, nan=0, posinf=1, neginf=-1)
    features = np.clip(features, -10, 10)

    prices = df['close'].iloc[i-256:i].values
    ratio = compression_ratio(prices)
    is_trending = ratio > 1.3

    action, conf = predict(features)

    if action == 2 or conf < 0.35:
        skipped += 1
        continue

    future_price = df['close'].iloc[i + 12]
    current_price = row['close']

    if action == 0:
        correct = future_price > current_price
    else:
        correct = future_price < current_price

    if is_trending:
        if correct:
            wins_trending += 1
        else:
            losses_trending += 1
    else:
        if correct:
            wins_choppy += 1
        else:
            losses_choppy += 1

total_trending = wins_trending + losses_trending
total_choppy = wins_choppy + losses_choppy
total = total_trending + total_choppy

print()
print('='*60)
print('73% ELITE EXPERT + COMPRESSION FILTER')
print('='*60)

if total_trending > 0:
    wr_trending = wins_trending / total_trending * 100
    print(f'TRENDING regime: {wins_trending}/{total_trending} = {wr_trending:.1f}%')
else:
    wr_trending = 0
    print('TRENDING: No trades')

if total_choppy > 0:
    wr_choppy = wins_choppy / total_choppy * 100
    print(f'CHOPPY regime:   {wins_choppy}/{total_choppy} = {wr_choppy:.1f}%')
else:
    wr_choppy = 0
    print('CHOPPY: No trades')

if total > 0:
    wr_total = (wins_trending + wins_choppy) / total * 100
    print(f'OVERALL:         {wins_trending + wins_choppy}/{total} = {wr_total:.1f}%')

print(f'Skipped: {skipped}')

if total_trending > 0 and total_choppy > 0:
    edge = wr_trending - wr_choppy
    print(f'\nCOMPRESSION EDGE: +{edge:.1f}%')
    print(f'\nTRADING ONLY IN TRENDING: {wr_trending:.1f}%')
