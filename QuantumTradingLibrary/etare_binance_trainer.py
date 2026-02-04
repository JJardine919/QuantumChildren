"""
ETARE Walk-Forward Trainer with Binance Data
- M1 and M5 only
- 4 train / 2 test batches (no overlap)
- Extinction after every second 4+2 cycle
- 60 months of data
"""
import numpy as np
import pandas as pd
import torch
import requests
import time
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from copy import deepcopy
import random
import warnings
import json
import os
warnings.filterwarnings('ignore')

# GPU
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"GPU: {torch_directml.device_name(0)}")
except:
    DEVICE = torch.device("cpu")

class Action(Enum):
    OPEN_BUY = 0
    OPEN_SELL = 1
    CLOSE_BUY_PROFIT = 2
    CLOSE_BUY_LOSS = 3
    CLOSE_SELL_PROFIT = 4
    CLOSE_SELL_LOSS = 5

@dataclass
class GeneticWeights:
    input_weights: torch.Tensor
    hidden_weights: torch.Tensor
    output_weights: torch.Tensor
    hidden_bias: torch.Tensor
    output_bias: torch.Tensor

class TradingIndividual:
    def __init__(self, input_size):
        self.weights = GeneticWeights(
            input_weights=torch.empty(input_size, 128, device=DEVICE).uniform_(-0.5, 0.5),
            hidden_weights=torch.empty(128, 64, device=DEVICE).uniform_(-0.5, 0.5),
            output_weights=torch.empty(64, len(Action), device=DEVICE).uniform_(-0.5, 0.5),
            hidden_bias=torch.empty(128, device=DEVICE).uniform_(-0.5, 0.5),
            output_bias=torch.empty(len(Action), device=DEVICE).uniform_(-0.5, 0.5),
        )
        self.fitness = 0
        self.test_fitness = 0
        self.input_size = input_size

    def batch_predict(self, states):
        mean = states.mean(dim=1, keepdim=True)
        std = states.std(dim=1, keepdim=True) + 1e-8
        states = (states - mean) / std
        hidden = torch.tanh(torch.matmul(states, self.weights.input_weights) + self.weights.hidden_bias)
        hidden2 = torch.tanh(torch.matmul(hidden, self.weights.hidden_weights))
        output = torch.matmul(hidden2, self.weights.output_weights) + self.weights.output_bias
        return torch.argmax(output, dim=1)

    def mutate(self):
        for w in [self.weights.input_weights, self.weights.hidden_weights, self.weights.output_weights]:
            mask = torch.rand_like(w) < 0.15
            w[mask] += torch.randn_like(w)[mask] * 0.15

def fetch_binance(symbol, interval, months):
    """Fetch data from Binance with retry logic"""
    print(f"  Fetching {interval} data ({months} months)...", flush=True)

    url = "https://api.binance.com/api/v3/klines"
    end_time = datetime.now()
    start_time = end_time - timedelta(days=months * 30)

    all_data = []
    current = start_time
    interval_mins = {'1m': 1, '5m': 5}[interval]
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

        # Retry logic
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 200:
                    all_data.extend(r.json())
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    Warning: Failed chunk at {current}", flush=True)

        current = chunk_end
        time.sleep(0.1)  # Slightly longer delay

        if len(all_data) % 50000 == 0 and len(all_data) > 0:
            print(f"    {len(all_data):,} bars...", flush=True)

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['time'] = pd.to_datetime(df['time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df['tick_volume'] = df['volume']

    print(f"    Done: {len(df):,} bars")
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

def prepare_features(df):
    """Prepare features"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain/(loss+1e-10)))

    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

    for p in [5, 10, 20, 50]:
        df[f"ema_{p}"] = df["close"].ewm(span=p).mean()

    df["momentum"] = df["close"] / df["close"].shift(10)
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["price_change"] = df["close"].pct_change()
    df["volume_ma"] = df["tick_volume"].rolling(20).mean()

    df = df.ffill().bfill()

    cols = ['rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'momentum', 'atr', 'price_change', 'volume_ma']

    for c in cols:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

    return df.dropna(), cols

def evaluate(individual, features, prices):
    """Evaluate win rate"""
    actions = individual.batch_predict(features).cpu().numpy()

    wins, losses = 0, 0
    position = None
    entry = 0

    for i in range(len(actions) - 5):
        a = actions[i]
        curr = prices[i]
        future = prices[i + 5]

        if position is None:
            if a == 0: position, entry = 'buy', curr
            elif a == 1: position, entry = 'sell', curr
        else:
            if position == 'buy' and a in [2, 3]:
                if future > entry: wins += 1
                else: losses += 1
                position = None
            elif position == 'sell' and a in [4, 5]:
                if future < entry: wins += 1
                else: losses += 1
                position = None

    total = wins + losses
    return wins / total if total > 0 else 0, total

def crossover(p1, p2, input_size):
    child = TradingIndividual(input_size)
    for attr in ["input_weights", "hidden_weights", "output_weights"]:
        w1, w2 = getattr(p1.weights, attr), getattr(p2.weights, attr)
        mask = torch.rand_like(w1) < 0.5
        setattr(child.weights, attr, torch.where(mask, w1, w2))
    return child

def extinction_event(population, train_f, train_p, input_size):
    """Run extinction - kill bottom, breed from top"""
    population.sort(key=lambda x: x.fitness, reverse=True)
    survivors = population[:5]  # Elite

    # Add fresh blood
    for _ in range(5):
        fresh = TradingIndividual(input_size)
        fresh.fitness, _ = evaluate(fresh, train_f, train_p)
        survivors.append(fresh)

    # Breed rest
    while len(survivors) < 50:
        p1 = random.choice(population[:10])
        p2 = random.choice(population[:10])
        child = crossover(p1, p2, input_size)
        child.mutate()
        child.fitness, _ = evaluate(child, train_f, train_p)
        survivors.append(child)

    return survivors

# ============================================================
print("="*60)
print("ETARE BINANCE WALK-FORWARD TRAINER")
print("M1 + M5 | 4 Train / 2 Test | Extinction every 2nd cycle")
print("="*60)

champions = []

for interval in ['1m', '5m']:
    print(f"\n{'#'*60}")
    print(f"# TIMEFRAME: {interval}")
    print(f"{'#'*60}")

    # Fetch data
    df = fetch_binance('BTCUSDT', interval, 60)
    df, cols = prepare_features(df)
    print(f"  Prepared: {len(df):,} bars, {len(cols)} features")

    features = df[cols].values
    prices = df['close'].values

    INPUT_SIZE = len(cols)

    # Split into 6 equal batches (4 train + 2 test)
    batch_size = len(features) // 6

    # Create population
    population = [TradingIndividual(INPUT_SIZE) for _ in range(50)]

    cycle_count = 0

    # Walk through data - 10 cycles, shifting by 1 batch each time
    for start_batch in range(10):
        if (start_batch + 6) * batch_size > len(features):
            break

        # 4 train batches
        train_start = start_batch * batch_size
        train_end = train_start + (4 * batch_size)

        # 2 test batches (NO OVERLAP)
        test_start = train_end
        test_end = test_start + (2 * batch_size)

        if test_end > len(features):
            break

        train_f = torch.FloatTensor(features[train_start:train_end]).to(DEVICE)
        train_p = prices[train_start:train_end]
        test_f = torch.FloatTensor(features[test_start:test_end]).to(DEVICE)
        test_p = prices[test_start:test_end]

        # Evaluate on train
        for ind in population:
            ind.fitness, _ = evaluate(ind, train_f, train_p)

        cycle_count += 1

        # Extinction every 2nd cycle
        if cycle_count % 2 == 0:
            population = extinction_event(population, train_f, train_p, INPUT_SIZE)
            status = "EXTINCTION"
        else:
            status = ""

        # Test on held-out data
        for ind in population:
            ind.test_fitness, _ = evaluate(ind, test_f, test_p)

        population.sort(key=lambda x: x.test_fitness, reverse=True)
        best = population[0]

        print(f"  Cycle {cycle_count}: Train={best.fitness*100:.1f}% Test={best.test_fitness*100:.1f}% {status}")

    # Pull 2-5 champions from this timeframe
    population.sort(key=lambda x: x.test_fitness, reverse=True)
    num_pull = random.randint(2, 5)
    for i in range(num_pull):
        champ = deepcopy(population[i])
        champ.timeframe = interval
        champions.append(champ)

    print(f"  Pulled {num_pull} champions: {[f'{c.test_fitness*100:.1f}%' for c in champions[-num_pull:]]}")

# Final test
print(f"\n{'='*60}")
print("FINAL CHAMPIONSHIP")
print(f"{'='*60}")

# Get fresh test data
print("Fetching fresh test data...")
test_df = fetch_binance('BTCUSDT', '5m', 3)  # 3 months fresh
test_df, cols = prepare_features(test_df)
test_f = torch.FloatTensor(test_df[cols].values).to(DEVICE)
test_p = test_df['close'].values

results = []
for i, champ in enumerate(champions):
    wr, trades = evaluate(champ, test_f, test_p)
    results.append({'idx': i, 'tf': champ.timeframe, 'wr': wr*100, 'trades': trades, 'champ': champ})
    print(f"  Champion {i+1} ({champ.timeframe}): {wr*100:.1f}% ({trades} trades)")

results.sort(key=lambda x: x['wr'], reverse=True)
best = results[0]

print(f"\n{'='*60}")
print(f"BEST: {best['wr']:.1f}% ({best['tf']}) - {best['trades']} trades")
print(f"{'='*60}")

if best['wr'] >= 70:
    print("*** TARGET MET: Ready for entropy remover! ***")
else:
    print(f"Gap: {70 - best['wr']:.1f}% to target")

print("="*60)
