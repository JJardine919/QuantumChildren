"""
ETARE Walk-Forward Trainer v2 - FIXED
- M1 and M5 only
- 4 train / 2 test batches (no overlap)
- Extinction after every second 4+2 cycle
- 60 months of data, 10 walk-forward windows
- SAVES 65%+ experts to keepers folder during extinction
"""
import numpy as np
import pandas as pd
import torch
import requests
import time
import os
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from copy import deepcopy
import random
import warnings
import sys
import json
warnings.filterwarnings('ignore')

# Create keepers folder
KEEPERS_DIR = "quantu/keepers"
os.makedirs(KEEPERS_DIR, exist_ok=True)

# Force unbuffered output
def print_flush(*args, **kwargs):
    sys.stdout.write(' '.join(map(str, args)) + '\n')
    sys.stdout.flush()
print = print_flush

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
        self.timeframe = ""

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
            mask = torch.rand_like(w) < 0.2
            w[mask] += torch.randn_like(w)[mask] * 0.2

def fetch_binance(symbol, interval, months):
    """Fetch data from Binance with retry"""
    print(f"  Fetching {interval} data ({months} months)...")

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

        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 200:
                    all_data.extend(r.json())
                    break
            except:
                if attempt < 2:
                    time.sleep(2)

        current = chunk_end
        time.sleep(0.05)  # Faster requests

        if len(all_data) % 10000 == 0 and len(all_data) > 0:
            print(f"    {len(all_data):,} bars...")

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

def save_keeper(individual, win_rate, timeframe, cycle_num):
    """Save a 65%+ expert to keepers folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"keeper_{timeframe}_{win_rate:.1f}pct_{timestamp}.pth"
    filepath = os.path.join(KEEPERS_DIR, filename)

    state = {
        'input_weights': individual.weights.input_weights.cpu(),
        'hidden_weights': individual.weights.hidden_weights.cpu(),
        'output_weights': individual.weights.output_weights.cpu(),
        'hidden_bias': individual.weights.hidden_bias.cpu(),
        'output_bias': individual.weights.output_bias.cpu(),
        'input_size': individual.input_size,
        'win_rate': win_rate,
        'timeframe': timeframe,
        'cycle': cycle_num,
        'saved_at': timestamp
    }
    torch.save(state, filepath)
    print(f"      *** KEEPER SAVED: {filename} ({win_rate:.1f}%) ***")

def extinction_event(population, train_f, train_p, input_size, timeframe="", cycle_num=0):
    """15 extinction cycles - saves 65%+ experts to keepers"""
    print("    >>> EXTINCTION EVENT <<<")

    keepers_found = 0

    for ext_round in range(15):
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Check for keepers (65%+) before extinction
        for ind in population:
            if ind.fitness >= 0.65:
                save_keeper(ind, ind.fitness * 100, timeframe, cycle_num)
                keepers_found += 1

        survivors = population[:3]  # Elite 3

        # Fresh diversity
        for _ in range(7):
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

        population = survivors

    if keepers_found > 0:
        print(f"    Saved {keepers_found} keepers (65%+) this extinction")

    return population

# ============================================================
print("="*60)
print("ETARE BINANCE WALK-FORWARD TRAINER v2")
print("M1 + M5 | 4 Train / 2 Test | 15 Extinctions every 2nd")
print("="*60)

champions = []

for interval in ['5m']:  # M5 only - better historical coverage on Binance
    print(f"\n{'#'*60}")
    print(f"# TIMEFRAME: {interval}")
    print(f"{'#'*60}")

    df = fetch_binance('BTCUSDT', interval, 36)  # 36 months for M5
    df, cols = prepare_features(df)
    print(f"  Prepared: {len(df):,} bars, {len(cols)} features")

    features = df[cols].values
    prices = df['close'].values

    INPUT_SIZE = len(cols)

    # 10 batches of 6 months each = walk forward
    # Each window: 4 months train, 2 months test (no overlap)
    # Window slides by 1 month each cycle

    total_bars = len(features)
    month_bars = total_bars // 36  # Approx bars per month (36 months of data)

    print(f"  ~{month_bars:,} bars per month")

    population = [TradingIndividual(INPUT_SIZE) for _ in range(50)]

    cycle_count = 0

    # 10 walk-forward cycles, sliding 1 month each time
    for cycle in range(10):
        # 4 months train, 2 months test
        train_start = cycle * month_bars
        train_end = train_start + (4 * month_bars)
        test_start = train_end  # NO OVERLAP
        test_end = test_start + (2 * month_bars)

        if test_end > total_bars:
            print(f"  Cycle {cycle+1}: Not enough data, stopping")
            break

        train_f = torch.FloatTensor(features[train_start:train_end]).to(DEVICE)
        train_p = prices[train_start:train_end]
        test_f = torch.FloatTensor(features[test_start:test_end]).to(DEVICE)
        test_p = prices[test_start:test_end]

        # Evaluate on train
        for ind in population:
            ind.fitness, _ = evaluate(ind, train_f, train_p)

        cycle_count += 1

        # Extinction every 2nd cycle (15 internal extinction rounds)
        if cycle_count % 2 == 0:
            population = extinction_event(population, train_f, train_p, INPUT_SIZE, interval, cycle_count)

        # Test on held-out
        for ind in population:
            ind.test_fitness, _ = evaluate(ind, test_f, test_p)
            ind.timeframe = interval  # Tag with timeframe

            # Save keepers hitting 65%+ on TEST data (more valuable)
            if ind.test_fitness >= 0.65:
                save_keeper(ind, ind.test_fitness * 100, interval, cycle_count)

        population.sort(key=lambda x: x.test_fitness, reverse=True)
        best = population[0]

        ext_status = "EXTINCTION" if cycle_count % 2 == 0 else ""
        print(f"  Cycle {cycle_count}: Train={best.fitness*100:.1f}% Test={best.test_fitness*100:.1f}% {ext_status}")

    # Pull 2-5 champions
    population.sort(key=lambda x: x.test_fitness, reverse=True)
    num_pull = random.randint(2, 5)
    for i in range(num_pull):
        champ = deepcopy(population[i])
        champ.timeframe = interval
        champions.append(champ)

    print(f"  Pulled {num_pull}: {[f'{c.test_fitness*100:.1f}%' for c in champions[-num_pull:]]}")

# Final championship
print(f"\n{'='*60}")
print("FINAL CHAMPIONSHIP TEST")
print(f"{'='*60}")

test_df = fetch_binance('BTCUSDT', '5m', 3)
test_df, cols = prepare_features(test_df)
test_f = torch.FloatTensor(test_df[cols].values).to(DEVICE)
test_p = test_df['close'].values

results = []
for i, champ in enumerate(champions):
    wr, trades = evaluate(champ, test_f, test_p)
    results.append({'idx': i, 'tf': champ.timeframe, 'wr': wr*100, 'trades': trades})
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

# Summary of keepers
keeper_files = [f for f in os.listdir(KEEPERS_DIR) if f.endswith('.pth')]
print(f"\n{'='*60}")
print(f"KEEPERS SAVED: {len(keeper_files)} experts (65%+)")
for kf in keeper_files:
    print(f"  - {kf}")
print("="*60)

# Save manifest
manifest = {
    'trained_at': datetime.now().isoformat(),
    'best_champion': {'win_rate': best['wr'], 'timeframe': best['tf'], 'trades': best['trades']},
    'total_champions': len(champions),
    'keepers_saved': len(keeper_files),
    'keepers': keeper_files
}
with open(os.path.join(KEEPERS_DIR, 'training_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

print("Training complete!")
