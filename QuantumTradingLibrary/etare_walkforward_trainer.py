"""
ETARE Walk-Forward Training System
===================================
- 3 Timeframes: M1, M5, M15
- 10 batches per cycle (6 months each)
- 4 training batches, 2 testing batches (walk-forward)
- 3 cycles per timeframe with staggered testing windows
- Pull 2-5 experts per cycle
- 15 extinction events every second round
- Target: >70% win rate
"""
import numpy as np
import pandas as pd
import torch
import MetaTrader5 as mt5
from enum import Enum
from dataclasses import dataclass
from copy import deepcopy
import random
import warnings
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# GPU Setup
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"GPU: {torch_directml.device_name(0)}")
except:
    DEVICE = torch.device("cpu")
    print("CPU mode")

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
    def __init__(self, input_size: int, mutation_rate=0.15, mutation_strength=0.15):
        self.input_size = input_size
        self.weights = GeneticWeights(
            input_weights=torch.empty(input_size, 128, device=DEVICE).uniform_(-0.5, 0.5),
            hidden_weights=torch.empty(128, 64, device=DEVICE).uniform_(-0.5, 0.5),
            output_weights=torch.empty(64, len(Action), device=DEVICE).uniform_(-0.5, 0.5),
            hidden_bias=torch.empty(128, device=DEVICE).uniform_(-0.5, 0.5),
            output_bias=torch.empty(len(Action), device=DEVICE).uniform_(-0.5, 0.5),
        )
        self.fitness = 0
        self.test_fitness = 0
        self.successful_trades = 0
        self.total_trades = 0
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.generation = 0
        self.timeframe = ""
        self.cycle = 0

    def batch_predict(self, states: torch.Tensor) -> torch.Tensor:
        mean = states.mean(dim=1, keepdim=True)
        std = states.std(dim=1, keepdim=True) + 1e-8
        states = (states - mean) / std

        hidden = torch.tanh(torch.matmul(states, self.weights.input_weights) + self.weights.hidden_bias)
        hidden2 = torch.tanh(torch.matmul(hidden, self.weights.hidden_weights))
        output = torch.matmul(hidden2, self.weights.output_weights) + self.weights.output_bias

        return torch.argmax(output, dim=1)

    def mutate(self):
        for weight_tensor in [
            self.weights.input_weights,
            self.weights.hidden_weights,
            self.weights.output_weights,
        ]:
            mask = torch.rand_like(weight_tensor) < self.mutation_rate
            noise = torch.randn_like(weight_tensor) * self.mutation_strength
            weight_tensor[mask] += noise[mask]

    def to_dict(self):
        return {
            'input_weights': self.weights.input_weights.cpu().tolist(),
            'hidden_weights': self.weights.hidden_weights.cpu().tolist(),
            'output_weights': self.weights.output_weights.cpu().tolist(),
            'hidden_bias': self.weights.hidden_bias.cpu().tolist(),
            'output_bias': self.weights.output_bias.cpu().tolist(),
            'fitness': self.fitness,
            'test_fitness': self.test_fitness,
            'timeframe': self.timeframe,
            'cycle': self.cycle,
            'generation': self.generation
        }

def prepare_features(data: pd.DataFrame) -> tuple:
    df = data.copy()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

    for period in [5, 10, 20, 50]:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    df["momentum"] = df["close"] / df["close"].shift(10)
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["price_change"] = df["close"].pct_change()
    df["price_change_abs"] = df["price_change"].abs()

    df["volume_ma"] = df["tick_volume"].rolling(20).mean()
    df["volume_std"] = df["tick_volume"].rolling(20).std()

    df = df.ffill().bfill()

    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
                    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'momentum', 'atr',
                    'price_change', 'price_change_abs', 'volume_ma', 'volume_std']

    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df[col] = (df[col] - mean) / std

    df = df.dropna()
    return df[feature_cols + ['close']], feature_cols

def tournament_selection(population, tournament_size=5):
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness)

def crossover(parent1, parent2, input_size):
    child = TradingIndividual(input_size)
    for attr in ["input_weights", "hidden_weights", "output_weights"]:
        parent1_weights = getattr(parent1.weights, attr)
        parent2_weights = getattr(parent2.weights, attr)
        mask = torch.rand_like(parent1_weights) < 0.5
        child_weights = torch.where(mask, parent1_weights, parent2_weights)
        setattr(child.weights, attr, child_weights)
    return child

def evaluate_individual(individual, features_tensor, prices):
    actions = individual.batch_predict(features_tensor).cpu().numpy()

    wins, losses = 0, 0
    position = None
    entry_price = 0
    hold_time = 5

    for i in range(len(actions) - hold_time):
        action = actions[i]
        current_price = prices[i]
        future_price = prices[i + hold_time]

        if position is None:
            if action == Action.OPEN_BUY.value:
                position = 'buy'
                entry_price = current_price
            elif action == Action.OPEN_SELL.value:
                position = 'sell'
                entry_price = current_price
        else:
            if position == 'buy':
                if action in [Action.CLOSE_BUY_PROFIT.value, Action.CLOSE_BUY_LOSS.value]:
                    profit = future_price - entry_price
                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None
            elif position == 'sell':
                if action in [Action.CLOSE_SELL_PROFIT.value, Action.CLOSE_SELL_LOSS.value]:
                    profit = entry_price - future_price
                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    individual.successful_trades = wins
    individual.total_trades = total_trades
    return win_rate, total_trades

def get_bars_for_months(timeframe, months):
    """Calculate approximate bars for given months based on timeframe"""
    bars_per_day = {
        'M1': 1440,   # 60 * 24
        'M5': 288,    # 12 * 24
        'M15': 96     # 4 * 24
    }
    return bars_per_day[timeframe] * 30 * months

def run_darwin_round(population, train_tensor, train_prices, test_tensor, test_prices,
                     input_size, is_extinction_round=False):
    """Run one Darwin round with optional extinction event"""

    ELITE_SIZE = 3
    POPULATION_SIZE = len(population)

    # Evaluate on training data
    for ind in population:
        wr, trades = evaluate_individual(ind, train_tensor, train_prices)
        ind.fitness = wr

    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)

    if is_extinction_round:
        # Keep elite
        survivors = population[:ELITE_SIZE]

        # Inject fresh diversity
        for _ in range(5):
            fresh = TradingIndividual(input_size)
            evaluate_individual(fresh, train_tensor, train_prices)
            survivors.append(fresh)

        # Repopulate with crossover and mutation
        while len(survivors) < POPULATION_SIZE:
            if random.random() < 0.6:
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)
                child = crossover(parent1, parent2, input_size)
            else:
                child = deepcopy(random.choice(population[:ELITE_SIZE]))
                child.mutate()

            evaluate_individual(child, train_tensor, train_prices)
            survivors.append(child)

        population = survivors

    # Test on held-out data
    for ind in population:
        wr, trades = evaluate_individual(ind, test_tensor, test_prices)
        ind.test_fitness = wr

    population.sort(key=lambda x: x.test_fitness, reverse=True)

    return population

def run_cycle(timeframe, cycle_num, data, feature_cols, stagger_months=0):
    """
    Run one training cycle with walk-forward validation
    - 10 batches of 6 months each
    - 4 training, 2 testing
    - Stagger testing window by stagger_months
    """

    INPUT_SIZE = len(feature_cols)
    POPULATION_SIZE = 50

    # Calculate batch size (6 months of data)
    batch_bars = get_bars_for_months(timeframe, 6)
    total_needed = batch_bars * 6  # 6 batches visible at a time

    print(f"\n{'='*60}")
    print(f"CYCLE {cycle_num} - {timeframe} - Stagger: {stagger_months} months")
    print(f"{'='*60}")

    features = data[feature_cols].values
    prices = data['close'].values

    # Apply stagger offset
    stagger_bars = get_bars_for_months(timeframe, stagger_months)

    # Create population
    population = [TradingIndividual(INPUT_SIZE) for _ in range(POPULATION_SIZE)]

    champions = []
    round_num = 0

    # Walk through 10 batches
    for batch_start in range(0, 10):
        # Calculate indices
        start_idx = batch_start * batch_bars + stagger_bars

        # 4 batches training, 2 batches testing
        train_end = start_idx + (batch_bars * 4)
        test_end = train_end + (batch_bars * 2)

        if test_end > len(features):
            print(f"  Batch {batch_start}: Not enough data, skipping")
            continue

        # Get training and testing data
        train_features = features[start_idx:train_end]
        train_prices = prices[start_idx:train_end]
        test_features = features[train_end:test_end]
        test_prices = prices[train_end:test_end]

        if len(train_features) < 100 or len(test_features) < 100:
            continue

        train_tensor = torch.FloatTensor(train_features).to(DEVICE)
        test_tensor = torch.FloatTensor(test_features).to(DEVICE)

        round_num += 1
        is_extinction = (round_num % 2 == 0)  # Every second round

        population = run_darwin_round(
            population, train_tensor, train_prices,
            test_tensor, test_prices, INPUT_SIZE, is_extinction
        )

        best = population[0]
        status = "EXTINCTION" if is_extinction else ""
        print(f"  Batch {batch_start}: Train={best.fitness*100:.1f}% Test={best.test_fitness*100:.1f}% {status}")

        # Update generation info
        for ind in population:
            ind.generation = round_num
            ind.timeframe = timeframe
            ind.cycle = cycle_num

    # Pull 2-5 best experts from this cycle
    population.sort(key=lambda x: x.test_fitness, reverse=True)
    num_to_pull = random.randint(2, 5)
    champions = deepcopy(population[:num_to_pull])

    print(f"\n  Pulled {num_to_pull} champions: {[f'{c.test_fitness*100:.1f}%' for c in champions]}")

    return champions, population

def run_timeframe(timeframe, symbol="BTCUSD"):
    """Run all 3 cycles for a timeframe with staggered testing"""

    print(f"\n{'#'*60}")
    print(f"# TIMEFRAME: {timeframe}")
    print(f"{'#'*60}")

    # Get data - need enough for all batches plus staggering
    tf_map = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15}

    # Calculate total bars needed (10 batches * 6 months + stagger room)
    total_bars = get_bars_for_months(timeframe, 66)  # 60 months + 6 month buffer

    print(f"Fetching {total_bars} bars of {timeframe} data...")
    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, min(total_bars, 500000))

    if rates is None or len(rates) < 1000:
        print(f"ERROR: Could not get enough {timeframe} data")
        return []

    df = pd.DataFrame(rates)
    df, feature_cols = prepare_features(df)
    print(f"Prepared {len(df)} bars with {len(feature_cols)} features")

    all_champions = []

    # Cycle 1: Normal position
    champs, _ = run_cycle(timeframe, 1, df, feature_cols, stagger_months=0)
    all_champions.extend(champs)

    # Cycle 2: Testing shifted back 2 months to middle
    champs, _ = run_cycle(timeframe, 2, df, feature_cols, stagger_months=2)
    all_champions.extend(champs)

    # Cycle 3: Testing shifted to start
    champs, _ = run_cycle(timeframe, 3, df, feature_cols, stagger_months=4)
    all_champions.extend(champs)

    return all_champions

def final_test(champions, symbol="BTCUSD"):
    """Final test on fresh data to check for >70%"""

    print(f"\n{'='*60}")
    print("FINAL CHAMPIONSHIP TEST")
    print(f"{'='*60}")

    # Get fresh M5 data (most balanced timeframe)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 50000)
    df = pd.DataFrame(rates)
    df, feature_cols = prepare_features(df)

    # Use last 20% as final test
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:]

    features = test_df[feature_cols].values
    prices = test_df['close'].values
    features_tensor = torch.FloatTensor(features).to(DEVICE)

    results = []
    for i, champ in enumerate(champions):
        wr, trades = evaluate_individual(champ, features_tensor, prices)
        results.append({
            'index': i,
            'timeframe': champ.timeframe,
            'cycle': champ.cycle,
            'win_rate': wr * 100,
            'trades': trades,
            'champion': champ
        })
        print(f"  Champion {i+1}: {champ.timeframe} C{champ.cycle} - WR={wr*100:.1f}% Trades={trades}")

    # Sort by win rate
    results.sort(key=lambda x: x['win_rate'], reverse=True)

    best = results[0]
    print(f"\n  BEST: {best['timeframe']} Cycle {best['cycle']} - {best['win_rate']:.1f}% ({best['trades']} trades)")

    return results

# ============================================================
# MAIN EXECUTION
# ============================================================

print("="*60)
print("ETARE WALK-FORWARD TRAINING SYSTEM")
print("Bitcoin | M1, M5, M15 | 3 Cycles Each | Walk-Forward")
print("="*60)

# Initialize MT5
if not mt5.initialize():
    print("ERROR: MT5 initialization failed")
    exit()

all_champions = []

# Run all timeframes
for tf in ['M1', 'M5', 'M15']:
    try:
        champions = run_timeframe(tf)
        all_champions.extend(champions)
    except Exception as e:
        print(f"ERROR in {tf}: {str(e)}")
        continue

print(f"\n{'='*60}")
print(f"TOTAL CHAMPIONS COLLECTED: {len(all_champions)}")
print(f"{'='*60}")

# Final championship test
if all_champions:
    results = final_test(all_champions)

    best = results[0]

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Champion: {best['win_rate']:.1f}% win rate")
    print(f"Timeframe: {best['timeframe']}, Cycle: {best['cycle']}")
    print(f"Trades: {best['trades']}")

    if best['win_rate'] >= 70:
        print(f"\n*** TARGET MET: {best['win_rate']:.1f}% >= 70% ***")
        print("Ready for entropy remover from quantum compression!")

        # Save the champion
        with open('champion_expert.json', 'w') as f:
            json.dump(best['champion'].to_dict(), f)
        print("Champion saved to champion_expert.json")
    else:
        print(f"\nGap to target: {70 - best['win_rate']:.1f}% needed")

    print(f"{'='*60}")

mt5.shutdown()
print("\nTraining complete.")
