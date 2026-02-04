"""
ETARE 50 Experts - FULL DARWIN EVOLUTION
More data, more extinction events, tighter selection
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
    def __init__(self, input_size: int):
        self.weights = GeneticWeights(
            input_weights=torch.empty(input_size, 128, device=DEVICE).uniform_(-0.5, 0.5),
            hidden_weights=torch.empty(128, 64, device=DEVICE).uniform_(-0.5, 0.5),
            output_weights=torch.empty(64, len(Action), device=DEVICE).uniform_(-0.5, 0.5),
            hidden_bias=torch.empty(128, device=DEVICE).uniform_(-0.5, 0.5),
            output_bias=torch.empty(len(Action), device=DEVICE).uniform_(-0.5, 0.5),
        )
        self.fitness = 0
        self.successful_trades = 0
        self.total_trades = 0
        self.mutation_rate = 0.15  # Slightly higher mutation
        self.mutation_strength = 0.15

    def batch_predict(self, states: torch.Tensor) -> torch.Tensor:
        """Batch prediction on GPU"""
        mean = states.mean(dim=1, keepdim=True)
        std = states.std(dim=1, keepdim=True) + 1e-8
        states = (states - mean) / std

        hidden = torch.tanh(torch.matmul(states, self.weights.input_weights) + self.weights.hidden_bias)
        hidden2 = torch.tanh(torch.matmul(hidden, self.weights.hidden_weights))
        output = torch.matmul(hidden2, self.weights.output_weights) + self.weights.output_bias

        return torch.argmax(output, dim=1)

    def mutate(self):
        if np.random.random() < self.mutation_rate:
            for weight_tensor in [
                self.weights.input_weights,
                self.weights.hidden_weights,
                self.weights.output_weights,
            ]:
                mask = torch.rand_like(weight_tensor) < 0.15
                noise = torch.randn_like(weight_tensor) * self.mutation_strength
                weight_tensor[mask] += noise[mask]

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
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

def tournament_selection(population, tournament_size=5):  # Increased pressure
    tournament = random.sample(population, tournament_size)
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
    individual.fitness = win_rate
    return win_rate, total_trades

print("="*60)
print("ETARE 50 EXPERTS - FULL DARWIN EVOLUTION")
print("More data | More extinction | Tighter selection")
print("="*60)

# Init MT5
mt5.initialize()
rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M5, 0, 8000)  # More data
df = pd.DataFrame(rates)
df, feature_cols = prepare_features(df)
print(f"Data: {len(df)} bars, Features: {len(feature_cols)}")

features = df[feature_cols].values
prices = df['close'].values
features_tensor = torch.FloatTensor(features).to(DEVICE)

INPUT_SIZE = len(feature_cols)
POPULATION_SIZE = 50
ELITE_SIZE = 3  # Tighter elite selection
EXTINCTION_ROUNDS = 10  # More rounds

# Create initial population
print(f"\nCreating {POPULATION_SIZE} experts...")
population = [TradingIndividual(INPUT_SIZE) for _ in range(POPULATION_SIZE)]

# Initial evaluation
print("Evaluating baseline (NO TRAINING)...")
for i, ind in enumerate(population):
    wr, trades = evaluate_individual(ind, features_tensor, prices)
    if (i+1) % 10 == 0:
        print(f"Evaluated {i+1}/50 - Best so far: {max(ind.fitness for ind in population[:i+1])*100:.1f}%")

all_wr = [ind.fitness * 100 for ind in population]
print(f"\nInitial baseline: Avg={np.mean(all_wr):.1f}%, Best={np.max(all_wr):.1f}%")

# Darwin selection - 10 rounds
print(f"\nRunning {EXTINCTION_ROUNDS} Darwin Extinction Events...")

for round_num in range(EXTINCTION_ROUNDS):
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)

    # Keep only top elite
    survivors = population[:ELITE_SIZE]

    # Repopulate with crossover and mutation
    while len(survivors) < POPULATION_SIZE:
        if random.random() < 0.7:  # 70% crossover
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover(parent1, parent2, INPUT_SIZE)
        else:  # 30% elite mutation
            child = deepcopy(random.choice(population[:ELITE_SIZE]))
            child.mutate()

        # Evaluate child
        evaluate_individual(child, features_tensor, prices)
        survivors.append(child)

    population = survivors

    best = max(population, key=lambda x: x.fitness)
    all_wr = [ind.fitness * 100 for ind in population]
    if (round_num + 1) % 2 == 0:
        print(f"Extinction {round_num + 1}: Best={best.fitness*100:.1f}%, Avg={np.mean(all_wr):.1f}%, Trades={best.total_trades}")

# Final results
population.sort(key=lambda x: x.fitness, reverse=True)
all_wr = [ind.fitness * 100 for ind in population]
all_trades = [ind.total_trades for ind in population]

print(f"\n{'='*60}")
print(f"FINAL RESULTS - {EXTINCTION_ROUNDS} DARWIN EXTINCTION EVENTS")
print(f"{'='*60}")
print(f"Average Win Rate: {np.mean(all_wr):.1f}%")
print(f"Best Win Rate: {np.max(all_wr):.1f}%")
print(f"Worst Win Rate: {np.min(all_wr):.1f}%")
print(f"Best Expert Trades: {population[0].total_trades}")
print(f"Top 5 Win Rates: {[f'{wr:.1f}%' for wr in all_wr[:5]]}")
print(f"{'='*60}")
print(f"TARGET: 63-65%")
if 63 <= np.max(all_wr) <= 65:
    print("TARGET MET!")
elif np.max(all_wr) >= 63:
    print(f"ABOVE TARGET: {np.max(all_wr):.1f}%")
else:
    print(f"Gap: {63 - np.max(all_wr):.1f}% needed")
print(f"{'='*60}")

mt5.shutdown()
