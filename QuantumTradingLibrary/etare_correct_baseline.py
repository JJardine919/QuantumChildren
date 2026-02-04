"""
ETARE 50 Experts - CORRECT ARCHITECTURE from ETARE_module.py
Feedforward network, 6 actions, Darwin selection
Testing baseline BEFORE training
"""
import numpy as np
import pandas as pd
import torch
import MetaTrader5 as mt5
from enum import Enum
from dataclasses import dataclass
from copy import deepcopy
import random

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
        self.total_profit = 0
        self.successful_trades = 0
        self.total_trades = 0
        self.epsilon = 0.1
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1

    def predict(self, state: np.ndarray) -> tuple:
        state_tensor = torch.FloatTensor(state).to(DEVICE)

        # Normalize
        mean = state_tensor.mean()
        std = state_tensor.std() + 1e-8
        state_tensor = (state_tensor - mean) / std

        # Forward pass (feedforward)
        hidden = torch.tanh(torch.matmul(state_tensor, self.weights.input_weights) + self.weights.hidden_bias)
        hidden2 = torch.tanh(torch.matmul(hidden, self.weights.hidden_weights))
        output = torch.matmul(hidden2, self.weights.output_weights) + self.weights.output_bias

        # Softmax
        probabilities = torch.softmax(output, dim=-1)
        probs_np = probabilities.cpu().detach().numpy()

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action = Action(np.random.randint(len(Action)))
        else:
            action = Action(np.argmax(probs_np))

        return action, probs_np

    def mutate(self):
        if np.random.random() < self.mutation_rate:
            for weight_tensor in [
                self.weights.input_weights,
                self.weights.hidden_weights,
                self.weights.output_weights,
            ]:
                mask = torch.rand_like(weight_tensor) < 0.1
                noise = torch.randn_like(weight_tensor) * self.mutation_strength
                weight_tensor[mask] += noise[mask]

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features - matching ETARE_module.py"""
    df = data.copy()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

    # EMAs
    for period in [5, 10, 20, 50]:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    # Momentum & ATR
    df["momentum"] = df["close"] / df["close"].shift(10)
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["price_change"] = df["close"].pct_change()
    df["price_change_abs"] = df["price_change"].abs()

    # Volume
    df["volume_ma"] = df["tick_volume"].rolling(20).mean()
    df["volume_std"] = df["tick_volume"].rolling(20).std()

    df = df.ffill().bfill()

    # Normalization
    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
                    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'momentum', 'atr',
                    'price_change', 'price_change_abs', 'volume_ma', 'volume_std']

    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df[col] = (df[col] - mean) / std

    df = df.dropna()
    return df[feature_cols + ['close']], feature_cols

def tournament_selection(population, tournament_size=3):
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

def evaluate_individual(individual, features, prices):
    """Evaluate trading performance"""
    wins, losses = 0, 0
    position = None  # None, 'buy', or 'sell'
    entry_price = 0

    for i in range(len(features) - 5):
        state = features[i]
        action, _ = individual.predict(state)
        current_price = prices[i]
        future_price = prices[min(i + 5, len(prices) - 1)]

        if position is None:
            if action == Action.OPEN_BUY:
                position = 'buy'
                entry_price = current_price
            elif action == Action.OPEN_SELL:
                position = 'sell'
                entry_price = current_price
        else:
            # Check close conditions
            if position == 'buy':
                profit = future_price - entry_price
                if action in [Action.CLOSE_BUY_PROFIT, Action.CLOSE_BUY_LOSS]:
                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None
            elif position == 'sell':
                profit = entry_price - future_price
                if action in [Action.CLOSE_SELL_PROFIT, Action.CLOSE_SELL_LOSS]:
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
print("ETARE 50 EXPERTS - CORRECT ARCHITECTURE BASELINE")
print("Feedforward Network | 6 Actions | Darwin Selection")
print("="*60)

# Init MT5
mt5.initialize()
rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M5, 0, 10000)
df = pd.DataFrame(rates)
df, feature_cols = prepare_features(df)
print(f"Data: {len(df)} bars, Features: {len(feature_cols)}")

features = df[feature_cols].values
prices = df['close'].values

INPUT_SIZE = len(feature_cols)
POPULATION_SIZE = 50
ELITE_SIZE = 5
EXTINCTION_RATE = 0.3

# Create initial population
print(f"\nCreating {POPULATION_SIZE} experts...")
population = [TradingIndividual(INPUT_SIZE) for _ in range(POPULATION_SIZE)]

# Evaluate initial population (NO TRAINING - just random weights)
print("Evaluating baseline (NO TRAINING)...")
all_wr = []
all_trades = []

for i, ind in enumerate(population):
    wr, trades = evaluate_individual(ind, features, prices)
    all_wr.append(wr * 100)
    all_trades.append(trades)
    if (i+1) % 10 == 0:
        print(f"Evaluated {i+1}/50 - WR: {wr*100:.1f}% - Trades: {trades}")

print(f"\n{'='*60}")
print(f"BASELINE RESULTS (50 EXPERTS - NO TRAINING)")
print(f"{'='*60}")
print(f"Average Win Rate: {np.mean(all_wr):.1f}%")
print(f"Best Win Rate: {np.max(all_wr):.1f}%")
print(f"Average Trades: {np.mean(all_trades):.0f}")
print(f"{'='*60}")

# Now run Darwin selection rounds
print(f"\nRunning Darwin Selection (3 rounds)...")

for round_num in range(3):
    print(f"\n--- EXTINCTION EVENT {round_num + 1} ---")

    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)

    # Keep elite
    survivors = population[:ELITE_SIZE]

    # Repopulate with crossover and mutation
    while len(survivors) < POPULATION_SIZE:
        if random.random() < 0.8:  # 80% crossover
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover(parent1, parent2, INPUT_SIZE)
        else:  # 20% elite mutation
            child = deepcopy(random.choice(population[:ELITE_SIZE]))
            child.mutate()

        # Evaluate child
        wr, trades = evaluate_individual(child, features, prices)
        survivors.append(child)

    population = survivors

    # Re-evaluate all
    all_wr = []
    for ind in population:
        wr, trades = evaluate_individual(ind, features, prices)
        all_wr.append(wr * 100)

    best = max(population, key=lambda x: x.fitness)
    print(f"Best after extinction: WR={best.fitness*100:.1f}%, Trades={best.total_trades}")
    print(f"Population avg: {np.mean(all_wr):.1f}%")

# Final results
population.sort(key=lambda x: x.fitness, reverse=True)
all_wr = [ind.fitness * 100 for ind in population]
all_trades = [ind.total_trades for ind in population]

print(f"\n{'='*60}")
print(f"FINAL RESULTS - DARWIN SELECTION (NO BACKPROP TRAINING)")
print(f"{'='*60}")
print(f"Average Win Rate: {np.mean(all_wr):.1f}%")
print(f"Best Win Rate: {np.max(all_wr):.1f}%")
print(f"Worst Win Rate: {np.min(all_wr):.1f}%")
print(f"Best Expert Trades: {population[0].total_trades}")
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
