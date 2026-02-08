"""
ETARE BTCUSD Trainer - True Blueprint Build
=============================================
Faithful to the original ETARE architecture:
- Numpy feedforward NN (20->128->64->6)
- 6-action space with grid trading simulation
- Q-learning with experience replay
- Genetic evolution with extinction events
- Walk-forward on BTCUSD M5 data from MT5

Run: .venv311\Scripts\python.exe ETARE_BTCUSD_Trainer.py
"""

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import sqlite3
import json
import logging
import os
import random
import time
from collections import deque
from copy import deepcopy
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from quantum_feature_defs import QUANTUM_FEATURE_NAMES, QUANTUM_FEATURE_DEFAULTS, QUANTUM_FEATURE_COUNT
from quantum_feature_fetcher import load_quantum_features_bulk

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("etare_btcusd_trainer.log"),
        logging.StreamHandler(),
    ],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
NUM_BARS = 5000
INPUT_SIZE = 27  # 20 technical + 7 quantum features
HIDDEN1 = 128
HIDDEN2 = 64
NUM_ACTIONS = 6

POPULATION_SIZE = 50
TOURNAMENT_SIZE = 3
ELITE_SIZE = 5
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
EXTINCTION_RATE = 0.3
EXTINCTION_INTERVAL = 10

MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 0.1

NUM_GENERATIONS = 50

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "ETARE_QuantumFusion", "models")
DB_PATH = os.path.join(SCRIPT_DIR, "etare_btcusd.db")
EXPORT_PATH = os.path.join(MODEL_DIR, "btcusd_etare_expert.json")


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------
class Action(Enum):
    OPEN_BUY = 0
    OPEN_SELL = 1
    CLOSE_BUY_PROFIT = 2
    CLOSE_BUY_LOSS = 3
    CLOSE_SELL_PROFIT = 4
    CLOSE_SELL_LOSS = 5


# ---------------------------------------------------------------------------
# Numpy feedforward weights
# ---------------------------------------------------------------------------
@dataclass
class GeneticWeights:
    input_weights: np.ndarray   # (INPUT_SIZE, HIDDEN1)
    hidden_weights: np.ndarray  # (HIDDEN1, HIDDEN2)
    output_weights: np.ndarray  # (HIDDEN2, NUM_ACTIONS)
    hidden_bias: np.ndarray     # (HIDDEN1,)
    hidden2_bias: np.ndarray    # (HIDDEN2,)
    output_bias: np.ndarray     # (NUM_ACTIONS,)


def random_weights() -> GeneticWeights:
    return GeneticWeights(
        input_weights=np.random.uniform(-0.5, 0.5, (INPUT_SIZE, HIDDEN1)),
        hidden_weights=np.random.uniform(-0.5, 0.5, (HIDDEN1, HIDDEN2)),
        output_weights=np.random.uniform(-0.5, 0.5, (HIDDEN2, NUM_ACTIONS)),
        hidden_bias=np.random.uniform(-0.5, 0.5, (HIDDEN1,)),
        hidden2_bias=np.random.uniform(-0.5, 0.5, (HIDDEN2,)),
        output_bias=np.random.uniform(-0.5, 0.5, (NUM_ACTIONS,)),
    )


# ---------------------------------------------------------------------------
# Experience replay
# ---------------------------------------------------------------------------
class RLMemory:
    def __init__(self, capacity: int = MEMORY_CAPACITY):
        self.memory: deque = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size: int) -> list:
        if len(self.memory) < batch_size:
            return list(self.memory)
        return random.sample(list(self.memory), batch_size)

    def __len__(self):
        return len(self.memory)


# ---------------------------------------------------------------------------
# Grid parameters (evolved per individual)
# ---------------------------------------------------------------------------
class GridParameters:
    def __init__(self):
        # BTCUSD-scale: price ~100k, so grid steps in dollars
        self.grid_step = np.random.uniform(50.0, 500.0)
        self.orders_count = np.random.randint(3, 10)
        self.base_volume = np.random.uniform(0.01, 0.05)
        self.volume_step = np.random.uniform(0.005, 0.02)
        self.profit_target = np.random.uniform(100.0, 1000.0)
        self.loss_limit = np.random.uniform(200.0, 1500.0)

    def mutate(self):
        self.grid_step = max(20.0, min(800.0, self.grid_step + np.random.normal(0, 50.0)))
        self.orders_count = max(2, min(12, self.orders_count + np.random.randint(-1, 2)))
        self.base_volume = max(0.01, min(0.1, self.base_volume + np.random.normal(0, 0.005)))
        self.volume_step = max(0.001, min(0.05, self.volume_step + np.random.normal(0, 0.003)))
        self.profit_target = max(50.0, min(2000.0, self.profit_target + np.random.normal(0, 100.0)))
        self.loss_limit = max(100.0, min(3000.0, self.loss_limit + np.random.normal(0, 150.0)))

    def to_dict(self) -> dict:
        return {
            "grid_step": float(self.grid_step),
            "orders_count": int(self.orders_count),
            "base_volume": float(self.base_volume),
            "volume_step": float(self.volume_step),
            "profit_target": float(self.profit_target),
            "loss_limit": float(self.loss_limit),
        }

    @staticmethod
    def from_dict(d: dict) -> "GridParameters":
        gp = GridParameters()
        gp.grid_step = d["grid_step"]
        gp.orders_count = d["orders_count"]
        gp.base_volume = d["base_volume"]
        gp.volume_step = d["volume_step"]
        gp.profit_target = d["profit_target"]
        gp.loss_limit = d["loss_limit"]
        return gp


# ---------------------------------------------------------------------------
# Simulated grid order / position
# ---------------------------------------------------------------------------
@dataclass
class SimOrder:
    """A pending grid order that hasn't filled yet."""
    direction: str  # "buy" or "sell"
    price: float
    volume: float


@dataclass
class SimPosition:
    """A filled position."""
    direction: str
    entry_price: float
    volume: float
    pnl: float = 0.0


# ---------------------------------------------------------------------------
# GridTrader individual
# ---------------------------------------------------------------------------
class GridTrader:
    def __init__(self):
        self.weights = random_weights()
        self.grid_params = GridParameters()
        self.memory = RLMemory()

        # Per-evaluation state (reset each generation)
        self.pending_orders: List[SimOrder] = []
        self.open_positions: List[SimPosition] = []
        self.closed_pnls: List[float] = []
        self.total_profit = 0.0
        self.wins = 0
        self.losses = 0
        self.fitness = 0.0
        self.max_drawdown = 0.0

    # --- Forward pass ---
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (q_values, hidden1_out, hidden2_out, state_used)."""
        s = state.reshape(1, -1)
        h1 = np.tanh(s @ self.weights.input_weights + self.weights.hidden_bias)
        h2 = np.tanh(h1 @ self.weights.hidden_weights + self.weights.hidden2_bias)
        q = h2 @ self.weights.output_weights + self.weights.output_bias
        return q, h1, h2, s

    def predict(self, state: np.ndarray) -> Tuple[Action, np.ndarray]:
        q, _, _, _ = self.forward(state)
        probs = _softmax(q[0])
        if np.random.random() < EPSILON:
            action = Action(np.random.randint(NUM_ACTIONS))
        else:
            action = Action(int(np.argmax(probs)))
        return action, probs

    # --- Q-learning update ---
    def store_experience(self, state: np.ndarray, action: Action, reward: float, next_state: np.ndarray):
        """Store experience. Only store non-trivial transitions to keep memory meaningful."""
        self.memory.add(state, action.value, reward, next_state)

    def train_from_memory(self, n_batches: int = 4):
        """Train on multiple batches from replay memory."""
        if len(self.memory) < BATCH_SIZE:
            return
        for _ in range(n_batches):
            batch = self.memory.sample(BATCH_SIZE)
            self._train_batch(batch)

    def _train_batch(self, batch: list):
        for (s, a, r, ns) in batch:
            q, h1, h2, s_2d = self.forward(s)
            nq, _, _, _ = self.forward(ns)
            target = q.copy()
            target[0, a] = r + GAMMA * np.max(nq)
            self._backprop(s_2d, h1, h2, q, target)

    def _backprop(self, s: np.ndarray, h1: np.ndarray, h2: np.ndarray,
                  q: np.ndarray, target: np.ndarray):
        # Output layer error
        out_err = (target - q) * LEARNING_RATE
        # Hidden2 error  (tanh derivative = 1 - h^2)
        h2_err = (out_err @ self.weights.output_weights.T) * (1 - h2 * h2)
        # Hidden1 error
        h1_err = (h2_err @ self.weights.hidden_weights.T) * (1 - h1 * h1)

        # Weight updates
        self.weights.output_weights += h2.T @ out_err
        self.weights.output_bias += out_err.sum(axis=0)
        self.weights.hidden_weights += h1.T @ h2_err
        self.weights.hidden2_bias += h2_err.sum(axis=0)
        self.weights.input_weights += s.T @ h1_err
        self.weights.hidden_bias += h1_err.sum(axis=0)

    # --- Mutation ---
    def mutate(self):
        for w in [self.weights.input_weights, self.weights.hidden_weights,
                  self.weights.output_weights]:
            mask = np.random.random(w.shape) < 0.1
            w[mask] += np.random.normal(0, 0.1, size=int(mask.sum()))
        for b in [self.weights.hidden_bias, self.weights.hidden2_bias,
                  self.weights.output_bias]:
            mask = np.random.random(b.shape) < 0.1
            b[mask] += np.random.normal(0, 0.1, size=int(mask.sum()))
        self.grid_params.mutate()

    # --- Reset per-generation state ---
    def reset_eval(self):
        self.pending_orders.clear()
        self.open_positions.clear()
        self.closed_pnls.clear()
        self.total_profit = 0.0
        self.wins = 0
        self.losses = 0
        self.max_drawdown = 0.0

    # --- Serialization ---
    def to_dict(self) -> dict:
        return {
            "input_weights": self.weights.input_weights.tolist(),
            "hidden_weights": self.weights.hidden_weights.tolist(),
            "output_weights": self.weights.output_weights.tolist(),
            "hidden_bias": self.weights.hidden_bias.tolist(),
            "hidden2_bias": self.weights.hidden2_bias.tolist(),
            "output_bias": self.weights.output_bias.tolist(),
            "grid_params": self.grid_params.to_dict(),
            "fitness": float(self.fitness),
        }

    @staticmethod
    def from_dict(d: dict) -> "GridTrader":
        gt = GridTrader()
        gt.weights.input_weights = np.array(d["input_weights"])
        gt.weights.hidden_weights = np.array(d["hidden_weights"])
        gt.weights.output_weights = np.array(d["output_weights"])
        gt.weights.hidden_bias = np.array(d["hidden_bias"])
        gt.weights.hidden2_bias = np.array(d["hidden2_bias"])
        gt.weights.output_bias = np.array(d["output_bias"])
        gt.grid_params = GridParameters.from_dict(d["grid_params"])
        gt.fitness = d.get("fitness", 0.0)
        return gt


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ---------------------------------------------------------------------------
# Feature preparation (20 indicators, faithful to original ETARE)
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame, quantum_array: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate 20 technical + 7 quantum features and return (features_matrix, close_prices).
    Features are z-score normalized and clipped to [-4, 4].
    Rows with NaN are dropped from the front.

    Args:
        df: DataFrame with OHLCV data
        quantum_array: Optional (n_rows, 7) array of quantum features aligned to df rows.
                       If None, fills with defaults.
    """
    d = df.copy()

    # --- RSI (14) ---
    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    d["rsi"] = 100 - (100 / (1 + rs))

    # --- MACD (12-26-9) ---
    exp12 = d["close"].ewm(span=12, adjust=False).mean()
    exp26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = exp12 - exp26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands ---
    d["bb_middle"] = d["close"].rolling(20).mean()
    d["bb_std"] = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_middle"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_middle"] - 2 * d["bb_std"]

    # --- EMAs ---
    for p in [5, 10, 20, 50]:
        d[f"ema_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

    # --- Momentum ---
    d["momentum"] = d["close"] / d["close"].shift(10)

    # --- ATR (14) ---
    d["atr"] = d["high"].rolling(14).max() - d["low"].rolling(14).min()

    # --- Price changes ---
    d["price_change"] = d["close"].pct_change()
    d["price_change_abs"] = d["price_change"].abs()

    # --- Volume ---
    d["volume_ma"] = d["tick_volume"].rolling(20).mean()
    d["volume_std"] = d["tick_volume"].rolling(20).std()

    # --- Stochastic K/D ---
    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = 100 * (d["close"] - low14) / (high14 - low14 + 1e-10)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # --- ROC ---
    d["roc"] = d["close"].pct_change(10) * 100

    # Drop NaN rows
    d = d.dropna()

    # 20 technical feature columns in canonical order
    feature_cols = [
        "rsi", "macd", "macd_signal",
        "bb_middle", "bb_upper", "bb_lower", "bb_std",
        "ema_5", "ema_10", "ema_20", "ema_50",
        "momentum", "atr",
        "price_change", "price_change_abs",
        "volume_ma", "volume_std",
        "stoch_k", "stoch_d",
        "roc",
    ]

    tech_features = d[feature_cols].values.astype(np.float64)
    n_rows = tech_features.shape[0]

    # Align and concatenate quantum features (7 columns)
    if quantum_array is not None and quantum_array.shape[0] >= len(df):
        # quantum_array was aligned to original df; slice to match after dropna
        # Use the last n_rows of quantum_array (NaN rows dropped from front)
        q = quantum_array[-n_rows:].astype(np.float64)
    elif quantum_array is not None and quantum_array.shape[0] == n_rows:
        q = quantum_array.astype(np.float64)
    else:
        # Fill with defaults
        defaults = np.array([QUANTUM_FEATURE_DEFAULTS[n] for n in QUANTUM_FEATURE_NAMES])
        q = np.tile(defaults, (n_rows, 1))

    features = np.concatenate([tech_features, q], axis=1)  # (n_rows, 27)

    # Z-score normalize per column (all 27)
    means = features.mean(axis=0)
    stds = features.std(axis=0) + 1e-8
    features = (features - means) / stds
    features = np.clip(features, -4.0, 4.0)

    close_prices = d["close"].values.astype(np.float64)

    return features, close_prices


# ---------------------------------------------------------------------------
# MT5 data fetching
# ---------------------------------------------------------------------------
def init_mt5() -> bool:
    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        return False
    logging.info("MT5 initialized")
    return True


def fetch_data(symbol: str = SYMBOL, bars: int = NUM_BARS) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, bars)
    if rates is None or len(rates) < 200:
        logging.error(f"Failed to fetch data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    logging.info(f"Fetched {len(df)} bars for {symbol}")
    return df


# ---------------------------------------------------------------------------
# Grid trading simulation (bar-by-bar)
# ---------------------------------------------------------------------------
def simulate_individual(individual: GridTrader, features: np.ndarray,
                        prices: np.ndarray) -> GridTrader:
    """
    Walk through bars, let the individual trade via simulated grid.
    Stores experiences and trains Q-learning periodically (not every bar).
    """
    individual.reset_eval()
    equity_peak = 0.0
    equity = 0.0
    ql_interval = 200  # Train Q-learning every N bars

    n_bars = len(features) - 1
    for i in range(n_bars):
        state = features[i]
        next_state = features[i + 1]
        price = prices[i]
        next_price = prices[i + 1]

        action, _ = individual.predict(state)
        reward = 0.0

        # ---- Process action ----
        if action == Action.OPEN_BUY and len(individual.open_positions) == 0:
            for g in range(individual.grid_params.orders_count):
                vol = individual.grid_params.base_volume + g * individual.grid_params.volume_step
                vol = round(max(0.01, vol), 2)
                order_price = price - (g + 1) * individual.grid_params.grid_step
                individual.pending_orders.append(SimOrder("buy", order_price, vol))
            individual.open_positions.append(
                SimPosition("buy", price, individual.grid_params.base_volume))

        elif action == Action.OPEN_SELL and len(individual.open_positions) == 0:
            for g in range(individual.grid_params.orders_count):
                vol = individual.grid_params.base_volume + g * individual.grid_params.volume_step
                vol = round(max(0.01, vol), 2)
                order_price = price + (g + 1) * individual.grid_params.grid_step
                individual.pending_orders.append(SimOrder("sell", order_price, vol))
            individual.open_positions.append(
                SimPosition("sell", price, individual.grid_params.base_volume))

        elif action == Action.CLOSE_BUY_PROFIT:
            still_open = []
            for pos in individual.open_positions:
                if pos.direction == "buy" and price > pos.entry_price:
                    pnl = (price - pos.entry_price) * pos.volume
                    individual.closed_pnls.append(pnl)
                    individual.total_profit += pnl
                    reward += pnl
                    individual.wins += 1
                else:
                    still_open.append(pos)
            individual.open_positions = still_open

        elif action == Action.CLOSE_BUY_LOSS:
            still_open = []
            for pos in individual.open_positions:
                if pos.direction == "buy" and price <= pos.entry_price:
                    pnl = (price - pos.entry_price) * pos.volume
                    individual.closed_pnls.append(pnl)
                    individual.total_profit += pnl
                    reward += pnl
                    individual.losses += 1
                else:
                    still_open.append(pos)
            individual.open_positions = still_open

        elif action == Action.CLOSE_SELL_PROFIT:
            still_open = []
            for pos in individual.open_positions:
                if pos.direction == "sell" and price < pos.entry_price:
                    pnl = (pos.entry_price - price) * pos.volume
                    individual.closed_pnls.append(pnl)
                    individual.total_profit += pnl
                    reward += pnl
                    individual.wins += 1
                else:
                    still_open.append(pos)
            individual.open_positions = still_open

        elif action == Action.CLOSE_SELL_LOSS:
            still_open = []
            for pos in individual.open_positions:
                if pos.direction == "sell" and price >= pos.entry_price:
                    pnl = (pos.entry_price - price) * pos.volume
                    individual.closed_pnls.append(pnl)
                    individual.total_profit += pnl
                    reward += pnl
                    individual.losses += 1
                else:
                    still_open.append(pos)
            individual.open_positions = still_open

        # ---- Fill pending grid orders ----
        filled = []
        for idx, order in enumerate(individual.pending_orders):
            if order.direction == "buy" and next_price <= order.price:
                individual.open_positions.append(
                    SimPosition("buy", order.price, order.volume))
                filled.append(idx)
            elif order.direction == "sell" and next_price >= order.price:
                individual.open_positions.append(
                    SimPosition("sell", order.price, order.volume))
                filled.append(idx)
        for idx in reversed(filled):
            individual.pending_orders.pop(idx)

        # ---- Check grid profit target / loss limit ----
        unrealized = 0.0
        for pos in individual.open_positions:
            if pos.direction == "buy":
                unrealized += (price - pos.entry_price) * pos.volume
            else:
                unrealized += (pos.entry_price - price) * pos.volume

        if len(individual.open_positions) > 0:
            if unrealized >= individual.grid_params.profit_target:
                for pos in individual.open_positions:
                    if pos.direction == "buy":
                        pnl = (price - pos.entry_price) * pos.volume
                    else:
                        pnl = (pos.entry_price - price) * pos.volume
                    individual.closed_pnls.append(pnl)
                    individual.total_profit += pnl
                    if pnl > 0:
                        individual.wins += 1
                    else:
                        individual.losses += 1
                    reward += pnl
                individual.open_positions.clear()
                individual.pending_orders.clear()

            elif unrealized <= -individual.grid_params.loss_limit:
                for pos in individual.open_positions:
                    if pos.direction == "buy":
                        pnl = (price - pos.entry_price) * pos.volume
                    else:
                        pnl = (pos.entry_price - price) * pos.volume
                    individual.closed_pnls.append(pnl)
                    individual.total_profit += pnl
                    if pnl > 0:
                        individual.wins += 1
                    else:
                        individual.losses += 1
                    reward += pnl
                individual.open_positions.clear()
                individual.pending_orders.clear()

        # ---- Drawdown tracking ----
        equity = individual.total_profit + unrealized
        if equity > equity_peak:
            equity_peak = equity
        dd = equity_peak - equity
        if dd > individual.max_drawdown:
            individual.max_drawdown = dd

        # ---- Store experience (only when something happened) ----
        if reward != 0.0 or action in (Action.OPEN_BUY, Action.OPEN_SELL):
            individual.store_experience(state, action, reward, next_state)

        # ---- Periodic Q-learning training ----
        if (i + 1) % ql_interval == 0:
            individual.train_from_memory(n_batches=2)

    # Final Q-learning pass
    individual.train_from_memory(n_batches=4)

    # Close any remaining positions at last price
    last_price = prices[-1]
    for pos in individual.open_positions:
        if pos.direction == "buy":
            pnl = (last_price - pos.entry_price) * pos.volume
        else:
            pnl = (pos.entry_price - last_price) * pos.volume
        individual.closed_pnls.append(pnl)
        individual.total_profit += pnl
        if pnl > 0:
            individual.wins += 1
        else:
            individual.losses += 1
    individual.open_positions.clear()
    individual.pending_orders.clear()

    return individual


# ---------------------------------------------------------------------------
# Fitness calculation
# ---------------------------------------------------------------------------
def calculate_fitness(ind: GridTrader) -> float:
    total_trades = ind.wins + ind.losses
    if total_trades == 0:
        return 0.0

    win_rate = ind.wins / total_trades

    # Accuracy proxy: fraction of profitable closed trades
    profitable = sum(1 for p in ind.closed_pnls if p > 0)
    accuracy = profitable / len(ind.closed_pnls) if ind.closed_pnls else 0.0

    # Profit factor
    gross_profit = sum(p for p in ind.closed_pnls if p > 0)
    gross_loss = abs(sum(p for p in ind.closed_pnls if p < 0))
    profit_factor = gross_profit / (gross_loss + 1e-8)
    # Cap profit factor contribution
    pf_score = min(profit_factor, 5.0) / 5.0

    # Drawdown resistance
    dd_resistance = 1.0 / (1.0 + ind.max_drawdown / 100.0)

    fitness = (
        accuracy * 0.3
        + win_rate * 0.4
        + pf_score * 0.2
        + dd_resistance * 0.1
    )
    return fitness


# ---------------------------------------------------------------------------
# Genetic operations
# ---------------------------------------------------------------------------
def tournament_select(population: List[GridTrader]) -> GridTrader:
    tournament = random.sample(population, min(TOURNAMENT_SIZE, len(population)))
    return max(tournament, key=lambda x: x.fitness)


def crossover(p1: GridTrader, p2: GridTrader) -> GridTrader:
    child = GridTrader()
    # Weight crossover: 50/50 mask
    for attr in ["input_weights", "hidden_weights", "output_weights"]:
        w1 = getattr(p1.weights, attr)
        w2 = getattr(p2.weights, attr)
        mask = np.random.random(w1.shape) < 0.5
        setattr(child.weights, attr, np.where(mask, w1, w2))

    for attr in ["hidden_bias", "hidden2_bias", "output_bias"]:
        b1 = getattr(p1.weights, attr)
        b2 = getattr(p2.weights, attr)
        mask = np.random.random(b1.shape) < 0.5
        setattr(child.weights, attr, np.where(mask, b1, b2))

    # Grid param inheritance: pick one parent
    donor = p1 if np.random.random() < 0.5 else p2
    child.grid_params.grid_step = donor.grid_params.grid_step
    child.grid_params.orders_count = donor.grid_params.orders_count
    child.grid_params.base_volume = donor.grid_params.base_volume
    child.grid_params.volume_step = donor.grid_params.volume_step
    child.grid_params.profit_target = donor.grid_params.profit_target
    child.grid_params.loss_limit = donor.grid_params.loss_limit

    return child


def extinction_event(population: List[GridTrader]) -> List[GridTrader]:
    """Kill bottom 30%, breed replacements from survivors."""
    population.sort(key=lambda x: x.fitness, reverse=True)
    cutoff = int(len(population) * (1 - EXTINCTION_RATE))
    survivors = population[:cutoff]

    while len(survivors) < POPULATION_SIZE:
        if random.random() < CROSSOVER_RATE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child = crossover(p1, p2)
        else:
            child = deepcopy(random.choice(survivors[:ELITE_SIZE]))
        child.mutate()
        survivors.append(child)

    return survivors[:POPULATION_SIZE]


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------
def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS population (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                individual TEXT NOT NULL,
                fitness REAL DEFAULT 0.0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER,
                best_fitness REAL,
                avg_fitness REAL,
                best_win_rate REAL,
                avg_win_rate REAL,
                best_profit REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return conn


def save_population(conn: sqlite3.Connection, population: List[GridTrader]):
    with conn:
        conn.execute("DELETE FROM population")
        for ind in population:
            conn.execute(
                "INSERT INTO population (individual, fitness) VALUES (?, ?)",
                (json.dumps(ind.to_dict()), float(ind.fitness)),
            )


def load_population(conn: sqlite3.Connection) -> List[GridTrader]:
    rows = conn.execute(
        "SELECT individual, fitness FROM population ORDER BY fitness DESC"
    ).fetchall()
    pop = []
    for row in rows:
        d = json.loads(row[0])
        gt = GridTrader.from_dict(d)
        gt.fitness = row[1]
        pop.append(gt)
    return pop


def save_generation_history(conn: sqlite3.Connection, generation: int,
                            population: List[GridTrader]):
    fitnesses = [ind.fitness for ind in population]
    win_rates = []
    profits = []
    for ind in population:
        total = ind.wins + ind.losses
        wr = ind.wins / total if total > 0 else 0.0
        win_rates.append(wr)
        profits.append(ind.total_profit)

    with conn:
        conn.execute(
            """INSERT INTO history
               (generation, best_fitness, avg_fitness, best_win_rate, avg_win_rate, best_profit)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                generation,
                float(max(fitnesses)),
                float(np.mean(fitnesses)),
                float(max(win_rates)),
                float(np.mean(win_rates)),
                float(max(profits)),
            ),
        )


# ---------------------------------------------------------------------------
# Export best expert
# ---------------------------------------------------------------------------
def export_best(population: List[GridTrader], path: str):
    best = max(population, key=lambda x: x.fitness)
    total = best.wins + best.losses
    win_rate = best.wins / total if total > 0 else 0.0

    export_data = {
        "model": "ETARE_BTCUSD",
        "version": 2,
        "architecture": f"numpy_feedforward_{INPUT_SIZE}_128_64_6",
        "input_size": INPUT_SIZE,
        "hidden1": HIDDEN1,
        "hidden2": HIDDEN2,
        "num_actions": NUM_ACTIONS,
        "quantum_features_included": True,
        "input_weights": best.weights.input_weights.tolist(),
        "hidden_weights": best.weights.hidden_weights.tolist(),
        "output_weights": best.weights.output_weights.tolist(),
        "hidden_bias": best.weights.hidden_bias.tolist(),
        "hidden2_bias": best.weights.hidden2_bias.tolist(),
        "output_bias": best.weights.output_bias.tolist(),
        "grid_params": best.grid_params.to_dict(),
        "fitness": float(best.fitness),
        "win_rate": float(win_rate),
        "total_profit": float(best.total_profit),
        "feature_order": [
            "rsi", "macd", "macd_signal",
            "bb_middle", "bb_upper", "bb_lower", "bb_std",
            "ema_5", "ema_10", "ema_20", "ema_50",
            "momentum", "atr",
            "price_change", "price_change_abs",
            "volume_ma", "volume_std",
            "stoch_k", "stoch_d",
            "roc",
        ] + QUANTUM_FEATURE_NAMES,
        "actions": [a.name for a in Action],
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(export_data, f, indent=2)
    logging.info(f"Exported best expert to {path} (fitness={best.fitness:.4f}, WR={win_rate*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    logging.info("=" * 70)
    logging.info("ETARE BTCUSD TRAINER - True Blueprint Build")
    logging.info(f"Population: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
    logging.info(f"Network: {INPUT_SIZE} -> {HIDDEN1} -> {HIDDEN2} -> {NUM_ACTIONS}")
    logging.info(f"Actions: {[a.name for a in Action]}")
    logging.info("=" * 70)

    # Init MT5
    if not init_mt5():
        return

    # Init database
    conn = init_db(DB_PATH)

    # Try to load existing population
    population = load_population(conn)
    start_gen = 0
    if population and len(population) >= POPULATION_SIZE:
        population = population[:POPULATION_SIZE]
        # Get last generation from history
        row = conn.execute("SELECT MAX(generation) FROM history").fetchone()
        start_gen = (row[0] or 0) + 1
        logging.info(f"Resumed {len(population)} individuals from DB, starting at generation {start_gen}")
    else:
        population = [GridTrader() for _ in range(POPULATION_SIZE)]
        logging.info(f"Initialized fresh population of {POPULATION_SIZE} individuals")

    # Fetch initial data
    df = fetch_data()
    if df is None:
        mt5.shutdown()
        return

    # Load quantum features aligned to M5 bar timestamps
    bar_timestamps = df["time"].values.astype("datetime64[s]").astype(np.float64)
    quantum_features = load_quantum_features_bulk(SYMBOL, bar_timestamps)
    logging.info(f"Quantum features loaded: {quantum_features.shape}")

    features, prices = prepare_features(df, quantum_array=quantum_features)
    logging.info(f"Features shape: {features.shape}, prices: {len(prices)} bars")

    best_ever_fitness = 0.0
    best_ever_wr = 0.0

    for gen in range(start_gen, start_gen + NUM_GENERATIONS):
        gen_start = time.time()

        # Optionally refresh data every 10 generations
        if gen > start_gen and gen % 10 == 0:
            new_df = fetch_data()
            if new_df is not None:
                new_bar_ts = new_df["time"].values.astype("datetime64[s]").astype(np.float64)
                quantum_features = load_quantum_features_bulk(SYMBOL, new_bar_ts)
                features, prices = prepare_features(new_df, quantum_array=quantum_features)
                logging.info(f"Refreshed data: {features.shape[0]} bars")

        # ---- Evaluate each individual ----
        for idx, ind in enumerate(population):
            simulate_individual(ind, features, prices)
            ind.fitness = calculate_fitness(ind)

        # ---- Stats ----
        fitnesses = [ind.fitness for ind in population]
        win_rates = []
        profits = []
        trade_counts = []
        for ind in population:
            total = ind.wins + ind.losses
            wr = ind.wins / total if total > 0 else 0.0
            win_rates.append(wr)
            profits.append(ind.total_profit)
            trade_counts.append(total)

        best_fit = max(fitnesses)
        avg_fit = np.mean(fitnesses)
        best_wr = max(win_rates)
        avg_wr = np.mean(win_rates)
        best_profit = max(profits)
        avg_trades = np.mean(trade_counts)

        if best_fit > best_ever_fitness:
            best_ever_fitness = best_fit
        if best_wr > best_ever_wr:
            best_ever_wr = best_wr

        elapsed = time.time() - gen_start

        logging.info(
            f"Gen {gen:3d} | "
            f"Fit: {best_fit:.4f} (avg {avg_fit:.4f}) | "
            f"WR: {best_wr*100:.1f}% (avg {avg_wr*100:.1f}%) | "
            f"Profit: ${best_profit:.2f} | "
            f"Trades: {avg_trades:.0f} | "
            f"{elapsed:.1f}s"
        )

        # ---- Save generation history ----
        save_generation_history(conn, gen, population)

        # ---- Extinction event ----
        if (gen + 1) % EXTINCTION_INTERVAL == 0:
            logging.info(f"=== EXTINCTION EVENT at generation {gen} ===")
            population = extinction_event(population)
            logging.info(f"Population rebuilt: {len(population)} individuals")
        else:
            # Standard evolution: replace bottom 30% each generation
            population.sort(key=lambda x: x.fitness, reverse=True)
            keep = int(POPULATION_SIZE * 0.7)
            new_pop = population[:keep]
            while len(new_pop) < POPULATION_SIZE:
                if random.random() < CROSSOVER_RATE:
                    p1 = tournament_select(population)
                    p2 = tournament_select(population)
                    child = crossover(p1, p2)
                else:
                    child = deepcopy(random.choice(new_pop[:ELITE_SIZE]))
                child.mutate()
                new_pop.append(child)
            population = new_pop[:POPULATION_SIZE]

        # ---- Save population every 5 generations ----
        if (gen + 1) % 5 == 0:
            save_population(conn, population)
            logging.info(f"Population saved to DB at generation {gen}")

        # ---- Export best every 10 generations ----
        if (gen + 1) % 10 == 0:
            export_best(population, EXPORT_PATH)

    # ---- Final export ----
    export_best(population, EXPORT_PATH)
    save_population(conn, population)

    # Summary
    best = max(population, key=lambda x: x.fitness)
    total = best.wins + best.losses
    final_wr = best.wins / total if total > 0 else 0.0

    logging.info("=" * 70)
    logging.info("TRAINING COMPLETE")
    logging.info(f"Best fitness ever: {best_ever_fitness:.4f}")
    logging.info(f"Best win rate ever: {best_ever_wr*100:.1f}%")
    logging.info(f"Final best: fitness={best.fitness:.4f}, WR={final_wr*100:.1f}%, profit=${best.total_profit:.2f}")
    logging.info(f"Expert saved to: {EXPORT_PATH}")
    logging.info(f"Database: {DB_PATH}")
    logging.info("=" * 70)

    conn.close()
    mt5.shutdown()


if __name__ == "__main__":
    main()
