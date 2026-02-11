"""
Phase 2: Retrain Champions on Live Collected Data
===================================================
Takes Phase 1 champions (trained on historical data) and retrains them
on REAL signals from:
  1. MT5 trade history (482 trades - ground truth with outcomes)
  2. Local signal backups (13,845 live BRAIN signals)
  3. Quantum features from bg_archive.db (3,867 quantum states)
  4. Phase 1 champion weights as starting population

Architecture:
  - Loads Phase 1 champions as elite seed population
  - Reconstructs market context at each real trade's timestamp
  - Labels each bar with REAL outcomes (win/loss from MT5 history)
  - Retrains with evolutionary pressure against real-world results
  - Exports best expert in ETARE format for BRAIN deployment

Usage:
    python phase2_retrain.py
    python phase2_retrain.py --symbol BTCUSD --days 30
"""

import numpy as np
import pandas as pd
import torch
import json
import time
import sys
import os
import gzip
import sqlite3
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy
from enum import Enum
from dataclasses import dataclass
import random
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# GPU
try:
    import torch_directml
    DEVICE = torch_directml.device()
    GPU_NAME = torch_directml.device_name(0)
    print(f"[PHASE2] GPU: {GPU_NAME}")
except ImportError:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU"
    print("[PHASE2] CPU mode")

import MetaTrader5 as mt5
from config_loader import CONFIDENCE_THRESHOLD

# Farm config
with open(SCRIPT_DIR / "signal_farm_config.json") as f:
    FARM_CONFIG = json.load(f)


# ============================================================
# GENETIC ALGORITHM (same architecture as Phase 1 + etare_expert)
# ============================================================

class Action(Enum):
    OPEN_BUY = 0
    OPEN_SELL = 1
    CLOSE_BUY_PROFIT = 2
    CLOSE_BUY_LOSS = 3
    CLOSE_SELL_PROFIT = 4
    CLOSE_SELL_LOSS = 5

# Direction mapping matching etare_expert.py
ACTION_TO_DIR = {0: "BUY", 1: "SELL", 2: "HOLD", 3: "HOLD", 4: "HOLD", 5: "HOLD"}


class TradingIndividual:
    """Genetic individual - same architecture as ETARE expert (input->128->64->6)."""

    def __init__(self, input_size: int):
        self.input_size = input_size
        gc = FARM_CONFIG["GENETIC"]
        self.mutation_rate = gc["MUTATION_RATE"]
        self.mutation_strength = gc["MUTATION_STRENGTH"]

        # Architecture: INPUT -> 128 (tanh) -> 64 (tanh) -> 6 (softmax)
        self.input_weights = torch.empty(input_size, 128, device=DEVICE).uniform_(-0.5, 0.5)
        self.hidden_weights = torch.empty(128, 64, device=DEVICE).uniform_(-0.5, 0.5)
        self.output_weights = torch.empty(64, 6, device=DEVICE).uniform_(-0.5, 0.5)
        self.hidden_bias = torch.empty(128, device=DEVICE).uniform_(-0.5, 0.5)
        self.hidden2_bias = torch.empty(64, device=DEVICE).uniform_(-0.5, 0.5)
        self.output_bias = torch.empty(6, device=DEVICE).uniform_(-0.5, 0.5)

        self.fitness = 0.0
        self.test_fitness = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        self.timeframe = ""
        self.cycle = 0
        self.generation = 0

    def batch_predict(self, states: torch.Tensor) -> torch.Tensor:
        mean = states.mean(dim=1, keepdim=True)
        std = states.std(dim=1, keepdim=True) + 1e-8
        states = (states - mean) / std
        h1 = torch.tanh(torch.matmul(states, self.input_weights) + self.hidden_bias)
        h2 = torch.tanh(torch.matmul(h1, self.hidden_weights) + self.hidden2_bias)
        out = torch.matmul(h2, self.output_weights) + self.output_bias
        return torch.argmax(out, dim=1)

    def predict_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities for a single state."""
        s = state.unsqueeze(0) if state.dim() == 1 else state
        mean = s.mean(dim=1, keepdim=True)
        std = s.std(dim=1, keepdim=True) + 1e-8
        s = (s - mean) / std
        h1 = torch.tanh(torch.matmul(s, self.input_weights) + self.hidden_bias)
        h2 = torch.tanh(torch.matmul(h1, self.hidden_weights) + self.hidden2_bias)
        logits = torch.matmul(h2, self.output_weights) + self.output_bias
        return torch.softmax(logits, dim=1)

    def mutate(self):
        for wt in [self.input_weights, self.hidden_weights, self.output_weights]:
            mask = torch.rand_like(wt) < self.mutation_rate
            noise = torch.randn_like(wt) * self.mutation_strength
            wt[mask] += noise[mask]

    def to_etare_dict(self, feature_order=None):
        """Export in ETARE expert JSON format (compatible with etare_expert.py)."""
        return {
            "model": "ETARE_BTCUSD",
            "architecture": f"numpy_feedforward_{self.input_size}_128_64_6",
            "version": 2,
            "input_size": self.input_size,
            "hidden1": 128,
            "hidden2": 64,
            "num_actions": 6,
            "quantum_features_included": self.input_size > 20,
            "fitness": float(self.fitness),
            "win_rate": float(self.test_fitness),
            "input_weights": self.input_weights.cpu().numpy().tolist(),
            "hidden_weights": self.hidden_weights.cpu().numpy().tolist(),
            "output_weights": self.output_weights.cpu().numpy().tolist(),
            "hidden_bias": self.hidden_bias.cpu().numpy().tolist(),
            "hidden2_bias": self.hidden2_bias.cpu().numpy().tolist(),
            "output_bias": self.output_bias.cpu().numpy().tolist(),
            "feature_order": feature_order or [],
            "trained_on": "phase2_live_data",
            "trained_at": datetime.now().isoformat(),
        }

    @classmethod
    def from_champion_json(cls, path, input_size):
        """Load a Phase 1 champion JSON into an individual."""
        with open(path) as f:
            data = json.load(f)

        ind = cls(input_size)
        # Phase 1 champions have input_weights etc as lists
        iw = torch.tensor(data["input_weights"], dtype=torch.float32, device=DEVICE)
        hw = torch.tensor(data["hidden_weights"], dtype=torch.float32, device=DEVICE)
        ow = torch.tensor(data["output_weights"], dtype=torch.float32, device=DEVICE)

        # Resize if needed (Phase 1 used 17 features, Phase 2 may use 20 or 27)
        if iw.shape[0] != input_size:
            # Pad or truncate input weights
            if iw.shape[0] < input_size:
                pad = torch.empty(input_size - iw.shape[0], 128, device=DEVICE).uniform_(-0.1, 0.1)
                iw = torch.cat([iw, pad], dim=0)
            else:
                iw = iw[:input_size]

        ind.input_weights = iw
        ind.hidden_weights = hw
        ind.output_weights = ow

        if "hidden_bias" in data:
            ind.hidden_bias = torch.tensor(data["hidden_bias"], dtype=torch.float32, device=DEVICE)
        if "output_bias" in data:
            ind.output_bias = torch.tensor(data["output_bias"], dtype=torch.float32, device=DEVICE)

        ind.fitness = data.get("fitness", 0)
        ind.test_fitness = data.get("test_fitness", data.get("win_rate", 0))
        ind.timeframe = data.get("timeframe", "")
        return ind


# ============================================================
# DATA: PULL REAL TRADES FROM MT5
# ============================================================

def pull_mt5_trades(symbol="BTCUSD", days=30):
    """
    Pull closed trades from MT5 with full context.
    Returns DataFrame with: time, direction, entry_price, exit_price, profit, outcome.
    """
    now = datetime.now()
    deals = mt5.history_deals_get(now - timedelta(days=days), now)

    if deals is None or len(deals) == 0:
        print("  No deals found")
        return pd.DataFrame()

    records = []
    for d in deals:
        if d.symbol != symbol:
            continue
        if d.entry != 1:  # Only closing deals
            continue
        if d.type not in [0, 1]:  # Buy or sell
            continue

        records.append({
            "time": datetime.fromtimestamp(d.time),
            "type": "BUY" if d.type == 0 else "SELL",
            "price": d.price,
            "profit": d.profit,
            "volume": d.volume,
            "outcome": "WIN" if d.profit > 0 else "LOSS",
            "commission": d.commission,
            "swap": d.swap,
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("time").reset_index(drop=True)
    print(f"  Pulled {len(df)} closed {symbol} trades from MT5")
    return df


# ============================================================
# DATA: LOAD LOCAL SIGNAL BACKUPS
# ============================================================

def load_local_signals():
    """Load all local signal .jsonl files."""
    sig_dir = SCRIPT_DIR / "quantum_data"
    records = []

    for f in sorted(sig_dir.glob("signals_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    print(f"  Loaded {len(records):,} local signal records")
    return records


# ============================================================
# DATA: LOAD QUANTUM FEATURES FROM bg_archive.db
# ============================================================

def load_quantum_features(symbol="BTCUSD"):
    """Load and decompress quantum features from bg_archive.db."""
    db_path = SCRIPT_DIR / "BlueGuardian_Deploy" / "bg_archive.db"
    if not db_path.exists():
        print(f"  WARNING: bg_archive.db not found at {db_path}")
        return []

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, features_compressed FROM quantum_features WHERE symbol=? ORDER BY timestamp",
        (symbol,)
    )
    rows = cur.fetchall()
    conn.close()

    features = []
    for ts, compressed in rows:
        try:
            decompressed = gzip.decompress(compressed)
            data = json.loads(decompressed)
            data["timestamp"] = ts
            features.append(data)
        except Exception:
            continue

    print(f"  Loaded {len(features):,} quantum feature snapshots")
    return features


# ============================================================
# FEATURE ENGINEERING (matching etare_expert.py's 20 features)
# ============================================================

def prepare_features_20(df):
    """Prepare the 20 technical features matching ETARE v1 expert."""
    d = df.copy()

    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss_s + 1e-10)
    d["rsi"] = 100 - (100 / (1 + rs))

    exp12 = d["close"].ewm(span=12, adjust=False).mean()
    exp26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = exp12 - exp26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()

    d["bb_middle"] = d["close"].rolling(20).mean()
    d["bb_std"] = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_middle"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_middle"] - 2 * d["bb_std"]

    for p in [5, 10, 20, 50]:
        d[f"ema_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

    d["momentum"] = d["close"] / d["close"].shift(10)
    d["atr"] = d["high"].rolling(14).max() - d["low"].rolling(14).min()
    d["price_change"] = d["close"].pct_change()
    d["price_change_abs"] = d["price_change"].abs()
    d["volume_ma"] = d["tick_volume"].rolling(20).mean()
    d["volume_std"] = d["tick_volume"].rolling(20).std()

    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = 100 * (d["close"] - low14) / (high14 - low14 + 1e-10)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()
    d["roc"] = d["close"].pct_change(10) * 100

    d = d.ffill().bfill()

    feature_cols = [
        "rsi", "macd", "macd_signal",
        "bb_middle", "bb_upper", "bb_lower", "bb_std",
        "ema_5", "ema_10", "ema_20", "ema_50",
        "momentum", "atr",
        "price_change", "price_change_abs",
        "volume_ma", "volume_std",
        "stoch_k", "stoch_d", "roc",
    ]

    for col in feature_cols:
        mean = d[col].mean()
        std = d[col].std() + 1e-8
        d[col] = (d[col] - mean) / std

    d = d.dropna()
    return d, feature_cols


# ============================================================
# BUILD LABELED TRAINING DATA FROM REAL TRADES
# ============================================================

def build_labeled_dataset(symbol="BTCUSD", days=30, timeframe="M5"):
    """
    Build training data by:
    1. Fetching M5 bars covering the trade period
    2. Computing features for every bar
    3. Labeling bars near trade timestamps with real outcomes
    4. Creating (features, label) pairs where label = correct action

    Labels:
      - Bars near winning BUY trades -> OPEN_BUY (0)
      - Bars near winning SELL trades -> OPEN_SELL (1)
      - Bars near losing BUY trades -> avoid OPEN_BUY
      - Bars near losing SELL trades -> avoid OPEN_SELL
    """
    TF_MAP = {"M5": mt5.TIMEFRAME_M5, "M1": mt5.TIMEFRAME_M1, "M15": mt5.TIMEFRAME_M15}
    tf = TF_MAP.get(timeframe, mt5.TIMEFRAME_M5)

    # Get bar data covering trade period
    bars_needed = 288 * days  # M5 = 288 bars/day
    print(f"  Fetching {bars_needed:,} {timeframe} bars for labeling...")
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, min(bars_needed, 500000))

    if rates is None or len(rates) < 500:
        print(f"  ERROR: Not enough bar data")
        return None, None, None, None

    bar_df = pd.DataFrame(rates)
    bar_df['time'] = pd.to_datetime(bar_df['time'], unit='s')

    # Compute features
    bar_df, feature_cols = prepare_features_20(bar_df)
    print(f"  Prepared {len(bar_df):,} bars with {len(feature_cols)} features")

    # Pull real trades
    trades_df = pull_mt5_trades(symbol, days)
    if trades_df.empty:
        print("  No trades to label with")
        return None, None, None, None

    # Label bars: for each trade, find nearest bar and label it
    features = bar_df[feature_cols].values
    prices = bar_df['close'].values
    bar_times = bar_df['time'].values

    # Create label array: -1 = unlabeled, 0-5 = action
    labels = np.full(len(features), -1, dtype=np.int32)
    # Also create outcome array for fitness (1=win, 0=loss)
    outcomes = np.full(len(features), -1, dtype=np.int32)

    labeled_count = 0
    for _, trade in trades_df.iterrows():
        trade_time = pd.Timestamp(trade['time'])
        # Find nearest bar
        time_diffs = np.abs(bar_times - np.datetime64(trade_time))
        nearest_idx = np.argmin(time_diffs)

        if trade['outcome'] == 'WIN':
            if trade['type'] == 'BUY':
                labels[nearest_idx] = Action.OPEN_BUY.value
            else:
                labels[nearest_idx] = Action.OPEN_SELL.value
            outcomes[nearest_idx] = 1
        else:
            # For losses, label the opposite action as correct
            # (should have gone the other way or held)
            if trade['type'] == 'BUY':
                labels[nearest_idx] = Action.CLOSE_BUY_LOSS.value
            else:
                labels[nearest_idx] = Action.CLOSE_SELL_LOSS.value
            outcomes[nearest_idx] = 0
        labeled_count += 1

    print(f"  Labeled {labeled_count} bars from real trades")

    return features, prices, labels, feature_cols


# ============================================================
# EVALUATION AGAINST REAL LABELED DATA
# ============================================================

def evaluate_on_labels(individual, features_tensor, labels, outcomes):
    """
    Evaluate individual against real labeled data.

    Fitness = weighted combination of:
    - Accuracy on labeled bars (did it predict the right action?)
    - Win rate on bars where it would have traded
    """
    predictions = individual.batch_predict(features_tensor).cpu().numpy()

    # Only evaluate on labeled bars
    labeled_mask = labels >= 0
    if labeled_mask.sum() == 0:
        return 0.0, 0

    labeled_preds = predictions[labeled_mask]
    labeled_true = labels[labeled_mask]
    labeled_outcomes = outcomes[labeled_mask]

    # Accuracy: did it predict the correct action?
    correct = (labeled_preds == labeled_true).sum()
    accuracy = correct / len(labeled_true)

    # Win-aligned: when it predicts BUY/SELL, does it match winners?
    trade_actions = (labeled_preds == 0) | (labeled_preds == 1)  # BUY or SELL
    if trade_actions.sum() > 0:
        trade_outcomes = labeled_outcomes[trade_actions]
        win_alignment = (trade_outcomes == 1).sum() / len(trade_outcomes)
    else:
        win_alignment = 0.0

    # Combined fitness (accuracy 40%, win-alignment 60%)
    fitness = accuracy * 0.4 + win_alignment * 0.6

    individual.total_trades = int(trade_actions.sum())
    individual.successful_trades = int((labeled_outcomes[trade_actions] == 1).sum()) if trade_actions.sum() > 0 else 0

    return float(fitness), int(trade_actions.sum())


def evaluate_walkforward(individual, features_tensor, prices, hold_time=5):
    """Traditional walk-forward evaluation (from Phase 1) as secondary metric."""
    actions = individual.batch_predict(features_tensor).cpu().numpy()
    wins, losses = 0, 0
    position = None
    entry_price = 0

    for i in range(len(actions) - hold_time):
        action = actions[i]
        cp = prices[i]
        fp = prices[i + hold_time]

        if position is None:
            if action == 0:
                position = 'buy'; entry_price = cp
            elif action == 1:
                position = 'sell'; entry_price = cp
        else:
            if position == 'buy' and action in [2, 3]:
                (wins if fp > entry_price else losses).__class__  # just counting
                if fp - entry_price > 0: wins += 1
                else: losses += 1
                position = None
            elif position == 'sell' and action in [4, 5]:
                if entry_price - fp > 0: wins += 1
                else: losses += 1
                position = None

    total = wins + losses
    return (wins / total) if total > 0 else 0, total


# ============================================================
# CROSSOVER / SELECTION
# ============================================================

def tournament_select(pop, size=5):
    t = random.sample(pop, min(size, len(pop)))
    return max(t, key=lambda x: x.fitness)


def crossover(p1, p2, input_size):
    child = TradingIndividual(input_size)
    for attr in ["input_weights", "hidden_weights", "output_weights"]:
        w1 = getattr(p1, attr)
        w2 = getattr(p2, attr)
        mask = torch.rand_like(w1) < 0.5
        setattr(child, attr, torch.where(mask, w1, w2))
    return child


# ============================================================
# MAIN PHASE 2 TRAINING
# ============================================================

def run_phase2(symbol="BTCUSD", days=30, generations=20, pop_size=50):
    """
    Phase 2: Retrain on live data.

    1. Load Phase 1 champions as seed population
    2. Build labeled dataset from real MT5 trades
    3. Evolve population against real labeled data
    4. Export best expert in ETARE format
    """

    print("=" * 70)
    print("  PHASE 2: RETRAIN ON LIVE COLLECTED DATA")
    print("=" * 70)
    print(f"  Symbol:       {symbol}")
    print(f"  History:      {days} days")
    print(f"  Generations:  {generations}")
    print(f"  Pop Size:     {pop_size}")
    print(f"  GPU:          {GPU_NAME}")
    print("=" * 70)

    # Init MT5
    if not mt5.initialize():
        print("ERROR: MT5 init failed")
        return

    start_time = time.time()

    # ---- STEP 1: Build labeled dataset ----
    print(f"\n--- STEP 1: Building labeled dataset from real trades ---")
    features, prices, labels, feature_cols = build_labeled_dataset(symbol, days, "M5")

    if features is None:
        print("ERROR: Could not build dataset")
        mt5.shutdown()
        return

    input_size = len(feature_cols)
    labeled_count = (labels >= 0).sum()
    win_labels = ((labels >= 0) & (labels <= 1)).sum()  # BUY or SELL (winning trades)
    print(f"  Input size: {input_size}")
    print(f"  Total bars: {len(features):,}")
    print(f"  Labeled bars: {labeled_count}")
    print(f"  Win-labeled: {win_labels}")

    # Split: 70% train, 30% test
    split = int(len(features) * 0.7)
    train_features = features[:split]
    train_prices = prices[:split]
    train_labels = labels[:split]
    train_outcomes = np.where(labels[:split] <= 1, 1, 0)  # BUY/SELL = win outcome

    test_features = features[split:]
    test_prices = prices[split:]
    test_labels = labels[split:]
    test_outcomes = np.where(labels[split:] <= 1, 1, 0)

    train_tensor = torch.FloatTensor(train_features).to(DEVICE)
    test_tensor = torch.FloatTensor(test_features).to(DEVICE)

    # ---- STEP 2: Load Phase 1 champions as seeds ----
    print(f"\n--- STEP 2: Loading Phase 1 champions ---")
    champion_dir = SCRIPT_DIR / "signal_farm_output"
    champion_files = sorted(champion_dir.glob("champion_*.json"))

    population = []

    for cf in champion_files:
        try:
            ind = TradingIndividual.from_champion_json(cf, input_size)
            population.append(ind)
            print(f"  Loaded: {cf.name} (WR={ind.test_fitness*100:.0f}%)")
        except Exception as e:
            print(f"  SKIP: {cf.name} ({e})")

    print(f"  Loaded {len(population)} Phase 1 champions as elite seeds")

    # Fill rest of population with fresh individuals
    while len(population) < pop_size:
        population.append(TradingIndividual(input_size))
    print(f"  Population size: {len(population)}")

    # ---- STEP 3: Load quantum features for enrichment ----
    print(f"\n--- STEP 3: Loading quantum features ---")
    quantum_features = load_quantum_features(symbol)

    # ---- STEP 4: Load local signals ----
    print(f"\n--- STEP 4: Loading local signal backups ---")
    local_signals = load_local_signals()

    # ---- STEP 5: Evolutionary training against real data ----
    print(f"\n--- STEP 5: Training ({generations} generations) ---")

    gc = FARM_CONFIG["GENETIC"]
    elite_size = gc["ELITE_SIZE"]
    best_ever_fitness = 0.0
    best_ever_individual = None

    for gen in range(1, generations + 1):
        t0 = time.time()
        is_extinction = (gen % gc["EXTINCTION_EVERY_N_ROUNDS"] == 0)

        # Evaluate on REAL labeled data
        for ind in population:
            fitness, trades = evaluate_on_labels(ind, train_tensor, train_labels, train_outcomes)
            ind.fitness = fitness

        population.sort(key=lambda x: x.fitness, reverse=True)

        # Test evaluation
        for ind in population[:10]:  # Only test top 10 for speed
            fitness, trades = evaluate_on_labels(ind, test_tensor, test_labels, test_outcomes)
            ind.test_fitness = fitness

        # Also do walk-forward test
        best = population[0]
        wf_wr, wf_trades = evaluate_walkforward(best, test_tensor, test_prices)

        elapsed = time.time() - t0
        ext_tag = " [EXTINCTION]" if is_extinction else ""
        print(f"  Gen {gen:>3}/{generations}: "
              f"Fit={best.fitness:.3f} TestFit={best.test_fitness:.3f} "
              f"WF_WR={wf_wr*100:.1f}% Trades={best.total_trades} "
              f"({elapsed:.1f}s){ext_tag}")

        # Track best ever
        if best.test_fitness > best_ever_fitness:
            best_ever_fitness = best.test_fitness
            best_ever_individual = deepcopy(best)

        # Extinction event
        if is_extinction:
            survivors = population[:elite_size]
            for _ in range(gc["FRESH_INJECTION_COUNT"]):
                fresh = TradingIndividual(input_size)
                survivors.append(fresh)

            while len(survivors) < pop_size:
                if random.random() < 0.6:
                    p1 = tournament_select(population)
                    p2 = tournament_select(population)
                    child = crossover(p1, p2, input_size)
                else:
                    child = deepcopy(random.choice(population[:elite_size]))
                    child.mutate()
                survivors.append(child)

            population = survivors
        else:
            # Normal evolution
            new_pop = population[:elite_size]  # Keep elite
            while len(new_pop) < pop_size:
                if random.random() < 0.7:
                    p1 = tournament_select(population)
                    p2 = tournament_select(population)
                    child = crossover(p1, p2, input_size)
                    child.mutate()
                else:
                    child = deepcopy(random.choice(population[:elite_size]))
                    child.mutate()
                new_pop.append(child)
            population = new_pop

    # ---- STEP 6: Final evaluation & export ----
    print(f"\n--- STEP 6: Final Evaluation ---")

    # Evaluate all on test set
    for ind in population:
        fitness, trades = evaluate_on_labels(ind, test_tensor, test_labels, test_outcomes)
        ind.test_fitness = fitness

    population.sort(key=lambda x: x.test_fitness, reverse=True)

    # Walk-forward on full test set for top 5
    print(f"\n  TOP 5 CANDIDATES:")
    for i, ind in enumerate(population[:5]):
        wf_wr, wf_trades = evaluate_walkforward(ind, test_tensor, test_prices)
        print(f"    #{i+1}: Fitness={ind.test_fitness:.3f} WF_WR={wf_wr*100:.1f}% "
              f"Trades={ind.total_trades} WFTrades={wf_trades}")

    # Export best as ETARE expert
    best = population[0]
    if best_ever_individual and best_ever_individual.test_fitness > best.test_fitness:
        best = best_ever_individual
        print(f"\n  Using best-ever individual (fitness={best.test_fitness:.3f})")

    output_dir = SCRIPT_DIR / "signal_farm_output"
    output_dir.mkdir(exist_ok=True)

    # Save in ETARE format (deployable by BRAIN scripts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    expert_path = output_dir / f"phase2_expert_{symbol}_{timestamp}.json"
    expert_data = best.to_etare_dict(feature_order=feature_cols)
    with open(expert_path, 'w') as f:
        json.dump(expert_data, f)
    print(f"\n  Expert saved: {expert_path}")

    # Also save to the ETARE deployment location
    deploy_path = SCRIPT_DIR / "ETARE_QuantumFusion" / "models" / f"{symbol.lower()}_etare_expert.json"
    deploy_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup current expert before overwriting
    if deploy_path.exists():
        backup_path = deploy_path.with_name(f"{symbol.lower()}_etare_expert_backup_{timestamp}.json")
        import shutil
        shutil.copy(deploy_path, backup_path)
        print(f"  Old expert backed up: {backup_path.name}")

    with open(deploy_path, 'w') as f:
        json.dump(expert_data, f)
    print(f"  Expert deployed: {deploy_path}")

    total_time = time.time() - start_time

    # Final report
    wf_wr, wf_trades = evaluate_walkforward(best, test_tensor, test_prices)
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2 COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Best Fitness:     {best.test_fitness:.3f}")
    print(f"  Walk-Forward WR:  {wf_wr*100:.1f}%")
    print(f"  WF Trades:        {wf_trades}")
    print(f"  Training Time:    {total_time:.1f}s")
    print(f"  Expert deployed to: {deploy_path}")
    print(f"  BRAIN scripts will pick up the new expert on next restart")
    print(f"{'=' * 70}")

    mt5.shutdown()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Retrain on Live Data")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--days", type=int, default=30, help="Days of trade history")
    parser.add_argument("--generations", type=int, default=20, help="Training generations")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    args = parser.parse_args()

    run_phase2(
        symbol=args.symbol,
        days=args.days,
        generations=args.generations,
        pop_size=args.pop_size,
    )
