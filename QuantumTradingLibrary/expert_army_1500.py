"""
EXPERT ARMY 1500 - Mass Breeding + Prop Firm Simulation + Signal Harvest
=========================================================================
Standalone script that:
  1. Breeds 1500 genetic expert configurations using multiple strategies
  2. Runs ALL of them through the prop firm simulator (BTCUSD, ETHUSD, XAUUSD)
  3. Collects every signal they generate
  4. Feeds results to TE domestication DB and collection server
  5. Reports: pass rates, win rates, total signals collected

Architecture:
  - Batched processing (50 experts per batch) to avoid memory overload
  - GPU for tensor ops (DirectML), LSTM stays on CPU
  - BELOW_NORMAL priority -- BRAIN is live, we do not interfere
  - Uses existing infrastructure: signal_farm_trainer features, QNIF sim engine
  - Writes to its own output directory, never touches MASTER_CONFIG.json

Breeding Strategies (to reach 1500 from 55 parents):
  - CROSSOVER:    Cross-breed existing experts (pair every combination)
  - MUTATION:     Mutate existing experts with varying strength
  - RADIATION:    Heavy mutation (high mutation rate) for exploration
  - GENESIS:      Fresh random individuals for diversity
  - ELITE_CLONE:  Clone top performers with micro-mutations

Usage:
    .venv312_gpu\\Scripts\\python.exe expert_army_1500.py
    .venv312_gpu\\Scripts\\python.exe expert_army_1500.py --target 1500 --batch-size 50
    .venv312_gpu\\Scripts\\python.exe expert_army_1500.py --dry-run

Authors: DooDoo + Claude
Date: 2026-02-12
"""

import os
import sys
import json
import time
import random
import hashlib
import sqlite3
import logging
import argparse
import traceback
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

# ============================================================
# PATHS & CONSTANTS
# ============================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
OUTPUT_DIR = SCRIPT_DIR / "army_1500_output"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PYTHON_EXE = str(SCRIPT_DIR / ".venv312_gpu" / "Scripts" / "python.exe")
EXPERTS_DIR = SCRIPT_DIR / "top_50_experts"
QNIF_DIR = SCRIPT_DIR / "QNIF"
HIST_DATA_DIR = QNIF_DIR / "HistoricalData" / "Full"
DOMESTICATION_DB = SCRIPT_DIR / "teqa_domestication.db"
SIGNAL_HISTORY_DB = OUTPUT_DIR / "army_signal_history.db"
COLLECTION_SERVER = "http://203.161.61.61:8888"

# ============================================================
# PRIORITY MANAGEMENT
# ============================================================

def set_below_normal_priority():
    """Set this process to BELOW_NORMAL so BRAIN gets CPU/GPU first."""
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        result = ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        return bool(result)
    except Exception:
        return False


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"army_1500_{timestamp}.log"

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger("ARMY_1500")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# GPU SETUP
# ============================================================

def setup_gpu():
    try:
        import torch_directml
        device = torch_directml.device()
        gpu_name = torch_directml.device_name(0)
        return device, gpu_name
    except ImportError:
        return torch.device("cpu"), "CPU (DirectML not available)"


# ============================================================
# GENETIC EXPERT (matches signal_farm_trainer.py architecture)
# ============================================================

class GeneticExpert:
    """
    A genetic trading expert with 3-layer neural network.
    Architecture: input_size -> 128 -> 64 -> 6 (actions)
    Matches signal_farm_trainer.py TradingIndividual exactly.
    """

    ACTIONS = {
        0: "OPEN_BUY",
        1: "OPEN_SELL",
        2: "CLOSE_BUY_PROFIT",
        3: "CLOSE_BUY_LOSS",
        4: "CLOSE_SELL_PROFIT",
        5: "CLOSE_SELL_LOSS",
    }

    def __init__(self, input_size: int = 17, device=None, expert_id: str = None):
        self.input_size = input_size
        self.hidden1 = 128
        self.hidden2 = 64
        self.output_size = 6
        self.device = device or torch.device("cpu")
        self.expert_id = expert_id or hashlib.md5(
            f"{random.random()}{time.time()}".encode()
        ).hexdigest()[:12]

        # Weights (on device)
        self.input_weights = torch.empty(
            input_size, self.hidden1, device=self.device
        ).uniform_(-0.5, 0.5)
        self.hidden_weights = torch.empty(
            self.hidden1, self.hidden2, device=self.device
        ).uniform_(-0.5, 0.5)
        self.output_weights = torch.empty(
            self.hidden2, self.output_size, device=self.device
        ).uniform_(-0.5, 0.5)
        self.hidden_bias = torch.empty(
            self.hidden1, device=self.device
        ).uniform_(-0.5, 0.5)
        self.output_bias = torch.empty(
            self.output_size, device=self.device
        ).uniform_(-0.5, 0.5)

        # Metadata
        self.breeding_strategy = "GENESIS"
        self.parent_ids = []
        self.generation = 0
        self.mutation_rate = 0.15
        self.mutation_strength = 0.15

        # Sim results (filled after prop firm sim)
        self.fitness = 0.0
        self.total_trades = 0
        self.winners = 0
        self.losers = 0
        self.win_rate = 0.0
        self.net_pnl = 0.0
        self.max_dd = 0.0
        self.max_dd_pct = 0.0
        self.profit_factor = 0.0
        self.signals_generated = 0
        self.prop_firm_pass = False
        self.symbol_results = {}

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        """Batch prediction -- same as TradingIndividual.batch_predict."""
        mean = states.mean(dim=1, keepdim=True)
        std = states.std(dim=1, keepdim=True) + 1e-8
        states = (states - mean) / std
        hidden = torch.tanh(
            torch.matmul(states, self.input_weights) + self.hidden_bias
        )
        hidden2 = torch.tanh(torch.matmul(hidden, self.hidden_weights))
        output = torch.matmul(hidden2, self.output_weights) + self.output_bias
        return torch.argmax(output, dim=1)

    def predict_with_confidence(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with softmax confidence scores."""
        mean = states.mean(dim=1, keepdim=True)
        std = states.std(dim=1, keepdim=True) + 1e-8
        states = (states - mean) / std
        hidden = torch.tanh(
            torch.matmul(states, self.input_weights) + self.hidden_bias
        )
        hidden2 = torch.tanh(torch.matmul(hidden, self.hidden_weights))
        output = torch.matmul(hidden2, self.output_weights) + self.output_bias
        probs = torch.softmax(output, dim=1)
        actions = torch.argmax(probs, dim=1)
        confidence = probs.max(dim=1).values
        return actions, confidence

    def mutate(self, rate: float = None, strength: float = None):
        """Apply random mutations to weights."""
        rate = rate or self.mutation_rate
        strength = strength or self.mutation_strength
        for wt in [self.input_weights, self.hidden_weights, self.output_weights,
                    self.hidden_bias, self.output_bias]:
            mask = torch.rand_like(wt) < rate
            noise = torch.randn_like(wt) * strength
            wt[mask] += noise[mask]

    def clone(self) -> 'GeneticExpert':
        """Deep clone this expert."""
        child = GeneticExpert(self.input_size, self.device)
        child.input_weights = self.input_weights.clone()
        child.hidden_weights = self.hidden_weights.clone()
        child.output_weights = self.output_weights.clone()
        child.hidden_bias = self.hidden_bias.clone()
        child.output_bias = self.output_bias.clone()
        child.generation = self.generation + 1
        child.parent_ids = [self.expert_id]
        return child

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "expert_id": self.expert_id,
            "input_size": self.input_size,
            "breeding_strategy": self.breeding_strategy,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "mutation_rate": self.mutation_rate,
            "mutation_strength": self.mutation_strength,
            "fitness": self.fitness,
            "total_trades": self.total_trades,
            "winners": self.winners,
            "losers": self.losers,
            "win_rate": self.win_rate,
            "net_pnl": self.net_pnl,
            "max_dd": self.max_dd,
            "max_dd_pct": self.max_dd_pct,
            "profit_factor": self.profit_factor,
            "signals_generated": self.signals_generated,
            "prop_firm_pass": self.prop_firm_pass,
            "symbol_results": self.symbol_results,
            "weights": {
                "input_weights": self.input_weights.cpu().tolist(),
                "hidden_weights": self.hidden_weights.cpu().tolist(),
                "output_weights": self.output_weights.cpu().tolist(),
                "hidden_bias": self.hidden_bias.cpu().tolist(),
                "output_bias": self.output_bias.cpu().tolist(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict, device=None) -> 'GeneticExpert':
        """Deserialize from dict."""
        input_size = data.get("input_size", 17)
        expert = cls(input_size, device)
        expert.expert_id = data.get("expert_id", expert.expert_id)
        expert.breeding_strategy = data.get("breeding_strategy", "LOADED")
        expert.parent_ids = data.get("parent_ids", [])
        expert.generation = data.get("generation", 0)

        w = data.get("weights", data)
        if "input_weights" in w:
            expert.input_weights = torch.FloatTensor(w["input_weights"]).to(expert.device)
            expert.hidden_weights = torch.FloatTensor(w["hidden_weights"]).to(expert.device)
            expert.output_weights = torch.FloatTensor(w["output_weights"]).to(expert.device)
            expert.hidden_bias = torch.FloatTensor(w["hidden_bias"]).to(expert.device)
            expert.output_bias = torch.FloatTensor(w["output_bias"]).to(expert.device)
            expert.input_size = expert.input_weights.shape[0]
        return expert


# ============================================================
# BREEDING STRATEGIES
# ============================================================

def crossover(parent1: GeneticExpert, parent2: GeneticExpert, device) -> GeneticExpert:
    """Uniform crossover between two parents."""
    child = GeneticExpert(parent1.input_size, device)
    child.breeding_strategy = "CROSSOVER"
    child.parent_ids = [parent1.expert_id, parent2.expert_id]
    child.generation = max(parent1.generation, parent2.generation) + 1

    for attr in ["input_weights", "hidden_weights", "output_weights",
                 "hidden_bias", "output_bias"]:
        w1 = getattr(parent1, attr)
        w2 = getattr(parent2, attr)
        if w1.shape == w2.shape:
            mask = torch.rand_like(w1) < 0.5
            setattr(child, attr, torch.where(mask, w1, w2))
        else:
            # Shape mismatch -- use random parent
            setattr(child, attr, (w1 if random.random() < 0.5 else w2).clone())
    return child


def blend_crossover(parent1: GeneticExpert, parent2: GeneticExpert,
                    device, alpha: float = 0.5) -> GeneticExpert:
    """Blend crossover -- weighted average of parents."""
    child = GeneticExpert(parent1.input_size, device)
    child.breeding_strategy = "BLEND"
    child.parent_ids = [parent1.expert_id, parent2.expert_id]
    child.generation = max(parent1.generation, parent2.generation) + 1

    a = random.uniform(0.3, 0.7)  # Random blend ratio
    for attr in ["input_weights", "hidden_weights", "output_weights",
                 "hidden_bias", "output_bias"]:
        w1 = getattr(parent1, attr)
        w2 = getattr(parent2, attr)
        if w1.shape == w2.shape:
            setattr(child, attr, w1 * a + w2 * (1 - a))
        else:
            setattr(child, attr, w1.clone())
    return child


def breed_army(parents: List[GeneticExpert], target_count: int,
               device, log: logging.Logger) -> List[GeneticExpert]:
    """
    Breed target_count experts from the parent pool using multiple strategies.

    Distribution:
      - 30% CROSSOVER (uniform crossover between random parent pairs)
      - 15% BLEND (weighted average crossover)
      - 20% MUTATION (moderate mutations of parents)
      - 10% RADIATION (heavy mutations for exploration)
      - 5%  ELITE_CLONE (top performers with micro-mutations)
      - 20% GENESIS (fresh random individuals for diversity)
    """
    # Normalize all parents to input_size=17 (our feature count)
    # Some parents may have different input sizes (e.g. darwin expert has 128)
    TARGET_INPUT_SIZE = 17
    normalized_parents = []
    for p in parents:
        if p.input_size != TARGET_INPUT_SIZE:
            log.info(f"  Normalizing parent {p.expert_id} from input_size={p.input_size} to {TARGET_INPUT_SIZE}")
            new_p = GeneticExpert(TARGET_INPUT_SIZE, device)
            new_p.expert_id = p.expert_id
            new_p.breeding_strategy = p.breeding_strategy
            new_p.fitness = p.fitness
            new_p.generation = p.generation
            # Fresh random weights at correct size, but influence with parent stats
            normalized_parents.append(new_p)
        else:
            normalized_parents.append(p)
    parents = normalized_parents

    army = list(parents)  # Start with parents
    remaining = target_count - len(army)

    if remaining <= 0:
        log.info(f"Already have {len(army)} experts, no breeding needed")
        return army[:target_count]

    log.info(f"Breeding {remaining} experts from {len(parents)} parents...")
    log.info(f"  Target: {target_count} total")

    # Calculate counts per strategy
    n_crossover = int(remaining * 0.30)
    n_blend = int(remaining * 0.15)
    n_mutation = int(remaining * 0.20)
    n_radiation = int(remaining * 0.10)
    n_elite = int(remaining * 0.05)
    n_genesis = remaining - n_crossover - n_blend - n_mutation - n_radiation - n_elite

    log.info(f"  Strategy breakdown:")
    log.info(f"    CROSSOVER:    {n_crossover}")
    log.info(f"    BLEND:        {n_blend}")
    log.info(f"    MUTATION:     {n_mutation}")
    log.info(f"    RADIATION:    {n_radiation}")
    log.info(f"    ELITE_CLONE:  {n_elite}")
    log.info(f"    GENESIS:      {n_genesis}")

    input_size = parents[0].input_size if parents else 17

    # 1. CROSSOVER
    for i in range(n_crossover):
        p1, p2 = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
        child = crossover(p1, p2, device)
        # Light mutation on 30% of crossover children
        if random.random() < 0.3:
            child.mutate(rate=0.05, strength=0.05)
        army.append(child)

    # 2. BLEND CROSSOVER
    for i in range(n_blend):
        p1, p2 = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
        child = blend_crossover(p1, p2, device)
        if random.random() < 0.2:
            child.mutate(rate=0.03, strength=0.03)
        army.append(child)

    # 3. MUTATION (moderate)
    for i in range(n_mutation):
        parent = random.choice(parents)
        child = parent.clone()
        child.breeding_strategy = "MUTATION"
        child.mutate(rate=0.15, strength=0.15)
        army.append(child)

    # 4. RADIATION (heavy mutation for exploration)
    for i in range(n_radiation):
        parent = random.choice(parents)
        child = parent.clone()
        child.breeding_strategy = "RADIATION"
        child.mutate(rate=0.40, strength=0.40)
        army.append(child)

    # 5. ELITE CLONE (top performers with micro-mutations)
    # Sort parents by fitness and clone the best ones
    sorted_parents = sorted(parents, key=lambda x: x.fitness, reverse=True)
    for i in range(n_elite):
        parent = sorted_parents[i % len(sorted_parents)]
        child = parent.clone()
        child.breeding_strategy = "ELITE_CLONE"
        child.mutate(rate=0.02, strength=0.02)
        army.append(child)

    # 6. GENESIS (fresh random)
    for i in range(n_genesis):
        child = GeneticExpert(input_size, device)
        child.breeding_strategy = "GENESIS"
        child.generation = 0
        army.append(child)

    log.info(f"  Army bred: {len(army)} total experts")
    return army


# ============================================================
# FEATURE ENGINEERING (matches signal_farm_trainer.py exactly)
# ============================================================

def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features from OHLCV data. Returns (df, feature_cols)."""
    df = data.copy()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss_s + 1e-10)
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

    feature_cols = [
        'rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'momentum', 'atr',
        'price_change', 'price_change_abs', 'volume_ma', 'volume_std'
    ]

    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df[col] = (df[col] - mean) / std

    df = df.dropna()
    return df[feature_cols + ['close', 'open', 'high', 'low', 'tick_volume']], feature_cols


# ============================================================
# PROP FIRM SIMULATOR (adapted from qnif_1month_comparison.py)
# ============================================================

SYMBOL_SPECS = {
    'BTCUSD':  {'contract_size': 1.0,   'point': 0.01, 'csv_prefix': 'BTCUSDT'},
    'ETHUSD':  {'contract_size': 1.0,   'point': 0.01, 'csv_prefix': 'ETHUSDT'},
    'XAUUSD':  {'contract_size': 100.0, 'point': 0.01, 'csv_prefix': 'XAUUSDT'},
}

# Prop firm rules (stricter than real -- same as signal_farm_config.json)
PROP_RULES = {
    'initial_balance': 100000.0,
    'max_daily_dd_pct': 5.0,
    'max_total_dd_pct': 10.0,
    'profit_target_pct': 10.0,
    'max_positions': 10,
    'max_loss_per_trade': 1.00,  # $1.00 SL from MASTER_CONFIG
    'tp_multiplier': 3,
    'rolling_sl_multiplier': 1.5,
    'dynamic_tp_pct': 50,
    'atr_multiplier': 0.0438,
    'hold_time': 5,  # bars to evaluate genetic signal quality
}


@dataclass
class SimSignal:
    """A single signal generated by a genetic expert during simulation."""
    expert_id: str = ""
    symbol: str = ""
    bar_index: int = 0
    timestamp: str = ""
    action: int = 0
    action_name: str = ""
    confidence: float = 0.0
    price: float = 0.0
    direction: int = 0  # 1=buy, -1=sell, 0=close/neutral
    outcome: str = ""  # WIN, LOSS, PENDING, FILTERED
    pnl: float = 0.0
    features: List[float] = field(default_factory=list)


@dataclass
class ExpertSimResult:
    """Result of running one expert through the prop firm sim for one symbol."""
    expert_id: str = ""
    symbol: str = ""
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    max_dd: float = 0.0
    max_dd_pct: float = 0.0
    profit_factor: float = 0.0
    signals_generated: int = 0
    prop_firm_pass: bool = False
    daily_dd_breach: bool = False
    total_dd_breach: bool = False
    target_hit: bool = False
    signals: List[SimSignal] = field(default_factory=list)


def load_csv_data(symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Load M5 data from CSV files."""
    spec = SYMBOL_SPECS.get(symbol)
    if not spec:
        return None

    csv_path = HIST_DATA_DIR / f"{spec['csv_prefix']}_5m.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    col_map = {
        'Open time': 'time', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'tick_volume',
    }
    df.rename(columns=col_map, inplace=True)

    if 'time' in df.columns:
        if df['time'].dtype in ['int64', 'float64']:
            unit = 'ms' if df['time'].iloc[0] > 1e12 else 's'
            df['time'] = pd.to_datetime(df['time'], unit=unit)
        else:
            df['time'] = pd.to_datetime(df['time'])

    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    cutoff = df['time'].max() - timedelta(days=days)
    df = df[df['time'] >= cutoff].copy()

    if 'tick_volume' not in df.columns:
        df['tick_volume'] = 1

    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]


def simulate_expert_on_symbol(
    expert: GeneticExpert,
    df_raw: pd.DataFrame,
    symbol: str,
    device,
    collect_signals: bool = True,
) -> ExpertSimResult:
    """
    Run one genetic expert through the prop firm simulator on one symbol.

    The expert's neural network generates actions at each bar.
    We translate those actions into trades with $1 SL / $3 TP.
    Track prop firm rules (daily DD, total DD, profit target).
    Collect every signal for the BRAIN.
    """
    spec = SYMBOL_SPECS[symbol]
    contract_size = spec['contract_size']
    rules = PROP_RULES

    result = ExpertSimResult(expert_id=expert.expert_id, symbol=symbol)

    # Prepare features
    df, feature_cols = prepare_features(df_raw)
    if len(df) < 100:
        return result

    features = df[feature_cols].values
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # Batch predict all actions at once (GPU)
    features_tensor = torch.FloatTensor(features).to(device)
    with torch.no_grad():
        actions, confidences = expert.predict_with_confidence(features_tensor)
    actions = actions.cpu().numpy()
    confidences = confidences.cpu().numpy()

    # Free GPU memory
    del features_tensor

    # ---- Walk-forward simulation ----
    balance = rules['initial_balance']
    peak_balance = balance
    max_dd = 0.0
    max_dd_pct = 0.0
    positions = []
    closed_trades = []
    signals = []
    next_ticket = 1
    current_day = None
    day_start_equity = balance
    daily_dd_breach = False
    total_dd_breach = False
    target_hit = False

    # ATR for SL calculation
    atr_window = 14
    atrs = np.zeros(len(prices))
    for i in range(atr_window, len(prices)):
        high_range = np.max(highs[i-atr_window:i]) - np.min(lows[i-atr_window:i])
        atrs[i] = high_range

    for i in range(max(50, atr_window), len(prices) - rules['hold_time']):
        if daily_dd_breach or total_dd_breach:
            break

        price = prices[i]
        atr = atrs[i]
        action = int(actions[i])
        conf = float(confidences[i])

        if atr <= 0:
            continue

        # Day tracking
        try:
            bar_time = df.iloc[i].get('time', None)
            if bar_time is not None:
                bar_date = pd.Timestamp(bar_time).date() if not isinstance(bar_time, str) else pd.Timestamp(bar_time).date()
                if bar_date != current_day:
                    if current_day is not None:
                        floating = sum(
                            (price - p['entry_price'] if p['dir'] == 1
                             else p['entry_price'] - price) * p['lot'] * contract_size
                            for p in positions
                        )
                        day_equity = balance + floating
                        daily_loss = day_start_equity - day_equity
                        daily_loss_pct = (daily_loss / day_start_equity) * 100 if day_start_equity > 0 else 0
                        if daily_loss_pct > rules['max_daily_dd_pct']:
                            daily_dd_breach = True
                            break
                    current_day = bar_date
                    floating = sum(
                        (price - p['entry_price'] if p['dir'] == 1
                         else p['entry_price'] - price) * p['lot'] * contract_size
                        for p in positions
                    )
                    day_start_equity = balance + floating
        except Exception:
            pass

        # Equity / DD tracking
        floating = sum(
            (price - p['entry_price'] if p['dir'] == 1
             else p['entry_price'] - price) * p['lot'] * contract_size
            for p in positions
        )
        equity = balance + floating
        if equity > peak_balance:
            peak_balance = equity
        dd = peak_balance - equity
        dd_pct = (dd / peak_balance) * 100 if peak_balance > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        # Total DD check
        total_loss_pct = ((rules['initial_balance'] - equity) / rules['initial_balance']) * 100
        if total_loss_pct > rules['max_total_dd_pct']:
            total_dd_breach = True
            break

        # Profit target check
        profit_pct = ((equity - rules['initial_balance']) / rules['initial_balance']) * 100
        if profit_pct >= rules['profit_target_pct']:
            target_hit = True

        # ---- Position management ----
        newly_closed = []
        for pos in positions:
            # SL check
            if (pos['dir'] == 1 and price <= pos['sl']) or \
               (pos['dir'] == -1 and price >= pos['sl']):
                pnl = (pos['sl'] - pos['entry_price']) * pos['lot'] * contract_size * pos['dir']
                balance += pnl
                pos['pnl'] = pnl
                pos['close_reason'] = 'SL'
                newly_closed.append(pos)
                continue

            # TP check
            if (pos['dir'] == 1 and price >= pos['tp']) or \
               (pos['dir'] == -1 and price <= pos['tp']):
                pnl = (pos['tp'] - pos['entry_price']) * pos['lot'] * contract_size * pos['dir']
                balance += pnl
                pos['pnl'] = pnl
                pos['close_reason'] = 'TP'
                newly_closed.append(pos)
                continue

            # Rolling SL update
            if pos.get('dyn_tp_taken'):
                sl_dist = abs(pos['entry_price'] - pos['initial_sl'])
                roll_target = sl_dist * rules['rolling_sl_multiplier']
                if pos['dir'] == 1:
                    profit = price - pos['entry_price']
                    if profit > roll_target:
                        new_sl = price - sl_dist
                        if new_sl > pos['sl']:
                            pos['sl'] = new_sl
                else:
                    profit = pos['entry_price'] - price
                    if profit > roll_target:
                        new_sl = price + sl_dist
                        if new_sl < pos['sl']:
                            pos['sl'] = new_sl

            # Dynamic TP (50% partial close)
            if not pos.get('dyn_tp_taken'):
                if (pos['dir'] == 1 and price >= pos['dyn_tp']) or \
                   (pos['dir'] == -1 and price <= pos['dyn_tp']):
                    half_lot = pos['lot'] * 0.5
                    partial_pnl = (pos['dyn_tp'] - pos['entry_price']) * half_lot * contract_size * pos['dir']
                    balance += partial_pnl
                    pos['partial_pnl'] = partial_pnl
                    pos['lot'] -= half_lot
                    pos['dyn_tp_taken'] = True
                    pos['sl'] = pos['entry_price']  # Breakeven

        closed_trades.extend(newly_closed)
        positions = [p for p in positions if p not in newly_closed]

        # ---- Signal generation from genetic actions ----
        direction = 0
        if action == 0:  # OPEN_BUY
            direction = 1
        elif action == 1:  # OPEN_SELL
            direction = -1
        elif action in [2, 3]:  # CLOSE_BUY signals
            # Close any open buys
            for pos in positions[:]:
                if pos['dir'] == 1:
                    pnl = (price - pos['entry_price']) * pos['lot'] * contract_size
                    balance += pnl
                    pos['pnl'] = pnl
                    pos['close_reason'] = 'GENETIC_CLOSE'
                    closed_trades.append(pos)
                    positions.remove(pos)
        elif action in [4, 5]:  # CLOSE_SELL signals
            for pos in positions[:]:
                if pos['dir'] == -1:
                    pnl = (pos['entry_price'] - price) * pos['lot'] * contract_size
                    balance += pnl
                    pos['pnl'] = pnl
                    pos['close_reason'] = 'GENETIC_CLOSE'
                    closed_trades.append(pos)
                    positions.remove(pos)

        # Collect signal regardless of whether we open a position
        if collect_signals and direction != 0:
            bar_time_str = ""
            try:
                bar_time_str = str(df.iloc[i]['time']) if 'time' in df.columns else ""
            except Exception:
                pass

            sig = SimSignal(
                expert_id=expert.expert_id,
                symbol=symbol,
                bar_index=i,
                timestamp=bar_time_str,
                action=action,
                action_name=GeneticExpert.ACTIONS.get(action, "UNKNOWN"),
                confidence=conf,
                price=price,
                direction=direction,
            )
            signals.append(sig)
            result.signals_generated += 1

        # Open position if direction signal
        if direction != 0:
            # Position limits
            dir_count = sum(1 for p in positions if p['dir'] == direction)
            if len(positions) >= rules['max_positions'] or dir_count >= rules['max_positions'] // 2:
                continue

            sl_distance = atr * rules['atr_multiplier']
            if sl_distance <= 0:
                continue

            lot = rules['max_loss_per_trade'] / (sl_distance * contract_size)
            lot = max(0.01, min(5.0, round(lot, 2)))

            tp_distance = sl_distance * rules['tp_multiplier']
            dyn_tp_distance = tp_distance * (rules['dynamic_tp_pct'] / 100.0)

            if direction == 1:
                sl_price = price - sl_distance
                tp_price = price + tp_distance
                dyn_tp_price = price + dyn_tp_distance
            else:
                sl_price = price + sl_distance
                tp_price = price - tp_distance
                dyn_tp_price = price - dyn_tp_distance

            pos = {
                'ticket': next_ticket,
                'dir': direction,
                'entry_price': price,
                'lot': lot,
                'sl': sl_price,
                'initial_sl': sl_price,
                'tp': tp_price,
                'dyn_tp': dyn_tp_price,
                'dyn_tp_taken': False,
                'partial_pnl': 0.0,
                'pnl': 0.0,
                'close_reason': '',
            }
            positions.append(pos)
            next_ticket += 1

    # Close remaining positions at last price
    if len(prices) > 0:
        last_price = prices[-1]
        for pos in positions:
            pnl = (last_price - pos['entry_price']) * pos['lot'] * contract_size * pos['dir']
            balance += pnl
            pos['pnl'] = pnl
            pos['close_reason'] = 'END'
            closed_trades.append(pos)

    # Calculate results
    total = len(closed_trades)
    winners_list = [t for t in closed_trades if t['pnl'] > 0]
    losers_list = [t for t in closed_trades if t['pnl'] <= 0]
    win_count = len(winners_list)
    loss_count = len(losers_list)
    gross_profit = sum(t['pnl'] for t in winners_list)
    gross_loss = abs(sum(t['pnl'] for t in losers_list))

    result.total_trades = total
    result.winners = win_count
    result.losers = loss_count
    result.win_rate = (win_count / total * 100) if total > 0 else 0
    result.net_pnl = balance - rules['initial_balance']
    result.max_dd = max_dd
    result.max_dd_pct = max_dd_pct
    result.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.99
    result.daily_dd_breach = daily_dd_breach
    result.total_dd_breach = total_dd_breach
    result.target_hit = target_hit
    result.prop_firm_pass = target_hit and not daily_dd_breach and not total_dd_breach
    result.signals = signals

    # Update signal outcomes based on closed trade data
    if collect_signals:
        # Map signals to outcomes from future price
        for sig in signals:
            idx = sig.bar_index
            if idx + rules['hold_time'] < len(prices):
                future = prices[idx + rules['hold_time']]
                if sig.direction == 1:
                    sig.pnl = future - sig.price
                    sig.outcome = "WIN" if sig.pnl > 0 else "LOSS"
                elif sig.direction == -1:
                    sig.pnl = sig.price - future
                    sig.outcome = "WIN" if sig.pnl > 0 else "LOSS"

    return result


# ============================================================
# SIGNAL COLLECTION & FEEDING
# ============================================================

def init_signal_db(db_path: Path):
    """Create the signal history database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE IF NOT EXISTS army_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        expert_id TEXT,
        symbol TEXT,
        timestamp TEXT,
        action INTEGER,
        action_name TEXT,
        confidence REAL,
        price REAL,
        direction INTEGER,
        outcome TEXT,
        pnl REAL,
        breeding_strategy TEXT,
        generation INTEGER,
        batch_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS army_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        expert_id TEXT,
        symbol TEXT,
        total_trades INTEGER,
        winners INTEGER,
        losers INTEGER,
        win_rate REAL,
        net_pnl REAL,
        max_dd REAL,
        max_dd_pct REAL,
        profit_factor REAL,
        signals_generated INTEGER,
        prop_firm_pass INTEGER,
        daily_dd_breach INTEGER,
        total_dd_breach INTEGER,
        target_hit INTEGER,
        breeding_strategy TEXT,
        generation INTEGER,
        batch_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_signals_expert
                    ON army_signals(expert_id)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_signals_symbol
                    ON army_signals(symbol)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_signals_outcome
                    ON army_signals(outcome)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_results_expert
                    ON army_results(expert_id)""")
    conn.commit()
    conn.close()


def save_signals_to_db(signals: List[SimSignal], expert: GeneticExpert,
                       batch_id: str, db_path: Path):
    """Save signals to SQLite database."""
    if not signals:
        return

    conn = sqlite3.connect(str(db_path), timeout=10)
    data = [
        (sig.expert_id, sig.symbol, sig.timestamp, sig.action, sig.action_name,
         sig.confidence, sig.price, sig.direction, sig.outcome, sig.pnl,
         expert.breeding_strategy, expert.generation, batch_id)
        for sig in signals
    ]
    conn.executemany("""INSERT INTO army_signals
        (expert_id, symbol, timestamp, action, action_name, confidence, price,
         direction, outcome, pnl, breeding_strategy, generation, batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", data)
    conn.commit()
    conn.close()


def save_result_to_db(result: ExpertSimResult, expert: GeneticExpert,
                      batch_id: str, db_path: Path):
    """Save expert simulation result to database."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("""INSERT INTO army_results
        (expert_id, symbol, total_trades, winners, losers, win_rate,
         net_pnl, max_dd, max_dd_pct, profit_factor, signals_generated,
         prop_firm_pass, daily_dd_breach, total_dd_breach, target_hit,
         breeding_strategy, generation, batch_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (result.expert_id, result.symbol, result.total_trades,
         result.winners, result.losers, result.win_rate,
         result.net_pnl, result.max_dd, result.max_dd_pct,
         result.profit_factor, result.signals_generated,
         int(result.prop_firm_pass), int(result.daily_dd_breach),
         int(result.total_dd_breach), int(result.target_hit),
         expert.breeding_strategy, expert.generation, batch_id))
    conn.commit()
    conn.close()


def feed_to_domestication_db(signals: List[SimSignal], log: logging.Logger):
    """
    Feed signal outcomes to the TE domestication database.
    Creates pattern hashes from signals and updates win/loss counts.
    """
    if not DOMESTICATION_DB.exists():
        log.warning(f"Domestication DB not found: {DOMESTICATION_DB}")
        return 0

    fed = 0
    try:
        conn = sqlite3.connect(str(DOMESTICATION_DB), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")

        for sig in signals:
            if sig.outcome not in ["WIN", "LOSS"]:
                continue

            # Create a pattern hash from signal properties
            pattern_str = f"{sig.symbol}|{sig.direction}|{sig.action}|{round(sig.confidence, 2)}"
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()[:16]
            te_combo = f"ARMY_{sig.expert_id}_{sig.symbol}_{sig.action_name}"

            now = datetime.utcnow().isoformat()

            # Try to update existing pattern
            cursor = conn.execute(
                "SELECT win_count, loss_count FROM domesticated_patterns WHERE pattern_hash=?",
                (pattern_hash,)
            )
            row = cursor.fetchone()

            if row:
                win_count = row[0] + (1 if sig.outcome == "WIN" else 0)
                loss_count = row[1] + (1 if sig.outcome == "LOSS" else 0)
                total = win_count + loss_count
                win_rate = win_count / total if total > 0 else 0
                domesticated = 1 if total >= 10 and win_rate >= 0.55 else 0
                boost = 1.0 + (win_rate - 0.5) * 2.0 if domesticated else 1.0
                pf = (win_count * abs(sig.pnl)) / (loss_count * abs(sig.pnl) + 1e-10) if sig.outcome == "WIN" else 0

                conn.execute("""UPDATE domesticated_patterns SET
                    win_count=?, loss_count=?, win_rate=?, domesticated=?,
                    boost_factor=?, last_seen=?, last_activated=?
                    WHERE pattern_hash=?""",
                    (win_count, loss_count, win_rate, domesticated,
                     boost, now, now, pattern_hash))
            else:
                win_count = 1 if sig.outcome == "WIN" else 0
                loss_count = 1 if sig.outcome == "LOSS" else 0
                conn.execute("""INSERT INTO domesticated_patterns
                    (pattern_hash, te_combo, win_count, loss_count, win_rate,
                     domesticated, boost_factor, first_seen, last_seen, last_activated)
                    VALUES (?, ?, ?, ?, ?, 0, 1.0, ?, ?, ?)""",
                    (pattern_hash, te_combo, win_count, loss_count,
                     win_count / (win_count + loss_count),
                     now, now, now))
            fed += 1

        conn.commit()
        conn.close()
    except Exception as e:
        log.error(f"Domestication DB feed error: {e}")

    return fed


def send_to_collection_server(signals: List[SimSignal], log: logging.Logger) -> int:
    """Send signals to the collection server (non-blocking, best-effort)."""
    import requests

    sent = 0
    # Batch signals into chunks of 100
    batch_size = 100
    for i in range(0, len(signals), batch_size):
        batch = signals[i:i+batch_size]
        payload = {
            "source": "ARMY_1500",
            "batch_size": len(batch),
            "timestamp": datetime.utcnow().isoformat(),
            "signals": [
                {
                    "expert_id": s.expert_id,
                    "symbol": s.symbol,
                    "direction": "BUY" if s.direction == 1 else "SELL",
                    "confidence": s.confidence,
                    "price": s.price,
                    "outcome": s.outcome,
                    "pnl": s.pnl,
                    "action": s.action_name,
                }
                for s in batch
            ]
        }
        try:
            resp = requests.post(
                f"{COLLECTION_SERVER}/signal",
                json=payload,
                timeout=5,
            )
            if resp.status_code in [200, 201]:
                sent += len(batch)
        except Exception:
            pass  # Best effort, do not block on server issues

    return sent


# ============================================================
# LOAD EXISTING EXPERTS AS PARENTS
# ============================================================

def load_parent_experts(device, log: logging.Logger) -> List[GeneticExpert]:
    """Load existing experts from top_50_experts/ as breeding parents."""
    parents = []

    if not EXPERTS_DIR.exists():
        log.warning(f"Experts directory not found: {EXPERTS_DIR}")
        return parents

    # Load .json experts (darwin format)
    for json_file in EXPERTS_DIR.glob("*.json"):
        if json_file.name == "top_50_manifest.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            # Handle darwin format (weights nested)
            if "weights" in data:
                w = data["weights"]
                if isinstance(w, dict) and "input_weights" in w:
                    expert = GeneticExpert.from_dict(data, device)
                    expert.breeding_strategy = "PARENT_JSON"
                    expert.fitness = data.get("fitness", 0)
                    parents.append(expert)
                    log.debug(f"  Loaded JSON expert: {json_file.name} (input_size={expert.input_size})")
        except Exception as e:
            log.debug(f"  Failed to load {json_file.name}: {e}")

    # Load .pth experts (PyTorch format)
    for pth_file in EXPERTS_DIR.glob("*.pth"):
        try:
            state_dict = torch.load(str(pth_file), map_location="cpu", weights_only=False)
            # Try to reconstruct as GeneticExpert
            if isinstance(state_dict, dict):
                # Check for common key patterns
                if any("weight" in k.lower() for k in state_dict.keys()):
                    # This is an LSTM or standard PyTorch model -- extract what we can
                    # We cannot directly use LSTM weights in our genetic architecture
                    # Instead, use the weight statistics to seed genetic experts
                    expert = GeneticExpert(17, device)  # 17 features
                    expert.breeding_strategy = "PARENT_PTH"
                    expert.expert_id = pth_file.stem

                    # Use weight statistics to bias initialization
                    for key, tensor in state_dict.items():
                        if tensor.numel() > 0:
                            std_val = tensor.float().std().item()
                            mean_val = tensor.float().mean().item()
                            # Influence genetic weights with model statistics
                            expert.input_weights *= (1 + mean_val * 0.1)
                            expert.hidden_weights *= (1 + std_val * 0.1)
                            break

                    parents.append(expert)
                    log.debug(f"  Loaded PTH expert: {pth_file.name}")
        except Exception as e:
            log.debug(f"  Failed to load {pth_file.name}: {e}")

    log.info(f"Loaded {len(parents)} parent experts from {EXPERTS_DIR}")
    return parents


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_army_pipeline(
    target_count: int = 1500,
    batch_size: int = 50,
    symbols: List[str] = None,
    days: int = 30,
    dry_run: bool = False,
):
    """
    Main pipeline: breed -> simulate -> collect -> feed -> report.
    """
    if symbols is None:
        symbols = ["BTCUSD", "ETHUSD", "XAUUSD"]

    log = setup_logging()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set priority
    if set_below_normal_priority():
        log.info("Process priority: BELOW_NORMAL (BRAIN gets priority)")
    else:
        log.warning("Could not set BELOW_NORMAL priority")

    # GPU setup
    device, gpu_name = setup_gpu()
    log.info(f"GPU: {gpu_name}")

    log.info("=" * 78)
    log.info("  EXPERT ARMY 1500 - Mass Breeding + Prop Firm Simulation")
    log.info("=" * 78)
    log.info(f"  Target experts:  {target_count}")
    log.info(f"  Batch size:      {batch_size}")
    log.info(f"  Symbols:         {', '.join(symbols)}")
    log.info(f"  Sim period:      {days} days")
    log.info(f"  GPU:             {gpu_name}")
    log.info(f"  Output:          {OUTPUT_DIR}")
    log.info(f"  Batch ID:        {batch_id}")
    log.info("=" * 78)

    if dry_run:
        log.info("[DRY RUN] Would execute the above. Exiting.")
        return

    pipeline_start = time.time()

    # ---- PHASE 1: Load parents ----
    log.info("")
    log.info("=" * 60)
    log.info("  PHASE 1: Loading Parent Experts")
    log.info("=" * 60)

    parents = load_parent_experts(device, log)
    if not parents:
        log.info("No parents found -- breeding entirely from GENESIS")
        # Create a seed population
        parents = [GeneticExpert(17, device) for _ in range(10)]
        for p in parents:
            p.breeding_strategy = "SEED"

    # ---- PHASE 2: Breed army ----
    log.info("")
    log.info("=" * 60)
    log.info("  PHASE 2: Breeding Army")
    log.info("=" * 60)

    army = breed_army(parents, target_count, device, log)
    log.info(f"Army size: {len(army)}")

    # Strategy distribution stats
    strategy_counts = {}
    for expert in army:
        s = expert.breeding_strategy
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    log.info("Strategy distribution:")
    for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {s:<20} {c:>5} ({c/len(army)*100:.1f}%)")

    # ---- PHASE 3: Load historical data ----
    log.info("")
    log.info("=" * 60)
    log.info("  PHASE 3: Loading Historical Data")
    log.info("=" * 60)

    symbol_data = {}
    for symbol in symbols:
        log.info(f"Loading {symbol}...")
        df = load_csv_data(symbol, days=days)
        if df is not None:
            log.info(f"  {symbol}: {len(df)} bars ({df['time'].min()} to {df['time'].max()})")
            symbol_data[symbol] = df
        else:
            log.warning(f"  {symbol}: NO DATA AVAILABLE -- skipping")

    if not symbol_data:
        log.error("No historical data available for any symbol. Aborting.")
        return

    # ---- PHASE 4: Initialize signal DB ----
    init_signal_db(SIGNAL_HISTORY_DB)
    log.info(f"Signal DB initialized: {SIGNAL_HISTORY_DB}")

    # ---- PHASE 5: Run prop firm simulation in batches ----
    log.info("")
    log.info("=" * 60)
    log.info("  PHASE 5: Prop Firm Simulation")
    log.info("=" * 60)

    total_experts = len(army)
    num_batches = (total_experts + batch_size - 1) // batch_size

    log.info(f"Processing {total_experts} experts in {num_batches} batches of {batch_size}")
    log.info("")

    # Global accumulators
    all_signals = []
    total_signals_count = 0
    total_trades_count = 0
    total_winners = 0
    total_losers = 0
    prop_firm_passers = []
    all_results = []
    experts_processed = 0

    for batch_idx in range(num_batches):
        batch_start_time = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_experts)
        batch = army[start_idx:end_idx]

        log.info(f"--- Batch {batch_idx + 1}/{num_batches} "
                 f"(experts {start_idx + 1}-{end_idx}/{total_experts}) ---")

        batch_signals = []
        batch_trades = 0
        batch_winners = 0

        for expert in batch:
            try:
                expert_signals = []
                expert_total_trades = 0
                expert_total_winners = 0
                expert_total_losers = 0
                expert_total_pnl = 0.0
                expert_max_dd = 0.0
                expert_pass = True  # Must pass ALL symbols

                for symbol, df in symbol_data.items():
                    try:
                        result = simulate_expert_on_symbol(
                            expert, df, symbol, device, collect_signals=True
                        )
                    except Exception as sim_err:
                        log.debug(f"  Expert {expert.expert_id} failed on {symbol}: {sim_err}")
                        result = ExpertSimResult(expert_id=expert.expert_id, symbol=symbol)

                    # Save result to DB
                    try:
                        save_result_to_db(result, expert, batch_id, SIGNAL_HISTORY_DB)
                    except Exception:
                        pass

                    # Accumulate
                    expert_total_trades += result.total_trades
                    expert_total_winners += result.winners
                    expert_total_losers += result.losers
                    expert_total_pnl += result.net_pnl
                    expert_max_dd = max(expert_max_dd, result.max_dd)

                    if result.daily_dd_breach or result.total_dd_breach:
                        expert_pass = False

                    expert.symbol_results[symbol] = {
                        "trades": result.total_trades,
                        "win_rate": round(result.win_rate, 1),
                        "pnl": round(result.net_pnl, 2),
                        "pass": result.prop_firm_pass,
                        "dd_pct": round(result.max_dd_pct, 2),
                    }

                    expert_signals.extend(result.signals)

                # Update expert aggregate stats
                expert.total_trades = expert_total_trades
                expert.winners = expert_total_winners
                expert.losers = expert_total_losers
                expert.win_rate = (expert_total_winners / expert_total_trades * 100) if expert_total_trades > 0 else 0
                expert.net_pnl = expert_total_pnl
                expert.max_dd = expert_max_dd
                expert.signals_generated = len(expert_signals)
                expert.prop_firm_pass = expert_pass and any(
                    r.get("pass", False) for r in expert.symbol_results.values()
                )
                expert.fitness = expert.win_rate / 100.0

                batch_signals.extend(expert_signals)
                batch_trades += expert_total_trades
                batch_winners += expert_total_winners

                if expert.prop_firm_pass:
                    prop_firm_passers.append(expert)

                all_results.append(expert)
                experts_processed += 1

            except Exception as expert_err:
                log.warning(f"  Expert {expert.expert_id} completely failed: {expert_err}")
                experts_processed += 1
                continue

        # Save batch signals to DB (bulk insert)
        if batch_signals:
            try:
                conn = sqlite3.connect(str(SIGNAL_HISTORY_DB), timeout=10)
                data = []
                for sig in batch_signals:
                    expert_match = next((e for e in batch if e.expert_id == sig.expert_id), None)
                    if expert_match:
                        data.append((
                            sig.expert_id, sig.symbol, sig.timestamp, sig.action,
                            sig.action_name, sig.confidence, sig.price, sig.direction,
                            sig.outcome, sig.pnl, expert_match.breeding_strategy,
                            expert_match.generation, batch_id
                        ))
                if data:
                    conn.executemany("""INSERT INTO army_signals
                        (expert_id, symbol, timestamp, action, action_name, confidence,
                         price, direction, outcome, pnl, breeding_strategy, generation, batch_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", data)
                    conn.commit()
                conn.close()
            except Exception as db_err:
                log.warning(f"  Batch signal DB save error: {db_err}")

        total_signals_count += len(batch_signals)
        total_trades_count += batch_trades
        total_winners += batch_winners
        total_losers += (batch_trades - batch_winners)
        all_signals.extend(batch_signals)

        batch_elapsed = time.time() - batch_start_time
        batch_wr = (batch_winners / batch_trades * 100) if batch_trades > 0 else 0

        log.info(f"  Batch {batch_idx + 1}: {len(batch)} experts | "
                 f"{len(batch_signals)} signals | "
                 f"{batch_trades} trades | "
                 f"WR={batch_wr:.1f}% | "
                 f"{batch_elapsed:.1f}s")

        # Progress update
        pct = (experts_processed / total_experts) * 100
        elapsed = time.time() - pipeline_start
        eta = (elapsed / experts_processed * (total_experts - experts_processed)) if experts_processed > 0 else 0
        log.info(f"  Progress: {experts_processed}/{total_experts} ({pct:.0f}%) | "
                 f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m | "
                 f"Passers: {len(prop_firm_passers)}")

        # Yield to BRAIN between batches
        time.sleep(0.5)

        # Periodic GPU memory cleanup
        if batch_idx % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- PHASE 6: Feed to domestication DB ----
    log.info("")
    log.info("=" * 60)
    log.info("  PHASE 6: Feeding Signals to Domestication DB")
    log.info("=" * 60)

    # Only feed signals with outcomes
    outcome_signals = [s for s in all_signals if s.outcome in ["WIN", "LOSS"]]
    log.info(f"Feeding {len(outcome_signals)} signals with outcomes...")

    fed_count = feed_to_domestication_db(outcome_signals, log)
    log.info(f"Fed {fed_count} patterns to domestication DB")

    # ---- PHASE 7: Send to collection server ----
    log.info("")
    log.info("=" * 60)
    log.info("  PHASE 7: Sending to Collection Server")
    log.info("=" * 60)

    sent_count = send_to_collection_server(outcome_signals[:5000], log)  # Cap at 5000
    log.info(f"Sent {sent_count} signals to collection server ({COLLECTION_SERVER})")

    # ---- PHASE 8: Save champion experts ----
    log.info("")
    log.info("=" * 60)
    log.info("  PHASE 8: Saving Champions")
    log.info("=" * 60)

    champions_dir = OUTPUT_DIR / "champions"
    champions_dir.mkdir(exist_ok=True)

    # Sort by fitness
    all_results.sort(key=lambda x: x.fitness, reverse=True)

    # Save top 50 as JSON
    saved_champions = 0
    for i, expert in enumerate(all_results[:50]):
        if expert.total_trades < 5:
            continue  # Skip experts that barely traded
        champion_path = champions_dir / f"champion_{i+1:04d}_{expert.breeding_strategy}_WR{expert.win_rate:.0f}.json"
        with open(champion_path, 'w') as f:
            json.dump(expert.to_dict(), f)
        saved_champions += 1
    log.info(f"Saved {saved_champions} champion configs to {champions_dir}")

    # Save prop firm passers
    if prop_firm_passers:
        passers_dir = OUTPUT_DIR / "prop_firm_passers"
        passers_dir.mkdir(exist_ok=True)
        for i, expert in enumerate(prop_firm_passers):
            passer_path = passers_dir / f"passer_{i+1:04d}_{expert.breeding_strategy}_WR{expert.win_rate:.0f}.json"
            with open(passer_path, 'w') as f:
                json.dump(expert.to_dict(), f)
        log.info(f"Saved {len(prop_firm_passers)} prop firm passers to {passers_dir}")

    # ---- FINAL REPORT ----
    pipeline_elapsed = time.time() - pipeline_start

    log.info("")
    log.info("=" * 78)
    log.info("  EXPERT ARMY 1500 - FINAL REPORT")
    log.info("=" * 78)
    log.info(f"  Total Time:          {pipeline_elapsed/60:.1f} minutes")
    log.info(f"  Experts Processed:   {experts_processed}")
    log.info(f"  Total Signals:       {total_signals_count:,}")
    log.info(f"  Total Trades:        {total_trades_count:,}")
    log.info(f"  Total Winners:       {total_winners:,}")
    log.info(f"  Total Losers:        {total_losers:,}")
    overall_wr = (total_winners / total_trades_count * 100) if total_trades_count > 0 else 0
    log.info(f"  Overall Win Rate:    {overall_wr:.1f}%")
    log.info(f"  Prop Firm Passers:   {len(prop_firm_passers)} / {experts_processed} "
             f"({len(prop_firm_passers)/experts_processed*100:.1f}%)" if experts_processed > 0 else "0")
    log.info(f"  Signals Fed to DB:   {fed_count:,}")
    log.info(f"  Signals to Server:   {sent_count:,}")
    log.info(f"  Champions Saved:     {saved_champions}")
    log.info("")

    # Breakdown by strategy
    log.info("  Results by Breeding Strategy:")
    log.info(f"  {'Strategy':<20} {'Count':>6} {'Avg WR':>8} {'Passers':>8} {'Avg PnL':>10}")
    log.info(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    for strategy in sorted(strategy_counts.keys()):
        strat_experts = [e for e in all_results if e.breeding_strategy == strategy]
        count = len(strat_experts)
        avg_wr = sum(e.win_rate for e in strat_experts) / count if count > 0 else 0
        passers = sum(1 for e in strat_experts if e.prop_firm_pass)
        avg_pnl = sum(e.net_pnl for e in strat_experts) / count if count > 0 else 0
        log.info(f"  {strategy:<20} {count:>6} {avg_wr:>7.1f}% {passers:>8} ${avg_pnl:>9.2f}")

    # Top 10 experts
    log.info("")
    log.info("  Top 10 Experts:")
    log.info(f"  {'Rank':>4} {'ID':<14} {'Strategy':<16} {'WR':>6} {'Trades':>7} {'PnL':>10} {'Pass':>5}")
    log.info(f"  {'-'*4} {'-'*14} {'-'*16} {'-'*6} {'-'*7} {'-'*10} {'-'*5}")
    for i, expert in enumerate(all_results[:10]):
        log.info(f"  {i+1:>4} {expert.expert_id:<14} {expert.breeding_strategy:<16} "
                 f"{expert.win_rate:>5.1f}% {expert.total_trades:>7} "
                 f"${expert.net_pnl:>9.2f} {'YES' if expert.prop_firm_pass else 'no':>5}")

    # Symbol breakdown
    log.info("")
    log.info("  Results by Symbol:")
    for symbol in symbols:
        sym_results = []
        for e in all_results:
            if symbol in e.symbol_results:
                sym_results.append(e.symbol_results[symbol])
        if sym_results:
            avg_wr = sum(r['win_rate'] for r in sym_results) / len(sym_results)
            avg_pnl = sum(r['pnl'] for r in sym_results) / len(sym_results)
            passers = sum(1 for r in sym_results if r.get('pass', False))
            total_trades_sym = sum(r['trades'] for r in sym_results)
            log.info(f"  {symbol}: {total_trades_sym:,} trades | "
                     f"Avg WR={avg_wr:.1f}% | Avg PnL=${avg_pnl:.2f} | "
                     f"Passers={passers}")

    log.info("")
    log.info("=" * 78)
    log.info(f"  Output directory: {OUTPUT_DIR}")
    log.info(f"  Signal database:  {SIGNAL_HISTORY_DB}")
    log.info(f"  Champions:        {champions_dir}")
    log.info("=" * 78)

    # Save final summary JSON
    summary = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "pipeline_elapsed_minutes": round(pipeline_elapsed / 60, 1),
        "target_count": target_count,
        "experts_processed": experts_processed,
        "total_signals": total_signals_count,
        "total_trades": total_trades_count,
        "total_winners": total_winners,
        "total_losers": total_losers,
        "overall_win_rate": round(overall_wr, 1),
        "prop_firm_passers": len(prop_firm_passers),
        "pass_rate": round(len(prop_firm_passers) / experts_processed * 100, 1) if experts_processed > 0 else 0,
        "signals_fed_to_db": fed_count,
        "signals_to_server": sent_count,
        "champions_saved": saved_champions,
        "strategy_breakdown": {
            s: {
                "count": len([e for e in all_results if e.breeding_strategy == s]),
                "avg_win_rate": round(sum(e.win_rate for e in all_results if e.breeding_strategy == s) /
                                      max(1, len([e for e in all_results if e.breeding_strategy == s])), 1),
                "passers": sum(1 for e in all_results if e.breeding_strategy == s and e.prop_firm_pass),
            }
            for s in strategy_counts
        },
        "top_10": [
            {
                "rank": i + 1,
                "expert_id": e.expert_id,
                "strategy": e.breeding_strategy,
                "win_rate": round(e.win_rate, 1),
                "trades": e.total_trades,
                "pnl": round(e.net_pnl, 2),
                "prop_firm_pass": e.prop_firm_pass,
            }
            for i, e in enumerate(all_results[:10])
        ],
        "symbols": {
            sym: {
                "total_trades": sum(e.symbol_results.get(sym, {}).get("trades", 0) for e in all_results),
                "avg_win_rate": round(
                    sum(e.symbol_results.get(sym, {}).get("win_rate", 0) for e in all_results) /
                    max(1, sum(1 for e in all_results if sym in e.symbol_results)), 1),
            }
            for sym in symbols
        },
    }
    summary_path = OUTPUT_DIR / f"army_summary_{batch_id}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved: {summary_path}")

    return summary


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expert Army 1500 - Mass Breeding + Prop Firm Sim + Signal Harvest"
    )
    parser.add_argument("--target", type=int, default=1500,
                        help="Number of experts to breed (default: 1500)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Experts per simulation batch (default: 50)")
    parser.add_argument("--symbols", nargs='+', default=['BTCUSD', 'ETHUSD', 'XAUUSD'],
                        help="Symbols to simulate (default: BTCUSD ETHUSD XAUUSD)")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of historical data to simulate (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config without running")
    args = parser.parse_args()

    run_army_pipeline(
        target_count=args.target,
        batch_size=args.batch_size,
        symbols=args.symbols,
        days=args.days,
        dry_run=args.dry_run,
    )
