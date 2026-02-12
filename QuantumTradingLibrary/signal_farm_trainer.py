"""
Signal Farm Trainer - 34 Round Protocol
========================================
Revised training pipeline:
  - 17 timeframes x 2 cycles = 34 total rounds
  - 1 year lookback per timeframe (12 months)
  - 4 months train / 2 months test per cycle
  - Cycle 2 shifts forward to jostle data alignment
  - Single symbol focus (BTCUSD to start)
  - Learn ALL timeframes first, then EA decides per-TF

Architecture:
  - Uses existing ETARE genetic algorithm from etare_walkforward_trainer.py
  - GPU-accelerated via DirectML (AMD RX 6800 XT)
  - LSTM stays on CPU per CLAUDE.md rules
  - Parallel account evaluation via gpu_pool_manager

DO NOT SKIP STEPS. Flag user if a step should be skipped.

Usage:
    python signal_farm_trainer.py
    python signal_farm_trainer.py --symbol BTCUSD --dry-run
"""

import numpy as np
import pandas as pd
import torch
import json
import time
import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy
from enum import Enum
from dataclasses import dataclass
import random
import warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# GPU Setup
try:
    import torch_directml
    DEVICE = torch_directml.device()
    GPU_NAME = torch_directml.device_name(0)
    print(f"[TRAINER] GPU: {GPU_NAME}")
except ImportError:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU"
    print("[TRAINER] WARNING: DirectML not available, using CPU")

# Load farm config
FARM_CONFIG_PATH = SCRIPT_DIR / "signal_farm_config.json"
with open(FARM_CONFIG_PATH) as f:
    FARM_CONFIG = json.load(f)

# MT5 timeframe constants
import MetaTrader5 as mt5

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M2":  mt5.TIMEFRAME_M2,
    "M3":  mt5.TIMEFRAME_M3,
    "M4":  mt5.TIMEFRAME_M4,
    "M5":  mt5.TIMEFRAME_M5,
    "M6":  mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H2":  mt5.TIMEFRAME_H2,
    "H3":  mt5.TIMEFRAME_H3,
    "H4":  mt5.TIMEFRAME_H4,
    "H6":  mt5.TIMEFRAME_H6,
    "H8":  mt5.TIMEFRAME_H8,
}


# ============================================================
# GENETIC ALGORITHM (from etare_walkforward_trainer.py)
# ============================================================

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
        gc = FARM_CONFIG["GENETIC"]
        self.mutation_rate = gc.get("MUTATION_RATE", mutation_rate)
        self.mutation_strength = gc.get("MUTATION_STRENGTH", mutation_strength)

        self.weights = GeneticWeights(
            input_weights=torch.empty(input_size, 128, device=DEVICE).uniform_(-0.5, 0.5),
            hidden_weights=torch.empty(128, 64, device=DEVICE).uniform_(-0.5, 0.5),
            output_weights=torch.empty(64, len(Action), device=DEVICE).uniform_(-0.5, 0.5),
            hidden_bias=torch.empty(128, device=DEVICE).uniform_(-0.5, 0.5),
            output_bias=torch.empty(len(Action), device=DEVICE).uniform_(-0.5, 0.5),
        )
        self.fitness = 0.0
        self.test_fitness = 0.0
        self.successful_trades = 0
        self.total_trades = 0
        self.generation = 0
        self.timeframe = ""
        self.cycle = 0
        self.round_num = 0

    def batch_predict(self, states: torch.Tensor) -> torch.Tensor:
        mean = states.mean(dim=1, keepdim=True)
        std = states.std(dim=1, keepdim=True) + 1e-8
        states = (states - mean) / std
        hidden = torch.tanh(torch.matmul(states, self.weights.input_weights) + self.weights.hidden_bias)
        hidden2 = torch.tanh(torch.matmul(hidden, self.weights.hidden_weights))
        output = torch.matmul(hidden2, self.weights.output_weights) + self.weights.output_bias
        return torch.argmax(output, dim=1)

    def mutate(self):
        for wt in [self.weights.input_weights, self.weights.hidden_weights, self.weights.output_weights]:
            mask = torch.rand_like(wt) < self.mutation_rate
            noise = torch.randn_like(wt) * self.mutation_strength
            wt[mask] += noise[mask]

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
            'generation': self.generation,
            'round': self.round_num,
        }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(data: pd.DataFrame) -> tuple:
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
    return df[feature_cols + ['close']], feature_cols


# ============================================================
# EVALUATION
# ============================================================

def evaluate_individual(individual, features_tensor, prices, hold_time=5):
    """Evaluate a genetic individual against data."""
    actions = individual.batch_predict(features_tensor).cpu().numpy()

    wins, losses = 0, 0
    position = None
    entry_price = 0

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
            if position == 'buy' and action in [Action.CLOSE_BUY_PROFIT.value, Action.CLOSE_BUY_LOSS.value]:
                if future_price - entry_price > 0:
                    wins += 1
                else:
                    losses += 1
                position = None
            elif position == 'sell' and action in [Action.CLOSE_SELL_PROFIT.value, Action.CLOSE_SELL_LOSS.value]:
                if entry_price - future_price > 0:
                    wins += 1
                else:
                    losses += 1
                position = None

    total = wins + losses
    wr = wins / total if total > 0 else 0
    individual.successful_trades = wins
    individual.total_trades = total
    return wr, total


def tournament_selection(population, size=5):
    tournament = random.sample(population, min(size, len(population)))
    return max(tournament, key=lambda x: x.fitness)


def crossover(p1, p2, input_size):
    child = TradingIndividual(input_size)
    for attr in ["input_weights", "hidden_weights", "output_weights"]:
        w1 = getattr(p1.weights, attr)
        w2 = getattr(p2.weights, attr)
        mask = torch.rand_like(w1) < 0.5
        setattr(child.weights, attr, torch.where(mask, w1, w2))
    return child


# ============================================================
# DARWIN ROUND
# ============================================================

def run_darwin_round(population, train_tensor, train_prices, test_tensor, test_prices,
                     input_size, is_extinction=False):
    """One evolutionary round with optional extinction event."""
    gc = FARM_CONFIG["GENETIC"]
    elite_size = gc["ELITE_SIZE"]
    pop_size = len(population)

    # Train evaluation
    for ind in population:
        wr, trades = evaluate_individual(ind, train_tensor, train_prices)
        ind.fitness = wr

    population.sort(key=lambda x: x.fitness, reverse=True)

    if is_extinction:
        survivors = population[:elite_size]
        for _ in range(gc["FRESH_INJECTION_COUNT"]):
            fresh = TradingIndividual(input_size)
            evaluate_individual(fresh, train_tensor, train_prices)
            survivors.append(fresh)

        while len(survivors) < pop_size:
            if random.random() < 0.6:
                p1 = tournament_selection(population, gc["TOURNAMENT_SIZE"])
                p2 = tournament_selection(population, gc["TOURNAMENT_SIZE"])
                child = crossover(p1, p2, input_size)
            else:
                child = deepcopy(random.choice(population[:elite_size]))
                child.mutate()
            evaluate_individual(child, train_tensor, train_prices)
            survivors.append(child)

        population = survivors

    # Test evaluation
    for ind in population:
        wr, trades = evaluate_individual(ind, test_tensor, test_prices)
        ind.test_fitness = wr

    population.sort(key=lambda x: x.test_fitness, reverse=True)
    return population


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_data(symbol, timeframe_name, months_back=12):
    """Fetch historical data from MT5."""
    tf_const = TF_MAP[timeframe_name]
    tf_info = next(t for t in FARM_CONFIG["TIMEFRAMES"] if t["name"] == timeframe_name)
    bars_needed = tf_info["bars_per_day"] * 30 * months_back

    # Cap at reasonable limit
    bars_needed = min(bars_needed, 500000)

    print(f"  Fetching {bars_needed:,} bars of {symbol} {timeframe_name}...")
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars_needed)

    if rates is None or len(rates) < 500:
        print(f"  ERROR: Only got {len(rates) if rates is not None else 0} bars")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"  Got {len(df):,} bars: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return df


# ============================================================
# SINGLE TIMEFRAME TRAINING (2 CYCLES)
# ============================================================

def train_timeframe(symbol, timeframe_name, all_champions, round_tracker):
    """
    Train on a single timeframe with 2 cycles.
    Each cycle: 4 months train, 2 months test.
    Cycle 2 shifts forward (jostle).

    Returns list of champion individuals.
    """
    protocol = FARM_CONFIG["PROTOCOL"]
    gc = FARM_CONFIG["GENETIC"]

    # Fetch 1 year of data
    df_raw = fetch_data(symbol, timeframe_name, months_back=protocol["LOOKBACK_MONTHS"])
    if df_raw is None:
        return []

    # Prepare features
    df, feature_cols = prepare_features(df_raw)
    features = df[feature_cols].values
    prices = df['close'].values
    input_size = len(feature_cols)

    print(f"\n  Prepared {len(df):,} bars, {input_size} features")

    tf_info = next(t for t in FARM_CONFIG["TIMEFRAMES"] if t["name"] == timeframe_name)
    bars_per_month = tf_info["bars_per_day"] * 30
    train_bars = bars_per_month * protocol["TRAIN_MONTHS"]
    test_bars = bars_per_month * protocol["TEST_MONTHS"]

    tf_champions = []

    for cycle in range(1, protocol["CYCLES_PER_TIMEFRAME"] + 1):
        # Cycle 2 shifts forward by 1 month (jostle)
        offset = bars_per_month * (cycle - 1) if cycle > 1 else 0

        train_start = offset
        train_end = train_start + train_bars
        test_start = train_end
        test_end = test_start + test_bars

        if test_end > len(features):
            # Not enough data - use what we have
            available = len(features) - train_start
            train_bars_actual = int(available * 0.67)
            test_bars_actual = available - train_bars_actual
            train_end = train_start + train_bars_actual
            test_start = train_end
            test_end = test_start + test_bars_actual
            print(f"  Adjusted: train={train_bars_actual:,}, test={test_bars_actual:,}")

        if train_end - train_start < 200 or test_end - test_start < 100:
            print(f"  SKIP: Not enough data for {timeframe_name} cycle {cycle}")
            continue

        train_features = features[train_start:train_end]
        train_prices = prices[train_start:train_end]
        test_features = features[test_start:test_end]
        test_prices = prices[test_start:test_end]

        train_tensor = torch.FloatTensor(train_features).to(DEVICE)
        test_tensor = torch.FloatTensor(test_features).to(DEVICE)

        # Create population
        population = [TradingIndividual(input_size) for _ in range(gc["POPULATION_SIZE"])]

        round_tracker["current"] += 1
        total = round_tracker["total"]
        current = round_tracker["current"]

        is_extinction = (gc["EXTINCTION_EVERY_N_ROUNDS"] > 0 and
                        current % gc["EXTINCTION_EVERY_N_ROUNDS"] == 0)

        ext_tag = " [EXTINCTION]" if is_extinction else ""
        print(f"\n  --- Round {current}/{total}: {timeframe_name} Cycle {cycle}{ext_tag} ---")
        print(f"  Train: bars {train_start:,}-{train_end:,} ({train_end-train_start:,})")
        print(f"  Test:  bars {test_start:,}-{test_end:,} ({test_end-test_start:,})")

        t0 = time.time()

        population = run_darwin_round(
            population, train_tensor, train_prices,
            test_tensor, test_prices, input_size, is_extinction
        )

        elapsed = time.time() - t0

        # Update metadata
        for ind in population:
            ind.timeframe = timeframe_name
            ind.cycle = cycle
            ind.round_num = current
            ind.generation = current

        best = population[0]
        print(f"  Best: Train={best.fitness*100:.1f}% Test={best.test_fitness*100:.1f}% "
              f"Trades={best.total_trades} ({elapsed:.1f}s)")

        # Pull champions
        min_champs, max_champs = gc["CHAMPIONS_PER_CYCLE"]
        num_pull = random.randint(min_champs, max_champs)
        cycle_champs = deepcopy(population[:num_pull])
        tf_champions.extend(cycle_champs)

        print(f"  Pulled {num_pull} champions: {[f'{c.test_fitness*100:.1f}%' for c in cycle_champs]}")

        # Free GPU memory
        del train_tensor, test_tensor

        # Track for decision gate
        round_tracker["results"].append({
            "round": current,
            "timeframe": timeframe_name,
            "cycle": cycle,
            "best_train_wr": round(best.fitness * 100, 1),
            "best_test_wr": round(best.test_fitness * 100, 1),
            "trades": best.total_trades,
            "is_extinction": is_extinction,
            "elapsed_s": round(elapsed, 1),
        })

    return tf_champions


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def run_signal_farm_training(symbol="BTCUSD", dry_run=False):
    """
    Execute the full 34-round training protocol.

    17 timeframes x 2 cycles = 34 rounds
    """
    protocol = FARM_CONFIG["PROTOCOL"]
    timeframes = [tf["name"] for tf in FARM_CONFIG["TIMEFRAMES"]]
    total_rounds = len(timeframes) * protocol["CYCLES_PER_TIMEFRAME"]

    print("=" * 70)
    print(f"  SIGNAL FARM TRAINER - {protocol['NAME']}")
    print("=" * 70)
    print(f"  Symbol:          {symbol}")
    print(f"  Timeframes:      {len(timeframes)} ({timeframes[0]} -> {timeframes[-1]})")
    print(f"  Cycles/TF:       {protocol['CYCLES_PER_TIMEFRAME']}")
    print(f"  Total Rounds:    {total_rounds}")
    print(f"  Lookback:        {protocol['LOOKBACK_MONTHS']} months")
    print(f"  Train/Test:      {protocol['TRAIN_MONTHS']}mo / {protocol['TEST_MONTHS']}mo")
    print(f"  GPU:             {GPU_NAME}")
    print(f"  Target WR:       {FARM_CONFIG['GENETIC']['TARGET_WIN_RATE']}%")
    print("=" * 70)

    if dry_run:
        print("\n  [DRY RUN] Would execute the above. Exiting.")
        return

    # Initialize MT5
    if not mt5.initialize():
        print("ERROR: MT5 initialization failed. Is MetaTrader 5 running?")
        return

    # Output directory
    output_dir = SCRIPT_DIR / FARM_CONFIG["SIGNALS"]["SIGNAL_OUTPUT_DIR"]
    output_dir.mkdir(exist_ok=True)

    all_champions = []
    round_tracker = {"current": 0, "total": total_rounds, "results": []}

    start_time = time.time()

    for tf_info in FARM_CONFIG["TIMEFRAMES"]:
        tf_name = tf_info["name"]

        print(f"\n{'#' * 70}")
        print(f"# TIMEFRAME: {tf_name} (order {tf_info['order']}/{len(timeframes)})")
        print(f"{'#' * 70}")

        try:
            champions = train_timeframe(symbol, tf_name, all_champions, round_tracker)
            all_champions.extend(champions)

            # Save intermediate results
            if FARM_CONFIG["SIGNALS"]["SAVE_EVERY_N_ROUNDS"] > 0:
                _save_checkpoint(output_dir, all_champions, round_tracker, symbol)

        except Exception as e:
            print(f"  ERROR in {tf_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Decision gate check after first full cycle through all TFs
        if round_tracker["current"] == len(timeframes):
            _check_decision_gate(round_tracker)

    total_time = time.time() - start_time

    # Final results
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total Champions: {len(all_champions)}")
    print(f"  Total Time:      {total_time/60:.1f} minutes")
    print(f"  Rounds:          {round_tracker['current']}/{total_rounds}")

    if all_champions:
        # Sort by test fitness
        all_champions.sort(key=lambda x: x.test_fitness, reverse=True)
        best = all_champions[0]
        print(f"\n  BEST OVERALL:")
        print(f"    Timeframe:  {best.timeframe}")
        print(f"    Cycle:      {best.cycle}")
        print(f"    Test WR:    {best.test_fitness*100:.1f}%")
        print(f"    Trades:     {best.total_trades}")

        target = FARM_CONFIG["GENETIC"]["TARGET_WIN_RATE"]
        if best.test_fitness * 100 >= target:
            print(f"\n  *** TARGET MET: {best.test_fitness*100:.1f}% >= {target}% ***")
        else:
            print(f"\n  Gap to target: {target - best.test_fitness*100:.1f}% needed")

        # Save final results
        _save_final(output_dir, all_champions, round_tracker, symbol, total_time)

    # Print round-by-round summary
    _print_summary(round_tracker)

    mt5.shutdown()
    print(f"\n{'=' * 70}")


def _save_checkpoint(output_dir, champions, tracker, symbol):
    """Save intermediate checkpoint."""
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "rounds_completed": tracker["current"],
        "total_champions": len(champions),
        "results": tracker["results"],
    }
    path = output_dir / "checkpoint.json"
    with open(path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def _save_final(output_dir, champions, tracker, symbol, total_time):
    """Save final training results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save champion experts
    for i, champ in enumerate(champions[:10]):  # Top 10
        expert_path = output_dir / f"champion_{symbol}_{champ.timeframe}_C{champ.cycle}_WR{champ.test_fitness*100:.0f}_{timestamp}.json"
        with open(expert_path, 'w') as f:
            json.dump(champ.to_dict(), f)

    # Save full results
    results_path = output_dir / f"training_results_{timestamp}.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "protocol": FARM_CONFIG["PROTOCOL"],
        "total_time_minutes": round(total_time / 60, 1),
        "total_champions": len(champions),
        "top_10": [
            {
                "rank": i + 1,
                "timeframe": c.timeframe,
                "cycle": c.cycle,
                "test_wr": round(c.test_fitness * 100, 1),
                "train_wr": round(c.fitness * 100, 1),
                "trades": c.total_trades,
            }
            for i, c in enumerate(champions[:10])
        ],
        "round_results": tracker["results"],
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {output_dir}")


def _check_decision_gate(tracker):
    """Check decision gate after first cycle through all timeframes."""
    gate = FARM_CONFIG["DECISION_GATES"]["AFTER_FIRST_CYCLE"]
    if not gate["ENABLED"]:
        return

    results = tracker["results"]
    if len(results) < 2:
        return

    # Check win rate trend
    first_wr = results[0]["best_test_wr"]
    last_wr = results[-1]["best_test_wr"]
    avg_wr = sum(r["best_test_wr"] for r in results) / len(results)
    improvement = last_wr - first_wr

    print(f"\n{'*' * 70}")
    print(f"  DECISION GATE - After First Cycle")
    print(f"{'*' * 70}")
    print(f"  First round WR:  {first_wr:.1f}%")
    print(f"  Last round WR:   {last_wr:.1f}%")
    print(f"  Average WR:      {avg_wr:.1f}%")
    print(f"  Improvement:     {improvement:+.1f}%")

    min_improvement = gate["MIN_WIN_RATE_IMPROVEMENT"]
    if improvement < min_improvement:
        print(f"\n  FLAG: Improvement ({improvement:+.1f}%) below threshold ({min_improvement}%)")
        print(f"  Action: {gate['ACTION_IF_NO_IMPROVEMENT']}")
    else:
        print(f"\n  PASS: Improvement on track")
    print(f"{'*' * 70}")


def _print_summary(tracker):
    """Print round-by-round summary table."""
    results = tracker["results"]
    if not results:
        return

    print(f"\n{'=' * 70}")
    print(f"  ROUND-BY-ROUND SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Round':>5} {'TF':>5} {'Cyc':>3} {'Train%':>7} {'Test%':>7} {'Trades':>6} {'Time':>6} {'Ext':>4}")
    print(f"  {'-'*5} {'-'*5} {'-'*3} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*4}")

    for r in results:
        ext = "YES" if r["is_extinction"] else ""
        print(f"  {r['round']:>5} {r['timeframe']:>5} {r['cycle']:>3} "
              f"{r['best_train_wr']:>6.1f}% {r['best_test_wr']:>6.1f}% "
              f"{r['trades']:>6} {r['elapsed_s']:>5.1f}s {ext:>4}")

    avg_test = sum(r["best_test_wr"] for r in results) / len(results)
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"\n  Average Test WR: {avg_test:.1f}%")
    print(f"  Total Training Time: {total_time/60:.1f} minutes")
    print(f"{'=' * 70}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Farm Trainer - 34 Round Protocol")
    parser.add_argument("--symbol", default="BTCUSD", help="Symbol to train on")
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")
    args = parser.parse_args()

    run_signal_farm_training(symbol=args.symbol, dry_run=args.dry_run)
