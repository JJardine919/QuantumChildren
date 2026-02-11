"""
PROP CHALLENGE CHAOS - Evolutionary Tournament with Control Groups
===================================================================
2000 simulated $100K prop firm accounts compete across 130+ symbols.

Three groups fight for supremacy:
  MAIN (1400):    Raw chaos. Maximum mutation. No filters. Pure DNA.
  JARDINE (300):  Same DNA but every signal passes through a Python port
                  of the 6-gate Jardine's Gate algorithm before entry.
  AOI (300):      Same DNA but signals pass through a simplified AOI
                  biological cascade (CRISPR, Protective Deletion,
                  Toxoplasma, Syncytin).

The question: Does filtering help, hurt, or make no difference when
the underlying DNA is random and mutations are cranked to maximum?

Evolutionary rounds:
  1. All 2000 run the challenge across ALL symbols on the same M5 data
  2. Rank by composite score (WR, PF, cross-symbol consistency, DD)
  3. Top 10% survive as elite (within each group separately)
  4. Next 20% breed (crossover parameters, within group)
  5. Bottom 70% die and are replaced by mutated offspring
  6. Repeat for N evolutionary rounds
  7. Compare group performance to measure filter alpha

Usage:
    python prop_challenge_chaos.py --symbols BTCUSD,ETHUSD,XAUUSD,...
    python prop_challenge_chaos.py --accounts 2000 --rounds 5 --symbols BTCUSD,ETHUSD
    python prop_challenge_chaos.py --accounts 500 --rounds 3 --symbols BTCUSD
"""

import numpy as np
import pandas as pd
import torch
import json
import time
import sys
import os
import argparse
import random
import math
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# ============================================================
# GPU SETUP (torch_directml for AMD RX 6800 XT)
# ============================================================
try:
    import torch_directml
    DEVICE = torch_directml.device()
    GPU_NAME = torch_directml.device_name(0)
except ImportError:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU"

import MetaTrader5 as mt5
from config_loader import (
    MAX_LOSS_DOLLARS, TP_MULTIPLIER, ROLLING_SL_MULTIPLIER,
    DYNAMIC_TP_PERCENT, CONFIDENCE_THRESHOLD
)

# Group identifiers
GROUP_MAIN = "MAIN"
GROUP_JARDINE = "JARDINE"
GROUP_AOI = "AOI"

# Default group sizes (ratios of total)
GROUP_RATIOS = {GROUP_MAIN: 0.70, GROUP_JARDINE: 0.15, GROUP_AOI: 0.15}


# ============================================================
# JARDINE'S GATE v2 - Python Port from JardinesGateAlgorithm.mqh
# ============================================================
# Six-gate quantum filter. Blocks bad signals, passes good ones.
# Ported faithfully from the MQL5 implementation.
# ============================================================

class JardinesGate:
    """
    Python port of the QuantumEdgeFilter from JardinesGateAlgorithm.mqh.

    Six gates in sequence:
      G1: Entropy Gate - block if entropy > threshold (noisy market)
      G2: Interference Gate - block if expert agreement too low
      G3: Confidence Gate - block if signal confidence too low
      G4: Probability Gate - P = amplitude^2 * entropy_factor * interference * confidence
      G5: Direction Gate - pass all (both directions allowed in chaos mode)
      G6: Kill Switch - block if recent win rate below threshold
    """

    def __init__(self,
                 entropy_clean: float = 0.90,
                 confidence_min: float = 0.20,
                 interference_min: float = 0.50,
                 probability_min: float = 0.60,
                 direction_bias: int = 0,       # 0=BOTH, 1=LONG, -1=SHORT
                 kill_switch_wr: float = 0.50,
                 kill_switch_lookback: int = 10):
        # Thresholds
        self.entropy_clean = entropy_clean
        self.confidence_min = confidence_min
        self.interference_min = interference_min
        self.probability_min = probability_min
        self.direction_bias = direction_bias
        self.kill_switch_wr = kill_switch_wr
        self.kill_switch_lookback = kill_switch_lookback

        # Trade history for kill switch
        self.trade_history = deque(maxlen=kill_switch_lookback * 2)

        # Stats
        self.total_checks = 0
        self.passed = 0
        self.blocked_g1 = 0
        self.blocked_g2 = 0
        self.blocked_g3 = 0
        self.blocked_g4 = 0
        self.blocked_g5 = 0
        self.blocked_g6 = 0

    def calculate_entropy_from_prices(self, prices: np.ndarray) -> float:
        """
        Calculate entropy from price array using three methods:
        1. Coefficient of variation of returns
        2. Autocorrelation (high autocorr = predictable = low entropy)
        3. Directional consistency

        Weighted combination: 0.4 * CV + 0.3 * autocorr + 0.3 * direction
        """
        n = len(prices)
        if n < 10:
            return 1.0

        # Returns
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        n_ret = len(returns)

        # Method 1: Coefficient of variation
        mean_abs = np.mean(np.abs(returns))
        std_dev = np.std(returns)
        cv_entropy = min(1.0, std_dev / (mean_abs + 1e-10))

        # Method 2: Autocorrelation (lag-1)
        autocorr = 0.0
        if n_ret > 2:
            x = returns[:-1]
            y = returns[1:]
            n_pairs = len(x)
            sum_xy = np.sum(x * y)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_x2 = np.sum(x * x)
            sum_y2 = np.sum(y * y)
            denom = math.sqrt(
                max(0, (n_pairs * sum_x2 - sum_x ** 2) *
                    (n_pairs * sum_y2 - sum_y ** 2))
            )
            if denom > 0:
                autocorr = (n_pairs * sum_xy - sum_x * sum_y) / denom
        autocorr_entropy = 1.0 - abs(autocorr)

        # Method 3: Directional consistency
        up = np.sum(returns > 0)
        down = np.sum(returns < 0)
        direction_entropy = 1.0 - abs(up - down) / max(n_ret, 1)

        # Weighted combination (matches MQL5 implementation)
        entropy = 0.4 * cv_entropy + 0.3 * autocorr_entropy + 0.3 * direction_entropy
        return max(0.0, min(1.0, entropy))

    def calculate_entropy_factor(self, entropy: float) -> float:
        """
        E(H) = 1.0                    if H < entropy_clean
               1.0 - 9*(H - 0.90)     if 0.90 <= H < 0.99
               0.1                     if H >= 0.99
        """
        if entropy < self.entropy_clean:
            return 1.0
        elif entropy >= 0.99:
            return 0.1
        else:
            return 1.0 - 9.0 * (entropy - self.entropy_clean)

    def calculate_probability(self, amplitude_squared: float, entropy: float,
                              interference: float, confidence: float) -> float:
        """P(trade) = |psi|^2 * E(entropy) * interference * confidence"""
        ef = self.calculate_entropy_factor(entropy)
        prob = amplitude_squared * ef * interference * confidence
        return max(0.0, min(1.0, prob))

    def record_trade(self, pnl: float):
        """Record a completed trade result for kill switch tracking."""
        self.trade_history.append(pnl > 0)

    def get_recent_win_rate(self) -> float:
        """Win rate over the last kill_switch_lookback trades."""
        if len(self.trade_history) == 0:
            return 1.0
        recent = list(self.trade_history)[-self.kill_switch_lookback:]
        if len(recent) == 0:
            return 1.0
        return sum(1 for w in recent if w) / len(recent)

    def is_kill_switch_triggered(self) -> bool:
        """Check if recent WR is below threshold."""
        if len(self.trade_history) < self.kill_switch_lookback:
            return False
        return self.get_recent_win_rate() < self.kill_switch_wr

    def should_trade(self, entropy: float, interference: float,
                     confidence: float, direction: int,
                     amplitude_squared: float = 0.5) -> bool:
        """
        Run all 6 gates in sequence. Returns True only if all pass.

        Signal --[G1]--[G2]--[G3]--[G4]--[G5]--[G6]--> EXECUTE
        """
        self.total_checks += 1

        # G1: Entropy Gate
        if entropy > self.entropy_clean:
            self.blocked_g1 += 1
            return False

        # G2: Interference Gate
        if interference < self.interference_min:
            self.blocked_g2 += 1
            return False

        # G3: Confidence Gate
        if confidence < self.confidence_min:
            self.blocked_g3 += 1
            return False

        # G4: Probability Gate
        probability = self.calculate_probability(
            amplitude_squared, entropy, interference, confidence
        )
        if probability < self.probability_min:
            self.blocked_g4 += 1
            return False

        # G5: Direction Gate (0 = BOTH allowed)
        if self.direction_bias != 0:
            if direction != self.direction_bias:
                self.blocked_g5 += 1
                return False

        # G6: Kill Switch
        if self.is_kill_switch_triggered():
            self.blocked_g6 += 1
            return False

        # ALL GATES PASSED
        self.passed += 1
        return True

    def get_stats(self) -> dict:
        """Return gate statistics."""
        return {
            "total_checks": self.total_checks,
            "passed": self.passed,
            "pass_rate": (self.passed / max(self.total_checks, 1)) * 100,
            "g1_entropy": self.blocked_g1,
            "g2_interference": self.blocked_g2,
            "g3_confidence": self.blocked_g3,
            "g4_probability": self.blocked_g4,
            "g5_direction": self.blocked_g5,
            "g6_killswitch": self.blocked_g6,
        }

    def reset_stats(self):
        """Reset gate counters (not trade history)."""
        self.total_checks = 0
        self.passed = 0
        self.blocked_g1 = 0
        self.blocked_g2 = 0
        self.blocked_g3 = 0
        self.blocked_g4 = 0
        self.blocked_g5 = 0
        self.blocked_g6 = 0


# ============================================================
# AOI BIOLOGICAL CASCADE - Simplified for simulation
# ============================================================
# Implements 4 key AOI algorithms without requiring external imports:
#   1. CRISPR Memory - track losing patterns, block repeats
#   2. Protective Deletion - skip after consecutive losses
#   3. Toxoplasma - regime mismatch detection
#   4. Syncytin - cross-symbol agreement boost
# ============================================================

class AOIBiologicalFilter:
    """
    Simplified AOI biological cascade for tournament simulation.

    This implements the 4 most impactful AOI algorithms as lightweight
    filters that operate during the challenge simulation loop.
    """

    def __init__(self):
        # CRISPR Memory: {(symbol, direction): deque of win/loss bools}
        self.crispr_memory: Dict[Tuple[str, int], deque] = defaultdict(
            lambda: deque(maxlen=20)
        )

        # Protective Deletion: consecutive loss counter per symbol
        self.consecutive_losses: Dict[str, int] = defaultdict(int)
        self.skip_next: Dict[str, bool] = defaultdict(bool)

        # Toxoplasma: baseline ATR ratio per symbol (set during warmup)
        self.baseline_atr_ratio: Dict[str, float] = {}

        # Syncytin: current signal directions across symbols
        self.current_directions: Dict[str, int] = {}

        # Stats
        self.crispr_blocks = 0
        self.protective_blocks = 0
        self.toxoplasma_adjustments = 0
        self.syncytin_boosts = 0

    def set_baseline_atr(self, symbol: str, atr_ratio: float):
        """Set the baseline ATR ratio for a symbol (from first N bars)."""
        self.baseline_atr_ratio[symbol] = atr_ratio

    def update_direction(self, symbol: str, direction: int):
        """Update the current signal direction for a symbol (for Syncytin)."""
        self.current_directions[symbol] = direction

    def record_trade_result(self, symbol: str, direction: int, won: bool):
        """Record a trade result for CRISPR memory and protective deletion."""
        key = (symbol, direction)
        self.crispr_memory[key].append(won)

        if won:
            self.consecutive_losses[symbol] = 0
            self.skip_next[symbol] = False
        else:
            self.consecutive_losses[symbol] += 1
            # Protective Deletion: 3+ consecutive losses triggers skip
            if self.consecutive_losses[symbol] >= 3:
                self.skip_next[symbol] = True

    def should_trade(self, symbol: str, direction: int,
                     confidence: float, current_atr_ratio: float) -> Tuple[bool, float]:
        """
        Run the AOI biological cascade on a signal.

        Returns:
            (should_trade: bool, modified_confidence: float)
        """
        modified_confidence = confidence

        # === 1. CRISPR Memory Gate ===
        # Block patterns with WR < 40% over last 20 trades
        key = (symbol, direction)
        history = self.crispr_memory[key]
        if len(history) >= 5:  # Need at least 5 trades to judge
            wr = sum(1 for w in history if w) / len(history)
            if wr < 0.40:
                self.crispr_blocks += 1
                return False, 0.0

        # === 2. Protective Deletion ===
        # If 3+ consecutive losses on this symbol, skip next signal
        if self.skip_next.get(symbol, False):
            self.protective_blocks += 1
            self.skip_next[symbol] = False  # Reset after one skip
            return False, 0.0

        # === 3. Toxoplasma - Regime mismatch detection ===
        # If current ATR ratio differs significantly from baseline, reduce confidence
        baseline = self.baseline_atr_ratio.get(symbol, current_atr_ratio)
        if baseline > 0:
            ratio_change = abs(current_atr_ratio - baseline) / (baseline + 1e-10)
            if ratio_change > 0.5:  # 50%+ regime shift
                modified_confidence *= 0.70  # Reduce by 30%
                self.toxoplasma_adjustments += 1

        # === 4. Syncytin - Cross-symbol agreement boost ===
        # If 2+ other symbols agree on direction, boost confidence
        agreeing = sum(
            1 for sym, d in self.current_directions.items()
            if sym != symbol and d == direction
        )
        if agreeing >= 2:
            modified_confidence *= 1.20  # Boost by 20%
            modified_confidence = min(1.0, modified_confidence)
            self.syncytin_boosts += 1

        return True, modified_confidence

    def get_stats(self) -> dict:
        """Return filter statistics."""
        return {
            "crispr_blocks": self.crispr_blocks,
            "protective_blocks": self.protective_blocks,
            "toxoplasma_adjustments": self.toxoplasma_adjustments,
            "syncytin_boosts": self.syncytin_boosts,
            "memory_patterns": len(self.crispr_memory),
        }


# ============================================================
# CHALLENGER DNA - Mutable trading parameters (CHAOS ranges)
# ============================================================

@dataclass
class ChallengerDNA:
    """
    Each challenger has unique mutated trading parameters.
    CHAOS version: wider ranges than prop_challenge_1000.py.
    """
    challenger_id: int
    group: str = GROUP_MAIN

    # Signal parameters (WIDER ranges)
    fast_ema: int = 5
    slow_ema: int = 13
    atr_period: int = 14
    confidence_threshold: float = 0.22

    # Risk parameters (WIDER ranges)
    sl_atr_mult: float = 1.0
    tp_multiplier: float = 3.0
    dyn_tp_percent: float = 50.0
    rolling_sl_mult: float = 1.5

    # Position sizing (WIDER ranges)
    lot_min: float = 0.01
    lot_max: float = 0.24
    max_positions: int = 20
    grid_spacing: int = 200

    # Confidence/DD lot weighting
    conf_weight: float = 0.7
    dd_weight: float = 0.3

    # Additional signal filters
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    use_rsi_filter: bool = False
    use_volume_filter: bool = False
    volume_ma_mult: float = 1.5

    # Metadata
    generation: int = 0
    parent_ids: list = field(default_factory=list)
    cumulative_score: float = 0.0
    rounds_survived: int = 0

    def mutate(self, strength: float = 0.4):
        """
        Mutate parameters with Gaussian noise.
        CHAOS version: strength=0.4 default, wider ranges.
        """
        def _m_int(val, low, high):
            new = val + int(random.gauss(0, max(1, (high - low) * strength)))
            return max(low, min(high, new))

        def _m_float(val, low, high):
            new = val + random.gauss(0, (high - low) * strength)
            return max(low, min(high, round(new, 4)))

        # CHAOS ranges (wider than prop_challenge_1000.py)
        self.fast_ema = _m_int(self.fast_ema, 2, 30)
        self.slow_ema = _m_int(self.slow_ema, 5, 80)
        if self.slow_ema <= self.fast_ema:
            self.slow_ema = self.fast_ema + 3
        self.atr_period = _m_int(self.atr_period, 3, 40)
        self.confidence_threshold = _m_float(self.confidence_threshold, 0.05, 0.60)
        self.sl_atr_mult = _m_float(self.sl_atr_mult, 0.2, 5.0)
        self.tp_multiplier = _m_float(self.tp_multiplier, 1.0, 8.0)
        self.dyn_tp_percent = _m_float(self.dyn_tp_percent, 15.0, 85.0)
        self.rolling_sl_mult = _m_float(self.rolling_sl_mult, 0.8, 4.0)
        self.max_positions = _m_int(self.max_positions, 3, 60)
        self.grid_spacing = _m_int(self.grid_spacing, 20, 800)
        self.conf_weight = _m_float(self.conf_weight, 0.2, 0.95)
        self.dd_weight = round(1.0 - self.conf_weight, 4)
        self.rsi_overbought = _m_float(self.rsi_overbought, 55.0, 90.0)
        self.rsi_oversold = _m_float(self.rsi_oversold, 10.0, 45.0)
        self.use_rsi_filter = random.random() < 0.35
        self.use_volume_filter = random.random() < 0.25
        self.volume_ma_mult = _m_float(self.volume_ma_mult, 0.8, 3.5)
        self.lot_max = _m_float(self.lot_max, 0.01, 0.50)
        if self.lot_max < self.lot_min:
            self.lot_max = self.lot_min

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def crossover_dna(parent1: ChallengerDNA, parent2: ChallengerDNA,
                  child_id: int, group: str) -> ChallengerDNA:
    """Breed two challengers - uniform crossover on each parameter."""
    child = ChallengerDNA(challenger_id=child_id, group=group)
    for attr in ['fast_ema', 'slow_ema', 'atr_period', 'confidence_threshold',
                 'sl_atr_mult', 'tp_multiplier', 'dyn_tp_percent', 'rolling_sl_mult',
                 'max_positions', 'grid_spacing', 'conf_weight',
                 'rsi_overbought', 'rsi_oversold', 'use_rsi_filter',
                 'use_volume_filter', 'volume_ma_mult', 'lot_max']:
        val = getattr(parent1, attr) if random.random() < 0.5 else getattr(parent2, attr)
        setattr(child, attr, val)
    child.dd_weight = round(1.0 - child.conf_weight, 4)
    child.parent_ids = [parent1.challenger_id, parent2.challenger_id]
    if child.slow_ema <= child.fast_ema:
        child.slow_ema = child.fast_ema + 3
    if child.lot_max < child.lot_min:
        child.lot_max = child.lot_min
    return child


# ============================================================
# CHALLENGE RESULT STRUCTURES
# ============================================================

@dataclass
class SymbolResult:
    """Per-symbol performance metrics."""
    symbol: str
    balance: float
    net_profit: float
    return_pct: float
    win_rate: float
    total_trades: int
    winners: int
    losers: int
    profit_factor: float
    max_dd_pct: float
    challenge_passed: bool
    challenge_failed: bool
    fail_reason: str
    days_traded: int
    signals_collected: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    avg_atr: float = 0.0
    regime_tag: str = ""


@dataclass
class ChallengeResult:
    challenger_id: int
    dna: dict
    group: str
    # Aggregate across all symbols
    balance: float
    net_profit: float
    return_pct: float
    win_rate: float
    total_trades: int
    winners: int
    losers: int
    profit_factor: float
    max_dd_pct: float
    challenge_passed: bool
    challenge_failed: bool
    fail_reason: str
    days_traded: int
    # Scoring
    score: float = 0.0
    # Signals
    signals_collected: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    # Multi-symbol metrics
    per_symbol: dict = field(default_factory=dict)
    best_symbol: str = ""
    worst_symbol: str = ""
    symbol_consistency: float = 0.0
    symbols_profitable: int = 0
    symbols_total: int = 0
    # Filter stats (for control groups)
    filter_stats: dict = field(default_factory=dict)


# ============================================================
# SINGLE-SYMBOL CHALLENGE RUNNER
# ============================================================

def run_single_symbol_challenge(
    dna: ChallengerDNA,
    df: pd.DataFrame,
    point: float = 0.01,
    contract_size: float = 1.0,
    jardine_gate: JardinesGate = None,
    aoi_filter: AOIBiologicalFilter = None,
    symbol: str = "",
    all_close_arrays: dict = None,
) -> Tuple[dict, list]:
    """
    Run one prop firm challenge for a single challenger on one symbol.

    Returns:
        (result_dict, closed_trades_list)
        where closed_trades_list contains (pnl, direction) tuples
    """
    INITIAL_BALANCE = 100000.0
    MAX_DAILY_DD = 5000.0    # 5%
    MAX_TOTAL_DD = 10000.0   # 10%
    PROFIT_TARGET = 10000.0  # 10%

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    tick_vol = df['tick_volume'].values if 'tick_volume' in df.columns else np.ones(len(df))

    n = len(close)
    if n < 100:
        return _empty_result(dna, symbol), []

    # Pre-compute indicators
    ema_fast = pd.Series(close).ewm(span=max(dna.fast_ema, 2), adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=max(dna.slow_ema, 3), adjust=False).mean().values

    # ATR
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(max(dna.atr_period, 2)).mean().values

    # RSI
    rsi = np.full(n, 50.0)
    if dna.use_rsi_filter:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss_arr = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss_arr).rolling(14).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

    # Volume MA
    vol_ma = pd.Series(tick_vol).rolling(20).mean().values

    # Baseline ATR ratio for Toxoplasma (first 500 bars)
    if aoi_filter is not None:
        warmup_end = min(500, n)
        warmup_atr = np.nanmean(atr[:warmup_end])
        warmup_price = np.nanmean(close[:warmup_end])
        baseline_atr_ratio = warmup_atr / (warmup_price + 1e-10)
        aoi_filter.set_baseline_atr(symbol, baseline_atr_ratio)

    # Simulation state
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    max_dd = 0.0
    max_dd_pct = 0.0
    positions = []
    closed = []
    challenge_passed = False
    challenge_failed = False
    fail_reason = ""
    current_day = -1
    day_start_eq = INITIAL_BALANCE
    days_traded = 0
    buy_signals = 0
    sell_signals = 0

    start_bar = max(dna.slow_ema, dna.atr_period, 26) + 5

    for i in range(start_bar, n):
        price = close[i]
        a = atr[i]
        if np.isnan(a) or a == 0:
            continue

        # Day tracking
        day_approx = i // 288
        if day_approx != current_day:
            if current_day >= 0:
                days_traded += 1
            current_day = day_approx
            floating = sum(
                (price - p[1] if p[0] == 1 else p[1] - price) * p[8] * contract_size
                for p in positions
            )
            day_start_eq = balance + floating

        # Equity
        floating = sum(
            (price - p[1] if p[0] == 1 else p[1] - price) * p[8] * contract_size
            for p in positions
        )
        equity = balance + floating

        # DD checks
        if day_start_eq - equity >= MAX_DAILY_DD:
            if not challenge_failed:
                challenge_failed = True
                fail_reason = "Daily DD"
            continue

        total_dd = INITIAL_BALANCE - equity
        if total_dd >= MAX_TOTAL_DD:
            if not challenge_failed:
                challenge_failed = True
                fail_reason = "Total DD"
            break

        if equity - INITIAL_BALANCE >= PROFIT_TARGET and not challenge_passed:
            challenge_passed = True

        dd_remaining_pct = max(0, (MAX_TOTAL_DD - max(0, total_dd)) / MAX_TOTAL_DD)

        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_p = (dd / peak) * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_p > max_dd_pct:
            max_dd_pct = dd_p

        # Manage positions
        new_positions = []
        for pos in positions:
            direction, entry, lot, sl, tp, dyn_tp, dyn_taken, rsl, rem_lot, ppnl = pos

            # SL hit
            if (direction == 1 and price <= rsl) or (direction == -1 and price >= rsl):
                pnl = (rsl - entry if direction == 1 else entry - rsl) * rem_lot * contract_size
                balance += pnl
                closed.append((pnl + ppnl, direction))
                # Record for filters
                won = (pnl + ppnl) > 0
                if jardine_gate is not None:
                    jardine_gate.record_trade(pnl + ppnl)
                if aoi_filter is not None:
                    aoi_filter.record_trade_result(symbol, direction, won)
                continue

            # TP hit
            if (direction == 1 and price >= tp) or (direction == -1 and price <= tp):
                pnl = (tp - entry if direction == 1 else entry - tp) * rem_lot * contract_size
                balance += pnl
                closed.append((pnl + ppnl, direction))
                won = (pnl + ppnl) > 0
                if jardine_gate is not None:
                    jardine_gate.record_trade(pnl + ppnl)
                if aoi_filter is not None:
                    aoi_filter.record_trade_result(symbol, direction, won)
                continue

            # Dynamic TP
            if not dyn_taken:
                if (direction == 1 and price >= dyn_tp) or (direction == -1 and price <= dyn_tp):
                    half = rem_lot * 0.5
                    partial = (price - entry if direction == 1 else entry - price) * half * contract_size
                    balance += partial
                    ppnl += partial
                    rem_lot -= half
                    dyn_taken = True
                    rsl = entry  # Breakeven

            # Rolling SL
            if dyn_taken:
                sl_dist = abs(entry - sl)
                target = sl_dist * dna.rolling_sl_mult
                profit = (price - entry) if direction == 1 else (entry - price)
                if profit > target:
                    new_sl = (price - sl_dist) if direction == 1 else (price + sl_dist)
                    if direction == 1 and new_sl > rsl:
                        rsl = new_sl
                    elif direction == -1 and (new_sl < rsl or rsl <= 0):
                        rsl = new_sl

            new_positions.append((direction, entry, lot, sl, tp, dyn_tp, dyn_taken, rsl, rem_lot, ppnl))

        positions = new_positions

        if challenge_failed:
            continue

        # Signal generation
        ef = ema_fast[i]
        es = ema_slow[i]
        separation = abs(ef - es)
        confidence = min(1.0, (separation / (a + 1e-10)) * 0.5)

        buy_sig = ef > es
        sell_sig = ef < es

        # RSI filter
        if dna.use_rsi_filter:
            r = rsi[i]
            if buy_sig and r > dna.rsi_overbought:
                buy_sig = False
            if sell_sig and r < dna.rsi_oversold:
                sell_sig = False

        # Volume filter
        if dna.use_volume_filter and not np.isnan(vol_ma[i]):
            if tick_vol[i] < vol_ma[i] * dna.volume_ma_mult:
                buy_sig = False
                sell_sig = False

        if buy_sig:
            buy_signals += 1
        if sell_sig:
            sell_signals += 1

        if confidence < dna.confidence_threshold:
            continue

        # Determine direction
        direction = 1 if buy_sig else (-1 if sell_sig else 0)
        if direction == 0:
            continue

        # ============================================================
        # CONTROL GROUP FILTERS
        # ============================================================

        # --- JARDINE'S GATE FILTER ---
        if jardine_gate is not None:
            # Calculate entropy from recent prices for G1
            lookback = min(100, i)
            entropy = jardine_gate.calculate_entropy_from_prices(close[i - lookback:i + 1])

            # Interference: agreement between fast and slow EMA direction
            # Both trending same way = high interference
            fast_dir = 1 if ema_fast[i] > ema_fast[max(0, i - 5)] else -1
            slow_dir = 1 if ema_slow[i] > ema_slow[max(0, i - 10)] else -1
            interference = 1.0 if fast_dir == slow_dir else 0.3

            if not jardine_gate.should_trade(entropy, interference, confidence, direction):
                continue

        # --- AOI BIOLOGICAL FILTER ---
        if aoi_filter is not None:
            # Update Syncytin direction tracking
            aoi_filter.update_direction(symbol, direction)

            # Calculate current ATR ratio for Toxoplasma
            current_atr_ratio = a / (price + 1e-10)

            should, modified_conf = aoi_filter.should_trade(
                symbol, direction, confidence, current_atr_ratio
            )
            if not should:
                continue
            confidence = modified_conf

        # Dynamic lot sizing
        cf = min(1.0, max(0.0, (confidence - dna.confidence_threshold) /
                          (1.0 - dna.confidence_threshold + 1e-10)))
        if dd_remaining_pct < 0.3:
            df_lot = 0.0
        elif dd_remaining_pct < 0.5:
            df_lot = (dd_remaining_pct - 0.3) / 0.2
        else:
            df_lot = 1.0
        combined = cf * dna.conf_weight + df_lot * dna.dd_weight
        lot = dna.lot_min + (dna.lot_max - dna.lot_min) * combined
        lot = max(dna.lot_min, min(dna.lot_max, round(lot, 2)))

        buy_count = sum(1 for p in positions if p[0] == 1)
        sell_count = sum(1 for p in positions if p[0] == -1)
        if buy_count + sell_count >= dna.max_positions:
            continue

        spacing = dna.grid_spacing * point

        # Entry
        if buy_sig and buy_count < dna.max_positions // 2:
            last_buy = max([p[1] for p in positions if p[0] == 1], default=0)
            if buy_count == 0 or price < last_buy - spacing:
                sd = a * dna.sl_atr_mult
                td = sd * dna.tp_multiplier
                dd_d = td * (dna.dyn_tp_percent / 100.0)
                pos = (1, price, lot, price - sd, price + td, price + dd_d, False, price - sd, lot, 0.0)
                positions.append(pos)

        if sell_sig and sell_count < dna.max_positions // 2:
            last_sell = min([p[1] for p in positions if p[0] == -1], default=1e18)
            if sell_count == 0 or price > last_sell + spacing:
                sd = a * dna.sl_atr_mult
                td = sd * dna.tp_multiplier
                dd_d = td * (dna.dyn_tp_percent / 100.0)
                pos = (-1, price, lot, price + sd, price - td, price - dd_d, False, price + sd, lot, 0.0)
                positions.append(pos)

    # Close remaining positions
    if n > 0:
        last_price = close[-1]
        for pos in positions:
            direction, entry, lot, sl, tp, dyn_tp, dyn_taken, rsl, rem_lot, ppnl = pos
            pnl = (last_price - entry if direction == 1 else entry - last_price) * rem_lot * contract_size
            closed.append((pnl + ppnl, direction))
            balance += pnl
            # Record for filters
            won = (pnl + ppnl) > 0
            if jardine_gate is not None:
                jardine_gate.record_trade(pnl + ppnl)
            if aoi_filter is not None:
                aoi_filter.record_trade_result(symbol, direction, won)

    days_traded += 1

    # Stats
    total_trades = len(closed)
    winners_list = [c for c in closed if c[0] > 0]
    losers_list = [c for c in closed if c[0] <= 0]
    win_count = len(winners_list)
    loss_count = len(losers_list)
    wr = (win_count / total_trades * 100) if total_trades > 0 else 0
    gp = sum(c[0] for c in winners_list)
    gl = abs(sum(c[0] for c in losers_list))
    pf = (gp / gl) if gl > 0 else 0.0
    net = balance - INITIAL_BALANCE

    return {
        "symbol": symbol,
        "balance": round(balance, 2),
        "net_profit": round(net, 2),
        "return_pct": round(net / INITIAL_BALANCE * 100, 2),
        "win_rate": round(wr, 1),
        "total_trades": total_trades,
        "winners": win_count,
        "losers": loss_count,
        "profit_factor": round(pf, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "challenge_passed": challenge_passed,
        "challenge_failed": challenge_failed,
        "fail_reason": fail_reason,
        "days_traded": days_traded,
        "signals_collected": buy_signals + sell_signals,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
    }, closed


def _empty_result(dna, symbol):
    """Return an empty result for symbols with insufficient data."""
    return {
        "symbol": symbol,
        "balance": 100000.0,
        "net_profit": 0.0,
        "return_pct": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "winners": 0,
        "losers": 0,
        "profit_factor": 0.0,
        "max_dd_pct": 0.0,
        "challenge_passed": False,
        "challenge_failed": False,
        "fail_reason": "insufficient_data",
        "days_traded": 0,
        "signals_collected": 0,
        "buy_signals": 0,
        "sell_signals": 0,
    }


# ============================================================
# MULTI-SYMBOL CHALLENGE ORCHESTRATOR
# ============================================================

def run_multi_symbol_challenge(
    dna: ChallengerDNA,
    symbol_data: dict,
    symbol_info: dict,
) -> ChallengeResult:
    """
    Run one challenger across ALL symbols. Aggregates results.
    Creates the appropriate filter based on group membership.
    """
    # Create filter based on group
    jardine_gate = None
    aoi_filter = None

    if dna.group == GROUP_JARDINE:
        jardine_gate = JardinesGate(
            entropy_clean=0.90,
            confidence_min=0.20,
            interference_min=0.50,
            probability_min=0.60,
            direction_bias=0,    # Both directions in chaos mode
            kill_switch_wr=0.50,
            kill_switch_lookback=10,
        )
    elif dna.group == GROUP_AOI:
        aoi_filter = AOIBiologicalFilter()

    per_symbol = {}
    total_trades = 0
    total_winners = 0
    total_losers = 0
    total_profit = 0.0
    total_signals = 0
    total_buy_sigs = 0
    total_sell_sigs = 0
    max_dd_all = 0.0
    any_passed = False
    any_failed = False
    fail_reasons = []

    for sym, df in symbol_data.items():
        pt, cs = symbol_info.get(sym, (0.01, 1.0))

        result_dict, _ = run_single_symbol_challenge(
            dna, df, pt, cs,
            jardine_gate=jardine_gate,
            aoi_filter=aoi_filter,
            symbol=sym,
        )

        # Regime tag
        close_vals = df['close'].values
        atr_vals = (df['high'] - df['low']).rolling(14).mean().values
        avg_atr = float(np.nanmean(atr_vals))
        atr_ratio = avg_atr / (np.nanmean(close_vals) + 1e-10)
        if atr_ratio < 0.005:
            regime = "clean"
        elif atr_ratio > 0.015:
            regime = "volatile"
        else:
            regime = "normal"

        sym_result = SymbolResult(
            symbol=sym,
            balance=result_dict["balance"],
            net_profit=result_dict["net_profit"],
            return_pct=result_dict["return_pct"],
            win_rate=result_dict["win_rate"],
            total_trades=result_dict["total_trades"],
            winners=result_dict["winners"],
            losers=result_dict["losers"],
            profit_factor=result_dict["profit_factor"],
            max_dd_pct=result_dict["max_dd_pct"],
            challenge_passed=result_dict["challenge_passed"],
            challenge_failed=result_dict["challenge_failed"],
            fail_reason=result_dict["fail_reason"],
            days_traded=result_dict["days_traded"],
            signals_collected=result_dict["signals_collected"],
            buy_signals=result_dict["buy_signals"],
            sell_signals=result_dict["sell_signals"],
            avg_atr=round(avg_atr, 6),
            regime_tag=regime,
        )
        per_symbol[sym] = sym_result

        total_trades += result_dict["total_trades"]
        total_winners += result_dict["winners"]
        total_losers += result_dict["losers"]
        total_profit += result_dict["net_profit"]
        total_signals += result_dict["signals_collected"]
        total_buy_sigs += result_dict["buy_signals"]
        total_sell_sigs += result_dict["sell_signals"]
        if result_dict["max_dd_pct"] > max_dd_all:
            max_dd_all = result_dict["max_dd_pct"]
        if result_dict["challenge_passed"]:
            any_passed = True
        if result_dict["challenge_failed"]:
            any_failed = True
            if result_dict["fail_reason"]:
                fail_reasons.append(f"{sym}: {result_dict['fail_reason']}")

    # Aggregate metrics
    agg_wr = (total_winners / total_trades * 100) if total_trades > 0 else 0

    sym_profits = [s.net_profit for s in per_symbol.values() if s.net_profit > 0]
    sym_losses = [abs(s.net_profit) for s in per_symbol.values() if s.net_profit < 0]
    agg_pf = (sum(sym_profits) / sum(sym_losses)) if sym_losses and sum(sym_losses) > 0 else 0.0

    sym_wrs = [s.win_rate for s in per_symbol.values() if s.total_trades > 0]
    consistency = float(np.std(sym_wrs)) if len(sym_wrs) > 1 else 0.0

    active_syms = {k: v for k, v in per_symbol.items() if v.total_trades > 0}
    best_sym = max(active_syms, key=lambda k: active_syms[k].win_rate) if active_syms else ""
    worst_sym = min(active_syms, key=lambda k: active_syms[k].win_rate) if active_syms else ""
    profitable_count = sum(1 for s in per_symbol.values() if s.net_profit > 0)

    # Composite score
    score = 0.0
    if total_trades >= 5:
        wr_score = min(agg_wr / 100, 1.0) * 25
        pf_score = min(agg_pf / 3.0, 1.0) * 20
        consist_score = max(0, 1.0 - consistency / 30) * 15
        profit_score = min(max(total_profit, 0) / 10000, 1.0) * 15
        dd_score = max(0, 1.0 - max_dd_all / 10) * 10
        multi_score = (profitable_count / max(len(per_symbol), 1)) * 10
        pass_score = 5.0 if any_passed else 0.0
        score = wr_score + pf_score + consist_score + profit_score + dd_score + multi_score + pass_score

    # Collect filter stats
    filter_stats = {}
    if jardine_gate is not None:
        filter_stats = jardine_gate.get_stats()
    elif aoi_filter is not None:
        filter_stats = aoi_filter.get_stats()

    return ChallengeResult(
        challenger_id=dna.challenger_id,
        dna=dna.to_dict(),
        group=dna.group,
        balance=round(100000 + total_profit, 2),
        net_profit=round(total_profit, 2),
        return_pct=round(total_profit / 100000 * 100, 2),
        win_rate=round(agg_wr, 1),
        total_trades=total_trades,
        winners=total_winners,
        losers=total_losers,
        profit_factor=round(agg_pf, 2),
        max_dd_pct=round(max_dd_all, 2),
        challenge_passed=any_passed,
        challenge_failed=any_failed,
        fail_reason="; ".join(fail_reasons[:5]) if fail_reasons else "",
        days_traded=max((s.days_traded for s in per_symbol.values()), default=0),
        score=round(score, 2),
        signals_collected=total_signals,
        buy_signals=total_buy_sigs,
        sell_signals=total_sell_sigs,
        per_symbol={k: v.__dict__ for k, v in per_symbol.items()},
        best_symbol=best_sym,
        worst_symbol=worst_sym,
        symbol_consistency=round(consistency, 2),
        symbols_profitable=profitable_count,
        symbols_total=len(per_symbol),
        filter_stats=filter_stats,
    )


# ============================================================
# DARK HORSE DETECTION
# ============================================================

def detect_dark_horses(results: List[ChallengeResult], symbol_data: dict,
                       top_n: int = 5) -> List[Tuple[str, int, float]]:
    """
    Identify "dark horse" symbols -- symbols that show up disproportionately
    as the best_symbol among top performers despite not being the most commonly
    traded or expected top symbols.

    Returns list of (symbol, appearances_in_winners, avg_wr_on_symbol).
    """
    # Count how many times each symbol appears as best_symbol in top 20%
    top_cutoff = max(10, len(results) // 5)
    top_results = results[:top_cutoff]

    best_sym_counts = defaultdict(int)
    for r in top_results:
        if r.best_symbol:
            best_sym_counts[r.best_symbol] += 1

    # Calculate average WR per symbol across ALL results (baseline expectation)
    sym_wr_all = defaultdict(list)
    for r in results:
        for sym, data in r.per_symbol.items():
            if isinstance(data, dict) and data.get('total_trades', 0) > 0:
                sym_wr_all[sym].append(data['win_rate'])

    # Baseline popularity: how many symbols are there?
    # "Dark horse" = appears in best_symbol more than expected
    expected_pct = 1.0 / max(len(symbol_data), 1) * top_cutoff

    dark_horses = []
    for sym, count in best_sym_counts.items():
        if count > expected_pct:  # More than random chance
            avg_wr = np.mean(sym_wr_all.get(sym, [0])) if sym in sym_wr_all else 0
            dark_horses.append((sym, count, round(avg_wr, 1)))

    dark_horses.sort(key=lambda x: x[1], reverse=True)
    return dark_horses[:top_n]


# ============================================================
# PER-SYMBOL LEADERBOARD
# ============================================================

def build_symbol_leaderboard(all_results: List[ChallengeResult]) -> List[dict]:
    """
    Build a comprehensive leaderboard ranking all symbols by performance.

    For each symbol, calculates:
      - Average WR across all challengers
      - Best individual WR achieved
      - Number of challengers who had it as best_symbol
      - Average PF
    """
    sym_stats = defaultdict(lambda: {
        "wrs": [], "pfs": [], "best_sym_count": 0, "total_challengers": 0,
    })

    for r in all_results:
        for sym, data in r.per_symbol.items():
            if isinstance(data, dict) and data.get('total_trades', 0) > 0:
                sym_stats[sym]["wrs"].append(data['win_rate'])
                sym_stats[sym]["pfs"].append(min(data['profit_factor'], 20))
                sym_stats[sym]["total_challengers"] += 1
        if r.best_symbol:
            sym_stats[r.best_symbol]["best_sym_count"] += 1

    leaderboard = []
    for sym, stats in sym_stats.items():
        if stats["wrs"]:
            leaderboard.append({
                "symbol": sym,
                "avg_wr": round(np.mean(stats["wrs"]), 1),
                "best_wr": round(max(stats["wrs"]), 1),
                "avg_pf": round(np.mean(stats["pfs"]), 2),
                "best_sym_count": stats["best_sym_count"],
                "total_challengers": stats["total_challengers"],
            })

    leaderboard.sort(key=lambda x: x["avg_wr"], reverse=True)
    return leaderboard


# ============================================================
# TOURNAMENT ENGINE
# ============================================================

def run_tournament(
    num_accounts: int = 2000,
    evo_rounds: int = 5,
    symbols: List[str] = None,
    days: int = 30,
    batch_size: int = 50,
    workers: int = 6,
    terminal_path: str = None,
):
    """
    Run the full CHAOS evolutionary tournament with control groups.

    2000 challengers split into MAIN / JARDINE / AOI groups.
    Each group evolves independently. Results compared at the end.
    """
    if symbols is None:
        symbols = ["BTCUSD", "ETHUSD", "XAUUSD"]

    # Calculate group sizes
    n_main = int(num_accounts * GROUP_RATIOS[GROUP_MAIN])
    n_jardine = int(num_accounts * GROUP_RATIOS[GROUP_JARDINE])
    n_aoi = num_accounts - n_main - n_jardine  # Remainder goes to AOI

    print("=" * 78)
    print("  PROP CHALLENGE CHAOS - EVOLUTIONARY TOURNAMENT WITH CONTROL GROUPS")
    print("  $100,000 simulated accounts. 130+ symbols. Maximum mutation.")
    print("=" * 78)
    print(f"  Total Challengers:  {num_accounts}")
    print(f"    MAIN group:       {n_main} (raw chaos, no filter)")
    print(f"    JARDINE group:    {n_jardine} (Jardine's Gate 6-gate filter)")
    print(f"    AOI group:        {n_aoi} (AOI biological cascade)")
    print(f"  Evo Rounds:         {evo_rounds}")
    print(f"  Symbols:            {len(symbols)} requested")
    print(f"  History:            {days} days")
    print(f"  GPU:                {GPU_NAME}")
    print(f"  Batch Size:         {batch_size}")
    print(f"  Workers:            {workers}")
    print(f"  Mutation Strength:  0.4 (MAXIMUM CHAOS)")
    if terminal_path:
        print(f"  Terminal:           {terminal_path}")
    print(f"  Config imports:     MAX_LOSS=${MAX_LOSS_DOLLARS} TP_MULT={TP_MULTIPLIER} "
          f"ROLL_SL={ROLLING_SL_MULTIPLIER} DYN_TP={DYNAMIC_TP_PERCENT}% "
          f"CONF={CONFIDENCE_THRESHOLD}")
    print("=" * 78)

    # ============================================================
    # INIT MT5 AND FETCH DATA
    # ============================================================
    if terminal_path:
        init_ok = mt5.initialize(path=terminal_path)
    else:
        init_ok = mt5.initialize()

    if not init_ok:
        print(f"ERROR: MT5 initialization failed: {mt5.last_error()}")
        return

    symbol_data = {}
    symbol_info_map = {}
    bars_needed = 288 * days
    skipped_symbols = []

    print(f"\n  Loading M5 data for {len(symbols)} symbols ({bars_needed:,} bars each)...")
    print(f"  {'Symbol':>12} {'Bars':>8} {'Start':>22} {'End':>22} {'Status':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*22} {'-'*22} {'-'*8}")

    for sym in symbols:
        sym = sym.strip()
        if not sym:
            continue

        try:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, min(bars_needed, 500000))
        except Exception as e:
            print(f"  {sym:>12} {'ERROR':>8} {str(e)[:44]:>44} {'SKIP':>8}")
            skipped_symbols.append(sym)
            continue

        if rates is None or len(rates) < 500:
            bar_count = len(rates) if rates is not None else 0
            print(f"  {sym:>12} {bar_count:>8} {'insufficient data':>44} {'SKIP':>8}")
            skipped_symbols.append(sym)
            continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        symbol_data[sym] = df

        si = mt5.symbol_info(sym)
        point = si.point if si else 0.01
        cs = si.trade_contract_size if si else 1.0
        symbol_info_map[sym] = (point, cs)

        start_t = str(df['time'].iloc[0])[:19]
        end_t = str(df['time'].iloc[-1])[:19]
        print(f"  {sym:>12} {len(df):>8,} {start_t:>22} {end_t:>22} {'OK':>8}")

    mt5.shutdown()

    if not symbol_data:
        print("\nERROR: No symbol data loaded. Check symbol names and MT5 connection.")
        return

    print(f"\n  Successfully loaded: {len(symbol_data)}/{len(symbols)} symbols")
    if skipped_symbols:
        print(f"  Skipped ({len(skipped_symbols)}): {', '.join(skipped_symbols[:20])}"
              f"{'...' if len(skipped_symbols) > 20 else ''}")

    # Output directory
    output_dir = SCRIPT_DIR / "signal_farm_output" / "challenge_chaos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # CREATE INITIAL POPULATION
    # ============================================================
    print(f"\n  Generating {num_accounts} challengers with CHAOS DNA (strength=0.4)...")

    population = []
    next_id = 1

    # MAIN group
    for i in range(n_main):
        dna = ChallengerDNA(challenger_id=next_id, group=GROUP_MAIN)
        dna.mutate(strength=0.4)
        population.append(dna)
        next_id += 1

    # JARDINE group
    for i in range(n_jardine):
        dna = ChallengerDNA(challenger_id=next_id, group=GROUP_JARDINE)
        dna.mutate(strength=0.4)
        population.append(dna)
        next_id += 1

    # AOI group
    for i in range(n_aoi):
        dna = ChallengerDNA(challenger_id=next_id, group=GROUP_AOI)
        dna.mutate(strength=0.4)
        population.append(dna)
        next_id += 1

    print(f"  Population ready: {n_main} MAIN + {n_jardine} JARDINE + {n_aoi} AOI")

    all_results_history = []
    all_round_results = []  # For final symbol leaderboard
    total_signals = 0
    tournament_start = time.time()

    # ============================================================
    # EVOLUTIONARY ROUNDS
    # ============================================================
    for evo_round in range(1, evo_rounds + 1):
        round_start = time.time()

        group_counts = defaultdict(int)
        for d in population:
            group_counts[d.group] += 1

        print(f"\n{'#' * 78}")
        print(f"# ROUND {evo_round}/{evo_rounds} - "
              f"{len(population)} CHALLENGERS x {len(symbol_data)} SYMBOLS")
        print(f"# Groups: MAIN={group_counts[GROUP_MAIN]} "
              f"JARDINE={group_counts[GROUP_JARDINE]} "
              f"AOI={group_counts[GROUP_AOI]}")
        print(f"{'#' * 78}")

        results: List[ChallengeResult] = []
        total_batches = (len(population) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start = batch_num * batch_size
            end = min(start + batch_size, len(population))
            batch_dna = population[start:end]
            batch_results = []

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        run_multi_symbol_challenge, dna, symbol_data, symbol_info_map
                    ): dna
                    for dna in batch_dna
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        dna = futures[future]
                        print(f"    ERROR: Challenger {dna.challenger_id} ({dna.group}): {e}")

            results.extend(batch_results)

            # Batch progress
            active = [r for r in batch_results if r.total_trades > 0]
            if active:
                avg_wr = sum(r.win_rate for r in active) / len(active)
                passed = sum(1 for r in active if r.challenge_passed)
                # Group breakdown for batch
                g_counts = defaultdict(int)
                for r in active:
                    g_counts[r.group] += 1
                grp_str = " ".join(f"{g}={c}" for g, c in sorted(g_counts.items()))
                print(f"    Batch {batch_num+1}/{total_batches}: "
                      f"{len(active)} active, WR={avg_wr:.1f}%, "
                      f"{passed} passed | {grp_str}")

        results.sort(key=lambda r: r.score, reverse=True)
        all_round_results.extend(results)

        active_results = [r for r in results if r.total_trades > 0]
        round_signals = sum(r.signals_collected for r in results)
        total_signals += round_signals
        passed_count = sum(1 for r in results if r.challenge_passed)
        failed_count = sum(1 for r in results if r.challenge_failed)
        round_elapsed = time.time() - round_start

        # ============================================================
        # PER-GROUP REPORTING
        # ============================================================
        print(f"\n  ROUND {evo_round} RESULTS ({round_elapsed:.1f}s):")
        print(f"  Total Active:  {len(active_results)}/{len(results)}")
        print(f"  Total Passed:  {passed_count}")
        print(f"  Total Failed:  {failed_count}")
        print(f"  Total Signals: {round_signals:,}")

        print(f"\n  {'='*74}")
        print(f"  PER-GROUP PERFORMANCE COMPARISON:")
        print(f"  {'='*74}")
        print(f"  {'Group':>8} {'Active':>7} {'AvgWR':>7} {'BestWR':>7} {'AvgPF':>7} "
              f"{'AvgScore':>9} {'Passed':>7} {'PassRate':>9} {'BestScore':>10}")
        print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} "
              f"{'-'*9} {'-'*7} {'-'*9} {'-'*10}")

        for group_name in [GROUP_MAIN, GROUP_JARDINE, GROUP_AOI]:
            group_results = [r for r in active_results if r.group == group_name]
            if not group_results:
                print(f"  {group_name:>8} {'(none)':>7}")
                continue

            g_wrs = [r.win_rate for r in group_results]
            g_pfs = [min(r.profit_factor, 20) for r in group_results]
            g_scores = [r.score for r in group_results]
            g_passed = sum(1 for r in group_results if r.challenge_passed)
            g_pass_rate = (g_passed / len(group_results)) * 100

            print(f"  {group_name:>8} {len(group_results):>7} "
                  f"{np.mean(g_wrs):>6.1f}% {max(g_wrs):>6.1f}% "
                  f"{np.mean(g_pfs):>7.2f} {np.mean(g_scores):>9.1f} "
                  f"{g_passed:>7} {g_pass_rate:>8.1f}% {max(g_scores):>10.1f}")

        # Filter-specific stats for control groups
        jardine_results = [r for r in active_results if r.group == GROUP_JARDINE and r.filter_stats]
        aoi_results = [r for r in active_results if r.group == GROUP_AOI and r.filter_stats]

        if jardine_results:
            avg_pass_rate = np.mean([r.filter_stats.get("pass_rate", 0) for r in jardine_results])
            avg_g1 = np.mean([r.filter_stats.get("g1_entropy", 0) for r in jardine_results])
            avg_g4 = np.mean([r.filter_stats.get("g4_probability", 0) for r in jardine_results])
            avg_g6 = np.mean([r.filter_stats.get("g6_killswitch", 0) for r in jardine_results])
            print(f"\n  JARDINE'S GATE STATS (avg across group):")
            print(f"    Gate pass rate:  {avg_pass_rate:.1f}%")
            print(f"    G1 blocks (entropy):     {avg_g1:.0f}")
            print(f"    G4 blocks (probability): {avg_g4:.0f}")
            print(f"    G6 blocks (kill switch): {avg_g6:.0f}")

        if aoi_results:
            avg_crispr = np.mean([r.filter_stats.get("crispr_blocks", 0) for r in aoi_results])
            avg_prot = np.mean([r.filter_stats.get("protective_blocks", 0) for r in aoi_results])
            avg_toxo = np.mean([r.filter_stats.get("toxoplasma_adjustments", 0) for r in aoi_results])
            avg_sync = np.mean([r.filter_stats.get("syncytin_boosts", 0) for r in aoi_results])
            print(f"\n  AOI BIOLOGICAL CASCADE STATS (avg across group):")
            print(f"    CRISPR blocks:           {avg_crispr:.0f}")
            print(f"    Protective deletions:    {avg_prot:.0f}")
            print(f"    Toxoplasma adjustments:  {avg_toxo:.0f}")
            print(f"    Syncytin boosts:         {avg_sync:.0f}")

        # ============================================================
        # DARK HORSE DETECTION
        # ============================================================
        dark_horses = detect_dark_horses(results, symbol_data)
        if dark_horses:
            print(f"\n  DARK HORSE SYMBOLS (unexpected top performers):")
            for rank, (sym, count, avg_wr) in enumerate(dark_horses, 1):
                print(f"    #{rank}: {sym} - appeared {count}x as best_symbol, avg WR={avg_wr}%")

        # Top 10 overall
        print(f"\n  TOP 10 OVERALL:")
        print(f"  {'#':>3} {'ID':>6} {'Group':>8} {'Score':>7} {'WR%':>6} {'PF':>6} "
              f"{'Trades':>7} {'Profit':>11} {'DD%':>6} {'Best':>8} {'Pass':>4}")
        print(f"  {'-'*3} {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*6} "
              f"{'-'*7} {'-'*11} {'-'*6} {'-'*8} {'-'*4}")
        for rank, r in enumerate(results[:10], 1):
            p = "YES" if r.challenge_passed else "no"
            bs = r.best_symbol[:8] if r.best_symbol else "-"
            print(f"  {rank:>3} {r.challenger_id:>6} {r.group:>8} {r.score:>7.1f} "
                  f"{r.win_rate:>5.1f}% {r.profit_factor:>6.2f} {r.total_trades:>7} "
                  f"${r.net_profit:>10,.2f} {r.max_dd_pct:>5.1f}% {bs:>8} {p:>4}")

        # Top 3 per group
        for group_name in [GROUP_MAIN, GROUP_JARDINE, GROUP_AOI]:
            group_sorted = [r for r in results if r.group == group_name]
            group_sorted.sort(key=lambda r: r.score, reverse=True)
            if group_sorted:
                top = group_sorted[0]
                print(f"\n  Best {group_name}: #{top.challenger_id} "
                      f"Score={top.score:.1f} WR={top.win_rate:.1f}% "
                      f"PF={top.profit_factor:.2f} Profit=${top.net_profit:,.2f} "
                      f"Best={top.best_symbol}")

        # Save round data
        all_results_history.append({
            "round": evo_round,
            "active": len(active_results),
            "passed": passed_count,
            "signals": round_signals,
            "elapsed_s": round(round_elapsed, 1),
            "group_stats": {
                group_name: {
                    "count": len([r for r in active_results if r.group == group_name]),
                    "avg_wr": round(np.mean([r.win_rate for r in active_results if r.group == group_name]) if any(r.group == group_name for r in active_results) else 0, 1),
                    "passed": sum(1 for r in results if r.group == group_name and r.challenge_passed),
                    "best_score": round(max((r.score for r in results if r.group == group_name), default=0), 1),
                }
                for group_name in [GROUP_MAIN, GROUP_JARDINE, GROUP_AOI]
            },
            "dark_horses": dark_horses,
            "top_10": [
                {
                    "id": r.challenger_id,
                    "group": r.group,
                    "score": r.score,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "net_profit": r.net_profit,
                    "trades": r.total_trades,
                    "passed": r.challenge_passed,
                    "best_symbol": r.best_symbol,
                    "dna": r.dna,
                }
                for r in results[:10]
            ],
        })

        # ============================================================
        # EVOLUTION (within each group separately)
        # ============================================================
        if evo_round < evo_rounds:
            result_map = {r.challenger_id: r for r in results}

            for dna in population:
                r = result_map.get(dna.challenger_id)
                if r:
                    dna.cumulative_score += r.score
                    dna.rounds_survived += 1

            new_population = []

            for group_name in [GROUP_MAIN, GROUP_JARDINE, GROUP_AOI]:
                group_pop = [d for d in population if d.group == group_name]
                if not group_pop:
                    continue

                # Sort by score
                group_pop.sort(
                    key=lambda d: result_map.get(d.challenger_id, ChallengeResult(
                        0, {}, group_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, "", 0
                    )).score,
                    reverse=True
                )

                group_size = len(group_pop)
                elite_count = max(5, group_size // 10)
                breed_count = max(10, group_size // 5)

                elite = group_pop[:elite_count]
                breeders = group_pop[:elite_count + breed_count]
                new_group = list(elite)

                while len(new_group) < group_size:
                    p1 = random.choice(breeders)
                    p2 = random.choice(breeders)
                    child = crossover_dna(p1, p2, next_id, group_name)
                    child.mutate(strength=0.4)  # MAXIMUM CHAOS every round
                    child.generation = evo_round
                    new_group.append(child)
                    next_id += 1

                new_population.extend(new_group)

            population = new_population

            g_counts = defaultdict(int)
            for d in population:
                g_counts[d.group] += 1
            print(f"\n  EVOLUTION: Elite survived within each group. "
                  f"New population: {' '.join(f'{g}={c}' for g, c in sorted(g_counts.items()))}")

    # ============================================================
    # FINAL RESULTS
    # ============================================================
    total_time = time.time() - tournament_start

    # Get final round results for winner
    final_results = all_round_results[-(len(population)):]  # Last round
    final_results.sort(key=lambda r: r.score, reverse=True)
    winner = final_results[0] if final_results else results[0]

    print(f"\n{'=' * 78}")
    print(f"  TOURNAMENT COMPLETE - CHAOS MODE")
    print(f"{'=' * 78}")
    print(f"  Total Time:       {total_time/60:.1f} minutes")
    print(f"  Total Signals:    {total_signals:,}")
    print(f"  Evo Rounds:       {evo_rounds}")
    print(f"  Challengers:      {num_accounts}")
    print(f"  Symbols Loaded:   {len(symbol_data)}")

    # ============================================================
    # FINAL GROUP COMPARISON
    # ============================================================
    print(f"\n  {'='*74}")
    print(f"  FINAL GROUP COMPARISON (last round):")
    print(f"  {'='*74}")

    for group_name in [GROUP_MAIN, GROUP_JARDINE, GROUP_AOI]:
        group_final = [r for r in final_results if r.group == group_name and r.total_trades > 0]
        if not group_final:
            continue

        g_wrs = [r.win_rate for r in group_final]
        g_pfs = [min(r.profit_factor, 20) for r in group_final]
        g_profits = [r.net_profit for r in group_final]
        g_scores = [r.score for r in group_final]
        g_passed = sum(1 for r in group_final if r.challenge_passed)
        g_dds = [r.max_dd_pct for r in group_final]

        print(f"\n  {group_name} GROUP ({len(group_final)} active):")
        print(f"    Avg WR:          {np.mean(g_wrs):.1f}%  (best: {max(g_wrs):.1f}%)")
        print(f"    Avg PF:          {np.mean(g_pfs):.2f}  (best: {max(g_pfs):.2f})")
        print(f"    Avg Profit:      ${np.mean(g_profits):,.2f}  (best: ${max(g_profits):,.2f})")
        print(f"    Avg Score:       {np.mean(g_scores):.1f}  (best: {max(g_scores):.1f})")
        print(f"    Avg DD:          {np.mean(g_dds):.1f}%  (worst: {max(g_dds):.1f}%)")
        print(f"    Pass Rate:       {g_passed}/{len(group_final)} "
              f"({g_passed/len(group_final)*100:.1f}%)")

    # ============================================================
    # WINNER
    # ============================================================
    print(f"\n  {'='*50}")
    print(f"  WINNER: Challenger #{winner.challenger_id} ({winner.group})")
    print(f"  {'='*50}")
    print(f"  Score:            {winner.score:.1f}")
    print(f"  Group:            {winner.group}")
    print(f"  Win Rate:         {winner.win_rate:.1f}%")
    print(f"  Profit Factor:    {winner.profit_factor:.2f}")
    print(f"  Net Profit:       ${winner.net_profit:,.2f}")
    print(f"  Total Trades:     {winner.total_trades}")
    print(f"  Max DD:           {winner.max_dd_pct:.1f}%")
    print(f"  Best Symbol:      {winner.best_symbol}")
    print(f"  Consistency:      {winner.symbol_consistency:.1f} (lower=better)")
    print(f"  Symbols Profit:   {winner.symbols_profitable}/{winner.symbols_total}")
    print(f"  Passed:           {'YES' if winner.challenge_passed else 'NO'}")

    if winner.filter_stats:
        print(f"\n  WINNER FILTER STATS:")
        for k, v in winner.filter_stats.items():
            print(f"    {k}: {v}")

    if winner.per_symbol:
        active_per_sym = {k: v for k, v in winner.per_symbol.items()
                         if isinstance(v, dict) and v.get('total_trades', 0) > 0}
        if active_per_sym:
            sorted_syms = sorted(active_per_sym.items(),
                                key=lambda x: x[1].get('win_rate', 0), reverse=True)
            print(f"\n  WINNER TOP SYMBOLS:")
            for sym, data in sorted_syms[:10]:
                print(f"    {sym}: WR={data.get('win_rate',0):.1f}% "
                      f"PF={data.get('profit_factor',0):.2f} "
                      f"Trades={data.get('total_trades',0)} "
                      f"Profit=${data.get('net_profit',0):,.2f} "
                      f"Regime={data.get('regime_tag','?')}")

    print(f"\n  WINNING DNA:")
    for k, v in winner.dna.items():
        if k not in ('challenger_id', 'parent_ids', 'cumulative_score',
                     'rounds_survived', 'generation', 'group'):
            print(f"    {k}: {v}")

    # ============================================================
    # PER-SYMBOL LEADERBOARD
    # ============================================================
    leaderboard = build_symbol_leaderboard(all_round_results)
    if leaderboard:
        print(f"\n  {'='*74}")
        print(f"  SYMBOL LEADERBOARD (ALL {len(leaderboard)} symbols, ranked by avg WR):")
        print(f"  {'='*74}")
        print(f"  {'#':>4} {'Symbol':>12} {'AvgWR':>7} {'BestWR':>7} {'AvgPF':>7} "
              f"{'BestSym#':>9} {'Challengers':>12}")
        print(f"  {'-'*4} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*12}")

        for rank, entry in enumerate(leaderboard, 1):
            print(f"  {rank:>4} {entry['symbol']:>12} {entry['avg_wr']:>6.1f}% "
                  f"{entry['best_wr']:>6.1f}% {entry['avg_pf']:>7.2f} "
                  f"{entry['best_sym_count']:>9} {entry['total_challengers']:>12}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    winner_path = output_dir / f"chaos_winner_{timestamp}.json"
    with open(winner_path, 'w') as f:
        json.dump({
            "winner": {
                "challenger_id": winner.challenger_id,
                "group": winner.group,
                "score": winner.score,
                "win_rate": winner.win_rate,
                "profit_factor": winner.profit_factor,
                "net_profit": winner.net_profit,
                "trades": winner.total_trades,
                "passed": winner.challenge_passed,
                "best_symbol": winner.best_symbol,
                "consistency": winner.symbol_consistency,
                "filter_stats": winner.filter_stats,
                "per_symbol": winner.per_symbol,
                "dna": winner.dna,
            },
            "tournament": {
                "accounts": num_accounts,
                "rounds": evo_rounds,
                "symbols_loaded": len(symbol_data),
                "symbols_requested": len(symbols),
                "total_signals": total_signals,
                "total_time_min": round(total_time / 60, 1),
                "mutation_strength": 0.4,
                "groups": {
                    GROUP_MAIN: n_main,
                    GROUP_JARDINE: n_jardine,
                    GROUP_AOI: n_aoi,
                },
            },
            "symbol_leaderboard": leaderboard[:50] if leaderboard else [],
            "history": all_results_history,
        }, f, indent=2, default=str)
    print(f"\n  Winner saved: {winner_path}")

    top100_path = output_dir / f"chaos_top100_{timestamp}.json"
    top100 = [
        {
            "rank": i + 1,
            "id": r.challenger_id,
            "group": r.group,
            "score": r.score,
            "win_rate": r.win_rate,
            "pf": r.profit_factor,
            "net_profit": r.net_profit,
            "best_sym": r.best_symbol,
            "consistency": r.symbol_consistency,
            "passed": r.challenge_passed,
            "filter_stats": r.filter_stats,
            "dna": r.dna,
        }
        for i, r in enumerate(final_results[:100])
    ]
    with open(top100_path, 'w') as f:
        json.dump(top100, f, indent=2, default=str)
    print(f"  Top 100 saved: {top100_path}")

    print(f"\n{'=' * 78}")
    print(f"  CHAOS TOURNAMENT FINISHED. THE FITTEST SURVIVE.")
    print(f"{'=' * 78}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prop Challenge CHAOS - 2000 Challengers, 3 Groups, 130+ Symbols"
    )
    parser.add_argument("--accounts", type=int, default=2000,
                        help="Total challengers (split 70/15/15 across groups)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Evolutionary rounds")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of M5 history to fetch")
    parser.add_argument("--symbols", type=str,
                        default="BTCUSD,ETHUSD,XAUUSD",
                        help="Comma-separated list of symbols (can be 130+)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for parallel execution (lower = less memory)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Thread pool workers")
    parser.add_argument("--terminal", type=str, default=None,
                        help="Path to terminal64.exe for specific broker")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 500 accounts, 3 rounds")

    args = parser.parse_args()

    if args.quick:
        args.accounts = 500
        args.rounds = 3

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    run_tournament(
        num_accounts=args.accounts,
        evo_rounds=args.rounds,
        symbols=symbols,
        days=args.days,
        batch_size=args.batch_size,
        workers=args.workers,
        terminal_path=args.terminal,
    )
