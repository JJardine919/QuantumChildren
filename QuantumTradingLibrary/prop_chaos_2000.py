"""
PROP CHAOS 2000 - Maximum Entropy Evolutionary Tournament
==========================================================
2000 challengers. ALL symbols. Random everything. Desperate survival mode.

Three groups competing:
  GROUP A (1400): Pure random DNA - EMA/ATR signal generation
  GROUP B (300):  Jardine's Gate control - signals filtered through quantum gates
  GROUP C (300):  AOI Pipeline control - signals filtered through 8 biological algorithms

Every challenger trades ALL available symbols simultaneously.
Maximum mutation strength. Maximum diversity. Survival of the desperate.

Usage:
    python prop_chaos_2000.py
    python prop_chaos_2000.py --accounts 2000 --rounds 5
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
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# GPU
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

# ============================================================
# AOI PIPELINE (control group B)
# ============================================================
AOI_AVAILABLE = False
aoi_pipeline_instance = None

try:
    from aoi_pipeline import AOIPipeline
    AOI_AVAILABLE = True
except ImportError:
    pass

# ============================================================
# CHALLENGER DNA
# ============================================================

@dataclass
class ChallengerDNA:
    """Each challenger has unique mutated trading parameters."""
    challenger_id: int
    group: str = "A"  # A=random, B=jardine, C=aoi
    # Signal parameters
    fast_ema: int = 5
    slow_ema: int = 13
    atr_period: int = 14
    confidence_threshold: float = 0.22
    # Risk parameters
    sl_atr_mult: float = 1.0
    tp_multiplier: float = 3.0
    dyn_tp_percent: float = 50.0
    rolling_sl_mult: float = 1.5
    # Position sizing
    lot_min: float = 0.01
    lot_max: float = 0.24
    max_positions: int = 20
    grid_spacing: int = 200
    # Confidence/DD lot weighting
    conf_weight: float = 0.7
    dd_weight: float = 0.3
    # Filters
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    use_rsi_filter: bool = False
    use_volume_filter: bool = False
    volume_ma_mult: float = 1.5
    # Jardine's Gate params (group B)
    jardine_entropy_weight: float = 0.5
    jardine_interference_decay: float = 0.85
    jardine_kill_threshold: float = 0.15
    # AOI params (group C)
    aoi_gate_strictness: float = 0.5
    aoi_te_boost: float = 1.3
    # Metadata
    generation: int = 0
    parent_ids: list = field(default_factory=list)
    cumulative_score: float = 0.0
    rounds_survived: int = 0

    def mutate(self, strength=0.3):
        """CHAOS mutation - extra strong for maximum diversity."""
        def _m_int(val, low, high):
            new = val + int(random.gauss(0, max(1, (high - low) * strength)))
            return max(low, min(high, new))

        def _m_float(val, low, high):
            new = val + random.gauss(0, (high - low) * strength)
            return max(low, min(high, round(new, 4)))

        self.fast_ema = _m_int(self.fast_ema, 2, 30)
        self.slow_ema = _m_int(self.slow_ema, 5, 80)
        if self.slow_ema <= self.fast_ema:
            self.slow_ema = self.fast_ema + 3
        self.atr_period = _m_int(self.atr_period, 3, 50)
        self.confidence_threshold = _m_float(self.confidence_threshold, 0.01, 0.60)
        self.sl_atr_mult = _m_float(self.sl_atr_mult, 0.1, 5.0)
        self.tp_multiplier = _m_float(self.tp_multiplier, 1.0, 8.0)
        self.dyn_tp_percent = _m_float(self.dyn_tp_percent, 10.0, 90.0)
        self.rolling_sl_mult = _m_float(self.rolling_sl_mult, 0.5, 4.0)
        self.max_positions = _m_int(self.max_positions, 1, 50)
        self.grid_spacing = _m_int(self.grid_spacing, 10, 1000)
        self.conf_weight = _m_float(self.conf_weight, 0.1, 0.95)
        self.dd_weight = round(1.0 - self.conf_weight, 4)
        self.lot_min = _m_float(self.lot_min, 0.01, 0.10)
        self.lot_max = _m_float(self.lot_max, 0.05, 1.0)
        if self.lot_max < self.lot_min:
            self.lot_max = self.lot_min + 0.05
        self.rsi_overbought = _m_float(self.rsi_overbought, 55.0, 90.0)
        self.rsi_oversold = _m_float(self.rsi_oversold, 10.0, 45.0)
        self.use_rsi_filter = random.random() < 0.4
        self.use_volume_filter = random.random() < 0.3
        self.volume_ma_mult = _m_float(self.volume_ma_mult, 0.5, 4.0)
        # Jardine params
        self.jardine_entropy_weight = _m_float(self.jardine_entropy_weight, 0.1, 0.9)
        self.jardine_interference_decay = _m_float(self.jardine_interference_decay, 0.5, 0.99)
        self.jardine_kill_threshold = _m_float(self.jardine_kill_threshold, 0.05, 0.40)
        # AOI params
        self.aoi_gate_strictness = _m_float(self.aoi_gate_strictness, 0.1, 0.9)
        self.aoi_te_boost = _m_float(self.aoi_te_boost, 0.5, 2.5)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def crossover_dna(parent1: ChallengerDNA, parent2: ChallengerDNA, child_id: int) -> ChallengerDNA:
    """Breed two challengers."""
    child = ChallengerDNA(challenger_id=child_id, group=parent1.group)
    for attr in ['fast_ema', 'slow_ema', 'atr_period', 'confidence_threshold',
                 'sl_atr_mult', 'tp_multiplier', 'dyn_tp_percent', 'rolling_sl_mult',
                 'max_positions', 'grid_spacing', 'conf_weight', 'lot_min', 'lot_max',
                 'rsi_overbought', 'rsi_oversold', 'use_rsi_filter',
                 'use_volume_filter', 'volume_ma_mult',
                 'jardine_entropy_weight', 'jardine_interference_decay',
                 'jardine_kill_threshold', 'aoi_gate_strictness', 'aoi_te_boost']:
        val = getattr(parent1, attr) if random.random() < 0.5 else getattr(parent2, attr)
        setattr(child, attr, val)
    child.dd_weight = round(1.0 - child.conf_weight, 4)
    child.parent_ids = [parent1.challenger_id, parent2.challenger_id]
    if child.slow_ema <= child.fast_ema:
        child.slow_ema = child.fast_ema + 3
    if child.lot_max < child.lot_min:
        child.lot_max = child.lot_min + 0.05
    return child


# ============================================================
# JARDINE'S GATE - Simplified Python Implementation
# ============================================================

def jardine_gate_filter(direction: int, confidence: float, atr: float,
                        price: float, ema_fast: float, ema_slow: float,
                        dna: ChallengerDNA) -> tuple:
    """
    Jardine's Gate quantum signal filter (Python approximation).
    6 gates: Entropy, Interference, Confidence, Probability, Direction, Kill-switch

    Returns: (filtered_direction, filtered_confidence, gate_pass)
    """
    # Gate 1: Entropy - signal decay based on noise
    noise = abs(price - ema_fast) / (atr + 1e-10)
    entropy = np.exp(-noise * dna.jardine_entropy_weight)
    conf = confidence * entropy

    # Gate 2: Interference - multi-path cancellation
    # If fast and slow EMAs are too close, signals interfere destructively
    separation = abs(ema_fast - ema_slow) / (atr + 1e-10)
    interference = 1.0 - dna.jardine_interference_decay ** (separation * 10)
    conf *= max(0.1, interference)

    # Gate 3: Confidence - subjective certainty threshold
    if conf < dna.confidence_threshold * 0.8:
        return (0, 0.0, False)

    # Gate 4: Probability collapse - P(trade) = |psi|^2
    psi_squared = conf ** 2
    if random.random() > psi_squared:
        return (0, 0.0, False)

    # Gate 5: Direction alignment - trend confirmation
    trend_aligned = (direction == 1 and ema_fast > ema_slow) or \
                    (direction == -1 and ema_fast < ema_slow)
    if not trend_aligned:
        conf *= 0.3  # Heavy penalty for counter-trend

    # Gate 6: Kill-switch - emergency halt
    if conf < dna.jardine_kill_threshold:
        return (0, 0.0, False)

    return (direction, round(conf, 4), True)


# ============================================================
# AOI FILTER - Simplified approximation for simulation speed
# ============================================================

def aoi_filter(direction: int, confidence: float, bars: np.ndarray,
               dna: ChallengerDNA, bar_idx: int) -> tuple:
    """
    Simplified AOI biological pipeline filter.
    Approximates the 8-algorithm cascade for simulation speed.

    Returns: (filtered_direction, filtered_confidence, gate_pass)
    """
    conf = confidence

    # 1. VDJ - Adaptive recombination: boost if recent bars show pattern diversity
    if bar_idx > 20:
        recent = bars[bar_idx-20:bar_idx]
        diversity = np.std(recent) / (np.mean(recent) + 1e-10)
        vdj_boost = min(1.3, 1.0 + diversity * 2)
        conf *= vdj_boost

    # 2. Protective Deletion - Suppress if recent pattern matches toxic profile
    if bar_idx > 10:
        recent_returns = np.diff(bars[bar_idx-10:bar_idx])
        consecutive_losses = 0
        for r in recent_returns:
            if (direction == 1 and r < 0) or (direction == -1 and r > 0):
                consecutive_losses += 1
        if consecutive_losses > 6:
            conf *= 0.3  # Toxic pattern suppression

    # 3. CRISPR - Immune memory: block if pattern fingerprint matches known loser
    if bar_idx > 30:
        fingerprint = np.mean(bars[bar_idx-5:bar_idx]) - np.mean(bars[bar_idx-30:bar_idx-5])
        if abs(fingerprint) < bars[bar_idx] * 0.0001:
            conf *= 0.5  # Flat market = known loser pattern

    # 4. Electric Organs - Cross-bar convergence (simplified)
    if bar_idx > 14:
        momentum = bars[bar_idx] - bars[bar_idx-14]
        if (direction == 1 and momentum > 0) or (direction == -1 and momentum < 0):
            conf *= 1.15  # Convergent signal boost

    # 5. KoRV - Signal domestication (gradual trust building)
    domestication = min(1.0, bar_idx / 2000) * dna.aoi_te_boost
    conf *= max(0.5, min(1.5, domestication))

    # 6. Toxoplasma - Regime hijack detection
    if bar_idx > 50:
        vol_recent = np.std(bars[bar_idx-10:bar_idx])
        vol_baseline = np.std(bars[bar_idx-50:bar_idx-10])
        if vol_recent > vol_baseline * 2.5:
            conf *= 0.4  # Regime hijack detected - reduce exposure

    # 7. Gate strictness
    if conf < dna.aoi_gate_strictness * dna.confidence_threshold:
        return (0, 0.0, False)

    return (direction, round(min(1.0, conf), 4), True)


# ============================================================
# CHALLENGE RUNNER
# ============================================================

@dataclass
class SymbolResult:
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
    group: str
    dna: dict
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
    score: float = 0.0
    signals_collected: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    per_symbol: dict = field(default_factory=dict)
    best_symbol: str = ""
    worst_symbol: str = ""
    symbol_consistency: float = 0.0
    symbols_profitable: int = 0
    symbols_total: int = 0


def run_challenge(dna: ChallengerDNA, df: pd.DataFrame, point: float = 0.01,
                  contract_size: float = 1.0) -> ChallengeResult:
    """Run one prop firm challenge with group-specific signal filtering."""
    INITIAL_BALANCE = 100000.0
    MAX_DAILY_DD = 5000.0
    MAX_TOTAL_DD = 10000.0
    PROFIT_TARGET = 10000.0

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    tick_vol = df['tick_volume'].values if 'tick_volume' in df.columns else np.ones(len(df))
    n = len(close)

    ema_fast = pd.Series(close).ewm(span=dna.fast_ema, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=dna.slow_ema, adjust=False).mean().values

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(dna.atr_period).mean().values

    rsi = np.full(n, 50.0)
    if dna.use_rsi_filter:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss_arr = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss_arr).rolling(14).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

    vol_ma = pd.Series(tick_vol).rolling(20).mean().values

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

        floating = sum(
            (price - p[1] if p[0] == 1 else p[1] - price) * p[8] * contract_size
            for p in positions
        )
        equity = balance + floating

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
            if (direction == 1 and price <= rsl) or (direction == -1 and price >= rsl):
                pnl = (rsl - entry if direction == 1 else entry - rsl) * rem_lot * contract_size
                balance += pnl
                closed.append((pnl + ppnl, direction))
                continue
            if (direction == 1 and price >= tp) or (direction == -1 and price <= tp):
                pnl = (tp - entry if direction == 1 else entry - tp) * rem_lot * contract_size
                balance += pnl
                closed.append((pnl + ppnl, direction))
                continue
            if not dyn_taken:
                if (direction == 1 and price >= dyn_tp) or (direction == -1 and price <= dyn_tp):
                    half = rem_lot * 0.5
                    partial = (price - entry if direction == 1 else entry - price) * half * contract_size
                    balance += partial
                    ppnl += partial
                    rem_lot -= half
                    dyn_taken = True
                    rsl = entry
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

        if dna.use_rsi_filter:
            r = rsi[i]
            if buy_sig and r > dna.rsi_overbought:
                buy_sig = False
            if sell_sig and r < dna.rsi_oversold:
                sell_sig = False

        if dna.use_volume_filter and not np.isnan(vol_ma[i]):
            if tick_vol[i] < vol_ma[i] * dna.volume_ma_mult:
                buy_sig = False
                sell_sig = False

        if buy_sig:
            buy_signals += 1
        if sell_sig:
            sell_signals += 1

        # ============================================================
        # GROUP-SPECIFIC SIGNAL FILTERING
        # ============================================================
        raw_direction = 1 if buy_sig else (-1 if sell_sig else 0)

        if raw_direction == 0 or confidence < dna.confidence_threshold:
            continue

        if dna.group == "B":
            # JARDINE'S GATE FILTER
            raw_direction, confidence, gate_pass = jardine_gate_filter(
                raw_direction, confidence, a, price, ef, es, dna
            )
            if not gate_pass:
                continue

        elif dna.group == "C":
            # AOI BIOLOGICAL PIPELINE FILTER
            raw_direction, confidence, gate_pass = aoi_filter(
                raw_direction, confidence, close, dna, i
            )
            if not gate_pass:
                continue

        buy_sig = raw_direction == 1
        sell_sig = raw_direction == -1

        if not buy_sig and not sell_sig:
            continue

        # Dynamic lot
        cf = min(1.0, max(0.0, (confidence - dna.confidence_threshold) / (1.0 - dna.confidence_threshold)))
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

        if buy_sig and buy_count < dna.max_positions // 2:
            last_buy = max([p[1] for p in positions if p[0] == 1], default=0)
            if buy_count == 0 or price < last_buy - spacing:
                sd = a * dna.sl_atr_mult
                td = sd * dna.tp_multiplier
                dd_d = td * (dna.dyn_tp_percent / 100.0)
                positions.append((1, price, lot, price - sd, price + td, price + dd_d, False, price - sd, lot, 0.0))

        if sell_sig and sell_count < dna.max_positions // 2:
            last_sell = min([p[1] for p in positions if p[0] == -1], default=1e9)
            if sell_count == 0 or price > last_sell + spacing:
                sd = a * dna.sl_atr_mult
                td = sd * dna.tp_multiplier
                dd_d = td * (dna.dyn_tp_percent / 100.0)
                positions.append((-1, price, lot, price + sd, price - td, price - dd_d, False, price + sd, lot, 0.0))

    # Close remaining
    if n > 0:
        last_price = close[-1]
        for pos in positions:
            direction, entry, lot, sl, tp, dyn_tp, dyn_taken, rsl, rem_lot, ppnl = pos
            pnl = (last_price - entry if direction == 1 else entry - last_price) * rem_lot * contract_size
            closed.append((pnl + ppnl, direction))
            balance += pnl

    days_traded += 1

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

    score = 0.0
    if total_trades >= 5:
        wr_score = min(wr / 100, 1.0) * 25
        pf_score = min(pf / 3.0, 1.0) * 20
        consist_score = 15  # placeholder for multi-symbol
        profit_score = min(max(net, 0) / PROFIT_TARGET, 1.0) * 15
        dd_score = max(0, 1.0 - max_dd_pct / 10) * 10
        pass_score = 5.0 if challenge_passed else 0.0
        score = wr_score + pf_score + profit_score + dd_score + pass_score

    return ChallengeResult(
        challenger_id=dna.challenger_id,
        group=dna.group,
        dna=dna.to_dict(),
        balance=round(balance, 2),
        net_profit=round(net, 2),
        return_pct=round(net / INITIAL_BALANCE * 100, 2),
        win_rate=round(wr, 1),
        total_trades=total_trades,
        winners=win_count,
        losers=loss_count,
        profit_factor=round(pf, 2),
        max_dd_pct=round(max_dd_pct, 2),
        challenge_passed=challenge_passed,
        challenge_failed=challenge_failed,
        fail_reason=fail_reason,
        days_traded=days_traded,
        score=round(score, 2),
        signals_collected=buy_signals + sell_signals,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
    )


def run_multi_symbol_challenge(dna, symbol_data, symbol_info):
    """Run one challenger across ALL symbols."""
    per_symbol = {}
    total_trades = 0
    total_winners = 0
    total_losers = 0
    total_profit = 0.0
    total_signals = 0
    total_buy = 0
    total_sell = 0
    max_dd_all = 0.0
    any_passed = False
    any_failed = False
    fail_reasons = []

    for symbol, df in symbol_data.items():
        pt, cs = symbol_info.get(symbol, (0.01, 1.0))
        result = run_challenge(dna, df, pt, cs)

        close_vals = df['close'].values
        atr_vals = (df['high'] - df['low']).rolling(14).mean().values
        avg_atr = float(np.nanmean(atr_vals))
        atr_ratio = avg_atr / (np.nanmean(close_vals) + 1e-10)
        regime = "clean" if atr_ratio < 0.005 else ("volatile" if atr_ratio > 0.015 else "normal")

        sym_result = SymbolResult(
            symbol=symbol, balance=result.balance, net_profit=result.net_profit,
            return_pct=result.return_pct, win_rate=result.win_rate,
            total_trades=result.total_trades, winners=result.winners,
            losers=result.losers, profit_factor=result.profit_factor,
            max_dd_pct=result.max_dd_pct, challenge_passed=result.challenge_passed,
            challenge_failed=result.challenge_failed, fail_reason=result.fail_reason,
            days_traded=result.days_traded, signals_collected=result.signals_collected,
            buy_signals=result.buy_signals, sell_signals=result.sell_signals,
            avg_atr=round(avg_atr, 2), regime_tag=regime,
        )
        per_symbol[symbol] = sym_result
        total_trades += result.total_trades
        total_winners += result.winners
        total_losers += result.losers
        total_profit += result.net_profit
        total_signals += result.signals_collected
        total_buy += result.buy_signals
        total_sell += result.sell_signals
        if result.max_dd_pct > max_dd_all:
            max_dd_all = result.max_dd_pct
        if result.challenge_passed:
            any_passed = True
        if result.challenge_failed:
            any_failed = True
            fail_reasons.append(f"{symbol}: {result.fail_reason}")

    agg_wr = (total_winners / total_trades * 100) if total_trades > 0 else 0
    sym_profits = [s.net_profit for s in per_symbol.values() if s.net_profit > 0]
    sym_losses = [abs(s.net_profit) for s in per_symbol.values() if s.net_profit < 0]
    agg_pf = (sum(sym_profits) / sum(sym_losses)) if sym_losses else 0.0

    sym_wrs = [s.win_rate for s in per_symbol.values() if s.total_trades > 0]
    consistency = float(np.std(sym_wrs)) if len(sym_wrs) > 1 else 0.0

    active_syms = {k: v for k, v in per_symbol.items() if v.total_trades > 0}
    best_sym = max(active_syms, key=lambda k: active_syms[k].win_rate) if active_syms else ""
    worst_sym = min(active_syms, key=lambda k: active_syms[k].win_rate) if active_syms else ""
    profitable_count = sum(1 for s in per_symbol.values() if s.net_profit > 0)

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

    return ChallengeResult(
        challenger_id=dna.challenger_id,
        group=dna.group,
        dna=dna.to_dict(),
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
        fail_reason="; ".join(fail_reasons) if fail_reasons else "",
        days_traded=max((s.days_traded for s in per_symbol.values()), default=0),
        score=round(score, 2),
        signals_collected=total_signals,
        buy_signals=total_buy,
        sell_signals=total_sell,
        per_symbol={k: v.__dict__ for k, v in per_symbol.items()},
        best_symbol=best_sym,
        worst_symbol=worst_sym,
        symbol_consistency=round(consistency, 2),
        symbols_profitable=profitable_count,
        symbols_total=len(per_symbol),
    )


# ============================================================
# TOURNAMENT
# ============================================================

def run_chaos_tournament(num_accounts=2000, evo_rounds=5, symbols=None,
                         days=30, batch_size=100, workers=8, terminal_path=None):
    """2000 challengers. ALL symbols. Three groups. Maximum chaos."""
    if symbols is None:
        symbols = ["XAUUSD"]

    # Group allocation: 70% random, 15% Jardine, 15% AOI
    n_a = int(num_accounts * 0.70)
    n_b = int(num_accounts * 0.15)
    n_c = num_accounts - n_a - n_b

    print("=" * 70)
    print("  PROP CHAOS 2000 - MAXIMUM ENTROPY TOURNAMENT")
    print("  DESPERATE SURVIVAL MODE - ALL SYMBOLS")
    print("=" * 70)
    print(f"  Total Challengers: {num_accounts}")
    print(f"    Group A (Random):   {n_a}")
    print(f"    Group B (Jardine):  {n_b}")
    print(f"    Group C (AOI Bio):  {n_c}")
    print(f"  Evo Rounds:    {evo_rounds}")
    print(f"  Symbols:       {len(symbols)} total")
    print(f"  History:       {days} days")
    print(f"  GPU:           {GPU_NAME}")
    print(f"  Workers:       {workers}")
    print("=" * 70)

    # Init MT5
    init_ok = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()
    if not init_ok:
        print("ERROR: MT5 init failed")
        return

    # Fetch data for ALL symbols
    symbol_data = {}
    symbol_info_map = {}
    bars_needed = 288 * days

    for symbol in symbols:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, min(bars_needed, 500000))
        if rates is None or len(rates) < 500:
            continue
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        symbol_data[symbol] = df
        si = mt5.symbol_info(symbol)
        point = si.point if si else 0.01
        cs = si.trade_contract_size if si else 1.0
        symbol_info_map[symbol] = (point, cs)

    mt5.shutdown()

    print(f"\n  Loaded {len(symbol_data)} symbols with data")
    if len(symbol_data) <= 20:
        print(f"  Symbols: {list(symbol_data.keys())}")
    else:
        print(f"  Sample: {list(symbol_data.keys())[:10]} ... +{len(symbol_data)-10} more")

    if not symbol_data:
        print("ERROR: No symbol data loaded")
        return

    output_dir = SCRIPT_DIR / "signal_farm_output" / "chaos_2000"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create population with maximum chaos diversity
    print(f"\n  Generating {num_accounts} DESPERATE challengers...")
    population = []
    cid = 1

    # Group A - Pure random
    for _ in range(n_a):
        dna = ChallengerDNA(challenger_id=cid, group="A")
        dna.mutate(strength=0.5)  # CHAOS mutation
        population.append(dna)
        cid += 1

    # Group B - Jardine's Gate
    for _ in range(n_b):
        dna = ChallengerDNA(challenger_id=cid, group="B")
        dna.mutate(strength=0.5)
        population.append(dna)
        cid += 1

    # Group C - AOI Bio
    for _ in range(n_c):
        dna = ChallengerDNA(challenger_id=cid, group="C")
        dna.mutate(strength=0.5)
        population.append(dna)
        cid += 1

    random.shuffle(population)  # Mix them up

    all_history = []
    total_signals = 0
    tournament_start = time.time()

    for evo_round in range(1, evo_rounds + 1):
        round_start = time.time()
        print(f"\n{'#' * 70}")
        print(f"# ROUND {evo_round}/{evo_rounds} - {len(population)} CHALLENGERS x {len(symbol_data)} SYMBOLS")
        print(f"{'#' * 70}")

        results = []
        total_batches = (len(population) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start = batch_num * batch_size
            end = min(start + batch_size, len(population))
            batch = population[start:end]

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_multi_symbol_challenge, dna,
                                   symbol_data, symbol_info_map): dna
                    for dna in batch
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        pass

            active = [r for r in results[start:] if r.total_trades > 0]
            if active:
                avg_wr = sum(r.win_rate for r in active) / len(active)
                passed = sum(1 for r in active if r.challenge_passed)
                print(f"    Batch {batch_num+1}/{total_batches}: "
                      f"{len(active)} active, avg WR={avg_wr:.1f}%, {passed} passed")

        results.sort(key=lambda r: r.score, reverse=True)
        active_results = [r for r in results if r.total_trades > 0]
        round_signals = sum(r.signals_collected for r in results)
        total_signals += round_signals
        round_elapsed = time.time() - round_start

        # Group breakdown
        grp_a = [r for r in active_results if r.group == "A"]
        grp_b = [r for r in active_results if r.group == "B"]
        grp_c = [r for r in active_results if r.group == "C"]

        print(f"\n  ROUND {evo_round} RESULTS ({round_elapsed:.1f}s):")
        print(f"  Active: {len(active_results)}/{len(results)}, Signals: {round_signals:,}")

        print(f"\n  GROUP COMPARISON:")
        print(f"  {'Group':>10} {'Count':>6} {'AvgWR':>7} {'BestWR':>7} {'AvgPF':>7} {'Passed':>7} {'AvgScore':>8}")
        print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
        for label, grp in [("A-Random", grp_a), ("B-Jardine", grp_b), ("C-AOI_Bio", grp_c)]:
            if grp:
                avg_wr = sum(r.win_rate for r in grp) / len(grp)
                best_wr = max(r.win_rate for r in grp)
                avg_pf = sum(min(r.profit_factor, 10) for r in grp) / len(grp)
                passed = sum(1 for r in grp if r.challenge_passed)
                avg_sc = sum(r.score for r in grp) / len(grp)
                print(f"  {label:>10} {len(grp):>6} {avg_wr:>6.1f}% {best_wr:>6.1f}% "
                      f"{avg_pf:>7.2f} {passed:>7} {avg_sc:>8.1f}")

        # Top 10
        print(f"\n  TOP 10 OVERALL:")
        print(f"  {'#':>3} {'ID':>5} {'Grp':>3} {'Score':>7} {'WR%':>6} {'PF':>6} {'Trades':>6} {'Profit':>10} {'DD%':>6} {'Best':>8} {'Pass':>4}")
        print(f"  {'-'*3} {'-'*5} {'-'*3} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*6} {'-'*8} {'-'*4}")
        for rank, r in enumerate(results[:10], 1):
            p = "YES" if r.challenge_passed else "no"
            bs = r.best_symbol[:8] if r.best_symbol else "-"
            print(f"  {rank:>3} {r.challenger_id:>5} {r.group:>3} {r.score:>7.1f} {r.win_rate:>5.1f}% "
                  f"{r.profit_factor:>6.2f} {r.total_trades:>6} ${r.net_profit:>9,.2f} "
                  f"{r.max_dd_pct:>5.1f}% {bs:>8} {p:>4}")

        # Save round
        all_history.append({
            "round": evo_round,
            "groups": {
                "A": {"count": len(grp_a), "avg_wr": round(sum(r.win_rate for r in grp_a)/max(len(grp_a),1), 1),
                       "passed": sum(1 for r in grp_a if r.challenge_passed)},
                "B": {"count": len(grp_b), "avg_wr": round(sum(r.win_rate for r in grp_b)/max(len(grp_b),1), 1),
                       "passed": sum(1 for r in grp_b if r.challenge_passed)},
                "C": {"count": len(grp_c), "avg_wr": round(sum(r.win_rate for r in grp_c)/max(len(grp_c),1), 1),
                       "passed": sum(1 for r in grp_c if r.challenge_passed)},
            },
            "top_10": [{"id": r.challenger_id, "group": r.group, "score": r.score,
                        "wr": r.win_rate, "pf": r.profit_factor, "profit": r.net_profit,
                        "best_sym": r.best_symbol} for r in results[:10]],
        })

        # EVOLUTION - groups breed within themselves
        if evo_round < evo_rounds:
            next_id = max(d.challenger_id for d in population) + 1
            new_pop = []

            for group_label in ["A", "B", "C"]:
                grp_dna = [d for d in population if d.group == group_label]
                grp_results = {r.challenger_id: r for r in results if r.group == group_label}

                grp_dna.sort(key=lambda d: grp_results.get(d.challenger_id, ChallengeResult(
                    0, group_label, {}, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, "", 0)).score, reverse=True)

                elite_count = max(5, len(grp_dna) // 10)
                breed_count = max(10, len(grp_dna) // 5)
                elite = grp_dna[:elite_count]
                breeders = grp_dna[:elite_count + breed_count]

                grp_new = list(elite)
                while len(grp_new) < len(grp_dna):
                    p1 = random.choice(breeders)
                    p2 = random.choice(breeders)
                    child = crossover_dna(p1, p2, next_id)
                    child.mutate(strength=0.15 + 0.1 * evo_round)
                    child.generation = evo_round
                    grp_new.append(child)
                    next_id += 1

                new_pop.extend(grp_new)

            random.shuffle(new_pop)
            population = new_pop
            print(f"\n  EVOLUTION: Groups bred independently, {len(population)} total")

    # FINAL RESULTS
    total_time = time.time() - tournament_start
    winner = results[0]

    print(f"\n{'=' * 70}")
    print(f"  CHAOS 2000 COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Time:          {total_time/60:.1f} minutes")
    print(f"  Signals:       {total_signals:,}")
    print(f"  Symbols:       {len(symbol_data)}")
    print(f"\n  FINAL GROUP RANKINGS:")

    for label, grp in [("A-Random", grp_a), ("B-Jardine", grp_b), ("C-AOI_Bio", grp_c)]:
        if grp:
            avg_wr = sum(r.win_rate for r in grp) / len(grp)
            avg_sc = sum(r.score for r in grp) / len(grp)
            passed = sum(1 for r in grp if r.challenge_passed)
            best = max(grp, key=lambda r: r.score)
            print(f"    {label}: AvgWR={avg_wr:.1f}%, AvgScore={avg_sc:.1f}, "
                  f"Passed={passed}, Best=#{best.challenger_id} ({best.score:.1f})")

    print(f"\n  WINNER: #{winner.challenger_id} (Group {winner.group})")
    print(f"  Score: {winner.score:.1f}, WR: {winner.win_rate:.1f}%, PF: {winner.profit_factor:.2f}")
    print(f"  Profit: ${winner.net_profit:,.2f}, DD: {winner.max_dd_pct:.1f}%")
    print(f"  Best Symbol: {winner.best_symbol}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"chaos_winner_{ts}.json", 'w') as f:
        json.dump({"winner": {"id": winner.challenger_id, "group": winner.group,
                              "score": winner.score, "wr": winner.win_rate,
                              "pf": winner.profit_factor, "profit": winner.net_profit,
                              "best_sym": winner.best_symbol, "per_symbol": winner.per_symbol,
                              "dna": winner.dna},
                   "history": all_history}, f, indent=2)
    print(f"  Saved: {output_dir / f'chaos_winner_{ts}.json'}")

    top100 = [{"rank": i+1, "id": r.challenger_id, "group": r.group,
               "score": r.score, "wr": r.win_rate, "pf": r.profit_factor,
               "profit": r.net_profit, "best_sym": r.best_symbol, "dna": r.dna}
              for i, r in enumerate(results[:100])]
    with open(output_dir / f"chaos_top100_{ts}.json", 'w') as f:
        json.dump(top100, f, indent=2)
    print(f"  Top 100: {output_dir / f'chaos_top100_{ts}.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prop Chaos 2000")
    parser.add_argument("--accounts", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols (default: ALL available)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--terminal", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.accounts = 500
        args.rounds = 3

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    # If no symbols specified, grab ALL tradeable
    if symbols is None:
        print("  Scanning ALL tradeable symbols...")
        init_ok = mt5.initialize(path=args.terminal) if args.terminal else mt5.initialize()
        if init_ok:
            all_syms = mt5.symbols_get()
            if all_syms:
                symbols = []
                for s in all_syms:
                    if s.trade_mode > 0:
                        rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_M5, 0, 10)
                        if rates is not None and len(rates) >= 5:
                            symbols.append(s.name)
                print(f"  Found {len(symbols)} tradeable symbols with data")
            mt5.shutdown()
        else:
            print("ERROR: MT5 init failed for symbol scan")
            sys.exit(1)

    run_chaos_tournament(
        num_accounts=args.accounts,
        evo_rounds=args.rounds,
        symbols=symbols,
        days=args.days,
        batch_size=args.batch_size,
        workers=args.workers,
        terminal_path=args.terminal,
    )
