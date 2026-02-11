"""
PROP CHALLENGE 1000 - Multi-Symbol Evolutionary Tournament
============================================================
1000 simulated $100K prop firm accounts compete on M5 data across
ALL available symbols (BTCUSD, ETHUSD, XAUUSD).

Each challenger trades ALL symbols simultaneously, collecting signals
and results per-symbol. This gives the EA the full market picture
so it learns cross-symbol patterns and regime awareness.

Evolutionary rounds:
  1. All 1000 run the challenge across ALL symbols on the same M5 data
  2. Rank by composite score (WR, PF, cross-symbol consistency, DD)
  3. Top 10% survive as elite
  4. Next 20% breed (crossover parameters)
  5. Bottom 70% die and are replaced by mutated offspring
  6. Repeat for N evolutionary rounds
  7. WINNER = highest combined score across all rounds + all symbols

Metrics tracked:
  - Per-symbol: WR, PF, trades, profit, DD, signals
  - Cross-symbol: consistency, best/worst symbol, regime tags
  - Aggregate: combined score, challenge pass rate, signal richness

Usage:
    python prop_challenge_1000.py
    python prop_challenge_1000.py --accounts 1000 --rounds 5
    python prop_challenge_1000.py --accounts 500 --rounds 3 --quick
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
from typing import List
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
# CHALLENGER DNA - Mutable trading parameters
# ============================================================

@dataclass
class ChallengerDNA:
    """Each challenger has unique mutated trading parameters."""
    challenger_id: int
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

    def mutate(self, strength=0.15):
        """Mutate parameters with Gaussian noise."""
        def _m_int(val, low, high):
            new = val + int(random.gauss(0, max(1, (high - low) * strength)))
            return max(low, min(high, new))

        def _m_float(val, low, high):
            new = val + random.gauss(0, (high - low) * strength)
            return max(low, min(high, round(new, 4)))

        self.fast_ema = _m_int(self.fast_ema, 2, 20)
        self.slow_ema = _m_int(self.slow_ema, 5, 50)
        if self.slow_ema <= self.fast_ema:
            self.slow_ema = self.fast_ema + 3
        self.atr_period = _m_int(self.atr_period, 5, 30)
        self.confidence_threshold = _m_float(self.confidence_threshold, 0.05, 0.50)
        self.sl_atr_mult = _m_float(self.sl_atr_mult, 0.3, 3.0)
        self.tp_multiplier = _m_float(self.tp_multiplier, 1.5, 6.0)
        self.dyn_tp_percent = _m_float(self.dyn_tp_percent, 20.0, 80.0)
        self.rolling_sl_mult = _m_float(self.rolling_sl_mult, 1.0, 3.0)
        self.max_positions = _m_int(self.max_positions, 5, 40)
        self.grid_spacing = _m_int(self.grid_spacing, 50, 500)
        self.conf_weight = _m_float(self.conf_weight, 0.3, 0.9)
        self.dd_weight = round(1.0 - self.conf_weight, 4)
        self.rsi_overbought = _m_float(self.rsi_overbought, 60.0, 85.0)
        self.rsi_oversold = _m_float(self.rsi_oversold, 15.0, 40.0)
        self.use_rsi_filter = random.random() < 0.3
        self.use_volume_filter = random.random() < 0.2
        self.volume_ma_mult = _m_float(self.volume_ma_mult, 1.0, 3.0)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def crossover_dna(parent1: ChallengerDNA, parent2: ChallengerDNA, child_id: int) -> ChallengerDNA:
    """Breed two challengers - uniform crossover on each parameter."""
    child = ChallengerDNA(challenger_id=child_id)
    for attr in ['fast_ema', 'slow_ema', 'atr_period', 'confidence_threshold',
                 'sl_atr_mult', 'tp_multiplier', 'dyn_tp_percent', 'rolling_sl_mult',
                 'max_positions', 'grid_spacing', 'conf_weight',
                 'rsi_overbought', 'rsi_oversold', 'use_rsi_filter',
                 'use_volume_filter', 'volume_ma_mult']:
        val = getattr(parent1, attr) if random.random() < 0.5 else getattr(parent2, attr)
        setattr(child, attr, val)
    child.dd_weight = round(1.0 - child.conf_weight, 4)
    child.parent_ids = [parent1.challenger_id, parent2.challenger_id]
    if child.slow_ema <= child.fast_ema:
        child.slow_ema = child.fast_ema + 3
    return child


# ============================================================
# CHALLENGE RUNNER - Simulate one account
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
    regime_tag: str = ""  # "clean", "volatile", "choppy"


@dataclass
class ChallengeResult:
    challenger_id: int
    dna: dict
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
    per_symbol: dict = field(default_factory=dict)  # symbol -> SymbolResult
    best_symbol: str = ""
    worst_symbol: str = ""
    symbol_consistency: float = 0.0  # std dev of win rates across symbols
    symbols_profitable: int = 0
    symbols_total: int = 0


def run_challenge(dna: ChallengerDNA, df: pd.DataFrame, point: float = 0.01,
                  contract_size: float = 1.0) -> ChallengeResult:
    """
    Run one prop firm challenge for a single challenger.
    Returns full result with score and signal counts.
    """
    INITIAL_BALANCE = 100000.0
    MAX_DAILY_DD = 5000.0    # 5%
    MAX_TOTAL_DD = 10000.0   # 10%
    PROFIT_TARGET = 10000.0  # 10%

    # Pre-compute indicators
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    tick_vol = df['tick_volume'].values if 'tick_volume' in df.columns else np.ones(len(df))

    n = len(close)

    # EMAs (vectorized)
    ema_fast = pd.Series(close).ewm(span=dna.fast_ema, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=dna.slow_ema, adjust=False).mean().values

    # ATR
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(dna.atr_period).mean().values

    # RSI (if filter enabled)
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

    # Simulation state
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    max_dd = 0.0
    max_dd_pct = 0.0

    positions = []  # (direction, entry, lot, sl, tp, dyn_tp, dyn_taken, rolling_sl, remaining_lot, partial_pnl)
    closed = []     # (pnl, direction)

    challenge_passed = False
    challenge_failed = False
    fail_reason = ""
    current_day = -1
    day_start_eq = INITIAL_BALANCE
    days_traded = 0

    buy_signals = 0
    sell_signals = 0
    next_ticket = dna.challenger_id * 100000

    start_bar = max(dna.slow_ema, dna.atr_period, 26) + 5

    for i in range(start_bar, n):
        price = close[i]
        a = atr[i]
        if np.isnan(a) or a == 0:
            continue

        # Day tracking (approximate from bar index for speed)
        day_approx = i // 288  # M5 = 288 bars/day
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

            # SL
            if (direction == 1 and price <= rsl) or (direction == -1 and price >= rsl):
                pnl = (rsl - entry if direction == 1 else entry - rsl) * rem_lot * contract_size
                balance += pnl
                closed.append((pnl + ppnl, direction))
                continue

            # TP
            if (direction == 1 and price >= tp) or (direction == -1 and price <= tp):
                pnl = (tp - entry if direction == 1 else entry - tp) * rem_lot * contract_size
                balance += pnl
                closed.append((pnl + ppnl, direction))
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
            last_sell = min([p[1] for p in positions if p[0] == -1], default=1e9)
            if sell_count == 0 or price > last_sell + spacing:
                sd = a * dna.sl_atr_mult
                td = sd * dna.tp_multiplier
                dd_d = td * (dna.dyn_tp_percent / 100.0)
                pos = (-1, price, lot, price + sd, price - td, price - dd_d, False, price + sd, lot, 0.0)
                positions.append(pos)

    # Close remaining
    if n > 0:
        last_price = close[-1]
        for pos in positions:
            direction, entry, lot, sl, tp, dyn_tp, dyn_taken, rsl, rem_lot, ppnl = pos
            pnl = (last_price - entry if direction == 1 else entry - last_price) * rem_lot * contract_size
            closed.append((pnl + ppnl, direction))
            balance += pnl

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

    # Composite score: weighted blend
    score = 0.0
    if total_trades >= 5:
        wr_score = min(wr / 100, 1.0) * 35        # 35% weight on win rate
        pf_score = min(pf / 3.0, 1.0) * 25         # 25% weight on profit factor
        ret_score = min(max(net, 0) / PROFIT_TARGET, 1.0) * 20  # 20% return progress
        dd_score = max(0, 1.0 - max_dd_pct / 10) * 10  # 10% low drawdown
        pass_score = 10.0 if challenge_passed else 0.0   # 10% passed challenge
        score = wr_score + pf_score + ret_score + dd_score + pass_score

    return ChallengeResult(
        challenger_id=dna.challenger_id,
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


# ============================================================
# TOURNAMENT ORCHESTRATOR
# ============================================================

def run_multi_symbol_challenge(dna: ChallengerDNA, symbol_data: dict,
                                symbol_info: dict) -> ChallengeResult:
    """
    Run one challenger across ALL symbols. Aggregates results.

    Args:
        dna: Challenger DNA
        symbol_data: {symbol: DataFrame} for each symbol
        symbol_info: {symbol: (point, contract_size)} for each symbol
    """
    per_symbol = {}
    total_trades = 0
    total_winners = 0
    total_losers = 0
    total_profit = 0.0
    total_gp = 0.0
    total_gl = 0.0
    total_signals = 0
    total_buy_sigs = 0
    total_sell_sigs = 0
    max_dd_all = 0.0
    any_passed = False
    any_failed = False
    fail_reasons = []

    for symbol, df in symbol_data.items():
        point, contract_size = symbol_info.get(symbol, (0.01, 1.0))
        result = run_challenge(dna, df, point, contract_size)

        # Tag regime from ATR volatility
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
            symbol=symbol,
            balance=result.balance,
            net_profit=result.net_profit,
            return_pct=result.return_pct,
            win_rate=result.win_rate,
            total_trades=result.total_trades,
            winners=result.winners,
            losers=result.losers,
            profit_factor=result.profit_factor,
            max_dd_pct=result.max_dd_pct,
            challenge_passed=result.challenge_passed,
            challenge_failed=result.challenge_failed,
            fail_reason=result.fail_reason,
            days_traded=result.days_traded,
            signals_collected=result.signals_collected,
            buy_signals=result.buy_signals,
            sell_signals=result.sell_signals,
            avg_atr=round(avg_atr, 2),
            regime_tag=regime,
        )
        per_symbol[symbol] = sym_result

        total_trades += result.total_trades
        total_winners += result.winners
        total_losers += result.losers
        total_profit += result.net_profit
        total_gp += result.net_profit if result.net_profit > 0 else 0
        total_signals += result.signals_collected
        total_buy_sigs += result.buy_signals
        total_sell_sigs += result.sell_signals
        if result.max_dd_pct > max_dd_all:
            max_dd_all = result.max_dd_pct
        if result.challenge_passed:
            any_passed = True
        if result.challenge_failed:
            any_failed = True
            fail_reasons.append(f"{symbol}: {result.fail_reason}")

    # Aggregate metrics
    agg_wr = (total_winners / total_trades * 100) if total_trades > 0 else 0

    # Profit factor across all symbols
    sym_profits = [s.net_profit for s in per_symbol.values() if s.net_profit > 0]
    sym_losses = [abs(s.net_profit) for s in per_symbol.values() if s.net_profit < 0]
    agg_pf = (sum(sym_profits) / sum(sym_losses)) if sym_losses else 0.0

    # Cross-symbol consistency (low std dev of win rates = consistent)
    sym_wrs = [s.win_rate for s in per_symbol.values() if s.total_trades > 0]
    consistency = 0.0
    if len(sym_wrs) > 1:
        consistency = float(np.std(sym_wrs))

    # Best/worst symbol
    active_syms = {k: v for k, v in per_symbol.items() if v.total_trades > 0}
    best_sym = max(active_syms, key=lambda k: active_syms[k].win_rate) if active_syms else ""
    worst_sym = min(active_syms, key=lambda k: active_syms[k].win_rate) if active_syms else ""
    profitable_count = sum(1 for s in per_symbol.values() if s.net_profit > 0)

    # Composite score (updated for multi-symbol)
    score = 0.0
    if total_trades >= 5:
        wr_score = min(agg_wr / 100, 1.0) * 25           # 25% win rate
        pf_score = min(agg_pf / 3.0, 1.0) * 20            # 20% profit factor
        consist_score = max(0, 1.0 - consistency / 30) * 15 # 15% cross-symbol consistency
        profit_score = min(max(total_profit, 0) / 10000, 1.0) * 15  # 15% profit
        dd_score = max(0, 1.0 - max_dd_all / 10) * 10      # 10% low DD
        multi_score = (profitable_count / max(len(per_symbol), 1)) * 10  # 10% multi-symbol profit
        pass_score = 5.0 if any_passed else 0.0              # 5% passed
        score = wr_score + pf_score + consist_score + profit_score + dd_score + multi_score + pass_score

    return ChallengeResult(
        challenger_id=dna.challenger_id,
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
        buy_signals=total_buy_sigs,
        sell_signals=total_sell_sigs,
        per_symbol={k: v.__dict__ for k, v in per_symbol.items()},
        best_symbol=best_sym,
        worst_symbol=worst_sym,
        symbol_consistency=round(consistency, 2),
        symbols_profitable=profitable_count,
        symbols_total=len(per_symbol),
    )


def run_tournament(num_accounts=1000, evo_rounds=5,
                   symbols=None, days=30, batch_size=100, workers=8,
                   terminal_path=None):
    """
    Run the full multi-symbol evolutionary tournament.

    1000 challengers x N rounds x ALL symbols.
    """
    if symbols is None:
        symbols = ["BTCUSD", "ETHUSD", "XAUUSD"]

    print("=" * 70)
    print("  PROP CHALLENGE 1000 - MULTI-SYMBOL EVOLUTIONARY TOURNAMENT")
    print("  $100,000 to the WINNER")
    print("=" * 70)
    print(f"  Challengers:   {num_accounts}")
    print(f"  Evo Rounds:    {evo_rounds}")
    print(f"  Symbols:       {', '.join(symbols)}")
    print(f"  History:       {days} days")
    print(f"  GPU:           {GPU_NAME}")
    print(f"  Batch Size:    {batch_size}")
    print(f"  Workers:       {workers}")
    if terminal_path:
        print(f"  Terminal:      {terminal_path}")
    print("=" * 70)

    # Init MT5
    init_ok = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()
    if not init_ok:
        print("ERROR: MT5 init failed")
        return

    # Fetch M5 data for ALL symbols
    symbol_data = {}
    symbol_info_map = {}
    bars_needed = 288 * days

    for symbol in symbols:
        print(f"\n  Fetching {bars_needed:,} M5 bars for {symbol}...")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, min(bars_needed, 500000))

        if rates is None or len(rates) < 500:
            print(f"  WARNING: {symbol} - only {len(rates) if rates is not None else 0} bars, skipping")
            continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        symbol_data[symbol] = df
        print(f"  Got {len(df):,} bars: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

        si = mt5.symbol_info(symbol)
        point = si.point if si else 0.01
        cs = si.trade_contract_size if si else 1.0
        symbol_info_map[symbol] = (point, cs)

    mt5.shutdown()

    if not symbol_data:
        print("ERROR: No symbol data loaded")
        return

    print(f"\n  Loaded {len(symbol_data)} symbols: {list(symbol_data.keys())}")

    # Output
    output_dir = SCRIPT_DIR / "signal_farm_output" / "challenge_1000"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create initial population
    print(f"\n  Generating {num_accounts} challengers with diverse DNA...")
    population = []
    for i in range(num_accounts):
        dna = ChallengerDNA(challenger_id=i + 1)
        if i > 0:
            dna.mutate(strength=0.3)
        population.append(dna)

    all_results_history = []
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
            batch_dna = population[start:end]
            batch_results = []

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_multi_symbol_challenge, dna,
                                   symbol_data, symbol_info_map): dna
                    for dna in batch_dna
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        dna = futures[future]
                        print(f"    Challenger {dna.challenger_id} error: {e}")

            results.extend(batch_results)

            active = [r for r in batch_results if r.total_trades > 0]
            if active:
                avg_wr = sum(r.win_rate for r in active) / len(active)
                passed = sum(1 for r in active if r.challenge_passed)
                print(f"    Batch {batch_num+1}/{total_batches}: "
                      f"{len(active)} active, avg WR={avg_wr:.1f}%, "
                      f"{passed} passed, "
                      f"avg consistency={sum(r.symbol_consistency for r in active)/len(active):.1f}")

        results.sort(key=lambda r: r.score, reverse=True)

        active_results = [r for r in results if r.total_trades > 0]
        round_signals = sum(r.signals_collected for r in results)
        total_signals += round_signals
        passed_count = sum(1 for r in results if r.challenge_passed)
        failed_count = sum(1 for r in results if r.challenge_failed)
        round_elapsed = time.time() - round_start

        # Per-symbol stats
        print(f"\n  ROUND {evo_round} RESULTS ({round_elapsed:.1f}s):")
        print(f"  Active:        {len(active_results)}/{len(results)}")
        print(f"  Passed:        {passed_count}")
        print(f"  Failed (DD):   {failed_count}")
        print(f"  Signals:       {round_signals:,}")

        if active_results:
            wrs = [r.win_rate for r in active_results]
            print(f"  Avg WR:        {sum(wrs)/len(wrs):.1f}%")
            print(f"  Best WR:       {max(wrs):.1f}%")

            # Per-symbol breakdown
            print(f"\n  PER-SYMBOL BREAKDOWN:")
            print(f"  {'Symbol':>8} {'AvgWR':>7} {'BestWR':>7} {'AvgPF':>7} {'AvgDD%':>7} {'Regime':>8} {'BestSym%':>8}")
            print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
            for sym in symbol_data.keys():
                sym_wrs = []
                sym_pfs = []
                sym_dds = []
                regime_counts = {}
                best_sym_count = 0
                for r in active_results:
                    if sym in r.per_symbol:
                        s = r.per_symbol[sym]
                        if isinstance(s, dict) and s.get('total_trades', 0) > 0:
                            sym_wrs.append(s['win_rate'])
                            sym_pfs.append(min(s['profit_factor'], 10))
                            sym_dds.append(s['max_dd_pct'])
                            rt = s.get('regime_tag', 'unknown')
                            regime_counts[rt] = regime_counts.get(rt, 0) + 1
                    if r.best_symbol == sym:
                        best_sym_count += 1

                if sym_wrs:
                    top_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "?"
                    best_pct = (best_sym_count / len(active_results) * 100)
                    print(f"  {sym:>8} {sum(sym_wrs)/len(sym_wrs):>6.1f}% "
                          f"{max(sym_wrs):>6.1f}% {sum(sym_pfs)/len(sym_pfs):>7.2f} "
                          f"{sum(sym_dds)/len(sym_dds):>6.1f}% {top_regime:>8} {best_pct:>7.1f}%")

        # Top 10
        print(f"\n  TOP 10 CHALLENGERS:")
        print(f"  {'#':>3} {'ID':>5} {'Score':>7} {'WR%':>6} {'PF':>6} {'Trades':>6} {'Profit':>10} {'DD%':>6} {'Best':>6} {'Con':>5} {'Pass':>4}")
        print(f"  {'-'*3} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*6} {'-'*6} {'-'*5} {'-'*4}")
        for rank, r in enumerate(results[:10], 1):
            p = "YES" if r.challenge_passed else "no"
            bs = r.best_symbol[:6] if r.best_symbol else "-"
            print(f"  {rank:>3} {r.challenger_id:>5} {r.score:>7.1f} {r.win_rate:>5.1f}% "
                  f"{r.profit_factor:>6.2f} {r.total_trades:>6} ${r.net_profit:>9,.2f} "
                  f"{r.max_dd_pct:>5.1f}% {bs:>6} {r.symbol_consistency:>4.1f} {p:>4}")

        # Save round
        all_results_history.append({
            "round": evo_round,
            "active": len(active_results),
            "passed": passed_count,
            "signals": round_signals,
            "symbols": list(symbol_data.keys()),
            "top_10": [
                {
                    "id": r.challenger_id,
                    "score": r.score,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "net_profit": r.net_profit,
                    "trades": r.total_trades,
                    "passed": r.challenge_passed,
                    "best_symbol": r.best_symbol,
                    "consistency": r.symbol_consistency,
                    "symbols_profitable": r.symbols_profitable,
                    "per_symbol": r.per_symbol,
                    "dna": r.dna,
                }
                for r in results[:10]
            ],
            "elapsed_s": round(round_elapsed, 1),
        })

        # EVOLUTION
        if evo_round < evo_rounds:
            elite_count = max(10, len(population) // 10)
            breed_count = max(20, len(population) // 5)

            result_map = {r.challenger_id: r for r in results}

            for dna in population:
                r = result_map.get(dna.challenger_id)
                if r:
                    dna.cumulative_score += r.score
                    dna.rounds_survived += 1

            population.sort(key=lambda d: result_map.get(d.challenger_id, ChallengeResult(
                0, {}, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, "", 0)).score, reverse=True)

            elite = population[:elite_count]
            breeders = population[:elite_count + breed_count]
            new_pop = list(elite)

            next_id = max(d.challenger_id for d in population) + 1
            while len(new_pop) < len(population):
                p1 = random.choice(breeders)
                p2 = random.choice(breeders)
                child = crossover_dna(p1, p2, next_id)
                child.mutate(strength=0.1 + 0.05 * evo_round)
                child.generation = evo_round
                new_pop.append(child)
                next_id += 1

            population = new_pop
            print(f"\n  EVOLUTION: {elite_count} elite survived, "
                  f"{len(population) - elite_count} new challengers bred")

    # ============================================================
    # CROWN THE WINNER
    # ============================================================
    total_time = time.time() - tournament_start
    winner = results[0]

    print(f"\n{'=' * 70}")
    print(f"  TOURNAMENT COMPLETE - MULTI-SYMBOL")
    print(f"{'=' * 70}")
    print(f"  Total Time:       {total_time/60:.1f} minutes")
    print(f"  Total Signals:    {total_signals:,}")
    print(f"  Evo Rounds:       {evo_rounds}")
    print(f"  Challengers:      {num_accounts}")
    print(f"  Symbols:          {', '.join(symbol_data.keys())}")

    print(f"\n  {'='*50}")
    print(f"  WINNER: Challenger #{winner.challenger_id}")
    print(f"  {'='*50}")
    print(f"  Score:            {winner.score:.1f}")
    print(f"  Win Rate:         {winner.win_rate:.1f}%")
    print(f"  Profit Factor:    {winner.profit_factor:.2f}")
    print(f"  Net Profit:       ${winner.net_profit:,.2f}")
    print(f"  Total Trades:     {winner.total_trades}")
    print(f"  Max DD:           {winner.max_dd_pct:.1f}%")
    print(f"  Best Symbol:      {winner.best_symbol}")
    print(f"  Consistency:      {winner.symbol_consistency:.1f} (lower=better)")
    print(f"  Symbols Profit:   {winner.symbols_profitable}/{winner.symbols_total}")
    print(f"  Passed:           {'YES' if winner.challenge_passed else 'NO'}")

    if winner.per_symbol:
        print(f"\n  WINNER PER-SYMBOL:")
        for sym, data in winner.per_symbol.items():
            if isinstance(data, dict):
                print(f"    {sym}: WR={data.get('win_rate',0):.1f}% "
                      f"PF={data.get('profit_factor',0):.2f} "
                      f"Trades={data.get('total_trades',0)} "
                      f"Profit=${data.get('net_profit',0):,.2f} "
                      f"Regime={data.get('regime_tag','?')}")

    print(f"\n  WINNING DNA:")
    for k, v in winner.dna.items():
        if k not in ('challenger_id', 'parent_ids', 'cumulative_score', 'rounds_survived', 'generation'):
            print(f"    {k}: {v}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    winner_path = output_dir / f"winner_dna_{timestamp}.json"
    with open(winner_path, 'w') as f:
        json.dump({
            "winner": {
                "challenger_id": winner.challenger_id,
                "score": winner.score,
                "win_rate": winner.win_rate,
                "profit_factor": winner.profit_factor,
                "net_profit": winner.net_profit,
                "trades": winner.total_trades,
                "passed": winner.challenge_passed,
                "best_symbol": winner.best_symbol,
                "consistency": winner.symbol_consistency,
                "per_symbol": winner.per_symbol,
                "dna": winner.dna,
            },
            "tournament": {
                "accounts": num_accounts,
                "rounds": evo_rounds,
                "symbols": list(symbol_data.keys()),
                "total_signals": total_signals,
                "total_time_min": round(total_time / 60, 1),
            },
            "history": all_results_history,
        }, f, indent=2)
    print(f"\n  Winner saved: {winner_path}")

    top50_path = output_dir / f"top50_dna_{timestamp}.json"
    top50 = [
        {"rank": i+1, "id": r.challenger_id, "score": r.score,
         "win_rate": r.win_rate, "pf": r.profit_factor,
         "best_sym": r.best_symbol, "consistency": r.symbol_consistency,
         "per_symbol": r.per_symbol, "dna": r.dna}
        for i, r in enumerate(results[:50])
    ]
    with open(top50_path, 'w') as f:
        json.dump(top50, f, indent=2)
    print(f"  Top 50 saved: {top50_path}")

    print(f"{'=' * 70}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prop Challenge 1000 - Evolutionary Tournament")
    parser.add_argument("--accounts", type=int, default=1000, help="Number of challengers")
    parser.add_argument("--rounds", type=int, default=5, help="Evolutionary rounds")
    parser.add_argument("--days", type=int, default=30, help="Days of M5 data")
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,XAUUSD",
                        help="Comma-separated symbols (default: BTCUSD,ETHUSD,XAUUSD)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--terminal", default=None,
                        help="Path to terminal64.exe (for specific broker)")
    parser.add_argument("--quick", action="store_true", help="Quick run (250 accounts, 3 rounds)")
    args = parser.parse_args()

    if args.quick:
        args.accounts = 250
        args.rounds = 3

    symbols = [s.strip() for s in args.symbols.split(",")]

    run_tournament(
        num_accounts=args.accounts,
        evo_rounds=args.rounds,
        symbols=symbols,
        days=args.days,
        batch_size=args.batch_size,
        workers=args.workers,
        terminal_path=args.terminal,
    )
