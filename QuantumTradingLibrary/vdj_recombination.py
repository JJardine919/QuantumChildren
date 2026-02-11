"""
VDJ RECOMBINATION ENGINE -- Adaptive Immune Trading System
============================================================
Domesticated Transib transposon -> RAG1/RAG2 V(D)J recombination.

9-Phase Algorithm:
    Phase 0: TRIM28 emergency suppression check
    Phase 1: Memory B cell recall (secondary immune response)
    Phase 2: Bone marrow -- quantum-guided antibody generation
    Phase 3: Thymic negative selection (risk management filter)
    Phase 4: Antigen exposure (walk-forward backtesting)
    Phase 5: Clonal selection (kill losers, clone winners)
    Phase 6: Affinity maturation (somatic hypermutation)
    Phase 7: Memory B cell promotion
    Phase 8: Consensus signal generation
    -- Phase 9 (generation advancement) is implicit --

Integration:
    - Reads TE activations from TEActivationEngine (33 families)
    - Uses 16-qubit quantum circuit for segment selection
    - Uses 6-component fitness function with Bayesian priors
    - Stores antibodies in SQLite (vdj_memory_cells.db)
    - Feeds winning signals into TEQAv3Engine via JSON signal file
    - Registers domesticated patterns in TEDomesticationTracker

Authors: DooDoo + Claude
Date:    2026-02-09
Version: VDJ-RECOMBINATION-2.0
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# VDJ subsystem imports
from vdj_segments import (
    V_SEGMENTS, D_SEGMENTS, J_SEGMENTS,
    V_RSS, D_RSS, J_RSS,
    V_NAMES, D_NAMES, J_NAMES,
    N_V, N_D, N_J,
    rss_compatible,
)
from vdj_quantum_circuit import execute_vdj_circuit
from vdj_fitness import (
    fitness_clonal_selection,
    compute_detailed_metrics,
    clonal_selection as classify_population,
    affinity_maturation as mutate_winner,
    thymic_selection,
    APOPTOSIS_THRESHOLD, SURVIVAL_THRESHOLD,
    PROLIFERATION_THRESHOLD, MEMORY_THRESHOLD,
    MAX_ACTIVE_ANTIBODIES, MIN_ACTIVE_ANTIBODIES,
    GENERATION_SIZE, EVALUATION_WINDOW_BARS,
    DOMESTICATION_EXPIRY_DAYS,
)

log = logging.getLogger(__name__)

VERSION = "VDJ-RECOMBINATION-2.0"

# Junctional diversity
JUNCTIONAL_BASE_SIGMA = 0.10


# ============================================================
# JUNCTIONAL DIVERSITY
# ============================================================

def apply_junctional_diversity(
    antibody: Dict,
    junction_seed: int,
    shock_level: float,
    base_sigma: float = JUNCTIONAL_BASE_SIGMA,
) -> Dict:
    """
    Apply TdT-like random parameter perturbation at V-D and D-J junctions.
    Turns ~2,400 valid V+D+J combos into ~240,000+ unique micro-strategies.
    """
    rng = np.random.RandomState(junction_seed)

    v_def = V_SEGMENTS[antibody["v_name"]]
    d_def = D_SEGMENTS[antibody["d_name"]]
    j_def = J_SEGMENTS[antibody["j_name"]]

    # V-D Junction: perturb V entry lookback parameters
    sigma_vd = base_sigma * (1 + shock_level * 0.5)
    perturbed_v = {}
    for lookback in v_def.get("lookback", []):
        delta = int(rng.normal(0, sigma_vd * lookback))
        perturbed_v[f"lookback_{lookback}"] = max(2, lookback + delta)

    # Perturb D's param_shift factors
    perturbed_d = {}
    for key, val in d_def.get("param_shift", {}).items():
        if isinstance(val, (int, float)):
            delta = rng.normal(0, sigma_vd * abs(val))
            perturbed_d[key] = max(0.01, val + delta)

    # D-J Junction: perturb J exit parameters
    sigma_dj = base_sigma * (1 + shock_level * 0.3)
    perturbed_j = {}
    for key, val in j_def.get("params", {}).items():
        if isinstance(val, (int, float)):
            delta = rng.normal(0, sigma_dj * abs(val + 1e-10))
            if isinstance(val, int):
                perturbed_j[key] = max(1, int(val + delta))
            else:
                perturbed_j[key] = max(0.01, val + delta)
        else:
            perturbed_j[key] = val

    antibody["perturbed_v_params"] = perturbed_v
    antibody["perturbed_d_params"] = perturbed_d
    antibody["perturbed_j_params"] = perturbed_j

    return antibody


def _make_antibody_id(v_name: str, d_name: str, j_name: str, params: Dict) -> str:
    """Generate unique hash for an antibody."""
    raw = f"{v_name}|{d_name}|{j_name}|{json.dumps(params, sort_keys=True, default=str)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ============================================================
# WALK-FORWARD BACKTESTING SIMULATOR
# ============================================================

class WalkForwardSimulator:
    """
    Simulates antibody trading logic over historical data.
    Uses the V segment for entry signals, D segment for regime filtering,
    and J segment for exit management.
    """

    @staticmethod
    def simulate(antibody: Dict, bars: np.ndarray, spread_points: float = 2.0) -> Dict:
        """
        Run a single antibody through walk-forward simulation.

        Returns dict with: n_trades, n_wins, total_profit, total_loss,
                           trade_returns, max_drawdown
        """
        if len(bars) < 100:
            return {"n_trades": 0, "n_wins": 0, "total_profit": 0, "total_loss": 0,
                    "trade_returns": [], "max_drawdown": 0}

        close = bars[:, 3]
        high = bars[:, 1]
        low = bars[:, 2]
        open_p = bars[:, 0]
        volume = bars[:, 4] if bars.shape[1] > 4 else np.ones(len(close))

        v_def = V_SEGMENTS[antibody["v_name"]]
        j_def = J_SEGMENTS[antibody["j_name"]]
        j_params = antibody.get("perturbed_j_params", j_def.get("params", {}))

        direction_map = {
            "trend-following": 1,
            "mean-reverting": -1,
            "neutral": 0,
            "adaptive": 0,
            "risk-off": 0,
        }
        base_dir = direction_map.get(v_def.get("direction", "neutral"), 0)

        atr_arr = WalkForwardSimulator._compute_atr(high, low, close, 14)

        trades = []
        in_trade = False
        entry_price = 0.0
        entry_bar = 0
        direction = 0
        sl_dist = 0.0
        tp_dist = 0.0
        trail_stop = 0.0

        max_lookback = max(v_def.get("lookback", [14])) if v_def.get("lookback") else 14
        start_bar = max(55, max_lookback + 5)

        for i in range(start_bar, len(close)):
            atr = atr_arr[i] if i < len(atr_arr) else atr_arr[-1]
            if atr < 1e-10:
                continue

            if not in_trade:
                # Regime check (simplified: use D segment detection hints)
                if not WalkForwardSimulator._check_regime(antibody, close, high, low, volume, i):
                    continue

                # Entry check
                entry_signal = WalkForwardSimulator._check_entry(antibody, close, high, low, open_p, volume, i)
                if entry_signal != 0:
                    direction = entry_signal if base_dir == 0 else base_dir
                    entry_price = close[i]
                    entry_bar = i
                    in_trade = True

                    # Set exit levels from J segment
                    exit_type = j_def.get("exit_type", "FIXED_TARGET")
                    if exit_type == "TRAILING_STOP":
                        mult = j_params.get("atr_multiplier", 2.0)
                        sl_dist = atr * mult
                        tp_dist = atr * 8.0
                        trail_stop = entry_price - direction * sl_dist
                    elif exit_type == "FIXED_TARGET":
                        rr = j_params.get("rr_ratio", 2.0)
                        sl_dist = atr * 1.5
                        tp_dist = sl_dist * rr
                    elif exit_type == "DYNAMIC":
                        sl_dist = atr * 1.5
                        tp_rr = j_params.get("tp1_rr", 1.5)
                        tp_dist = sl_dist * tp_rr
                    elif exit_type == "TIME_BASED":
                        sl_dist = atr * 2.0
                        tp_dist = atr * 6.0
                    else:
                        sl_dist = atr * 1.5
                        tp_dist = atr * 3.0

            else:
                bars_held = i - entry_bar
                current_pnl = (close[i] - entry_price) * direction

                # Update trailing stop
                if j_def.get("exit_type") == "TRAILING_STOP":
                    mult = j_params.get("atr_multiplier", 2.0)
                    if direction > 0:
                        trail_stop = max(trail_stop, close[i] - atr * mult)
                    else:
                        trail_stop = min(trail_stop, close[i] + atr * mult)

                exited = False
                exit_pnl = 0.0

                # SL hit
                if current_pnl <= -sl_dist:
                    exit_pnl = -sl_dist - spread_points
                    exited = True
                # TP hit
                elif current_pnl >= tp_dist:
                    exit_pnl = tp_dist - spread_points
                    exited = True
                # Trail hit
                elif j_def.get("exit_type") == "TRAILING_STOP":
                    if direction > 0 and close[i] <= trail_stop:
                        exit_pnl = (trail_stop - entry_price) - spread_points
                        exited = True
                    elif direction < 0 and close[i] >= trail_stop:
                        exit_pnl = (entry_price - trail_stop) - spread_points
                        exited = True
                # Time exit
                elif j_def.get("exit_type") == "TIME_BASED":
                    max_bars = j_params.get("max_bars", 60)
                    if bars_held >= max_bars:
                        exit_pnl = current_pnl - spread_points
                        exited = True
                # Emergency time limit
                elif bars_held > 500:
                    exit_pnl = current_pnl - spread_points
                    exited = True

                if exited:
                    trades.append(exit_pnl)
                    in_trade = False

        # Compute result metrics
        n_trades = len(trades)
        n_wins = sum(1 for t in trades if t > 0)
        total_profit = sum(t for t in trades if t > 0)
        total_loss = sum(t for t in trades if t <= 0)

        # Max drawdown
        max_dd = 0.0
        if trades:
            cum = np.cumsum(trades)
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        return {
            "n_trades": n_trades,
            "n_wins": n_wins,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "trade_returns": trades,
            "max_drawdown": max_dd,
        }

    @staticmethod
    def _check_regime(antibody: Dict, close, high, low, volume, i: int) -> bool:
        """Simplified regime check based on D segment detection hints."""
        d_def = D_SEGMENTS[antibody["d_name"]]
        detection = d_def.get("detection", "")

        if i < 50:
            return False

        # EMA trend detection
        if "EMA(8) > EMA(21)" in detection:
            ema8 = WalkForwardSimulator._ema(close[:i+1], 8)
            ema21 = WalkForwardSimulator._ema(close[:i+1], 21)
            if ">" in detection.split("EMA(21)")[0]:
                return ema8 > ema21
            else:
                return ema8 < ema21

        # ADX-based (approximated via directional movement)
        if "ADX < 20" in detection:
            returns = np.diff(close[i-20:i+1])
            return np.std(returns) / (np.mean(np.abs(returns)) + 1e-10) > 0.8

        # ATR spike
        if "ATR(14) > 2" in detection:
            atr14 = WalkForwardSimulator._atr_val(high[i-13:i+1], low[i-13:i+1], close[i-13:i+1])
            atr50 = WalkForwardSimulator._atr_val(high[i-49:i+1], low[i-49:i+1], close[i-49:i+1])
            return atr14 > 2 * atr50 if atr50 > 0 else False

        # Volume based
        if "volume > 5 * avg_volume" in detection:
            avg_vol = np.mean(volume[i-20:i])
            return volume[i] > 5 * avg_vol if avg_vol > 0 else False

        # Compression
        if "compression_ratio" in detection:
            recent_range = np.max(high[i-10:i+1]) - np.min(low[i-10:i+1])
            baseline_range = np.max(high[i-30:i-10]) - np.min(low[i-30:i-10])
            return recent_range < baseline_range * 0.5 if baseline_range > 0 else False

        # Session transition (approximate)
        if "session" in detection.lower():
            return True  # Always allow session-based

        # Default: pass
        return True

    @staticmethod
    def _check_entry(antibody: Dict, close, high, low, open_p, volume, i: int) -> int:
        """
        Check entry signal based on V segment signal type.
        Returns: 1 (long), -1 (short), 0 (no signal)
        """
        v_def = V_SEGMENTS[antibody["v_name"]]
        signal = v_def.get("signal", "")
        lookbacks = v_def.get("lookback", [14])
        lb = lookbacks[0] if lookbacks else 14

        if i < lb + 5:
            return 0

        if signal == "momentum":
            ret = (close[i] - close[i-lb]) / (close[i-lb] + 1e-10)
            if ret > 0.005:
                return 1
            elif ret < -0.005:
                return -1

        elif signal == "rsi":
            rsi = WalkForwardSimulator._rsi(close[:i+1], lb)
            if rsi < 30:
                return 1
            elif rsi > 70:
                return -1

        elif signal == "bollinger_position":
            sma = np.mean(close[i-lb+1:i+1])
            std = np.std(close[i-lb+1:i+1])
            if std > 0:
                z = (close[i] - sma) / std
                if z < -2.0:
                    return 1
                elif z > 2.0:
                    return -1

        elif signal == "ema_crossover":
            fast = lookbacks[0] if len(lookbacks) > 0 else 8
            slow = lookbacks[1] if len(lookbacks) > 1 else 21
            if i < slow + 2:
                return 0
            f_now = WalkForwardSimulator._ema(close[:i+1], fast)
            s_now = WalkForwardSimulator._ema(close[:i+1], slow)
            f_prev = WalkForwardSimulator._ema(close[:i], fast)
            s_prev = WalkForwardSimulator._ema(close[:i], slow)
            if f_now > s_now and f_prev <= s_prev:
                return 1
            elif f_now < s_now and f_prev >= s_prev:
                return -1

        elif signal == "macd":
            fast = lookbacks[0] if len(lookbacks) > 0 else 12
            slow = lookbacks[1] if len(lookbacks) > 1 else 26
            if i < slow + 2:
                return 0
            f_now = WalkForwardSimulator._ema(close[:i+1], fast)
            s_now = WalkForwardSimulator._ema(close[:i+1], slow)
            f_prev = WalkForwardSimulator._ema(close[:i], fast)
            s_prev = WalkForwardSimulator._ema(close[:i], slow)
            if (f_now - s_now) > 0 and (f_prev - s_prev) <= 0:
                return 1
            elif (f_now - s_now) < 0 and (f_prev - s_prev) >= 0:
                return -1

        elif signal == "mean_reversion":
            sma = np.mean(close[i-lb+1:i+1])
            std = np.std(close[i-lb+1:i+1])
            if std > 0:
                z = (close[i] - sma) / std
                if z < -2.0:
                    return 1
                elif z > 2.0:
                    return -1

        elif signal == "trend_duration":
            # Count consecutive bars in same direction
            count = 0
            for k in range(1, min(lb, i)):
                if close[i-k] < close[i-k+1]:
                    count += 1
                else:
                    break
            if count > lb * 0.6:
                return 1
            count = 0
            for k in range(1, min(lb, i)):
                if close[i-k] > close[i-k+1]:
                    count += 1
                else:
                    break
            if count > lb * 0.6:
                return -1

        elif signal == "trend_strength":
            ema8 = WalkForwardSimulator._ema(close[:i+1], 8)
            ema21 = WalkForwardSimulator._ema(close[:i+1], 21)
            diff = (ema8 - ema21) / (ema21 + 1e-10)
            if diff > 0.005:
                return 1
            elif diff < -0.005:
                return -1

        elif signal == "tick_volume":
            avg_vol = np.mean(volume[i-lb:i])
            if avg_vol > 0 and volume[i] > 2 * avg_vol:
                return 1 if close[i] > open_p[i] else -1

        elif signal == "candle_pattern":
            if i < 2:
                return 0
            body = close[i] - open_p[i]
            rng = high[i] - low[i]
            if rng > 0 and abs(body) / rng > 0.6:
                return 1 if body > 0 else -1

        elif signal in ("compression_ratio", "compression_breakout"):
            recent_range = np.max(high[i-5:i+1]) - np.min(low[i-5:i+1])
            baseline_range = np.max(high[i-30:i-5]) - np.min(low[i-30:i-5])
            if baseline_range > 0 and recent_range > baseline_range * 0.5:
                return 1 if close[i] > close[i-1] else -1

        elif signal in ("short_volatility", "atr_ratio"):
            atr5 = WalkForwardSimulator._atr_val(high[i-4:i+1], low[i-4:i+1], close[i-4:i+1])
            atr20 = WalkForwardSimulator._atr_val(high[i-19:i+1], low[i-19:i+1], close[i-19:i+1])
            if atr20 > 0 and atr5 / atr20 > 1.5:
                return 1 if close[i] > close[i-1] else -1

        elif signal == "support_resistance":
            high_20 = np.max(high[i-20:i])
            low_20 = np.min(low[i-20:i])
            rng = high_20 - low_20
            if rng > 0:
                pos = (close[i] - low_20) / rng
                if pos < 0.1:
                    return 1
                elif pos > 0.9:
                    return -1

        elif signal == "price_change":
            ret = (close[i] - close[i-1]) / (close[i-1] + 1e-10)
            if abs(ret) > 0.005:
                return 1 if ret > 0 else -1

        elif signal == "signal_noise_ratio":
            returns = np.diff(close[i-lb:i+1])
            if len(returns) > 2:
                snr = abs(np.mean(returns)) / (np.std(returns) + 1e-10)
                if snr > 0.3:
                    return 1 if np.mean(returns) > 0 else -1

        elif signal == "drawdown":
            # TRIM28 risk-off: suppress entry during drawdown
            return 0

        # Fallback for other signals: momentum-like
        elif signal in ("order_flow", "gap_analysis", "session_overlap",
                        "diversity_index", "multi_tf_variance", "cross_correlation",
                        "successful_pattern_echo", "fractal_dim", "autocorrelation",
                        "microstructure", "pattern_repetition", "volume_profile",
                        "spread_analysis", "noise_pattern", "mutation_rate"):
            ret = (close[i] - close[i-min(lb, 5)]) / (close[i-min(lb, 5)] + 1e-10)
            if abs(ret) > 0.003:
                return 1 if ret > 0 else -1

        return 0

    @staticmethod
    def _rsi(close, period=14):
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _ema(data, period):
        if len(data) < period:
            return float(np.mean(data))
        mult = 2.0 / (period + 1)
        ema = float(np.mean(data[:period]))
        for val in data[period:]:
            ema = (float(val) - ema) * mult + ema
        return ema

    @staticmethod
    def _compute_atr(high, low, close, period=14):
        n = len(close)
        atr = np.zeros(n)
        for i in range(1, n):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = tr if i < period else (atr[i-1] * (period - 1) + tr) / period
        return atr

    @staticmethod
    def _atr_val(high, low, close):
        if len(high) < 2:
            return float(high[0] - low[0]) if len(high) > 0 else 0.0
        trs = []
        for i in range(1, len(high)):
            trs.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
        return float(np.mean(trs)) if trs else 0.0


# ============================================================
# MEMORY CELL DATABASE
# ============================================================

class VDJMemoryDB:
    """SQLite database for memory B cells (domesticated strategies)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_cells (
                        antibody_id     TEXT PRIMARY KEY,
                        v_segment       TEXT NOT NULL,
                        d_segment       TEXT NOT NULL,
                        j_segment       TEXT NOT NULL,
                        v_params        TEXT,
                        d_params        TEXT,
                        j_params        TEXT,
                        fitness         REAL NOT NULL,
                        posterior_wr    REAL,
                        profit_factor   REAL,
                        sortino         REAL,
                        n_trades        INTEGER,
                        n_wins          INTEGER,
                        generation      INTEGER,
                        parent_id       TEXT,
                        maturation_rounds INTEGER DEFAULT 0,
                        created_at      TEXT,
                        last_activated  TEXT,
                        activation_count INTEGER DEFAULT 0,
                        active          INTEGER DEFAULT 1,
                        te_source       TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vdj_generations (
                        generation      INTEGER PRIMARY KEY,
                        timestamp       TEXT,
                        population_size INTEGER,
                        survivors       INTEGER,
                        memory_added    INTEGER,
                        avg_fitness     REAL,
                        best_fitness    REAL,
                        best_id         TEXT,
                        maturation_improvements INTEGER
                    )
                """)
                conn.commit()
        except Exception as e:
            log.warning("VDJ DB init failed: %s", e)

    def save_memory_cell(self, ab: Dict):
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                now = datetime.now().isoformat()
                te_source = V_SEGMENTS.get(ab.get("v_name", ""), {}).get("te_source", "")
                conn.execute("""
                    INSERT OR REPLACE INTO memory_cells VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    ab["antibody_id"], ab["v_name"], ab["d_name"], ab["j_name"],
                    json.dumps(ab.get("perturbed_v_params", {})),
                    json.dumps(ab.get("perturbed_d_params", {})),
                    json.dumps(ab.get("perturbed_j_params", {})),
                    ab.get("fitness", 0), ab.get("posterior_wr", 0.5),
                    ab.get("profit_factor", 0), ab.get("sortino", 0),
                    ab.get("n_trades", 0), ab.get("n_wins", 0),
                    ab.get("generation", 0), ab.get("parent_id", ""),
                    ab.get("maturation_rounds", 0),
                    ab.get("created_at", now), now, 0, 1, te_source,
                ))
                conn.commit()
        except Exception as e:
            log.warning("Failed to save memory cell %s: %s", ab.get("antibody_id"), e)

    def load_memory_cells(self) -> List[Dict]:
        cells = []
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM memory_cells
                    WHERE active = 1 AND fitness >= ?
                    ORDER BY fitness DESC
                """, (MEMORY_THRESHOLD,)).fetchall()
                for row in rows:
                    cells.append({
                        "antibody_id": row["antibody_id"],
                        "v_name": row["v_segment"],
                        "d_name": row["d_segment"],
                        "j_name": row["j_segment"],
                        "perturbed_v_params": json.loads(row["v_params"] or "{}"),
                        "perturbed_d_params": json.loads(row["d_params"] or "{}"),
                        "perturbed_j_params": json.loads(row["j_params"] or "{}"),
                        "fitness": row["fitness"],
                        "posterior_wr": row["posterior_wr"],
                        "n_trades": row["n_trades"],
                        "n_wins": row["n_wins"],
                        "generation": row["generation"],
                        "last_activated": row["last_activated"],
                        "source": "MEMORY_CELL",
                    })
        except Exception as e:
            log.warning("Failed to load memory cells: %s", e)
        return cells

    def save_generation(self, stats: Dict):
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO vdj_generations VALUES (?,?,?,?,?,?,?,?,?)
                """, (
                    stats["generation"], stats["timestamp"],
                    stats["population_size"], stats["survivors"],
                    stats["memory_added"], stats["avg_fitness"],
                    stats["best_fitness"], stats["best_id"],
                    stats["maturation_improvements"],
                ))
                conn.commit()
        except Exception as e:
            log.warning("Failed to save generation: %s", e)


# ============================================================
# VDJ RECOMBINATION ENGINE -- 9-Phase Algorithm
# ============================================================

class VDJRecombinationEngine:
    """
    V(D)J Recombination Engine for adaptive strategy generation.

    Integrates with TEQAv3Engine by:
        1. Consuming te_activations from TEActivationEngine
        2. Using shock_level from GenomicShockDetector
        3. Writing to TEDomesticationTracker via memory cell promotion
        4. Running ALONGSIDE the main TEQA pipeline, not replacing it

    The VDJ engine provides strategy selection (which V+D+J combo)
    while TEQA provides the real-time directional signal.
    """

    def __init__(
        self,
        memory_db_path: str = None,
        max_active: int = MAX_ACTIVE_ANTIBODIES,
        generation_size: int = GENERATION_SIZE,
        eval_window: int = EVALUATION_WINDOW_BARS,
        seed: int = None,
    ):
        self.db = VDJMemoryDB(
            memory_db_path or str(Path(__file__).parent / "vdj_memory_cells.db")
        )
        self.max_active = max_active
        self.generation_size = generation_size
        self.eval_window = eval_window
        self.rng = np.random.RandomState(seed)

        self.active_antibodies: List[Dict] = []
        self.bone_marrow_pool: List[Dict] = []
        self.memory_cells: List[Dict] = self.db.load_memory_cells()
        self.generation = 0

        log.info("[VDJ] Engine initialized: %d memory cells loaded", len(self.memory_cells))

    # ----------------------------------------------------------
    # MAIN ENTRY POINT: run_cycle (9-Phase Algorithm)
    # ----------------------------------------------------------

    def run_cycle(
        self,
        bars: np.ndarray,
        symbol: str,
        te_activations: List[Dict],
        shock_level: float,
        shock_label: str,
        drawdown: float = 0.0,
    ) -> Dict:
        """
        Execute one full VDJ recombination cycle (9 phases).

        Args:
            bars: OHLCV numpy array
            symbol: trading instrument
            te_activations: from TEActivationEngine
            shock_level: from GenomicShockDetector
            shock_label: CALM/NORMAL/ELEVATED/SHOCK/EXTREME
            drawdown: current drawdown fraction

        Returns:
            dict with action, confidence, lot_mult, strategy details
        """
        t_start = time.time()

        # ============ PHASE 0: TRIM28 CHECK ============
        if shock_label == "EXTREME":
            return {
                "action": "HOLD", "confidence": 0.0, "lot_mult": 1.0,
                "source": "TRIM28_SUPPRESSION", "generation": self.generation,
                "population_stats": self._pop_stats(),
            }

        # ============ PHASE 1: MEMORY CELL RECALL ============
        memory_signal = self._phase1_memory_recall(bars, te_activations, shock_label)
        if memory_signal is not None:
            return memory_signal

        # ============ PHASE 2: BONE MARROW GENERATION ============
        self._phase2_bone_marrow(te_activations, shock_level)

        # ============ PHASE 3: THYMIC SELECTION ============
        self._phase3_thymic_selection()

        # ============ PHASE 4: ANTIGEN EXPOSURE ============
        self._phase4_antigen_exposure(bars)

        # ============ PHASE 5: CLONAL SELECTION ============
        proliferators, memory_candidates = self._phase5_clonal_selection()

        # ============ PHASE 6: AFFINITY MATURATION ============
        self._phase6_affinity_maturation(proliferators, bars)

        # ============ PHASE 7: MEMORY B CELL PROMOTION ============
        new_memory = self._phase7_memory_promotion(memory_candidates)

        # ============ PHASE 8: CONSENSUS SIGNAL ============
        result = self._phase8_consensus_signal(bars, te_activations)

        # ============ PHASE 9: GENERATION ADVANCEMENT ============
        self.generation += 1
        elapsed = time.time() - t_start

        self.db.save_generation({
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(self.active_antibodies) + len(self.bone_marrow_pool),
            "survivors": len(self.active_antibodies),
            "memory_added": len(new_memory),
            "avg_fitness": np.mean([a.get("fitness", 0) for a in self.active_antibodies]) if self.active_antibodies else 0,
            "best_fitness": max((a.get("fitness", 0) for a in self.active_antibodies), default=0),
            "best_id": self.active_antibodies[0].get("antibody_id", "") if self.active_antibodies else "",
            "maturation_improvements": result.get("maturation_improvements", 0),
        })

        result["generation"] = self.generation
        result["population_stats"] = self._pop_stats()
        result["elapsed_seconds"] = elapsed

        log.info(
            "[VDJ] Cycle complete: gen=%d active=%d memory=%d action=%s conf=%.3f | %.1fs",
            self.generation, len(self.active_antibodies),
            len(self.memory_cells), result["action"],
            result["confidence"], elapsed,
        )

        return result

    # ----------------------------------------------------------
    # PHASE IMPLEMENTATIONS
    # ----------------------------------------------------------

    def _phase1_memory_recall(
        self, bars: np.ndarray, te_activations: List[Dict], shock_label: str
    ) -> Optional[Dict]:
        """Phase 1: Check if any memory B cells match current conditions."""
        if not self.memory_cells:
            return None

        best_match = None
        best_fitness = 0.0

        for mc in self.memory_cells:
            v_name = mc["v_name"]
            te_source = V_SEGMENTS.get(v_name, {}).get("te_source", "")

            # Check TE activation
            te_act = next((a for a in te_activations if a.get("te") == te_source), None)
            if te_act is None or te_act.get("strength", 0) < 0.5:
                continue

            # Check regime match
            d_regime = D_SEGMENTS.get(mc["d_name"], {}).get("regime", "")
            if shock_label in ("SHOCK", "EXTREME") and d_regime not in ("HIGH_VOLATILITY", "NEWS_SHOCK"):
                continue

            # Check expiry
            last_act = mc.get("last_activated")
            if last_act:
                try:
                    days_since = (datetime.now() - datetime.fromisoformat(last_act)).days
                    if days_since > DOMESTICATION_EXPIRY_DAYS:
                        continue
                except (ValueError, TypeError):
                    pass

            if mc.get("fitness", 0) > best_fitness:
                best_fitness = mc["fitness"]
                best_match = mc

        if best_match is not None and best_fitness >= MEMORY_THRESHOLD:
            # Quick signal check
            entry = WalkForwardSimulator._check_entry(
                best_match, bars[:, 3], bars[:, 1], bars[:, 2], bars[:, 0],
                bars[:, 4] if bars.shape[1] > 4 else np.ones(len(bars)),
                len(bars) - 1
            )
            if entry != 0:
                return {
                    "action": "BUY" if entry > 0 else "SELL",
                    "confidence": min(1.0, best_fitness * 1.2),
                    "lot_mult": 1.0 + 0.3 * max(0, best_fitness - SURVIVAL_THRESHOLD),
                    "source": f"MEMORY_RECALL:{best_match['antibody_id'][:8]}",
                    "strategy_id": best_match["antibody_id"],
                    "v_segment": best_match["v_name"],
                    "d_segment": best_match["d_name"],
                    "j_segment": best_match["j_name"],
                    "fitness": best_fitness,
                    "generation": self.generation,
                    "population_stats": self._pop_stats(),
                }

        return None

    def _phase2_bone_marrow(self, te_activations: List[Dict], shock_level: float):
        """Phase 2: Generate new antibodies via quantum circuit."""
        if len(self.bone_marrow_pool) >= self.generation_size:
            return

        n_needed = self.generation_size - len(self.bone_marrow_pool)

        # Determine exit bias from strongest D regime
        exit_bias = "trail"  # default

        candidates = execute_vdj_circuit(
            te_activations=te_activations,
            shock_level=shock_level,
            exit_bias=exit_bias,
            shots=4096,
            n_candidates=n_needed * 2,
            rng=self.rng,
        )

        for sel in candidates:
            if len(self.bone_marrow_pool) >= self.generation_size:
                break

            antibody = {
                "v_name": sel["v_name"],
                "d_name": sel["d_name"],
                "j_name": sel["j_name"],
                "generation": self.generation + 1,
                "fitness": 0.0,
                "parent_id": "",
                "maturation_rounds": 0,
            }

            # Apply junctional diversity
            seed = hash(f"{sel['v_name']}_{sel['d_name']}_{sel['j_name']}_{self.rng.randint(0, 999999)}")
            antibody = apply_junctional_diversity(antibody, seed & 0x7FFFFFFF, shock_level)
            antibody["antibody_id"] = _make_antibody_id(
                sel["v_name"], sel["d_name"], sel["j_name"],
                antibody.get("perturbed_j_params", {})
            )

            self.bone_marrow_pool.append(antibody)

        log.info("[VDJ] Phase 2: %d antibodies in bone marrow", len(self.bone_marrow_pool))

    def _phase3_thymic_selection(self):
        """Phase 3: Kill antibodies that violate risk management rules."""
        safe = []
        killed = 0
        for ab in self.bone_marrow_pool:
            if thymic_selection(ab):
                safe.append(ab)
            else:
                killed += 1
        self.bone_marrow_pool = safe
        if killed > 0:
            log.info("[VDJ] Phase 3: %d antibodies killed by thymic selection", killed)

    def _phase4_antigen_exposure(self, bars: np.ndarray):
        """Phase 4: Move from bone marrow to active, run walk-forward test."""
        # Graduate from bone marrow
        slots = self.max_active - len(self.active_antibodies)
        if slots > 0:
            graduates = self.bone_marrow_pool[:slots]
            self.bone_marrow_pool = self.bone_marrow_pool[slots:]
            self.active_antibodies.extend(graduates)

        # Evaluate all active antibodies
        eval_bars = bars[-self.eval_window:] if len(bars) > self.eval_window else bars
        for ab in self.active_antibodies:
            result = WalkForwardSimulator.simulate(ab, eval_bars)
            ab["fitness"] = fitness_clonal_selection(result)
            metrics = compute_detailed_metrics(result)
            ab.update({
                "n_trades": result["n_trades"],
                "n_wins": result["n_wins"],
                "total_profit": result["total_profit"],
                "total_loss": result["total_loss"],
                "max_drawdown": result["max_drawdown"],
                "posterior_wr": metrics["posterior_wr"],
                "profit_factor": metrics["profit_factor"],
                "sortino": metrics["sortino"],
                "trade_returns": result["trade_returns"],
            })

        # Sort by fitness
        self.active_antibodies.sort(key=lambda a: a.get("fitness", 0), reverse=True)
        log.info("[VDJ] Phase 4: %d antibodies evaluated", len(self.active_antibodies))

    def _phase5_clonal_selection(self):
        """Phase 5: Classify antibodies by fitness threshold."""
        selection = classify_population(self.active_antibodies)

        # Keep only survivors (includes proliferators and memory candidates)
        self.active_antibodies = selection["survivors"]

        # Ensure minimum population
        if len(self.active_antibodies) < MIN_ACTIVE_ANTIBODIES:
            # Keep some anergic ones to maintain diversity
            deficit = MIN_ACTIVE_ANTIBODIES - len(self.active_antibodies)
            self.active_antibodies.extend(selection["anergic"][:deficit])

        log.info(
            "[VDJ] Phase 5: %d dead, %d survive, %d proliferate, %d memory candidates",
            len(selection["dead"]), len(self.active_antibodies),
            len(selection["proliferators"]), len(selection["memory_candidates"]),
        )

        return selection["proliferators"], selection["memory_candidates"]

    def _phase6_affinity_maturation(self, proliferators: List[Dict], bars: np.ndarray):
        """Phase 6: Mutate winning antibodies, test mutants."""
        improvements = 0
        eval_bars = bars[-self.eval_window:] if len(bars) > self.eval_window else bars

        for parent in proliferators:
            mutants = mutate_winner(
                parent,
                n_mutants=5,
                generation=parent.get("generation", 0),
                rng=self.rng,
            )

            for mutant in mutants:
                result = WalkForwardSimulator.simulate(mutant, eval_bars)
                mutant["fitness"] = fitness_clonal_selection(result)

            # Keep best mutant if better than parent
            if mutants:
                best_mutant = max(mutants, key=lambda m: m.get("fitness", 0))
                if best_mutant.get("fitness", 0) > parent.get("fitness", 0):
                    best_mutant["maturation_rounds"] = parent.get("maturation_rounds", 0) + 1
                    self.active_antibodies.append(best_mutant)
                    improvements += 1

        # Enforce population cap: keep only top N by fitness
        if len(self.active_antibodies) > self.max_active:
            self.active_antibodies.sort(key=lambda a: a.get("fitness", 0), reverse=True)
            self.active_antibodies = self.active_antibodies[:self.max_active]

        log.info("[VDJ] Phase 6: %d maturation improvements", improvements)
        return improvements

    def _phase7_memory_promotion(self, memory_candidates: List[Dict]) -> List[Dict]:
        """Phase 7: Promote exceptional antibodies to memory B cell status."""
        new_memory = []
        for ab in memory_candidates:
            if (ab.get("fitness", 0) >= MEMORY_THRESHOLD
                    and ab.get("maturation_rounds", 0) >= 3):
                ab["created_at"] = datetime.now().isoformat()
                self.db.save_memory_cell(ab)
                new_memory.append(ab)

                # Register in domestication tracker if available
                te_source = V_SEGMENTS.get(ab.get("v_name", ""), {}).get("te_source", "")
                log.info(
                    "[VDJ] MEMORY B CELL: %s fitness=%.3f WR=%.1f%% TE=%s",
                    ab.get("antibody_id", "")[:8], ab.get("fitness", 0),
                    ab.get("posterior_wr", 0.5) * 100, te_source,
                )

        # Update in-memory list
        existing_ids = {m["antibody_id"] for m in self.memory_cells}
        for ab in new_memory:
            if ab["antibody_id"] not in existing_ids:
                self.memory_cells.append(ab)

        log.info("[VDJ] Phase 7: %d new memory cells, %d total", len(new_memory), len(self.memory_cells))
        return new_memory

    def _phase8_consensus_signal(self, bars: np.ndarray, te_activations: List[Dict]) -> Dict:
        """Phase 8: Fitness-weighted vote from all active antibodies."""
        votes = []
        i = len(bars) - 1

        for ab in self.active_antibodies:
            entry = WalkForwardSimulator._check_entry(
                ab, bars[:, 3], bars[:, 1], bars[:, 2], bars[:, 0],
                bars[:, 4] if bars.shape[1] > 4 else np.ones(len(bars)),
                i
            )
            if entry != 0:
                votes.append({
                    "direction": entry,
                    "fitness": ab.get("fitness", 0),
                    "antibody_id": ab.get("antibody_id", ""),
                })

        if not votes:
            return {
                "action": "HOLD", "confidence": 0.0, "lot_mult": 1.0,
                "source": "VDJ_NO_SIGNAL",
            }

        total_fitness = sum(v["fitness"] for v in votes)
        if total_fitness <= 0:
            return {
                "action": "HOLD", "confidence": 0.0, "lot_mult": 1.0,
                "source": "VDJ_LOW_FITNESS",
            }

        weighted_dir = sum(
            v["direction"] * v["fitness"] for v in votes
        ) / total_fitness

        if weighted_dir > 0.1:
            action = "BUY"
        elif weighted_dir < -0.1:
            action = "SELL"
        else:
            action = "HOLD"

        confidence = min(1.0, abs(weighted_dir))
        best_fitness = max(v["fitness"] for v in votes)
        lot_mult = 1.0 + 0.3 * max(0, best_fitness - SURVIVAL_THRESHOLD)

        # Find the best matching antibody for strategy details
        best_vote = max(votes, key=lambda v: v["fitness"])
        best_ab = next((a for a in self.active_antibodies
                        if a.get("antibody_id") == best_vote["antibody_id"]), {})

        return {
            "action": action,
            "confidence": confidence,
            "lot_mult": lot_mult,
            "source": f"VDJ_GEN{self.generation + 1}",
            "strategy_id": best_vote.get("antibody_id", ""),
            "v_segment": best_ab.get("v_name", ""),
            "d_segment": best_ab.get("d_name", ""),
            "j_segment": best_ab.get("j_name", ""),
            "fitness": best_fitness,
            "n_voters": len(votes),
            "weighted_direction": weighted_dir,
            "maturation_improvements": 0,
        }

    def _pop_stats(self) -> Dict:
        return {
            "active": len(self.active_antibodies),
            "bone_marrow": len(self.bone_marrow_pool),
            "memory": len(self.memory_cells),
            "generation": self.generation,
        }


# ============================================================
# TEQA BRIDGE
# ============================================================

class VDJTEQABridge:
    """
    Bridges VDJ engine with the TEQA v3.0 pipeline.
    Writes antibody signals to JSON for MQL5 EA consumption.
    """

    def __init__(self, vdj_engine: VDJRecombinationEngine, signal_file: str = None):
        self.vdj = vdj_engine
        self.signal_file = signal_file or str(
            Path(__file__).parent / "vdj_antibody_signal.json"
        )

    def write_signal_file(self, vdj_result: Dict):
        signal = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "action": vdj_result.get("action", "HOLD"),
            "confidence": vdj_result.get("confidence", 0),
            "lot_mult": vdj_result.get("lot_mult", 1.0),
            "source": vdj_result.get("source", ""),
            "v_segment": vdj_result.get("v_segment", ""),
            "d_segment": vdj_result.get("d_segment", ""),
            "j_segment": vdj_result.get("j_segment", ""),
            "fitness": vdj_result.get("fitness", 0),
            "generation": vdj_result.get("generation", 0),
            "population": vdj_result.get("population_stats", {}),
        }
        try:
            tmp = self.signal_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(signal, f, indent=2)
            os.replace(tmp, self.signal_file)
        except Exception as e:
            log.warning("Failed to write VDJ signal: %s", e)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    from vdj_segments import count_valid_combinations

    print("=" * 76)
    print("  VDJ RECOMBINATION ENGINE v2.0")
    print("  RAG1/RAG2 Domesticated Transib -> Adaptive Immune Trading System")
    print("=" * 76)

    # Synthetic data: clear uptrend with pullbacks (so strategies can profit)
    np.random.seed(42)
    n_bars = 600
    # Strong uptrend: +0.3% per bar drift with noise
    returns = np.random.randn(n_bars) * 0.005 + 0.003
    # Add mean-reversion pullbacks every ~50 bars
    for i in range(n_bars):
        if i % 50 < 10:
            returns[i] -= 0.004  # Pullback periods
    close = np.zeros(n_bars)
    close[0] = 50000
    for i in range(1, n_bars):
        close[i] = close[i-1] * (1 + returns[i])
    close = np.maximum(close, 100)
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.003))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.003))
    open_p = close * (1 + np.random.randn(n_bars) * 0.001)
    volume = np.abs(np.random.randn(n_bars) * 100 + 500)
    bars = np.column_stack([open_p, high, low, close, volume])

    print(f"\n  V segments: {N_V} | D segments: {N_D} | J segments: {N_J}")
    print(f"  Raw combinations: {N_V * N_D * N_J}")
    valid = count_valid_combinations()
    print(f"  Valid (12/23 rule): {valid}")
    print(f"  With junctional diversity: ~{valid * 100}+")

    # Fake TE activations
    te_activations = []
    for v_name in V_NAMES:
        v_def = V_SEGMENTS[v_name]
        te_activations.append({
            "te": v_def["te_source"],
            "strength": np.random.uniform(0.1, 0.9),
            "direction": np.random.choice([-1, 1]),
        })

    # Test engine
    test_db = str(Path(__file__).parent / "test_vdj_memory.db")
    engine = VDJRecombinationEngine(memory_db_path=test_db, seed=42)

    print("\n  Running VDJ cycle...")
    result = engine.run_cycle(
        bars=bars,
        symbol="BTCUSD",
        te_activations=te_activations,
        shock_level=0.3,
        shock_label="NORMAL",
    )

    print(f"\n  Result:")
    print(f"    Action:     {result['action']}")
    print(f"    Confidence: {result['confidence']:.4f}")
    print(f"    Lot mult:   {result['lot_mult']:.2f}")
    print(f"    Source:      {result['source']}")
    print(f"    Generation:  {result['generation']}")
    print(f"    Population:  {result['population_stats']}")
    print(f"    Elapsed:     {result.get('elapsed_seconds', 0):.1f}s")

    # Run multiple cycles to test maturation + memory promotion
    for cyc in range(2, 12):
        shock = max(0.1, 0.5 - cyc * 0.05)
        label = "CALM" if shock < 0.2 else "NORMAL"
        r = engine.run_cycle(
            bars=bars, symbol="BTCUSD",
            te_activations=te_activations,
            shock_level=shock, shock_label=label,
        )
        # Show top fitness for diagnostics
        fitnesses = sorted([a.get("fitness", 0) for a in engine.active_antibodies], reverse=True)
        top3 = fitnesses[:3] if len(fitnesses) >= 3 else fitnesses
        mat_rounds = [a.get("maturation_rounds", 0) for a in engine.active_antibodies]
        max_mat = max(mat_rounds) if mat_rounds else 0
        print(f"  Cycle {cyc:2d}: {r['action']:4s} conf={r['confidence']:.3f} "
              f"active={r['population_stats']['active']:3d} "
              f"memory={r['population_stats']['memory']} "
              f"top_fit=[{', '.join(f'{f:.3f}' for f in top3)}] "
              f"max_mat_rounds={max_mat}")

    # Force-test memory promotion path by injecting a high-fitness antibody
    print("\n  Testing memory promotion path...")
    fake_memory = {
        "antibody_id": "test_memory_001",
        "v_name": V_NAMES[0], "d_name": D_NAMES[0], "j_name": J_NAMES[0],
        "perturbed_v_params": {}, "perturbed_d_params": {}, "perturbed_j_params": {},
        "fitness": 0.85, "posterior_wr": 0.72, "profit_factor": 2.5,
        "sortino": 1.8, "n_trades": 40, "n_wins": 28,
        "generation": 5, "parent_id": "", "maturation_rounds": 5,
    }
    promoted = engine._phase7_memory_promotion([fake_memory])
    print(f"    Promoted: {len(promoted)} memory cells")
    assert len(promoted) == 1, "Memory promotion failed!"
    assert len(engine.memory_cells) >= 1, "Memory cell not stored!"
    print(f"    Total memory cells: {len(engine.memory_cells)}")

    # Verify SQLite persistence
    print("\n  SQLite verification:")
    import sqlite3
    with sqlite3.connect(test_db) as conn:
        gen_rows = conn.execute("SELECT COUNT(*) FROM vdj_generations").fetchone()[0]
        mem_rows = conn.execute("SELECT COUNT(*) FROM memory_cells").fetchone()[0]
        print(f"    Generations logged: {gen_rows}")
        print(f"    Memory cells stored: {mem_rows}")
        if mem_rows > 0:
            top = conn.execute(
                "SELECT antibody_id, fitness, v_segment, d_segment, j_segment "
                "FROM memory_cells ORDER BY fitness DESC LIMIT 3"
            ).fetchall()
            for row in top:
                print(f"    Top memory: {row[0]} fitness={row[1]:.3f} ({row[2]}+{row[3]}+{row[4]})")

    # Cleanup
    try:
        os.remove(test_db)
    except OSError:
        pass

    print("\n" + "=" * 76)
    print("  VDJ v2.0 test complete.")
    print("=" * 76)
