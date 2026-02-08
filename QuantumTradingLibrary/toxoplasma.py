"""
TOXOPLASMA ENGINE -- Regime Behavior Modification
===================================================
Detects when the market is hijacking your strategies through regime shifts,
scores the degree of behavioral infection, and activates countermeasures.

Biological basis:
    Toxoplasma gondii modifies host behavior by producing tyrosine hydroxylase
    (increasing dopamine) and secreting effector proteins (ROP16, GRA15, GRA16)
    that hijack host gene expression. Infected rodents lose fear of cats,
    effectively running toward their own predator.

    The market does the same thing: regime shifts make strategies act against
    their own nature. Trending markets "infect" mean-reversion strategies.
    Ranging markets "infect" momentum strategies. Volatility spikes act as
    "dopamine" that makes the system overtrade.

    This engine detects the infection, scores its severity, and activates
    proportional countermeasures -- from position reduction to full signal
    inversion.

Integration:
    - Reads TE activations from TEActivationEngine (33 families)
    - Tracks TE patterns associated with regime manipulation
    - Feeds manipulation flags back to TE Domestication tracker
    - Writes infection status to JSON for BRAIN script consumption
    - Uses config_loader for all trading thresholds (never hardcoded)

Authors: DooDoo + Claude
Date:    2026-02-08
Version: TOXOPLASMA-1.0
"""

import json
import logging
import math
import os
import hashlib
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import trading settings from config_loader (NEVER hardcode)
from config_loader import CONFIDENCE_THRESHOLD, MAX_LOSS_DOLLARS

log = logging.getLogger(__name__)

VERSION = "TOXOPLASMA-1.0"

# ============================================================
# CONSTANTS (behavioral thresholds -- not trading values)
# ============================================================

# Behavioral baseline calibration
BASELINE_MIN_TRADES = 30
BASELINE_WINDOW_TRADES = 100
BASELINE_RECALIBRATE_HOURS = 24

# Infection detection thresholds
INFECTION_LOOKBACK_TRADES = 20
INFECTION_MILD_THRESHOLD = 0.35
INFECTION_MODERATE_THRESHOLD = 0.55
INFECTION_SEVERE_THRESHOLD = 0.75
INFECTION_CRITICAL = 0.90

# Dopamine proxy (volatility/volume excitement index)
DOPAMINE_VOL_LOOKBACK = 20
DOPAMINE_SPIKE_MULT = 1.8
DOPAMINE_VOLUME_SPIKE_MULT = 2.0
DOPAMINE_DECAY_RATE = 0.85

# Anti-parasitic countermeasures
COUNTERMEASURE_SIZE_REDUCTION = 0.50
COUNTERMEASURE_CONFIDENCE_BOOST = 0.15
COUNTERMEASURE_HOLD_REDUCTION = 0.60
COUNTERMEASURE_INVERSION_THRESH = 0.80

# Chronic vs acute classification
ACUTE_WINDOW_HOURS = 4
CHRONIC_WINDOW_HOURS = 48
CHRONIC_ADAPTATION_RATE = 0.02

# TE epigenetic tracking
TE_MANIPULATION_MIN_COUNT = 5

# Bayesian prior for smoothing (consistent with VDJ / domestication)
PRIOR_ALPHA = 8
PRIOR_BETA = 8


# ============================================================
# ENUMS
# ============================================================

class StrategyType(Enum):
    """Classification of trading strategy behavior."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    HYBRID = "hybrid"


class InfectionPhase(Enum):
    """Severity phase of market regime hijacking."""
    HEALTHY = "healthy"
    ACUTE = "acute"
    CHRONIC = "chronic"
    CRITICAL = "critical"


class RegimeType(Enum):
    """Current market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    COMPRESSED = "compressed"
    TRANSITIONING = "transitioning"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class BehavioralBaseline:
    """
    The 'healthy self' profile for a strategy.

    Like the immune system learning self-antigens during thymic selection,
    this captures what the strategy looks like when NOT being manipulated
    by the market. Deviations from this baseline indicate infection.
    """
    strategy_id: str = ""
    strategy_type: str = StrategyType.HYBRID.value
    symbol: str = ""
    # Overall statistics
    baseline_win_rate: float = 0.5
    baseline_avg_hold: float = 10.0
    baseline_trade_freq: float = 1.0
    baseline_avg_pnl: float = 0.0
    baseline_sharpe: float = 0.0
    # Regime-specific win rates
    wr_trending_up: float = 0.5
    wr_trending_down: float = 0.5
    wr_ranging: float = 0.5
    wr_volatile: float = 0.5
    wr_compressed: float = 0.5
    wr_transitioning: float = 0.5
    # Calibration metadata
    total_trades: int = 0
    last_calibrated: str = ""
    calibration_valid: bool = False


@dataclass
class DopamineState:
    """
    The volatility/volume excitement index.

    Toxoplasma produces tyrosine hydroxylase, increasing dopamine in the
    amygdala. In markets, volatility and volume spikes are the dopamine
    that makes the system want to trade more aggressively.
    """
    current_level: float = 0.0
    atr_ratio: float = 1.0
    volume_ratio: float = 1.0
    spike_count_1h: int = 0
    decay_factor: float = DOPAMINE_DECAY_RATE


@dataclass
class InfectionScore:
    """
    Composite infection diagnostic.

    Each component measures a different dimension of behavioral deviation
    from the healthy baseline. Like a blood panel for Toxoplasma antibodies.
    """
    strategy_id: str = ""
    symbol: str = ""
    timestamp: str = ""
    # Component scores (each 0.0 to 1.0)
    win_rate_deviation: float = 0.0
    hold_time_deviation: float = 0.0
    frequency_deviation: float = 0.0
    pnl_deviation: float = 0.0
    regime_mismatch: float = 0.0
    dopamine_level: float = 0.0
    # Composite
    infection_score: float = 0.0
    infection_phase: str = InfectionPhase.HEALTHY.value
    # Duration tracking
    infection_start: str = ""
    infection_duration_hours: float = 0.0


@dataclass
class CountermeasureSet:
    """
    Anti-parasitic countermeasures applied when infection is detected.

    Like the IFN-gamma / IL-12 immune response against Toxoplasma,
    these measures resist the market's behavioral hijacking.
    """
    position_size_mult: float = 1.0
    confidence_offset: float = 0.0
    hold_time_mult: float = 1.0
    signal_inversion: bool = False
    strategy_pause: bool = False
    active_measures: List[str] = field(default_factory=list)


@dataclass
class TradeRecord:
    """Minimal trade record for baseline calibration and infection detection."""
    ticket: int = 0
    symbol: str = ""
    direction: int = 0  # 1 = long, -1 = short
    profit: float = 0.0
    hold_time_bars: float = 0.0
    regime: str = RegimeType.RANGING.value
    timestamp: str = ""
    active_tes: List[str] = field(default_factory=list)


# ============================================================
# REGIME MISMATCH MATRIX
# ============================================================

# The core "infection vector" mapping.
# High values = the market is trying to make you trade against your nature.
# This is the "running toward the cat" detector.

_MISMATCH_MATRIX: Dict[Tuple[str, str], float] = {
    # MOMENTUM strategies
    (StrategyType.MOMENTUM.value, RegimeType.TRENDING_UP.value): 0.05,
    (StrategyType.MOMENTUM.value, RegimeType.TRENDING_DOWN.value): 0.10,
    (StrategyType.MOMENTUM.value, RegimeType.RANGING.value): 0.90,
    (StrategyType.MOMENTUM.value, RegimeType.VOLATILE.value): 0.60,
    (StrategyType.MOMENTUM.value, RegimeType.COMPRESSED.value): 0.40,
    (StrategyType.MOMENTUM.value, RegimeType.TRANSITIONING.value): 0.70,

    # MEAN REVERSION strategies
    (StrategyType.MEAN_REVERSION.value, RegimeType.TRENDING_UP.value): 0.85,
    (StrategyType.MEAN_REVERSION.value, RegimeType.TRENDING_DOWN.value): 0.85,
    (StrategyType.MEAN_REVERSION.value, RegimeType.RANGING.value): 0.05,
    (StrategyType.MEAN_REVERSION.value, RegimeType.VOLATILE.value): 0.70,
    (StrategyType.MEAN_REVERSION.value, RegimeType.COMPRESSED.value): 0.20,
    (StrategyType.MEAN_REVERSION.value, RegimeType.TRANSITIONING.value): 0.80,

    # VOLATILITY strategies
    (StrategyType.VOLATILITY.value, RegimeType.TRENDING_UP.value): 0.30,
    (StrategyType.VOLATILITY.value, RegimeType.TRENDING_DOWN.value): 0.30,
    (StrategyType.VOLATILITY.value, RegimeType.RANGING.value): 0.40,
    (StrategyType.VOLATILITY.value, RegimeType.VOLATILE.value): 0.05,
    (StrategyType.VOLATILITY.value, RegimeType.COMPRESSED.value): 0.80,
    (StrategyType.VOLATILITY.value, RegimeType.TRANSITIONING.value): 0.15,

    # HYBRID strategies
    (StrategyType.HYBRID.value, RegimeType.TRENDING_UP.value): 0.25,
    (StrategyType.HYBRID.value, RegimeType.TRENDING_DOWN.value): 0.25,
    (StrategyType.HYBRID.value, RegimeType.RANGING.value): 0.30,
    (StrategyType.HYBRID.value, RegimeType.VOLATILE.value): 0.35,
    (StrategyType.HYBRID.value, RegimeType.COMPRESSED.value): 0.40,
    (StrategyType.HYBRID.value, RegimeType.TRANSITIONING.value): 0.45,
}


def get_regime_mismatch(strategy_type: str, regime: str) -> float:
    """Look up the mismatch score for a strategy-type / regime pair."""
    return _MISMATCH_MATRIX.get((strategy_type, regime), 0.30)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _sigmoid(x: float) -> float:
    """Standard sigmoid function, clamped to avoid overflow."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from a to b by factor t (0..1)."""
    return a + (b - a) * _clamp(t)


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> np.ndarray:
    """Compute ATR array from OHLC data."""
    n = len(close)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        if i < period:
            atr[i] = tr
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period
    return atr


def _atr_single(high: np.ndarray, low: np.ndarray,
                close: np.ndarray) -> float:
    """Compute a single ATR value from a slice of bars."""
    if len(high) < 2:
        return float(high[0] - low[0]) if len(high) > 0 else 0.0
    trs = []
    for i in range(1, len(high)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs)) if trs else 0.0


def _ema_val(data: np.ndarray, period: int) -> float:
    """Compute the most recent EMA value."""
    if len(data) < period:
        return float(np.mean(data))
    mult = 2.0 / (period + 1)
    ema = float(np.mean(data[:period]))
    for val in data[period:]:
        ema = (float(val) - ema) * mult + ema
    return ema


# ============================================================
# REGIME CLASSIFIER
# ============================================================

class RegimeClassifier:
    """
    Classifies the current market regime from OHLCV data.

    This is the "environmental scanner" -- it tells the Toxoplasma engine
    what type of market the strategy is facing, so it can assess whether
    the strategy is well-suited or being manipulated.
    """

    @staticmethod
    def classify(bars: np.ndarray) -> RegimeType:
        """
        Classify regime from OHLCV bars (N x 5: open, high, low, close, volume).

        Returns one of: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE,
                         COMPRESSED, TRANSITIONING
        """
        if len(bars) < 50:
            return RegimeType.RANGING

        close = bars[:, 3]
        high = bars[:, 1]
        low = bars[:, 2]

        # EMA trend detection
        ema8 = _ema_val(close, 8)
        ema21 = _ema_val(close, 21)
        ema_diff_pct = (ema8 - ema21) / (ema21 + 1e-10)

        # ATR for volatility measurement
        atr_recent = _atr_single(high[-5:], low[-5:], close[-5:])
        atr_baseline = _atr_single(high[-25:-5], low[-25:-5], close[-25:-5])
        atr_ratio = atr_recent / (atr_baseline + 1e-10)

        # Bollinger width for range detection
        sma20 = float(np.mean(close[-20:]))
        std20 = float(np.std(close[-20:]))
        bb_width = (2 * std20) / (sma20 + 1e-10)

        # Compression detection: was squeezed then expanded?
        atr_prev_5 = _atr_single(high[-10:-5], low[-10:-5], close[-10:-5])
        was_compressed = atr_prev_5 / (atr_baseline + 1e-10) < 0.6
        now_expanding = atr_ratio > 1.5

        # Classification logic
        if was_compressed and now_expanding:
            return RegimeType.TRANSITIONING

        if atr_ratio < 0.5:
            return RegimeType.COMPRESSED

        if atr_ratio > 2.0:
            return RegimeType.VOLATILE

        if abs(ema_diff_pct) > 0.005:
            if ema_diff_pct > 0:
                return RegimeType.TRENDING_UP
            else:
                return RegimeType.TRENDING_DOWN

        if bb_width < 0.02:
            return RegimeType.RANGING

        # Default: check recent trend persistence
        if ema_diff_pct > 0.002:
            return RegimeType.TRENDING_UP
        elif ema_diff_pct < -0.002:
            return RegimeType.TRENDING_DOWN

        return RegimeType.RANGING


# ============================================================
# TOXOPLASMA ENGINE -- Core Implementation
# ============================================================

class ToxoplasmaEngine:
    """
    Core Toxoplasma engine that detects regime hijacking and activates
    countermeasures.

    This is the anti-parasitic immune system for the trading pipeline.
    It monitors each registered strategy for signs of behavioral infection
    (deviation from baseline), computes a composite infection score, and
    activates proportional countermeasures.

    Lifecycle:
        1. Register strategies with their type (momentum, mean-reversion, etc.)
        2. Feed trade outcomes to build behavioral baselines
        3. Each cycle: classify regime, compute dopamine, score infection
        4. Apply countermeasures to trading signals
        5. Track TE patterns associated with infection periods
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            self.db_path = str(
                Path(__file__).parent / "toxoplasma_infection.db"
            )
        else:
            self.db_path = db_path

        self._init_db()

        # In-memory state
        self._baselines: Dict[str, BehavioralBaseline] = {}
        self._trade_history: Dict[str, List[TradeRecord]] = {}
        self._dopamine_prev: Dict[str, float] = {}
        self._infection_starts: Dict[str, Optional[str]] = {}
        self._latest_scores: Dict[str, InfectionScore] = {}
        self._latest_countermeasures: Dict[str, CountermeasureSet] = {}

        # Regime classifier
        self.regime_classifier = RegimeClassifier()

        # Load persisted baselines
        self._load_baselines()

        log.info(
            "[Toxoplasma] Engine initialized | DB=%s | baselines=%d",
            self.db_path, len(self._baselines),
        )

    # ----------------------------------------------------------
    # DATABASE
    # ----------------------------------------------------------

    def _init_db(self):
        """Initialize SQLite database with all required tables."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")

                # Strategy baselines
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_baselines (
                        strategy_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy_type TEXT NOT NULL,
                        baseline_win_rate REAL DEFAULT 0.5,
                        baseline_avg_hold REAL DEFAULT 10.0,
                        baseline_trade_freq REAL DEFAULT 1.0,
                        baseline_avg_pnl REAL DEFAULT 0.0,
                        baseline_sharpe REAL DEFAULT 0.0,
                        wr_trending_up REAL DEFAULT 0.5,
                        wr_trending_down REAL DEFAULT 0.5,
                        wr_ranging REAL DEFAULT 0.5,
                        wr_volatile REAL DEFAULT 0.5,
                        wr_compressed REAL DEFAULT 0.5,
                        wr_transitioning REAL DEFAULT 0.5,
                        total_trades INTEGER DEFAULT 0,
                        last_calibrated TEXT,
                        calibration_valid INTEGER DEFAULT 0,
                        PRIMARY KEY (strategy_id, symbol)
                    )
                """)

                # Infection score log
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS infection_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        win_rate_deviation REAL DEFAULT 0.0,
                        hold_time_deviation REAL DEFAULT 0.0,
                        frequency_deviation REAL DEFAULT 0.0,
                        pnl_deviation REAL DEFAULT 0.0,
                        regime_mismatch REAL DEFAULT 0.0,
                        dopamine_level REAL DEFAULT 0.0,
                        infection_score REAL DEFAULT 0.0,
                        infection_phase TEXT DEFAULT 'healthy',
                        infection_start TEXT,
                        infection_duration_hours REAL DEFAULT 0.0,
                        regime TEXT DEFAULT 'ranging'
                    )
                """)

                # Dopamine state log
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dopamine_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        current_level REAL DEFAULT 0.0,
                        atr_ratio REAL DEFAULT 1.0,
                        volume_ratio REAL DEFAULT 1.0,
                        spike_count_1h INTEGER DEFAULT 0
                    )
                """)

                # Countermeasure activation log
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS countermeasure_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        position_size_mult REAL DEFAULT 1.0,
                        confidence_offset REAL DEFAULT 0.0,
                        hold_time_mult REAL DEFAULT 1.0,
                        signal_inversion INTEGER DEFAULT 0,
                        strategy_pause INTEGER DEFAULT 0,
                        active_measures TEXT DEFAULT '[]',
                        infection_score REAL DEFAULT 0.0,
                        infection_phase TEXT DEFAULT 'healthy'
                    )
                """)

                # TE manipulation tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS te_manipulation (
                        te_combo_hash TEXT PRIMARY KEY,
                        te_combo TEXT NOT NULL,
                        seen_during_infection INTEGER DEFAULT 0,
                        seen_during_healthy INTEGER DEFAULT 0,
                        infection_correlation REAL DEFAULT 0.0,
                        infection_wins INTEGER DEFAULT 0,
                        infection_losses INTEGER DEFAULT 0,
                        infection_wr REAL DEFAULT 0.5,
                        is_manipulation_tool INTEGER DEFAULT 0,
                        first_seen TEXT,
                        last_seen TEXT
                    )
                """)

                # Trade history for baseline calibration
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trade_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        ticket INTEGER DEFAULT 0,
                        direction INTEGER DEFAULT 0,
                        profit REAL DEFAULT 0.0,
                        hold_time_bars REAL DEFAULT 0.0,
                        regime TEXT DEFAULT 'ranging',
                        timestamp TEXT NOT NULL,
                        active_tes TEXT DEFAULT '[]'
                    )
                """)

                conn.commit()
        except Exception as e:
            log.warning("[Toxoplasma] DB init failed: %s", e)

    def _load_baselines(self):
        """Load persisted strategy baselines from the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM strategy_baselines")
                for row in cursor.fetchall():
                    key = f"{row['strategy_id']}|{row['symbol']}"
                    bl = BehavioralBaseline(
                        strategy_id=row["strategy_id"],
                        strategy_type=row["strategy_type"],
                        symbol=row["symbol"],
                        baseline_win_rate=row["baseline_win_rate"],
                        baseline_avg_hold=row["baseline_avg_hold"],
                        baseline_trade_freq=row["baseline_trade_freq"],
                        baseline_avg_pnl=row["baseline_avg_pnl"],
                        baseline_sharpe=row["baseline_sharpe"],
                        wr_trending_up=row["wr_trending_up"],
                        wr_trending_down=row["wr_trending_down"],
                        wr_ranging=row["wr_ranging"],
                        wr_volatile=row["wr_volatile"],
                        wr_compressed=row["wr_compressed"],
                        wr_transitioning=row["wr_transitioning"],
                        total_trades=row["total_trades"],
                        last_calibrated=row["last_calibrated"] or "",
                        calibration_valid=bool(row["calibration_valid"]),
                    )
                    self._baselines[key] = bl
                log.info(
                    "[Toxoplasma] Loaded %d baselines from DB",
                    len(self._baselines),
                )
        except Exception as e:
            log.warning("[Toxoplasma] Failed to load baselines: %s", e)

    def _save_baseline(self, bl: BehavioralBaseline):
        """Persist a behavioral baseline to the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT OR REPLACE INTO strategy_baselines
                    (strategy_id, symbol, strategy_type,
                     baseline_win_rate, baseline_avg_hold,
                     baseline_trade_freq, baseline_avg_pnl, baseline_sharpe,
                     wr_trending_up, wr_trending_down, wr_ranging,
                     wr_volatile, wr_compressed, wr_transitioning,
                     total_trades, last_calibrated, calibration_valid)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bl.strategy_id, bl.symbol, bl.strategy_type,
                    bl.baseline_win_rate, bl.baseline_avg_hold,
                    bl.baseline_trade_freq, bl.baseline_avg_pnl,
                    bl.baseline_sharpe,
                    bl.wr_trending_up, bl.wr_trending_down, bl.wr_ranging,
                    bl.wr_volatile, bl.wr_compressed, bl.wr_transitioning,
                    bl.total_trades, bl.last_calibrated,
                    1 if bl.calibration_valid else 0,
                ))
                conn.commit()
        except Exception as e:
            log.warning("[Toxoplasma] Failed to save baseline: %s", e)

    def _log_infection(self, score: InfectionScore, regime: str):
        """Log an infection score to the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO infection_log
                    (strategy_id, symbol, timestamp,
                     win_rate_deviation, hold_time_deviation,
                     frequency_deviation, pnl_deviation,
                     regime_mismatch, dopamine_level,
                     infection_score, infection_phase,
                     infection_start, infection_duration_hours, regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    score.strategy_id, score.symbol, score.timestamp,
                    score.win_rate_deviation, score.hold_time_deviation,
                    score.frequency_deviation, score.pnl_deviation,
                    score.regime_mismatch, score.dopamine_level,
                    score.infection_score, score.infection_phase,
                    score.infection_start, score.infection_duration_hours,
                    regime,
                ))
                conn.commit()
        except Exception as e:
            log.warning("[Toxoplasma] Failed to log infection: %s", e)

    def _log_dopamine(self, symbol: str, state: DopamineState):
        """Log a dopamine state reading."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO dopamine_log
                    (symbol, timestamp, current_level, atr_ratio,
                     volume_ratio, spike_count_1h)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    symbol, datetime.now().isoformat(),
                    state.current_level, state.atr_ratio,
                    state.volume_ratio, state.spike_count_1h,
                ))
                conn.commit()
        except Exception as e:
            log.warning("[Toxoplasma] Failed to log dopamine: %s", e)

    def _log_countermeasures(self, strategy_id: str, symbol: str,
                             measures: CountermeasureSet,
                             infection: InfectionScore):
        """Log activated countermeasures."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO countermeasure_log
                    (strategy_id, symbol, timestamp,
                     position_size_mult, confidence_offset,
                     hold_time_mult, signal_inversion, strategy_pause,
                     active_measures, infection_score, infection_phase)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id, symbol, datetime.now().isoformat(),
                    measures.position_size_mult, measures.confidence_offset,
                    measures.hold_time_mult,
                    1 if measures.signal_inversion else 0,
                    1 if measures.strategy_pause else 0,
                    json.dumps(measures.active_measures),
                    infection.infection_score, infection.infection_phase,
                ))
                conn.commit()
        except Exception as e:
            log.warning("[Toxoplasma] Failed to log countermeasures: %s", e)

    def _save_trade(self, strategy_id: str, trade: TradeRecord):
        """Persist a trade record for baseline calibration."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO trade_history
                    (strategy_id, symbol, ticket, direction, profit,
                     hold_time_bars, regime, timestamp, active_tes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id, trade.symbol, trade.ticket,
                    trade.direction, trade.profit, trade.hold_time_bars,
                    trade.regime, trade.timestamp,
                    json.dumps(trade.active_tes),
                ))
                conn.commit()
        except Exception as e:
            log.warning("[Toxoplasma] Failed to save trade: %s", e)

    def _load_trade_history(self, strategy_id: str,
                            symbol: str) -> List[TradeRecord]:
        """Load trade history from the database for a strategy/symbol pair."""
        trades = []
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM trade_history
                    WHERE strategy_id = ? AND symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (strategy_id, symbol, BASELINE_WINDOW_TRADES))
                for row in cursor.fetchall():
                    tr = TradeRecord(
                        ticket=row["ticket"],
                        symbol=row["symbol"],
                        direction=row["direction"],
                        profit=row["profit"],
                        hold_time_bars=row["hold_time_bars"],
                        regime=row["regime"],
                        timestamp=row["timestamp"],
                        active_tes=json.loads(row["active_tes"])
                        if row["active_tes"] else [],
                    )
                    trades.append(tr)
                # Reverse so oldest is first (we SELECTed DESC)
                trades.reverse()
        except Exception as e:
            log.warning("[Toxoplasma] Failed to load trade history: %s", e)
        return trades

    # ----------------------------------------------------------
    # STRATEGY REGISTRATION
    # ----------------------------------------------------------

    def register_strategy(self, strategy_id: str, strategy_type: StrategyType,
                          symbol: str):
        """
        Register a strategy for Toxoplasma monitoring.

        Must be called before the engine can monitor a strategy.
        Creates an initial (uncalibrated) baseline.
        """
        key = f"{strategy_id}|{symbol}"
        if key not in self._baselines:
            bl = BehavioralBaseline(
                strategy_id=strategy_id,
                strategy_type=strategy_type.value,
                symbol=symbol,
                calibration_valid=False,
            )
            self._baselines[key] = bl
            self._trade_history[key] = []
            log.info(
                "[Toxoplasma] Registered strategy '%s' (%s) on %s",
                strategy_id, strategy_type.value, symbol,
            )
        else:
            log.info(
                "[Toxoplasma] Strategy '%s' on %s already registered",
                strategy_id, symbol,
            )

    # ----------------------------------------------------------
    # PHASE 1: BEHAVIORAL BASELINE CALIBRATION
    # ----------------------------------------------------------

    def calibrate_baseline(self, strategy_id: str, symbol: str,
                           trade_history: List[TradeRecord] = None
                           ) -> BehavioralBaseline:
        """
        Establish the 'healthy self' behavioral profile for a strategy.

        This is thymic selection: learning what is 'self' so deviations
        (infections) can be detected. Requires BASELINE_MIN_TRADES trades.
        """
        key = f"{strategy_id}|{symbol}"

        # Load from DB if no explicit history provided
        if trade_history is None:
            trade_history = self._load_trade_history(strategy_id, symbol)

        if len(trade_history) < BASELINE_MIN_TRADES:
            log.info(
                "[Toxoplasma] Baseline for '%s|%s': need %d trades, have %d",
                strategy_id, symbol, BASELINE_MIN_TRADES, len(trade_history),
            )
            bl = self._baselines.get(key, BehavioralBaseline(
                strategy_id=strategy_id, symbol=symbol,
                calibration_valid=False,
            ))
            bl.total_trades = len(trade_history)
            return bl

        recent = trade_history[-BASELINE_WINDOW_TRADES:]
        bl = self._baselines.get(key, BehavioralBaseline())
        bl.strategy_id = strategy_id
        bl.symbol = symbol
        bl.total_trades = len(recent)
        bl.last_calibrated = datetime.now().isoformat()

        # Preserve strategy type from registration
        if key in self._baselines:
            bl.strategy_type = self._baselines[key].strategy_type

        # Overall statistics
        wins = [t for t in recent if t.profit > 0]
        bl.baseline_win_rate = len(wins) / len(recent) if recent else 0.5

        hold_times = [t.hold_time_bars for t in recent if t.hold_time_bars > 0]
        bl.baseline_avg_hold = float(np.mean(hold_times)) if hold_times else 10.0

        # Trade frequency: trades per day
        if len(recent) >= 2 and recent[-1].timestamp and recent[0].timestamp:
            try:
                t_last = datetime.fromisoformat(recent[-1].timestamp)
                t_first = datetime.fromisoformat(recent[0].timestamp)
                span_days = max(0.01, (t_last - t_first).total_seconds() / 86400)
                bl.baseline_trade_freq = len(recent) / span_days
            except (ValueError, TypeError):
                bl.baseline_trade_freq = 1.0
        else:
            bl.baseline_trade_freq = 1.0

        pnls = [t.profit for t in recent]
        bl.baseline_avg_pnl = float(np.mean(pnls)) if pnls else 0.0

        if len(pnls) > 1:
            mean_pnl = float(np.mean(pnls))
            std_pnl = float(np.std(pnls))
            bl.baseline_sharpe = (mean_pnl / std_pnl) * math.sqrt(252) if std_pnl > 0 else 0.0
        else:
            bl.baseline_sharpe = 0.0

        # Regime-specific win rates
        for regime in RegimeType:
            regime_trades = [t for t in recent if t.regime == regime.value]
            if len(regime_trades) >= 5:
                regime_wins = [t for t in regime_trades if t.profit > 0]
                wr = len(regime_wins) / len(regime_trades)
            else:
                wr = bl.baseline_win_rate  # Fallback to overall
            setattr(bl, f"wr_{regime.value}", wr)

        bl.calibration_valid = True
        self._baselines[key] = bl
        self._save_baseline(bl)

        log.info(
            "[Toxoplasma] Baseline calibrated for '%s|%s': WR=%.1f%% "
            "hold=%.1f freq=%.1f pnl=%.4f trades=%d",
            strategy_id, symbol,
            bl.baseline_win_rate * 100, bl.baseline_avg_hold,
            bl.baseline_trade_freq, bl.baseline_avg_pnl, bl.total_trades,
        )

        return bl

    # ----------------------------------------------------------
    # PHASE 2: DOPAMINE STATE COMPUTATION
    # ----------------------------------------------------------

    def compute_dopamine(self, bars: np.ndarray,
                         symbol: str) -> DopamineState:
        """
        Compute the 'dopamine proxy' from volatility and volume data.

        Toxoplasma produces tyrosine hydroxylase, increasing dopamine in the
        amygdala. In markets, volatility and volume spikes are the dopamine
        injection that makes the system want to trade aggressively.
        """
        if len(bars) < DOPAMINE_VOL_LOOKBACK + 5:
            return DopamineState()

        high = bars[:, 1]
        low = bars[:, 2]
        close = bars[:, 3]
        volume = bars[:, 4] if bars.shape[1] > 4 else np.ones(len(close))

        # ATR ratio: current (5-bar) vs baseline (full lookback)
        atr_current = _atr_single(high[-5:], low[-5:], close[-5:])
        atr_baseline = _atr_single(
            high[-DOPAMINE_VOL_LOOKBACK:], low[-DOPAMINE_VOL_LOOKBACK:],
            close[-DOPAMINE_VOL_LOOKBACK:],
        )
        atr_ratio = atr_current / (atr_baseline + 1e-10)

        # Volume ratio: recent vs baseline
        vol_current = float(np.mean(volume[-5:])) if len(volume) >= 5 else 0.0
        vol_baseline = float(np.mean(volume[-DOPAMINE_VOL_LOOKBACK:])) \
            if len(volume) >= DOPAMINE_VOL_LOOKBACK else vol_current
        volume_ratio = vol_current / (vol_baseline + 1e-10)

        # Spike counting: bars with extreme ATR or volume in recent window
        # Use min(60, available bars) as the scan window
        scan_len = min(60, len(close) - 1)
        spike_count = 0
        if scan_len > 5:
            atr_arr = _compute_atr(high[-scan_len:], low[-scan_len:],
                                   close[-scan_len:], 14)
            scan_atr_mean = float(np.mean(atr_arr[1:])) if len(atr_arr) > 1 else 1e-10
            scan_vol_mean = float(np.mean(volume[-scan_len:])) + 1e-10
            for j in range(1, scan_len):
                if (atr_arr[j] > scan_atr_mean * DOPAMINE_SPIKE_MULT
                        or volume[-scan_len + j] > scan_vol_mean * DOPAMINE_VOLUME_SPIKE_MULT):
                    spike_count += 1

        # Composite dopamine level (sigmoid mapped)
        atr_component = _sigmoid(3.0 * (atr_ratio - 1.5))
        vol_component = _sigmoid(2.0 * (volume_ratio - 2.0))
        spike_component = min(1.0, spike_count / 10.0)

        dopamine_level = (
            atr_component * 0.40
            + vol_component * 0.35
            + spike_component * 0.25
        )

        # Decay from previous reading (dopamine does not stay elevated forever)
        prev = self._dopamine_prev.get(symbol, 0.0)
        decayed = prev * DOPAMINE_DECAY_RATE
        dopamine_level = max(dopamine_level, decayed)

        dopamine_level = _clamp(dopamine_level, 0.0, 1.0)
        self._dopamine_prev[symbol] = dopamine_level

        state = DopamineState(
            current_level=dopamine_level,
            atr_ratio=float(atr_ratio),
            volume_ratio=float(volume_ratio),
            spike_count_1h=spike_count,
            decay_factor=DOPAMINE_DECAY_RATE,
        )

        self._log_dopamine(symbol, state)
        return state

    # ----------------------------------------------------------
    # PHASE 3: INFECTION SCORING
    # ----------------------------------------------------------

    def compute_infection_score(
        self,
        strategy_id: str,
        symbol: str,
        baseline: BehavioralBaseline,
        recent_trades: List[TradeRecord],
        dopamine: DopamineState,
        current_regime: RegimeType,
    ) -> InfectionScore:
        """
        Compute the composite infection score for a strategy.

        This is the core diagnostic. Each component measures a different
        dimension of behavioral deviation from the healthy baseline.
        Like a blood panel for Toxoplasma antibodies -- multiple markers
        combined into a single infection index.
        """
        now_str = datetime.now().isoformat()

        if not baseline.calibration_valid or len(recent_trades) < 5:
            score = InfectionScore(
                strategy_id=strategy_id,
                symbol=symbol,
                timestamp=now_str,
                infection_score=0.0,
                infection_phase=InfectionPhase.HEALTHY.value,
            )
            key = f"{strategy_id}|{symbol}"
            self._latest_scores[key] = score
            return score

        recent = recent_trades[-INFECTION_LOOKBACK_TRADES:]

        # ------ Component 1: Win rate deviation ------
        # Expected WR for this regime
        expected_wr = getattr(
            baseline, f"wr_{current_regime.value}",
            baseline.baseline_win_rate,
        )
        if expected_wr is None or expected_wr <= 0:
            expected_wr = baseline.baseline_win_rate

        actual_wins = sum(1 for t in recent if t.profit > 0)
        actual_wr = actual_wins / len(recent)

        # Bayesian smoothing
        posterior_wr = (PRIOR_ALPHA + actual_wr * len(recent)) / (
            PRIOR_ALPHA + PRIOR_BETA + len(recent)
        )
        wr_deviation = _clamp(
            (expected_wr - posterior_wr) / (expected_wr + 1e-10)
        )

        # ------ Component 2: Hold time deviation ------
        hold_times = [t.hold_time_bars for t in recent if t.hold_time_bars > 0]
        if hold_times and baseline.baseline_avg_hold > 0:
            actual_hold = float(np.mean(hold_times))
            hold_ratio = actual_hold / baseline.baseline_avg_hold
            hold_deviation = _clamp(abs(hold_ratio - 1.0))
        else:
            hold_deviation = 0.0

        # ------ Component 3: Frequency deviation ------
        if len(recent) >= 2 and recent[-1].timestamp and recent[0].timestamp:
            try:
                t_last = datetime.fromisoformat(recent[-1].timestamp)
                t_first = datetime.fromisoformat(recent[0].timestamp)
                span_days = max(0.01, (t_last - t_first).total_seconds() / 86400)
                actual_freq = len(recent) / span_days
            except (ValueError, TypeError):
                actual_freq = baseline.baseline_trade_freq
        else:
            actual_freq = baseline.baseline_trade_freq

        freq_ratio = actual_freq / (baseline.baseline_trade_freq + 1e-10)
        # Overtrading is the primary infection symptom (dopamine-driven)
        freq_deviation = _clamp(max(0.0, freq_ratio - 1.0))

        # ------ Component 4: P/L deviation ------
        actual_pnls = [t.profit for t in recent]
        actual_avg_pnl = float(np.mean(actual_pnls)) if actual_pnls else 0.0
        if baseline.baseline_avg_pnl > 0:
            pnl_deviation = _clamp(
                (baseline.baseline_avg_pnl - actual_avg_pnl) /
                (baseline.baseline_avg_pnl + 1e-10)
            )
        else:
            # If baseline P/L is non-positive, measure absolute deterioration
            pnl_deviation = _clamp(-actual_avg_pnl) if actual_avg_pnl < 0 else 0.0

        # ------ Component 5: Regime mismatch ------
        mismatch = get_regime_mismatch(
            baseline.strategy_type, current_regime.value,
        )

        # ------ Component 6: Dopamine influence ------
        dopamine_component = dopamine.current_level

        # ------ COMPOSITE INFECTION SCORE ------
        infection_score = (
            wr_deviation * 0.25
            + hold_deviation * 0.10
            + freq_deviation * 0.15
            + pnl_deviation * 0.15
            + mismatch * 0.25
            + dopamine_component * 0.10
        )

        # Dopamine AMPLIFICATION: high dopamine makes all infection worse
        # (Toxoplasma's TH enzyme makes the host more susceptible)
        amplification = 1.0 + 0.5 * dopamine_component
        infection_score = _clamp(infection_score * amplification)

        # ------ Phase classification ------
        key = f"{strategy_id}|{symbol}"
        infection_start_str = self._infection_starts.get(key)
        duration_hours = 0.0

        if infection_score < INFECTION_MILD_THRESHOLD:
            phase = InfectionPhase.HEALTHY
            self._infection_starts[key] = None
            infection_start_str = ""
        else:
            if not infection_start_str:
                infection_start_str = now_str
                self._infection_starts[key] = infection_start_str

            try:
                start_dt = datetime.fromisoformat(infection_start_str)
                duration_hours = (
                    datetime.now() - start_dt
                ).total_seconds() / 3600.0
            except (ValueError, TypeError):
                duration_hours = 0.0

            if infection_score >= INFECTION_CRITICAL:
                phase = InfectionPhase.CRITICAL
            elif duration_hours > CHRONIC_WINDOW_HOURS:
                phase = InfectionPhase.CHRONIC
            else:
                phase = InfectionPhase.ACUTE

        score = InfectionScore(
            strategy_id=strategy_id,
            symbol=symbol,
            timestamp=now_str,
            win_rate_deviation=float(wr_deviation),
            hold_time_deviation=float(hold_deviation),
            frequency_deviation=float(freq_deviation),
            pnl_deviation=float(pnl_deviation),
            regime_mismatch=float(mismatch),
            dopamine_level=float(dopamine_component),
            infection_score=float(infection_score),
            infection_phase=phase.value,
            infection_start=infection_start_str or "",
            infection_duration_hours=float(duration_hours),
        )

        self._latest_scores[key] = score
        self._log_infection(score, current_regime.value)

        return score

    # ----------------------------------------------------------
    # PHASE 4: ANTI-PARASITIC COUNTERMEASURES
    # ----------------------------------------------------------

    def activate_countermeasures(
        self,
        infection: InfectionScore,
    ) -> CountermeasureSet:
        """
        Activate proportional countermeasures based on infection severity.

        Like the immune system fighting Toxoplasma:
          - Mild: IFN-gamma (moderate suppression)
          - Severe: full inflammatory cascade
          - Chronic: tolerance + adaptation
          - Critical: quarantine (halt strategy)
        """
        measures = CountermeasureSet()
        score = infection.infection_score
        phase_str = infection.infection_phase

        if phase_str == InfectionPhase.HEALTHY.value:
            return measures

        # ---- MILD INFECTION (0.35 - 0.55) ----
        if score >= INFECTION_MILD_THRESHOLD:
            t = _clamp((score - 0.35) / 0.20)
            reduction = _lerp(0.0, 0.25, t)
            measures.position_size_mult = 1.0 - reduction
            measures.confidence_offset = reduction * COUNTERMEASURE_CONFIDENCE_BOOST
            measures.active_measures.append("mild_position_reduction")
            measures.active_measures.append("mild_confidence_tightening")

        # ---- MODERATE INFECTION (0.55 - 0.75) ----
        if score >= INFECTION_MODERATE_THRESHOLD:
            measures.position_size_mult = COUNTERMEASURE_SIZE_REDUCTION
            measures.confidence_offset = COUNTERMEASURE_CONFIDENCE_BOOST
            measures.hold_time_mult = COUNTERMEASURE_HOLD_REDUCTION
            measures.active_measures.append("moderate_position_reduction")
            measures.active_measures.append("moderate_confidence_tightening")
            measures.active_measures.append("hold_time_shortened")

        # ---- SEVERE INFECTION (0.75 - 0.90) ----
        if score >= INFECTION_SEVERE_THRESHOLD:
            measures.position_size_mult = 0.25
            measures.confidence_offset = 0.25
            measures.hold_time_mult = 0.40
            measures.active_measures.append("severe_position_reduction")
            measures.active_measures.append("severe_confidence_lockdown")

            # Signal inversion: if the market is hijacking you,
            # doing the opposite is the correct play
            if score >= COUNTERMEASURE_INVERSION_THRESH:
                measures.signal_inversion = True
                measures.active_measures.append("signal_inversion_active")

        # ---- CRITICAL INFECTION (>0.90) ----
        if score >= INFECTION_CRITICAL:
            measures.strategy_pause = True
            measures.position_size_mult = 0.0
            measures.active_measures.append(
                "STRATEGY_PAUSED_CRITICAL_INFECTION"
            )

        # ---- CHRONIC ADAPTATION ----
        if phase_str == InfectionPhase.CHRONIC.value:
            duration = infection.infection_duration_hours
            tolerance = min(0.5, duration * CHRONIC_ADAPTATION_RATE)
            measures.position_size_mult = min(
                0.75, measures.position_size_mult + tolerance * 0.5,
            )
            measures.confidence_offset = max(
                0.05, measures.confidence_offset - tolerance * 0.1,
            )
            measures.active_measures.append("chronic_tolerance_adaptation")

        # Log and cache
        key = f"{infection.strategy_id}|{infection.symbol}"
        self._latest_countermeasures[key] = measures
        self._log_countermeasures(
            infection.strategy_id, infection.symbol, measures, infection,
        )

        return measures

    # ----------------------------------------------------------
    # PHASE 5: TE EPIGENETIC TRACKING
    # ----------------------------------------------------------

    def track_te_manipulation(
        self,
        active_tes: List[str],
        infection: InfectionScore,
        trade_won: bool,
    ):
        """
        Track TE activation patterns during infection vs healthy periods.

        Toxoplasma activates host L1/SINE elements as part of its epigenetic
        hijacking. Similarly, certain TE activation patterns in our system
        may be ASSOCIATED with regime manipulation.

        If specific TE combos consistently appear during infection AND lead
        to losses, they are classified as 'manipulation tools' -- the
        market's effector proteins.
        """
        if not active_tes:
            return

        combo = "+".join(sorted(active_tes))
        combo_hash = hashlib.md5(combo.encode()).hexdigest()[:16]
        is_infected = infection.infection_score >= INFECTION_MILD_THRESHOLD
        now_str = datetime.now().isoformat()

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM te_manipulation WHERE te_combo_hash = ?",
                    (combo_hash,),
                )
                row = cursor.fetchone()

                if row is not None:
                    # Unpack existing row
                    (_, _, seen_inf, seen_healthy, inf_corr,
                     inf_wins, inf_losses, inf_wr, is_manip,
                     first_seen, _) = row

                    if is_infected:
                        seen_inf += 1
                        if trade_won:
                            inf_wins += 1
                        else:
                            inf_losses += 1
                    else:
                        seen_healthy += 1

                    total_inf_total = seen_inf + seen_healthy
                    inf_corr = seen_inf / (total_inf_total + 1e-10)

                    total_inf_trades = inf_wins + inf_losses
                    inf_wr = (inf_wins / total_inf_trades
                              if total_inf_trades > 0 else 0.5)

                    # Flag as manipulation tool if strongly correlated with
                    # infection AND low win rate during infection
                    is_manip_new = int(
                        inf_corr > 0.60
                        and inf_wr < 0.45
                        and total_inf_trades >= TE_MANIPULATION_MIN_COUNT
                    )

                    conn.execute("""
                        UPDATE te_manipulation
                        SET seen_during_infection = ?,
                            seen_during_healthy = ?,
                            infection_correlation = ?,
                            infection_wins = ?,
                            infection_losses = ?,
                            infection_wr = ?,
                            is_manipulation_tool = ?,
                            last_seen = ?
                        WHERE te_combo_hash = ?
                    """, (
                        seen_inf, seen_healthy, inf_corr,
                        inf_wins, inf_losses, inf_wr,
                        is_manip_new, now_str, combo_hash,
                    ))
                else:
                    # First observation of this TE combo
                    conn.execute("""
                        INSERT INTO te_manipulation
                        (te_combo_hash, te_combo,
                         seen_during_infection, seen_during_healthy,
                         infection_correlation,
                         infection_wins, infection_losses, infection_wr,
                         is_manipulation_tool, first_seen, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """, (
                        combo_hash, combo,
                        1 if is_infected else 0,
                        0 if is_infected else 1,
                        1.0 if is_infected else 0.0,
                        1 if (is_infected and trade_won) else 0,
                        1 if (is_infected and not trade_won) else 0,
                        1.0 if trade_won else 0.0,
                        now_str, now_str,
                    ))
                conn.commit()
        except Exception as e:
            log.warning("[Toxoplasma] TE manipulation tracking error: %s", e)

    def get_manipulation_tools(self) -> List[Dict]:
        """
        Retrieve all TE patterns classified as manipulation tools.

        These are fed back to the TE Domestication tracker as suppression
        targets -- preventing the system from 'domesticating' patterns
        that only work because the market is tricking it.
        """
        results = []
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM te_manipulation
                    WHERE is_manipulation_tool = 1
                    ORDER BY infection_correlation DESC
                """)
                for row in cursor.fetchall():
                    results.append({
                        "te_combo_hash": row["te_combo_hash"],
                        "te_combo": row["te_combo"],
                        "infection_correlation": row["infection_correlation"],
                        "infection_wr": row["infection_wr"],
                        "seen_during_infection": row["seen_during_infection"],
                        "seen_during_healthy": row["seen_during_healthy"],
                    })
        except Exception as e:
            log.warning("[Toxoplasma] Failed to get manipulation tools: %s", e)
        return results

    # ----------------------------------------------------------
    # TRADE RECORDING
    # ----------------------------------------------------------

    def record_trade(self, strategy_id: str, trade: TradeRecord):
        """
        Record a completed trade outcome for baseline calibration
        and infection detection.
        """
        key = f"{strategy_id}|{trade.symbol}"
        if key not in self._trade_history:
            self._trade_history[key] = []

        self._trade_history[key].append(trade)

        # Keep in-memory history bounded
        if len(self._trade_history[key]) > BASELINE_WINDOW_TRADES * 2:
            self._trade_history[key] = (
                self._trade_history[key][-BASELINE_WINDOW_TRADES:]
            )

        # Persist to DB
        self._save_trade(strategy_id, trade)

        # Track TE manipulation if we have an infection score
        infection = self._latest_scores.get(key)
        if infection and trade.active_tes:
            self.track_te_manipulation(
                trade.active_tes, infection, trade.profit > 0,
            )

        log.debug(
            "[Toxoplasma] Recorded trade for '%s|%s': profit=%.2f regime=%s",
            strategy_id, trade.symbol, trade.profit, trade.regime,
        )

    # ----------------------------------------------------------
    # FULL CYCLE: Classify -> Dopamine -> Score -> Countermeasure
    # ----------------------------------------------------------

    def run_cycle(
        self,
        strategy_id: str,
        symbol: str,
        bars: np.ndarray,
        active_tes: List[str] = None,
    ) -> Dict:
        """
        Run a complete Toxoplasma diagnostic cycle for one strategy.

        This is the main entry point called once per TEQA/BRAIN cycle (~60s).

        Steps:
            1. Classify current market regime
            2. Compute dopamine state (volatility/volume excitement)
            3. Ensure behavioral baseline is calibrated
            4. Score infection level
            5. Activate proportional countermeasures
            6. Return full diagnostic for BRAIN script consumption

        Args:
            strategy_id: unique identifier for the strategy
            symbol: trading instrument
            bars: OHLCV numpy array (N x 5)
            active_tes: current TE activation names (optional, for tracking)

        Returns:
            Dict with infection status, countermeasures, and diagnostics.
        """
        key = f"{strategy_id}|{symbol}"

        # Step 1: Regime classification
        regime = self.regime_classifier.classify(bars)

        # Step 2: Dopamine state
        dopamine = self.compute_dopamine(bars, symbol)

        # Step 3: Get or calibrate baseline
        baseline = self._baselines.get(key)
        if baseline is None:
            # Auto-register as hybrid if not previously registered
            self.register_strategy(
                strategy_id, StrategyType.HYBRID, symbol,
            )
            baseline = self._baselines[key]

        # Re-calibrate if stale
        if baseline.last_calibrated:
            try:
                last_cal = datetime.fromisoformat(baseline.last_calibrated)
                hours_since = (
                    datetime.now() - last_cal
                ).total_seconds() / 3600.0
                if hours_since > BASELINE_RECALIBRATE_HOURS:
                    baseline = self.calibrate_baseline(strategy_id, symbol)
            except (ValueError, TypeError):
                baseline = self.calibrate_baseline(strategy_id, symbol)
        else:
            baseline = self.calibrate_baseline(strategy_id, symbol)

        # Step 4: Gather recent trades for infection scoring
        trade_history = self._trade_history.get(key, [])
        if not trade_history:
            trade_history = self._load_trade_history(strategy_id, symbol)
            self._trade_history[key] = trade_history

        # Step 5: Infection scoring
        infection = self.compute_infection_score(
            strategy_id, symbol, baseline,
            trade_history, dopamine, regime,
        )

        # Step 6: Countermeasures
        measures = self.activate_countermeasures(infection)

        # Build result
        # NOTE: confidence_offset is ADDED to CONFIDENCE_THRESHOLD from config_loader
        # The BRAIN script computes: effective_threshold = CONFIDENCE_THRESHOLD + confidence_offset
        result = {
            "version": VERSION,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            # Regime
            "regime": regime.value,
            # Dopamine
            "dopamine_level": dopamine.current_level,
            "atr_ratio": dopamine.atr_ratio,
            "volume_ratio": dopamine.volume_ratio,
            # Infection
            "infection_score": infection.infection_score,
            "infection_phase": infection.infection_phase,
            "infection_duration_hours": infection.infection_duration_hours,
            # Component breakdown
            "components": {
                "win_rate_deviation": infection.win_rate_deviation,
                "hold_time_deviation": infection.hold_time_deviation,
                "frequency_deviation": infection.frequency_deviation,
                "pnl_deviation": infection.pnl_deviation,
                "regime_mismatch": infection.regime_mismatch,
                "dopamine_level": infection.dopamine_level,
            },
            # Countermeasures
            "position_size_mult": measures.position_size_mult,
            "confidence_offset": measures.confidence_offset,
            "hold_time_mult": measures.hold_time_mult,
            "signal_inversion": measures.signal_inversion,
            "strategy_pause": measures.strategy_pause,
            "active_measures": measures.active_measures,
            # Baseline status
            "baseline_calibrated": baseline.calibration_valid,
            "baseline_trades": baseline.total_trades,
            "baseline_win_rate": baseline.baseline_win_rate,
        }

        log.info(
            "[Toxoplasma] %s|%s: regime=%s infection=%.2f [%s] "
            "dopamine=%.2f size_mult=%.2f conf_offset=%.3f%s",
            strategy_id, symbol, regime.value,
            infection.infection_score, infection.infection_phase,
            dopamine.current_level,
            measures.position_size_mult, measures.confidence_offset,
            " INVERTED" if measures.signal_inversion else "",
        )

        return result

    # ----------------------------------------------------------
    # SIGNAL FILE OUTPUT (for BRAIN script consumption)
    # ----------------------------------------------------------

    def write_signal_file(
        self,
        result: Dict,
        signal_file: str = None,
    ):
        """
        Write Toxoplasma status to JSON for BRAIN script consumption.

        The BRAIN script reads this file each cycle and applies the
        countermeasures to its trading decisions.
        """
        if signal_file is None:
            signal_file = str(
                Path(__file__).parent / "toxoplasma_signal.json"
            )

        try:
            tmp_path = signal_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(result, f, indent=2)
            os.replace(tmp_path, signal_file)
        except Exception as e:
            log.warning("[Toxoplasma] Failed to write signal file: %s", e)

    # ----------------------------------------------------------
    # QUERY METHODS
    # ----------------------------------------------------------

    def get_infection_status(self, strategy_id: str,
                             symbol: str) -> Optional[InfectionScore]:
        """Get the latest infection score for a strategy/symbol pair."""
        key = f"{strategy_id}|{symbol}"
        return self._latest_scores.get(key)

    def get_countermeasures(self, strategy_id: str,
                            symbol: str) -> Optional[CountermeasureSet]:
        """Get the latest countermeasures for a strategy/symbol pair."""
        key = f"{strategy_id}|{symbol}"
        return self._latest_countermeasures.get(key)

    def get_all_registered_strategies(self) -> List[str]:
        """Get all registered strategy|symbol keys."""
        return list(self._baselines.keys())

    def get_infection_history(self, strategy_id: str, symbol: str,
                              hours: int = 24) -> List[Dict]:
        """Get infection score history for a strategy over the last N hours."""
        results = []
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM infection_log
                    WHERE strategy_id = ? AND symbol = ?
                      AND timestamp > ?
                    ORDER BY timestamp ASC
                """, (strategy_id, symbol, cutoff))
                for row in cursor.fetchall():
                    results.append(dict(row))
        except Exception as e:
            log.warning("[Toxoplasma] Failed to get infection history: %s", e)
        return results

    def get_summary(self) -> Dict:
        """Get a summary of all monitored strategies and their infection status."""
        summary = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "strategies": {},
            "manipulation_tools": len(self.get_manipulation_tools()),
        }
        for key, baseline in self._baselines.items():
            infection = self._latest_scores.get(key)
            measures = self._latest_countermeasures.get(key)
            summary["strategies"][key] = {
                "strategy_type": baseline.strategy_type,
                "calibrated": baseline.calibration_valid,
                "baseline_wr": baseline.baseline_win_rate,
                "baseline_trades": baseline.total_trades,
                "infection_score": infection.infection_score if infection else 0.0,
                "infection_phase": infection.infection_phase if infection else "healthy",
                "position_size_mult": measures.position_size_mult if measures else 1.0,
                "signal_inversion": measures.signal_inversion if measures else False,
                "strategy_pause": measures.strategy_pause if measures else False,
            }
        return summary


# ============================================================
# TEQA INTEGRATION: Toxoplasma Bridge
# ============================================================

class ToxoplasmaTEQABridge:
    """
    Bridges the Toxoplasma engine with the TEQA v3.0 pipeline.

    The bridge:
      1. Feeds infection context into TEQA confidence adjustments
      2. Reports manipulation tools to TE Domestication tracker
      3. Provides a unified interface for BRAIN scripts
      4. Manages the signal file lifecycle

    This is how the anti-parasitic immune response integrates with
    the broader transposon ecosystem.
    """

    def __init__(
        self,
        engine: ToxoplasmaEngine,
        signal_file: str = None,
    ):
        self.engine = engine
        if signal_file is None:
            self.signal_file = str(
                Path(__file__).parent / "toxoplasma_signal.json"
            )
        else:
            self.signal_file = signal_file

    def get_toxoplasma_gate_result(
        self,
        strategy_id: str,
        symbol: str,
        bars: np.ndarray,
        active_tes: List[str] = None,
    ) -> Dict:
        """
        Compute Toxoplasma gate result for integration with Jardine's Gate.

        This becomes a modifier gate in the extended gate system.
        Unlike binary pass/fail gates, Toxoplasma provides CONTINUOUS
        modulation of confidence and position sizing.

        Returns:
            {
                "gate_pass": bool (False only if strategy_pause is True),
                "confidence_modifier": float (added to threshold),
                "size_modifier": float (multiplied with position size),
                "signal_inversion": bool,
                "infection_score": float,
                "infection_phase": str,
                "regime": str,
            }
        """
        result = self.engine.run_cycle(
            strategy_id, symbol, bars, active_tes,
        )

        gate_pass = not result.get("strategy_pause", False)

        return {
            "gate_pass": gate_pass,
            "confidence_modifier": result.get("confidence_offset", 0.0),
            "size_modifier": result.get("position_size_mult", 1.0),
            "signal_inversion": result.get("signal_inversion", False),
            "infection_score": result.get("infection_score", 0.0),
            "infection_phase": result.get("infection_phase", "healthy"),
            "regime": result.get("regime", "ranging"),
            "dopamine_level": result.get("dopamine_level", 0.0),
            "hold_time_mult": result.get("hold_time_mult", 1.0),
            "active_measures": result.get("active_measures", []),
        }

    def apply_to_signal(
        self,
        strategy_id: str,
        symbol: str,
        original_direction: int,
        original_confidence: float,
        bars: np.ndarray,
        active_tes: List[str] = None,
    ) -> Tuple[int, float, float]:
        """
        Apply Toxoplasma modulation to a trading signal.

        This is the primary integration point for BRAIN scripts.
        It takes the raw signal and returns the modified version
        after applying all countermeasures.

        Args:
            strategy_id: strategy identifier
            symbol: trading instrument
            original_direction: 1 (long) or -1 (short)
            original_confidence: raw confidence score
            bars: OHLCV data
            active_tes: current TE activations

        Returns:
            (modified_direction, modified_confidence, position_size_mult)

        Usage in BRAIN:
            direction, confidence, size_mult = bridge.apply_to_signal(
                "teqa_main", "BTCUSD", signal_dir, signal_conf, bars, tes
            )
            # confidence is already adjusted -- compare directly to CONFIDENCE_THRESHOLD
            # size_mult is applied to lot calculation
        """
        gate = self.get_toxoplasma_gate_result(
            strategy_id, symbol, bars, active_tes,
        )

        if not gate["gate_pass"]:
            # Strategy paused -- return zero confidence to block trade
            return 0, 0.0, 0.0

        modified_direction = original_direction
        if gate["signal_inversion"]:
            modified_direction = -original_direction

        # Confidence offset is SUBTRACTED from signal confidence
        # (higher offset = harder to meet threshold)
        modified_confidence = original_confidence - gate["confidence_modifier"]

        size_mult = gate["size_modifier"]

        return modified_direction, modified_confidence, size_mult

    def get_manipulation_suppression_list(self) -> List[Dict]:
        """
        Get TE patterns that should receive domestication penalties.

        This is fed to the TE Domestication tracker:
          For each manipulation tool TE combo:
            domestication_db.SUPPRESS(pattern, penalty=0.15)

        Prevents the system from rewarding patterns that are associated
        with regime manipulation rather than genuine market edge.
        """
        return self.engine.get_manipulation_tools()

    def write_status(
        self,
        strategy_id: str,
        symbol: str,
        bars: np.ndarray,
        active_tes: List[str] = None,
    ):
        """
        Run a full cycle and write the result to the signal file.
        Convenience method for BRAIN script integration.
        """
        result = self.engine.run_cycle(
            strategy_id, symbol, bars, active_tes,
        )
        self.engine.write_signal_file(result, self.signal_file)
        return result


# ============================================================
# STANDALONE TEST / DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 76)
    print("  TOXOPLASMA ENGINE -- Regime Behavior Modification")
    print("  Detecting market hijacking of trading strategies")
    print("=" * 76)

    # Generate synthetic OHLCV data with regime shifts
    np.random.seed(42)
    n_bars = 600

    # Phase 1 (0-200): trending up
    # Phase 2 (200-400): ranging (whipsaw zone)
    # Phase 3 (400-600): volatile breakdown
    returns = np.zeros(n_bars)
    for i in range(n_bars):
        if i < 200:
            returns[i] = np.random.randn() * 0.005 + 0.001  # Uptrend
        elif i < 400:
            returns[i] = np.random.randn() * 0.003  # Range
        else:
            returns[i] = np.random.randn() * 0.015 - 0.001  # Volatile down

    close = 50000 + np.cumsum(returns) * 50000
    close = np.maximum(close, 100)
    high = close + np.abs(np.random.randn(n_bars) * close * 0.003)
    low = close - np.abs(np.random.randn(n_bars) * close * 0.003)
    open_p = close + np.random.randn(n_bars) * close * 0.001
    volume = np.abs(np.random.randn(n_bars) * 100 + 500)
    # Volume spikes during volatile phase
    volume[400:] *= 2.5
    bars = np.column_stack([open_p, high, low, close, volume])

    # Use temp DB for test
    test_db = str(Path(__file__).parent / "test_toxoplasma.db")

    print(f"\n  Synthetic data: {n_bars} bars")
    print("  Phase 1 (0-200):   Trending up")
    print("  Phase 2 (200-400): Ranging (whipsaw zone)")
    print("  Phase 3 (400-600): Volatile breakdown")

    # Initialize engine
    engine = ToxoplasmaEngine(db_path=test_db)

    # Register a momentum strategy
    engine.register_strategy("test_momentum", StrategyType.MOMENTUM, "BTCUSD")

    # Simulate trade history during trending phase (healthy behavior)
    print("\n  --- PHASE 1: Building Healthy Baseline (Trending) ---")
    for i in range(50):
        won = np.random.random() < 0.62  # 62% win rate during trend
        regime = "trending_up"
        ts = (datetime(2026, 2, 1) + timedelta(hours=i)).isoformat()
        trade = TradeRecord(
            ticket=1000 + i,
            symbol="BTCUSD",
            direction=1,
            profit=1.50 if won else -0.80,
            hold_time_bars=float(np.random.randint(5, 20)),
            regime=regime,
            timestamp=ts,
            active_tes=["BEL_Pao", "DIRS1", "LINE"],
        )
        engine.record_trade("test_momentum", trade)

    # Calibrate baseline
    baseline = engine.calibrate_baseline("test_momentum", "BTCUSD")
    print(f"  Baseline WR:        {baseline.baseline_win_rate:.1%}")
    print(f"  Baseline avg hold:  {baseline.baseline_avg_hold:.1f} bars")
    print(f"  Baseline avg P/L:   ${baseline.baseline_avg_pnl:.4f}")
    print(f"  Valid:              {baseline.calibration_valid}")

    # Test infection detection during trending regime (should be healthy)
    print("\n  --- PHASE 2: Healthy Check (Trending Regime) ---")
    result = engine.run_cycle("test_momentum", "BTCUSD", bars[:200])
    print(f"  Regime:             {result['regime']}")
    print(f"  Infection score:    {result['infection_score']:.3f}")
    print(f"  Infection phase:    {result['infection_phase']}")
    print(f"  Dopamine level:     {result['dopamine_level']:.3f}")
    print(f"  Position size mult: {result['position_size_mult']:.2f}")
    print(f"  Confidence offset:  {result['confidence_offset']:.3f}")

    # Now simulate trades during ranging phase (strategy gets infected)
    print("\n  --- PHASE 3: Infection Period (Ranging Regime) ---")
    for i in range(25):
        won = np.random.random() < 0.35  # 35% WR -- momentum fails in range
        ts = (datetime(2026, 2, 5) + timedelta(hours=i)).isoformat()
        trade = TradeRecord(
            ticket=2000 + i,
            symbol="BTCUSD",
            direction=1,
            profit=1.20 if won else -1.00,
            hold_time_bars=float(np.random.randint(2, 8)),
            regime="ranging",
            timestamp=ts,
            active_tes=["Mariner_Tc1", "Mutator", "RTE"],
        )
        engine.record_trade("test_momentum", trade)

    # Run cycle during ranging period
    result = engine.run_cycle("test_momentum", "BTCUSD", bars[200:400])
    print(f"  Regime:             {result['regime']}")
    print(f"  Infection score:    {result['infection_score']:.3f}")
    print(f"  Infection phase:    {result['infection_phase']}")
    print(f"  Dopamine level:     {result['dopamine_level']:.3f}")
    print(f"  Position size mult: {result['position_size_mult']:.2f}")
    print(f"  Confidence offset:  {result['confidence_offset']:.3f}")
    print(f"  Signal inversion:   {result['signal_inversion']}")
    print(f"  Active measures:    {result['active_measures']}")
    print(f"  Components:")
    for k, v in result["components"].items():
        print(f"    {k:25s}: {v:.3f}")

    # Test during volatile breakdown (high dopamine)
    print("\n  --- PHASE 4: Volatile Breakdown (High Dopamine) ---")
    for i in range(20):
        won = np.random.random() < 0.25  # 25% WR -- everything fails
        ts = (datetime(2026, 2, 7) + timedelta(hours=i)).isoformat()
        trade = TradeRecord(
            ticket=3000 + i,
            symbol="BTCUSD",
            direction=1,
            profit=0.80 if won else -1.00,
            hold_time_bars=float(np.random.randint(1, 5)),
            regime="volatile",
            timestamp=ts,
            active_tes=["VIPER_Ngaro", "Alu", "SINE"],
        )
        engine.record_trade("test_momentum", trade)

    result = engine.run_cycle("test_momentum", "BTCUSD", bars[400:])
    print(f"  Regime:             {result['regime']}")
    print(f"  Infection score:    {result['infection_score']:.3f}")
    print(f"  Infection phase:    {result['infection_phase']}")
    print(f"  Dopamine level:     {result['dopamine_level']:.3f}")
    print(f"  Position size mult: {result['position_size_mult']:.2f}")
    print(f"  Confidence offset:  {result['confidence_offset']:.3f}")
    print(f"  Signal inversion:   {result['signal_inversion']}")
    print(f"  Strategy pause:     {result['strategy_pause']}")
    print(f"  Active measures:    {result['active_measures']}")

    # Test the bridge integration
    print("\n  --- PHASE 5: TEQA Bridge Integration ---")
    bridge = ToxoplasmaTEQABridge(engine)

    # Apply to a hypothetical BUY signal with 0.35 confidence
    direction, confidence, size_mult = bridge.apply_to_signal(
        "test_momentum", "BTCUSD",
        original_direction=1,
        original_confidence=0.35,
        bars=bars[400:],
        active_tes=["VIPER_Ngaro", "Alu"],
    )
    print(f"  Original signal:    BUY @ confidence 0.35")
    dir_str = "BUY" if direction > 0 else ("SELL" if direction < 0 else "NONE")
    print(f"  Modified signal:    {dir_str} @ confidence {confidence:.3f}")
    print(f"  Size multiplier:    {size_mult:.2f}")
    print(f"  CONFIDENCE_THRESHOLD (from config_loader): {CONFIDENCE_THRESHOLD}")
    would_trade = confidence >= CONFIDENCE_THRESHOLD and size_mult > 0
    print(f"  Would trade?        {'YES' if would_trade else 'NO'}")

    # Get manipulation tools
    print("\n  --- PHASE 6: TE Manipulation Tools ---")
    tools = engine.get_manipulation_tools()
    if tools:
        print(f"  Identified {len(tools)} manipulation tool patterns:")
        for tool in tools:
            print(
                f"    {tool['te_combo'][:40]:40s} | "
                f"corr={tool['infection_correlation']:.2f} "
                f"wr={tool['infection_wr']:.2f}"
            )
    else:
        print("  No manipulation tools identified yet (need more data)")

    # Get summary
    print("\n  --- SUMMARY ---")
    summary = engine.get_summary()
    for key, info in summary["strategies"].items():
        print(f"  Strategy: {key}")
        print(f"    Type:            {info['strategy_type']}")
        print(f"    Calibrated:      {info['calibrated']}")
        print(f"    Baseline WR:     {info['baseline_wr']:.1%}")
        print(f"    Infection score: {info['infection_score']:.3f}")
        print(f"    Infection phase: {info['infection_phase']}")
        print(f"    Size mult:       {info['position_size_mult']:.2f}")
        print(f"    Inversion:       {info['signal_inversion']}")
        print(f"    Paused:          {info['strategy_pause']}")

    # Write signal file
    signal_path = str(Path(__file__).parent / "test_toxoplasma_signal.json")
    engine.write_signal_file(result, signal_path)
    print(f"\n  Signal file written to: {signal_path}")

    # Cleanup test files
    try:
        os.remove(test_db)
    except OSError:
        pass
    try:
        os.remove(signal_path)
    except OSError:
        pass

    print("\n" + "=" * 76)
    print("  Toxoplasma Engine test complete.")
    print("  The system that survives is the one that knows when it is infected.")
    print("=" * 76)
