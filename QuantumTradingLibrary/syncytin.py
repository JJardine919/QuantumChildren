"""
SYNCYTIN FUSION ENGINE -- Strategy Fusion into Hybrid Organisms
================================================================
Algorithm #8 in the Quantum Children biological algorithm series.

Domesticated retroviral envelope protein -> placental cell fusion.

Biological basis:
    Syncytin is a retroviral env gene domesticated by mammals to fuse
    trophoblast cells into the syncytiotrophoblast -- the outer layer
    of the placenta. A mechanism designed for viral INVASION was
    repurposed for NURTURING. The fusion creates a barrier that protects
    the fetus from the mother's immune system while allowing nutrient
    exchange. This happened INDEPENDENTLY in multiple mammal lineages.

Trading translation:
    Strategy A = wins in trends, bleeds in ranges
    Strategy B = wins in ranges, bleeds in trends
    Syncytin FUSES them into a hybrid organism that routes to the
    appropriate sub-strategy based on detected market regime, while
    maintaining a shared risk envelope (the membrane) and compartmentalized
    drawdown limits (the immune barrier).

    The hybrid must outperform either parent alone. If it does not,
    the fusion is dissolved (defusion = spontaneous abortion).

Integration:
    - Reads VDJ memory cells (Algorithm #1) as fusion candidates
    - Reads TE domestication records (Algorithm #2) for fitness data
    - Stores fusion records in SQLite (syncytin_fusions.db)
    - Emits hybrid signals to JSON for BRAIN script consumption
    - All trading values sourced from config_loader (no hardcoded values)

Authors: DooDoo + Claude
Date:    2026-02-08
Version: SYNCYTIN-1.0
"""

import json
import hashlib
import logging
import math
import os
import sqlite3
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np

# All trading values from config_loader -- never hardcoded
from config_loader import (
    MAX_LOSS_DOLLARS,
    CONFIDENCE_THRESHOLD,
)

log = logging.getLogger(__name__)

VERSION = "SYNCYTIN-1.0"

# ============================================================
# CONSTANTS (algorithm-specific, NOT trading values)
# ============================================================

# Fusion candidate screening
MIN_PARENT_TRADES = 20
MIN_PARENT_WIN_RATE = 0.55
MAX_RETURN_CORRELATION = 0.30
MIN_REGIME_COMPLEMENTARITY = 0.40

# Compatibility scoring (receptor ASCT2 check)
COMPATIBILITY_THRESHOLD = 0.60

# Envelope protein (shared risk management)
# NOTE: max SL is read from config_loader.MAX_LOSS_DOLLARS at runtime
ENVELOPE_MAX_DRAWDOWN_PCT = 0.05
ENVELOPE_POSITION_LIMIT = 2

# Syncytiotrophoblast routing
REGIME_DETECTION_LOOKBACK = 50
REGIME_SWITCH_COOLDOWN = 5
EQUITY_SHARING_RATE = 0.20

# Immune barrier (compartmentalized risk)
COMPARTMENT_MAX_BUDGET_PCT = 0.60
COMPARTMENT_KILL_THRESHOLD = 0.03
COMPARTMENT_RESUME_RATIO = 0.50

# Fitness monitoring
FUSION_MIN_TRADES = 30
FUSION_MIN_IMPROVEMENT = 0.05
DEFUSION_DEGRADATION_PCT = -0.10
FUSION_REEVAL_INTERVAL = 50

# Population limits
MAX_ACTIVE_FUSIONS = 10
MAX_FUSION_CANDIDATES = 50

# Bayesian prior for hybrid win rate estimation
PRIOR_ALPHA = 8
PRIOR_BETA = 8


# ============================================================
# ENUMS
# ============================================================

class RegimeType(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    COMPRESSED = "compressed"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


class FusionType(Enum):
    """How two strategies are fused."""
    REGIME_SWITCH = "regime_switch"       # Pure routing by regime
    WEIGHTED_BLEND = "weighted_blend"     # Weighted average of both signals
    CASCADE = "cascade"                   # A proposes, B confirms


class HybridStatus(Enum):
    """Lifecycle status of a hybrid organism."""
    ACTIVE = "active"
    PROBATION = "probation"
    DEFUSED = "defused"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class StrategyProfile:
    """
    Profile of a single strategy that can participate in fusion.

    This can be sourced from:
      - VDJ memory B cells (Algorithm #1)
      - TE domestication records (Algorithm #2)
      - Custom strategy definitions
    """
    strategy_id: str
    strategy_type: str = "custom"         # "vdj_antibody" | "te_pattern" | "custom"
    source_id: str = ""                   # antibody_id or pattern_hash

    # Regime affinity: how well this strategy performs in each regime
    # Values 0.0 to 1.0 representing win rate or fitness in that regime
    regime_affinity: Dict[str, float] = field(default_factory=lambda: {
        "trending_up": 0.5,
        "trending_down": 0.5,
        "ranging": 0.5,
        "volatile": 0.5,
        "compressed": 0.5,
        "breakout": 0.5,
    })

    # Fitness metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    fitness: float = 0.0

    # Historical return stream (for correlation analysis)
    return_stream: List[float] = field(default_factory=list)

    # Per-regime trade counts
    regime_wins: Dict[str, int] = field(default_factory=dict)
    regime_losses: Dict[str, int] = field(default_factory=dict)

    # Signal generation callback (set at runtime, not persisted)
    _signal_fn: Optional[Callable] = field(default=None, repr=False)

    def generate_signal(self, bars: np.ndarray) -> Dict:
        """Generate a trading signal from this strategy."""
        if self._signal_fn is not None:
            return self._signal_fn(bars)
        return {"direction": 0, "confidence": 0.0}


@dataclass
class FusionCandidate:
    """
    A screened pair of strategies evaluated for fusion compatibility.

    The compatibility score is the "receptor match" -- like syncytin
    binding to ASCT2. Without receptor compatibility, fusion cannot occur.
    """
    strategy_a_id: str
    strategy_b_id: str
    compatibility: float = 0.0
    return_correlation: float = 0.0
    regime_complementarity: float = 0.0
    drawdown_overlap: float = 0.0
    frequency_ratio: float = 0.0
    receptor_match: bool = False
    screened_at: str = ""

    def to_dict(self) -> Dict:
        return {
            "strategy_a_id": self.strategy_a_id,
            "strategy_b_id": self.strategy_b_id,
            "compatibility": self.compatibility,
            "return_correlation": self.return_correlation,
            "regime_complementarity": self.regime_complementarity,
            "drawdown_overlap": self.drawdown_overlap,
            "frequency_ratio": self.frequency_ratio,
            "receptor_match": self.receptor_match,
            "screened_at": self.screened_at,
        }


@dataclass
class RiskCompartment:
    """
    Compartmentalized risk for one sub-strategy within a hybrid.

    This is the IMMUNE BARRIER -- prevents one sub-strategy's losses
    from killing the entire hybrid organism. Like the placental barrier
    preventing maternal immune cells from attacking fetal tissue.
    """
    strategy_id: str
    budget_pct: float = 0.50          # Fraction of total risk budget
    current_dd: float = 0.0           # Current drawdown in this compartment
    peak_equity: float = 0.0          # Peak equity for drawdown calculation
    equity_balance: float = 0.0       # Running equity for this compartment
    suspended: bool = False           # TRUE = compartment hit kill threshold
    trades_since_susp: int = 0        # Trades since last suspension
    total_trades: int = 0
    wins: int = 0
    losses: int = 0

    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades


@dataclass
class EnvelopeProtein:
    """
    Shared risk management layer wrapping both sub-strategies.

    This is the syncytin envelope -- the fused membrane that creates
    a unified organism from two distinct cells. It enforces:
      - Combined position limit
      - Combined drawdown limit
      - SL from config_loader (never hardcoded)
    """
    max_sl_dollars: float = 0.0       # From config_loader at init time
    max_drawdown_pct: float = ENVELOPE_MAX_DRAWDOWN_PCT
    position_limit: int = ENVELOPE_POSITION_LIMIT
    current_positions: int = 0
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    total_equity: float = 0.0

    def can_trade(self) -> bool:
        """Check if the envelope allows a new trade."""
        if self.current_positions >= self.position_limit:
            return False
        if self.current_drawdown > self.max_drawdown_pct:
            return False
        return True

    def update_equity(self, pnl: float):
        """Update envelope equity tracking."""
        self.total_equity += pnl
        if self.total_equity > self.peak_equity:
            self.peak_equity = self.total_equity
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.total_equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0


@dataclass
class HybridOrganism:
    """
    A fused hybrid of two complementary strategies.

    This is the syncytiotrophoblast -- a multinucleated organism formed
    by cell-cell fusion. It has:
      - Two nuclei (strategy A and strategy B signal logic)
      - One shared membrane (envelope protein = risk management)
      - Compartmentalized risk (immune barrier)
      - Regime-based routing (nutrient transport selectivity)
    """
    hybrid_id: str
    strategy_a_id: str
    strategy_b_id: str
    fusion_type: str = FusionType.REGIME_SWITCH.value

    # Shared membrane
    envelope: EnvelopeProtein = field(default_factory=EnvelopeProtein)

    # Immune barrier
    compartment_a: RiskCompartment = field(default_factory=lambda: RiskCompartment(""))
    compartment_b: RiskCompartment = field(default_factory=lambda: RiskCompartment(""))

    # Regime routing state
    current_regime: str = RegimeType.UNKNOWN.value
    active_strategy: str = "BOTH"
    regime_switch_bar: int = 0

    # Fitness tracking
    hybrid_trades: int = 0
    hybrid_wins: int = 0
    hybrid_losses: int = 0
    hybrid_pnl: float = 0.0
    hybrid_returns: List[float] = field(default_factory=list)
    parent_a_solo_returns: List[float] = field(default_factory=list)
    parent_b_solo_returns: List[float] = field(default_factory=list)
    fusion_alpha: float = 0.0

    # Posterior win rate (Bayesian)
    posterior_wr: float = 0.5

    # Status
    status: str = HybridStatus.ACTIVE.value
    created_at: str = ""
    last_evaluated: str = ""
    last_trade_at: str = ""

    def compute_id(self, a_id: str, b_id: str, ftype: str) -> str:
        """Generate unique hash from the fusion components."""
        # Canonical ordering so A+B == B+A
        ids = sorted([a_id, b_id])
        raw = f"{ids[0]}|{ids[1]}|{ftype}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def win_rate(self) -> float:
        if self.hybrid_trades == 0:
            return 0.0
        return self.hybrid_wins / self.hybrid_trades

    def profit_factor(self) -> float:
        wins_sum = sum(r for r in self.hybrid_returns if r > 0)
        loss_sum = abs(sum(r for r in self.hybrid_returns if r <= 0))
        if loss_sum == 0:
            return 99.0 if wins_sum > 0 else 0.0
        return wins_sum / loss_sum

    def sharpe(self) -> float:
        if len(self.hybrid_returns) < 2:
            return 0.0
        arr = np.array(self.hybrid_returns)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        return float(np.mean(arr) / std * np.sqrt(252))

    def summary(self) -> str:
        return (
            f"[{self.hybrid_id[:8]}] {self.fusion_type} | "
            f"A={self.strategy_a_id[:8]} B={self.strategy_b_id[:8]} | "
            f"WR={self.win_rate():.1%} PF={self.profit_factor():.2f} "
            f"trades={self.hybrid_trades} alpha={self.fusion_alpha:+.2%} "
            f"status={self.status}"
        )


# ============================================================
# REGIME DETECTION ENGINE
# ============================================================

class RegimeDetector:
    """
    Classifies the current market regime from price bars.

    The regime determines which sub-strategy gets control of the hybrid.
    This is the selectivity mechanism of the syncytiotrophoblast --
    like how the placenta selectively transports specific nutrients
    depending on fetal needs.
    """

    @staticmethod
    def detect(
        bars: np.ndarray,
        lookback: int = REGIME_DETECTION_LOOKBACK,
    ) -> RegimeType:
        """
        Classify market regime from OHLCV bars.

        Args:
            bars: numpy array of shape (N, >=4) with [open, high, low, close, ...]
            lookback: number of bars to analyze

        Returns:
            RegimeType enum value
        """
        if len(bars) < lookback:
            return RegimeType.UNKNOWN

        close = bars[-lookback:, 3]
        high = bars[-lookback:, 1]
        low = bars[-lookback:, 2]

        n = len(close)
        if n < 20:
            return RegimeType.UNKNOWN

        # ATR ratio: recent vs baseline
        atr_recent = RegimeDetector._atr_window(high[-5:], low[-5:], close[-5:])
        atr_full = RegimeDetector._atr_window(high, low, close)
        atr_ratio = atr_recent / (atr_full + 1e-10)

        # EMA spread for trend detection
        ema_fast = RegimeDetector._ema_val(close, 8)
        ema_slow = RegimeDetector._ema_val(close, 21)
        ema_diff = (ema_fast - ema_slow) / (ema_slow + 1e-10)

        # Bollinger width for ranging/compression
        sma_20 = float(np.mean(close[-20:]))
        std_20 = float(np.std(close[-20:]))
        bb_width = (2 * std_20) / (sma_20 + 1e-10)

        # Classification hierarchy
        if atr_ratio < 0.6:
            return RegimeType.COMPRESSED

        if atr_ratio > 1.5:
            # Check if this was preceded by compression (= breakout)
            if n >= 10:
                atr_prev = RegimeDetector._atr_window(
                    high[-10:-5], low[-10:-5], close[-10:-5]
                )
                if atr_full > 0 and atr_prev / (atr_full + 1e-10) < 0.7:
                    return RegimeType.BREAKOUT
            return RegimeType.VOLATILE

        if abs(ema_diff) > 0.005:
            if ema_diff > 0:
                return RegimeType.TRENDING_UP
            else:
                return RegimeType.TRENDING_DOWN

        if bb_width < 0.02:
            return RegimeType.RANGING

        return RegimeType.RANGING

    @staticmethod
    def _atr_window(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Compute average true range over a window."""
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

    @staticmethod
    def _ema_val(data: np.ndarray, period: int) -> float:
        """Compute single EMA value at the end of the array."""
        if len(data) < period:
            return float(np.mean(data))
        mult = 2.0 / (period + 1)
        ema = float(np.mean(data[:period]))
        for val in data[period:]:
            ema = (float(val) - ema) * mult + ema
        return ema

    @staticmethod
    def compute_regime_affinity(
        return_stream: List[float],
        regime_labels: List[str],
    ) -> Dict[str, float]:
        """
        Compute regime affinity from a return stream and corresponding regime labels.

        For each regime, computes the win rate of the strategy when that regime
        was active. This tells us which regimes the strategy is "good at."

        Args:
            return_stream: list of trade PnL values
            regime_labels: list of regime labels, one per trade

        Returns:
            Dict mapping regime name to win rate in that regime
        """
        affinity = {}
        for regime in RegimeType:
            rname = regime.value
            trades_in_regime = [
                r for r, lbl in zip(return_stream, regime_labels)
                if lbl == rname
            ]
            if len(trades_in_regime) >= 3:
                wins = sum(1 for r in trades_in_regime if r > 0)
                affinity[rname] = wins / len(trades_in_regime)
            else:
                affinity[rname] = 0.5  # Prior: uncertain
        return affinity


# ============================================================
# SYNCYTIN FUSION ENGINE
# ============================================================

class SyncytinFusionEngine:
    """
    Core engine for strategy fusion.

    This is the syncytin protein equivalent. It:
      1. Screens strategy pairs for fusion compatibility (receptor check)
      2. Fuses compatible pairs into hybrid organisms
      3. Routes signals through the hybrid based on market regime
      4. Manages compartmentalized risk (immune barrier)
      5. Monitors fusion fitness and triggers defusion if degraded
      6. Persists all state in SQLite for learning across sessions
    """

    def __init__(
        self,
        db_path: str = None,
        strategy_pool: List[StrategyProfile] = None,
    ):
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "syncytin_fusions.db")
        else:
            self.db_path = db_path

        # Strategy pool (populated by load or external injection)
        self.strategy_pool: Dict[str, StrategyProfile] = {}
        if strategy_pool:
            for sp in strategy_pool:
                self.strategy_pool[sp.strategy_id] = sp

        # Active hybrid organisms
        self.hybrids: Dict[str, HybridOrganism] = {}

        # Screened candidates
        self.candidates: List[FusionCandidate] = []

        # Regime detector
        self.regime_detector = RegimeDetector()

        # Initialize database
        self._init_db()

        # Load existing hybrids
        self._load_hybrids()

    # ----------------------------------------------------------
    # DATABASE
    # ----------------------------------------------------------

    def _init_db(self):
        """Initialize the syncytin fusion database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS hybrid_organisms (
                        hybrid_id TEXT PRIMARY KEY,
                        strategy_a_id TEXT NOT NULL,
                        strategy_b_id TEXT NOT NULL,
                        fusion_type TEXT NOT NULL,
                        status TEXT DEFAULT 'active',

                        -- Envelope protein
                        envelope_max_dd_pct REAL DEFAULT 0.05,
                        envelope_position_limit INTEGER DEFAULT 2,
                        envelope_current_dd REAL DEFAULT 0.0,
                        envelope_peak_equity REAL DEFAULT 0.0,
                        envelope_total_equity REAL DEFAULT 0.0,

                        -- Compartment A
                        comp_a_budget_pct REAL DEFAULT 0.50,
                        comp_a_equity REAL DEFAULT 0.0,
                        comp_a_peak_equity REAL DEFAULT 0.0,
                        comp_a_current_dd REAL DEFAULT 0.0,
                        comp_a_suspended INTEGER DEFAULT 0,
                        comp_a_trades INTEGER DEFAULT 0,
                        comp_a_wins INTEGER DEFAULT 0,
                        comp_a_losses INTEGER DEFAULT 0,

                        -- Compartment B
                        comp_b_budget_pct REAL DEFAULT 0.50,
                        comp_b_equity REAL DEFAULT 0.0,
                        comp_b_peak_equity REAL DEFAULT 0.0,
                        comp_b_current_dd REAL DEFAULT 0.0,
                        comp_b_suspended INTEGER DEFAULT 0,
                        comp_b_trades INTEGER DEFAULT 0,
                        comp_b_wins INTEGER DEFAULT 0,
                        comp_b_losses INTEGER DEFAULT 0,

                        -- Regime state
                        current_regime TEXT DEFAULT 'unknown',
                        active_strategy TEXT DEFAULT 'BOTH',
                        regime_switch_bar INTEGER DEFAULT 0,

                        -- Fitness
                        hybrid_trades INTEGER DEFAULT 0,
                        hybrid_wins INTEGER DEFAULT 0,
                        hybrid_losses INTEGER DEFAULT 0,
                        hybrid_pnl REAL DEFAULT 0.0,
                        fusion_alpha REAL DEFAULT 0.0,
                        posterior_wr REAL DEFAULT 0.5,

                        -- Timestamps
                        created_at TEXT,
                        last_evaluated TEXT,
                        last_trade_at TEXT
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fusion_trades (
                        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        hybrid_id TEXT NOT NULL,
                        active_strategy TEXT NOT NULL,
                        regime TEXT,
                        direction INTEGER,
                        pnl REAL,
                        compartment_equity_after REAL,
                        envelope_equity_after REAL,
                        timestamp TEXT,
                        FOREIGN KEY (hybrid_id) REFERENCES hybrid_organisms(hybrid_id)
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fusion_candidates (
                        candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_a_id TEXT NOT NULL,
                        strategy_b_id TEXT NOT NULL,
                        compatibility REAL,
                        return_correlation REAL,
                        regime_complementarity REAL,
                        drawdown_overlap REAL,
                        frequency_ratio REAL,
                        receptor_match INTEGER,
                        screened_at TEXT
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fusion_fitness_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        hybrid_id TEXT NOT NULL,
                        evaluation_time TEXT,
                        hybrid_trades INTEGER,
                        hybrid_pnl REAL,
                        solo_a_pnl REAL,
                        solo_b_pnl REAL,
                        fusion_alpha REAL,
                        status_before TEXT,
                        status_after TEXT,
                        FOREIGN KEY (hybrid_id) REFERENCES hybrid_organisms(hybrid_id)
                    )
                """)

                # Index for fast lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_fusion_trades_hybrid
                    ON fusion_trades(hybrid_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_hybrid_status
                    ON hybrid_organisms(status)
                """)

                conn.commit()
        except Exception as e:
            log.warning("Syncytin DB init failed: %s", e)

    def _load_hybrids(self):
        """Load active hybrid organisms from database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM hybrid_organisms
                    WHERE status IN ('active', 'probation')
                    ORDER BY fusion_alpha DESC
                    LIMIT ?
                """, (MAX_ACTIVE_FUSIONS,))

                for row in cursor.fetchall():
                    hybrid = self._row_to_hybrid(row)
                    self.hybrids[hybrid.hybrid_id] = hybrid

                log.info(
                    "Loaded %d active hybrid organisms from database",
                    len(self.hybrids),
                )
        except Exception as e:
            log.warning("Failed to load hybrids: %s", e)

    def _row_to_hybrid(self, row) -> HybridOrganism:
        """Convert a database row to a HybridOrganism."""
        envelope = EnvelopeProtein(
            max_sl_dollars=MAX_LOSS_DOLLARS,
            max_drawdown_pct=row["envelope_max_dd_pct"] or ENVELOPE_MAX_DRAWDOWN_PCT,
            position_limit=row["envelope_position_limit"] or ENVELOPE_POSITION_LIMIT,
            current_drawdown=row["envelope_current_dd"] or 0.0,
            peak_equity=row["envelope_peak_equity"] or 0.0,
            total_equity=row["envelope_total_equity"] or 0.0,
        )

        comp_a = RiskCompartment(
            strategy_id=row["strategy_a_id"],
            budget_pct=row["comp_a_budget_pct"] or 0.5,
            equity_balance=row["comp_a_equity"] or 0.0,
            peak_equity=row["comp_a_peak_equity"] or 0.0,
            current_dd=row["comp_a_current_dd"] or 0.0,
            suspended=bool(row["comp_a_suspended"]),
            total_trades=row["comp_a_trades"] or 0,
            wins=row["comp_a_wins"] or 0,
            losses=row["comp_a_losses"] or 0,
        )

        comp_b = RiskCompartment(
            strategy_id=row["strategy_b_id"],
            budget_pct=row["comp_b_budget_pct"] or 0.5,
            equity_balance=row["comp_b_equity"] or 0.0,
            peak_equity=row["comp_b_peak_equity"] or 0.0,
            current_dd=row["comp_b_current_dd"] or 0.0,
            suspended=bool(row["comp_b_suspended"]),
            total_trades=row["comp_b_trades"] or 0,
            wins=row["comp_b_wins"] or 0,
            losses=row["comp_b_losses"] or 0,
        )

        hybrid = HybridOrganism(
            hybrid_id=row["hybrid_id"],
            strategy_a_id=row["strategy_a_id"],
            strategy_b_id=row["strategy_b_id"],
            fusion_type=row["fusion_type"],
            envelope=envelope,
            compartment_a=comp_a,
            compartment_b=comp_b,
            current_regime=row["current_regime"] or RegimeType.UNKNOWN.value,
            active_strategy=row["active_strategy"] or "BOTH",
            regime_switch_bar=row["regime_switch_bar"] or 0,
            hybrid_trades=row["hybrid_trades"] or 0,
            hybrid_wins=row["hybrid_wins"] or 0,
            hybrid_losses=row["hybrid_losses"] or 0,
            hybrid_pnl=row["hybrid_pnl"] or 0.0,
            fusion_alpha=row["fusion_alpha"] or 0.0,
            posterior_wr=row["posterior_wr"] or 0.5,
            status=row["status"] or HybridStatus.ACTIVE.value,
            created_at=row["created_at"] or "",
            last_evaluated=row["last_evaluated"] or "",
            last_trade_at=row["last_trade_at"] or "",
        )

        # Load return history from trade log
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn2:
                cursor2 = conn2.cursor()
                cursor2.execute("""
                    SELECT pnl FROM fusion_trades
                    WHERE hybrid_id = ?
                    ORDER BY trade_id ASC
                """, (hybrid.hybrid_id,))
                hybrid.hybrid_returns = [r[0] for r in cursor2.fetchall()]
        except Exception:
            pass

        return hybrid

    def _save_hybrid(self, hybrid: HybridOrganism):
        """Persist a hybrid organism to the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                now = datetime.now().isoformat()
                conn.execute("""
                    INSERT OR REPLACE INTO hybrid_organisms (
                        hybrid_id, strategy_a_id, strategy_b_id, fusion_type, status,
                        envelope_max_dd_pct, envelope_position_limit,
                        envelope_current_dd, envelope_peak_equity, envelope_total_equity,
                        comp_a_budget_pct, comp_a_equity, comp_a_peak_equity,
                        comp_a_current_dd, comp_a_suspended, comp_a_trades,
                        comp_a_wins, comp_a_losses,
                        comp_b_budget_pct, comp_b_equity, comp_b_peak_equity,
                        comp_b_current_dd, comp_b_suspended, comp_b_trades,
                        comp_b_wins, comp_b_losses,
                        current_regime, active_strategy, regime_switch_bar,
                        hybrid_trades, hybrid_wins, hybrid_losses, hybrid_pnl,
                        fusion_alpha, posterior_wr,
                        created_at, last_evaluated, last_trade_at
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?
                    )
                """, (
                    hybrid.hybrid_id,
                    hybrid.strategy_a_id,
                    hybrid.strategy_b_id,
                    hybrid.fusion_type,
                    hybrid.status,
                    hybrid.envelope.max_drawdown_pct,
                    hybrid.envelope.position_limit,
                    hybrid.envelope.current_drawdown,
                    hybrid.envelope.peak_equity,
                    hybrid.envelope.total_equity,
                    hybrid.compartment_a.budget_pct,
                    hybrid.compartment_a.equity_balance,
                    hybrid.compartment_a.peak_equity,
                    hybrid.compartment_a.current_dd,
                    1 if hybrid.compartment_a.suspended else 0,
                    hybrid.compartment_a.total_trades,
                    hybrid.compartment_a.wins,
                    hybrid.compartment_a.losses,
                    hybrid.compartment_b.budget_pct,
                    hybrid.compartment_b.equity_balance,
                    hybrid.compartment_b.peak_equity,
                    hybrid.compartment_b.current_dd,
                    1 if hybrid.compartment_b.suspended else 0,
                    hybrid.compartment_b.total_trades,
                    hybrid.compartment_b.wins,
                    hybrid.compartment_b.losses,
                    hybrid.current_regime,
                    hybrid.active_strategy,
                    hybrid.regime_switch_bar,
                    hybrid.hybrid_trades,
                    hybrid.hybrid_wins,
                    hybrid.hybrid_losses,
                    hybrid.hybrid_pnl,
                    hybrid.fusion_alpha,
                    hybrid.posterior_wr,
                    hybrid.created_at or now,
                    hybrid.last_evaluated or now,
                    hybrid.last_trade_at or "",
                ))
                conn.commit()
        except Exception as e:
            log.warning("Failed to save hybrid %s: %s", hybrid.hybrid_id, e)

    def _save_trade(self, hybrid: HybridOrganism, active: str, regime: str,
                    direction: int, pnl: float):
        """Record a trade in the fusion trade log."""
        try:
            comp = hybrid.compartment_a if active == "A" else hybrid.compartment_b
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO fusion_trades
                    (hybrid_id, active_strategy, regime, direction, pnl,
                     compartment_equity_after, envelope_equity_after, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hybrid.hybrid_id, active, regime, direction, pnl,
                    comp.equity_balance, hybrid.envelope.total_equity,
                    datetime.now().isoformat(),
                ))
                conn.commit()
        except Exception as e:
            log.warning("Failed to save fusion trade: %s", e)

    def _save_candidate(self, candidate: FusionCandidate):
        """Save a screened fusion candidate."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO fusion_candidates
                    (strategy_a_id, strategy_b_id, compatibility,
                     return_correlation, regime_complementarity,
                     drawdown_overlap, frequency_ratio, receptor_match, screened_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    candidate.strategy_a_id,
                    candidate.strategy_b_id,
                    candidate.compatibility,
                    candidate.return_correlation,
                    candidate.regime_complementarity,
                    candidate.drawdown_overlap,
                    candidate.frequency_ratio,
                    1 if candidate.receptor_match else 0,
                    candidate.screened_at,
                ))
                conn.commit()
        except Exception as e:
            log.warning("Failed to save fusion candidate: %s", e)

    def _save_fitness_log(self, hybrid: HybridOrganism, solo_a_pnl: float,
                          solo_b_pnl: float, status_before: str, status_after: str):
        """Record a fusion fitness evaluation."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO fusion_fitness_log
                    (hybrid_id, evaluation_time, hybrid_trades, hybrid_pnl,
                     solo_a_pnl, solo_b_pnl, fusion_alpha,
                     status_before, status_after)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hybrid.hybrid_id,
                    datetime.now().isoformat(),
                    hybrid.hybrid_trades,
                    hybrid.hybrid_pnl,
                    solo_a_pnl,
                    solo_b_pnl,
                    hybrid.fusion_alpha,
                    status_before,
                    status_after,
                ))
                conn.commit()
        except Exception as e:
            log.warning("Failed to save fitness log: %s", e)

    # ----------------------------------------------------------
    # PHASE 1: CANDIDATE SCREENING (receptor compatibility check)
    # ----------------------------------------------------------

    def screen_candidates(
        self,
        strategy_pool: List[StrategyProfile] = None,
    ) -> List[FusionCandidate]:
        """
        Screen strategy pairs for fusion compatibility.

        This is the RECEPTOR CHECK -- like syncytin binding to ASCT2.
        Without receptor compatibility, fusion cannot occur.

        Checks:
          1. Each parent meets minimum quality thresholds
          2. Return streams are weakly or negatively correlated
          3. Regime coverage is complementary (non-overlapping)
          4. Drawdown periods do not overlap heavily
          5. Trade frequencies are reasonably balanced

        Args:
            strategy_pool: list of StrategyProfile objects to screen.
                           Defaults to self.strategy_pool.values()

        Returns:
            List of FusionCandidate objects sorted by compatibility.
        """
        pool = strategy_pool or list(self.strategy_pool.values())

        # Step 1: Filter qualified strategies
        qualified = [
            s for s in pool
            if s.total_trades >= MIN_PARENT_TRADES
            and s.win_rate >= MIN_PARENT_WIN_RATE
        ]

        log.info(
            "[SYNCYTIN] Screening %d qualified strategies from %d total",
            len(qualified), len(pool),
        )

        if len(qualified) < 2:
            log.info("[SYNCYTIN] Need at least 2 qualified strategies for fusion")
            return []

        # Step 2: Generate all unique pairs and score compatibility
        candidates = []
        now_str = datetime.now().isoformat()

        for i in range(len(qualified)):
            for j in range(i + 1, len(qualified)):
                a = qualified[i]
                b = qualified[j]

                # Already fused? Skip
                existing_id = HybridOrganism(
                    hybrid_id="", strategy_a_id=a.strategy_id,
                    strategy_b_id=b.strategy_id
                ).compute_id(a.strategy_id, b.strategy_id, "any")
                # Check all fusion types
                already_fused = False
                for ft in FusionType:
                    check_id = HybridOrganism(
                        hybrid_id="", strategy_a_id="", strategy_b_id=""
                    ).compute_id(a.strategy_id, b.strategy_id, ft.value)
                    if check_id in self.hybrids:
                        already_fused = True
                        break
                if already_fused:
                    continue

                # 2a. Return correlation
                corr = self._return_correlation(a.return_stream, b.return_stream)
                if corr > MAX_RETURN_CORRELATION:
                    continue

                # 2b. Regime complementarity
                regime_comp = self._regime_complementarity(
                    a.regime_affinity, b.regime_affinity
                )
                if regime_comp < MIN_REGIME_COMPLEMENTARITY:
                    continue

                # 2c. Drawdown overlap
                dd_overlap = self._drawdown_overlap(a.return_stream, b.return_stream)

                # 2d. Frequency ratio
                freq_ratio = self._frequency_ratio(a.total_trades, b.total_trades)

                # 2e. Composite compatibility score
                compatibility = (
                    (1.0 - max(0.0, corr)) * 0.35
                    + regime_comp * 0.30
                    + (1.0 - max(0.0, dd_overlap)) * 0.20
                    + freq_ratio * 0.15
                )

                receptor_match = compatibility >= COMPATIBILITY_THRESHOLD

                candidate = FusionCandidate(
                    strategy_a_id=a.strategy_id,
                    strategy_b_id=b.strategy_id,
                    compatibility=compatibility,
                    return_correlation=corr,
                    regime_complementarity=regime_comp,
                    drawdown_overlap=dd_overlap,
                    frequency_ratio=freq_ratio,
                    receptor_match=receptor_match,
                    screened_at=now_str,
                )

                if receptor_match:
                    candidates.append(candidate)
                    self._save_candidate(candidate)

        # Sort by compatibility descending
        candidates.sort(key=lambda c: c.compatibility, reverse=True)
        candidates = candidates[:MAX_FUSION_CANDIDATES]

        self.candidates = candidates

        log.info(
            "[SYNCYTIN] Screened %d compatible fusion candidates",
            len(candidates),
        )

        return candidates

    @staticmethod
    def _return_correlation(stream_a: List[float], stream_b: List[float]) -> float:
        """Compute Pearson correlation between two return streams."""
        if len(stream_a) < 5 or len(stream_b) < 5:
            return 0.0

        # Align lengths
        min_len = min(len(stream_a), len(stream_b))
        a = np.array(stream_a[-min_len:])
        b = np.array(stream_b[-min_len:])

        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0

        corr_matrix = np.corrcoef(a, b)
        return float(corr_matrix[0, 1])

    @staticmethod
    def _regime_complementarity(
        affinity_a: Dict[str, float],
        affinity_b: Dict[str, float],
    ) -> float:
        """
        Compute how complementary two strategies' regime coverage is.

        High complementarity = they cover DIFFERENT regimes well.
        Low complementarity = they are strong in the SAME regimes (redundant).
        """
        regimes = set(affinity_a.keys()) | set(affinity_b.keys())
        if not regimes:
            return 0.0

        overlap_sum = 0.0
        for regime in regimes:
            val_a = affinity_a.get(regime, 0.5)
            val_b = affinity_b.get(regime, 0.5)
            # Overlap: min of the two affinities (both strong = overlap)
            overlap_sum += min(val_a, val_b)

        avg_overlap = overlap_sum / len(regimes)
        complementarity = 1.0 - avg_overlap
        return max(0.0, min(1.0, complementarity))

    @staticmethod
    def _drawdown_overlap(stream_a: List[float], stream_b: List[float]) -> float:
        """
        Compute correlation of drawdown periods between two return streams.

        Low overlap = they draw down at DIFFERENT times (good for fusion).
        High overlap = they draw down together (bad, no diversification).
        """
        if len(stream_a) < 10 or len(stream_b) < 10:
            return 0.0

        min_len = min(len(stream_a), len(stream_b))
        a = np.array(stream_a[-min_len:])
        b = np.array(stream_b[-min_len:])

        # Compute rolling drawdown indicator (1 = in drawdown, 0 = at equity high)
        def drawdown_mask(returns: np.ndarray) -> np.ndarray:
            cum = np.cumsum(returns)
            peak = np.maximum.accumulate(cum)
            return (peak - cum > 0).astype(float)

        dd_a = drawdown_mask(a)
        dd_b = drawdown_mask(b)

        if np.std(dd_a) < 1e-10 or np.std(dd_b) < 1e-10:
            return 0.0

        corr_matrix = np.corrcoef(dd_a, dd_b)
        return float(corr_matrix[0, 1])

    @staticmethod
    def _frequency_ratio(trades_a: int, trades_b: int) -> float:
        """Compute balance ratio of trade frequencies."""
        if trades_a == 0 or trades_b == 0:
            return 0.0
        return min(trades_a, trades_b) / max(trades_a, trades_b)

    # ----------------------------------------------------------
    # PHASE 2: FUSION (create the hybrid organism)
    # ----------------------------------------------------------

    def fuse(self, candidate: FusionCandidate) -> Optional[HybridOrganism]:
        """
        Fuse two strategies into a hybrid organism.

        This is the actual syncytin-mediated cell fusion event.
        The envelope protein pulls the two membranes together,
        creating a multinucleated syncytium.

        Args:
            candidate: a screened FusionCandidate with receptor_match=True

        Returns:
            HybridOrganism if fusion succeeds, None if blocked.
        """
        if not candidate.receptor_match:
            log.warning(
                "[SYNCYTIN] Cannot fuse: receptor mismatch (compat=%.3f)",
                candidate.compatibility,
            )
            return None

        if len(self.hybrids) >= MAX_ACTIVE_FUSIONS:
            log.warning(
                "[SYNCYTIN] Cannot fuse: max active fusions reached (%d)",
                MAX_ACTIVE_FUSIONS,
            )
            return None

        a_id = candidate.strategy_a_id
        b_id = candidate.strategy_b_id

        # Determine fusion type from regime complementarity
        if candidate.regime_complementarity > 0.70:
            fusion_type = FusionType.REGIME_SWITCH
        elif candidate.regime_complementarity > 0.40:
            fusion_type = FusionType.WEIGHTED_BLEND
        else:
            fusion_type = FusionType.CASCADE

        # Build envelope protein (shared risk layer)
        # SL comes from config_loader -- never hardcoded
        envelope = EnvelopeProtein(
            max_sl_dollars=MAX_LOSS_DOLLARS,
            max_drawdown_pct=ENVELOPE_MAX_DRAWDOWN_PCT,
            position_limit=ENVELOPE_POSITION_LIMIT,
        )

        # Compute risk budget allocation weighted by fitness
        a_profile = self.strategy_pool.get(a_id)
        b_profile = self.strategy_pool.get(b_id)

        fitness_a = a_profile.fitness if a_profile else 0.5
        fitness_b = b_profile.fitness if b_profile else 0.5
        total_fitness = fitness_a + fitness_b
        if total_fitness > 0:
            budget_a = fitness_a / total_fitness
        else:
            budget_a = 0.5

        # Clamp to compartment limits
        budget_a = max(1.0 - COMPARTMENT_MAX_BUDGET_PCT, min(COMPARTMENT_MAX_BUDGET_PCT, budget_a))
        budget_b = 1.0 - budget_a

        # Build risk compartments (immune barrier)
        comp_a = RiskCompartment(strategy_id=a_id, budget_pct=budget_a)
        comp_b = RiskCompartment(strategy_id=b_id, budget_pct=budget_b)

        # Create the hybrid
        hybrid = HybridOrganism(
            hybrid_id="",
            strategy_a_id=a_id,
            strategy_b_id=b_id,
            fusion_type=fusion_type.value,
            envelope=envelope,
            compartment_a=comp_a,
            compartment_b=comp_b,
            status=HybridStatus.ACTIVE.value,
            created_at=datetime.now().isoformat(),
            last_evaluated=datetime.now().isoformat(),
        )
        hybrid.hybrid_id = hybrid.compute_id(a_id, b_id, fusion_type.value)

        # Store
        self.hybrids[hybrid.hybrid_id] = hybrid
        self._save_hybrid(hybrid)

        log.info(
            "[SYNCYTIN] FUSION: %s + %s -> hybrid %s (type=%s, compat=%.3f, budget=%.0f/%.0f)",
            a_id[:8], b_id[:8], hybrid.hybrid_id[:8],
            fusion_type.value, candidate.compatibility,
            budget_a * 100, budget_b * 100,
        )

        return hybrid

    # ----------------------------------------------------------
    # PHASE 3: SIGNAL ROUTING (syncytiotrophoblast in action)
    # ----------------------------------------------------------

    def get_hybrid_signal(
        self,
        hybrid: HybridOrganism,
        bars: np.ndarray,
        current_bar: int = -1,
    ) -> Dict:
        """
        Route through a hybrid organism to produce a trading signal.

        Based on the detected market regime, routes to the appropriate
        sub-strategy (or blends/cascades them). Then checks the envelope
        and compartment constraints before emitting the signal.

        Args:
            hybrid: the HybridOrganism to route through
            bars: OHLCV numpy array
            current_bar: bar index (default -1 = latest)

        Returns:
            Dict with {direction, confidence, active_strategy, regime, hybrid_id, ...}
        """
        if hybrid.status == HybridStatus.DEFUSED.value:
            return {"direction": 0, "confidence": 0.0, "hybrid_id": hybrid.hybrid_id,
                    "reason": "defused"}

        # Step 1: Detect regime
        regime = self.regime_detector.detect(bars)

        # Step 2: Check cooldown before switching
        bar_idx = current_bar if current_bar >= 0 else len(bars) - 1
        bars_since_switch = bar_idx - hybrid.regime_switch_bar

        if regime.value != hybrid.current_regime and bars_since_switch >= REGIME_SWITCH_COOLDOWN:
            old_regime = hybrid.current_regime
            hybrid.current_regime = regime.value
            hybrid.regime_switch_bar = bar_idx
            log.info(
                "[SYNCYTIN] Regime switch: %s -> %s (hybrid %s)",
                old_regime, regime.value, hybrid.hybrid_id[:8],
            )

        # Step 3: Get sub-strategy signals
        a_profile = self.strategy_pool.get(hybrid.strategy_a_id)
        b_profile = self.strategy_pool.get(hybrid.strategy_b_id)

        signal_a = a_profile.generate_signal(bars) if a_profile else {"direction": 0, "confidence": 0.0}
        signal_b = b_profile.generate_signal(bars) if b_profile else {"direction": 0, "confidence": 0.0}

        # Step 4: Route based on fusion type
        if hybrid.fusion_type == FusionType.REGIME_SWITCH.value:
            result = self._route_regime_switch(
                hybrid, signal_a, signal_b,
                a_profile, b_profile, regime,
            )
        elif hybrid.fusion_type == FusionType.WEIGHTED_BLEND.value:
            result = self._route_weighted_blend(
                hybrid, signal_a, signal_b,
                a_profile, b_profile, regime,
            )
        elif hybrid.fusion_type == FusionType.CASCADE.value:
            result = self._route_cascade(
                hybrid, signal_a, signal_b,
            )
        else:
            result = {"direction": 0, "confidence": 0.0, "active_strategy": "NONE"}

        # Step 5: Envelope check (shared risk management)
        if not hybrid.envelope.can_trade():
            result["direction"] = 0
            result["blocked_by"] = "envelope"

        # Step 6: Compartment check (immune barrier)
        active = result.get("active_strategy", "A")
        if active == "A" and hybrid.compartment_a.suspended:
            result["direction"] = 0
            result["blocked_by"] = "compartment_a_suspended"
        elif active == "B" and hybrid.compartment_b.suspended:
            result["direction"] = 0
            result["blocked_by"] = "compartment_b_suspended"

        # Add metadata
        result["hybrid_id"] = hybrid.hybrid_id
        result["fusion_type"] = hybrid.fusion_type
        result["regime"] = hybrid.current_regime
        result["status"] = hybrid.status
        result["hybrid_trades"] = hybrid.hybrid_trades
        result["fusion_alpha"] = hybrid.fusion_alpha

        return result

    def _route_regime_switch(
        self,
        hybrid: HybridOrganism,
        signal_a: Dict, signal_b: Dict,
        a_profile: Optional[StrategyProfile],
        b_profile: Optional[StrategyProfile],
        regime: RegimeType,
    ) -> Dict:
        """
        Pure regime-based routing. One strategy gets full control.

        Like the syncytiotrophoblast selectively transporting different
        nutrients in different conditions.
        """
        affinity_a = a_profile.regime_affinity.get(regime.value, 0.5) if a_profile else 0.5
        affinity_b = b_profile.regime_affinity.get(regime.value, 0.5) if b_profile else 0.5

        if affinity_a >= affinity_b:
            hybrid.active_strategy = "A"
            return {
                "direction": signal_a.get("direction", 0),
                "confidence": signal_a.get("confidence", 0.0),
                "active_strategy": "A",
                "affinity_a": affinity_a,
                "affinity_b": affinity_b,
            }
        else:
            hybrid.active_strategy = "B"
            return {
                "direction": signal_b.get("direction", 0),
                "confidence": signal_b.get("confidence", 0.0),
                "active_strategy": "B",
                "affinity_a": affinity_a,
                "affinity_b": affinity_b,
            }

    def _route_weighted_blend(
        self,
        hybrid: HybridOrganism,
        signal_a: Dict, signal_b: Dict,
        a_profile: Optional[StrategyProfile],
        b_profile: Optional[StrategyProfile],
        regime: RegimeType,
    ) -> Dict:
        """
        Weighted blend of both signals based on regime affinity.

        Both nuclei contribute, but with different weights depending
        on which regime is active.
        """
        affinity_a = a_profile.regime_affinity.get(regime.value, 0.5) if a_profile else 0.5
        affinity_b = b_profile.regime_affinity.get(regime.value, 0.5) if b_profile else 0.5
        total_aff = affinity_a + affinity_b

        if total_aff < 1e-10:
            return {"direction": 0, "confidence": 0.0, "active_strategy": "BOTH"}

        weight_a = affinity_a / total_aff
        weight_b = affinity_b / total_aff

        dir_a = signal_a.get("direction", 0)
        dir_b = signal_b.get("direction", 0)
        conf_a = signal_a.get("confidence", 0.0)
        conf_b = signal_b.get("confidence", 0.0)

        blended_score = dir_a * conf_a * weight_a + dir_b * conf_b * weight_b
        blended_direction = 1 if blended_score > 0 else (-1 if blended_score < 0 else 0)
        blended_confidence = abs(blended_score)

        hybrid.active_strategy = "BOTH"

        return {
            "direction": blended_direction,
            "confidence": blended_confidence,
            "active_strategy": "BOTH",
            "weight_a": weight_a,
            "weight_b": weight_b,
            "blended_score": blended_score,
        }

    def _route_cascade(
        self,
        hybrid: HybridOrganism,
        signal_a: Dict, signal_b: Dict,
    ) -> Dict:
        """
        Cascade routing: Strategy A proposes, Strategy B confirms.

        A generates a candidate trade. B checks if it agrees.
        Both must point the same direction for the trade to proceed.
        """
        dir_a = signal_a.get("direction", 0)
        conf_a = signal_a.get("confidence", 0.0)
        dir_b = signal_b.get("direction", 0)
        conf_b = signal_b.get("confidence", 0.0)

        if conf_a > CONFIDENCE_THRESHOLD and dir_a != 0:
            if dir_a == dir_b:
                # Both agree
                hybrid.active_strategy = "A+B"
                return {
                    "direction": dir_a,
                    "confidence": max(conf_a, conf_b),
                    "active_strategy": "A+B",
                    "cascade": "confirmed",
                }
            else:
                # Disagreement
                hybrid.active_strategy = "NONE"
                return {
                    "direction": 0,
                    "confidence": 0.0,
                    "active_strategy": "NONE",
                    "cascade": "rejected",
                }

        hybrid.active_strategy = "NONE"
        return {
            "direction": 0,
            "confidence": 0.0,
            "active_strategy": "NONE",
            "cascade": "no_proposal",
        }

    # ----------------------------------------------------------
    # PHASE 4: TRADE OUTCOME & COMPARTMENT ACCOUNTING
    # ----------------------------------------------------------

    def record_trade(
        self,
        hybrid_id: str,
        active_strategy: str,
        direction: int,
        pnl: float,
        account_equity: float = 10000.0,
    ):
        """
        Record a trade outcome for a hybrid organism.

        Updates the appropriate risk compartment, checks the immune barrier,
        executes nutrient exchange if applicable, and persists everything.

        Args:
            hybrid_id: the hybrid's ID
            active_strategy: "A" or "B" (which sub-strategy was active)
            direction: 1 (long) or -1 (short)
            pnl: trade profit/loss in dollars
            account_equity: current account equity (for DD percentage calc)
        """
        hybrid = self.hybrids.get(hybrid_id)
        if hybrid is None:
            log.warning("[SYNCYTIN] Unknown hybrid: %s", hybrid_id)
            return

        won = pnl > 0

        # Step 1: Update hybrid-level stats
        hybrid.hybrid_trades += 1
        hybrid.hybrid_pnl += pnl
        hybrid.hybrid_returns.append(pnl)
        if won:
            hybrid.hybrid_wins += 1
        else:
            hybrid.hybrid_losses += 1

        # Bayesian posterior win rate
        hybrid.posterior_wr = (
            (PRIOR_ALPHA + hybrid.hybrid_wins)
            / (PRIOR_ALPHA + PRIOR_BETA + hybrid.hybrid_trades)
        )

        hybrid.last_trade_at = datetime.now().isoformat()

        # Step 2: Update the appropriate compartment
        if active_strategy == "A":
            comp = hybrid.compartment_a
            other_comp = hybrid.compartment_b
            other_label = "B"
        else:
            comp = hybrid.compartment_b
            other_comp = hybrid.compartment_a
            other_label = "A"

        comp.equity_balance += pnl
        comp.total_trades += 1
        if won:
            comp.wins += 1
        else:
            comp.losses += 1

        # Update compartment peak and drawdown
        if comp.equity_balance > comp.peak_equity:
            comp.peak_equity = comp.equity_balance
        if comp.peak_equity > 0:
            comp.current_dd = (comp.peak_equity - comp.equity_balance) / comp.peak_equity
        elif comp.equity_balance < 0:
            comp.current_dd = abs(comp.equity_balance) / (account_equity + 1e-10)
        else:
            comp.current_dd = 0.0

        # Step 3: Immune barrier check -- suspend compartment if DD too high
        if comp.current_dd > COMPARTMENT_KILL_THRESHOLD:
            if not comp.suspended:
                comp.suspended = True
                comp.trades_since_susp = 0
                log.info(
                    "[SYNCYTIN] IMMUNE BARRIER: Compartment %s suspended "
                    "(DD=%.2f%%, threshold=%.2f%%) in hybrid %s",
                    active_strategy, comp.current_dd * 100,
                    COMPARTMENT_KILL_THRESHOLD * 100, hybrid.hybrid_id[:8],
                )

        # Step 4: Nutrient exchange (equity sharing)
        # If active compartment is profitable and other is in drawdown,
        # transfer a fraction of profits to subsidize the other
        if comp.equity_balance > 0 and other_comp.equity_balance < 0:
            subsidy = comp.equity_balance * EQUITY_SHARING_RATE
            other_comp.equity_balance += subsidy
            comp.equity_balance -= subsidy

            # Recalculate other compartment drawdown after subsidy
            if other_comp.peak_equity > 0:
                other_comp.current_dd = max(0.0,
                    (other_comp.peak_equity - other_comp.equity_balance) / other_comp.peak_equity
                )

            log.info(
                "[SYNCYTIN] NUTRIENT EXCHANGE: $%.4f from %s to %s in hybrid %s",
                subsidy, active_strategy, other_label, hybrid.hybrid_id[:8],
            )

        # Step 5: Check if suspended compartment can resume
        if comp.suspended:
            comp.trades_since_susp += 1
            if comp.current_dd < COMPARTMENT_KILL_THRESHOLD * COMPARTMENT_RESUME_RATIO:
                comp.suspended = False
                log.info(
                    "[SYNCYTIN] IMMUNE BARRIER LIFTED: Compartment %s resumed in hybrid %s",
                    active_strategy, hybrid.hybrid_id[:8],
                )

        # Step 6: Update envelope
        hybrid.envelope.update_equity(pnl)

        # Step 7: Save trade and updated hybrid state
        self._save_trade(hybrid, active_strategy, hybrid.current_regime, direction, pnl)
        self._save_hybrid(hybrid)

        # Step 8: Periodic fusion fitness evaluation
        if hybrid.hybrid_trades % FUSION_REEVAL_INTERVAL == 0:
            self.evaluate_fusion_fitness(hybrid)

    # ----------------------------------------------------------
    # PHASE 5: FUSION FITNESS MONITORING
    # ----------------------------------------------------------

    def evaluate_fusion_fitness(
        self,
        hybrid: HybridOrganism,
        solo_a_returns: List[float] = None,
        solo_b_returns: List[float] = None,
    ) -> Dict:
        """
        Evaluate whether the fusion is producing alpha over either parent alone.

        This is the viability check -- like testing if the placenta is
        functioning properly. A non-viable fusion (no alpha) means the
        pregnancy should be terminated (defusion).

        Args:
            hybrid: the HybridOrganism to evaluate
            solo_a_returns: shadow returns of strategy A trading alone
            solo_b_returns: shadow returns of strategy B trading alone

        Returns:
            Dict with fitness evaluation results
        """
        if hybrid.hybrid_trades < FUSION_MIN_TRADES:
            return {
                "hybrid_id": hybrid.hybrid_id,
                "status": "insufficient_data",
                "trades": hybrid.hybrid_trades,
                "min_required": FUSION_MIN_TRADES,
            }

        status_before = hybrid.status

        # Hybrid PnL
        hybrid_total = hybrid.hybrid_pnl

        # Solo parent PnL (from shadow execution or compartment data)
        solo_a_pnl = sum(solo_a_returns) if solo_a_returns else hybrid.compartment_a.equity_balance
        solo_b_pnl = sum(solo_b_returns) if solo_b_returns else hybrid.compartment_b.equity_balance
        best_parent = max(solo_a_pnl, solo_b_pnl)

        # Fusion alpha
        if abs(best_parent) > 1e-10:
            fusion_alpha = (hybrid_total - best_parent) / abs(best_parent)
        else:
            fusion_alpha = 0.0 if hybrid_total == 0 else 1.0

        hybrid.fusion_alpha = fusion_alpha

        # Decision
        if fusion_alpha >= FUSION_MIN_IMPROVEMENT:
            hybrid.status = HybridStatus.ACTIVE.value
            log.info(
                "[SYNCYTIN] FITNESS: Hybrid %s VIABLE (alpha=%.2f%%, hybrid=$%.2f, best_parent=$%.2f)",
                hybrid.hybrid_id[:8], fusion_alpha * 100, hybrid_total, best_parent,
            )
        elif fusion_alpha >= 0:
            hybrid.status = HybridStatus.PROBATION.value
            log.info(
                "[SYNCYTIN] FITNESS: Hybrid %s PROBATION (alpha=%.2f%%)",
                hybrid.hybrid_id[:8], fusion_alpha * 100,
            )
        elif fusion_alpha < DEFUSION_DEGRADATION_PCT:
            hybrid.status = HybridStatus.DEFUSED.value
            log.info(
                "[SYNCYTIN] DEFUSION: Hybrid %s DISSOLVED (alpha=%.2f%% < %.2f%%)",
                hybrid.hybrid_id[:8], fusion_alpha * 100, DEFUSION_DEGRADATION_PCT * 100,
            )

        hybrid.last_evaluated = datetime.now().isoformat()

        self._save_hybrid(hybrid)
        self._save_fitness_log(hybrid, solo_a_pnl, solo_b_pnl, status_before, hybrid.status)

        return {
            "hybrid_id": hybrid.hybrid_id,
            "fusion_alpha": fusion_alpha,
            "hybrid_pnl": hybrid_total,
            "solo_a_pnl": solo_a_pnl,
            "solo_b_pnl": solo_b_pnl,
            "best_parent_pnl": best_parent,
            "status_before": status_before,
            "status_after": hybrid.status,
            "hybrid_trades": hybrid.hybrid_trades,
            "win_rate": hybrid.win_rate(),
            "profit_factor": hybrid.profit_factor(),
        }

    # ----------------------------------------------------------
    # FULL PIPELINE
    # ----------------------------------------------------------

    def run_screening_and_fusion(
        self,
        strategy_pool: List[StrategyProfile] = None,
    ) -> Dict:
        """
        Run the complete screening and fusion pipeline.

        1. Screen all strategy pairs for compatibility
        2. Fuse the top compatible candidates
        3. Return summary of new hybrids created

        Args:
            strategy_pool: strategies to screen (defaults to self.strategy_pool)

        Returns:
            Dict with pipeline results
        """
        t_start = time.time()

        # Update pool if provided
        if strategy_pool:
            for sp in strategy_pool:
                self.strategy_pool[sp.strategy_id] = sp

        # Screen
        candidates = self.screen_candidates()

        # Fuse top candidates (up to available slots)
        new_hybrids = []
        available_slots = MAX_ACTIVE_FUSIONS - len(self.hybrids)

        for candidate in candidates[:available_slots]:
            hybrid = self.fuse(candidate)
            if hybrid:
                new_hybrids.append(hybrid)

        elapsed = time.time() - t_start

        log.info(
            "[SYNCYTIN] Pipeline complete: %d candidates screened, %d new fusions, "
            "%d total active hybrids | %.1fs",
            len(candidates), len(new_hybrids), len(self.hybrids), elapsed,
        )

        return {
            "candidates_screened": len(candidates),
            "new_fusions": len(new_hybrids),
            "total_active_hybrids": len(self.hybrids),
            "new_hybrid_ids": [h.hybrid_id for h in new_hybrids],
            "elapsed_seconds": elapsed,
        }

    def get_all_hybrid_signals(
        self,
        bars: np.ndarray,
    ) -> List[Dict]:
        """
        Get signals from all active hybrid organisms.

        Returns a list of signals, one per active hybrid.
        """
        signals = []
        for hybrid_id, hybrid in self.hybrids.items():
            if hybrid.status in (HybridStatus.ACTIVE.value, HybridStatus.PROBATION.value):
                signal = self.get_hybrid_signal(hybrid, bars)
                signals.append(signal)
        return signals

    def get_consensus_signal(
        self,
        bars: np.ndarray,
    ) -> Dict:
        """
        Compute consensus across all active hybrids.

        Like the polyclonal immune response: multiple hybrids voting.

        Returns:
            Dict with consensus direction, confidence, and hybrid details.
        """
        signals = self.get_all_hybrid_signals(bars)
        active_signals = [s for s in signals if s.get("direction", 0) != 0]

        if not active_signals:
            return {
                "direction": 0,
                "confidence": 0.0,
                "n_active": 0,
                "n_hybrids": len(self.hybrids),
                "signals": [],
            }

        n_long = sum(1 for s in active_signals if s["direction"] > 0)
        n_short = sum(1 for s in active_signals if s["direction"] < 0)

        weighted_long = sum(
            s.get("confidence", 0.0) for s in active_signals if s["direction"] > 0
        )
        weighted_short = sum(
            s.get("confidence", 0.0) for s in active_signals if s["direction"] < 0
        )
        total_weight = weighted_long + weighted_short

        if total_weight > 0:
            if weighted_long > weighted_short:
                direction = 1
                confidence = weighted_long / total_weight
            elif weighted_short > weighted_long:
                direction = -1
                confidence = weighted_short / total_weight
            else:
                direction = 0
                confidence = 0.0
        else:
            direction = 0
            confidence = 0.0

        return {
            "direction": direction,
            "confidence": float(confidence),
            "n_active": len(active_signals),
            "n_hybrids": len(self.hybrids),
            "n_long": n_long,
            "n_short": n_short,
            "weighted_long": float(weighted_long),
            "weighted_short": float(weighted_short),
            "signals": active_signals,
        }

    # ----------------------------------------------------------
    # SIGNAL FILE OUTPUT (for BRAIN script consumption)
    # ----------------------------------------------------------

    def write_signal_file(
        self,
        bars: np.ndarray,
        signal_path: str = None,
    ):
        """
        Write hybrid consensus signal to JSON for BRAIN script consumption.

        Args:
            bars: OHLCV numpy array
            signal_path: output file path (defaults to syncytin_signal.json)
        """
        if signal_path is None:
            signal_path = str(Path(__file__).parent / "syncytin_signal.json")

        consensus = self.get_consensus_signal(bars)

        output = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "direction": consensus["direction"],
            "confidence": consensus["confidence"],
            "n_active_hybrids": consensus["n_active"],
            "n_total_hybrids": consensus["n_hybrids"],
            "n_long": consensus.get("n_long", 0),
            "n_short": consensus.get("n_short", 0),
            "hybrids": [
                {
                    "id": s.get("hybrid_id", "")[:8],
                    "dir": s.get("direction", 0),
                    "conf": round(s.get("confidence", 0.0), 4),
                    "type": s.get("fusion_type", ""),
                    "regime": s.get("regime", ""),
                    "active": s.get("active_strategy", ""),
                    "alpha": round(s.get("fusion_alpha", 0.0), 4),
                }
                for s in consensus.get("signals", [])
            ],
        }

        try:
            tmp_path = signal_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(output, f, indent=2)
            os.replace(tmp_path, signal_path)
        except Exception as e:
            log.warning("Failed to write syncytin signal file: %s", e)

    # ----------------------------------------------------------
    # STATUS & REPORTING
    # ----------------------------------------------------------

    def get_status(self) -> Dict:
        """Get a summary of all hybrid organisms and their status."""
        active = [h for h in self.hybrids.values() if h.status == HybridStatus.ACTIVE.value]
        probation = [h for h in self.hybrids.values() if h.status == HybridStatus.PROBATION.value]
        defused = [h for h in self.hybrids.values() if h.status == HybridStatus.DEFUSED.value]

        return {
            "total_hybrids": len(self.hybrids),
            "active": len(active),
            "probation": len(probation),
            "defused": len(defused),
            "strategy_pool_size": len(self.strategy_pool),
            "candidates_screened": len(self.candidates),
            "hybrids": [h.summary() for h in self.hybrids.values()],
        }


# ============================================================
# STANDALONE TEST / DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    print("=" * 76)
    print("  SYNCYTIN FUSION ENGINE -- Strategy Fusion into Hybrid Organisms")
    print("  Algorithm #8: Retroviral Envelope -> Placental Cell Fusion")
    print("=" * 76)

    # ──────────────────────────────────────────────────────────
    # Generate synthetic test data
    # ──────────────────────────────────────────────────────────
    np.random.seed(42)
    n_bars = 500
    returns = np.random.randn(n_bars) * 0.008 + 0.0002
    for i in range(1, len(returns)):
        returns[i] -= 0.3 * returns[i - 1]
    close = 50000 + np.cumsum(returns) * 50000
    close = np.maximum(close, 100)
    high = close + np.abs(np.random.randn(n_bars) * close * 0.003)
    low = close - np.abs(np.random.randn(n_bars) * close * 0.003)
    open_p = close + np.random.randn(n_bars) * close * 0.001
    volume = np.abs(np.random.randn(n_bars) * 100 + 500)
    bars = np.column_stack([open_p, high, low, close, volume])

    print(f"\n  Synthetic data: {n_bars} bars (BTC-like)")

    # ──────────────────────────────────────────────────────────
    # Create synthetic strategy profiles for testing
    # ──────────────────────────────────────────────────────────

    # Strategy A: Trend follower (good in trends, bad in ranges)
    rng_a = np.random.RandomState(100)
    returns_a = []
    for _ in range(50):
        # Simulated: wins big in trends, small losses in ranges
        if rng_a.random() < 0.60:
            returns_a.append(rng_a.uniform(0.5, 2.0))
        else:
            returns_a.append(rng_a.uniform(-1.0, -0.1))

    strategy_a = StrategyProfile(
        strategy_id="trend_follower_01",
        strategy_type="custom",
        source_id="manual_A",
        regime_affinity={
            "trending_up": 0.80,
            "trending_down": 0.75,
            "ranging": 0.25,
            "volatile": 0.55,
            "compressed": 0.20,
            "breakout": 0.70,
        },
        win_rate=0.60,
        profit_factor=1.8,
        sharpe_ratio=1.2,
        max_drawdown=0.08,
        total_trades=50,
        total_pnl=sum(returns_a),
        fitness=0.65,
        return_stream=returns_a,
    )
    strategy_a._signal_fn = lambda bars: {
        "direction": 1 if bars[-1, 3] > bars[-2, 3] else -1,
        "confidence": 0.55,
    }

    # Strategy B: Mean reversion (good in ranges, bad in trends)
    rng_b = np.random.RandomState(200)
    returns_b = []
    for _ in range(45):
        if rng_b.random() < 0.58:
            returns_b.append(rng_b.uniform(0.3, 1.5))
        else:
            returns_b.append(rng_b.uniform(-1.2, -0.2))

    strategy_b = StrategyProfile(
        strategy_id="mean_reversion_01",
        strategy_type="custom",
        source_id="manual_B",
        regime_affinity={
            "trending_up": 0.20,
            "trending_down": 0.25,
            "ranging": 0.80,
            "volatile": 0.40,
            "compressed": 0.70,
            "breakout": 0.15,
        },
        win_rate=0.58,
        profit_factor=1.5,
        sharpe_ratio=0.9,
        max_drawdown=0.06,
        total_trades=45,
        total_pnl=sum(returns_b),
        fitness=0.58,
        return_stream=returns_b,
    )
    strategy_b._signal_fn = lambda bars: {
        "direction": -1 if bars[-1, 3] > np.mean(bars[-20:, 3]) else 1,
        "confidence": 0.50,
    }

    # Strategy C: Breakout trader (redundant with A -- should NOT fuse well with A)
    rng_c = np.random.RandomState(300)
    # Make C's returns CORRELATED with A (same regime strength)
    returns_c = [r * 0.8 + rng_c.normal(0, 0.3) for r in returns_a[:40]]

    strategy_c = StrategyProfile(
        strategy_id="breakout_trader_01",
        strategy_type="custom",
        source_id="manual_C",
        regime_affinity={
            "trending_up": 0.75,
            "trending_down": 0.70,
            "ranging": 0.30,
            "volatile": 0.60,
            "compressed": 0.25,
            "breakout": 0.85,
        },
        win_rate=0.57,
        profit_factor=1.4,
        sharpe_ratio=0.8,
        max_drawdown=0.10,
        total_trades=40,
        total_pnl=sum(returns_c),
        fitness=0.55,
        return_stream=returns_c,
    )
    strategy_c._signal_fn = lambda bars: {
        "direction": 1 if bars[-1, 3] > np.max(bars[-20:, 1]) else 0,
        "confidence": 0.45,
    }

    print(f"\n  Strategy A: Trend Follower  | WR={strategy_a.win_rate:.0%} PF={strategy_a.profit_factor:.2f} PnL=${strategy_a.total_pnl:.2f}")
    print(f"  Strategy B: Mean Reversion  | WR={strategy_b.win_rate:.0%} PF={strategy_b.profit_factor:.2f} PnL=${strategy_b.total_pnl:.2f}")
    print(f"  Strategy C: Breakout (corr) | WR={strategy_c.win_rate:.0%} PF={strategy_c.profit_factor:.2f} PnL=${strategy_c.total_pnl:.2f}")

    # ──────────────────────────────────────────────────────────
    # Test: Regime Detection
    # ──────────────────────────────────────────────────────────
    print("\n  --- REGIME DETECTION ---")
    regime = RegimeDetector.detect(bars)
    print(f"  Current regime: {regime.value}")

    # ──────────────────────────────────────────────────────────
    # Test: Full Pipeline
    # ──────────────────────────────────────────────────────────
    test_db = str(Path(__file__).parent / "test_syncytin_fusions.db")

    engine = SyncytinFusionEngine(
        db_path=test_db,
        strategy_pool=[strategy_a, strategy_b, strategy_c],
    )

    print("\n  --- PHASE 1: CANDIDATE SCREENING (Receptor Check) ---")
    candidates = engine.screen_candidates()
    print(f"  Qualified candidates: {len(candidates)}")
    for c in candidates:
        print(f"    {c.strategy_a_id[:16]} + {c.strategy_b_id[:16]} | "
              f"compat={c.compatibility:.3f} corr={c.return_correlation:.3f} "
              f"regime_comp={c.regime_complementarity:.3f} "
              f"receptor={'MATCH' if c.receptor_match else 'NO MATCH'}")

    print("\n  --- PHASE 2: FUSION ---")
    results = engine.run_screening_and_fusion()
    print(f"  New fusions: {results['new_fusions']}")
    print(f"  Active hybrids: {results['total_active_hybrids']}")

    for hid, hybrid in engine.hybrids.items():
        print(f"    {hybrid.summary()}")

    print("\n  --- PHASE 3: SIGNAL ROUTING ---")
    for hid, hybrid in engine.hybrids.items():
        signal = engine.get_hybrid_signal(hybrid, bars)
        dir_str = "LONG" if signal["direction"] > 0 else (
            "SHORT" if signal["direction"] < 0 else "FLAT"
        )
        print(f"    Hybrid {hid[:8]}: {dir_str} conf={signal.get('confidence', 0):.4f} "
              f"active={signal.get('active_strategy', '?')} regime={signal.get('regime', '?')}")

    print("\n  --- PHASE 4: SIMULATE TRADES ---")
    # Simulate some trades for the first hybrid
    if engine.hybrids:
        first_hybrid = list(engine.hybrids.values())[0]
        sim_rng = np.random.RandomState(42)

        for i in range(35):
            active = "A" if sim_rng.random() < 0.5 else "B"
            won = sim_rng.random() < 0.60
            pnl = sim_rng.uniform(0.3, 1.5) if won else sim_rng.uniform(-1.0, -0.2)
            engine.record_trade(first_hybrid.hybrid_id, active, 1, pnl)

        print(f"  Simulated {first_hybrid.hybrid_trades} trades for hybrid {first_hybrid.hybrid_id[:8]}")
        print(f"  Hybrid PnL: ${first_hybrid.hybrid_pnl:.2f}")
        print(f"  Win Rate:   {first_hybrid.win_rate():.1%}")
        print(f"  PF:         {first_hybrid.profit_factor():.2f}")
        print(f"  Comp A eq:  ${first_hybrid.compartment_a.equity_balance:.2f}")
        print(f"  Comp B eq:  ${first_hybrid.compartment_b.equity_balance:.2f}")

    print("\n  --- PHASE 5: FITNESS EVALUATION ---")
    if engine.hybrids:
        first_hybrid = list(engine.hybrids.values())[0]
        eval_result = engine.evaluate_fusion_fitness(first_hybrid)
        print(f"  Hybrid:    {eval_result['hybrid_id'][:8]}")
        print(f"  Alpha:     {eval_result['fusion_alpha']:+.2%}")
        print(f"  Status:    {eval_result['status_after']}")
        print(f"  Hybrid $:  ${eval_result['hybrid_pnl']:.2f}")
        print(f"  Solo A $:  ${eval_result['solo_a_pnl']:.2f}")
        print(f"  Solo B $:  ${eval_result['solo_b_pnl']:.2f}")

    print("\n  --- CONSENSUS SIGNAL ---")
    consensus = engine.get_consensus_signal(bars)
    dir_str = "LONG" if consensus["direction"] > 0 else (
        "SHORT" if consensus["direction"] < 0 else "FLAT"
    )
    print(f"  Direction:  {dir_str}")
    print(f"  Confidence: {consensus['confidence']:.4f}")
    print(f"  Active:     {consensus['n_active']} hybrids")

    # Write signal file
    signal_path = str(Path(__file__).parent / "test_syncytin_signal.json")
    engine.write_signal_file(bars, signal_path)
    print(f"\n  Signal file written to: {signal_path}")

    # Status report
    print("\n  --- STATUS REPORT ---")
    status = engine.get_status()
    print(f"  Total hybrids:     {status['total_hybrids']}")
    print(f"  Active:            {status['active']}")
    print(f"  Probation:         {status['probation']}")
    print(f"  Defused:           {status['defused']}")
    print(f"  Strategy pool:     {status['strategy_pool_size']}")

    # Cleanup test files
    try:
        os.remove(test_db)
    except OSError:
        pass
    try:
        os.remove(signal_path)
    except OSError:
        pass
    try:
        os.remove(signal_path + ".tmp")
    except OSError:
        pass

    print("\n" + "=" * 76)
    print("  Syncytin Fusion Engine test complete.")
    print("  BIOLOGICAL INVARIANT:")
    print("    'A mechanism designed for invasion was repurposed for nurturing.'")
    print("    'The fusion creates something neither parent could be alone.'")
    print("=" * 76)
