"""
VDJ RECOMBINATION ENGINE -- Adaptive Immune Trading System
============================================================
Domesticated Transib transposon -> RAG1/RAG2 V(D)J recombination.

Biological basis:
    RAG1/RAG2 is a domesticated Transib DNA transposon (TE family #24 in
    our system) that became the foundation of the vertebrate adaptive
    immune system ~500 million years ago. It works by:

    1. V(D)J Recombination: Randomly selecting one V (Variable), one D
       (Diversity), and one J (Joining) gene segment from large pools
    2. Junctional Diversity: Adding random nucleotides at V-D and D-J
       junctions during the cutting/joining process
    3. Clonal Selection: Testing billions of unique antibodies against
       pathogens -- winners survive, losers undergo apoptosis
    4. Affinity Maturation: Somatic hypermutation in germinal centers
       optimizes the binding affinity of surviving antibodies
    5. Memory B Cells: The best antibodies are stored permanently for
       rapid recall upon re-exposure

Trading translation:
    V segments = entry signal generators (RSI, MACD, Stoch, BB, etc.)
    D segments = market regime classifiers (trending, ranging, volatile)
    J segments = exit strategies (trailing, partial, time-based, etc.)
    "Antibody" = unique V+D+J combination = a micro-strategy
    Junctional diversity = random parameter offsets at combination points
    Clonal selection = backtest all antibodies, kill losers, clone winners
    Affinity maturation = mutate parameters of winning strategies
    Memory B cells = persistent database of winning V+D+J combos

Integration:
    - Reads TE activations from TEActivationEngine (33 families)
    - Stores antibodies in SQLite alongside domestication DB
    - Feeds winning antibody signals into TEQAv3Engine pipeline
    - MQL5 EA reads antibody consensus via JSON signal file

Authors: DooDoo + Claude
Date:    2026-02-08
Version: VDJ-RECOMBINATION-1.0
"""

import json
import math
import os
import random
import logging
import sqlite3
import hashlib
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

# Import from existing TEQA system
from teqa_v3_neural_te import (
    ALL_TE_FAMILIES,
    TEActivationEngine,
    TEClass,
    N_QUBITS,
)

log = logging.getLogger(__name__)

VERSION = "VDJ-RECOMBINATION-1.0"

# ============================================================
# CONSTANTS
# ============================================================

# Antibody population
DEFAULT_POPULATION_SIZE = 100       # Initial antibody pool per generation
ELITE_SURVIVORS = 10                # Top N survive clonal selection
MEMORY_CELL_LIMIT = 50              # Max permanently stored antibodies

# Clonal selection thresholds
MIN_BACKTEST_BARS = 200             # Minimum bars for fitness evaluation
MIN_TRADES_FOR_FITNESS = 10         # Must generate >= 10 trades to be scored
MIN_WIN_RATE_SURVIVE = 0.55         # Below this = apoptosis
MIN_PROFIT_FACTOR_SURVIVE = 1.2     # avg_win/avg_loss minimum

# Affinity maturation
MATURATION_ROUNDS = 5               # Number of hypermutation rounds per winner
MATURATION_MUTATION_RATE = 0.20     # Probability of mutating each parameter
MATURATION_PARAM_JITTER = 0.15      # Max fractional change per parameter

# Junctional diversity (N-nucleotide additions)
JUNCTIONAL_NOISE_SCALE = 0.10      # Random parameter offset at V-D and D-J junctions

# Memory B cell thresholds
MEMORY_MIN_WIN_RATE = 0.65          # Must exceed this to become a memory cell
MEMORY_MIN_PROFIT_FACTOR = 1.5      # Must exceed this
MEMORY_MIN_TRADES = 20              # Must have this many trades
MEMORY_EXPIRY_DAYS = 90             # Re-evaluate after 90 days of inactivity

# Bayesian prior for win rate estimation (same philosophy as domestication)
PRIOR_ALPHA = 8
PRIOR_BETA = 8


# ============================================================
# V SEGMENTS -- Entry Signal Generators
# ============================================================

class VSegmentType(Enum):
    """Each V segment corresponds to a specific entry signal logic."""
    RSI_OVERSOLD     = "rsi_oversold"
    RSI_OVERBOUGHT   = "rsi_overbought"
    MACD_CROSS_UP    = "macd_cross_up"
    MACD_CROSS_DOWN  = "macd_cross_down"
    BB_LOWER_TOUCH   = "bb_lower_touch"
    BB_UPPER_TOUCH   = "bb_upper_touch"
    EMA_CROSS_UP     = "ema_cross_up"
    EMA_CROSS_DOWN   = "ema_cross_down"
    STOCH_OVERSOLD   = "stoch_oversold"
    STOCH_OVERBOUGHT = "stoch_overbought"
    MOMENTUM_LONG    = "momentum_long"
    MOMENTUM_SHORT   = "momentum_short"
    VOLUME_SPIKE_UP  = "volume_spike_up"
    MEAN_REVERT_LONG = "mean_revert_long"
    MEAN_REVERT_SHORT = "mean_revert_short"
    BREAKOUT_HIGH    = "breakout_high"
    BREAKOUT_LOW     = "breakout_low"
    CANDLE_ENGULF_UP = "candle_engulf_up"
    CANDLE_ENGULF_DN = "candle_engulf_dn"


@dataclass
class VSegment:
    """
    Variable segment: entry signal generator with tunable parameters.
    Maps to one or more TE families from the 33-family system.
    """
    segment_type: VSegmentType
    # Tunable parameters (specific to each type)
    params: Dict[str, float] = field(default_factory=dict)
    # Which TE families this V segment draws from
    te_sources: List[str] = field(default_factory=list)

    def default_params(self) -> Dict[str, float]:
        """Default parameters for each V segment type."""
        defaults = {
            VSegmentType.RSI_OVERSOLD:     {"period": 14, "threshold": 30},
            VSegmentType.RSI_OVERBOUGHT:   {"period": 14, "threshold": 70},
            VSegmentType.MACD_CROSS_UP:    {"fast": 12, "slow": 26, "signal": 9},
            VSegmentType.MACD_CROSS_DOWN:  {"fast": 12, "slow": 26, "signal": 9},
            VSegmentType.BB_LOWER_TOUCH:   {"period": 20, "std_mult": 2.0},
            VSegmentType.BB_UPPER_TOUCH:   {"period": 20, "std_mult": 2.0},
            VSegmentType.EMA_CROSS_UP:     {"fast": 8, "slow": 21},
            VSegmentType.EMA_CROSS_DOWN:   {"fast": 8, "slow": 21},
            VSegmentType.STOCH_OVERSOLD:   {"k_period": 14, "d_period": 3, "threshold": 20},
            VSegmentType.STOCH_OVERBOUGHT: {"k_period": 14, "d_period": 3, "threshold": 80},
            VSegmentType.MOMENTUM_LONG:    {"lookback": 10, "threshold": 0.01},
            VSegmentType.MOMENTUM_SHORT:   {"lookback": 10, "threshold": -0.01},
            VSegmentType.VOLUME_SPIKE_UP:  {"lookback": 20, "spike_mult": 2.0},
            VSegmentType.MEAN_REVERT_LONG: {"lookback": 20, "z_threshold": -2.0},
            VSegmentType.MEAN_REVERT_SHORT:{"lookback": 20, "z_threshold": 2.0},
            VSegmentType.BREAKOUT_HIGH:    {"lookback": 20, "buffer_pct": 0.001},
            VSegmentType.BREAKOUT_LOW:     {"lookback": 20, "buffer_pct": 0.001},
            VSegmentType.CANDLE_ENGULF_UP: {"min_body_ratio": 0.6},
            VSegmentType.CANDLE_ENGULF_DN: {"min_body_ratio": 0.6},
        }
        return defaults.get(self.segment_type, {})

    def te_family_mapping(self) -> List[str]:
        """Map V segments to source TE families from the 33-family system."""
        mapping = {
            VSegmentType.RSI_OVERSOLD:     ["Ty1_copia"],       # rsi
            VSegmentType.RSI_OVERBOUGHT:   ["Ty1_copia"],
            VSegmentType.MACD_CROSS_UP:    ["Ty3_gypsy"],       # macd
            VSegmentType.MACD_CROSS_DOWN:  ["Ty3_gypsy"],
            VSegmentType.BB_LOWER_TOUCH:   ["Ty5"],             # bollinger_position
            VSegmentType.BB_UPPER_TOUCH:   ["Ty5"],
            VSegmentType.EMA_CROSS_UP:     ["CACTA"],           # ema_crossover
            VSegmentType.EMA_CROSS_DOWN:   ["CACTA"],
            VSegmentType.STOCH_OVERSOLD:   ["BEL_Pao"],         # momentum proxy
            VSegmentType.STOCH_OVERBOUGHT: ["BEL_Pao"],
            VSegmentType.MOMENTUM_LONG:    ["BEL_Pao", "LINE"], # momentum + price_change
            VSegmentType.MOMENTUM_SHORT:   ["BEL_Pao", "LINE"],
            VSegmentType.VOLUME_SPIKE_UP:  ["SINE", "Helitron"],# tick_volume + volume_profile
            VSegmentType.MEAN_REVERT_LONG: ["RTE"],             # mean_reversion
            VSegmentType.MEAN_REVERT_SHORT:["RTE"],
            VSegmentType.BREAKOUT_HIGH:    ["I_element", "SVA_Regulatory"],  # support_resistance + compression_breakout
            VSegmentType.BREAKOUT_LOW:     ["I_element", "SVA_Regulatory"],
            VSegmentType.CANDLE_ENGULF_UP: ["hobo"],            # candle_pattern
            VSegmentType.CANDLE_ENGULF_DN: ["hobo"],
        }
        return mapping.get(self.segment_type, [])


# ============================================================
# D SEGMENTS -- Market Regime Classifiers
# ============================================================

class DSegmentType(Enum):
    """Each D segment classifies the current market regime."""
    TRENDING_UP   = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING       = "ranging"
    VOLATILE      = "volatile"
    COMPRESSED    = "compressed"
    BREAKOUT      = "breakout"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class DSegment:
    """
    Diversity segment: market regime classifier with tunable parameters.
    Acts as a filter -- the antibody only fires when the regime matches.
    """
    segment_type: DSegmentType
    params: Dict[str, float] = field(default_factory=dict)
    te_sources: List[str] = field(default_factory=list)

    def default_params(self) -> Dict[str, float]:
        defaults = {
            DSegmentType.TRENDING_UP:    {"adx_min": 25, "ema_diff_pct": 0.005},
            DSegmentType.TRENDING_DOWN:  {"adx_min": 25, "ema_diff_pct": -0.005},
            DSegmentType.RANGING:        {"adx_max": 20, "bb_width_max": 0.02},
            DSegmentType.VOLATILE:       {"atr_ratio_min": 1.5},
            DSegmentType.COMPRESSED:     {"atr_ratio_max": 0.6, "bb_squeeze": True},
            DSegmentType.BREAKOUT:       {"compression_then_expansion": True, "lookback": 20},
            DSegmentType.MEAN_REVERTING: {"hurst_max": 0.4},
        }
        return defaults.get(self.segment_type, {})

    def te_family_mapping(self) -> List[str]:
        mapping = {
            DSegmentType.TRENDING_UP:    ["DIRS1", "Penelope"],       # trend_strength + trend_duration
            DSegmentType.TRENDING_DOWN:  ["DIRS1", "Penelope"],
            DSegmentType.RANGING:        ["Mariner_Tc1", "Mutator"],  # fractal_dim + mutation_rate
            DSegmentType.VOLATILE:       ["VIPER_Ngaro", "Alu"],      # atr_ratio + short_volatility
            DSegmentType.COMPRESSED:     ["Crypton", "SVA_Regulatory"],# compression_ratio + compression_breakout
            DSegmentType.BREAKOUT:       ["SVA_Regulatory", "P_element"], # compression_breakout + spread_analysis
            DSegmentType.MEAN_REVERTING: ["RTE", "Alu_Exonization"],  # mean_reversion + noise_pattern
        }
        return mapping.get(self.segment_type, [])


# ============================================================
# J SEGMENTS -- Exit Strategies
# ============================================================

class JSegmentType(Enum):
    """Each J segment defines an exit strategy."""
    FIXED_TP           = "fixed_tp"
    TRAILING_STOP      = "trailing_stop"
    PARTIAL_CLOSE      = "partial_close"
    TIME_BASED         = "time_based"
    SIGNAL_REVERSAL    = "signal_reversal"
    ATR_TRAILING       = "atr_trailing"
    BREAKEVEN_TRAIL    = "breakeven_trail"
    OPPOSITE_BB        = "opposite_bb"


@dataclass
class JSegment:
    """
    Joining segment: exit strategy with tunable parameters.
    Determines how and when profits are taken / losses are cut.
    """
    segment_type: JSegmentType
    params: Dict[str, float] = field(default_factory=dict)
    te_sources: List[str] = field(default_factory=list)

    def default_params(self) -> Dict[str, float]:
        defaults = {
            JSegmentType.FIXED_TP:        {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0},
            JSegmentType.TRAILING_STOP:   {"trail_atr_mult": 1.5, "activation_atr": 1.0},
            JSegmentType.PARTIAL_CLOSE:   {"partial_pct": 0.50, "partial_at_atr": 1.5, "remainder_trail": 2.0},
            JSegmentType.TIME_BASED:      {"max_bars": 50, "min_profit_pct": 0.001},
            JSegmentType.SIGNAL_REVERSAL: {"reversal_threshold": 0.6},
            JSegmentType.ATR_TRAILING:    {"atr_period": 14, "atr_mult": 2.0},
            JSegmentType.BREAKEVEN_TRAIL: {"be_trigger_atr": 1.0, "trail_after_be": 1.5},
            JSegmentType.OPPOSITE_BB:     {"bb_period": 20, "bb_std": 2.0},
        }
        return defaults.get(self.segment_type, {})

    def te_family_mapping(self) -> List[str]:
        mapping = {
            JSegmentType.FIXED_TP:        ["VIPER_Ngaro"],           # atr_ratio
            JSegmentType.TRAILING_STOP:   ["VIPER_Ngaro", "LINE"],   # atr + price_change
            JSegmentType.PARTIAL_CLOSE:   ["Arc_Capsid"],            # successful pattern echo
            JSegmentType.TIME_BASED:      ["pogo"],                  # session_overlap
            JSegmentType.SIGNAL_REVERSAL: ["L1_Neuronal"],           # pattern_repetition
            JSegmentType.ATR_TRAILING:    ["VIPER_Ngaro"],
            JSegmentType.BREAKEVEN_TRAIL: ["TRIM28_Silencer"],       # drawdown management
            JSegmentType.OPPOSITE_BB:     ["Ty5"],                   # bollinger_position
        }
        return mapping.get(self.segment_type, [])


# ============================================================
# ANTIBODY -- A unique V+D+J combination
# ============================================================

@dataclass
class Antibody:
    """
    A unique micro-strategy formed by V(D)J recombination.

    The antibody encodes:
      - WHAT signal triggers entry (V segment)
      - WHEN that signal is valid (D segment = regime filter)
      - HOW to exit the trade (J segment)
      - Junctional noise at V-D and D-J boundaries

    Each antibody can be evaluated against historical data for fitness.
    """
    antibody_id: str           # Unique hash
    v_segment: VSegment        # Entry signal
    d_segment: DSegment        # Regime filter
    j_segment: JSegment        # Exit strategy

    # Junctional diversity (random offsets at combination boundaries)
    vd_junction_noise: Dict[str, float] = field(default_factory=dict)
    dj_junction_noise: Dict[str, float] = field(default_factory=dict)

    # Fitness metrics (populated after backtesting)
    fitness: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Bayesian posterior win rate
    posterior_wr: float = 0.5

    # Lineage tracking
    generation: int = 0
    parent_id: Optional[str] = None
    mutation_count: int = 0

    # Memory cell status
    is_memory_cell: bool = False
    memory_since: Optional[str] = None

    def compute_id(self) -> str:
        """Generate unique hash from V+D+J combination + junction noise."""
        raw = (
            f"{self.v_segment.segment_type.value}"
            f"|{json.dumps(self.v_segment.params, sort_keys=True)}"
            f"|{self.d_segment.segment_type.value}"
            f"|{json.dumps(self.d_segment.params, sort_keys=True)}"
            f"|{self.j_segment.segment_type.value}"
            f"|{json.dumps(self.j_segment.params, sort_keys=True)}"
            f"|{json.dumps(self.vd_junction_noise, sort_keys=True)}"
            f"|{json.dumps(self.dj_junction_noise, sort_keys=True)}"
        )
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def get_direction(self) -> int:
        """Infer trade direction from V segment type."""
        long_types = {
            VSegmentType.RSI_OVERSOLD, VSegmentType.MACD_CROSS_UP,
            VSegmentType.BB_LOWER_TOUCH, VSegmentType.EMA_CROSS_UP,
            VSegmentType.STOCH_OVERSOLD, VSegmentType.MOMENTUM_LONG,
            VSegmentType.VOLUME_SPIKE_UP, VSegmentType.MEAN_REVERT_LONG,
            VSegmentType.BREAKOUT_HIGH, VSegmentType.CANDLE_ENGULF_UP,
        }
        return 1 if self.v_segment.segment_type in long_types else -1

    def get_te_fingerprint(self) -> List[str]:
        """Get all TE families involved in this antibody."""
        tes = set()
        tes.update(self.v_segment.te_family_mapping())
        tes.update(self.d_segment.te_family_mapping())
        tes.update(self.j_segment.te_family_mapping())
        return sorted(tes)

    def summary(self) -> str:
        d = self.get_direction()
        dir_str = "LONG" if d > 0 else "SHORT"
        return (
            f"[{self.antibody_id[:8]}] {dir_str} | "
            f"V={self.v_segment.segment_type.value} "
            f"D={self.d_segment.segment_type.value} "
            f"J={self.j_segment.segment_type.value} | "
            f"WR={self.posterior_wr:.1%} PF={self.profit_factor:.2f} "
            f"trades={self.total_trades} fit={self.fitness:.4f}"
        )


# ============================================================
# VDJ RECOMBINATION ENGINE
# ============================================================

class VDJRecombinationEngine:
    """
    Core engine that performs V(D)J recombination to generate antibodies.

    This is the RAG1/RAG2 recombinase equivalent. It:
      1. Maintains pools of V, D, and J gene segments
      2. Randomly combines one V + one D + one J to create an antibody
      3. Adds junctional diversity (random parameter offsets at join points)
      4. Evaluates antibody fitness via backtesting
      5. Performs clonal selection (kill losers, clone winners)
      6. Runs affinity maturation (somatic hypermutation on winners)
      7. Stores the best as memory B cells in a persistent database
    """

    def __init__(
        self,
        db_path: str = None,
        population_size: int = DEFAULT_POPULATION_SIZE,
        seed: int = None,
    ):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.population_size = population_size

        # Gene segment pools
        self.v_pool: List[VSegmentType] = list(VSegmentType)
        self.d_pool: List[DSegmentType] = list(DSegmentType)
        self.j_pool: List[JSegmentType] = list(JSegmentType)

        # Current antibody population
        self.population: List[Antibody] = []

        # Memory B cells (persisted winners)
        self.memory_cells: List[Antibody] = []

        # Database
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "vdj_antibodies.db")
        else:
            self.db_path = db_path
        self._init_db()

        # Load existing memory cells
        self._load_memory_cells()

        # Generation counter
        self.generation = 0

        # TE activation engine for signal evaluation
        self.te_engine = TEActivationEngine()

    # ----------------------------------------------------------
    # DATABASE
    # ----------------------------------------------------------

    def _init_db(self):
        """Initialize the antibody database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS antibodies (
                        antibody_id TEXT PRIMARY KEY,
                        v_type TEXT NOT NULL,
                        v_params TEXT NOT NULL,
                        d_type TEXT NOT NULL,
                        d_params TEXT NOT NULL,
                        j_type TEXT NOT NULL,
                        j_params TEXT NOT NULL,
                        vd_junction TEXT DEFAULT '{}',
                        dj_junction TEXT DEFAULT '{}',
                        fitness REAL DEFAULT 0.0,
                        win_rate REAL DEFAULT 0.0,
                        posterior_wr REAL DEFAULT 0.5,
                        profit_factor REAL DEFAULT 0.0,
                        total_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0.0,
                        sharpe_ratio REAL DEFAULT 0.0,
                        max_drawdown REAL DEFAULT 0.0,
                        avg_win REAL DEFAULT 0.0,
                        avg_loss REAL DEFAULT 0.0,
                        generation INTEGER DEFAULT 0,
                        parent_id TEXT,
                        mutation_count INTEGER DEFAULT 0,
                        is_memory_cell INTEGER DEFAULT 0,
                        memory_since TEXT,
                        te_fingerprint TEXT DEFAULT '',
                        direction INTEGER DEFAULT 0,
                        created_at TEXT,
                        last_evaluated TEXT,
                        last_activated TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS vdj_generations (
                        generation INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        population_size INTEGER,
                        survivors INTEGER,
                        memory_cells_added INTEGER,
                        avg_fitness REAL,
                        best_fitness REAL,
                        best_antibody_id TEXT,
                        maturation_improvements INTEGER
                    )
                """)
                conn.commit()
        except Exception as e:
            log.warning("VDJ DB init failed: %s", e)

    def _load_memory_cells(self):
        """Load memory B cells from database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT antibody_id, v_type, v_params, d_type, d_params,
                           j_type, j_params, vd_junction, dj_junction,
                           fitness, win_rate, posterior_wr, profit_factor,
                           total_trades, total_pnl, sharpe_ratio, max_drawdown,
                           avg_win, avg_loss, generation, parent_id,
                           mutation_count, memory_since
                    FROM antibodies
                    WHERE is_memory_cell = 1
                    ORDER BY fitness DESC
                    LIMIT ?
                """, (MEMORY_CELL_LIMIT,))

                self.memory_cells = []
                for row in cursor.fetchall():
                    ab = self._row_to_antibody(row)
                    ab.is_memory_cell = True
                    self.memory_cells.append(ab)

                log.info("Loaded %d memory B cells from database", len(self.memory_cells))

        except Exception as e:
            log.warning("Failed to load memory cells: %s", e)

    def _row_to_antibody(self, row) -> Antibody:
        """Convert a database row to an Antibody object."""
        v_seg = VSegment(
            segment_type=VSegmentType(row[1]),
            params=json.loads(row[2]),
        )
        v_seg.te_sources = v_seg.te_family_mapping()

        d_seg = DSegment(
            segment_type=DSegmentType(row[3]),
            params=json.loads(row[4]),
        )
        d_seg.te_sources = d_seg.te_family_mapping()

        j_seg = JSegment(
            segment_type=JSegmentType(row[5]),
            params=json.loads(row[6]),
        )
        j_seg.te_sources = j_seg.te_family_mapping()

        return Antibody(
            antibody_id=row[0],
            v_segment=v_seg,
            d_segment=d_seg,
            j_segment=j_seg,
            vd_junction_noise=json.loads(row[7]) if row[7] else {},
            dj_junction_noise=json.loads(row[8]) if row[8] else {},
            fitness=row[9] or 0.0,
            win_rate=row[10] or 0.0,
            posterior_wr=row[11] or 0.5,
            profit_factor=row[12] or 0.0,
            total_trades=row[13] or 0,
            total_pnl=row[14] or 0.0,
            sharpe_ratio=row[15] or 0.0,
            max_drawdown=row[16] or 0.0,
            avg_win=row[17] or 0.0,
            avg_loss=row[18] or 0.0,
            generation=row[19] or 0,
            parent_id=row[20],
            mutation_count=row[21] or 0,
            memory_since=row[22],
        )

    def _save_antibody(self, ab: Antibody):
        """Save a single antibody to the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                now = datetime.now().isoformat()
                conn.execute("""
                    INSERT OR REPLACE INTO antibodies
                    (antibody_id, v_type, v_params, d_type, d_params,
                     j_type, j_params, vd_junction, dj_junction,
                     fitness, win_rate, posterior_wr, profit_factor,
                     total_trades, total_pnl, sharpe_ratio, max_drawdown,
                     avg_win, avg_loss, generation, parent_id,
                     mutation_count, is_memory_cell, memory_since,
                     te_fingerprint, direction, created_at, last_evaluated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ab.antibody_id,
                    ab.v_segment.segment_type.value,
                    json.dumps(ab.v_segment.params, sort_keys=True),
                    ab.d_segment.segment_type.value,
                    json.dumps(ab.d_segment.params, sort_keys=True),
                    ab.j_segment.segment_type.value,
                    json.dumps(ab.j_segment.params, sort_keys=True),
                    json.dumps(ab.vd_junction_noise, sort_keys=True),
                    json.dumps(ab.dj_junction_noise, sort_keys=True),
                    ab.fitness, ab.win_rate, ab.posterior_wr, ab.profit_factor,
                    ab.total_trades, ab.total_pnl, ab.sharpe_ratio,
                    ab.max_drawdown, ab.avg_win, ab.avg_loss,
                    ab.generation, ab.parent_id, ab.mutation_count,
                    1 if ab.is_memory_cell else 0,
                    ab.memory_since,
                    "+".join(ab.get_te_fingerprint()),
                    ab.get_direction(),
                    now, now,
                ))
                conn.commit()
        except Exception as e:
            log.warning("Failed to save antibody %s: %s", ab.antibody_id, e)

    def _save_generation(self, gen_stats: Dict):
        """Save generation statistics."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT OR REPLACE INTO vdj_generations
                    (generation, timestamp, population_size, survivors,
                     memory_cells_added, avg_fitness, best_fitness,
                     best_antibody_id, maturation_improvements)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gen_stats["generation"],
                    gen_stats["timestamp"],
                    gen_stats["population_size"],
                    gen_stats["survivors"],
                    gen_stats["memory_cells_added"],
                    gen_stats["avg_fitness"],
                    gen_stats["best_fitness"],
                    gen_stats["best_antibody_id"],
                    gen_stats["maturation_improvements"],
                ))
                conn.commit()
        except Exception as e:
            log.warning("Failed to save generation stats: %s", e)

    # ----------------------------------------------------------
    # STEP 1: V(D)J RECOMBINATION -- Generate Antibodies
    # ----------------------------------------------------------

    def recombine(self, n: int = None) -> List[Antibody]:
        """
        Generate a pool of antibodies through V(D)J recombination.

        This is the RAG1/RAG2 cut-and-paste reaction:
          1. Randomly select one V, one D, one J segment
          2. Initialize with default parameters
          3. Add junctional diversity (N-nucleotide additions)
          4. Compute unique antibody ID

        Returns list of newly created antibodies.
        """
        n = n or self.population_size
        self.generation += 1
        antibodies = []

        for _ in range(n):
            # RAG1/RAG2 selects one segment from each pool
            v_type = self.rng.choice(self.v_pool)
            d_type = self.rng.choice(self.d_pool)
            j_type = self.rng.choice(self.j_pool)

            # Create segments with default parameters
            v_seg = VSegment(segment_type=v_type)
            v_seg.params = v_seg.default_params()
            v_seg.te_sources = v_seg.te_family_mapping()

            d_seg = DSegment(segment_type=d_type)
            d_seg.params = d_seg.default_params()
            d_seg.te_sources = d_seg.te_family_mapping()

            j_seg = JSegment(segment_type=j_type)
            j_seg.params = j_seg.default_params()
            j_seg.te_sources = j_seg.te_family_mapping()

            # Junctional diversity: random offsets at V-D boundary
            # (Like N-nucleotide additions by TdT enzyme)
            vd_noise = {}
            for key in v_seg.params:
                if isinstance(v_seg.params[key], (int, float)):
                    noise = self.rng.gauss(0, JUNCTIONAL_NOISE_SCALE)
                    vd_noise[f"v_{key}"] = noise
                    v_seg.params[key] *= (1.0 + noise)

            # Junctional diversity at D-J boundary
            dj_noise = {}
            for key in j_seg.params:
                if isinstance(j_seg.params[key], (int, float)):
                    noise = self.rng.gauss(0, JUNCTIONAL_NOISE_SCALE)
                    dj_noise[f"j_{key}"] = noise
                    j_seg.params[key] *= (1.0 + noise)

            # Also jitter D segment params
            for key in d_seg.params:
                if isinstance(d_seg.params[key], (int, float)):
                    noise = self.rng.gauss(0, JUNCTIONAL_NOISE_SCALE * 0.5)
                    d_seg.params[key] *= (1.0 + noise)

            ab = Antibody(
                antibody_id="",  # Will compute
                v_segment=v_seg,
                d_segment=d_seg,
                j_segment=j_seg,
                vd_junction_noise=vd_noise,
                dj_junction_noise=dj_noise,
                generation=self.generation,
            )
            ab.antibody_id = ab.compute_id()
            antibodies.append(ab)

        self.population = antibodies
        log.info(
            "[VDJ] Generation %d: recombined %d antibodies from %dV x %dD x %dJ segments",
            self.generation, n,
            len(self.v_pool), len(self.d_pool), len(self.j_pool),
        )
        return antibodies

    # ----------------------------------------------------------
    # STEP 2: FITNESS EVALUATION (Antigen Testing)
    # ----------------------------------------------------------

    def evaluate_fitness(
        self,
        bars: np.ndarray,
        antibodies: List[Antibody] = None,
        spread_points: float = 2.0,
    ) -> List[Antibody]:
        """
        Evaluate each antibody's fitness by backtesting against price data.

        This is the "antigen presentation" step: each antibody is tested
        against the market (the pathogen) to see if it can produce a
        profitable immune response (trading edge).

        Args:
            bars: OHLCV numpy array (N x 5), minimum 200 rows
            antibodies: list to evaluate (defaults to self.population)
            spread_points: trading cost in price points per round trip

        Returns:
            Same antibodies list with fitness metrics populated.
        """
        if antibodies is None:
            antibodies = self.population

        if len(bars) < MIN_BACKTEST_BARS:
            log.warning("Need >= %d bars for fitness eval, got %d", MIN_BACKTEST_BARS, len(bars))
            return antibodies

        close = bars[:, 3]
        high = bars[:, 1]
        low = bars[:, 2]
        open_p = bars[:, 0]
        volume = bars[:, 4] if bars.shape[1] > 4 else np.ones(len(close))

        for ab in antibodies:
            trades = self._simulate_antibody(ab, open_p, high, low, close, volume, spread_points)
            self._compute_fitness_metrics(ab, trades)

        # Sort by fitness
        antibodies.sort(key=lambda a: a.fitness, reverse=True)

        log.info(
            "[VDJ] Evaluated %d antibodies | Best: %s (fit=%.4f) | Worst: %s (fit=%.4f)",
            len(antibodies),
            antibodies[0].antibody_id[:8] if antibodies else "N/A",
            antibodies[0].fitness if antibodies else 0,
            antibodies[-1].antibody_id[:8] if antibodies else "N/A",
            antibodies[-1].fitness if antibodies else 0,
        )
        return antibodies

    def _simulate_antibody(
        self,
        ab: Antibody,
        open_p: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        spread: float,
    ) -> List[Dict]:
        """
        Simulate an antibody's trading logic over historical data.
        Returns list of trade results: [{entry_bar, exit_bar, direction, pnl}, ...]
        """
        trades = []
        in_trade = False
        entry_price = 0.0
        entry_bar = 0
        direction = ab.get_direction()
        trail_stop = 0.0
        partial_closed = False

        # ATR for position sizing reference
        atr_arr = self._compute_atr(high, low, close, 14)

        # Need enough lookback for indicators
        start_bar = max(50, int(ab.v_segment.params.get("period", 14)) + 5)

        for i in range(start_bar, len(close)):
            atr = atr_arr[i] if i < len(atr_arr) else atr_arr[-1]
            if atr < 1e-10:
                continue

            if not in_trade:
                # Check D segment (regime filter)
                if not self._check_regime(ab.d_segment, open_p, high, low, close, volume, i):
                    continue

                # Check V segment (entry signal)
                if self._check_entry(ab.v_segment, open_p, high, low, close, volume, i):
                    entry_price = close[i]
                    entry_bar = i
                    in_trade = True
                    partial_closed = False

                    # Initialize exit parameters from J segment
                    j_params = ab.j_segment.params
                    if ab.j_segment.segment_type == JSegmentType.FIXED_TP:
                        tp_dist = atr * j_params.get("tp_atr_mult", 3.0)
                        sl_dist = atr * j_params.get("sl_atr_mult", 1.0)
                    elif ab.j_segment.segment_type == JSegmentType.TRAILING_STOP:
                        trail_stop = entry_price - direction * atr * j_params.get("trail_atr_mult", 1.5)
                        tp_dist = atr * 5.0  # Wide TP, let trail do the work
                        sl_dist = atr * j_params.get("trail_atr_mult", 1.5)
                    elif ab.j_segment.segment_type == JSegmentType.ATR_TRAILING:
                        trail_stop = entry_price - direction * atr * j_params.get("atr_mult", 2.0)
                        tp_dist = atr * 6.0
                        sl_dist = atr * j_params.get("atr_mult", 2.0)
                    else:
                        tp_dist = atr * 3.0
                        sl_dist = atr * 1.5

            else:
                # In trade: check exit conditions
                bars_held = i - entry_bar
                current_pnl = (close[i] - entry_price) * direction

                # Update trailing stop if applicable
                if ab.j_segment.segment_type in (JSegmentType.TRAILING_STOP, JSegmentType.ATR_TRAILING):
                    if direction > 0:
                        new_trail = close[i] - atr * ab.j_segment.params.get("trail_atr_mult",
                                                     ab.j_segment.params.get("atr_mult", 1.5))
                        trail_stop = max(trail_stop, new_trail)
                    else:
                        new_trail = close[i] + atr * ab.j_segment.params.get("trail_atr_mult",
                                                     ab.j_segment.params.get("atr_mult", 1.5))
                        trail_stop = min(trail_stop, new_trail)

                # Check exit conditions
                exited = False
                exit_pnl = 0.0

                # Stop loss hit
                if current_pnl <= -sl_dist:
                    exit_pnl = -sl_dist - spread
                    exited = True

                # Take profit hit
                elif current_pnl >= tp_dist:
                    exit_pnl = tp_dist - spread
                    exited = True

                # Trailing stop hit
                elif ab.j_segment.segment_type in (JSegmentType.TRAILING_STOP, JSegmentType.ATR_TRAILING):
                    if direction > 0 and close[i] <= trail_stop:
                        exit_pnl = (trail_stop - entry_price) * direction - spread
                        exited = True
                    elif direction < 0 and close[i] >= trail_stop:
                        exit_pnl = (entry_price - trail_stop) * abs(direction) - spread
                        exited = True

                # Time-based exit
                elif ab.j_segment.segment_type == JSegmentType.TIME_BASED:
                    max_bars = int(ab.j_segment.params.get("max_bars", 50))
                    if bars_held >= max_bars:
                        exit_pnl = current_pnl - spread
                        exited = True

                # Partial close logic
                if (not exited and not partial_closed
                        and ab.j_segment.segment_type == JSegmentType.PARTIAL_CLOSE):
                    partial_at = atr * ab.j_segment.params.get("partial_at_atr", 1.5)
                    if current_pnl >= partial_at:
                        # Record partial profit
                        partial_pct = ab.j_segment.params.get("partial_pct", 0.5)
                        partial_profit = current_pnl * partial_pct - spread * 0.5
                        trades.append({
                            "entry_bar": entry_bar,
                            "exit_bar": i,
                            "direction": direction,
                            "pnl": partial_profit,
                            "partial": True,
                        })
                        partial_closed = True
                        # Move SL to breakeven for remainder
                        sl_dist = 0.0

                if exited:
                    # Adjust if partial was already taken
                    if partial_closed:
                        remaining_pct = 1.0 - ab.j_segment.params.get("partial_pct", 0.5)
                        exit_pnl *= remaining_pct

                    trades.append({
                        "entry_bar": entry_bar,
                        "exit_bar": i,
                        "direction": direction,
                        "pnl": exit_pnl,
                        "partial": False,
                    })
                    in_trade = False

        return trades

    def _check_entry(
        self, v: VSegment, open_p, high, low, close, volume, i: int
    ) -> bool:
        """Check if V segment entry condition is met at bar i."""
        p = v.params
        vt = v.segment_type

        if vt == VSegmentType.RSI_OVERSOLD:
            rsi = self._rsi(close[:i+1], int(p.get("period", 14)))
            return rsi < p.get("threshold", 30)

        elif vt == VSegmentType.RSI_OVERBOUGHT:
            rsi = self._rsi(close[:i+1], int(p.get("period", 14)))
            return rsi > p.get("threshold", 70)

        elif vt == VSegmentType.MACD_CROSS_UP:
            fast = int(p.get("fast", 12))
            slow = int(p.get("slow", 26))
            if i < slow + 2:
                return False
            ema_fast_now = self._ema_val(close[:i+1], fast)
            ema_slow_now = self._ema_val(close[:i+1], slow)
            ema_fast_prev = self._ema_val(close[:i], fast)
            ema_slow_prev = self._ema_val(close[:i], slow)
            return (ema_fast_now - ema_slow_now) > 0 and (ema_fast_prev - ema_slow_prev) <= 0

        elif vt == VSegmentType.MACD_CROSS_DOWN:
            fast = int(p.get("fast", 12))
            slow = int(p.get("slow", 26))
            if i < slow + 2:
                return False
            ema_fast_now = self._ema_val(close[:i+1], fast)
            ema_slow_now = self._ema_val(close[:i+1], slow)
            ema_fast_prev = self._ema_val(close[:i], fast)
            ema_slow_prev = self._ema_val(close[:i], slow)
            return (ema_fast_now - ema_slow_now) < 0 and (ema_fast_prev - ema_slow_prev) >= 0

        elif vt == VSegmentType.BB_LOWER_TOUCH:
            period = int(p.get("period", 20))
            std_mult = p.get("std_mult", 2.0)
            if i < period:
                return False
            sma = np.mean(close[i-period+1:i+1])
            std = np.std(close[i-period+1:i+1])
            lower_band = sma - std_mult * std
            return close[i] <= lower_band

        elif vt == VSegmentType.BB_UPPER_TOUCH:
            period = int(p.get("period", 20))
            std_mult = p.get("std_mult", 2.0)
            if i < period:
                return False
            sma = np.mean(close[i-period+1:i+1])
            std = np.std(close[i-period+1:i+1])
            upper_band = sma + std_mult * std
            return close[i] >= upper_band

        elif vt == VSegmentType.EMA_CROSS_UP:
            fast = int(p.get("fast", 8))
            slow = int(p.get("slow", 21))
            if i < slow + 2:
                return False
            f_now = self._ema_val(close[:i+1], fast)
            s_now = self._ema_val(close[:i+1], slow)
            f_prev = self._ema_val(close[:i], fast)
            s_prev = self._ema_val(close[:i], slow)
            return f_now > s_now and f_prev <= s_prev

        elif vt == VSegmentType.EMA_CROSS_DOWN:
            fast = int(p.get("fast", 8))
            slow = int(p.get("slow", 21))
            if i < slow + 2:
                return False
            f_now = self._ema_val(close[:i+1], fast)
            s_now = self._ema_val(close[:i+1], slow)
            f_prev = self._ema_val(close[:i], fast)
            s_prev = self._ema_val(close[:i], slow)
            return f_now < s_now and f_prev >= s_prev

        elif vt == VSegmentType.STOCH_OVERSOLD:
            k_period = int(p.get("k_period", 14))
            threshold = p.get("threshold", 20)
            if i < k_period:
                return False
            highest = np.max(high[i-k_period+1:i+1])
            lowest = np.min(low[i-k_period+1:i+1])
            if highest == lowest:
                return False
            k_val = 100 * (close[i] - lowest) / (highest - lowest)
            return k_val < threshold

        elif vt == VSegmentType.STOCH_OVERBOUGHT:
            k_period = int(p.get("k_period", 14))
            threshold = p.get("threshold", 80)
            if i < k_period:
                return False
            highest = np.max(high[i-k_period+1:i+1])
            lowest = np.min(low[i-k_period+1:i+1])
            if highest == lowest:
                return False
            k_val = 100 * (close[i] - lowest) / (highest - lowest)
            return k_val > threshold

        elif vt == VSegmentType.MOMENTUM_LONG:
            lb = int(p.get("lookback", 10))
            threshold = p.get("threshold", 0.01)
            if i < lb:
                return False
            ret = (close[i] - close[i-lb]) / (close[i-lb] + 1e-10)
            return ret > threshold

        elif vt == VSegmentType.MOMENTUM_SHORT:
            lb = int(p.get("lookback", 10))
            threshold = p.get("threshold", -0.01)
            if i < lb:
                return False
            ret = (close[i] - close[i-lb]) / (close[i-lb] + 1e-10)
            return ret < threshold

        elif vt == VSegmentType.VOLUME_SPIKE_UP:
            lb = int(p.get("lookback", 20))
            spike_mult = p.get("spike_mult", 2.0)
            if i < lb:
                return False
            avg_vol = np.mean(volume[i-lb:i])
            return volume[i] > avg_vol * spike_mult and close[i] > open_p[i]

        elif vt == VSegmentType.MEAN_REVERT_LONG:
            lb = int(p.get("lookback", 20))
            z_thresh = p.get("z_threshold", -2.0)
            if i < lb:
                return False
            sma = np.mean(close[i-lb+1:i+1])
            std = np.std(close[i-lb+1:i+1])
            if std < 1e-10:
                return False
            z = (close[i] - sma) / std
            return z < z_thresh

        elif vt == VSegmentType.MEAN_REVERT_SHORT:
            lb = int(p.get("lookback", 20))
            z_thresh = p.get("z_threshold", 2.0)
            if i < lb:
                return False
            sma = np.mean(close[i-lb+1:i+1])
            std = np.std(close[i-lb+1:i+1])
            if std < 1e-10:
                return False
            z = (close[i] - sma) / std
            return z > z_thresh

        elif vt == VSegmentType.BREAKOUT_HIGH:
            lb = int(p.get("lookback", 20))
            buffer = p.get("buffer_pct", 0.001)
            if i < lb + 1:
                return False
            prev_high = np.max(high[i-lb:i])
            return close[i] > prev_high * (1 + buffer)

        elif vt == VSegmentType.BREAKOUT_LOW:
            lb = int(p.get("lookback", 20))
            buffer = p.get("buffer_pct", 0.001)
            if i < lb + 1:
                return False
            prev_low = np.min(low[i-lb:i])
            return close[i] < prev_low * (1 - buffer)

        elif vt == VSegmentType.CANDLE_ENGULF_UP:
            min_ratio = p.get("min_body_ratio", 0.6)
            if i < 2:
                return False
            prev_body = open_p[i-1] - close[i-1]  # Negative if prev was bearish
            curr_body = close[i] - open_p[i]       # Positive if current is bullish
            curr_range = high[i] - low[i]
            if curr_range < 1e-10:
                return False
            body_ratio = abs(curr_body) / curr_range
            return prev_body < 0 and curr_body > 0 and curr_body > abs(prev_body) and body_ratio > min_ratio

        elif vt == VSegmentType.CANDLE_ENGULF_DN:
            min_ratio = p.get("min_body_ratio", 0.6)
            if i < 2:
                return False
            prev_body = close[i-1] - open_p[i-1]
            curr_body = open_p[i] - close[i]
            curr_range = high[i] - low[i]
            if curr_range < 1e-10:
                return False
            body_ratio = abs(curr_body) / curr_range
            return prev_body > 0 and curr_body > 0 and curr_body > abs(prev_body) and body_ratio > min_ratio

        return False

    def _check_regime(
        self, d: DSegment, open_p, high, low, close, volume, i: int
    ) -> bool:
        """Check if D segment regime condition is met at bar i."""
        p = d.params
        dt = d.segment_type

        if dt == DSegmentType.TRENDING_UP:
            if i < 30:
                return False
            ema_diff_pct = p.get("ema_diff_pct", 0.005)
            ema8 = self._ema_val(close[:i+1], 8)
            ema21 = self._ema_val(close[:i+1], 21)
            diff = (ema8 - ema21) / (ema21 + 1e-10)
            return diff > ema_diff_pct

        elif dt == DSegmentType.TRENDING_DOWN:
            if i < 30:
                return False
            ema_diff_pct = p.get("ema_diff_pct", -0.005)
            ema8 = self._ema_val(close[:i+1], 8)
            ema21 = self._ema_val(close[:i+1], 21)
            diff = (ema8 - ema21) / (ema21 + 1e-10)
            return diff < ema_diff_pct

        elif dt == DSegmentType.RANGING:
            if i < 30:
                return False
            bb_width_max = p.get("bb_width_max", 0.02)
            sma = np.mean(close[i-19:i+1])
            std = np.std(close[i-19:i+1])
            bb_width = (2 * std) / (sma + 1e-10)
            return bb_width < bb_width_max

        elif dt == DSegmentType.VOLATILE:
            if i < 30:
                return False
            atr_ratio_min = p.get("atr_ratio_min", 1.5)
            atr_recent = self._atr_val(high[i-4:i+1], low[i-4:i+1], close[i-4:i+1])
            atr_baseline = self._atr_val(high[i-24:i-4], low[i-24:i-4], close[i-24:i-4])
            if atr_baseline < 1e-10:
                return False
            return atr_recent / atr_baseline > atr_ratio_min

        elif dt == DSegmentType.COMPRESSED:
            if i < 30:
                return False
            atr_ratio_max = p.get("atr_ratio_max", 0.6)
            atr_recent = self._atr_val(high[i-4:i+1], low[i-4:i+1], close[i-4:i+1])
            atr_baseline = self._atr_val(high[i-24:i-4], low[i-24:i-4], close[i-24:i-4])
            if atr_baseline < 1e-10:
                return False
            return atr_recent / atr_baseline < atr_ratio_max

        elif dt == DSegmentType.BREAKOUT:
            if i < 30:
                return False
            # Compression followed by expansion
            atr_5 = self._atr_val(high[i-4:i+1], low[i-4:i+1], close[i-4:i+1])
            atr_prev_5 = self._atr_val(high[i-9:i-4], low[i-9:i-4], close[i-9:i-4])
            atr_20 = self._atr_val(high[i-19:i+1], low[i-19:i+1], close[i-19:i+1])
            if atr_20 < 1e-10 or atr_prev_5 < 1e-10:
                return False
            was_compressed = atr_prev_5 / atr_20 < 0.7
            now_expanding = atr_5 / atr_prev_5 > 1.5
            return was_compressed and now_expanding

        elif dt == DSegmentType.MEAN_REVERTING:
            if i < 30:
                return False
            # Approximate Hurst exponent via R/S analysis
            returns = np.diff(close[i-29:i+1]) / (close[i-29:i] + 1e-10)
            if len(returns) < 10:
                return False
            mean_r = np.mean(returns)
            deviations = np.cumsum(returns - mean_r)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(returns)
            if s < 1e-10:
                return False
            hurst_approx = np.log(r / s + 1e-10) / np.log(len(returns))
            return hurst_approx < p.get("hurst_max", 0.4)

        # Default: regime check passes (permissive)
        return True

    def _compute_fitness_metrics(self, ab: Antibody, trades: List[Dict]):
        """Compute fitness metrics from trade results."""
        if not trades:
            ab.fitness = 0.0
            ab.win_rate = 0.0
            ab.profit_factor = 0.0
            ab.total_trades = 0
            ab.total_pnl = 0.0
            ab.posterior_wr = (PRIOR_ALPHA) / (PRIOR_ALPHA + PRIOR_BETA)
            return

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        ab.total_trades = len(trades)
        ab.total_pnl = sum(pnls)
        ab.win_rate = len(wins) / len(trades) if trades else 0
        ab.avg_win = np.mean(wins) if wins else 0.0
        ab.avg_loss = abs(np.mean(losses)) if losses else 0.0
        ab.profit_factor = ab.avg_win / ab.avg_loss if ab.avg_loss > 0 else (99.0 if ab.avg_win > 0 else 0.0)

        # Bayesian posterior win rate
        ab.posterior_wr = (PRIOR_ALPHA + len(wins)) / (PRIOR_ALPHA + PRIOR_BETA + len(trades))

        # Sharpe ratio (annualized, assuming daily bars)
        if len(pnls) > 1:
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            ab.sharpe_ratio = (mean_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0.0
        else:
            ab.sharpe_ratio = 0.0

        # Max drawdown
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = peak - cum_pnl
        ab.max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Composite fitness function
        # Weighted: posterior_wr * 0.30 + profit_factor * 0.25 + sharpe * 0.25 + trade_count * 0.10 + low_dd * 0.10
        wr_score = ab.posterior_wr
        pf_score = min(1.0, ab.profit_factor / 3.0) if ab.profit_factor > 0 else 0.0
        sharpe_score = min(1.0, max(0.0, ab.sharpe_ratio / 3.0))
        trade_score = min(1.0, ab.total_trades / 50.0)
        dd_score = max(0.0, 1.0 - ab.max_drawdown / (abs(ab.total_pnl) + 1e-10))

        ab.fitness = (
            wr_score * 0.30
            + pf_score * 0.25
            + sharpe_score * 0.25
            + trade_score * 0.10
            + dd_score * 0.10
        )

    # ----------------------------------------------------------
    # STEP 3: CLONAL SELECTION (Kill Losers, Clone Winners)
    # ----------------------------------------------------------

    def clonal_selection(
        self,
        antibodies: List[Antibody] = None,
        n_survivors: int = ELITE_SURVIVORS,
    ) -> Tuple[List[Antibody], List[Antibody]]:
        """
        Clonal selection: test all antibodies, kill the losers, keep the winners.

        Selection criteria (must meet ALL to survive):
          1. Minimum number of trades (proves it actually fires)
          2. Posterior win rate above threshold
          3. Profit factor above threshold

        Antibodies that fail undergo apoptosis (programmed cell death).

        Returns:
            (survivors, dead) -- lists of surviving and dead antibodies
        """
        if antibodies is None:
            antibodies = self.population

        survivors = []
        dead = []

        for ab in antibodies:
            if (ab.total_trades >= MIN_TRADES_FOR_FITNESS
                    and ab.posterior_wr >= MIN_WIN_RATE_SURVIVE
                    and ab.profit_factor >= MIN_PROFIT_FACTOR_SURVIVE):
                survivors.append(ab)
            else:
                dead.append(ab)

        # Sort survivors by fitness and keep top N
        survivors.sort(key=lambda a: a.fitness, reverse=True)
        if len(survivors) > n_survivors:
            dead.extend(survivors[n_survivors:])
            survivors = survivors[:n_survivors]

        log.info(
            "[VDJ] Clonal selection: %d survived / %d died (%.1f%% survival rate)",
            len(survivors), len(dead),
            100 * len(survivors) / max(1, len(survivors) + len(dead)),
        )

        if survivors:
            log.info(
                "[VDJ] Best survivor: %s | WR=%.1f%% PF=%.2f trades=%d",
                survivors[0].antibody_id[:8],
                survivors[0].posterior_wr * 100,
                survivors[0].profit_factor,
                survivors[0].total_trades,
            )

        return survivors, dead

    # ----------------------------------------------------------
    # STEP 4: AFFINITY MATURATION (Somatic Hypermutation)
    # ----------------------------------------------------------

    def affinity_maturation(
        self,
        survivors: List[Antibody],
        bars: np.ndarray,
        rounds: int = MATURATION_ROUNDS,
        spread_points: float = 2.0,
    ) -> List[Antibody]:
        """
        Affinity maturation: mutate parameters of winning antibodies to
        find even better versions. This mimics somatic hypermutation in
        germinal centers.

        For each survivor:
          1. Create N mutant copies with small parameter changes
          2. Evaluate all mutants
          3. If a mutant is better than the parent, it replaces the parent
          4. Repeat for multiple rounds (affinity maturation cycles)

        Returns:
            List of matured antibodies (may be same or improved versions).
        """
        if not survivors:
            return survivors

        matured = []
        total_improvements = 0

        for parent in survivors:
            best = deepcopy(parent)

            for round_num in range(rounds):
                # Generate mutant offspring
                mutant = self._hypermutate(best)

                # Evaluate mutant
                self.evaluate_fitness(bars, [mutant], spread_points)

                # Selection: better mutant replaces parent
                if mutant.fitness > best.fitness:
                    mutant.parent_id = best.antibody_id
                    mutant.mutation_count = best.mutation_count + 1
                    best = mutant
                    total_improvements += 1

            matured.append(best)

        log.info(
            "[VDJ] Affinity maturation: %d rounds x %d survivors = %d improvements",
            rounds, len(survivors), total_improvements,
        )

        return matured

    def _hypermutate(self, parent: Antibody) -> Antibody:
        """
        Create a mutant copy of an antibody with small parameter changes.
        This is somatic hypermutation: random point mutations in the
        variable region to improve binding affinity.
        """
        child = deepcopy(parent)

        # Mutate V segment params
        for key, val in child.v_segment.params.items():
            if isinstance(val, (int, float)) and self.rng.random() < MATURATION_MUTATION_RATE:
                jitter = self.rng.gauss(0, MATURATION_PARAM_JITTER)
                new_val = val * (1.0 + jitter)
                # Keep integer params as integers
                child.v_segment.params[key] = int(round(new_val)) if isinstance(val, int) else new_val

        # Mutate D segment params
        for key, val in child.d_segment.params.items():
            if isinstance(val, (int, float)) and self.rng.random() < MATURATION_MUTATION_RATE:
                jitter = self.rng.gauss(0, MATURATION_PARAM_JITTER)
                new_val = val * (1.0 + jitter)
                child.d_segment.params[key] = int(round(new_val)) if isinstance(val, int) else new_val

        # Mutate J segment params
        for key, val in child.j_segment.params.items():
            if isinstance(val, (int, float)) and self.rng.random() < MATURATION_MUTATION_RATE:
                jitter = self.rng.gauss(0, MATURATION_PARAM_JITTER)
                new_val = val * (1.0 + jitter)
                child.j_segment.params[key] = int(round(new_val)) if isinstance(val, int) else new_val

        # Recompute ID (mutation changes the antibody)
        child.antibody_id = child.compute_id()
        child.generation = parent.generation
        child.fitness = 0.0  # Must be re-evaluated

        return child

    # ----------------------------------------------------------
    # STEP 5: MEMORY B CELL FORMATION
    # ----------------------------------------------------------

    def form_memory_cells(self, matured: List[Antibody]) -> List[Antibody]:
        """
        Promote the best matured antibodies to memory B cell status.
        Memory cells are persisted in the database and recalled for
        rapid deployment when similar market conditions return.

        Criteria for memory promotion:
          - Posterior win rate >= 65%
          - Profit factor >= 1.5
          - Minimum 20 trades
        """
        new_memory = []

        for ab in matured:
            if (ab.posterior_wr >= MEMORY_MIN_WIN_RATE
                    and ab.profit_factor >= MEMORY_MIN_PROFIT_FACTOR
                    and ab.total_trades >= MEMORY_MIN_TRADES):
                ab.is_memory_cell = True
                ab.memory_since = datetime.now().isoformat()
                new_memory.append(ab)
                self._save_antibody(ab)

        # Update in-memory list
        existing_ids = {m.antibody_id for m in self.memory_cells}
        for ab in new_memory:
            if ab.antibody_id not in existing_ids:
                self.memory_cells.append(ab)

        # Trim memory to limit
        if len(self.memory_cells) > MEMORY_CELL_LIMIT:
            self.memory_cells.sort(key=lambda a: a.fitness, reverse=True)
            self.memory_cells = self.memory_cells[:MEMORY_CELL_LIMIT]

        log.info(
            "[VDJ] Memory B cells: %d new, %d total stored",
            len(new_memory), len(self.memory_cells),
        )

        return new_memory

    # ----------------------------------------------------------
    # FULL PIPELINE: Recombine -> Evaluate -> Select -> Mature -> Remember
    # ----------------------------------------------------------

    def run_full_cycle(
        self,
        bars: np.ndarray,
        population_size: int = None,
        spread_points: float = 2.0,
    ) -> Dict:
        """
        Run the complete VDJ immune response cycle:

        1. V(D)J Recombination: generate diverse antibody pool
        2. Antigen Testing: evaluate fitness via backtesting
        3. Clonal Selection: kill losers, keep winners
        4. Affinity Maturation: optimize winner parameters
        5. Memory Formation: persist the best permanently

        Args:
            bars: OHLCV numpy array (N x 5)
            population_size: number of antibodies to generate
            spread_points: trading cost

        Returns:
            Dict with cycle statistics and results.
        """
        t_start = time.time()
        n = population_size or self.population_size

        # Step 1: Recombination
        antibodies = self.recombine(n)

        # Step 2: Fitness evaluation
        antibodies = self.evaluate_fitness(bars, antibodies, spread_points)

        # Step 3: Clonal selection
        survivors, dead = self.clonal_selection(antibodies)

        # Step 4: Affinity maturation
        matured = self.affinity_maturation(survivors, bars, spread_points=spread_points)

        # Re-evaluate matured antibodies for final ranking
        matured = self.evaluate_fitness(bars, matured, spread_points)

        # Step 5: Memory formation
        new_memory = self.form_memory_cells(matured)

        elapsed = time.time() - t_start

        # Generation statistics
        gen_stats = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": n,
            "survivors": len(survivors),
            "memory_cells_added": len(new_memory),
            "avg_fitness": float(np.mean([a.fitness for a in matured])) if matured else 0.0,
            "best_fitness": matured[0].fitness if matured else 0.0,
            "best_antibody_id": matured[0].antibody_id if matured else "",
            "maturation_improvements": sum(1 for m, s in zip(matured, survivors) if m.fitness > s.fitness),
            "total_memory_cells": len(self.memory_cells),
            "elapsed_seconds": elapsed,
        }
        self._save_generation(gen_stats)

        log.info(
            "[VDJ] Full cycle complete: gen=%d | %d generated -> %d survived -> %d matured -> %d memorized | %.1fs",
            self.generation, n, len(survivors), len(matured), len(new_memory), elapsed,
        )

        return {
            "generation_stats": gen_stats,
            "matured_antibodies": matured,
            "new_memory_cells": new_memory,
            "all_memory_cells": self.memory_cells,
            "dead_count": len(dead),
        }

    # ----------------------------------------------------------
    # SIGNAL GENERATION (for live trading integration)
    # ----------------------------------------------------------

    def get_active_antibody_signals(
        self,
        bars: np.ndarray,
    ) -> List[Dict]:
        """
        Query all memory B cells for current signals.
        This is the "secondary immune response" -- rapid recall of
        known-good antibodies when the pathogen (market condition) returns.

        Returns list of active antibody signals with direction and confidence.
        """
        if not self.memory_cells:
            return []

        close = bars[:, 3]
        high = bars[:, 1]
        low = bars[:, 2]
        open_p = bars[:, 0]
        volume = bars[:, 4] if bars.shape[1] > 4 else np.ones(len(close))

        active_signals = []
        i = len(close) - 1  # Current bar

        for ab in self.memory_cells:
            # Check regime filter
            if not self._check_regime(ab.d_segment, open_p, high, low, close, volume, i):
                continue

            # Check entry signal
            if self._check_entry(ab.v_segment, open_p, high, low, close, volume, i):
                active_signals.append({
                    "antibody_id": ab.antibody_id,
                    "direction": ab.get_direction(),
                    "confidence": ab.fitness,
                    "win_rate": ab.posterior_wr,
                    "profit_factor": ab.profit_factor,
                    "v_type": ab.v_segment.segment_type.value,
                    "d_type": ab.d_segment.segment_type.value,
                    "j_type": ab.j_segment.segment_type.value,
                    "te_fingerprint": ab.get_te_fingerprint(),
                    "total_trades": ab.total_trades,
                })

        return active_signals

    def get_antibody_consensus(
        self,
        bars: np.ndarray,
    ) -> Dict:
        """
        Compute consensus signal from all active memory antibodies.
        This is analogous to the polyclonal immune response where
        multiple antibody types converge on the same target.

        Returns:
            {direction, confidence, n_active, n_long, n_short, details}
        """
        signals = self.get_active_antibody_signals(bars)

        if not signals:
            return {
                "direction": 0,
                "confidence": 0.0,
                "n_active": 0,
                "n_long": 0,
                "n_short": 0,
                "signals": [],
            }

        n_long = sum(1 for s in signals if s["direction"] > 0)
        n_short = sum(1 for s in signals if s["direction"] < 0)

        # Weighted vote by fitness
        weighted_long = sum(s["confidence"] for s in signals if s["direction"] > 0)
        weighted_short = sum(s["confidence"] for s in signals if s["direction"] < 0)
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
            "n_active": len(signals),
            "n_long": n_long,
            "n_short": n_short,
            "weighted_long": float(weighted_long),
            "weighted_short": float(weighted_short),
            "signals": signals,
        }

    # ----------------------------------------------------------
    # HELPER FUNCTIONS
    # ----------------------------------------------------------

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> float:
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
    def _ema_val(data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(np.mean(data))
        mult = 2.0 / (period + 1)
        ema = float(np.mean(data[:period]))
        for val in data[period:]:
            ema = (float(val) - ema) * mult + ema
        return ema

    @staticmethod
    def _compute_atr(high, low, close, period=14) -> np.ndarray:
        """Compute ATR array."""
        n = len(close)
        atr = np.zeros(n)
        for i in range(1, n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1]),
            )
            if i < period:
                atr[i] = tr
            else:
                atr[i] = (atr[i-1] * (period - 1) + tr) / period
        return atr

    @staticmethod
    def _atr_val(high, low, close) -> float:
        """Compute single ATR value."""
        if len(high) < 2:
            return float(high[0] - low[0]) if len(high) > 0 else 0.0
        trs = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1]),
            )
            trs.append(tr)
        return float(np.mean(trs)) if trs else 0.0


# ============================================================
# TEQA INTEGRATION: VDJ Bridge
# ============================================================

class VDJTEQABridge:
    """
    Bridges VDJ Recombination engine with the existing TEQA v3.0 system.

    The bridge:
      1. Maps antibody signals to TE activation patterns
      2. Feeds antibody consensus into the TEQA pipeline as an additional gate
      3. Records trade outcomes back to the VDJ system for re-evaluation
      4. Writes antibody signals to JSON for MQL5 EA consumption

    This is how the domesticated Transib (RAG1/RAG2) feeds back into
    the broader transposon ecosystem.
    """

    def __init__(
        self,
        vdj_engine: VDJRecombinationEngine,
        signal_file: str = None,
    ):
        self.vdj = vdj_engine
        if signal_file is None:
            self.signal_file = str(
                Path(__file__).parent / "vdj_antibody_signal.json"
            )
        else:
            self.signal_file = signal_file

    def get_vdj_gate_result(self, bars: np.ndarray) -> Dict:
        """
        Compute VDJ antibody gate result for integration with Jardine's Gate.

        This becomes Gate G11 (VDJ Immune Response) in the extended gate system.

        Returns:
            {
                "gate_pass": bool,
                "direction": int,
                "confidence": float,
                "n_active_antibodies": int,
                "antibody_consensus": dict,
            }
        """
        consensus = self.vdj.get_antibody_consensus(bars)

        # Gate passes if:
        # 1. At least 2 antibodies are active (polyclonal response)
        # 2. Consensus confidence exceeds 0.6
        gate_pass = (
            consensus["n_active"] >= 2
            and consensus["confidence"] >= 0.6
        )

        return {
            "gate_pass": gate_pass,
            "direction": consensus["direction"],
            "confidence": consensus["confidence"],
            "n_active_antibodies": consensus["n_active"],
            "antibody_consensus": consensus,
        }

    def write_signal_file(self, bars: np.ndarray):
        """
        Write current antibody signals to JSON for MQL5 EA consumption.

        The MQL5 EA reads this file to incorporate immune response signals
        into its trading decisions.
        """
        consensus = self.vdj.get_antibody_consensus(bars)
        gate = self.get_vdj_gate_result(bars)

        signal = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "direction": consensus["direction"],
            "confidence": consensus["confidence"],
            "gate_pass": gate["gate_pass"],
            "n_active": consensus["n_active"],
            "n_long": consensus["n_long"],
            "n_short": consensus["n_short"],
            "weighted_long": consensus.get("weighted_long", 0),
            "weighted_short": consensus.get("weighted_short", 0),
            "memory_cells_total": len(self.vdj.memory_cells),
            "generation": self.vdj.generation,
            # Individual antibody signals for the EA
            "antibodies": [
                {
                    "id": s["antibody_id"][:8],
                    "dir": s["direction"],
                    "conf": round(s["confidence"], 4),
                    "wr": round(s["win_rate"], 4),
                    "pf": round(s["profit_factor"], 2),
                    "v": s["v_type"],
                    "d": s["d_type"],
                    "j": s["j_type"],
                }
                for s in consensus.get("signals", [])
            ],
        }

        try:
            tmp_path = self.signal_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(signal, f, indent=2)
            os.replace(tmp_path, self.signal_file)
        except Exception as e:
            log.warning("Failed to write VDJ signal file: %s", e)

    def record_trade_outcome(self, antibody_id: str, won: bool, pnl: float = 0.0):
        """
        Record a trade outcome for a specific antibody.
        This feeds back into the immune system for continued learning.
        """
        for ab in self.vdj.memory_cells:
            if ab.antibody_id == antibody_id or ab.antibody_id[:8] == antibody_id:
                if won:
                    ab.total_trades += 1
                    # Update running stats
                    ab.posterior_wr = (
                        (PRIOR_ALPHA + ab.total_trades * ab.win_rate + 1)
                        / (PRIOR_ALPHA + PRIOR_BETA + ab.total_trades + 1)
                    )
                else:
                    ab.total_trades += 1
                    ab.posterior_wr = (
                        (PRIOR_ALPHA + ab.total_trades * ab.win_rate)
                        / (PRIOR_ALPHA + PRIOR_BETA + ab.total_trades + 1)
                    )
                self.vdj._save_antibody(ab)
                break


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
    print("  VDJ RECOMBINATION ENGINE -- Adaptive Immune Trading System")
    print("  RAG1/RAG2 Domesticated Transib -> Vertebrate Adaptive Immunity")
    print("=" * 76)

    # Generate synthetic OHLCV data with a trend + mean-reversion pattern
    np.random.seed(42)
    n_bars = 500
    # Create a market with a slight upward trend and mean-reversion
    returns = np.random.randn(n_bars) * 0.008 + 0.0002  # Slight drift
    # Add mean-reversion autocorrelation
    for i in range(1, len(returns)):
        returns[i] -= 0.3 * returns[i-1]  # Mean-reverting component
    close = 50000 + np.cumsum(returns) * 50000
    close = np.maximum(close, 100)  # Floor

    high = close + np.abs(np.random.randn(n_bars) * close * 0.003)
    low = close - np.abs(np.random.randn(n_bars) * close * 0.003)
    open_p = close + np.random.randn(n_bars) * close * 0.001
    volume = np.abs(np.random.randn(n_bars) * 100 + 500)
    bars = np.column_stack([open_p, high, low, close, volume])

    # Use temp DB for test
    test_db = str(Path(__file__).parent / "test_vdj_antibodies.db")

    print(f"\n  Synthetic data: {n_bars} bars (BTC-like)")
    print(f"  V segments: {len(list(VSegmentType))} entry signals")
    print(f"  D segments: {len(list(DSegmentType))} regime classifiers")
    print(f"  J segments: {len(list(JSegmentType))} exit strategies")
    print(f"  Combinatorial space: {len(list(VSegmentType)) * len(list(DSegmentType)) * len(list(JSegmentType))} base combinations")
    print(f"  (+ junctional diversity = effectively infinite)")

    # Initialize engine
    engine = VDJRecombinationEngine(
        db_path=test_db,
        population_size=100,
        seed=42,
    )

    print("\n  --- STEP 1: V(D)J RECOMBINATION ---")
    print("  Generating 100 antibodies...")
    antibodies = engine.recombine(100)
    print(f"  Generated {len(antibodies)} unique antibodies")

    # Show a few examples
    print("\n  Sample antibodies:")
    for ab in antibodies[:5]:
        d = ab.get_direction()
        dir_str = "LONG" if d > 0 else "SHORT"
        print(f"    [{ab.antibody_id[:8]}] {dir_str}: "
              f"V={ab.v_segment.segment_type.value} | "
              f"D={ab.d_segment.segment_type.value} | "
              f"J={ab.j_segment.segment_type.value} | "
              f"TEs={ab.get_te_fingerprint()}")

    print("\n  --- STEP 2: ANTIGEN TESTING (Backtest) ---")
    antibodies = engine.evaluate_fitness(bars, antibodies)
    active_abs = [a for a in antibodies if a.total_trades > 0]
    print(f"  {len(active_abs)} antibodies generated trades")
    print(f"  Best fitness: {antibodies[0].fitness:.4f}")
    print(f"  Avg trades per antibody: {np.mean([a.total_trades for a in antibodies]):.1f}")

    print("\n  Top 5 antibodies after testing:")
    for ab in antibodies[:5]:
        print(f"    {ab.summary()}")

    print("\n  --- STEP 3: CLONAL SELECTION ---")
    survivors, dead = engine.clonal_selection(antibodies)
    print(f"  Survivors: {len(survivors)}")
    print(f"  Dead (apoptosis): {len(dead)}")

    if survivors:
        print("\n  Surviving antibodies:")
        for ab in survivors:
            print(f"    {ab.summary()}")

    print("\n  --- STEP 4: AFFINITY MATURATION ---")
    matured = engine.affinity_maturation(survivors, bars)
    matured = engine.evaluate_fitness(bars, matured)

    print("  Matured antibodies:")
    for i, (orig, mat) in enumerate(zip(survivors[:len(matured)], matured)):
        improvement = mat.fitness - orig.fitness
        print(f"    {mat.summary()} | improvement={improvement:+.4f}")

    print("\n  --- STEP 5: MEMORY B CELL FORMATION ---")
    new_memory = engine.form_memory_cells(matured)
    print(f"  New memory cells: {len(new_memory)}")
    print(f"  Total memory cells: {len(engine.memory_cells)}")

    if engine.memory_cells:
        print("\n  Memory B cells (persistent winners):")
        for mc in engine.memory_cells:
            print(f"    {mc.summary()}")

    print("\n  --- LIVE SIGNAL TEST ---")
    consensus = engine.get_antibody_consensus(bars)
    dir_str = "LONG" if consensus["direction"] > 0 else ("SHORT" if consensus["direction"] < 0 else "NEUTRAL")
    print(f"  Direction:  {dir_str}")
    print(f"  Confidence: {consensus['confidence']:.4f}")
    print(f"  Active:     {consensus['n_active']} antibodies")
    print(f"  Long:       {consensus['n_long']}")
    print(f"  Short:      {consensus['n_short']}")

    # Write signal file
    bridge = VDJTEQABridge(engine)
    bridge.write_signal_file(bars)
    print(f"\n  Signal file written to: {bridge.signal_file}")

    # Cleanup test DB
    try:
        os.remove(test_db)
    except OSError:
        pass

    print("\n" + "=" * 76)
    print("  VDJ Recombination Engine test complete.")
    print("=" * 76)
