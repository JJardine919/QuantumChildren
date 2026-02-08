"""
PROTECTIVE DELETION ENGINE -- Loss-of-Function Defense
========================================================
Algorithm #2 in the Quantum Children biological algorithm series.

This is the INVERSE of TE domestication. Where domestication BOOSTS signal
combos that precede wins (gain-of-function), Protective Deletion SUPPRESSES
signal combos that precede losses (loss-of-function defense).

Biological basis:
    CCR5-delta32 deletion provides HIV resistance by removing the receptor.
    Heterozygous (one copy deleted) = partial resistance.
    Homozygous (both copies deleted) = near-complete resistance.

    In trading: toxic TE combos get their signal strength "deleted" --
    reduced to 0.5x (heterozygous) or 0.1x (homozygous) of normal.

Integration:
    - Reads TE activations from TEActivationEngine (33 families)
    - Shares pattern hash scheme with TEDomesticationTracker
    - Applied at the same point as get_boost() in TEQA pipeline
    - Combined modifier = boost * suppression (safety-first: suppression wins conflicts)

Authors: DooDoo + Claude
Date:    2026-02-08
Version: PROTECTIVE-DELETION-1.0
"""

import hashlib
import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import from existing TEQA system
from teqa_v3_neural_te import (
    TEDomesticationTracker,
    N_QUBITS,
)

# Import trading config values from config_loader (CLAUDE.md rule: no hardcoded trading values)
from config_loader import MAX_LOSS_DOLLARS, CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)

VERSION = "PROTECTIVE-DELETION-1.0"

# ============================================================
# CONSTANTS
# ============================================================

# Toxic pattern detection thresholds
DELETION_HETEROZYGOUS_MIN_TRADES = 15     # Stage 1: partial evidence required
DELETION_HETEROZYGOUS_MIN_LR = 0.65       # Stage 1: loss rate >= 65% (posterior)
DELETION_HETEROZYGOUS_FACTOR = 0.50       # Stage 1: half signal strength

DELETION_HOMOZYGOUS_MIN_TRADES = 25       # Stage 2: strong evidence required
DELETION_HOMOZYGOUS_MIN_LR = 0.70         # Stage 2: loss rate >= 70% (posterior)
DELETION_HOMOZYGOUS_FACTOR = 0.10         # Stage 2: 90% signal reduction

# Hysteresis: recovery requires loss rate to drop well below flagging threshold
DELETION_RECOVERY_LR = 0.50              # Unflag only when posterior LR < 50%

# Bayesian prior for loss rate estimation (mirrors domestication's Beta prior)
# Beta(10, 10) starts at 50% expectation, requires real evidence to shift
DELETION_PRIOR_ALPHA = 10                 # Prior "loss" observations
DELETION_PRIOR_BETA = 10                  # Prior "win" observations

# Drawdown pressure thresholds (% of peak equity)
DRAWDOWN_PRESSURE_MILD = 0.03            # 3% drawdown
DRAWDOWN_PRESSURE_MODERATE = 0.05        # 5% drawdown
DRAWDOWN_PRESSURE_SEVERE = 0.08          # 8% drawdown

# How much to reduce thresholds under drawdown pressure
DRAWDOWN_LR_REDUCTION_MILD = 0.03        # Lower LR threshold by 3%
DRAWDOWN_LR_REDUCTION_MODERATE = 0.05    # Lower LR threshold by 5%
DRAWDOWN_LR_REDUCTION_SEVERE = 0.08      # Lower LR threshold by 8%
DRAWDOWN_TRADES_REDUCTION_MILD = 2       # Reduce min trades by 2
DRAWDOWN_TRADES_REDUCTION_MODERATE = 4   # Reduce min trades by 4
DRAWDOWN_TRADES_REDUCTION_SEVERE = 6     # Reduce min trades by 6

# Minimum floors for adaptive thresholds (never go below these)
DELETION_MIN_TRADES_FLOOR = 8            # Never flag with fewer than 8 trades
DELETION_MIN_HOM_TRADES_FLOOR = 15       # Never fully suppress with < 15 trades

# Allele frequency monitoring thresholds
ALLELE_FREQ_WARNING_THRESHOLD = 0.30     # Warn if > 30% of patterns suppressed
ALLELE_FREQ_CRITICAL_THRESHOLD = 0.50    # Critical: > 50% suppressed

# Quantum integration
QUANTUM_SUPPRESSION_ANGLE_SCALE = 0.3    # Dampen qubit rotation for suppressed TEs

# Heterozygote advantage tracking
HET_ADVANTAGE_EVAL_TRADES = 50           # Minimum trades to evaluate het vs hom
HET_ADVANTAGE_TRACKING_DAYS = 30         # Window for advantage tracking

# Expiry: suppressed patterns get a second chance after prolonged inactivity
DELETION_EXPIRY_DAYS = 45                # Longer than domestication (deletion is cautious)


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class DeletionStage(Enum):
    """The allele state of a pattern's deletion."""
    NONE = "none"                    # Wild-type: no deletion
    HETEROZYGOUS = "heterozygous"    # One copy deleted: partial suppression
    HOMOZYGOUS = "homozygous"        # Both copies deleted: near-complete suppression


class PressureLevel(Enum):
    """Drawdown pressure intensity."""
    NONE = "NONE"
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


@dataclass
class PressureResult:
    """Result from drawdown pressure computation."""
    lr_reduction: float = 0.0         # How much to lower loss-rate thresholds
    trades_reduction: int = 0         # How much to lower min-trades thresholds
    pressure_level: PressureLevel = PressureLevel.NONE
    drawdown_pct: float = 0.0


@dataclass
class AlleleFrequencyReport:
    """Population-level health metric for suppressed patterns."""
    total_patterns: int = 0
    het_patterns: int = 0
    hom_patterns: int = 0
    suppressed_count: int = 0
    allele_frequency: float = 0.0     # Fraction of patterns that are suppressed
    health: str = "HEALTHY"           # HEALTHY / WARNING / CRITICAL
    avg_suppression: float = 1.0      # Mean suppression factor across suppressed patterns
    worst_pattern_hash: str = ""
    worst_pattern_combo: str = ""
    worst_posterior_lr: float = 0.0
    timestamp: str = ""


@dataclass
class HetAdvantageRecord:
    """Tracks whether partial suppression outperforms full deletion."""
    pattern_hash: str = ""
    het_period_pnl: float = 0.0
    het_period_trades: int = 0
    hom_period_pnl: float = 0.0
    hom_period_trades: int = 0
    advantage_detected: bool = False
    last_evaluated: str = ""


# ============================================================
# DRAWDOWN PRESSURE ENGINE
# ============================================================

class DrawdownPressure:
    """
    Monitors account equity and computes adaptive pressure on deletion thresholds.

    Under drawdown stress, the system becomes more aggressive at detecting
    and suppressing toxic patterns. This mirrors how selective pressure
    intensifies in hostile environments -- organisms under stress evolve
    faster because the cost of NOT adapting is death.

    The drawdown percentage is computed externally (from MT5 account equity)
    and passed in. This class translates it into threshold adjustments.
    """

    def __init__(self):
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._drawdown_pct: float = 0.0
        self._pressure_history: List[Dict] = []

    def update_equity(self, equity: float) -> float:
        """
        Update equity tracking and return current drawdown percentage.

        Args:
            equity: Current account equity from MT5.

        Returns:
            Current drawdown as a fraction (0.05 = 5% drawdown).
        """
        if equity > self._peak_equity:
            self._peak_equity = equity
        self._current_equity = equity

        if self._peak_equity > 0:
            self._drawdown_pct = (self._peak_equity - equity) / self._peak_equity
        else:
            self._drawdown_pct = 0.0

        return self._drawdown_pct

    def compute_pressure(self, drawdown_pct: float = None) -> PressureResult:
        """
        Compute deletion threshold adjustments based on drawdown severity.

        Under stress, thresholds are lowered so toxic patterns get flagged
        faster. This is the "evolutionary pressure" that accelerates adaptation.

        Args:
            drawdown_pct: Override drawdown percentage (otherwise uses internal tracking).

        Returns:
            PressureResult with threshold adjustments.
        """
        dd = drawdown_pct if drawdown_pct is not None else self._drawdown_pct

        if dd >= DRAWDOWN_PRESSURE_SEVERE:
            result = PressureResult(
                lr_reduction=DRAWDOWN_LR_REDUCTION_SEVERE,
                trades_reduction=DRAWDOWN_TRADES_REDUCTION_SEVERE,
                pressure_level=PressureLevel.SEVERE,
                drawdown_pct=dd,
            )
        elif dd >= DRAWDOWN_PRESSURE_MODERATE:
            result = PressureResult(
                lr_reduction=DRAWDOWN_LR_REDUCTION_MODERATE,
                trades_reduction=DRAWDOWN_TRADES_REDUCTION_MODERATE,
                pressure_level=PressureLevel.MODERATE,
                drawdown_pct=dd,
            )
        elif dd >= DRAWDOWN_PRESSURE_MILD:
            result = PressureResult(
                lr_reduction=DRAWDOWN_LR_REDUCTION_MILD,
                trades_reduction=DRAWDOWN_TRADES_REDUCTION_MILD,
                pressure_level=PressureLevel.MILD,
                drawdown_pct=dd,
            )
        else:
            result = PressureResult(
                lr_reduction=0.0,
                trades_reduction=0,
                pressure_level=PressureLevel.NONE,
                drawdown_pct=dd,
            )

        # Log pressure changes
        self._pressure_history.append({
            "timestamp": datetime.now().isoformat(),
            "drawdown_pct": dd,
            "pressure_level": result.pressure_level.value,
            "lr_reduction": result.lr_reduction,
            "trades_reduction": result.trades_reduction,
        })

        # Keep history bounded
        if len(self._pressure_history) > 1000:
            self._pressure_history = self._pressure_history[-500:]

        return result

    @property
    def drawdown_pct(self) -> float:
        return self._drawdown_pct

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    def get_pressure_summary(self) -> Dict:
        """Return current pressure state for logging."""
        return {
            "peak_equity": self._peak_equity,
            "current_equity": self._current_equity,
            "drawdown_pct": round(self._drawdown_pct, 4),
            "pressure_level": self.compute_pressure().pressure_level.value,
            "history_length": len(self._pressure_history),
        }


# ============================================================
# ALLELE FREQUENCY MONITOR
# ============================================================

class AlleleFrequencyMonitor:
    """
    Tracks what percentage of observed TE patterns are currently suppressed.

    This is a population-level health metric. If too many patterns are toxic,
    it suggests either:
      1. The system is over-suppressing (thresholds too aggressive), or
      2. The market regime has fundamentally shifted (all old patterns are bad)

    In genetics, high allele frequency of a deletion variant can indicate
    strong selective pressure (the deletion is very beneficial) or genetic
    drift (random chance in a small population).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def compute_frequency(self) -> AlleleFrequencyReport:
        """
        Compute current allele frequency of suppressed patterns.

        Returns:
            AlleleFrequencyReport with population-level metrics.
        """
        report = AlleleFrequencyReport(timestamp=datetime.now().isoformat())

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()

                # Total patterns observed
                cursor.execute("SELECT COUNT(*) FROM protective_deletions")
                report.total_patterns = cursor.fetchone()[0]

                if report.total_patterns == 0:
                    return report

                # Heterozygous deletions
                cursor.execute(
                    "SELECT COUNT(*) FROM protective_deletions WHERE deletion_stage=?",
                    (DeletionStage.HETEROZYGOUS.value,)
                )
                report.het_patterns = cursor.fetchone()[0]

                # Homozygous deletions
                cursor.execute(
                    "SELECT COUNT(*) FROM protective_deletions WHERE deletion_stage=?",
                    (DeletionStage.HOMOZYGOUS.value,)
                )
                report.hom_patterns = cursor.fetchone()[0]

                report.suppressed_count = report.het_patterns + report.hom_patterns
                report.allele_frequency = report.suppressed_count / report.total_patterns

                # Average suppression factor across suppressed patterns
                cursor.execute(
                    "SELECT AVG(suppression_factor) FROM protective_deletions "
                    "WHERE deletion_stage != ?",
                    (DeletionStage.NONE.value,)
                )
                row = cursor.fetchone()
                report.avg_suppression = float(row[0]) if row and row[0] is not None else 1.0

                # Worst pattern (highest posterior loss rate)
                cursor.execute(
                    "SELECT pattern_hash, te_combo, posterior_lr "
                    "FROM protective_deletions "
                    "ORDER BY posterior_lr DESC LIMIT 1"
                )
                worst = cursor.fetchone()
                if worst:
                    report.worst_pattern_hash = worst[0]
                    report.worst_pattern_combo = worst[1]
                    report.worst_posterior_lr = float(worst[2]) if worst[2] else 0.0

                # Health classification
                if report.allele_frequency > ALLELE_FREQ_CRITICAL_THRESHOLD:
                    report.health = "CRITICAL"
                elif report.allele_frequency > ALLELE_FREQ_WARNING_THRESHOLD:
                    report.health = "WARNING"
                else:
                    report.health = "HEALTHY"

        except Exception as e:
            log.warning("Allele frequency computation failed: %s", e)

        return report


# ============================================================
# TOXIC PATTERN DETECTOR
# ============================================================

class ToxicPatternDetector:
    """
    Detects toxic TE signal combinations that consistently precede losing trades.

    Uses two-stage classification with Bayesian posterior loss rate estimation
    and hysteresis to prevent oscillation at threshold boundaries.

    Stage 1 (Heterozygous): posterior_lr >= 0.65 AND trades >= 15
        -> suppression_factor = 0.50

    Stage 2 (Homozygous): posterior_lr >= 0.70 AND trades >= 25
        -> suppression_factor = 0.10

    Recovery: posterior_lr must drop below 0.50 to unflag (hysteresis gap).
    """

    def __init__(self, drawdown_pressure: DrawdownPressure = None):
        self.drawdown_pressure = drawdown_pressure or DrawdownPressure()

    def classify_pattern(
        self,
        win_count: int,
        loss_count: int,
        posterior_lr: float,
        current_stage: DeletionStage,
        account_drawdown_pct: float = 0.0,
        het_advantage: bool = False,
    ) -> Tuple[DeletionStage, float]:
        """
        Classify a pattern's deletion stage based on its statistics.

        Implements two-stage detection with hysteresis and drawdown-adaptive thresholds.

        Args:
            win_count: Total wins for this pattern.
            loss_count: Total losses for this pattern.
            posterior_lr: Bayesian posterior loss rate.
            current_stage: Current deletion stage (for hysteresis).
            account_drawdown_pct: Current account drawdown (0.0 to 1.0).
            het_advantage: Whether heterozygote advantage has been detected.

        Returns:
            (new_stage, suppression_factor)
        """
        total = win_count + loss_count

        # Compute drawdown-adaptive thresholds
        pressure = self.drawdown_pressure.compute_pressure(account_drawdown_pct)
        effective_het_lr = DELETION_HETEROZYGOUS_MIN_LR - pressure.lr_reduction
        effective_hom_lr = DELETION_HOMOZYGOUS_MIN_LR - pressure.lr_reduction
        effective_het_trades = max(
            DELETION_MIN_TRADES_FLOOR,
            DELETION_HETEROZYGOUS_MIN_TRADES - pressure.trades_reduction,
        )
        effective_hom_trades = max(
            DELETION_MIN_HOM_TRADES_FLOOR,
            DELETION_HOMOZYGOUS_MIN_TRADES - pressure.trades_reduction,
        )

        # HYSTERESIS: Different logic based on current state
        if current_stage == DeletionStage.HOMOZYGOUS:
            # Already fully suppressed: only recover if loss rate drops well below threshold
            if posterior_lr < DELETION_RECOVERY_LR:
                return DeletionStage.NONE, 1.0
            # Stay homozygous (hysteresis prevents oscillation)
            return DeletionStage.HOMOZYGOUS, DELETION_HOMOZYGOUS_FACTOR

        elif current_stage == DeletionStage.HETEROZYGOUS:
            # Partially suppressed: can escalate or recover
            if posterior_lr < DELETION_RECOVERY_LR:
                return DeletionStage.NONE, 1.0
            elif total >= effective_hom_trades and posterior_lr >= effective_hom_lr:
                # Escalate to homozygous... unless het advantage detected
                if het_advantage:
                    log.info(
                        "[DEL] Heterozygote advantage detected -- maintaining partial "
                        "suppression (posterior_lr=%.3f, trades=%d)",
                        posterior_lr, total,
                    )
                    return DeletionStage.HETEROZYGOUS, DELETION_HETEROZYGOUS_FACTOR
                return DeletionStage.HOMOZYGOUS, DELETION_HOMOZYGOUS_FACTOR
            # Stay heterozygous
            return DeletionStage.HETEROZYGOUS, DELETION_HETEROZYGOUS_FACTOR

        else:  # DeletionStage.NONE
            # Not yet flagged: evaluate for initial flagging
            if total >= effective_hom_trades and posterior_lr >= effective_hom_lr:
                # Strong evidence: skip to homozygous
                return DeletionStage.HOMOZYGOUS, DELETION_HOMOZYGOUS_FACTOR
            elif total >= effective_het_trades and posterior_lr >= effective_het_lr:
                # Partial evidence: heterozygous
                return DeletionStage.HETEROZYGOUS, DELETION_HETEROZYGOUS_FACTOR
            # Not enough evidence yet
            return DeletionStage.NONE, 1.0


# ============================================================
# SUPPRESSION ENGINE
# ============================================================

class SuppressionEngine:
    """
    Applies suppression factors to TE signal combinations.

    This is the execution layer -- it reads suppression factors from the database
    and applies them to signal strength. It also handles quantum integration
    by reducing qubit rotation angles for suppressed patterns.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_suppression(self, active_tes: List[str]) -> float:
        """
        Get the suppression factor for a TE combination.

        Returns:
            1.0 if no suppression (normal signal strength).
            0.5 for heterozygous deletion (partial suppression).
            0.1 for homozygous deletion (near-complete suppression).
        """
        if not active_tes:
            return 1.0

        combo = "+".join(sorted(active_tes))
        pattern_hash = hashlib.md5(combo.encode()).hexdigest()[:16]

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT suppression_factor, deletion_stage, last_seen "
                    "FROM protective_deletions WHERE pattern_hash=?",
                    (pattern_hash,)
                )
                row = cursor.fetchone()

            if row is None:
                return 1.0  # Unknown pattern, no suppression

            suppression_factor = float(row[0])
            deletion_stage = row[1]
            last_seen = row[2]

            # No suppression if stage is "none"
            if deletion_stage == DeletionStage.NONE.value:
                return 1.0

            # Check expiry: don't suppress patterns that haven't been seen recently
            if last_seen:
                try:
                    last_dt = datetime.fromisoformat(last_seen)
                    age_days = (datetime.now() - last_dt).days
                    if age_days > DELETION_EXPIRY_DAYS:
                        log.debug(
                            "Suppressed pattern %s expired (%d days old, limit=%d) -- releasing",
                            combo, age_days, DELETION_EXPIRY_DAYS,
                        )
                        return 1.0
                except (ValueError, TypeError):
                    pass

            return suppression_factor

        except Exception as e:
            log.warning("Suppression lookup failed for %s: %s", combo, e)
            return 1.0

    def get_quantum_angle_modifier(self, active_tes: List[str]) -> float:
        """
        Get the quantum rotation angle modifier for suppressed patterns.

        Suppressed patterns get their qubit rotation angles scaled down,
        which reduces their influence on the quantum circuit output.

        Returns:
            1.0 for normal patterns.
            suppression_factor * QUANTUM_SUPPRESSION_ANGLE_SCALE for suppressed.
        """
        suppression = self.get_suppression(active_tes)
        if suppression >= 1.0:
            return 1.0
        # Scale the suppression for quantum circuit application
        return max(0.01, suppression * QUANTUM_SUPPRESSION_ANGLE_SCALE)

    def get_combined_modifier(
        self,
        active_tes: List[str],
        domestication_boost: float = 1.0,
    ) -> float:
        """
        Compute the combined signal modifier from both domestication and deletion.

        combined = boost * suppression

        If domesticated:  1.25 * 1.0 = 1.25 (boosted)
        If neutral:       1.0  * 1.0 = 1.00 (unchanged)
        If het-deleted:   1.0  * 0.5 = 0.50 (partially suppressed)
        If hom-deleted:   1.0  * 0.1 = 0.10 (nearly muted)
        If CONFLICTED:    1.25 * 0.5 = 0.625 (suppression wins, safety first)

        Args:
            active_tes: List of active TE family names.
            domestication_boost: Boost factor from TEDomesticationTracker.

        Returns:
            Combined modifier to apply to signal confidence.
        """
        suppression = self.get_suppression(active_tes)
        return domestication_boost * suppression


# ============================================================
# PROTECTIVE DELETION TRACKER (main class)
# ============================================================

class ProtectiveDeletionTracker:
    """
    Main tracker for protective deletion -- the inverse of TE domestication.

    Where TEDomesticationTracker identifies winning patterns and boosts them,
    ProtectiveDeletionTracker identifies LOSING patterns and suppresses them.

    This class orchestrates:
      - ToxicPatternDetector: classifies patterns into deletion stages
      - SuppressionEngine: applies suppression factors to signals
      - DrawdownPressure: adapts thresholds under account stress
      - AlleleFrequencyMonitor: tracks population-level suppression health
      - Heterozygote advantage tracking: detects when partial > full suppression

    Integration with domestication:
      - Shares the same pattern_hash scheme (MD5 of sorted TE combo)
      - Applied at the same point in the TEQA pipeline (alongside get_boost)
      - Combined modifier = boost * suppression
    """

    def __init__(
        self,
        db_path: str = None,
        domestication_tracker: TEDomesticationTracker = None,
    ):
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "teqa_protective_deletion.db")
        else:
            self.db_path = db_path

        self.domestication = domestication_tracker

        # Sub-components
        self.drawdown = DrawdownPressure()
        self.detector = ToxicPatternDetector(self.drawdown)
        self.suppression_engine = SuppressionEngine(self.db_path)
        self.allele_monitor = AlleleFrequencyMonitor(self.db_path)

        # Initialize database
        self._init_db()

        # In-memory caches
        self._het_advantage_cache: Dict[str, HetAdvantageRecord] = {}
        self._last_allele_report: Optional[AlleleFrequencyReport] = None
        self._allele_report_interval = 300  # Recompute every 5 minutes
        self._last_allele_report_time: float = 0.0

    def _init_db(self):
        """Initialize the protective deletion database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS protective_deletions (
                        pattern_hash TEXT PRIMARY KEY,
                        te_combo TEXT NOT NULL,
                        win_count INTEGER DEFAULT 0,
                        loss_count INTEGER DEFAULT 0,
                        loss_rate REAL DEFAULT 0.0,
                        posterior_lr REAL DEFAULT 0.5,
                        deletion_stage TEXT DEFAULT 'none',
                        suppression_factor REAL DEFAULT 1.0,
                        flagged_at TEXT,
                        recovered_at TEXT,
                        first_seen TEXT,
                        last_seen TEXT,
                        total_pnl REAL DEFAULT 0.0,
                        total_loss_pnl REAL DEFAULT 0.0,
                        total_win_pnl REAL DEFAULT 0.0,
                        avg_loss_magnitude REAL DEFAULT 0.0,
                        drawdown_at_flag REAL DEFAULT 0.0,
                        het_advantage_score REAL DEFAULT 0.0,
                        het_period_pnl REAL DEFAULT 0.0,
                        het_period_trades INTEGER DEFAULT 0,
                        hom_period_pnl REAL DEFAULT 0.0,
                        hom_period_trades INTEGER DEFAULT 0,
                        transitions INTEGER DEFAULT 0
                    )
                """)

                # History table for tracking stage transitions (audit trail)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS deletion_transitions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_hash TEXT NOT NULL,
                        te_combo TEXT NOT NULL,
                        old_stage TEXT NOT NULL,
                        new_stage TEXT NOT NULL,
                        posterior_lr REAL,
                        total_trades INTEGER,
                        drawdown_pct REAL DEFAULT 0.0,
                        pressure_level TEXT DEFAULT 'NONE',
                        timestamp TEXT NOT NULL
                    )
                """)

                # Allele frequency snapshots (time series of population health)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS allele_frequency_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_patterns INTEGER,
                        het_patterns INTEGER,
                        hom_patterns INTEGER,
                        allele_frequency REAL,
                        health TEXT,
                        avg_suppression REAL
                    )
                """)

                conn.commit()
        except Exception as e:
            log.warning("Protective deletion DB init failed: %s", e)

    # ----------------------------------------------------------
    # CORE: Record trade outcome and update deletion classification
    # ----------------------------------------------------------

    def record_outcome(
        self,
        active_tes: List[str],
        won: bool,
        profit: float = 0.0,
        account_drawdown_pct: float = 0.0,
    ):
        """
        Record the outcome of a trade where these TEs were active.

        This is the primary feedback function. It:
          1. Updates win/loss counts and Bayesian posterior loss rate
          2. Runs the ToxicPatternDetector to classify the deletion stage
          3. Logs any stage transitions for audit
          4. Tracks heterozygote advantage data

        Called from the same feedback loop as domestication's record_pattern().

        Args:
            active_tes: List of TE family names that were active at signal time.
            won: Whether the trade was profitable.
            profit: Trade P/L in dollars.
            account_drawdown_pct: Current account drawdown as fraction (0.05 = 5%).
        """
        if not active_tes:
            return

        combo = "+".join(sorted(active_tes))
        pattern_hash = hashlib.md5(combo.encode()).hexdigest()[:16]
        now = datetime.now().isoformat()

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT win_count, loss_count, deletion_stage, suppression_factor, "
                    "total_pnl, total_loss_pnl, total_win_pnl, avg_loss_magnitude, "
                    "het_period_pnl, het_period_trades, hom_period_pnl, hom_period_trades, "
                    "het_advantage_score, transitions "
                    "FROM protective_deletions WHERE pattern_hash=?",
                    (pattern_hash,)
                )
                row = cursor.fetchone()

                if row:
                    # Update existing record
                    win_count = row[0] + (1 if won else 0)
                    loss_count = row[1] + (0 if won else 1)
                    total = win_count + loss_count
                    loss_rate = loss_count / total if total > 0 else 0.0

                    current_stage = DeletionStage(row[2])
                    old_suppression = float(row[3])
                    total_pnl = (row[4] or 0.0) + profit
                    total_loss_pnl = (row[5] or 0.0) + (abs(profit) if not won else 0.0)
                    total_win_pnl = (row[6] or 0.0) + (profit if won else 0.0)
                    het_advantage_score = float(row[12] or 0.0)
                    transitions = int(row[13] or 0)

                    # Running average of loss magnitude
                    if not won:
                        old_avg_loss = float(row[7] or 0.0)
                        old_loss_count = row[1]  # Before increment
                        if old_loss_count > 0:
                            avg_loss_mag = (old_avg_loss * old_loss_count + abs(profit)) / (old_loss_count + 1)
                        else:
                            avg_loss_mag = abs(profit)
                    else:
                        avg_loss_mag = float(row[7] or 0.0)

                    # Track P/L by deletion stage (for heterozygote advantage detection)
                    het_period_pnl = float(row[8] or 0.0)
                    het_period_trades = int(row[9] or 0)
                    hom_period_pnl = float(row[10] or 0.0)
                    hom_period_trades = int(row[11] or 0)

                    if current_stage == DeletionStage.HETEROZYGOUS:
                        het_period_pnl += profit
                        het_period_trades += 1
                    elif current_stage == DeletionStage.HOMOZYGOUS:
                        hom_period_pnl += profit
                        hom_period_trades += 1

                    # Bayesian posterior loss rate
                    # Beta(alpha + losses, beta + wins) -> posterior mean
                    posterior_lr = (DELETION_PRIOR_ALPHA + loss_count) / (
                        DELETION_PRIOR_ALPHA + DELETION_PRIOR_BETA + total
                    )

                    # Check heterozygote advantage
                    het_advantage = self._check_het_advantage(
                        pattern_hash, het_period_pnl, het_period_trades,
                        hom_period_pnl, hom_period_trades,
                    )

                    # Classify deletion stage
                    new_stage, new_suppression = self.detector.classify_pattern(
                        win_count=win_count,
                        loss_count=loss_count,
                        posterior_lr=posterior_lr,
                        current_stage=current_stage,
                        account_drawdown_pct=account_drawdown_pct,
                        het_advantage=het_advantage,
                    )

                    # Log stage transition if changed
                    if new_stage != current_stage:
                        transitions += 1
                        pressure = self.drawdown.compute_pressure(account_drawdown_pct)
                        cursor.execute("""
                            INSERT INTO deletion_transitions
                            (pattern_hash, te_combo, old_stage, new_stage,
                             posterior_lr, total_trades, drawdown_pct, pressure_level, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pattern_hash, combo,
                            current_stage.value, new_stage.value,
                            posterior_lr, total,
                            account_drawdown_pct,
                            pressure.pressure_level.value,
                            now,
                        ))

                        if new_stage != DeletionStage.NONE:
                            log.info(
                                "[DEL] Pattern %s (%s) -> %s | posterior_lr=%.3f "
                                "trades=%d suppression=%.2f pressure=%s",
                                pattern_hash[:8], combo, new_stage.value,
                                posterior_lr, total, new_suppression,
                                pressure.pressure_level.value,
                            )
                        else:
                            log.info(
                                "[DEL] Pattern %s (%s) RECOVERED | posterior_lr=%.3f trades=%d",
                                pattern_hash[:8], combo, posterior_lr, total,
                            )

                    # Update heterozygote advantage score
                    if het_advantage:
                        het_advantage_score += 1.0

                    # Update the record
                    flagged_at_clause = ""
                    recovered_at_clause = ""

                    if new_stage != DeletionStage.NONE and current_stage == DeletionStage.NONE:
                        flagged_at_clause = f", flagged_at='{now}'"
                    if new_stage == DeletionStage.NONE and current_stage != DeletionStage.NONE:
                        recovered_at_clause = f", recovered_at='{now}'"

                    cursor.execute(f"""
                        UPDATE protective_deletions
                        SET win_count=?, loss_count=?, loss_rate=?, posterior_lr=?,
                            deletion_stage=?, suppression_factor=?,
                            total_pnl=?, total_loss_pnl=?, total_win_pnl=?,
                            avg_loss_magnitude=?,
                            het_period_pnl=?, het_period_trades=?,
                            hom_period_pnl=?, hom_period_trades=?,
                            het_advantage_score=?, transitions=?,
                            last_seen=?
                            {flagged_at_clause}{recovered_at_clause}
                        WHERE pattern_hash=?
                    """, (
                        win_count, loss_count, loss_rate, posterior_lr,
                        new_stage.value, new_suppression,
                        total_pnl, total_loss_pnl, total_win_pnl,
                        avg_loss_mag,
                        het_period_pnl, het_period_trades,
                        hom_period_pnl, hom_period_trades,
                        het_advantage_score, transitions,
                        now,
                        pattern_hash,
                    ))

                else:
                    # First time seeing this TE combination
                    initial_lr = (DELETION_PRIOR_ALPHA + (0 if won else 1)) / (
                        DELETION_PRIOR_ALPHA + DELETION_PRIOR_BETA + 1
                    )
                    cursor.execute("""
                        INSERT INTO protective_deletions
                        (pattern_hash, te_combo, win_count, loss_count, loss_rate,
                         posterior_lr, deletion_stage, suppression_factor,
                         first_seen, last_seen, total_pnl, total_loss_pnl, total_win_pnl,
                         avg_loss_magnitude)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern_hash, combo,
                        1 if won else 0,
                        0 if won else 1,
                        0.0 if won else 1.0,
                        initial_lr,
                        DeletionStage.NONE.value,
                        1.0,
                        now, now,
                        profit,
                        abs(profit) if not won else 0.0,
                        profit if won else 0.0,
                        abs(profit) if not won else 0.0,
                    ))

                conn.commit()

        except Exception as e:
            log.warning("Protective deletion record failed for %s: %s", combo, e)

    # ----------------------------------------------------------
    # SIGNAL APPLICATION: Get suppression and combined modifier
    # ----------------------------------------------------------

    def get_suppression(self, active_tes: List[str]) -> float:
        """
        Get suppression factor for a TE combination.

        Returns 1.0 (no effect), 0.5 (het deletion), or 0.1 (hom deletion).
        """
        return self.suppression_engine.get_suppression(active_tes)

    def get_combined_modifier(self, active_tes: List[str]) -> float:
        """
        Get the combined domestication boost * protective deletion suppression.

        This is the single number that modifies signal confidence.
        Uses the domestication tracker if one was provided at construction.

        Args:
            active_tes: List of active TE family names.

        Returns:
            Combined modifier (e.g., 1.25, 1.0, 0.5, 0.1, or 0.625 for conflicts).
        """
        # Get domestication boost (>= 1.0 for winners)
        boost = 1.0
        if self.domestication is not None:
            boost = self.domestication.get_boost(active_tes)

        # Get deletion suppression (<= 1.0 for losers)
        suppression = self.get_suppression(active_tes)

        return boost * suppression

    def get_quantum_angle_modifier(self, active_tes: List[str]) -> float:
        """
        Get quantum circuit rotation angle modifier for suppressed patterns.

        Suppressed patterns get reduced qubit rotation angles, decreasing
        their influence on the quantum circuit output.
        """
        return self.suppression_engine.get_quantum_angle_modifier(active_tes)

    # ----------------------------------------------------------
    # DRAWDOWN INTEGRATION
    # ----------------------------------------------------------

    def update_equity(self, equity: float) -> float:
        """
        Update the drawdown pressure engine with current account equity.

        Args:
            equity: Current account equity from MT5.

        Returns:
            Current drawdown percentage.
        """
        return self.drawdown.update_equity(equity)

    # ----------------------------------------------------------
    # ALLELE FREQUENCY MONITORING
    # ----------------------------------------------------------

    def get_allele_frequency_report(self, force: bool = False) -> AlleleFrequencyReport:
        """
        Get the current allele frequency report.

        Cached and recomputed every 5 minutes (or on force=True).

        Returns:
            AlleleFrequencyReport with population health metrics.
        """
        now = time.time()
        if (not force
                and self._last_allele_report is not None
                and now - self._last_allele_report_time < self._allele_report_interval):
            return self._last_allele_report

        report = self.allele_monitor.compute_frequency()
        self._last_allele_report = report
        self._last_allele_report_time = now

        # Log the report to the database for time-series tracking
        self._log_allele_frequency(report)

        # Log warnings
        if report.health == "CRITICAL":
            log.warning(
                "[DEL] CRITICAL allele frequency: %.1f%% of patterns suppressed "
                "(%d het + %d hom / %d total). Consider regime change evaluation.",
                report.allele_frequency * 100,
                report.het_patterns, report.hom_patterns, report.total_patterns,
            )
        elif report.health == "WARNING":
            log.info(
                "[DEL] Elevated allele frequency: %.1f%% of patterns suppressed "
                "(%d het + %d hom / %d total)",
                report.allele_frequency * 100,
                report.het_patterns, report.hom_patterns, report.total_patterns,
            )

        return report

    def _log_allele_frequency(self, report: AlleleFrequencyReport):
        """Log allele frequency snapshot to database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO allele_frequency_log
                    (timestamp, total_patterns, het_patterns, hom_patterns,
                     allele_frequency, health, avg_suppression)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    report.timestamp,
                    report.total_patterns,
                    report.het_patterns,
                    report.hom_patterns,
                    report.allele_frequency,
                    report.health,
                    report.avg_suppression,
                ))
                conn.commit()
        except Exception as e:
            log.debug("Allele frequency log failed: %s", e)

    # ----------------------------------------------------------
    # HETEROZYGOTE ADVANTAGE
    # ----------------------------------------------------------

    def _check_het_advantage(
        self,
        pattern_hash: str,
        het_pnl: float,
        het_trades: int,
        hom_pnl: float,
        hom_trades: int,
    ) -> bool:
        """
        Check if partial suppression outperforms full deletion for this pattern.

        This is analogous to sickle-cell heterozygote advantage: carriers
        of one copy of the sickle-cell allele have an advantage over both
        homozygous normal (malaria susceptible) and homozygous sickle-cell
        (sickle-cell disease).

        A fully deleted signal provides ZERO information. A partially
        suppressed signal still contributes SOME information, which can
        be useful in transitional market regimes.

        Returns True if heterozygote advantage is detected.
        """
        # Need sufficient data in both periods
        if het_trades < HET_ADVANTAGE_EVAL_TRADES:
            return False
        if hom_trades < 10:
            # Not enough homozygous data to compare -- default to no advantage
            return False

        het_avg_pnl = het_pnl / het_trades
        hom_avg_pnl = hom_pnl / hom_trades

        # Het advantage: partial suppression produces better average P/L
        advantage = het_avg_pnl > hom_avg_pnl

        if advantage:
            # Cache the finding
            self._het_advantage_cache[pattern_hash] = HetAdvantageRecord(
                pattern_hash=pattern_hash,
                het_period_pnl=het_pnl,
                het_period_trades=het_trades,
                hom_period_pnl=hom_pnl,
                hom_period_trades=hom_trades,
                advantage_detected=True,
                last_evaluated=datetime.now().isoformat(),
            )

        return advantage

    # ----------------------------------------------------------
    # STATISTICS AND REPORTING
    # ----------------------------------------------------------

    def get_deletion_stats(self) -> Dict:
        """
        Get comprehensive statistics about the protective deletion system.

        Returns dict with counts, frequencies, pressure state, and top toxic patterns.
        """
        stats = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "total_patterns": 0,
            "none_count": 0,
            "het_count": 0,
            "hom_count": 0,
            "allele_frequency": 0.0,
            "health": "HEALTHY",
            "drawdown": self.drawdown.get_pressure_summary(),
            "top_toxic_patterns": [],
            "recent_transitions": [],
            "het_advantages_detected": len([
                h for h in self._het_advantage_cache.values()
                if h.advantage_detected
            ]),
        }

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()

                # Counts by stage
                cursor.execute("SELECT COUNT(*) FROM protective_deletions")
                stats["total_patterns"] = cursor.fetchone()[0]

                for stage in DeletionStage:
                    cursor.execute(
                        "SELECT COUNT(*) FROM protective_deletions WHERE deletion_stage=?",
                        (stage.value,)
                    )
                    count = cursor.fetchone()[0]
                    stats[f"{stage.value}_count"] = count

                # Allele frequency
                if stats["total_patterns"] > 0:
                    suppressed = stats.get("heterozygous_count", 0) + stats.get("homozygous_count", 0)
                    stats["allele_frequency"] = suppressed / stats["total_patterns"]

                # Health classification
                if stats["allele_frequency"] > ALLELE_FREQ_CRITICAL_THRESHOLD:
                    stats["health"] = "CRITICAL"
                elif stats["allele_frequency"] > ALLELE_FREQ_WARNING_THRESHOLD:
                    stats["health"] = "WARNING"

                # Top 10 most toxic patterns
                cursor.execute("""
                    SELECT pattern_hash, te_combo, loss_count, win_count,
                           posterior_lr, deletion_stage, suppression_factor, total_pnl
                    FROM protective_deletions
                    WHERE deletion_stage != 'none'
                    ORDER BY posterior_lr DESC
                    LIMIT 10
                """)
                for r in cursor.fetchall():
                    stats["top_toxic_patterns"].append({
                        "hash": r[0][:8],
                        "combo": r[1],
                        "losses": r[2],
                        "wins": r[3],
                        "posterior_lr": round(float(r[4]), 4),
                        "stage": r[5],
                        "suppression": round(float(r[6]), 2),
                        "total_pnl": round(float(r[7] or 0), 2),
                    })

                # Recent transitions (last 20)
                cursor.execute("""
                    SELECT pattern_hash, te_combo, old_stage, new_stage,
                           posterior_lr, total_trades, pressure_level, timestamp
                    FROM deletion_transitions
                    ORDER BY id DESC
                    LIMIT 20
                """)
                for r in cursor.fetchall():
                    stats["recent_transitions"].append({
                        "hash": r[0][:8],
                        "combo": r[1],
                        "from": r[2],
                        "to": r[3],
                        "posterior_lr": round(float(r[4] or 0), 4),
                        "trades": r[5],
                        "pressure": r[6],
                        "time": r[7],
                    })

        except Exception as e:
            log.warning("Failed to get deletion stats: %s", e)

        return stats

    def get_pattern_detail(self, active_tes: List[str]) -> Optional[Dict]:
        """
        Get detailed information about a specific TE pattern's deletion status.

        Args:
            active_tes: List of TE family names.

        Returns:
            Dict with all pattern fields, or None if pattern not found.
        """
        if not active_tes:
            return None

        combo = "+".join(sorted(active_tes))
        pattern_hash = hashlib.md5(combo.encode()).hexdigest()[:16]

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM protective_deletions WHERE pattern_hash=?",
                    (pattern_hash,)
                )
                row = cursor.fetchone()
                if row is None:
                    return None

                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))

        except Exception as e:
            log.warning("Pattern detail lookup failed: %s", e)
            return None

    def reset_pattern(self, active_tes: List[str]):
        """
        Reset a specific pattern's deletion status back to neutral.

        Use with caution -- this removes learned suppression.

        Args:
            active_tes: List of TE family names.
        """
        if not active_tes:
            return

        combo = "+".join(sorted(active_tes))
        pattern_hash = hashlib.md5(combo.encode()).hexdigest()[:16]
        now = datetime.now().isoformat()

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    UPDATE protective_deletions
                    SET deletion_stage=?, suppression_factor=?, recovered_at=?
                    WHERE pattern_hash=?
                """, (DeletionStage.NONE.value, 1.0, now, pattern_hash))
                conn.commit()
                log.info("[DEL] Pattern %s (%s) manually reset to neutral", pattern_hash[:8], combo)
        except Exception as e:
            log.warning("Pattern reset failed: %s", e)


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
    print("  PROTECTIVE DELETION ENGINE -- Loss-of-Function Defense")
    print("  CCR5-delta32 Receptor Deletion -> Immune Resistance")
    print("=" * 76)

    # Use temp DB for test
    test_db = str(Path(__file__).parent / "test_protective_deletion.db")

    # Clean up any previous test DB
    try:
        os.remove(test_db)
    except OSError:
        pass

    # Initialize tracker (no domestication tracker for standalone test)
    tracker = ProtectiveDeletionTracker(db_path=test_db)

    print("\n  --- PHASE 1: TOXIC PATTERN ACCUMULATION ---")
    print("  Simulating a toxic TE combo that consistently precedes losses...")
    print("  (Bayesian Beta(10,10) prior requires real evidence to shift from 50%)")

    toxic_tes = ["L1_Neuronal", "Alu_Exonization", "HERV_Synapse"]
    neutral_tes = ["Ty1_copia", "CACTA", "Mariner_Tc1"]
    mixed_tes = ["Ty3_gypsy", "BEL_Pao", "DIRS1"]
    drawdown_tes = ["SINE", "Helitron", "Crypton"]  # Tests drawdown pressure

    # Simulate 50 trades for the toxic pattern (80% loss rate)
    # With Beta(10,10) prior, need ~50 trades at 80% LR to push posterior past 0.70
    # posterior_lr = (10 + 40) / (10 + 10 + 50) = 50/70 = 0.714 (crosses hom threshold)
    for i in range(50):
        won = i % 5 == 0  # 20% win rate = 80% loss rate
        profit = 0.80 if won else -1.00
        tracker.record_outcome(toxic_tes, won, profit, account_drawdown_pct=0.02)

    # Simulate 40 trades for a neutral pattern (50% win rate)
    for i in range(40):
        won = i % 2 == 0  # 50% win rate
        profit = 1.50 if won else -1.00
        tracker.record_outcome(neutral_tes, won, profit, account_drawdown_pct=0.02)

    # Simulate 25 trades for a borderline pattern (~32% win rate = 68% loss rate)
    # Use modular arithmetic that works within the range: win if i in {0,8,16,24,...}
    # i % 3 == 0 gives ~33% win rate for 25 trades = 9 wins, 16 losses
    # posterior_lr = (10 + 16) / (10 + 10 + 25) = 26/45 = 0.578 -- below het threshold
    for i in range(25):
        won = (i % 3 == 0)  # ~33% win rate = ~67% loss rate
        profit = 0.90 if won else -1.00
        tracker.record_outcome(mixed_tes, won, profit, account_drawdown_pct=0.02)

    # Simulate 40 trades under drawdown pressure (~35% win rate = 65% loss rate)
    # Under 6% drawdown (MODERATE pressure):
    #   effective_het_lr = 0.65 - 0.05 = 0.60
    #   effective_het_trades = max(8, 15-4) = 11
    # i % 20 < 7 gives 35% win rate for 40 trades = 14 wins, 26 losses
    # posterior_lr = (10 + 26) / (10 + 10 + 40) = 36/60 = 0.600 -- hits effective het threshold
    for i in range(40):
        won = (i % 20 < 7)  # 35% win rate = 65% loss rate
        profit = 0.70 if won else -1.00
        tracker.record_outcome(drawdown_tes, won, profit, account_drawdown_pct=0.06)

    print("\n  --- PHASE 2: DELETION CLASSIFICATION ---")
    toxic_detail = tracker.get_pattern_detail(toxic_tes)
    neutral_detail = tracker.get_pattern_detail(neutral_tes)
    mixed_detail = tracker.get_pattern_detail(mixed_tes)
    drawdown_detail = tracker.get_pattern_detail(drawdown_tes)

    for label, tes, detail in [
        ("Toxic (80% LR, 50 trades)", toxic_tes, toxic_detail),
        ("Neutral (50% LR, 40 trades)", neutral_tes, neutral_detail),
        ("Mixed (67% LR, 25 trades)", mixed_tes, mixed_detail),
        ("Drawdown-pressured (65% LR, 40 trades, 6% DD)", drawdown_tes, drawdown_detail),
    ]:
        print(f"\n  {label} ({'+'.join(tes)}):")
        if detail:
            print(f"    Stage: {detail['deletion_stage']}")
            print(f"    Posterior LR: {detail['posterior_lr']:.4f}")
            print(f"    Suppression: {detail['suppression_factor']:.2f}")
            print(f"    Wins: {detail['win_count']} | Losses: {detail['loss_count']}")
            print(f"    Total PnL: ${detail.get('total_pnl', 0):.2f}")

    print("\n  --- PHASE 3: SUPPRESSION APPLICATION ---")
    s_toxic = tracker.get_suppression(toxic_tes)
    s_neutral = tracker.get_suppression(neutral_tes)
    s_mixed = tracker.get_suppression(mixed_tes)
    s_drawdown = tracker.get_suppression(drawdown_tes)
    s_unknown = tracker.get_suppression(["Penelope", "RTE"])

    print(f"  Toxic suppression:    {s_toxic:.2f}x")
    print(f"  Neutral suppression:  {s_neutral:.2f}x")
    print(f"  Mixed suppression:    {s_mixed:.2f}x")
    print(f"  Drawdown suppression: {s_drawdown:.2f}x")
    print(f"  Unknown suppression:  {s_unknown:.2f}x (no data)")

    print("\n  --- PHASE 4: COMBINED MODIFIER (with domestication) ---")
    # Simulate combined modifiers
    print(f"  Toxic + neutral boost:    {1.0 * s_toxic:.3f} (boost=1.0, suppression={s_toxic:.2f})")
    print(f"  Neutral + neutral boost:  {1.0 * s_neutral:.3f} (boost=1.0, suppression={s_neutral:.2f})")
    print(f"  Toxic + domesticated:     {1.25 * s_toxic:.3f} (boost=1.25, suppression={s_toxic:.2f})")
    print(f"  Neutral + domesticated:   {1.25 * s_neutral:.3f} (boost=1.25, suppression={s_neutral:.2f})")

    print("\n  --- PHASE 5: DRAWDOWN PRESSURE ---")
    # Test drawdown pressure at various levels
    for dd_pct in [0.01, 0.03, 0.05, 0.08, 0.12]:
        pressure = tracker.drawdown.compute_pressure(dd_pct)
        eff_het_lr = DELETION_HETEROZYGOUS_MIN_LR - pressure.lr_reduction
        eff_hom_lr = DELETION_HOMOZYGOUS_MIN_LR - pressure.lr_reduction
        eff_het_trades = max(
            DELETION_MIN_TRADES_FLOOR,
            DELETION_HETEROZYGOUS_MIN_TRADES - pressure.trades_reduction,
        )
        eff_hom_trades = max(
            DELETION_MIN_HOM_TRADES_FLOOR,
            DELETION_HOMOZYGOUS_MIN_TRADES - pressure.trades_reduction,
        )
        print(
            f"  DD={dd_pct*100:.0f}%: pressure={pressure.pressure_level.value:8s} | "
            f"het_lr={eff_het_lr:.2f} hom_lr={eff_hom_lr:.2f} | "
            f"het_trades={eff_het_trades} hom_trades={eff_hom_trades}"
        )

    print("\n  --- PHASE 6: ALLELE FREQUENCY ---")
    report = tracker.get_allele_frequency_report(force=True)
    print(f"  Total patterns:  {report.total_patterns}")
    print(f"  Heterozygous:    {report.het_patterns}")
    print(f"  Homozygous:      {report.hom_patterns}")
    print(f"  Allele frequency: {report.allele_frequency:.1%}")
    print(f"  Health:          {report.health}")
    if report.worst_pattern_combo:
        print(f"  Worst pattern:   {report.worst_pattern_combo} (LR={report.worst_posterior_lr:.4f})")

    print("\n  --- PHASE 7: QUANTUM ANGLE MODIFIER ---")
    qa_toxic = tracker.get_quantum_angle_modifier(toxic_tes)
    qa_neutral = tracker.get_quantum_angle_modifier(neutral_tes)
    print(f"  Toxic qubit angle modifier:   {qa_toxic:.4f} (rotation severely dampened)")
    print(f"  Neutral qubit angle modifier: {qa_neutral:.4f} (rotation unchanged)")

    print("\n  --- PHASE 8: HYSTERESIS TEST ---")
    print("  Adding wins to the toxic pattern to test recovery threshold...")
    print("  Hysteresis: flagged at LR>0.70, but only recovers when LR drops below 0.50")

    # Add 20 wins first -- should NOT recover yet (hysteresis holds)
    print("\n  Step A: Adding 20 wins...")
    for i in range(20):
        tracker.record_outcome(toxic_tes, won=True, profit=1.50, account_drawdown_pct=0.01)

    toxic_detail_a = tracker.get_pattern_detail(toxic_tes)
    if toxic_detail_a:
        recovered_a = toxic_detail_a['deletion_stage'] == DeletionStage.NONE.value
        print(f"    Stage: {toxic_detail_a['deletion_stage']}  |  "
              f"Posterior LR: {toxic_detail_a['posterior_lr']:.4f}  |  "
              f"Win/Loss: {toxic_detail_a['win_count']}/{toxic_detail_a['loss_count']}  |  "
              f"Recovered: {'YES' if recovered_a else 'NO'}")

    # Add 30 more wins -- should push posterior LR below 0.50 and recover
    print("\n  Step B: Adding 30 more wins (total 50 added)...")
    for i in range(30):
        tracker.record_outcome(toxic_tes, won=True, profit=1.50, account_drawdown_pct=0.01)

    toxic_detail_b = tracker.get_pattern_detail(toxic_tes)
    if toxic_detail_b:
        recovered_b = toxic_detail_b['deletion_stage'] == DeletionStage.NONE.value
        print(f"    Stage: {toxic_detail_b['deletion_stage']}  |  "
              f"Posterior LR: {toxic_detail_b['posterior_lr']:.4f}  |  "
              f"Win/Loss: {toxic_detail_b['win_count']}/{toxic_detail_b['loss_count']}  |  "
              f"Recovered: {'YES -- receptor restored' if recovered_b else 'NO'}")

    print("\n  --- COMPREHENSIVE STATS ---")
    stats = tracker.get_deletion_stats()
    print(f"  Version: {stats['version']}")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Health: {stats['health']}")
    print(f"  Drawdown: {stats['drawdown']}")

    if stats["top_toxic_patterns"]:
        print("\n  Top toxic patterns:")
        for p in stats["top_toxic_patterns"]:
            print(f"    [{p['hash']}] {p['combo']} | "
                  f"LR={p['posterior_lr']:.3f} stage={p['stage']} "
                  f"supp={p['suppression']}x PnL=${p['total_pnl']:.2f}")

    if stats["recent_transitions"]:
        print("\n  Recent transitions:")
        for t in stats["recent_transitions"][:5]:
            print(f"    [{t['hash']}] {t['from']} -> {t['to']} | "
                  f"LR={t['posterior_lr']:.3f} trades={t['trades']} "
                  f"pressure={t['pressure']}")

    # Cleanup test DB
    try:
        os.remove(test_db)
    except OSError:
        pass

    print("\n" + "=" * 76)
    print("  Protective Deletion Engine test complete.")
    print("  The receptor has been deleted. The attack vector is gone.")
    print("=" * 76)
