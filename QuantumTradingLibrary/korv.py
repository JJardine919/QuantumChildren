"""
KoRV DOMESTICATION ENGINE -- Real-Time Retroviral Signal Onboarding
====================================================================
Algorithm #5 in the Quantum Children biological algorithm series.

Manages the staged integration of NEW signal types into the trading system,
inspired by the real-time domestication of Koala Retrovirus (KoRV) in
Australian koala populations.

Every new signal must earn its place through four stages:
  INFECTION       -> Probationary observation (no weight changes)
  IMMUNE_RESPONSE -> Methylation silencing (weight decay for toxic signals)
  TOLERANCE       -> Neutral coexistence (neither boosted nor suppressed)
  DOMESTICATED    -> Permanent integration (weight boost for proven signals)

Each {signal_type, instrument, timeframe} population is tracked independently,
mirroring how different koala populations domesticate KoRV at different rates.

Integration:
    - Reads CONFIDENCE_THRESHOLD from config_loader (no hardcoded trading values)
    - Stores all state in SQLite (korv_domestication.db)
    - Provides get_signal_weight() for BRAIN scripts and TEQA pipeline
    - Tracks pairwise signal interactions (recombination risk)

Authors: DooDoo + Claude
Date:    2026-02-08
Version: KORV-1.0
"""

import hashlib
import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import trading values from config_loader (NEVER hardcode)
try:
    from config_loader import CONFIDENCE_THRESHOLD
except ImportError:
    # Fallback for standalone testing only
    CONFIDENCE_THRESHOLD = 0.22

log = logging.getLogger(__name__)

VERSION = "KORV-1.0"

# ============================================================
# CONSTANTS
# ============================================================

# Onboarding stages
STAGE_INFECTION = "infection"
STAGE_IMMUNE_RESPONSE = "immune_response"
STAGE_TOLERANCE = "tolerance"
STAGE_DOMESTICATED = "domesticated"
STAGE_DORMANT = "dormant"

ALL_STAGES = [
    STAGE_INFECTION,
    STAGE_IMMUNE_RESPONSE,
    STAGE_TOLERANCE,
    STAGE_DOMESTICATED,
    STAGE_DORMANT,
]

# Probationary window (infection stage)
INFECTION_MIN_TRADES = 15
INFECTION_MIN_DAYS = 7
INFECTION_MAX_TRADES = 100

# Immune response (methylation / silencing)
METHYLATION_TOXICITY_THRESHOLD = 0.60
METHYLATION_INITIAL_WEIGHT = 0.70
METHYLATION_HEAVY_WEIGHT = 0.30
METHYLATION_SILENCED_WEIGHT = 0.05
METHYLATION_ESCALATION_LR = 0.70
METHYLATION_SILENCE_LR = 0.80
IMMUNE_RESPONSE_SPEED_FAST = 10
IMMUNE_RESPONSE_SPEED_SLOW = 30

# Tolerance (neutral zone)
TOLERANCE_WR_LOW = 0.42
TOLERANCE_WR_HIGH = 0.58
TOLERANCE_REEVAL_INTERVAL_DAYS = 14
TOLERANCE_MAX_IDLE_DAYS = 60

# Domestication (permanent integration)
DOMESTICATION_MIN_TRADES = 30
DOMESTICATION_MIN_POSTERIOR_WR = 0.62
DOMESTICATION_MIN_PROFIT_FACTOR = 1.40
DOMESTICATION_BOOST_BASE = 1.10
DOMESTICATION_BOOST_MAX = 1.35
DOMESTICATION_DE_TRIGGER_WR = 0.50
DOMESTICATION_DE_TRIGGER_TRADES = 20

# Bayesian prior (Beta distribution)
PRIOR_ALPHA = 8
PRIOR_BETA = 8

# Interaction tracking
INTERACTION_MIN_CO_OCCURRENCES = 10
INTERACTION_SYNERGY_THRESHOLD = 0.10
INTERACTION_TOXICITY_THRESHOLD = -0.10

# Endogenization report interval
ENDOGENIZATION_REPORT_INTERVAL_SEC = 3600


# ============================================================
# HELPER: Sigmoid
# ============================================================

def _sigmoid(x: float) -> float:
    """Standard sigmoid function, clamped to avoid overflow."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _running_average(current_avg: float, new_value: float, count: int) -> float:
    """Incremental running average."""
    if count <= 1:
        return new_value
    return current_avg + (new_value - current_avg) / count


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SignalOnboardRecord:
    """Tracks a single signal's onboarding status for one population."""
    record_id: str
    signal_type: str
    instrument: str
    timeframe: str
    population_key: str
    stage: str = STAGE_INFECTION
    win_count: int = 0
    loss_count: int = 0
    total_trades: int = 0
    posterior_wr: float = 0.5
    posterior_lr: float = 0.5
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    current_weight: float = 1.0
    methylation_level: float = 0.0
    domestication_boost: float = 1.0
    stage_entered_at: str = ""
    first_seen: str = ""
    last_seen: str = ""
    last_reeval: str = ""
    immune_response_speed: float = IMMUNE_RESPONSE_SPEED_SLOW
    de_domestication_count: int = 0
    notes: str = ""
    # Post-domestication tracking
    post_dom_wins: int = 0
    post_dom_losses: int = 0


@dataclass
class InteractionRecord:
    """Tracks pairwise signal interaction (recombination risk)."""
    interaction_id: str
    signal_a: str
    signal_b: str
    instrument: str
    timeframe: str
    co_occurrence_count: int = 0
    co_win_count: int = 0
    co_loss_count: int = 0
    individual_wr_a: float = 0.5
    individual_wr_b: float = 0.5
    combined_wr: float = 0.5
    synergy_score: float = 0.0
    classification: str = "neutral"
    first_seen: str = ""
    last_seen: str = ""


# ============================================================
# KoRV DOMESTICATION ENGINE
# ============================================================

class KoRVDomesticationEngine:
    """
    Core engine for real-time retroviral signal domestication.

    Manages the staged integration of new signal types into the trading
    system. Each signal is tracked per-population (instrument + timeframe),
    and progresses through INFECTION -> IMMUNE_RESPONSE / TOLERANCE /
    DOMESTICATED based on observed trade outcomes.

    This is the algorithmic equivalent of watching KoRV domestication
    unfold across koala populations in real time.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            self.db_path = str(
                Path(__file__).parent / "korv_domestication.db"
            )
        else:
            self.db_path = db_path

        self._init_db()
        self._last_report_time = 0.0

        log.info(
            "[KoRV] Engine initialized | db=%s | version=%s",
            self.db_path, VERSION,
        )

    # ----------------------------------------------------------
    # DATABASE INITIALIZATION
    # ----------------------------------------------------------

    def _init_db(self):
        """Create all tables for KoRV domestication tracking."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")

                # Main signal onboarding table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_onboard (
                        record_id TEXT PRIMARY KEY,
                        signal_type TEXT NOT NULL,
                        instrument TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        population_key TEXT NOT NULL UNIQUE,
                        stage TEXT NOT NULL DEFAULT 'infection',
                        win_count INTEGER DEFAULT 0,
                        loss_count INTEGER DEFAULT 0,
                        total_trades INTEGER DEFAULT 0,
                        posterior_wr REAL DEFAULT 0.5,
                        posterior_lr REAL DEFAULT 0.5,
                        profit_factor REAL DEFAULT 0.0,
                        total_pnl REAL DEFAULT 0.0,
                        avg_win REAL DEFAULT 0.0,
                        avg_loss REAL DEFAULT 0.0,
                        current_weight REAL DEFAULT 1.0,
                        methylation_level REAL DEFAULT 0.0,
                        domestication_boost REAL DEFAULT 1.0,
                        stage_entered_at TEXT,
                        first_seen TEXT,
                        last_seen TEXT,
                        last_reeval TEXT,
                        immune_response_speed REAL DEFAULT 30.0,
                        de_domestication_count INTEGER DEFAULT 0,
                        notes TEXT DEFAULT '',
                        post_dom_wins INTEGER DEFAULT 0,
                        post_dom_losses INTEGER DEFAULT 0
                    )
                """)

                # Pairwise interaction tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        interaction_id TEXT PRIMARY KEY,
                        signal_a TEXT NOT NULL,
                        signal_b TEXT NOT NULL,
                        instrument TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        co_occurrence_count INTEGER DEFAULT 0,
                        co_win_count INTEGER DEFAULT 0,
                        co_loss_count INTEGER DEFAULT 0,
                        individual_wr_a REAL DEFAULT 0.5,
                        individual_wr_b REAL DEFAULT 0.5,
                        combined_wr REAL DEFAULT 0.5,
                        synergy_score REAL DEFAULT 0.0,
                        classification TEXT DEFAULT 'neutral',
                        first_seen TEXT,
                        last_seen TEXT
                    )
                """)

                # Stage transition audit log
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS stage_transitions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        record_id TEXT NOT NULL,
                        from_stage TEXT NOT NULL,
                        to_stage TEXT NOT NULL,
                        reason TEXT DEFAULT '',
                        posterior_wr REAL DEFAULT 0.5,
                        posterior_lr REAL DEFAULT 0.5,
                        profit_factor REAL DEFAULT 0.0,
                        total_trades INTEGER DEFAULT 0,
                        current_weight REAL DEFAULT 1.0,
                        timestamp TEXT NOT NULL
                    )
                """)

                # Indices for performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_onboard_stage
                    ON signal_onboard(stage)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_onboard_instrument
                    ON signal_onboard(instrument)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_transitions_record
                    ON stage_transitions(record_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_interactions_class
                    ON interactions(classification)
                """)

                conn.commit()
        except Exception as e:
            log.warning("[KoRV] DB init failed: %s", e)

    # ----------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------

    @staticmethod
    def _make_record_id(signal_type: str, instrument: str, timeframe: str) -> str:
        """Generate deterministic record ID from population key."""
        population_key = f"{signal_type}|{instrument}|{timeframe}"
        return hashlib.md5(population_key.encode()).hexdigest()[:16]

    @staticmethod
    def _make_population_key(signal_type: str, instrument: str, timeframe: str) -> str:
        return f"{signal_type}|{instrument}|{timeframe}"

    @staticmethod
    def _make_interaction_id(
        signal_a: str, signal_b: str, instrument: str, timeframe: str
    ) -> str:
        pair = "|".join(sorted([signal_a, signal_b]))
        key = f"{pair}|{instrument}|{timeframe}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()

    def _log_stage_transition(
        self,
        record_id: str,
        from_stage: str,
        to_stage: str,
        reason: str,
        row: SignalOnboardRecord,
    ):
        """Write an audit entry for a stage transition."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO stage_transitions
                    (record_id, from_stage, to_stage, reason,
                     posterior_wr, posterior_lr, profit_factor,
                     total_trades, current_weight, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record_id, from_stage, to_stage, reason,
                    row.posterior_wr, row.posterior_lr, row.profit_factor,
                    row.total_trades, row.current_weight,
                    self._now_iso(),
                ))
                conn.commit()
        except Exception as e:
            log.warning("[KoRV] Failed to log transition: %s", e)

        log.info(
            "[KoRV] STAGE TRANSITION: %s -> %s | %s | WR=%.1f%% PF=%.2f trades=%d weight=%.2f | %s",
            from_stage, to_stage,
            row.population_key,
            row.posterior_wr * 100, row.profit_factor,
            row.total_trades, row.current_weight,
            reason,
        )

    def _load_record(self, record_id: str) -> Optional[SignalOnboardRecord]:
        """Load a signal onboard record from the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM signal_onboard WHERE record_id = ?",
                    (record_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return SignalOnboardRecord(
                    record_id=row["record_id"],
                    signal_type=row["signal_type"],
                    instrument=row["instrument"],
                    timeframe=row["timeframe"],
                    population_key=row["population_key"],
                    stage=row["stage"],
                    win_count=row["win_count"],
                    loss_count=row["loss_count"],
                    total_trades=row["total_trades"],
                    posterior_wr=row["posterior_wr"],
                    posterior_lr=row["posterior_lr"],
                    profit_factor=row["profit_factor"],
                    total_pnl=row["total_pnl"],
                    avg_win=row["avg_win"],
                    avg_loss=row["avg_loss"],
                    current_weight=row["current_weight"],
                    methylation_level=row["methylation_level"],
                    domestication_boost=row["domestication_boost"],
                    stage_entered_at=row["stage_entered_at"] or "",
                    first_seen=row["first_seen"] or "",
                    last_seen=row["last_seen"] or "",
                    last_reeval=row["last_reeval"] or "",
                    immune_response_speed=row["immune_response_speed"],
                    de_domestication_count=row["de_domestication_count"],
                    notes=row["notes"] or "",
                    post_dom_wins=row["post_dom_wins"],
                    post_dom_losses=row["post_dom_losses"],
                )
        except Exception as e:
            log.warning("[KoRV] Failed to load record %s: %s", record_id, e)
            return None

    def _save_record(self, rec: SignalOnboardRecord):
        """Persist a signal onboard record to the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT OR REPLACE INTO signal_onboard
                    (record_id, signal_type, instrument, timeframe,
                     population_key, stage, win_count, loss_count,
                     total_trades, posterior_wr, posterior_lr,
                     profit_factor, total_pnl, avg_win, avg_loss,
                     current_weight, methylation_level, domestication_boost,
                     stage_entered_at, first_seen, last_seen, last_reeval,
                     immune_response_speed, de_domestication_count, notes,
                     post_dom_wins, post_dom_losses)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?)
                """, (
                    rec.record_id, rec.signal_type, rec.instrument,
                    rec.timeframe, rec.population_key, rec.stage,
                    rec.win_count, rec.loss_count, rec.total_trades,
                    rec.posterior_wr, rec.posterior_lr,
                    rec.profit_factor, rec.total_pnl,
                    rec.avg_win, rec.avg_loss,
                    rec.current_weight, rec.methylation_level,
                    rec.domestication_boost,
                    rec.stage_entered_at, rec.first_seen,
                    rec.last_seen, rec.last_reeval,
                    rec.immune_response_speed,
                    rec.de_domestication_count, rec.notes,
                    rec.post_dom_wins, rec.post_dom_losses,
                ))
                conn.commit()
        except Exception as e:
            log.warning("[KoRV] Failed to save record %s: %s", rec.record_id, e)

    # ----------------------------------------------------------
    # PHASE 1: INFECTION -- New Signal Registration
    # ----------------------------------------------------------

    def register_new_signal(
        self,
        signal_type: str,
        instrument: str,
        timeframe: str,
    ) -> SignalOnboardRecord:
        """
        Register a new signal type for a specific population.

        This is the "initial infection" -- the retrovirus has entered
        the genome and begins its probationary period.

        Args:
            signal_type: Name of the new signal/indicator
            instrument:  Trading instrument (e.g. "XAUUSD")
            timeframe:   Timeframe (e.g. "M5", "H1")

        Returns:
            The newly created (or existing) SignalOnboardRecord.
        """
        record_id = self._make_record_id(signal_type, instrument, timeframe)

        # Check if already registered
        existing = self._load_record(record_id)
        if existing is not None:
            return existing

        now = self._now_iso()
        initial_wr = PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_BETA)

        rec = SignalOnboardRecord(
            record_id=record_id,
            signal_type=signal_type,
            instrument=instrument,
            timeframe=timeframe,
            population_key=self._make_population_key(
                signal_type, instrument, timeframe
            ),
            stage=STAGE_INFECTION,
            win_count=0,
            loss_count=0,
            total_trades=0,
            posterior_wr=initial_wr,
            posterior_lr=1.0 - initial_wr,
            profit_factor=0.0,
            total_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            current_weight=1.0,
            methylation_level=0.0,
            domestication_boost=1.0,
            stage_entered_at=now,
            first_seen=now,
            last_seen=now,
            last_reeval=now,
            immune_response_speed=IMMUNE_RESPONSE_SPEED_SLOW,
            de_domestication_count=0,
            notes="",
            post_dom_wins=0,
            post_dom_losses=0,
        )

        self._save_record(rec)
        self._log_stage_transition(
            record_id, "none", STAGE_INFECTION,
            "New signal registered", rec,
        )

        log.info(
            "[KoRV] New signal registered: %s | %s/%s | stage=INFECTION",
            signal_type, instrument, timeframe,
        )

        return rec

    # ----------------------------------------------------------
    # PHASE 2: OUTCOME RECORDING
    # ----------------------------------------------------------

    def record_outcome(
        self,
        signal_type: str,
        instrument: str,
        timeframe: str,
        won: bool,
        pnl: float = 0.0,
    ) -> SignalOnboardRecord:
        """
        Record a trade outcome for a signal in a specific population.

        This is the feedback loop -- every trade result updates the signal's
        posterior win rate and triggers stage evaluation.

        Args:
            signal_type: Name of the signal that contributed to this trade
            instrument:  Trading instrument
            timeframe:   Timeframe
            won:         True if trade was profitable
            pnl:         Profit/loss amount in dollars

        Returns:
            Updated SignalOnboardRecord.
        """
        record_id = self._make_record_id(signal_type, instrument, timeframe)

        rec = self._load_record(record_id)
        if rec is None:
            rec = self.register_new_signal(signal_type, instrument, timeframe)

        # Update counts
        if won:
            rec.win_count += 1
        else:
            rec.loss_count += 1
        rec.total_trades = rec.win_count + rec.loss_count
        rec.total_pnl += pnl
        rec.last_seen = self._now_iso()

        # Track post-domestication outcomes separately
        if rec.stage == STAGE_DOMESTICATED:
            if won:
                rec.post_dom_wins += 1
            else:
                rec.post_dom_losses += 1

        # Running averages
        if won and pnl > 0:
            rec.avg_win = _running_average(rec.avg_win, pnl, rec.win_count)
        elif not won and pnl < 0:
            rec.avg_loss = _running_average(rec.avg_loss, abs(pnl), rec.loss_count)

        # Bayesian posterior
        rec.posterior_wr = (PRIOR_ALPHA + rec.win_count) / (
            PRIOR_ALPHA + PRIOR_BETA + rec.total_trades
        )
        rec.posterior_lr = 1.0 - rec.posterior_wr

        # Profit factor
        if rec.avg_loss > 0:
            rec.profit_factor = rec.avg_win / rec.avg_loss
        elif rec.avg_win > 0:
            rec.profit_factor = 99.0
        else:
            rec.profit_factor = 0.0

        # Save before stage evaluation (so eval sees updated stats)
        self._save_record(rec)

        # Trigger stage evaluation
        self._evaluate_stage_transition(rec)

        return rec

    # ----------------------------------------------------------
    # PHASE 3: STAGE EVALUATION ENGINE
    # ----------------------------------------------------------

    def _evaluate_stage_transition(self, rec: SignalOnboardRecord):
        """
        Evaluate whether a signal should transition between stages.

        This is the core domestication logic -- the algorithm that decides
        whether KoRV becomes part of the host genome or gets silenced.
        """
        current_stage = rec.stage

        if current_stage == STAGE_INFECTION:
            self._eval_from_infection(rec)
        elif current_stage == STAGE_IMMUNE_RESPONSE:
            self._eval_from_immune_response(rec)
        elif current_stage == STAGE_TOLERANCE:
            self._eval_from_tolerance(rec)
        elif current_stage == STAGE_DOMESTICATED:
            self._eval_from_domesticated(rec)
        # DORMANT: no evaluation -- signal is parked

    def _eval_from_infection(self, rec: SignalOnboardRecord):
        """
        Evaluate a signal in the INFECTION (probationary) stage.

        Must meet minimum observation window before any transition.
        """
        if rec.total_trades < INFECTION_MIN_TRADES:
            return  # Still collecting data

        # Check calendar minimum
        days_in_stage = self._days_since(rec.stage_entered_at)
        if days_in_stage < INFECTION_MIN_DAYS and rec.total_trades < INFECTION_MAX_TRADES:
            return  # Still in calendar probation

        # TOXIC: posterior loss rate exceeds threshold -> IMMUNE_RESPONSE
        if rec.posterior_lr >= METHYLATION_TOXICITY_THRESHOLD:
            self._transition_to_immune_response(rec)
            return

        # BENEFICIAL: strong WR + good PF + enough trades -> DOMESTICATED
        if (
            rec.posterior_wr >= DOMESTICATION_MIN_POSTERIOR_WR
            and rec.profit_factor >= DOMESTICATION_MIN_PROFIT_FACTOR
            and rec.total_trades >= DOMESTICATION_MIN_TRADES
        ):
            self._transition_to_domesticated(rec)
            return

        # NEUTRAL: win rate in the tolerance band -> TOLERANCE
        if TOLERANCE_WR_LOW <= rec.posterior_wr <= TOLERANCE_WR_HIGH:
            self._transition_to_tolerance(rec)
            return

        # AMBIGUOUS: between neutral and toxic/beneficial -- keep observing
        if rec.total_trades >= INFECTION_MAX_TRADES:
            # Force a decision at max trades
            if rec.posterior_wr > 0.50:
                self._transition_to_tolerance(rec)
            else:
                self._transition_to_immune_response(rec)

    def _eval_from_immune_response(self, rec: SignalOnboardRecord):
        """
        Evaluate a signal under IMMUNE_RESPONSE (methylation/silencing).

        Methylation escalates with continued poor performance.
        Recovery is possible if performance improves.
        """
        # Extreme toxicity: fast-track silencing
        if rec.posterior_lr >= METHYLATION_SILENCE_LR:
            rec.methylation_level = min(1.0, rec.methylation_level + 0.20)
            rec.current_weight = max(
                METHYLATION_SILENCED_WEIGHT,
                1.0 - rec.methylation_level,
            )

        # Escalating methylation
        elif rec.posterior_lr >= METHYLATION_ESCALATION_LR:
            trades_in_stage = self._trades_since_stage_entered(rec)
            rate = 0.10 if trades_in_stage < rec.immune_response_speed else 0.05
            rec.methylation_level = min(1.0, rec.methylation_level + rate)
            rec.current_weight = max(
                METHYLATION_HEAVY_WEIGHT,
                1.0 - rec.methylation_level,
            )

        # Still above toxicity threshold: hold position
        elif rec.posterior_lr >= METHYLATION_TOXICITY_THRESHOLD:
            pass  # No change

        # Performance improving: de-methylation possible
        else:
            trades_in_stage = self._trades_since_stage_entered(rec)

            # Remarkable recovery -> domesticate
            if (
                rec.posterior_wr >= DOMESTICATION_MIN_POSTERIOR_WR
                and trades_in_stage >= 20
                and rec.profit_factor >= DOMESTICATION_MIN_PROFIT_FACTOR
            ):
                self._transition_to_domesticated(rec)
                return

            # Stopped being toxic -> move toward tolerance
            if rec.posterior_lr < METHYLATION_TOXICITY_THRESHOLD:
                rec.methylation_level = max(0.0, rec.methylation_level - 0.10)
                rec.current_weight = 1.0 - rec.methylation_level
                if rec.methylation_level <= 0.0:
                    self._transition_to_tolerance(rec)
                    return

        self._save_record(rec)

    def _eval_from_tolerance(self, rec: SignalOnboardRecord):
        """
        Evaluate a signal in TOLERANCE (neutral coexistence).

        Periodic re-evaluation checks for toxicity onset or domestication.
        """
        days_since_reeval = self._days_since(rec.last_reeval)
        if days_since_reeval < TOLERANCE_REEVAL_INTERVAL_DAYS:
            return  # Not time for re-evaluation yet

        rec.last_reeval = self._now_iso()

        # Check for idleness
        days_idle = self._days_since(rec.last_seen)
        if days_idle > TOLERANCE_MAX_IDLE_DAYS:
            self._transition_to_dormant(rec)
            return

        # Turned toxic: move to immune response
        if rec.posterior_lr >= METHYLATION_TOXICITY_THRESHOLD:
            self._transition_to_immune_response(rec)
            return

        # Became beneficial: domesticate
        if (
            rec.posterior_wr >= DOMESTICATION_MIN_POSTERIOR_WR
            and rec.profit_factor >= DOMESTICATION_MIN_PROFIT_FACTOR
            and rec.total_trades >= DOMESTICATION_MIN_TRADES
        ):
            self._transition_to_domesticated(rec)
            return

        # Still neutral: no change
        self._save_record(rec)

    def _eval_from_domesticated(self, rec: SignalOnboardRecord):
        """
        Evaluate a DOMESTICATED signal for potential de-domestication.

        De-domestication is rare and requires strong evidence of collapse.
        This mirrors KoRV copies reverting to pathogenic behavior.
        """
        post_dom_total = rec.post_dom_wins + rec.post_dom_losses

        if post_dom_total >= DOMESTICATION_DE_TRIGGER_TRADES:
            recent_wr = rec.post_dom_wins / post_dom_total if post_dom_total > 0 else 0.5

            if recent_wr < DOMESTICATION_DE_TRIGGER_WR:
                # Performance collapsed: de-domesticate
                rec.de_domestication_count += 1
                self._transition_to_immune_response(rec)
                return

        # Update boost based on overall posterior WR
        wr_delta = rec.posterior_wr - 0.55
        boost_range = DOMESTICATION_BOOST_MAX - DOMESTICATION_BOOST_BASE
        boost = DOMESTICATION_BOOST_BASE + boost_range * _sigmoid(10.0 * wr_delta)
        rec.domestication_boost = min(DOMESTICATION_BOOST_MAX, boost)
        rec.current_weight = rec.domestication_boost

        self._save_record(rec)

    # ----------------------------------------------------------
    # STAGE TRANSITION FUNCTIONS
    # ----------------------------------------------------------

    def _transition_to_immune_response(self, rec: SignalOnboardRecord):
        """Transition a signal to IMMUNE_RESPONSE (methylation/silencing)."""
        previous = rec.stage
        rec.stage = STAGE_IMMUNE_RESPONSE
        rec.stage_entered_at = self._now_iso()
        rec.domestication_boost = 1.0
        rec.post_dom_wins = 0
        rec.post_dom_losses = 0

        # Faster immune response for previously de-domesticated signals
        if rec.de_domestication_count > 0:
            rec.immune_response_speed = IMMUNE_RESPONSE_SPEED_FAST
            rec.methylation_level = 0.50
            rec.current_weight = METHYLATION_HEAVY_WEIGHT
        else:
            rec.immune_response_speed = IMMUNE_RESPONSE_SPEED_SLOW
            rec.methylation_level = 0.30
            rec.current_weight = METHYLATION_INITIAL_WEIGHT

        self._save_record(rec)
        self._log_stage_transition(
            rec.record_id, previous, STAGE_IMMUNE_RESPONSE,
            f"Posterior LR={rec.posterior_lr:.1%}, "
            f"de_dom_count={rec.de_domestication_count}",
            rec,
        )

    def _transition_to_tolerance(self, rec: SignalOnboardRecord):
        """Transition a signal to TOLERANCE (neutral coexistence)."""
        previous = rec.stage
        rec.stage = STAGE_TOLERANCE
        rec.stage_entered_at = self._now_iso()
        rec.last_reeval = self._now_iso()
        rec.methylation_level = 0.0
        rec.current_weight = 1.0
        rec.domestication_boost = 1.0
        rec.post_dom_wins = 0
        rec.post_dom_losses = 0

        self._save_record(rec)
        self._log_stage_transition(
            rec.record_id, previous, STAGE_TOLERANCE,
            f"Posterior WR={rec.posterior_wr:.1%}",
            rec,
        )

    def _transition_to_domesticated(self, rec: SignalOnboardRecord):
        """Transition a signal to DOMESTICATED (permanent integration)."""
        previous = rec.stage
        rec.stage = STAGE_DOMESTICATED
        rec.stage_entered_at = self._now_iso()
        rec.methylation_level = 0.0
        rec.domestication_boost = DOMESTICATION_BOOST_BASE
        rec.current_weight = DOMESTICATION_BOOST_BASE
        rec.post_dom_wins = 0
        rec.post_dom_losses = 0

        self._save_record(rec)
        self._log_stage_transition(
            rec.record_id, previous, STAGE_DOMESTICATED,
            f"Posterior WR={rec.posterior_wr:.1%}, PF={rec.profit_factor:.2f}",
            rec,
        )

    def _transition_to_dormant(self, rec: SignalOnboardRecord):
        """Transition a signal to DORMANT (idle, no longer tracked)."""
        previous = rec.stage
        rec.stage = STAGE_DORMANT
        rec.stage_entered_at = self._now_iso()
        rec.current_weight = 0.0

        self._save_record(rec)
        self._log_stage_transition(
            rec.record_id, previous, STAGE_DORMANT,
            f"Idle for >{TOLERANCE_MAX_IDLE_DAYS} days",
            rec,
        )

    # ----------------------------------------------------------
    # PHASE 4: WEIGHT QUERY (called during signal generation)
    # ----------------------------------------------------------

    def get_signal_weight(
        self,
        signal_type: str,
        instrument: str,
        timeframe: str,
    ) -> float:
        """
        Get the current weight modifier for a signal in a population.

        This is the primary query interface for BRAIN scripts and the
        TEQA pipeline. Returns a float that modifies signal influence:

            INFECTION:       1.00 (unmodified, under observation)
            IMMUNE_RESPONSE: 0.05 to 0.70 (methylation-dependent)
            TOLERANCE:       1.00 (neutral)
            DOMESTICATED:    1.10 to 1.35 (boosted)
            DORMANT:         0.00 (inactive)
            Unknown:         1.00 (not yet registered)

        Args:
            signal_type: Signal name
            instrument:  Trading instrument
            timeframe:   Timeframe

        Returns:
            Weight modifier (float).
        """
        record_id = self._make_record_id(signal_type, instrument, timeframe)
        rec = self._load_record(record_id)

        if rec is None:
            return 1.0  # Unknown signal, no modification

        return rec.current_weight

    def get_signal_stage(
        self,
        signal_type: str,
        instrument: str,
        timeframe: str,
    ) -> str:
        """Get the current stage for a signal in a population."""
        record_id = self._make_record_id(signal_type, instrument, timeframe)
        rec = self._load_record(record_id)
        if rec is None:
            return "unknown"
        return rec.stage

    def get_signal_record(
        self,
        signal_type: str,
        instrument: str,
        timeframe: str,
    ) -> Optional[SignalOnboardRecord]:
        """Get the full record for a signal in a population."""
        record_id = self._make_record_id(signal_type, instrument, timeframe)
        return self._load_record(record_id)

    def get_all_records(self) -> List[SignalOnboardRecord]:
        """Load all signal onboard records."""
        records = []
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT record_id FROM signal_onboard ORDER BY last_seen DESC")
                for row in cursor.fetchall():
                    rec = self._load_record(row["record_id"])
                    if rec is not None:
                        records.append(rec)
        except Exception as e:
            log.warning("[KoRV] Failed to load all records: %s", e)
        return records

    # ----------------------------------------------------------
    # PHASE 5: RECOMBINATION RISK -- Interaction Tracking
    # ----------------------------------------------------------

    def record_interaction(
        self,
        signal_a: str,
        signal_b: str,
        instrument: str,
        timeframe: str,
        won: bool,
    ):
        """
        Record a co-occurrence of two signals on the same trade.

        This tracks potential recombination effects -- when a new signal
        (KoRV) fires alongside an existing domesticated signal, the
        combination may be synergistic, neutral, or interfering.

        Args:
            signal_a: First signal type
            signal_b: Second signal type
            instrument: Trading instrument
            timeframe: Timeframe
            won: Whether the trade was profitable
        """
        interaction_id = self._make_interaction_id(
            signal_a, signal_b, instrument, timeframe
        )

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT * FROM interactions WHERE interaction_id = ?",
                    (interaction_id,),
                )
                row = cursor.fetchone()
                now = self._now_iso()

                if row is None:
                    # First co-occurrence
                    conn.execute("""
                        INSERT INTO interactions
                        (interaction_id, signal_a, signal_b, instrument,
                         timeframe, co_occurrence_count, co_win_count,
                         co_loss_count, first_seen, last_seen)
                        VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                    """, (
                        interaction_id,
                        min(signal_a, signal_b),
                        max(signal_a, signal_b),
                        instrument, timeframe,
                        1 if won else 0,
                        0 if won else 1,
                        now, now,
                    ))
                else:
                    co_count = row["co_occurrence_count"] + 1
                    co_wins = row["co_win_count"] + (1 if won else 0)
                    co_losses = row["co_loss_count"] + (0 if won else 1)

                    # Compute synergy score after enough data
                    combined_wr = co_wins / co_count if co_count > 0 else 0.5
                    synergy = 0.0
                    classification = "neutral"

                    if co_count >= INTERACTION_MIN_CO_OCCURRENCES:
                        wr_a = self._get_population_wr(
                            min(signal_a, signal_b), instrument, timeframe
                        )
                        wr_b = self._get_population_wr(
                            max(signal_a, signal_b), instrument, timeframe
                        )
                        expected = max(wr_a, wr_b)
                        synergy = combined_wr - expected

                        if synergy > INTERACTION_SYNERGY_THRESHOLD:
                            classification = "synergistic"
                        elif synergy < INTERACTION_TOXICITY_THRESHOLD:
                            classification = "interfering"
                        else:
                            classification = "neutral"

                    conn.execute("""
                        UPDATE interactions SET
                            co_occurrence_count = ?,
                            co_win_count = ?,
                            co_loss_count = ?,
                            combined_wr = ?,
                            synergy_score = ?,
                            classification = ?,
                            individual_wr_a = ?,
                            individual_wr_b = ?,
                            last_seen = ?
                        WHERE interaction_id = ?
                    """, (
                        co_count, co_wins, co_losses,
                        combined_wr, synergy, classification,
                        self._get_population_wr(
                            min(signal_a, signal_b), instrument, timeframe
                        ),
                        self._get_population_wr(
                            max(signal_a, signal_b), instrument, timeframe
                        ),
                        now,
                        interaction_id,
                    ))

                conn.commit()

        except Exception as e:
            log.warning("[KoRV] Failed to record interaction: %s", e)

    def _get_population_wr(
        self, signal_type: str, instrument: str, timeframe: str
    ) -> float:
        """Get the posterior win rate for a signal's population."""
        record_id = self._make_record_id(signal_type, instrument, timeframe)
        rec = self._load_record(record_id)
        if rec is None:
            return 0.5
        return rec.posterior_wr

    def get_interactions(
        self,
        instrument: str = None,
        classification: str = None,
        limit: int = 20,
    ) -> List[InteractionRecord]:
        """Query interaction records with optional filters."""
        records = []
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM interactions WHERE 1=1"
                params = []
                if instrument:
                    query += " AND instrument = ?"
                    params.append(instrument)
                if classification:
                    query += " AND classification = ?"
                    params.append(classification)
                query += " ORDER BY co_occurrence_count DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                for row in cursor.fetchall():
                    records.append(InteractionRecord(
                        interaction_id=row["interaction_id"],
                        signal_a=row["signal_a"],
                        signal_b=row["signal_b"],
                        instrument=row["instrument"],
                        timeframe=row["timeframe"],
                        co_occurrence_count=row["co_occurrence_count"],
                        co_win_count=row["co_win_count"],
                        co_loss_count=row["co_loss_count"],
                        individual_wr_a=row["individual_wr_a"],
                        individual_wr_b=row["individual_wr_b"],
                        combined_wr=row["combined_wr"],
                        synergy_score=row["synergy_score"],
                        classification=row["classification"],
                        first_seen=row["first_seen"] or "",
                        last_seen=row["last_seen"] or "",
                    ))
        except Exception as e:
            log.warning("[KoRV] Failed to query interactions: %s", e)
        return records

    # ----------------------------------------------------------
    # PHASE 6: ENDOGENIZATION REPORT
    # ----------------------------------------------------------

    def generate_endogenization_report(self) -> Dict:
        """
        Generate a comprehensive report on the current state of
        signal domestication across all populations.

        This is the genomic census -- how many KoRV copies are at each
        stage of integration across all koala populations.

        Returns:
            Dict with stage counts, instrument breakdown, interaction
            hotspots, and recent transitions.
        """
        report = {
            "timestamp": self._now_iso(),
            "version": VERSION,
            "stage_counts": {},
            "instrument_breakdown": {},
            "top_synergies": [],
            "top_interferences": [],
            "recent_transitions": [],
            "total_signals": 0,
            "domestication_rate": 0.0,
        }

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Stage counts
                cursor.execute("""
                    SELECT stage, COUNT(*) as cnt,
                           AVG(posterior_wr) as avg_wr,
                           AVG(current_weight) as avg_weight,
                           SUM(total_trades) as total_trades
                    FROM signal_onboard
                    GROUP BY stage
                """)
                total = 0
                domesticated_count = 0
                for row in cursor.fetchall():
                    stage = row["stage"]
                    cnt = row["cnt"]
                    report["stage_counts"][stage] = {
                        "count": cnt,
                        "avg_wr": round(row["avg_wr"] or 0.5, 4),
                        "avg_weight": round(row["avg_weight"] or 1.0, 4),
                        "total_trades": row["total_trades"] or 0,
                    }
                    total += cnt
                    if stage == STAGE_DOMESTICATED:
                        domesticated_count = cnt

                report["total_signals"] = total
                report["domestication_rate"] = (
                    domesticated_count / total if total > 0 else 0.0
                )

                # Per-instrument breakdown
                cursor.execute("""
                    SELECT instrument, stage, COUNT(*) as cnt,
                           AVG(posterior_wr) as avg_wr,
                           AVG(profit_factor) as avg_pf
                    FROM signal_onboard
                    GROUP BY instrument, stage
                    ORDER BY instrument, stage
                """)
                for row in cursor.fetchall():
                    inst = row["instrument"]
                    if inst not in report["instrument_breakdown"]:
                        report["instrument_breakdown"][inst] = {}
                    report["instrument_breakdown"][inst][row["stage"]] = {
                        "count": row["cnt"],
                        "avg_wr": round(row["avg_wr"] or 0.5, 4),
                        "avg_pf": round(row["avg_pf"] or 0.0, 2),
                    }

                # Top synergies
                cursor.execute("""
                    SELECT * FROM interactions
                    WHERE classification = 'synergistic'
                    ORDER BY synergy_score DESC LIMIT 10
                """)
                for row in cursor.fetchall():
                    report["top_synergies"].append({
                        "signal_a": row["signal_a"],
                        "signal_b": row["signal_b"],
                        "instrument": row["instrument"],
                        "synergy_score": round(row["synergy_score"], 4),
                        "combined_wr": round(row["combined_wr"], 4),
                        "co_occurrences": row["co_occurrence_count"],
                    })

                # Top interferences
                cursor.execute("""
                    SELECT * FROM interactions
                    WHERE classification = 'interfering'
                    ORDER BY synergy_score ASC LIMIT 10
                """)
                for row in cursor.fetchall():
                    report["top_interferences"].append({
                        "signal_a": row["signal_a"],
                        "signal_b": row["signal_b"],
                        "instrument": row["instrument"],
                        "synergy_score": round(row["synergy_score"], 4),
                        "combined_wr": round(row["combined_wr"], 4),
                        "co_occurrences": row["co_occurrence_count"],
                    })

                # Recent transitions
                cursor.execute("""
                    SELECT * FROM stage_transitions
                    ORDER BY timestamp DESC LIMIT 20
                """)
                for row in cursor.fetchall():
                    report["recent_transitions"].append({
                        "record_id": row["record_id"],
                        "from": row["from_stage"],
                        "to": row["to_stage"],
                        "reason": row["reason"],
                        "wr": round(row["posterior_wr"], 4),
                        "pf": round(row["profit_factor"], 2),
                        "trades": row["total_trades"],
                        "weight": round(row["current_weight"], 4),
                        "timestamp": row["timestamp"],
                    })

        except Exception as e:
            log.warning("[KoRV] Failed to generate report: %s", e)

        return report

    def maybe_generate_report(self) -> Optional[Dict]:
        """
        Generate a report if enough time has elapsed since the last one.
        Returns None if not yet time.
        """
        now = time.time()
        if now - self._last_report_time >= ENDOGENIZATION_REPORT_INTERVAL_SEC:
            self._last_report_time = now
            return self.generate_endogenization_report()
        return None

    # ----------------------------------------------------------
    # BATCH OPERATIONS
    # ----------------------------------------------------------

    def record_multi_signal_outcome(
        self,
        signal_types: List[str],
        instrument: str,
        timeframe: str,
        won: bool,
        pnl: float = 0.0,
    ):
        """
        Record a trade outcome for multiple signals that fired together.

        Handles both individual signal tracking AND pairwise interaction
        tracking for all signal combinations.

        Args:
            signal_types: List of signal type names that contributed
            instrument:   Trading instrument
            timeframe:    Timeframe
            won:          Whether the trade was profitable
            pnl:          Profit/loss amount
        """
        # Record individual outcomes
        for sig in signal_types:
            self.record_outcome(sig, instrument, timeframe, won, pnl)

        # Record all pairwise interactions
        for i in range(len(signal_types)):
            for j in range(i + 1, len(signal_types)):
                self.record_interaction(
                    signal_types[i], signal_types[j],
                    instrument, timeframe, won,
                )

    def get_population_weights(
        self,
        signal_types: List[str],
        instrument: str,
        timeframe: str,
    ) -> Dict[str, float]:
        """
        Get weights for multiple signals in a specific population.

        Args:
            signal_types: List of signal names
            instrument:   Trading instrument
            timeframe:    Timeframe

        Returns:
            Dict mapping signal_type -> weight.
        """
        weights = {}
        for sig in signal_types:
            weights[sig] = self.get_signal_weight(sig, instrument, timeframe)
        return weights

    def get_population_summary(self, instrument: str) -> Dict:
        """
        Get a summary of all signals for a specific instrument.

        Returns:
            Dict with per-stage counts and signal details.
        """
        summary = {
            "instrument": instrument,
            "stages": {},
            "signals": [],
        }

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM signal_onboard
                    WHERE instrument = ?
                    ORDER BY stage, posterior_wr DESC
                """, (instrument,))

                for row in cursor.fetchall():
                    stage = row["stage"]
                    if stage not in summary["stages"]:
                        summary["stages"][stage] = 0
                    summary["stages"][stage] += 1

                    summary["signals"].append({
                        "signal_type": row["signal_type"],
                        "timeframe": row["timeframe"],
                        "stage": stage,
                        "wr": round(row["posterior_wr"], 4),
                        "pf": round(row["profit_factor"], 2),
                        "weight": round(row["current_weight"], 4),
                        "methylation": round(row["methylation_level"], 4),
                        "trades": row["total_trades"],
                        "pnl": round(row["total_pnl"], 2),
                    })

        except Exception as e:
            log.warning("[KoRV] Failed to get population summary: %s", e)

        return summary

    # ----------------------------------------------------------
    # INTERNAL UTILITIES
    # ----------------------------------------------------------

    def _days_since(self, iso_timestamp: str) -> float:
        """Calculate days elapsed since an ISO timestamp."""
        if not iso_timestamp:
            return 999.0
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            delta = datetime.now() - dt
            return delta.total_seconds() / 86400.0
        except (ValueError, TypeError):
            return 999.0

    def _trades_since_stage_entered(self, rec: SignalOnboardRecord) -> int:
        """
        Estimate trades since the signal entered its current stage.

        Uses the stage transition log for accuracy.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT total_trades FROM stage_transitions
                    WHERE record_id = ? AND to_stage = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (rec.record_id, rec.stage))
                row = cursor.fetchone()
                if row is not None:
                    trades_at_transition = row["total_trades"]
                    return max(0, rec.total_trades - trades_at_transition)
        except Exception:
            pass

        # Fallback: return total trades (conservative estimate)
        return rec.total_trades


# ============================================================
# TEQA INTEGRATION: KoRV Bridge
# ============================================================

class KoRVTEQABridge:
    """
    Bridges the KoRV Domestication Engine with the existing TEQA v3.0
    pipeline and BRAIN scripts.

    The bridge:
      1. Provides per-signal weight modifiers for the TEQA confidence calculation
      2. Records trade outcomes back to the KoRV engine
      3. Writes a status JSON file for monitoring
      4. Integrates with the existing domestication and CRISPR systems

    This is how the real-time retroviral domestication process interfaces
    with the rest of the trading immune system.
    """

    def __init__(
        self,
        engine: KoRVDomesticationEngine,
        status_file: str = None,
    ):
        self.engine = engine
        if status_file is None:
            self.status_file = str(
                Path(__file__).parent / "korv_status.json"
            )
        else:
            self.status_file = status_file

    def get_weighted_confidence(
        self,
        base_confidence: float,
        signal_types: List[str],
        instrument: str,
        timeframe: str,
    ) -> float:
        """
        Apply KoRV weights to a base confidence score.

        For multiple signals, the combined weight is the geometric mean
        of individual weights. This ensures that:
          - One silenced signal heavily drags down the combined weight
          - One boosted signal moderately lifts the combined weight
          - Neutral signals (1.0) have no effect

        Args:
            base_confidence: Original confidence from TEQA pipeline
            signal_types:    List of contributing signal names
            instrument:      Trading instrument
            timeframe:       Timeframe

        Returns:
            Modified confidence value.
        """
        if not signal_types:
            return base_confidence

        weights = self.engine.get_population_weights(
            signal_types, instrument, timeframe
        )

        if not weights:
            return base_confidence

        # Geometric mean of weights
        product = 1.0
        for w in weights.values():
            product *= max(0.01, w)  # Floor at 0.01 to avoid zero
        geo_mean = product ** (1.0 / len(weights))

        modified = base_confidence * geo_mean
        return max(0.0, min(1.0, modified))

    def record_trade_result(
        self,
        signal_types: List[str],
        instrument: str,
        timeframe: str,
        won: bool,
        pnl: float = 0.0,
    ):
        """
        Record a trade result for all contributing signals.

        Should be called after every trade closes, with the list of
        signal types that contributed to the entry decision.
        """
        self.engine.record_multi_signal_outcome(
            signal_types, instrument, timeframe, won, pnl
        )

    def write_status_file(self):
        """
        Write current KoRV domestication status to JSON for monitoring.

        This file can be read by dashboards, MQL5 EAs, or other processes
        that need to know the current state of signal onboarding.
        """
        report = self.engine.generate_endogenization_report()

        try:
            tmp_path = self.status_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            os.replace(tmp_path, self.status_file)
        except Exception as e:
            log.warning("[KoRV] Failed to write status file: %s", e)

    def get_gate_result(
        self,
        signal_types: List[str],
        instrument: str,
        timeframe: str,
    ) -> Dict:
        """
        Compute a gate result for integration with the Jardine's Gate system.

        The KoRV gate checks whether any contributing signal is under
        heavy methylation (silencing). If the combined weight is too low,
        the gate blocks the trade.

        Returns:
            {
                "gate_pass": bool,
                "combined_weight": float,
                "signal_weights": dict,
                "any_silenced": bool,
                "any_domesticated": bool,
            }
        """
        weights = self.engine.get_population_weights(
            signal_types, instrument, timeframe
        )

        any_silenced = any(w < METHYLATION_HEAVY_WEIGHT for w in weights.values())
        any_domesticated = any(w > 1.0 for w in weights.values())

        # Combined weight (geometric mean)
        if weights:
            product = 1.0
            for w in weights.values():
                product *= max(0.01, w)
            combined = product ** (1.0 / len(weights))
        else:
            combined = 1.0

        # Gate passes if combined weight is above a threshold
        # A silenced signal (0.05) among normal signals drags the
        # geometric mean below the confidence threshold
        gate_pass = combined >= CONFIDENCE_THRESHOLD

        return {
            "gate_pass": gate_pass,
            "combined_weight": round(combined, 4),
            "signal_weights": {k: round(v, 4) for k, v in weights.items()},
            "any_silenced": any_silenced,
            "any_domesticated": any_domesticated,
        }


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
    print("  KoRV DOMESTICATION ENGINE -- Real-Time Retroviral Signal Onboarding")
    print("  Algorithm #5: Koala Retrovirus Domestication")
    print("=" * 76)

    # Use temp DB for testing
    test_db = str(Path(__file__).parent / "test_korv_domestication.db")

    # Clean up previous test
    try:
        os.remove(test_db)
    except OSError:
        pass

    engine = KoRVDomesticationEngine(db_path=test_db)
    bridge = KoRVTEQABridge(engine)

    # Helper: backdate timestamps so calendar checks pass in test
    def _backdate_record(sig_type, inst, tf, days_ago=10):
        """Simulate that the signal was registered N days ago."""
        rid = engine._make_record_id(sig_type, inst, tf)
        past = (datetime.now() - timedelta(days=days_ago)).isoformat()
        try:
            with sqlite3.connect(test_db, timeout=5) as conn:
                conn.execute(
                    "UPDATE signal_onboard SET stage_entered_at = ?, last_reeval = ? "
                    "WHERE record_id = ?",
                    (past, past, rid),
                )
                conn.commit()
        except Exception:
            pass

    # ----- Scenario 1: TOXIC signal gets methylated/silenced -----
    print("\n  --- SCENARIO 1: TOXIC SIGNAL (gets silenced) ---")
    print("  Signal: 'lunar_phase_indicator' on XAUUSD M5")
    print("  This signal is garbage -- feeds only losses\n")

    np.random.seed(42)
    engine.register_new_signal("lunar_phase_indicator", "XAUUSD", "M5")
    _backdate_record("lunar_phase_indicator", "XAUUSD", "M5")

    # Phase A: deterministic toxic trades to force immune response
    for i in range(20):
        # Deterministic: 3 wins out of 20 = 15% raw WR
        won = (i % 7 == 0)
        pnl = 0.80 if won else -1.20
        engine.record_outcome("lunar_phase_indicator", "XAUUSD", "M5", won, pnl)

    rec = engine.get_signal_record("lunar_phase_indicator", "XAUUSD", "M5")
    print(f"  After 20 trades (deterministic toxic):")
    print(f"    Stage:        {rec.stage}")
    print(f"    Weight:       {rec.current_weight:.4f}")
    print(f"    Methylation:  {rec.methylation_level:.4f}")
    print(f"    Posterior WR: {rec.posterior_wr:.1%}")
    print(f"    Posterior LR: {rec.posterior_lr:.1%}")

    # Phase B: more toxic trades to escalate methylation
    for i in range(20):
        won = (i % 10 == 0)
        pnl = 0.60 if won else -1.10
        engine.record_outcome("lunar_phase_indicator", "XAUUSD", "M5", won, pnl)

    rec = engine.get_signal_record("lunar_phase_indicator", "XAUUSD", "M5")
    print(f"  After 40 total trades:")
    print(f"    Stage:        {rec.stage}")
    print(f"    Weight:       {rec.current_weight:.4f}")
    print(f"    Methylation:  {rec.methylation_level:.4f}")
    print(f"    Posterior LR: {rec.posterior_lr:.1%}")
    print(f"    Total PnL:    ${rec.total_pnl:.2f}")

    # ----- Scenario 2: PROFITABLE signal gets domesticated -----
    print("\n  --- SCENARIO 2: PROFITABLE SIGNAL (gets domesticated) ---")
    print("  Signal: 'order_flow_imbalance_v2' on XAUUSD M5")
    print("  This signal has a 70%+ win rate with good PF\n")

    engine.register_new_signal("order_flow_imbalance_v2", "XAUUSD", "M5")
    _backdate_record("order_flow_imbalance_v2", "XAUUSD", "M5")

    # Deterministic: 7 wins per 10 trades = 70% WR, good PF
    for i in range(50):
        won = (i % 10) < 7  # 70% win rate deterministic
        pnl = np.random.uniform(1.50, 2.50) if won else -np.random.uniform(0.60, 0.90)
        engine.record_outcome("order_flow_imbalance_v2", "XAUUSD", "M5", won, pnl)

    rec = engine.get_signal_record("order_flow_imbalance_v2", "XAUUSD", "M5")
    print(f"  Stage:        {rec.stage}")
    print(f"  Weight:       {rec.current_weight:.4f}")
    print(f"  Boost:        {rec.domestication_boost:.4f}")
    print(f"  Posterior WR: {rec.posterior_wr:.1%}")
    print(f"  Profit factor: {rec.profit_factor:.2f}")
    print(f"  Total trades: {rec.total_trades}")
    print(f"  Total PnL:    ${rec.total_pnl:.2f}")

    # ----- Scenario 3: NEUTRAL signal enters tolerance -----
    print("\n  --- SCENARIO 3: NEUTRAL SIGNAL (enters tolerance) ---")
    print("  Signal: 'sunspot_correlation' on BTCUSD H1")
    print("  This signal is around 50% win rate -- harmless but useless\n")

    engine.register_new_signal("sunspot_correlation", "BTCUSD", "H1")
    _backdate_record("sunspot_correlation", "BTCUSD", "H1")

    for i in range(25):
        won = (i % 2 == 0)  # Exactly 50% WR
        pnl = np.random.uniform(0.50, 1.00) if won else -np.random.uniform(0.50, 1.00)
        engine.record_outcome("sunspot_correlation", "BTCUSD", "H1", won, pnl)

    rec = engine.get_signal_record("sunspot_correlation", "BTCUSD", "H1")
    print(f"  Stage:        {rec.stage}")
    print(f"  Weight:       {rec.current_weight:.4f}")
    print(f"  Posterior WR: {rec.posterior_wr:.1%}")
    print(f"  Total trades: {rec.total_trades}")

    # ----- Scenario 4: POPULATION VARIATION -----
    print("\n  --- SCENARIO 4: POPULATION VARIATION ---")
    print("  Same signal 'vwap_deviation' on different instruments")
    print("  XAUUSD: profitable | BTCUSD: toxic | EURUSD: neutral\n")

    populations = [
        ("XAUUSD", 7, 10),  # 7/10 = 70% deterministic WR
        ("BTCUSD", 2, 10),  # 2/10 = 20% deterministic WR
        ("EURUSD", 5, 10),  # 5/10 = 50% deterministic WR
    ]
    for instrument, wins_per_cycle, cycle_len in populations:
        engine.register_new_signal("vwap_deviation", instrument, "M5")
        _backdate_record("vwap_deviation", instrument, "M5")
        for i in range(40):
            won = (i % cycle_len) < wins_per_cycle
            pnl = np.random.uniform(1.00, 2.00) if won else -np.random.uniform(0.50, 1.00)
            engine.record_outcome("vwap_deviation", instrument, "M5", won, pnl)

        rec = engine.get_signal_record("vwap_deviation", instrument, "M5")
        print(f"  {instrument}: stage={rec.stage:16s} weight={rec.current_weight:.4f} "
              f"WR={rec.posterior_wr:.1%} trades={rec.total_trades}")

    # ----- Scenario 5: INTERACTION TRACKING -----
    print("\n  --- SCENARIO 5: INTERACTION TRACKING ---")
    print("  Testing signal pair co-occurrences\n")

    for i in range(15):
        won = np.random.random() < 0.80  # Good synergy when both fire
        engine.record_interaction(
            "order_flow_imbalance_v2", "vwap_deviation",
            "XAUUSD", "M5", won,
        )

    interactions = engine.get_interactions(instrument="XAUUSD")
    for ix in interactions:
        print(f"  {ix.signal_a} + {ix.signal_b}: "
              f"class={ix.classification} synergy={ix.synergy_score:+.4f} "
              f"co_WR={ix.combined_wr:.1%} n={ix.co_occurrence_count}")

    # ----- Scenario 6: BRIDGE INTEGRATION -----
    print("\n  --- SCENARIO 6: BRIDGE INTEGRATION ---")
    print("  Testing weighted confidence and gate results\n")

    base_conf = 0.65
    signals = ["order_flow_imbalance_v2", "lunar_phase_indicator"]
    modified = bridge.get_weighted_confidence(base_conf, signals, "XAUUSD", "M5")
    gate = bridge.get_gate_result(signals, "XAUUSD", "M5")

    print(f"  Base confidence:     {base_conf:.4f}")
    print(f"  Modified confidence: {modified:.4f}")
    print(f"  Gate pass:           {gate['gate_pass']}")
    print(f"  Combined weight:     {gate['combined_weight']:.4f}")
    print(f"  Any silenced:        {gate['any_silenced']}")
    print(f"  Any domesticated:    {gate['any_domesticated']}")
    print(f"  Signal weights:      {gate['signal_weights']}")

    # ----- Full Report -----
    print("\n  --- ENDOGENIZATION REPORT ---")
    report = engine.generate_endogenization_report()

    print(f"  Total signals tracked: {report['total_signals']}")
    print(f"  Domestication rate:    {report['domestication_rate']:.1%}")
    print(f"\n  Stage counts:")
    for stage, data in report["stage_counts"].items():
        print(f"    {stage:16s}: {data['count']} signals, "
              f"avg WR={data['avg_wr']:.1%}, avg weight={data['avg_weight']:.4f}")

    print(f"\n  Instrument breakdown:")
    for inst, stages in report["instrument_breakdown"].items():
        print(f"    {inst}:")
        for stage, data in stages.items():
            print(f"      {stage:16s}: {data['count']} signals, "
                  f"avg WR={data['avg_wr']:.1%}")

    if report["recent_transitions"]:
        print(f"\n  Recent stage transitions (last {len(report['recent_transitions'])}):")
        for t in report["recent_transitions"][:5]:
            print(f"    {t['from']:16s} -> {t['to']:16s} | "
                  f"WR={t['wr']:.1%} PF={t['pf']:.2f} weight={t['weight']:.4f} | "
                  f"{t['reason']}")

    # Write status file
    bridge.write_status_file()
    print(f"\n  Status file written to: {bridge.status_file}")

    # Cleanup
    try:
        os.remove(test_db)
    except OSError:
        pass

    # Remove temp status file
    try:
        status_path = str(Path(__file__).parent / "korv_status.json")
        os.remove(status_path)
    except OSError:
        pass

    print("\n" + "=" * 76)
    print("  KoRV Domestication Engine test complete.")
    print("  Every new signal must earn its place in the genome.")
    print("=" * 76)
