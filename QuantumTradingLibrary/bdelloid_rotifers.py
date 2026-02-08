"""
BDELLOID ROTIFERS ENGINE -- Horizontal Gene Transfer for Trading Strategies
============================================================================
Algorithm #6 in the Quantum Children biological algorithm series.

Manages cross-strategy component theft during drawdown-triggered desiccation
events, inspired by the horizontal gene transfer (HGT) mechanism of bdelloid
rotifers -- microscopic animals that have thrived for 80+ million years of
asexual reproduction by stealing DNA from bacteria, fungi, and other organisms.

When a strategy enters drawdown (desiccation), its parameters shatter.
During reassembly, proven components from high-performing donor strategies
are incorporated at TE-mediated integration sites. The reconstructed hybrid
strategy enters quarantine, and if it passes, the foreign genes persist.
Strategies accumulate foreign DNA with each crisis, becoming more robust.

Integration:
    - Reads CONFIDENCE_THRESHOLD from config_loader (no hardcoded trading values)
    - Stores all state in SQLite (bdelloid_rotifers.db)
    - Provides desiccation monitoring, foreign DNA scanning, quarantine evaluation
    - Tracks foreign gene lineage and contribution scores
    - Interfaces with VDJ antibodies as donor pool and TE engine for integration sites

Authors: DooDoo + Claude
Date:    2026-02-08
Version: BDELLOID-HGT-1.0
"""

import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import trading values from config_loader (NEVER hardcode)
try:
    from config_loader import CONFIDENCE_THRESHOLD
except ImportError:
    # Fallback for standalone testing only
    CONFIDENCE_THRESHOLD = 0.22

log = logging.getLogger(__name__)

VERSION = "BDELLOID-HGT-1.0"

# ============================================================
# CONSTANTS
# ============================================================

# Desiccation trigger thresholds
DESICCATION_DRAWDOWN_PCT = 0.10         # 10% drawdown from peak triggers desiccation
DESICCATION_LOSING_STREAK = 5           # 5 consecutive losses triggers desiccation
DESICCATION_WIN_RATE_FLOOR = 0.40       # WR below 40% over recent window triggers it
DESICCATION_EVAL_WINDOW = 20            # Look at last 20 trades for WR floor check

# Foreign DNA scanning (donor requirements)
DONOR_MIN_WIN_RATE = 0.58               # Donor must have posterior WR >= 58%
DONOR_MIN_TRADES = 15                   # Donor needs >= 15 trades to be credible
DONOR_MIN_PROFIT_FACTOR = 1.30          # Donor needs PF >= 1.30
DONOR_RECENCY_DAYS = 30                 # Only consider donors active in last 30 days

# HGT incorporation
HGT_MAX_COMPONENTS_PER_EVENT = 3        # Max foreign components absorbed per desiccation
HGT_COMPONENT_ACCEPT_PROB = 0.60        # 60% base acceptance probability
HGT_PARAM_BLEND_RATIO = 0.70           # 70% donor, 30% original (not full overwrite)
HGT_TE_INTEGRATION_BOOST = 0.15        # TE-active integration sites get +15% acceptance
HGT_CROSS_INSTRUMENT_PENALTY = 0.70    # 30% penalty for cross-instrument transfers

# Reassembly quarantine
REASSEMBLY_QUARANTINE_TRADES = 10       # Trades observed during quarantine
REASSEMBLY_VALIDATION_MIN_WR = 0.50     # Must achieve 50% WR to keep foreign DNA
REASSEMBLY_REVERT_ON_FAILURE = True     # Revert to pre-desiccation state on quarantine fail

# Foreign gene lifecycle
FOREIGN_GENE_MAX_AGE_DAYS = 90          # Re-evaluate foreign genes after 90 days
FOREIGN_GENE_MIN_CONTRIBUTION = 0.05    # Must contribute >= 5% improvement to persist

# Desiccation resistance (builds with each survived cycle)
RESISTANCE_BONUS_PER_SURVIVAL = 0.02    # +2% resistance per survived desiccation
RESISTANCE_MAX_BONUS = 0.20             # Capped at 20%
RESISTANCE_REDUCES_TRIGGER = True       # Resistant strategies need deeper drawdown

# Component shattering probabilities
SHATTER_BASE_PROB = 0.40                # Base 40% shatter chance per component
SHATTER_ENTRY_EXIT_BONUS = 0.10         # Entry/exit components shatter more easily
SHATTER_STALE_GENE_BONUS = 0.10         # Per stale foreign gene in component
SHATTER_MAX_PROB = 0.90                 # Maximum shatter probability

# TE integration families (biological: non-LTR retrotransposons mark integration sites)
TE_INTEGRATION_FAMILIES = [
    "L1_Neuronal",
    "Alu_Exonization",
    "HERV_Synapse",
    "Mariner_Tc1",
    "hobo",
]
TE_INTEGRATION_WEIGHT_THRESHOLD = 0.50

# Bayesian prior
PRIOR_ALPHA = 8
PRIOR_BETA = 8

# Component types available for HGT
COMPONENT_ENTRY = "entry_logic"
COMPONENT_EXIT = "exit_logic"
COMPONENT_REGIME = "regime_filter"
COMPONENT_TE_WEIGHTS = "te_weights"
COMPONENT_RISK = "risk_params"
COMPONENT_TIMING = "timing_params"

ALL_COMPONENT_TYPES = [
    COMPONENT_ENTRY,
    COMPONENT_EXIT,
    COMPONENT_REGIME,
    COMPONENT_TE_WEIGHTS,
    COMPONENT_RISK,
    COMPONENT_TIMING,
]

# Foreign gene statuses
GENE_STATUS_ACTIVE = "active"
GENE_STATUS_NEUTRAL = "neutral"
GENE_STATUS_REJECTED = "rejected"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ForeignGene:
    """A foreign gene incorporated from a donor strategy during HGT."""
    gene_id: str
    component_type: str
    donor_strategy_id: str
    donor_name: str
    donated_params: Dict[str, Any]
    incorporated_at: str
    desiccation_event: int
    te_integration_site: str
    contribution_score: float = 0.0
    status: str = GENE_STATUS_ACTIVE

    def to_dict(self) -> Dict:
        return {
            "gene_id": self.gene_id,
            "component_type": self.component_type,
            "donor_strategy_id": self.donor_strategy_id,
            "donor_name": self.donor_name,
            "donated_params": self.donated_params,
            "incorporated_at": self.incorporated_at,
            "desiccation_event": self.desiccation_event,
            "te_integration_site": self.te_integration_site,
            "contribution_score": self.contribution_score,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ForeignGene":
        return cls(
            gene_id=d["gene_id"],
            component_type=d["component_type"],
            donor_strategy_id=d["donor_strategy_id"],
            donor_name=d["donor_name"],
            donated_params=d.get("donated_params", {}),
            incorporated_at=d["incorporated_at"],
            desiccation_event=d.get("desiccation_event", 0),
            te_integration_site=d.get("te_integration_site", "L1_Neuronal"),
            contribution_score=d.get("contribution_score", 0.0),
            status=d.get("status", GENE_STATUS_ACTIVE),
        )


@dataclass
class StrategyOrganism:
    """
    A trading strategy viewed as a biological organism with a genome
    that can undergo desiccation, HGT, and reassembly.
    """
    strategy_id: str
    strategy_name: str
    instrument: str
    timeframe: str

    # Core DNA (strategy parameters)
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Performance tracking
    win_count: int = 0
    loss_count: int = 0
    total_trades: int = 0
    posterior_wr: float = 0.5
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    peak_pnl: float = 0.0
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Recent trade results (sliding window for recent_wr)
    recent_results: List[bool] = field(default_factory=list)

    # Desiccation state
    is_desiccated: bool = False
    desiccation_count: int = 0
    desiccation_resistance: float = 0.0
    last_desiccation: Optional[str] = None
    pre_desiccation_snapshot: Optional[Dict] = None

    # Foreign gene inventory
    foreign_genes: List[ForeignGene] = field(default_factory=list)

    # Quarantine state
    in_quarantine: bool = False
    quarantine_start: Optional[str] = None
    quarantine_trades: int = 0
    quarantine_wins: int = 0

    # Metadata
    created_at: str = ""
    last_updated: str = ""
    generation: int = 0

    def compute_id(self) -> str:
        raw = f"{self.strategy_name}|{self.instrument}|{self.timeframe}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @property
    def recent_wr(self) -> float:
        if not self.recent_results:
            return self.posterior_wr
        return sum(self.recent_results) / len(self.recent_results)

    @property
    def active_foreign_gene_count(self) -> int:
        return sum(1 for g in self.foreign_genes if g.status == GENE_STATUS_ACTIVE)

    @property
    def foreign_gene_pct(self) -> float:
        total = len(ALL_COMPONENT_TYPES)
        if total == 0:
            return 0.0
        return self.active_foreign_gene_count / total

    def summary(self) -> str:
        return (
            f"[{self.strategy_id[:8]}] {self.strategy_name} | "
            f"{self.instrument} {self.timeframe} | "
            f"WR={self.posterior_wr:.1%} PF={self.profit_factor:.2f} "
            f"trades={self.total_trades} | "
            f"DD={self.current_drawdown_pct:.1%} "
            f"desic={self.desiccation_count} "
            f"foreign={self.active_foreign_gene_count} "
            f"gen={self.generation}"
        )


# ============================================================
# BDELLOID HGT ENGINE
# ============================================================

class BdelloidHGTEngine:
    """
    Core engine implementing Horizontal Gene Transfer for trading strategies.

    This is the bdelloid rotifer equivalent: it monitors strategies for
    desiccation conditions, shatters their components, scans for foreign
    DNA from high-performing donors, incorporates viable foreign genes,
    reassembles the hybrid strategy, and manages quarantine validation.
    """

    def __init__(
        self,
        db_path: str = None,
        seed: int = None,
    ):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Database
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "bdelloid_rotifers.db")
        else:
            self.db_path = db_path
        self._init_db()

        # In-memory strategy cache
        self.strategies: Dict[str, StrategyOrganism] = {}
        self._load_strategies()

    # ----------------------------------------------------------
    # DATABASE
    # ----------------------------------------------------------

    def _init_db(self):
        """Initialize the bdelloid rotifers database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategies (
                        strategy_id TEXT PRIMARY KEY,
                        strategy_name TEXT NOT NULL,
                        instrument TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        components TEXT DEFAULT '{}',
                        win_count INTEGER DEFAULT 0,
                        loss_count INTEGER DEFAULT 0,
                        total_trades INTEGER DEFAULT 0,
                        posterior_wr REAL DEFAULT 0.5,
                        profit_factor REAL DEFAULT 0.0,
                        total_pnl REAL DEFAULT 0.0,
                        peak_pnl REAL DEFAULT 0.0,
                        current_drawdown_pct REAL DEFAULT 0.0,
                        consecutive_losses INTEGER DEFAULT 0,
                        avg_win REAL DEFAULT 0.0,
                        avg_loss REAL DEFAULT 0.0,
                        recent_results TEXT DEFAULT '[]',
                        is_desiccated INTEGER DEFAULT 0,
                        desiccation_count INTEGER DEFAULT 0,
                        desiccation_resistance REAL DEFAULT 0.0,
                        last_desiccation TEXT,
                        pre_desiccation_snapshot TEXT,
                        foreign_genes TEXT DEFAULT '[]',
                        in_quarantine INTEGER DEFAULT 0,
                        quarantine_start TEXT,
                        quarantine_trades INTEGER DEFAULT 0,
                        quarantine_wins INTEGER DEFAULT 0,
                        created_at TEXT,
                        last_updated TEXT,
                        generation INTEGER DEFAULT 0
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS desiccation_events (
                        event_id TEXT PRIMARY KEY,
                        strategy_id TEXT NOT NULL,
                        triggered_at TEXT NOT NULL,
                        trigger_reason TEXT NOT NULL,
                        drawdown_at_trigger REAL DEFAULT 0.0,
                        components_shattered TEXT DEFAULT '[]',
                        donor_strategies_scanned INTEGER DEFAULT 0,
                        foreign_genes_incorporated INTEGER DEFAULT 0,
                        reassembly_completed_at TEXT,
                        quarantine_passed INTEGER,
                        net_improvement REAL DEFAULT 0.0
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS donor_scans (
                        scan_id TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL,
                        donor_strategy_id TEXT NOT NULL,
                        donor_name TEXT NOT NULL,
                        component_type TEXT NOT NULL,
                        donor_fitness REAL DEFAULT 0.0,
                        donor_pf REAL DEFAULT 0.0,
                        te_compatibility REAL DEFAULT 0.0,
                        accepted INTEGER DEFAULT 0,
                        acceptance_reason TEXT DEFAULT ''
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS foreign_gene_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_id TEXT NOT NULL,
                        gene_id TEXT NOT NULL,
                        component_type TEXT NOT NULL,
                        donor_strategy_id TEXT NOT NULL,
                        donor_name TEXT NOT NULL,
                        te_integration_site TEXT,
                        action TEXT NOT NULL,
                        details TEXT DEFAULT ''
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS hgt_statistics (
                        stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_strategies INTEGER DEFAULT 0,
                        total_desiccations INTEGER DEFAULT 0,
                        total_foreign_genes INTEGER DEFAULT 0,
                        active_foreign_genes INTEGER DEFAULT 0,
                        quarantine_survival_rate REAL DEFAULT 0.0,
                        avg_foreign_gene_pct REAL DEFAULT 0.0,
                        avg_desiccation_resistance REAL DEFAULT 0.0,
                        report_json TEXT DEFAULT '{}'
                    )
                """)
                conn.commit()
        except Exception as e:
            log.warning("Bdelloid DB init failed: %s", e)

    def _load_strategies(self):
        """Load all strategies from the database into memory."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM strategies")
                for row in cursor.fetchall():
                    s = self._row_to_strategy(row)
                    self.strategies[s.strategy_id] = s
                log.info(
                    "[HGT] Loaded %d strategies from database", len(self.strategies)
                )
        except Exception as e:
            log.warning("Failed to load strategies: %s", e)

    def _row_to_strategy(self, row) -> StrategyOrganism:
        """Convert a database row to a StrategyOrganism."""
        foreign_genes_raw = json.loads(row["foreign_genes"] or "[]")
        foreign_genes = [ForeignGene.from_dict(g) for g in foreign_genes_raw]

        snapshot_raw = row["pre_desiccation_snapshot"]
        snapshot = json.loads(snapshot_raw) if snapshot_raw else None

        recent_raw = row["recent_results"]
        recent = json.loads(recent_raw) if recent_raw else []

        return StrategyOrganism(
            strategy_id=row["strategy_id"],
            strategy_name=row["strategy_name"],
            instrument=row["instrument"],
            timeframe=row["timeframe"],
            components=json.loads(row["components"] or "{}"),
            win_count=row["win_count"] or 0,
            loss_count=row["loss_count"] or 0,
            total_trades=row["total_trades"] or 0,
            posterior_wr=row["posterior_wr"] or 0.5,
            profit_factor=row["profit_factor"] or 0.0,
            total_pnl=row["total_pnl"] or 0.0,
            peak_pnl=row["peak_pnl"] or 0.0,
            current_drawdown_pct=row["current_drawdown_pct"] or 0.0,
            consecutive_losses=row["consecutive_losses"] or 0,
            avg_win=row["avg_win"] or 0.0,
            avg_loss=row["avg_loss"] or 0.0,
            recent_results=recent,
            is_desiccated=bool(row["is_desiccated"]),
            desiccation_count=row["desiccation_count"] or 0,
            desiccation_resistance=row["desiccation_resistance"] or 0.0,
            last_desiccation=row["last_desiccation"],
            pre_desiccation_snapshot=snapshot,
            foreign_genes=foreign_genes,
            in_quarantine=bool(row["in_quarantine"]),
            quarantine_start=row["quarantine_start"],
            quarantine_trades=row["quarantine_trades"] or 0,
            quarantine_wins=row["quarantine_wins"] or 0,
            created_at=row["created_at"] or "",
            last_updated=row["last_updated"] or "",
            generation=row["generation"] or 0,
        )

    def _save_strategy(self, s: StrategyOrganism):
        """Save a strategy to the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                foreign_genes_json = json.dumps(
                    [g.to_dict() for g in s.foreign_genes], sort_keys=True
                )
                snapshot_json = (
                    json.dumps(s.pre_desiccation_snapshot, sort_keys=True)
                    if s.pre_desiccation_snapshot
                    else None
                )
                recent_json = json.dumps(s.recent_results[-DESICCATION_EVAL_WINDOW:])
                conn.execute(
                    """
                    INSERT OR REPLACE INTO strategies
                    (strategy_id, strategy_name, instrument, timeframe,
                     components, win_count, loss_count, total_trades,
                     posterior_wr, profit_factor, total_pnl, peak_pnl,
                     current_drawdown_pct, consecutive_losses, avg_win, avg_loss,
                     recent_results, is_desiccated, desiccation_count,
                     desiccation_resistance, last_desiccation,
                     pre_desiccation_snapshot, foreign_genes,
                     in_quarantine, quarantine_start, quarantine_trades,
                     quarantine_wins, created_at, last_updated, generation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        s.strategy_id,
                        s.strategy_name,
                        s.instrument,
                        s.timeframe,
                        json.dumps(s.components, sort_keys=True),
                        s.win_count,
                        s.loss_count,
                        s.total_trades,
                        s.posterior_wr,
                        s.profit_factor,
                        s.total_pnl,
                        s.peak_pnl,
                        s.current_drawdown_pct,
                        s.consecutive_losses,
                        s.avg_win,
                        s.avg_loss,
                        recent_json,
                        1 if s.is_desiccated else 0,
                        s.desiccation_count,
                        s.desiccation_resistance,
                        s.last_desiccation,
                        snapshot_json,
                        foreign_genes_json,
                        1 if s.in_quarantine else 0,
                        s.quarantine_start,
                        s.quarantine_trades,
                        s.quarantine_wins,
                        s.created_at,
                        s.last_updated,
                        s.generation,
                    ),
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to save strategy %s: %s", s.strategy_id, e)

    def _save_desiccation_event(self, event: Dict):
        """Save a desiccation event to the database."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO desiccation_events
                    (event_id, strategy_id, triggered_at, trigger_reason,
                     drawdown_at_trigger, components_shattered,
                     donor_strategies_scanned, foreign_genes_incorporated,
                     reassembly_completed_at, quarantine_passed, net_improvement)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["event_id"],
                        event["strategy_id"],
                        event["triggered_at"],
                        event["trigger_reason"],
                        event.get("drawdown_at_trigger", 0.0),
                        json.dumps(event.get("components_shattered", [])),
                        event.get("donor_strategies_scanned", 0),
                        event.get("foreign_genes_incorporated", 0),
                        event.get("reassembly_completed_at"),
                        event.get("quarantine_passed"),
                        event.get("net_improvement", 0.0),
                    ),
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to save desiccation event: %s", e)

    def _save_donor_scan(self, scan: Dict):
        """Save a donor scan record."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO donor_scans
                    (scan_id, event_id, donor_strategy_id, donor_name,
                     component_type, donor_fitness, donor_pf,
                     te_compatibility, accepted, acceptance_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scan["scan_id"],
                        scan["event_id"],
                        scan["donor_strategy_id"],
                        scan["donor_name"],
                        scan["component_type"],
                        scan.get("donor_fitness", 0.0),
                        scan.get("donor_pf", 0.0),
                        scan.get("te_compatibility", 0.0),
                        1 if scan.get("accepted") else 0,
                        scan.get("acceptance_reason", ""),
                    ),
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to save donor scan: %s", e)

    def _log_foreign_gene_action(
        self, strategy_id: str, gene: ForeignGene, action: str, details: str = ""
    ):
        """Log a foreign gene action to the audit log."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    INSERT INTO foreign_gene_log
                    (timestamp, strategy_id, gene_id, component_type,
                     donor_strategy_id, donor_name, te_integration_site,
                     action, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        strategy_id,
                        gene.gene_id,
                        gene.component_type,
                        gene.donor_strategy_id,
                        gene.donor_name,
                        gene.te_integration_site,
                        action,
                        details,
                    ),
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to log foreign gene action: %s", e)

    # ----------------------------------------------------------
    # STRATEGY REGISTRATION
    # ----------------------------------------------------------

    def register_strategy(
        self,
        strategy_name: str,
        instrument: str,
        timeframe: str,
        components: Dict[str, Dict[str, Any]] = None,
    ) -> StrategyOrganism:
        """
        Register a new strategy organism in the HGT ecosystem.

        Args:
            strategy_name: Human-readable strategy name
            instrument: Trading instrument (e.g. "XAUUSD")
            timeframe: Timeframe (e.g. "M5")
            components: Initial strategy components (entry, exit, regime, etc.)

        Returns:
            The newly created StrategyOrganism.
        """
        s = StrategyOrganism(
            strategy_id="",
            strategy_name=strategy_name,
            instrument=instrument,
            timeframe=timeframe,
            components=components or {},
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )
        s.strategy_id = s.compute_id()

        # Initialize default components if not provided
        for ct in ALL_COMPONENT_TYPES:
            if ct not in s.components:
                s.components[ct] = {}

        # Check if already registered
        if s.strategy_id in self.strategies:
            log.info(
                "[HGT] Strategy %s already registered, returning existing",
                s.strategy_name,
            )
            return self.strategies[s.strategy_id]

        self.strategies[s.strategy_id] = s
        self._save_strategy(s)
        log.info(
            "[HGT] Registered strategy: %s (%s %s) id=%s",
            strategy_name,
            instrument,
            timeframe,
            s.strategy_id[:8],
        )
        return s

    # ----------------------------------------------------------
    # PHASE 1: DESICCATION DETECTION
    # ----------------------------------------------------------

    def check_desiccation(self, strategy_id: str) -> bool:
        """
        Check if a strategy should undergo desiccation (drawdown-triggered
        parameter shattering).

        Returns True if desiccation was triggered.
        """
        s = self.strategies.get(strategy_id)
        if s is None:
            return False

        if s.is_desiccated or s.in_quarantine:
            return False

        # Calculate effective drawdown trigger (resistance raises the bar)
        effective_trigger = DESICCATION_DRAWDOWN_PCT
        if RESISTANCE_REDUCES_TRIGGER:
            effective_trigger += s.desiccation_resistance

        # TRIGGER 1: Drawdown from peak
        if s.current_drawdown_pct >= effective_trigger:
            self._trigger_desiccation(s, "drawdown")
            return True

        # TRIGGER 2: Consecutive losing streak
        if s.consecutive_losses >= DESICCATION_LOSING_STREAK:
            self._trigger_desiccation(s, "losing_streak")
            return True

        # TRIGGER 3: Recent win rate below floor
        if (
            s.total_trades >= DESICCATION_EVAL_WINDOW
            and s.recent_wr < DESICCATION_WIN_RATE_FLOOR
        ):
            self._trigger_desiccation(s, "wr_floor")
            return True

        return False

    # ----------------------------------------------------------
    # PHASE 2: DNA SHATTERING
    # ----------------------------------------------------------

    def _trigger_desiccation(self, s: StrategyOrganism, reason: str):
        """
        Trigger desiccation: snapshot state, shatter components, scan donors,
        incorporate foreign DNA, and reassemble.
        """
        # Snapshot current state for potential rollback
        s.pre_desiccation_snapshot = deepcopy(s.components)
        s.is_desiccated = True
        s.desiccation_count += 1
        s.last_desiccation = datetime.now().isoformat()
        s.generation += 1

        # Determine which components shatter
        shattered = self._determine_shattered_components(s)

        # Create desiccation event
        event_raw = f"{s.strategy_id}|{datetime.now().isoformat()}|{s.desiccation_count}"
        event_id = hashlib.md5(event_raw.encode()).hexdigest()[:16]
        event = {
            "event_id": event_id,
            "strategy_id": s.strategy_id,
            "triggered_at": datetime.now().isoformat(),
            "trigger_reason": reason,
            "drawdown_at_trigger": s.current_drawdown_pct,
            "components_shattered": shattered,
            "donor_strategies_scanned": 0,
            "foreign_genes_incorporated": 0,
            "reassembly_completed_at": None,
            "quarantine_passed": None,
            "net_improvement": 0.0,
        }
        self._save_desiccation_event(event)

        log.info(
            "[HGT] DESICCATION triggered: %s | reason=%s DD=%.1f%% "
            "shattered=%s cycle=%d",
            s.strategy_name,
            reason,
            s.current_drawdown_pct * 100,
            shattered,
            s.desiccation_count,
        )

        # Scan for foreign DNA and incorporate
        incorporated = self._scan_and_incorporate(s, event, shattered)

        # Reassemble
        self._reassemble(s, event, incorporated)

    def _determine_shattered_components(
        self, s: StrategyOrganism
    ) -> List[str]:
        """
        Determine which components shatter during desiccation.
        Components with worse performance or stale foreign genes shatter first.
        """
        shattered = []

        for ct in ALL_COMPONENT_TYPES:
            prob = SHATTER_BASE_PROB

            # Entry/exit components are more market-sensitive
            if ct in (COMPONENT_ENTRY, COMPONENT_EXIT):
                prob += SHATTER_ENTRY_EXIT_BONUS

            # Stale foreign genes increase shatter probability
            stale_count = sum(
                1
                for g in s.foreign_genes
                if g.component_type == ct and g.status == GENE_STATUS_NEUTRAL
            )
            prob += min(
                SHATTER_MAX_PROB - SHATTER_BASE_PROB,
                stale_count * SHATTER_STALE_GENE_BONUS,
            )

            prob = min(SHATTER_MAX_PROB, prob)

            if self.rng.random() < prob:
                shattered.append(ct)

        # At least one component must shatter
        if not shattered:
            # Force-shatter the component with most stale foreign genes, or entry
            stale_counts = {}
            for ct in ALL_COMPONENT_TYPES:
                stale_counts[ct] = sum(
                    1
                    for g in s.foreign_genes
                    if g.component_type == ct and g.status == GENE_STATUS_NEUTRAL
                )
            weakest = max(stale_counts, key=stale_counts.get)
            if stale_counts[weakest] == 0:
                weakest = COMPONENT_ENTRY  # Default to entry logic
            shattered.append(weakest)

        return shattered

    # ----------------------------------------------------------
    # PHASE 3: FOREIGN DNA SCANNING
    # ----------------------------------------------------------

    def _scan_and_incorporate(
        self,
        s: StrategyOrganism,
        event: Dict,
        shattered: List[str],
    ) -> List[ForeignGene]:
        """
        Scan all other strategies for viable donor components and incorporate
        foreign DNA at shattered positions.
        """
        # Find viable donors
        now = datetime.now()
        cutoff = (now - timedelta(days=DONOR_RECENCY_DAYS)).isoformat()

        donors = [
            d
            for d in self.strategies.values()
            if d.strategy_id != s.strategy_id
            and d.total_trades >= DONOR_MIN_TRADES
            and d.posterior_wr >= DONOR_MIN_WIN_RATE
            and d.profit_factor >= DONOR_MIN_PROFIT_FACTOR
            and d.last_updated >= cutoff
        ]

        # Sort donors by fitness (best first)
        donors.sort(key=lambda d: d.posterior_wr, reverse=True)

        if not donors:
            log.info(
                "[HGT] No viable donors found for %s. Reassembling with original DNA.",
                s.strategy_name,
            )
            return []

        incorporated = []
        scan_count = 0

        for ct in shattered:
            if len(incorporated) >= HGT_MAX_COMPONENTS_PER_EVENT:
                break

            for donor in donors:
                scan_count += 1

                # Check TE compatibility at integration site
                te_compat = self._compute_te_compatibility(s, donor, ct)

                # Calculate acceptance probability
                accept_prob = HGT_COMPONENT_ACCEPT_PROB

                # TE-mediated boost
                if te_compat > TE_INTEGRATION_WEIGHT_THRESHOLD:
                    accept_prob += HGT_TE_INTEGRATION_BOOST

                # Cross-instrument penalty
                if donor.instrument != s.instrument:
                    accept_prob *= HGT_CROSS_INSTRUMENT_PENALTY

                # Donor fitness bonus
                fitness_bonus = (donor.posterior_wr - DONOR_MIN_WIN_RATE) * 0.50
                accept_prob += fitness_bonus

                accept_prob = max(0.0, min(1.0, accept_prob))

                # Record the scan
                scan_raw = f"{event['event_id']}|{donor.strategy_id}|{ct}"
                scan_id = hashlib.md5(scan_raw.encode()).hexdigest()[:16]
                scan = {
                    "scan_id": scan_id,
                    "event_id": event["event_id"],
                    "donor_strategy_id": donor.strategy_id,
                    "donor_name": donor.strategy_name,
                    "component_type": ct,
                    "donor_fitness": donor.posterior_wr,
                    "donor_pf": donor.profit_factor,
                    "te_compatibility": te_compat,
                    "accepted": False,
                    "acceptance_reason": "",
                }

                if self.rng.random() < accept_prob:
                    # ACCEPT: incorporate foreign DNA
                    gene = self._incorporate_foreign_gene(s, donor, ct)
                    incorporated.append(gene)
                    scan["accepted"] = True
                    scan["acceptance_reason"] = (
                        f"prob={accept_prob:.2f} te_compat={te_compat:.2f}"
                    )
                    self._save_donor_scan(scan)
                    break  # Move to next shattered component
                else:
                    scan["acceptance_reason"] = (
                        f"prob={accept_prob:.2f} random_rejected"
                    )
                    self._save_donor_scan(scan)

        # Update event
        event["donor_strategies_scanned"] = scan_count
        event["foreign_genes_incorporated"] = len(incorporated)
        self._save_desiccation_event(event)

        log.info(
            "[HGT] Scanned %d donors, incorporated %d foreign genes for %s",
            scan_count,
            len(incorporated),
            s.strategy_name,
        )

        return incorporated

    # ----------------------------------------------------------
    # PHASE 4: HGT INCORPORATION
    # ----------------------------------------------------------

    def _incorporate_foreign_gene(
        self,
        s: StrategyOrganism,
        donor: StrategyOrganism,
        component_type: str,
    ) -> ForeignGene:
        """
        Incorporate a foreign gene from a donor strategy into the recipient.
        Uses parameter blending (70/30 donor/original) rather than full overwrite.
        """
        donor_params = donor.components.get(component_type, {})
        original_params = s.components.get(component_type, {})

        # Blend parameters
        blended = {}
        all_keys = set(list(donor_params.keys()) + list(original_params.keys()))

        for key in all_keys:
            d_val = donor_params.get(key)
            o_val = original_params.get(key)

            if d_val is not None and o_val is not None:
                if isinstance(d_val, (int, float)) and isinstance(o_val, (int, float)):
                    blended_val = (
                        HGT_PARAM_BLEND_RATIO * d_val
                        + (1.0 - HGT_PARAM_BLEND_RATIO) * o_val
                    )
                    blended[key] = (
                        int(round(blended_val)) if isinstance(d_val, int) and isinstance(o_val, int) else blended_val
                    )
                else:
                    # Non-numeric: take donor version
                    blended[key] = d_val
            elif d_val is not None:
                blended[key] = d_val
            else:
                blended[key] = o_val

        # Select integration TE
        integration_te = self._select_integration_te(s)

        # Create foreign gene record
        gene_raw = f"{donor.strategy_id}|{component_type}|{datetime.now().isoformat()}"
        gene_id = hashlib.md5(gene_raw.encode()).hexdigest()[:16]

        gene = ForeignGene(
            gene_id=gene_id,
            component_type=component_type,
            donor_strategy_id=donor.strategy_id,
            donor_name=donor.strategy_name,
            donated_params=donor_params,
            incorporated_at=datetime.now().isoformat(),
            desiccation_event=s.desiccation_count,
            te_integration_site=integration_te,
            contribution_score=0.0,
            status=GENE_STATUS_ACTIVE,
        )

        # Apply blended params
        s.components[component_type] = blended
        s.foreign_genes.append(gene)

        # Log the action
        self._log_foreign_gene_action(
            s.strategy_id,
            gene,
            "incorporated",
            f"blend={HGT_PARAM_BLEND_RATIO:.0%}donor/{1-HGT_PARAM_BLEND_RATIO:.0%}orig "
            f"te={integration_te} donor_wr={donor.posterior_wr:.1%}",
        )

        log.info(
            "[HGT] Foreign gene incorporated: %s from %s via TE:%s gene=%s",
            component_type,
            donor.strategy_name,
            integration_te,
            gene_id[:8],
        )

        return gene

    def _compute_te_compatibility(
        self,
        recipient: StrategyOrganism,
        donor: StrategyOrganism,
        component_type: str,
    ) -> float:
        """
        Compute TE compatibility between recipient and donor at the
        integration site for a given component type.

        Higher compatibility = more active TEs at homologous positions =
        better chance of successful foreign DNA integration.
        """
        r_te = recipient.components.get(COMPONENT_TE_WEIGHTS, {})
        d_te = donor.components.get(COMPONENT_TE_WEIGHTS, {})

        if not r_te and not d_te:
            return 0.5  # No TE data: assume neutral compatibility

        compatibility = 0.0
        n_checked = 0

        for te_family in TE_INTEGRATION_FAMILIES:
            r_weight = r_te.get(te_family, 0.0)
            d_weight = d_te.get(te_family, 0.0)

            if isinstance(r_weight, (int, float)) and isinstance(d_weight, (int, float)):
                overlap = min(float(r_weight), float(d_weight))
                compatibility += overlap
                n_checked += 1

        if n_checked > 0:
            compatibility /= n_checked

        return compatibility

    def _select_integration_te(self, s: StrategyOrganism) -> str:
        """
        Select which TE family facilitates integration.
        Prefers TEs with highest activation weight.
        """
        te_weights = s.components.get(COMPONENT_TE_WEIGHTS, {})

        best_te = "L1_Neuronal"  # Default
        best_weight = -1.0

        for te_family in TE_INTEGRATION_FAMILIES:
            w = te_weights.get(te_family, 0.0)
            if isinstance(w, (int, float)) and w > best_weight:
                best_weight = w
                best_te = te_family

        return best_te

    # ----------------------------------------------------------
    # PHASE 5: REASSEMBLY
    # ----------------------------------------------------------

    def _reassemble(
        self,
        s: StrategyOrganism,
        event: Dict,
        incorporated: List[ForeignGene],
    ):
        """
        Reassemble the strategy with hybrid DNA and enter quarantine.
        """
        s.is_desiccated = False
        s.in_quarantine = True
        s.quarantine_start = datetime.now().isoformat()
        s.quarantine_trades = 0
        s.quarantine_wins = 0

        # Build desiccation resistance
        s.desiccation_resistance = min(
            RESISTANCE_MAX_BONUS,
            s.desiccation_resistance + RESISTANCE_BONUS_PER_SURVIVAL,
        )

        # Update event
        event["reassembly_completed_at"] = datetime.now().isoformat()
        self._save_desiccation_event(event)

        self._save_strategy(s)

        log.info(
            "[HGT] Strategy %s reassembled: %d foreign genes, gen=%d, "
            "resistance=%.1f%%",
            s.strategy_name,
            len(incorporated),
            s.generation,
            s.desiccation_resistance * 100,
        )

    # ----------------------------------------------------------
    # PHASE 6: QUARANTINE EVALUATION
    # ----------------------------------------------------------

    def record_quarantine_outcome(
        self, strategy_id: str, won: bool, pnl: float = 0.0
    ) -> Optional[Dict]:
        """
        Record a trade outcome during quarantine. Returns a result dict
        if quarantine evaluation is complete, None otherwise.
        """
        s = self.strategies.get(strategy_id)
        if s is None or not s.in_quarantine:
            return None

        s.quarantine_trades += 1
        if won:
            s.quarantine_wins += 1

        result = None

        # Check if quarantine period is complete
        if s.quarantine_trades >= REASSEMBLY_QUARANTINE_TRADES:
            q_wr = s.quarantine_wins / s.quarantine_trades

            if q_wr >= REASSEMBLY_VALIDATION_MIN_WR:
                # QUARANTINE PASSED
                s.in_quarantine = False
                s.consecutive_losses = 0

                # Score foreign gene contributions
                self._score_foreign_gene_contributions(s)

                log.info(
                    "[HGT] Quarantine PASSED: %s WR=%.1f%% over %d trades",
                    s.strategy_name,
                    q_wr * 100,
                    s.quarantine_trades,
                )

                # Update desiccation event
                self._update_latest_event_quarantine(s, True)

                result = {
                    "passed": True,
                    "quarantine_wr": q_wr,
                    "trades": s.quarantine_trades,
                    "foreign_genes_kept": s.active_foreign_gene_count,
                }

            else:
                # QUARANTINE FAILED
                if (
                    REASSEMBLY_REVERT_ON_FAILURE
                    and s.pre_desiccation_snapshot is not None
                ):
                    s.components = s.pre_desiccation_snapshot
                    # Mark foreign genes from this cycle as rejected
                    for g in s.foreign_genes:
                        if g.desiccation_event == s.desiccation_count:
                            g.status = GENE_STATUS_REJECTED
                            self._log_foreign_gene_action(
                                s.strategy_id,
                                g,
                                "rejected",
                                f"quarantine_failed wr={q_wr:.1%}",
                            )

                    log.info(
                        "[HGT] Quarantine FAILED: %s WR=%.1f%%, REVERTING",
                        s.strategy_name,
                        q_wr * 100,
                    )
                else:
                    log.info(
                        "[HGT] Quarantine FAILED: %s WR=%.1f%%, no revert",
                        s.strategy_name,
                        q_wr * 100,
                    )

                s.in_quarantine = False
                self._update_latest_event_quarantine(s, False)

                result = {
                    "passed": False,
                    "quarantine_wr": q_wr,
                    "trades": s.quarantine_trades,
                    "reverted": REASSEMBLY_REVERT_ON_FAILURE,
                }

        self._save_strategy(s)
        return result

    def _update_latest_event_quarantine(self, s: StrategyOrganism, passed: bool):
        """Update the most recent desiccation event with quarantine result."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                # SQLite does not support ORDER BY in UPDATE, so use subquery
                conn.execute(
                    """
                    UPDATE desiccation_events
                    SET quarantine_passed = ?
                    WHERE event_id = (
                        SELECT event_id FROM desiccation_events
                        WHERE strategy_id = ?
                        ORDER BY triggered_at DESC
                        LIMIT 1
                    )
                    """,
                    (1 if passed else 0, s.strategy_id),
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to update quarantine status: %s", e)

    def _score_foreign_gene_contributions(self, s: StrategyOrganism):
        """
        Score how much each foreign gene from the latest desiccation cycle
        contributed to the quarantine performance improvement.
        """
        # Estimate pre-desiccation WR from recent results before desiccation
        pre_wr = 0.5
        if s.recent_results:
            # Use the results before desiccation as baseline
            pre_results = s.recent_results[:-s.quarantine_trades] if len(s.recent_results) > s.quarantine_trades else s.recent_results
            if pre_results:
                pre_wr = sum(pre_results) / len(pre_results)

        post_wr = s.quarantine_wins / max(1, s.quarantine_trades)
        improvement = post_wr - pre_wr

        # Distribute improvement across active foreign genes from this cycle
        cycle_genes = [
            g
            for g in s.foreign_genes
            if g.desiccation_event == s.desiccation_count
            and g.status == GENE_STATUS_ACTIVE
        ]

        if cycle_genes:
            per_gene = improvement / len(cycle_genes)
            for g in cycle_genes:
                g.contribution_score = per_gene
                self._log_foreign_gene_action(
                    s.strategy_id,
                    g,
                    "scored",
                    f"contribution={per_gene:+.3f} improvement={improvement:+.3f}",
                )

    # ----------------------------------------------------------
    # PHASE 7: FOREIGN GENE MAINTENANCE
    # ----------------------------------------------------------

    def evaluate_foreign_genes(self, strategy_id: str):
        """
        Evaluate all active foreign genes in a strategy.
        Mark stale or non-contributing genes as neutral.
        """
        s = self.strategies.get(strategy_id)
        if s is None:
            return

        now = datetime.now()
        changed = False

        for g in s.foreign_genes:
            if g.status != GENE_STATUS_ACTIVE:
                continue

            # Check age
            try:
                inc_time = datetime.fromisoformat(g.incorporated_at)
                days_since = (now - inc_time).days
            except (ValueError, TypeError):
                days_since = 0

            if days_since > FOREIGN_GENE_MAX_AGE_DAYS:
                if g.contribution_score < FOREIGN_GENE_MIN_CONTRIBUTION:
                    g.status = GENE_STATUS_NEUTRAL
                    changed = True
                    self._log_foreign_gene_action(
                        s.strategy_id,
                        g,
                        "marked_neutral",
                        f"age={days_since}d contribution={g.contribution_score:.3f}",
                    )
                    log.info(
                        "[HGT] Foreign gene %s from %s marked NEUTRAL: "
                        "age=%dd contribution=%.3f",
                        g.gene_id[:8],
                        g.donor_name,
                        days_since,
                        g.contribution_score,
                    )

        if changed:
            self._save_strategy(s)

    def evaluate_all_foreign_genes(self):
        """Evaluate foreign genes across all strategies."""
        for sid in list(self.strategies.keys()):
            self.evaluate_foreign_genes(sid)

    # ----------------------------------------------------------
    # PHASE 8: TRADE OUTCOME RECORDING (main entry point)
    # ----------------------------------------------------------

    def record_trade_outcome(
        self, strategy_id: str, won: bool, pnl: float = 0.0
    ) -> Dict:
        """
        Record a trade outcome for a strategy. This is the main entry point
        that drives the entire HGT lifecycle.

        Returns a status dict describing what happened.
        """
        s = self.strategies.get(strategy_id)
        if s is None:
            return {"error": f"Strategy {strategy_id} not found"}

        now = datetime.now().isoformat()

        # Update core metrics
        if won:
            s.win_count += 1
            if pnl > 0:
                # Running average for avg_win
                s.avg_win = (
                    (s.avg_win * (s.win_count - 1) + pnl) / s.win_count
                    if s.win_count > 1
                    else pnl
                )
        else:
            s.loss_count += 1
            if pnl < 0:
                s.avg_loss = (
                    (s.avg_loss * (s.loss_count - 1) + abs(pnl)) / s.loss_count
                    if s.loss_count > 1
                    else abs(pnl)
                )

        s.total_trades = s.win_count + s.loss_count
        s.total_pnl += pnl
        s.last_updated = now

        # Consecutive losses
        if won:
            s.consecutive_losses = 0
        else:
            s.consecutive_losses += 1

        # Bayesian posterior WR
        s.posterior_wr = (PRIOR_ALPHA + s.win_count) / (
            PRIOR_ALPHA + PRIOR_BETA + s.total_trades
        )

        # Peak PnL and drawdown
        if s.total_pnl > s.peak_pnl:
            s.peak_pnl = s.total_pnl
        if s.peak_pnl > 0:
            s.current_drawdown_pct = (s.peak_pnl - s.total_pnl) / s.peak_pnl
        else:
            s.current_drawdown_pct = 0.0

        # Recent results (sliding window)
        s.recent_results.append(won)
        if len(s.recent_results) > DESICCATION_EVAL_WINDOW:
            s.recent_results = s.recent_results[-DESICCATION_EVAL_WINDOW:]

        # Profit factor
        if s.avg_loss > 0:
            s.profit_factor = s.avg_win / s.avg_loss
        else:
            s.profit_factor = 99.0 if s.avg_win > 0 else 0.0

        self._save_strategy(s)

        result = {
            "strategy_id": strategy_id,
            "won": won,
            "pnl": pnl,
            "posterior_wr": s.posterior_wr,
            "drawdown": s.current_drawdown_pct,
            "consecutive_losses": s.consecutive_losses,
            "desiccated": False,
            "quarantine_result": None,
        }

        # If in quarantine, feed to quarantine tracker
        if s.in_quarantine:
            q_result = self.record_quarantine_outcome(strategy_id, won, pnl)
            result["quarantine_result"] = q_result
            return result

        # Check for desiccation trigger
        if self.check_desiccation(strategy_id):
            result["desiccated"] = True

        return result

    # ----------------------------------------------------------
    # PHASE 9: HGT STATISTICS & REPORTING
    # ----------------------------------------------------------

    def generate_hgt_report(self) -> Dict:
        """
        Generate a comprehensive HGT report across all strategies.
        """
        strategies = list(self.strategies.values())

        if not strategies:
            return {
                "timestamp": datetime.now().isoformat(),
                "total_strategies": 0,
                "total_desiccations": 0,
                "total_foreign_genes": 0,
                "active_foreign_genes": 0,
                "quarantine_survival_rate": 0.0,
                "avg_foreign_gene_pct": 0.0,
                "top_donors": [],
                "component_distribution": {},
                "avg_desiccation_resistance": 0.0,
            }

        total_desiccations = sum(s.desiccation_count for s in strategies)
        total_foreign = sum(len(s.foreign_genes) for s in strategies)
        active_foreign = sum(s.active_foreign_gene_count for s in strategies)

        # Donor frequency map
        donor_freq: Dict[str, int] = {}
        for s in strategies:
            for g in s.foreign_genes:
                if g.status == GENE_STATUS_ACTIVE:
                    donor_freq[g.donor_name] = donor_freq.get(g.donor_name, 0) + 1

        # Component type distribution
        comp_dist: Dict[str, int] = {}
        for s in strategies:
            for g in s.foreign_genes:
                if g.status == GENE_STATUS_ACTIVE:
                    comp_dist[g.component_type] = (
                        comp_dist.get(g.component_type, 0) + 1
                    )

        # Quarantine survival rate from DB
        survival_rate = 0.0
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM desiccation_events WHERE quarantine_passed IS NOT NULL"
                )
                total_evaluated = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT COUNT(*) FROM desiccation_events WHERE quarantine_passed = 1"
                )
                total_passed = cursor.fetchone()[0]
                if total_evaluated > 0:
                    survival_rate = total_passed / total_evaluated
        except Exception:
            pass

        # Foreign gene percentages
        foreign_pcts = [s.foreign_gene_pct for s in strategies]
        avg_pct = float(np.mean(foreign_pcts)) if foreign_pcts else 0.0

        # Average resistance
        resistances = [s.desiccation_resistance for s in strategies]
        avg_resistance = float(np.mean(resistances)) if resistances else 0.0

        # Sort donors
        top_donors = sorted(donor_freq.items(), key=lambda x: -x[1])[:10]

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_strategies": len(strategies),
            "total_desiccations": total_desiccations,
            "total_foreign_genes": total_foreign,
            "active_foreign_genes": active_foreign,
            "quarantine_survival_rate": survival_rate,
            "avg_foreign_gene_pct": avg_pct,
            "top_donors": top_donors,
            "component_distribution": comp_dist,
            "avg_desiccation_resistance": avg_resistance,
            "strategies": [
                {
                    "name": s.strategy_name,
                    "id": s.strategy_id[:8],
                    "instrument": s.instrument,
                    "wr": s.posterior_wr,
                    "pf": s.profit_factor,
                    "trades": s.total_trades,
                    "desiccations": s.desiccation_count,
                    "foreign_genes": s.active_foreign_gene_count,
                    "foreign_pct": s.foreign_gene_pct,
                    "resistance": s.desiccation_resistance,
                    "generation": s.generation,
                    "status": (
                        "quarantine"
                        if s.in_quarantine
                        else ("desiccated" if s.is_desiccated else "active")
                    ),
                }
                for s in strategies
            ],
        }

        # Save to statistics table
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    INSERT INTO hgt_statistics
                    (timestamp, total_strategies, total_desiccations,
                     total_foreign_genes, active_foreign_genes,
                     quarantine_survival_rate, avg_foreign_gene_pct,
                     avg_desiccation_resistance, report_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        report["timestamp"],
                        report["total_strategies"],
                        report["total_desiccations"],
                        report["total_foreign_genes"],
                        report["active_foreign_genes"],
                        report["quarantine_survival_rate"],
                        report["avg_foreign_gene_pct"],
                        report["avg_desiccation_resistance"],
                        json.dumps(report, sort_keys=True, default=str),
                    ),
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to save HGT statistics: %s", e)

        return report

    # ----------------------------------------------------------
    # QUERY HELPERS
    # ----------------------------------------------------------

    def get_strategy(self, strategy_id: str) -> Optional[StrategyOrganism]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)

    def get_strategy_by_name(
        self, name: str, instrument: str, timeframe: str
    ) -> Optional[StrategyOrganism]:
        """Get a strategy by name + instrument + timeframe."""
        sid = hashlib.md5(
            f"{name}|{instrument}|{timeframe}".encode()
        ).hexdigest()[:16]
        return self.strategies.get(sid)

    def get_all_strategies(self) -> List[StrategyOrganism]:
        """Get all registered strategies."""
        return list(self.strategies.values())

    def get_foreign_gene_census(self, strategy_id: str) -> Dict:
        """
        Get a census of foreign genes in a strategy.
        Analogous to cataloging non-metazoan genes in a rotifer genome.
        """
        s = self.strategies.get(strategy_id)
        if s is None:
            return {"error": "Strategy not found"}

        census = {
            "strategy_name": s.strategy_name,
            "total_foreign_genes": len(s.foreign_genes),
            "active": sum(1 for g in s.foreign_genes if g.status == GENE_STATUS_ACTIVE),
            "neutral": sum(1 for g in s.foreign_genes if g.status == GENE_STATUS_NEUTRAL),
            "rejected": sum(1 for g in s.foreign_genes if g.status == GENE_STATUS_REJECTED),
            "foreign_gene_pct": s.foreign_gene_pct,
            "by_component": {},
            "by_donor": {},
            "genes": [],
        }

        for g in s.foreign_genes:
            # By component
            ct = g.component_type
            if ct not in census["by_component"]:
                census["by_component"][ct] = {"active": 0, "neutral": 0, "rejected": 0}
            census["by_component"][ct][g.status] = (
                census["by_component"][ct].get(g.status, 0) + 1
            )

            # By donor
            dn = g.donor_name
            if dn not in census["by_donor"]:
                census["by_donor"][dn] = 0
            if g.status == GENE_STATUS_ACTIVE:
                census["by_donor"][dn] += 1

            # Gene details
            census["genes"].append(
                {
                    "gene_id": g.gene_id[:8],
                    "type": g.component_type,
                    "donor": g.donor_name,
                    "te_site": g.te_integration_site,
                    "contribution": g.contribution_score,
                    "status": g.status,
                    "desiccation_cycle": g.desiccation_event,
                }
            )

        return census

    def get_desiccation_history(self, strategy_id: str) -> List[Dict]:
        """Get the desiccation history for a strategy."""
        events = []
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM desiccation_events
                    WHERE strategy_id = ?
                    ORDER BY triggered_at DESC
                    """,
                    (strategy_id,),
                )
                for row in cursor.fetchall():
                    events.append(
                        {
                            "event_id": row["event_id"],
                            "triggered_at": row["triggered_at"],
                            "reason": row["trigger_reason"],
                            "drawdown": row["drawdown_at_trigger"],
                            "shattered": json.loads(
                                row["components_shattered"] or "[]"
                            ),
                            "donors_scanned": row["donor_strategies_scanned"],
                            "genes_incorporated": row["foreign_genes_incorporated"],
                            "reassembled_at": row["reassembly_completed_at"],
                            "quarantine_passed": (
                                bool(row["quarantine_passed"])
                                if row["quarantine_passed"] is not None
                                else None
                            ),
                        }
                    )
        except Exception as e:
            log.warning("Failed to get desiccation history: %s", e)
        return events


# ============================================================
# TEQA INTEGRATION: Bdelloid HGT Bridge
# ============================================================

class BdelloidTEQABridge:
    """
    Bridges the Bdelloid HGT engine with the existing TEQA v3.0 system.

    The bridge:
      1. Maps TE activation patterns to HGT integration site compatibility
      2. Feeds desiccation/reassembly events into the TEQA pipeline
      3. Writes HGT status to JSON for MQL5 EA consumption
      4. Provides a gate result for Jardine's Gate integration

    This is how horizontal gene transfer feeds back into the broader
    transposon ecosystem.
    """

    def __init__(
        self,
        hgt_engine: BdelloidHGTEngine,
        signal_file: str = None,
    ):
        self.hgt = hgt_engine
        if signal_file is None:
            self.signal_file = str(
                Path(__file__).parent / "bdelloid_hgt_signal.json"
            )
        else:
            self.signal_file = signal_file

    def get_hgt_gate_result(self, strategy_id: str) -> Dict:
        """
        Compute HGT gate result for integration with Jardine's Gate.

        A strategy in quarantine or active desiccation should NOT trade
        (or should trade with reduced confidence). A strategy with many
        successful foreign genes gets a confidence boost.

        Returns:
            {
                gate_pass: bool,
                confidence_modifier: float,
                status: str,
                foreign_gene_pct: float,
                desiccation_resistance: float,
            }
        """
        s = self.hgt.get_strategy(strategy_id)
        if s is None:
            return {
                "gate_pass": True,
                "confidence_modifier": 1.0,
                "status": "unknown",
                "foreign_gene_pct": 0.0,
                "desiccation_resistance": 0.0,
            }

        # Gate blocks during desiccation and quarantine
        if s.is_desiccated or s.in_quarantine:
            return {
                "gate_pass": False,
                "confidence_modifier": 0.0,
                "status": "quarantine" if s.in_quarantine else "desiccated",
                "foreign_gene_pct": s.foreign_gene_pct,
                "desiccation_resistance": s.desiccation_resistance,
            }

        # Confidence boost from accumulated foreign genes and resistance
        # More foreign genes + more resistance = higher confidence modifier
        gene_boost = s.active_foreign_gene_count * 0.02  # +2% per active gene
        resistance_boost = s.desiccation_resistance * 0.50  # Up to +10%
        confidence_mod = 1.0 + min(0.15, gene_boost + resistance_boost)

        return {
            "gate_pass": True,
            "confidence_modifier": confidence_mod,
            "status": "active",
            "foreign_gene_pct": s.foreign_gene_pct,
            "desiccation_resistance": s.desiccation_resistance,
        }

    def write_signal_file(self):
        """
        Write current HGT status to JSON for MQL5 EA consumption.
        """
        report = self.hgt.generate_hgt_report()

        signal = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "total_strategies": report["total_strategies"],
            "total_desiccations": report["total_desiccations"],
            "active_foreign_genes": report["active_foreign_genes"],
            "quarantine_survival_rate": report["quarantine_survival_rate"],
            "avg_foreign_gene_pct": report["avg_foreign_gene_pct"],
            "strategies": report.get("strategies", []),
        }

        try:
            tmp_path = self.signal_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(signal, f, indent=2)
            os.replace(tmp_path, self.signal_file)
        except Exception as e:
            log.warning("Failed to write HGT signal file: %s", e)


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
    print("  BDELLOID ROTIFERS ENGINE -- Horizontal Gene Transfer")
    print("  Strategy DNA Theft via Desiccation-Triggered Component Absorption")
    print("=" * 76)

    # Use temp DB for test
    test_db = str(Path(__file__).parent / "test_bdelloid_rotifers.db")

    engine = BdelloidHGTEngine(db_path=test_db, seed=42)

    # ----------------------------------------------------------
    # Register several strategy organisms with different components
    # ----------------------------------------------------------
    print("\n  --- STEP 1: REGISTER STRATEGY ORGANISMS ---")

    s1 = engine.register_strategy(
        strategy_name="momentum_breakout",
        instrument="XAUUSD",
        timeframe="M5",
        components={
            COMPONENT_ENTRY: {
                "type": "breakout_high",
                "lookback": 20,
                "buffer_pct": 0.001,
            },
            COMPONENT_EXIT: {
                "type": "trailing_stop",
                "trail_atr_mult": 1.5,
                "activation_atr": 1.0,
            },
            COMPONENT_REGIME: {
                "type": "trending_up",
                "adx_min": 25,
                "ema_diff_pct": 0.005,
            },
            COMPONENT_TE_WEIGHTS: {
                "L1_Neuronal": 0.80,
                "Alu_Exonization": 0.65,
                "HERV_Synapse": 0.40,
                "Mariner_Tc1": 0.70,
                "hobo": 0.55,
            },
            COMPONENT_RISK: {"position_pct": 0.02, "max_correlated": 3},
            COMPONENT_TIMING: {"session": "london_ny_overlap", "weight": 1.2},
        },
    )
    print(f"    {s1.summary()}")

    s2 = engine.register_strategy(
        strategy_name="mean_revert_rsi",
        instrument="XAUUSD",
        timeframe="M5",
        components={
            COMPONENT_ENTRY: {
                "type": "rsi_oversold",
                "period": 14,
                "threshold": 30,
            },
            COMPONENT_EXIT: {
                "type": "fixed_tp",
                "tp_atr_mult": 2.0,
                "sl_atr_mult": 1.0,
            },
            COMPONENT_REGIME: {
                "type": "ranging",
                "adx_max": 20,
                "bb_width_max": 0.02,
            },
            COMPONENT_TE_WEIGHTS: {
                "L1_Neuronal": 0.60,
                "Alu_Exonization": 0.75,
                "HERV_Synapse": 0.80,
                "Mariner_Tc1": 0.50,
                "hobo": 0.45,
            },
            COMPONENT_RISK: {"position_pct": 0.01, "max_correlated": 2},
            COMPONENT_TIMING: {"session": "asia", "weight": 0.8},
        },
    )
    print(f"    {s2.summary()}")

    s3 = engine.register_strategy(
        strategy_name="ema_crossover",
        instrument="BTCUSD",
        timeframe="H1",
        components={
            COMPONENT_ENTRY: {
                "type": "ema_cross_up",
                "fast": 8,
                "slow": 21,
            },
            COMPONENT_EXIT: {
                "type": "atr_trailing",
                "atr_period": 14,
                "atr_mult": 2.0,
            },
            COMPONENT_REGIME: {
                "type": "volatile",
                "atr_ratio_min": 1.5,
            },
            COMPONENT_TE_WEIGHTS: {
                "L1_Neuronal": 0.70,
                "Alu_Exonization": 0.55,
                "HERV_Synapse": 0.60,
                "Mariner_Tc1": 0.85,
                "hobo": 0.75,
            },
            COMPONENT_RISK: {"position_pct": 0.015, "max_correlated": 4},
            COMPONENT_TIMING: {"session": "all", "weight": 1.0},
        },
    )
    print(f"    {s3.summary()}")

    s4 = engine.register_strategy(
        strategy_name="stoch_momentum",
        instrument="XAUUSD",
        timeframe="M5",
        components={
            COMPONENT_ENTRY: {
                "type": "stoch_oversold",
                "k_period": 14,
                "d_period": 3,
                "threshold": 20,
            },
            COMPONENT_EXIT: {
                "type": "partial_close",
                "partial_pct": 0.50,
                "partial_at_atr": 1.5,
                "remainder_trail": 2.0,
            },
            COMPONENT_REGIME: {
                "type": "mean_reverting",
                "hurst_max": 0.4,
            },
            COMPONENT_TE_WEIGHTS: {
                "L1_Neuronal": 0.90,
                "Alu_Exonization": 0.70,
                "HERV_Synapse": 0.55,
                "Mariner_Tc1": 0.65,
                "hobo": 0.80,
            },
            COMPONENT_RISK: {"position_pct": 0.02, "max_correlated": 2},
            COMPONENT_TIMING: {"session": "ny", "weight": 1.1},
        },
    )
    print(f"    {s4.summary()}")

    print(f"\n    Total strategies registered: {len(engine.strategies)}")

    # ----------------------------------------------------------
    # Simulate trade outcomes: make s2/s4 winners, s1/s3 losers
    # ----------------------------------------------------------
    print("\n  --- STEP 2: SIMULATE TRADE OUTCOMES ---")

    rng = random.Random(42)

    # s2 (mean_revert_rsi) performs well: ~70% WR
    print("\n    Simulating: mean_revert_rsi (STRONG performer)")
    for i in range(25):
        won = rng.random() < 0.70
        pnl = rng.uniform(0.50, 2.00) if won else -rng.uniform(0.30, 1.00)
        engine.record_trade_outcome(s2.strategy_id, won, pnl)
    s2_live = engine.get_strategy(s2.strategy_id)
    print(f"    {s2_live.summary()}")

    # s4 (stoch_momentum) performs well: ~65% WR
    print("\n    Simulating: stoch_momentum (GOOD performer)")
    for i in range(20):
        won = rng.random() < 0.65
        pnl = rng.uniform(0.40, 1.80) if won else -rng.uniform(0.20, 0.90)
        engine.record_trade_outcome(s4.strategy_id, won, pnl)
    s4_live = engine.get_strategy(s4.strategy_id)
    print(f"    {s4_live.summary()}")

    # s3 (ema_crossover BTC) performs okay: ~55% WR
    print("\n    Simulating: ema_crossover BTCUSD (OKAY performer)")
    for i in range(18):
        won = rng.random() < 0.55
        pnl = rng.uniform(0.30, 1.50) if won else -rng.uniform(0.30, 1.20)
        engine.record_trade_outcome(s3.strategy_id, won, pnl)
    s3_live = engine.get_strategy(s3.strategy_id)
    print(f"    {s3_live.summary()}")

    # s1 (momentum_breakout) performs badly: ~30% WR -> should trigger desiccation
    print("\n    Simulating: momentum_breakout (POOR performer -> desiccation candidate)")
    for i in range(20):
        won = rng.random() < 0.30
        pnl = rng.uniform(0.30, 1.00) if won else -rng.uniform(0.50, 1.50)
        result = engine.record_trade_outcome(s1.strategy_id, won, pnl)
        if result.get("desiccated"):
            print(f"    *** DESICCATION TRIGGERED at trade {i+1}! ***")
            break

    s1_live = engine.get_strategy(s1.strategy_id)
    print(f"    {s1_live.summary()}")

    # ----------------------------------------------------------
    # Check desiccation status
    # ----------------------------------------------------------
    print("\n  --- STEP 3: POST-DESICCATION STATUS ---")

    s1_live = engine.get_strategy(s1.strategy_id)
    print(f"    Desiccated: {s1_live.is_desiccated}")
    print(f"    In quarantine: {s1_live.in_quarantine}")
    print(f"    Desiccation count: {s1_live.desiccation_count}")
    print(f"    Generation: {s1_live.generation}")
    print(f"    Resistance: {s1_live.desiccation_resistance:.1%}")
    print(f"    Foreign genes: {s1_live.active_foreign_gene_count}")

    # ----------------------------------------------------------
    # Foreign gene census
    # ----------------------------------------------------------
    print("\n  --- STEP 4: FOREIGN GENE CENSUS ---")
    census = engine.get_foreign_gene_census(s1.strategy_id)
    print(f"    Total foreign genes: {census['total_foreign_genes']}")
    print(f"    Active: {census['active']}")
    print(f"    Neutral: {census['neutral']}")
    print(f"    Rejected: {census['rejected']}")
    print(f"    Foreign gene %: {census['foreign_gene_pct']:.1%}")

    if census["genes"]:
        print("\n    Individual foreign genes:")
        for g in census["genes"]:
            print(
                f"      [{g['gene_id']}] {g['type']} from {g['donor']} "
                f"via TE:{g['te_site']} status={g['status']}"
            )

    # ----------------------------------------------------------
    # Run quarantine trades
    # ----------------------------------------------------------
    print("\n  --- STEP 5: QUARANTINE TRADES ---")
    if s1_live.in_quarantine:
        print(
            f"    Running {REASSEMBLY_QUARANTINE_TRADES} quarantine trades..."
        )
        for i in range(REASSEMBLY_QUARANTINE_TRADES):
            # Simulate: hybrid strategy performs better (~55% WR)
            won = rng.random() < 0.55
            pnl = rng.uniform(0.30, 1.20) if won else -rng.uniform(0.30, 0.80)
            result = engine.record_trade_outcome(s1.strategy_id, won, pnl)
            if result.get("quarantine_result"):
                q = result["quarantine_result"]
                print(
                    f"    Quarantine complete: {'PASSED' if q['passed'] else 'FAILED'} "
                    f"WR={q['quarantine_wr']:.1%}"
                )
                break

    s1_final = engine.get_strategy(s1.strategy_id)
    print(f"\n    Final state: {s1_final.summary()}")

    # ----------------------------------------------------------
    # Desiccation history
    # ----------------------------------------------------------
    print("\n  --- STEP 6: DESICCATION HISTORY ---")
    history = engine.get_desiccation_history(s1.strategy_id)
    for ev in history:
        print(
            f"    Event {ev['event_id'][:8]}: reason={ev['reason']} "
            f"DD={ev['drawdown']:.1%} shattered={ev['shattered']} "
            f"genes_inc={ev['genes_incorporated']} "
            f"quarantine={'PASS' if ev['quarantine_passed'] else 'FAIL' if ev['quarantine_passed'] is not None else 'PENDING'}"
        )

    # ----------------------------------------------------------
    # HGT Report
    # ----------------------------------------------------------
    print("\n  --- STEP 7: HGT ECOSYSTEM REPORT ---")
    report = engine.generate_hgt_report()
    print(f"    Total strategies:       {report['total_strategies']}")
    print(f"    Total desiccations:     {report['total_desiccations']}")
    print(f"    Total foreign genes:    {report['total_foreign_genes']}")
    print(f"    Active foreign genes:   {report['active_foreign_genes']}")
    print(
        f"    Quarantine survival:    {report['quarantine_survival_rate']:.1%}"
    )
    print(f"    Avg foreign gene %:     {report['avg_foreign_gene_pct']:.1%}")
    print(
        f"    Avg resistance:         {report['avg_desiccation_resistance']:.1%}"
    )

    if report["top_donors"]:
        print("\n    Top donors:")
        for donor_name, count in report["top_donors"]:
            print(f"      {donor_name}: {count} active genes donated")

    if report["component_distribution"]:
        print("\n    Component distribution:")
        for ct, count in report["component_distribution"].items():
            print(f"      {ct}: {count} active foreign genes")

    # ----------------------------------------------------------
    # TEQA bridge test
    # ----------------------------------------------------------
    print("\n  --- STEP 8: TEQA BRIDGE GATE TEST ---")
    bridge = BdelloidTEQABridge(engine)

    for s in engine.get_all_strategies():
        gate = bridge.get_hgt_gate_result(s.strategy_id)
        print(
            f"    {s.strategy_name}: gate_pass={gate['gate_pass']} "
            f"conf_mod={gate['confidence_modifier']:.3f} "
            f"status={gate['status']} "
            f"foreign={gate['foreign_gene_pct']:.1%}"
        )

    # Write signal file
    bridge.write_signal_file()
    print(f"\n    Signal file written to: {bridge.signal_file}")

    # ----------------------------------------------------------
    # Cleanup test DB
    # ----------------------------------------------------------
    try:
        os.remove(test_db)
    except OSError:
        pass

    # Remove test signal file if created
    try:
        test_signal = str(Path(__file__).parent / "bdelloid_hgt_signal.json")
        if os.path.exists(test_signal):
            os.remove(test_signal)
    except OSError:
        pass

    print("\n" + "=" * 76)
    print("  Bdelloid Rotifers HGT Engine test complete.")
    print(
        "  \"Diversity without sex -- theft under duress.\""
    )
    print("=" * 76)
