"""
ELECTRIC ORGANS ENGINE -- Convergent Signal Evolution
======================================================
Cross-instrument convergent pattern detection for universal edge discovery.

Biological basis:
    Electric organs evolved INDEPENDENTLY at least 6 times in unrelated
    fish lineages (electric eels, torpedo rays, electric catfish,
    elephantfish, stargazers, skates). Each time, the same solution
    emerged: muscle cells transform into electrocytes by upregulating
    sodium channel genes and downregulating contractile genes.
    Transposable elements played a key role in this regulatory rewiring.

Trading translation:
    Each traded instrument (XAUUSD, BTCUSD, ETHUSD, NAS100) is an
    independent "species". If 3+ instruments independently domesticate
    the same TE pattern combo, that pattern is probably not noise --
    it is a UNIVERSAL edge (convergent evolution). Convergent winners
    get a super-boost (sodium channel amplification). Convergent losers
    get super-suppressed (contractile gene suppression).

Integration:
    - Reads domestication DBs from multiple instruments
    - Identifies TE combos independently domesticated in 3+ lineages
    - Applies convergence super-boost/suppress into TEQA pipeline
    - Persists convergence findings in SQLite
    - Verifies independence via rolling correlation

Authors: DooDoo + Claude
Date:    2026-02-08
Version: ELECTRIC-ORGANS-1.0
"""

import json
import hashlib
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Config loader for trading values -- no hardcoded trading params
from config_loader import CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)

VERSION = "ELECTRIC-ORGANS-1.0"

# ============================================================
# CONSTANTS (algorithm-internal, NOT trading values)
# ============================================================

# Convergence detection
CONVERGENCE_MIN_INSTRUMENTS = 3        # Minimum instruments sharing pattern
CONVERGENCE_ELECTROCYTE_THRESH = 0.60  # Ratio to achieve electrocyte status
CONVERGENCE_SCAN_INTERVAL_SEC = 300    # 5 min between convergence scans
CONVERGENCE_MIN_WR = 0.65             # Min avg WR for super-boost
CONVERGENCE_MIN_PF = 1.50             # Min avg PF for super-boost

# Super-boost / super-suppress multipliers
SODIUM_CHANNEL_BOOST = 1.50           # 1.5x on top of domestication
CONTRACTILE_SUPPRESS = 0.30           # 0.3x for convergent losers
PARTIAL_CONVERGENCE_SCALE = 0.50      # Scale factor for partial convergence

# Independence verification
MIN_INDEPENDENCE_THRESHOLD = 0.40     # Max absolute correlation for "independent"
INDEPENDENCE_WINDOW_BARS = 100        # Window for rolling correlation
INDEPENDENCE_CHECK_INTERVAL = 3600    # Re-verify independence every hour
INDEPENDENCE_PENALTY = 0.30           # Reduce boost to 30% if correlated

# Pattern maturity for convergence consideration
CONVERGENCE_MIN_TRADES = 10           # Min trades per instrument for pattern
CONVERGENT_LOSER_MAX_WR = 0.45        # Below this = convergent loser
CONVERGENT_LOSER_MIN_OBS = 3          # Must fail in 3+ instruments

# Staleness
CONVERGENCE_EXPIRY_DAYS = 60          # Re-evaluate after 60 days

# Default instruments (independent lineages)
DEFAULT_LINEAGES = ["XAUUSD", "BTCUSD", "ETHUSD", "NAS100"]

# Domestication DB min trades threshold for "observed"
OBSERVED_MIN_TRADES = 5


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Lineage:
    """
    An independent instrument lineage with its own domestication history.
    Biological parallel: a separate fish species in which electric organs
    may or may not have independently evolved.
    """
    symbol: str
    domestication_db_path: str
    is_active: bool = True

    def __repr__(self):
        status = "active" if self.is_active else "inactive"
        return f"Lineage({self.symbol}, {status})"


@dataclass
class ConvergentPattern:
    """
    A TE combo that has been independently domesticated in multiple instruments.
    Biological parallel: a trait that evolved convergently in separate lineages.
    """
    pattern_hash: str
    te_combo: str
    n_instruments: int = 0
    n_observed: int = 0
    convergence_score: float = 0.0
    is_electrocyte: bool = False
    super_boost: float = 1.0
    avg_win_rate: float = 0.0
    avg_profit_factor: float = 0.0
    first_detected: str = ""
    last_verified: str = ""
    independence_ok: bool = True
    lineages_present: List[str] = field(default_factory=list)


@dataclass
class ConvergenceSignal:
    """
    Output signal from the convergence engine for a specific TE combo.
    This gets applied on top of the domestication boost in the TEQA pipeline.
    """
    pattern_hash: str
    te_combo: str
    convergence_boost: float  # Multiplier: >1 = amplify, <1 = suppress
    convergence_score: float
    is_electrocyte: bool
    n_instruments: int
    independence_verified: bool
    lineages: List[str]


# ============================================================
# CONVERGENT SIGNAL ENGINE
# ============================================================

class ConvergentSignalEngine:
    """
    Core engine implementing the Electric Organs algorithm.

    Scans domestication databases from multiple independent instruments
    (lineages) to identify TE patterns that have been convergently
    domesticated. Convergent winners get amplified (sodium channel boost).
    Convergent losers get suppressed (contractile suppression).

    This provides cross-instrument validation that is the strongest
    evidence against overfitting: if multiple unrelated instruments
    independently discovered the same winning pattern, it is probably real.
    """

    def __init__(
        self,
        lineage_symbols: List[str] = None,
        db_dir: str = None,
        convergence_db_path: str = None,
    ):
        """
        Initialize the convergent signal engine.

        Args:
            lineage_symbols: List of instrument symbols to monitor.
                Defaults to DEFAULT_LINEAGES.
            db_dir: Directory containing domestication DBs.
                Defaults to the directory of this script.
            convergence_db_path: Path for the convergence results DB.
                Defaults to electric_organs_convergence.db in db_dir.
        """
        self.symbols = lineage_symbols or list(DEFAULT_LINEAGES)
        self.db_dir = db_dir or str(Path(__file__).parent)

        if convergence_db_path is None:
            self.convergence_db_path = str(
                Path(self.db_dir) / "electric_organs_convergence.db"
            )
        else:
            self.convergence_db_path = convergence_db_path

        # Discover lineages
        self.lineages: List[Lineage] = []
        self._discover_lineages()

        # Initialize convergence DB
        self._init_convergence_db()

        # Cache for convergence lookups (pattern_hash -> ConvergentPattern)
        self._convergence_cache: Dict[str, ConvergentPattern] = {}
        self._cache_loaded_at: float = 0.0

        # Timing
        self._last_scan_time: float = 0.0
        self._last_independence_check: float = 0.0

        # Signal output file
        self.signal_file = str(
            Path(self.db_dir) / "electric_organs_signal.json"
        )

        log.info(
            "[ELECTRIC_ORGANS] Initialized with %d lineages: %s",
            len(self.lineages),
            [l.symbol for l in self.lineages],
        )

    # ----------------------------------------------------------
    # LINEAGE DISCOVERY
    # ----------------------------------------------------------

    def _discover_lineages(self):
        """
        Discover domestication databases for each configured instrument.
        Each instrument with a domestication DB becomes an independent lineage.
        """
        self.lineages = []

        for symbol in self.symbols:
            db_path = self._find_domestication_db(symbol)
            if db_path:
                self.lineages.append(Lineage(
                    symbol=symbol,
                    domestication_db_path=db_path,
                    is_active=True,
                ))
                log.info(
                    "[ELECTRIC_ORGANS] Discovered lineage: %s -> %s",
                    symbol, db_path,
                )
            else:
                log.debug(
                    "[ELECTRIC_ORGANS] No domestication DB for %s, skipping",
                    symbol,
                )

        if len(self.lineages) < 2:
            log.warning(
                "[ELECTRIC_ORGANS] Only %d lineages found. Need >= 2 for "
                "convergence detection. Patterns from: %s",
                len(self.lineages),
                [l.symbol for l in self.lineages],
            )

    def _find_domestication_db(self, symbol: str) -> Optional[str]:
        """
        Locate the domestication database for a given instrument.

        Search order:
          1. teqa_domestication_{symbol}.db  (per-instrument DB)
          2. teqa_domestication.db           (shared DB -- check for symbol data)
        """
        # Per-instrument DB (preferred)
        per_instrument = Path(self.db_dir) / f"teqa_domestication_{symbol}.db"
        if per_instrument.exists():
            return str(per_instrument)

        # Shared DB fallback
        shared_db = Path(self.db_dir) / "teqa_domestication.db"
        if shared_db.exists():
            # The shared DB is valid for all instruments when there is only
            # one domestication DB. Return it as the source.
            return str(shared_db)

        return None

    # ----------------------------------------------------------
    # CONVERGENCE DATABASE
    # ----------------------------------------------------------

    def _init_convergence_db(self):
        """Initialize the convergence results database."""
        try:
            with sqlite3.connect(self.convergence_db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS convergent_patterns (
                        pattern_hash TEXT PRIMARY KEY,
                        te_combo TEXT NOT NULL,
                        n_instruments INTEGER DEFAULT 0,
                        n_observed INTEGER DEFAULT 0,
                        convergence_score REAL DEFAULT 0.0,
                        is_electrocyte INTEGER DEFAULT 0,
                        super_boost REAL DEFAULT 1.0,
                        avg_win_rate REAL DEFAULT 0.0,
                        avg_profit_factor REAL DEFAULT 0.0,
                        first_detected TEXT,
                        last_verified TEXT,
                        independence_ok INTEGER DEFAULT 1,
                        lineages_present TEXT DEFAULT '[]',
                        max_pairwise_corr REAL DEFAULT 0.0
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS convergence_scans (
                        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        n_lineages INTEGER,
                        n_convergent_found INTEGER,
                        n_electrocytes INTEGER,
                        n_convergent_losers INTEGER,
                        scan_duration_ms REAL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS lineage_pattern_details (
                        pattern_hash TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        posterior_wr REAL DEFAULT 0.5,
                        profit_factor REAL DEFAULT 0.0,
                        win_count INTEGER DEFAULT 0,
                        loss_count INTEGER DEFAULT 0,
                        domesticated INTEGER DEFAULT 0,
                        boost_factor REAL DEFAULT 1.0,
                        last_updated TEXT,
                        PRIMARY KEY (pattern_hash, symbol)
                    )
                """)
                conn.commit()
        except Exception as e:
            log.warning("[ELECTRIC_ORGANS] Convergence DB init failed: %s", e)

    def _save_convergent_pattern(self, pattern: ConvergentPattern):
        """Persist a convergent pattern to the database."""
        try:
            with sqlite3.connect(self.convergence_db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                now = datetime.now().isoformat()
                conn.execute("""
                    INSERT OR REPLACE INTO convergent_patterns
                    (pattern_hash, te_combo, n_instruments, n_observed,
                     convergence_score, is_electrocyte, super_boost,
                     avg_win_rate, avg_profit_factor, first_detected,
                     last_verified, independence_ok, lineages_present)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_hash,
                    pattern.te_combo,
                    pattern.n_instruments,
                    pattern.n_observed,
                    pattern.convergence_score,
                    1 if pattern.is_electrocyte else 0,
                    pattern.super_boost,
                    pattern.avg_win_rate,
                    pattern.avg_profit_factor,
                    pattern.first_detected or now,
                    now,
                    1 if pattern.independence_ok else 0,
                    json.dumps(pattern.lineages_present),
                ))
                conn.commit()
        except Exception as e:
            log.warning(
                "[ELECTRIC_ORGANS] Failed to save pattern %s: %s",
                pattern.pattern_hash, e,
            )

    def _save_lineage_detail(
        self,
        pattern_hash: str,
        symbol: str,
        posterior_wr: float,
        profit_factor: float,
        win_count: int,
        loss_count: int,
        domesticated: bool,
        boost_factor: float,
    ):
        """Save per-lineage detail for a convergent pattern."""
        try:
            with sqlite3.connect(self.convergence_db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT OR REPLACE INTO lineage_pattern_details
                    (pattern_hash, symbol, posterior_wr, profit_factor,
                     win_count, loss_count, domesticated, boost_factor,
                     last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_hash, symbol, posterior_wr, profit_factor,
                    win_count, loss_count, 1 if domesticated else 0,
                    boost_factor, datetime.now().isoformat(),
                ))
                conn.commit()
        except Exception as e:
            log.debug(
                "[ELECTRIC_ORGANS] Failed to save lineage detail: %s", e,
            )

    def _save_scan_record(
        self,
        n_convergent: int,
        n_electrocytes: int,
        n_losers: int,
        duration_ms: float,
    ):
        """Record scan metadata."""
        try:
            with sqlite3.connect(self.convergence_db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO convergence_scans
                    (timestamp, n_lineages, n_convergent_found,
                     n_electrocytes, n_convergent_losers, scan_duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    len(self.lineages),
                    n_convergent,
                    n_electrocytes,
                    n_losers,
                    duration_ms,
                ))
                conn.commit()
        except Exception as e:
            log.debug("[ELECTRIC_ORGANS] Failed to save scan record: %s", e)

    # ----------------------------------------------------------
    # PHASE 2: CONVERGENCE SCAN
    # ----------------------------------------------------------

    def convergence_scan(self) -> List[ConvergentPattern]:
        """
        Scan all lineage domestication databases and identify TE patterns
        that have been independently domesticated in multiple instruments.

        This is the core of the algorithm: finding convergent evolution
        across independent trading lineages.

        Returns:
            List of ConvergentPattern objects for patterns meeting the
            convergence threshold.
        """
        t_start = time.time()

        if len(self.lineages) < 2:
            log.debug(
                "[ELECTRIC_ORGANS] Need >= 2 lineages, have %d. Skipping scan.",
                len(self.lineages),
            )
            return []

        # Step 1: Harvest domesticated and observed patterns from each lineage
        # domesticated_map: pattern_hash -> {symbols, win_rates, pfs, te_combo}
        domesticated_map: Dict[str, Dict] = {}
        # observed_map: pattern_hash -> set of symbols that observed it
        observed_map: Dict[str, set] = {}

        for lineage in self.lineages:
            if not lineage.is_active:
                continue

            domesticated_rows, observed_hashes = self._harvest_lineage(lineage)

            for row in domesticated_rows:
                p_hash = row["pattern_hash"]
                if p_hash not in domesticated_map:
                    domesticated_map[p_hash] = {
                        "symbols": [],
                        "win_rates": [],
                        "profit_factors": [],
                        "te_combo": row["te_combo"],
                        "details": [],
                    }
                domesticated_map[p_hash]["symbols"].append(lineage.symbol)
                domesticated_map[p_hash]["win_rates"].append(row["posterior_wr"])
                domesticated_map[p_hash]["profit_factors"].append(row["profit_factor"])
                domesticated_map[p_hash]["details"].append(row)

            for p_hash in observed_hashes:
                if p_hash not in observed_map:
                    observed_map[p_hash] = set()
                observed_map[p_hash].add(lineage.symbol)

        # Step 2: Identify convergent patterns (domesticated in 3+ lineages)
        convergent_patterns: List[ConvergentPattern] = []
        n_electrocytes = 0
        n_losers = 0

        for p_hash, data in domesticated_map.items():
            n_instruments = len(data["symbols"])
            n_observed = len(observed_map.get(p_hash, set(data["symbols"])))
            convergence_score = n_instruments / max(1, n_observed)

            if n_instruments < CONVERGENCE_MIN_INSTRUMENTS:
                continue

            avg_wr = float(np.mean(data["win_rates"]))
            avg_pf = float(np.mean(data["profit_factors"]))

            is_electrocyte = convergence_score >= CONVERGENCE_ELECTROCYTE_THRESH

            # Determine super-boost
            if is_electrocyte and avg_wr >= CONVERGENCE_MIN_WR and avg_pf >= CONVERGENCE_MIN_PF:
                # SODIUM CHANNEL AMPLIFICATION -- convergent winner
                super_boost = SODIUM_CHANNEL_BOOST
                n_electrocytes += 1
            elif is_electrocyte and avg_wr < CONVERGENT_LOSER_MAX_WR:
                # CONTRACTILE SUPPRESSION -- convergent loser
                super_boost = CONTRACTILE_SUPPRESS
                n_losers += 1
            else:
                # Partial convergence -- mild boost scaled by score
                super_boost = 1.0 + 0.25 * (convergence_score - 0.5)

            pattern = ConvergentPattern(
                pattern_hash=p_hash,
                te_combo=data["te_combo"],
                n_instruments=n_instruments,
                n_observed=n_observed,
                convergence_score=convergence_score,
                is_electrocyte=is_electrocyte,
                super_boost=super_boost,
                avg_win_rate=avg_wr,
                avg_profit_factor=avg_pf,
                first_detected=datetime.now().isoformat(),
                last_verified=datetime.now().isoformat(),
                independence_ok=True,  # Will be verified separately
                lineages_present=data["symbols"],
            )

            convergent_patterns.append(pattern)

            # Persist pattern
            self._save_convergent_pattern(pattern)

            # Persist per-lineage details
            for detail in data["details"]:
                self._save_lineage_detail(
                    pattern_hash=p_hash,
                    symbol=detail["symbol"],
                    posterior_wr=detail["posterior_wr"],
                    profit_factor=detail["profit_factor"],
                    win_count=detail["win_count"],
                    loss_count=detail["loss_count"],
                    domesticated=True,
                    boost_factor=detail.get("boost_factor", 1.0),
                )

        # Step 3: Check for convergent losers (not domesticated anywhere, failed everywhere)
        convergent_losers = self._find_convergent_losers()
        for loser in convergent_losers:
            # Only add if not already in convergent_patterns
            existing = {p.pattern_hash for p in convergent_patterns}
            if loser.pattern_hash not in existing:
                convergent_patterns.append(loser)
                self._save_convergent_pattern(loser)
                n_losers += 1

        # Update cache
        self._convergence_cache = {p.pattern_hash: p for p in convergent_patterns}
        self._cache_loaded_at = time.time()
        self._last_scan_time = time.time()

        elapsed_ms = (time.time() - t_start) * 1000
        self._save_scan_record(
            len(convergent_patterns), n_electrocytes, n_losers, elapsed_ms,
        )

        log.info(
            "[ELECTRIC_ORGANS] Convergence scan complete: %d convergent patterns "
            "(%d electrocytes, %d losers) from %d lineages [%.0fms]",
            len(convergent_patterns), n_electrocytes, n_losers,
            len(self.lineages), elapsed_ms,
        )

        return convergent_patterns

    def _harvest_lineage(
        self, lineage: Lineage
    ) -> Tuple[List[Dict], List[str]]:
        """
        Read domesticated patterns and observed patterns from a lineage's DB.

        Returns:
            (domesticated_rows, observed_hashes)
            domesticated_rows: list of dicts for domesticated patterns
            observed_hashes: list of pattern_hashes with enough trades
        """
        domesticated_rows: List[Dict] = []
        observed_hashes: List[str] = []

        try:
            if not Path(lineage.domestication_db_path).exists():
                lineage.is_active = False
                return [], []

            with sqlite3.connect(
                lineage.domestication_db_path, timeout=5
            ) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()

                # Get domesticated patterns
                cursor.execute("""
                    SELECT pattern_hash, te_combo, posterior_wr, profit_factor,
                           win_count, loss_count, boost_factor
                    FROM domesticated_patterns
                    WHERE domesticated = 1
                """)
                for row in cursor.fetchall():
                    domesticated_rows.append({
                        "pattern_hash": row[0],
                        "te_combo": row[1],
                        "posterior_wr": row[2] or 0.5,
                        "profit_factor": row[3] or 0.0,
                        "win_count": row[4] or 0,
                        "loss_count": row[5] or 0,
                        "boost_factor": row[6] or 1.0,
                        "symbol": lineage.symbol,
                    })

                # Get all observed patterns (enough trades to evaluate)
                cursor.execute("""
                    SELECT pattern_hash
                    FROM domesticated_patterns
                    WHERE (win_count + loss_count) >= ?
                """, (OBSERVED_MIN_TRADES,))
                observed_hashes = [row[0] for row in cursor.fetchall()]

        except Exception as e:
            log.warning(
                "[ELECTRIC_ORGANS] Failed to harvest lineage %s: %s",
                lineage.symbol, e,
            )
            lineage.is_active = False

        return domesticated_rows, observed_hashes

    def _find_convergent_losers(self) -> List[ConvergentPattern]:
        """
        Find patterns that have been observed in 3+ instruments and
        FAILED to domesticate in all of them. These are convergent losers
        that should be suppressed.

        Biological parallel: a mutation that is deleterious across
        multiple independent lineages -- convergent purifying selection.
        """
        losers: List[ConvergentPattern] = []

        # Collect failure data across lineages
        # failure_map: hash -> {symbols_failed, symbols_observed, te_combo}
        failure_map: Dict[str, Dict] = {}

        for lineage in self.lineages:
            if not lineage.is_active:
                continue
            try:
                with sqlite3.connect(
                    lineage.domestication_db_path, timeout=5
                ) as conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT pattern_hash, te_combo, posterior_wr,
                               win_count, loss_count, domesticated
                        FROM domesticated_patterns
                        WHERE (win_count + loss_count) >= ?
                    """, (CONVERGENCE_MIN_TRADES,))

                    for row in cursor.fetchall():
                        p_hash = row[0]
                        if p_hash not in failure_map:
                            failure_map[p_hash] = {
                                "te_combo": row[1],
                                "observed": set(),
                                "failed": set(),
                                "win_rates": [],
                            }
                        failure_map[p_hash]["observed"].add(lineage.symbol)
                        posterior_wr = row[2] or 0.5
                        domesticated = row[5]
                        failure_map[p_hash]["win_rates"].append(posterior_wr)

                        if not domesticated and posterior_wr < CONVERGENT_LOSER_MAX_WR:
                            failure_map[p_hash]["failed"].add(lineage.symbol)

            except Exception as e:
                log.debug(
                    "[ELECTRIC_ORGANS] Failed to check losers for %s: %s",
                    lineage.symbol, e,
                )

        # Identify convergent losers
        for p_hash, data in failure_map.items():
            n_observed = len(data["observed"])
            n_failed = len(data["failed"])

            if n_observed >= CONVERGENT_LOSER_MIN_OBS and n_failed >= CONVERGENT_LOSER_MIN_OBS:
                avg_wr = float(np.mean(data["win_rates"])) if data["win_rates"] else 0.5
                losers.append(ConvergentPattern(
                    pattern_hash=p_hash,
                    te_combo=data["te_combo"],
                    n_instruments=0,  # 0 domesticated
                    n_observed=n_observed,
                    convergence_score=0.0,
                    is_electrocyte=False,
                    super_boost=CONTRACTILE_SUPPRESS,
                    avg_win_rate=avg_wr,
                    avg_profit_factor=0.0,
                    first_detected=datetime.now().isoformat(),
                    last_verified=datetime.now().isoformat(),
                    independence_ok=True,
                    lineages_present=list(data["failed"]),
                ))

        if losers:
            log.info(
                "[ELECTRIC_ORGANS] Found %d convergent losers (failed in 3+ instruments)",
                len(losers),
            )

        return losers

    # ----------------------------------------------------------
    # PHASE 3: INDEPENDENCE VERIFICATION
    # ----------------------------------------------------------

    def verify_independence(
        self,
        convergent_patterns: List[ConvergentPattern],
        bars_dict: Dict[str, np.ndarray],
    ) -> List[ConvergentPattern]:
        """
        Verify that convergent patterns come from truly independent lineages
        by checking the rolling correlation of returns between instrument pairs.

        If instruments are highly correlated, convergence is less meaningful --
        it might just be the same signal duplicated across correlated markets.

        Args:
            convergent_patterns: Patterns to verify.
            bars_dict: {symbol: np.ndarray} of recent OHLCV bars.

        Returns:
            Updated patterns with independence_ok flag set.
        """
        if not bars_dict or len(bars_dict) < 2:
            log.debug(
                "[ELECTRIC_ORGANS] Not enough bars_dict entries for "
                "independence check (%d)",
                len(bars_dict),
            )
            return convergent_patterns

        # Pre-compute returns for each instrument
        returns_dict: Dict[str, np.ndarray] = {}
        for symbol, bars in bars_dict.items():
            if len(bars) >= INDEPENDENCE_WINDOW_BARS + 1:
                close = bars[:, 3]  # Close prices
                log_returns = np.diff(np.log(close + 1e-10))
                returns_dict[symbol] = log_returns[-INDEPENDENCE_WINDOW_BARS:]
            else:
                log.debug(
                    "[ELECTRIC_ORGANS] Not enough bars for %s (%d < %d)",
                    symbol, len(bars), INDEPENDENCE_WINDOW_BARS + 1,
                )

        for pattern in convergent_patterns:
            symbols = pattern.lineages_present
            max_corr = 0.0

            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    s1, s2 = symbols[i], symbols[j]
                    if s1 in returns_dict and s2 in returns_dict:
                        r1 = returns_dict[s1]
                        r2 = returns_dict[s2]
                        # Align lengths
                        min_len = min(len(r1), len(r2))
                        if min_len < 20:
                            continue
                        corr = self._pearson_correlation(
                            r1[-min_len:], r2[-min_len:]
                        )
                        max_corr = max(max_corr, abs(corr))

            # Apply independence check
            pattern.independence_ok = max_corr < MIN_INDEPENDENCE_THRESHOLD

            if not pattern.independence_ok:
                # Reduce super-boost for correlated lineages
                original_boost = pattern.super_boost
                if pattern.super_boost > 1.0:
                    # Scale down the excess boost
                    excess = pattern.super_boost - 1.0
                    pattern.super_boost = 1.0 + excess * INDEPENDENCE_PENALTY
                elif pattern.super_boost < 1.0:
                    # For suppressors, reduce the suppression (move toward 1.0)
                    deficit = 1.0 - pattern.super_boost
                    pattern.super_boost = 1.0 - deficit * INDEPENDENCE_PENALTY

                log.info(
                    "[ELECTRIC_ORGANS] Pattern %s (%s) failed independence "
                    "check: max_corr=%.3f. Boost adjusted %.3f -> %.3f",
                    pattern.pattern_hash[:8], pattern.te_combo,
                    max_corr, original_boost, pattern.super_boost,
                )

                # Update DB
                self._save_convergent_pattern(pattern)

        self._last_independence_check = time.time()
        return convergent_patterns

    @staticmethod
    def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Pearson correlation coefficient between two arrays."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_diff = x - x_mean
        y_diff = y - y_mean
        numerator = np.sum(x_diff * y_diff)
        denom = np.sqrt(np.sum(x_diff ** 2) * np.sum(y_diff ** 2))
        if denom < 1e-15:
            return 0.0
        return float(numerator / denom)

    # ----------------------------------------------------------
    # PHASE 4: SIGNAL APPLICATION
    # ----------------------------------------------------------

    def get_convergence_boost(
        self,
        active_tes: List[str],
        symbol: str = "",
    ) -> ConvergenceSignal:
        """
        Get the convergence boost for a specific TE combination.

        This is called from the TEQA pipeline to apply the convergence
        multiplier on top of the domestication boost.

        Args:
            active_tes: List of active TE family names.
            symbol: Current instrument (for logging).

        Returns:
            ConvergenceSignal with the boost multiplier and metadata.
        """
        if not active_tes:
            return ConvergenceSignal(
                pattern_hash="",
                te_combo="",
                convergence_boost=1.0,
                convergence_score=0.0,
                is_electrocyte=False,
                n_instruments=0,
                independence_verified=True,
                lineages=[],
            )

        combo = "+".join(sorted(active_tes))
        pattern_hash = hashlib.md5(combo.encode()).hexdigest()[:16]

        # Check cache first
        pattern = self._convergence_cache.get(pattern_hash)

        if pattern is None:
            # Cache miss -- check DB
            pattern = self._lookup_pattern(pattern_hash)

        if pattern is None:
            # Not a convergent pattern
            return ConvergenceSignal(
                pattern_hash=pattern_hash,
                te_combo=combo,
                convergence_boost=1.0,
                convergence_score=0.0,
                is_electrocyte=False,
                n_instruments=0,
                independence_verified=True,
                lineages=[],
            )

        # Check expiry
        if pattern.last_verified:
            try:
                last_dt = datetime.fromisoformat(pattern.last_verified)
                age_days = (datetime.now() - last_dt).days
                if age_days > CONVERGENCE_EXPIRY_DAYS:
                    log.debug(
                        "[ELECTRIC_ORGANS] Convergent pattern %s expired "
                        "(%d days old)",
                        combo, age_days,
                    )
                    return ConvergenceSignal(
                        pattern_hash=pattern_hash,
                        te_combo=combo,
                        convergence_boost=1.0,
                        convergence_score=pattern.convergence_score,
                        is_electrocyte=False,
                        n_instruments=pattern.n_instruments,
                        independence_verified=pattern.independence_ok,
                        lineages=pattern.lineages_present,
                    )
            except (ValueError, TypeError):
                pass

        # Compute effective boost
        if pattern.is_electrocyte and pattern.independence_ok:
            # Full electrocyte -- sodium channel boost
            convergence_boost = pattern.super_boost
            log.info(
                "[ELECTROCYTE] %s | conv=%.2f | %d instruments | "
                "boost=%.3f | %s",
                combo, pattern.convergence_score,
                pattern.n_instruments, convergence_boost,
                symbol,
            )
        elif pattern.convergence_score >= 0.40:
            # Partial convergence -- reduced boost
            excess = pattern.super_boost - 1.0
            convergence_boost = 1.0 + excess * PARTIAL_CONVERGENCE_SCALE
            log.debug(
                "[PARTIAL_CONV] %s | conv=%.2f | partial_boost=%.3f | %s",
                combo, pattern.convergence_score,
                convergence_boost, symbol,
            )
        elif pattern.super_boost < 1.0:
            # Convergent loser -- apply suppression
            convergence_boost = pattern.super_boost
            log.info(
                "[CONTRACTILE] %s | conv_loser | suppress=%.3f | %s",
                combo, convergence_boost, symbol,
            )
        else:
            convergence_boost = 1.0

        return ConvergenceSignal(
            pattern_hash=pattern_hash,
            te_combo=combo,
            convergence_boost=convergence_boost,
            convergence_score=pattern.convergence_score,
            is_electrocyte=pattern.is_electrocyte,
            n_instruments=pattern.n_instruments,
            independence_verified=pattern.independence_ok,
            lineages=pattern.lineages_present,
        )

    def apply_convergence_boost(
        self,
        active_tes: List[str],
        domestication_boost: float,
        symbol: str = "",
    ) -> float:
        """
        Convenience method: apply convergence boost on top of domestication boost.

        This is the primary integration point for BRAIN scripts.

        Args:
            active_tes: Active TE family names from the current signal.
            domestication_boost: The boost from TEDomesticationTracker.
            symbol: Current instrument.

        Returns:
            Final combined boost (domestication * convergence).
        """
        signal = self.get_convergence_boost(active_tes, symbol)
        final_boost = domestication_boost * signal.convergence_boost
        return final_boost

    def check_convergent_loser(
        self,
        active_tes: List[str],
        symbol: str = "",
    ) -> float:
        """
        Check if a TE combo is a convergent loser (failed across instruments).

        This provides an independent suppression signal that can be used
        even when the pattern is not in the convergence cache (direct DB check).

        Args:
            active_tes: Active TE family names.
            symbol: Current instrument.

        Returns:
            Suppression multiplier (1.0 = no suppression, <1.0 = suppress).
        """
        if not active_tes:
            return 1.0

        combo = "+".join(sorted(active_tes))
        pattern_hash = hashlib.md5(combo.encode()).hexdigest()[:16]

        n_failed = 0
        n_observed = 0

        for lineage in self.lineages:
            if not lineage.is_active:
                continue
            try:
                with sqlite3.connect(
                    lineage.domestication_db_path, timeout=3
                ) as conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT posterior_wr, domesticated, win_count, loss_count
                        FROM domesticated_patterns
                        WHERE pattern_hash = ?
                    """, (pattern_hash,))
                    row = cursor.fetchone()

                    if row:
                        total = (row[2] or 0) + (row[3] or 0)
                        if total >= CONVERGENCE_MIN_TRADES:
                            n_observed += 1
                            posterior_wr = row[0] or 0.5
                            domesticated = row[1]
                            if not domesticated and posterior_wr < CONVERGENT_LOSER_MAX_WR:
                                n_failed += 1

            except Exception:
                pass

        if n_observed >= CONVERGENT_LOSER_MIN_OBS and n_failed >= CONVERGENT_LOSER_MIN_OBS:
            log.info(
                "[CONTRACTILE_SUPPRESS] %s failed in %d/%d instruments | %s",
                combo, n_failed, n_observed, symbol,
            )
            return CONTRACTILE_SUPPRESS

        return 1.0

    def _lookup_pattern(self, pattern_hash: str) -> Optional[ConvergentPattern]:
        """Look up a convergent pattern from the database."""
        try:
            with sqlite3.connect(
                self.convergence_db_path, timeout=3
            ) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT pattern_hash, te_combo, n_instruments, n_observed,
                           convergence_score, is_electrocyte, super_boost,
                           avg_win_rate, avg_profit_factor, first_detected,
                           last_verified, independence_ok, lineages_present
                    FROM convergent_patterns
                    WHERE pattern_hash = ?
                """, (pattern_hash,))
                row = cursor.fetchone()

                if row:
                    return ConvergentPattern(
                        pattern_hash=row[0],
                        te_combo=row[1],
                        n_instruments=row[2] or 0,
                        n_observed=row[3] or 0,
                        convergence_score=row[4] or 0.0,
                        is_electrocyte=bool(row[5]),
                        super_boost=row[6] or 1.0,
                        avg_win_rate=row[7] or 0.0,
                        avg_profit_factor=row[8] or 0.0,
                        first_detected=row[9] or "",
                        last_verified=row[10] or "",
                        independence_ok=bool(row[11]),
                        lineages_present=json.loads(row[12]) if row[12] else [],
                    )
        except Exception as e:
            log.debug(
                "[ELECTRIC_ORGANS] Pattern lookup failed: %s", e,
            )
        return None

    # ----------------------------------------------------------
    # CACHE MANAGEMENT
    # ----------------------------------------------------------

    def refresh_cache(self):
        """Reload convergence cache from database."""
        try:
            with sqlite3.connect(
                self.convergence_db_path, timeout=5
            ) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT pattern_hash, te_combo, n_instruments, n_observed,
                           convergence_score, is_electrocyte, super_boost,
                           avg_win_rate, avg_profit_factor, first_detected,
                           last_verified, independence_ok, lineages_present
                    FROM convergent_patterns
                """)

                self._convergence_cache = {}
                for row in cursor.fetchall():
                    p = ConvergentPattern(
                        pattern_hash=row[0],
                        te_combo=row[1],
                        n_instruments=row[2] or 0,
                        n_observed=row[3] or 0,
                        convergence_score=row[4] or 0.0,
                        is_electrocyte=bool(row[5]),
                        super_boost=row[6] or 1.0,
                        avg_win_rate=row[7] or 0.0,
                        avg_profit_factor=row[8] or 0.0,
                        first_detected=row[9] or "",
                        last_verified=row[10] or "",
                        independence_ok=bool(row[11]),
                        lineages_present=json.loads(row[12]) if row[12] else [],
                    )
                    self._convergence_cache[p.pattern_hash] = p

                self._cache_loaded_at = time.time()
                log.debug(
                    "[ELECTRIC_ORGANS] Cache refreshed: %d patterns",
                    len(self._convergence_cache),
                )

        except Exception as e:
            log.warning("[ELECTRIC_ORGANS] Cache refresh failed: %s", e)

    def should_scan(self) -> bool:
        """Check if enough time has passed for another convergence scan."""
        return (time.time() - self._last_scan_time) >= CONVERGENCE_SCAN_INTERVAL_SEC

    def should_check_independence(self) -> bool:
        """Check if enough time has passed for an independence re-check."""
        return (time.time() - self._last_independence_check) >= INDEPENDENCE_CHECK_INTERVAL

    # ----------------------------------------------------------
    # SIGNAL OUTPUT
    # ----------------------------------------------------------

    def write_signal_file(self):
        """
        Write current convergence state to JSON for external consumers.
        """
        electrocytes = [
            p for p in self._convergence_cache.values()
            if p.is_electrocyte and p.independence_ok
        ]
        suppressors = [
            p for p in self._convergence_cache.values()
            if p.super_boost < 1.0
        ]

        signal = {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "n_lineages": len(self.lineages),
            "lineage_symbols": [l.symbol for l in self.lineages],
            "n_convergent_patterns": len(self._convergence_cache),
            "n_electrocytes": len(electrocytes),
            "n_suppressors": len(suppressors),
            "electrocytes": [
                {
                    "hash": p.pattern_hash[:8],
                    "combo": p.te_combo,
                    "score": round(p.convergence_score, 3),
                    "boost": round(p.super_boost, 3),
                    "instruments": p.n_instruments,
                    "avg_wr": round(p.avg_win_rate, 4),
                    "avg_pf": round(p.avg_profit_factor, 2),
                    "independent": p.independence_ok,
                    "lineages": p.lineages_present,
                }
                for p in electrocytes
            ],
            "suppressors": [
                {
                    "hash": p.pattern_hash[:8],
                    "combo": p.te_combo,
                    "suppress": round(p.super_boost, 3),
                    "n_failed": p.n_observed,
                }
                for p in suppressors
            ],
        }

        try:
            tmp_path = self.signal_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(signal, f, indent=2)
            os.replace(tmp_path, self.signal_file)
        except Exception as e:
            log.warning(
                "[ELECTRIC_ORGANS] Failed to write signal file: %s", e,
            )

    # ----------------------------------------------------------
    # FULL CYCLE
    # ----------------------------------------------------------

    def run_cycle(
        self,
        bars_dict: Optional[Dict[str, np.ndarray]] = None,
        force_scan: bool = False,
    ) -> Dict:
        """
        Run a full Electric Organs cycle:
          1. Convergence scan (if due or forced)
          2. Independence verification (if bars provided and due)
          3. Write signal file

        Args:
            bars_dict: Optional {symbol: np.ndarray} for independence check.
            force_scan: Force a convergence scan regardless of timing.

        Returns:
            Dict with cycle results.
        """
        results = {
            "scanned": False,
            "independence_checked": False,
            "n_convergent": len(self._convergence_cache),
            "n_electrocytes": 0,
            "n_suppressors": 0,
        }

        # Step 1: Convergence scan
        if force_scan or self.should_scan():
            patterns = self.convergence_scan()
            results["scanned"] = True
            results["n_convergent"] = len(patterns)

        # Step 2: Independence verification
        if bars_dict and (force_scan or self.should_check_independence()):
            cached_patterns = list(self._convergence_cache.values())
            if cached_patterns:
                self.verify_independence(cached_patterns, bars_dict)
                results["independence_checked"] = True

        # Step 3: Write signal file
        self.write_signal_file()

        # Count electrocytes and suppressors
        results["n_electrocytes"] = sum(
            1 for p in self._convergence_cache.values()
            if p.is_electrocyte and p.independence_ok
        )
        results["n_suppressors"] = sum(
            1 for p in self._convergence_cache.values()
            if p.super_boost < 1.0
        )

        return results

    # ----------------------------------------------------------
    # REPORTING
    # ----------------------------------------------------------

    def get_summary(self) -> Dict:
        """Get summary of current convergence state."""
        patterns = list(self._convergence_cache.values())
        electrocytes = [
            p for p in patterns if p.is_electrocyte and p.independence_ok
        ]
        suppressors = [p for p in patterns if p.super_boost < 1.0]
        partial = [
            p for p in patterns
            if not p.is_electrocyte and p.super_boost > 1.0
        ]

        return {
            "version": VERSION,
            "n_lineages": len(self.lineages),
            "lineages": [
                {"symbol": l.symbol, "active": l.is_active}
                for l in self.lineages
            ],
            "n_total_patterns": len(patterns),
            "n_electrocytes": len(electrocytes),
            "n_partial_convergent": len(partial),
            "n_convergent_losers": len(suppressors),
            "electrocyte_patterns": [
                {
                    "combo": p.te_combo,
                    "score": round(p.convergence_score, 3),
                    "boost": round(p.super_boost, 3),
                    "instruments": p.n_instruments,
                    "avg_wr": round(p.avg_win_rate, 4),
                    "lineages": p.lineages_present,
                }
                for p in sorted(
                    electrocytes,
                    key=lambda x: x.convergence_score,
                    reverse=True,
                )
            ],
            "suppressed_patterns": [
                {
                    "combo": p.te_combo,
                    "suppress_factor": round(p.super_boost, 3),
                    "n_observed": p.n_observed,
                }
                for p in suppressors
            ],
            "last_scan": datetime.fromtimestamp(
                self._last_scan_time
            ).isoformat() if self._last_scan_time > 0 else "never",
            "last_independence_check": datetime.fromtimestamp(
                self._last_independence_check
            ).isoformat() if self._last_independence_check > 0 else "never",
        }


# ============================================================
# TEQA INTEGRATION: Electric Organs Bridge
# ============================================================

class ElectricOrgansBridge:
    """
    Bridges the Electric Organs convergence engine with the TEQA pipeline.

    Usage in BRAIN scripts:

        from electric_organs import ElectricOrgansBridge

        eo_bridge = ElectricOrgansBridge()

        # During each TEQA cycle:
        domestic_boost = domestication_tracker.get_boost(active_tes)
        final_boost = eo_bridge.apply(active_tes, domestic_boost, symbol)
        # final_boost = domestication_boost * convergence_boost

    The bridge handles:
      - Lazy initialization of the convergence engine
      - Periodic convergence scans on a background timer
      - Cache management
      - Signal file output
    """

    def __init__(
        self,
        lineage_symbols: List[str] = None,
        db_dir: str = None,
    ):
        self.engine = ConvergentSignalEngine(
            lineage_symbols=lineage_symbols,
            db_dir=db_dir,
        )
        self._initialized = False

    def apply(
        self,
        active_tes: List[str],
        domestication_boost: float,
        symbol: str = "",
    ) -> float:
        """
        Apply convergence boost on top of domestication boost.

        This is the one-line integration point for BRAIN scripts.

        Args:
            active_tes: Active TE families from current signal.
            domestication_boost: Boost from TEDomesticationTracker.
            symbol: Current instrument.

        Returns:
            Final combined boost.
        """
        # Lazy scan on first call
        if not self._initialized:
            self.engine.convergence_scan()
            self._initialized = True

        # Periodic re-scan
        if self.engine.should_scan():
            self.engine.convergence_scan()

        return self.engine.apply_convergence_boost(
            active_tes, domestication_boost, symbol,
        )

    def check_loser(
        self,
        active_tes: List[str],
        symbol: str = "",
    ) -> float:
        """Check for convergent loser suppression."""
        return self.engine.check_convergent_loser(active_tes, symbol)

    def run_full_cycle(
        self,
        bars_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict:
        """Run a full convergence cycle with optional independence check."""
        return self.engine.run_cycle(bars_dict, force_scan=True)


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
    print("  ELECTRIC ORGANS ENGINE -- Convergent Signal Evolution")
    print("  Algorithm #4: Convergent Evolution of Trading Patterns")
    print("=" * 76)

    # -------------------------------------------------------
    # Create synthetic domestication DBs for 4 instruments
    # -------------------------------------------------------

    test_dir = Path(__file__).parent / "_test_electric_organs"
    test_dir.mkdir(exist_ok=True)

    # TE combos that will be "convergently domesticated"
    convergent_winners = [
        "Alu+CACTA+L1_Neuronal",          # Winner in 4/4 instruments
        "BEL_Pao+DIRS1+Ty3_gypsy",        # Winner in 3/4 instruments
    ]
    convergent_losers = [
        "hobo+Mutator+P_element",          # Loser in 4/4 instruments
    ]
    instrument_specific = [
        "Crypton+Helitron+SVA_Regulatory",  # Only domesticated in BTCUSD
    ]

    symbols = ["XAUUSD", "BTCUSD", "ETHUSD", "NAS100"]

    print(f"\n  Creating synthetic domestication DBs for: {symbols}")
    print(f"  Convergent winners: {convergent_winners}")
    print(f"  Convergent losers:  {convergent_losers}")
    print(f"  Instrument-specific: {instrument_specific}")

    for symbol in symbols:
        db_path = str(test_dir / f"teqa_domestication_{symbol}.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS domesticated_patterns (
                    pattern_hash TEXT PRIMARY KEY,
                    te_combo TEXT,
                    win_count INTEGER DEFAULT 0,
                    loss_count INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    domesticated INTEGER DEFAULT 0,
                    boost_factor REAL DEFAULT 1.0,
                    first_seen TEXT,
                    last_seen TEXT,
                    last_activated TEXT,
                    topology_hash TEXT DEFAULT '',
                    avg_win REAL DEFAULT 0.0,
                    avg_loss REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,
                    posterior_wr REAL DEFAULT 0.5,
                    total_win_pnl REAL DEFAULT 0.0,
                    total_loss_pnl REAL DEFAULT 0.0
                )
            """)

            now = datetime.now().isoformat()

            # Insert convergent winners (domesticated in all instruments)
            for combo in convergent_winners:
                if combo == convergent_winners[1] and symbol == "NAS100":
                    # Second winner only in 3/4
                    continue
                p_hash = hashlib.md5(combo.encode()).hexdigest()[:16]
                conn.execute("""
                    INSERT OR REPLACE INTO domesticated_patterns
                    (pattern_hash, te_combo, win_count, loss_count, win_rate,
                     domesticated, boost_factor, posterior_wr, profit_factor,
                     avg_win, avg_loss, first_seen, last_seen, last_activated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p_hash, combo,
                    30, 10, 0.75,       # 75% raw WR
                    1, 1.25,            # Domesticated
                    0.72,               # Bayesian posterior WR
                    2.1,                # Profit factor
                    1.50, 0.71,         # Avg win / avg loss
                    now, now, now,
                ))

            # Insert convergent losers (observed everywhere, domesticated nowhere)
            for combo in convergent_losers:
                p_hash = hashlib.md5(combo.encode()).hexdigest()[:16]
                conn.execute("""
                    INSERT OR REPLACE INTO domesticated_patterns
                    (pattern_hash, te_combo, win_count, loss_count, win_rate,
                     domesticated, boost_factor, posterior_wr, profit_factor,
                     avg_win, avg_loss, first_seen, last_seen, last_activated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p_hash, combo,
                    8, 22, 0.267,       # 27% raw WR
                    0, 1.0,             # NOT domesticated
                    0.40,               # Low posterior WR
                    0.5,                # Bad profit factor
                    0.30, 0.60,
                    now, now, now,
                ))

            # Insert instrument-specific pattern (only for BTCUSD)
            if symbol == "BTCUSD":
                for combo in instrument_specific:
                    p_hash = hashlib.md5(combo.encode()).hexdigest()[:16]
                    conn.execute("""
                        INSERT OR REPLACE INTO domesticated_patterns
                        (pattern_hash, te_combo, win_count, loss_count, win_rate,
                         domesticated, boost_factor, posterior_wr, profit_factor,
                         avg_win, avg_loss, first_seen, last_seen, last_activated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        p_hash, combo,
                        25, 5, 0.833,
                        1, 1.28,
                        0.76,
                        2.5,
                        2.00, 0.80,
                        now, now, now,
                    ))

            conn.commit()

    print("  Synthetic DBs created.\n")

    # -------------------------------------------------------
    # Initialize engine and run convergence scan
    # -------------------------------------------------------

    print("  --- PHASE 1: LINEAGE DISCOVERY ---")
    engine = ConvergentSignalEngine(
        lineage_symbols=symbols,
        db_dir=str(test_dir),
        convergence_db_path=str(test_dir / "test_convergence.db"),
    )
    print(f"  Discovered {len(engine.lineages)} lineages")
    for lineage in engine.lineages:
        print(f"    {lineage}")

    print("\n  --- PHASE 2: CONVERGENCE SCAN ---")
    patterns = engine.convergence_scan()
    print(f"  Found {len(patterns)} convergent patterns:")
    for p in patterns:
        status = "ELECTROCYTE" if p.is_electrocyte else ("SUPPRESSOR" if p.super_boost < 1.0 else "PARTIAL")
        print(
            f"    [{status:11s}] {p.te_combo:40s} | "
            f"conv={p.convergence_score:.2f} | "
            f"boost={p.super_boost:.3f} | "
            f"{p.n_instruments} instruments | "
            f"avg_wr={p.avg_win_rate:.3f}"
        )

    print("\n  --- PHASE 3: INDEPENDENCE VERIFICATION ---")
    # Create synthetic bars for independence check
    np.random.seed(42)
    n_bars = 200
    bars_dict = {}
    for i, symbol in enumerate(symbols):
        # Make BTCUSD and ETHUSD somewhat correlated
        base_returns = np.random.randn(n_bars) * 0.01
        if symbol == "ETHUSD":
            btc_returns = np.diff(np.log(bars_dict["BTCUSD"][:, 3] + 1))
            eth_noise = np.random.randn(n_bars - 1) * 0.01
            # 50% correlated with BTC
            close = 3000 + np.cumsum(
                np.concatenate([[0], 0.5 * btc_returns + 0.5 * eth_noise])
            ) * 100
        else:
            close = [1000 + i * 10000][0] + np.cumsum(base_returns) * 500
        close = np.maximum(close, 10)
        high = close + abs(np.random.randn(n_bars) * close * 0.003)
        low = close - abs(np.random.randn(n_bars) * close * 0.003)
        open_p = close + np.random.randn(n_bars) * close * 0.001
        volume = abs(np.random.randn(n_bars) * 100 + 500)
        bars_dict[symbol] = np.column_stack([open_p, high, low, close, volume])

    patterns = engine.verify_independence(patterns, bars_dict)
    for p in patterns:
        ind_str = "INDEPENDENT" if p.independence_ok else "CORRELATED"
        print(
            f"    {p.te_combo:40s} | {ind_str:11s} | boost={p.super_boost:.3f}"
        )

    print("\n  --- PHASE 4: SIGNAL APPLICATION ---")
    # Test convergence boost for a known convergent winner
    test_tes_winner = ["Alu", "CACTA", "L1_Neuronal"]
    signal = engine.get_convergence_boost(test_tes_winner, "XAUUSD")
    print(f"  Test TEs: {test_tes_winner}")
    print(f"    Convergence boost:  {signal.convergence_boost:.3f}")
    print(f"    Convergence score:  {signal.convergence_score:.3f}")
    print(f"    Is electrocyte:     {signal.is_electrocyte}")
    print(f"    Instruments:        {signal.n_instruments}")
    print(f"    Independent:        {signal.independence_verified}")

    # Combined with domestication
    mock_domestication_boost = 1.25
    final = engine.apply_convergence_boost(
        test_tes_winner, mock_domestication_boost, "XAUUSD"
    )
    print(f"    Domestication boost: {mock_domestication_boost:.3f}")
    print(f"    Final combined:      {final:.3f} "
          f"({mock_domestication_boost:.3f} x {signal.convergence_boost:.3f})")

    print("\n  Test TEs (convergent loser): hobo+Mutator+P_element")
    loser_suppress = engine.check_convergent_loser(
        ["hobo", "Mutator", "P_element"], "BTCUSD"
    )
    print(f"    Suppression factor: {loser_suppress:.3f}")

    print("\n  Test TEs (instrument-specific): Crypton+Helitron+SVA_Regulatory")
    specific_signal = engine.get_convergence_boost(
        ["Crypton", "Helitron", "SVA_Regulatory"], "BTCUSD"
    )
    print(f"    Convergence boost: {specific_signal.convergence_boost:.3f}")
    print(f"    (Expected 1.0 -- only in 1 instrument, no convergence)")

    print("\n  --- PHASE 5: BRIDGE INTEGRATION ---")
    bridge = ElectricOrgansBridge(
        lineage_symbols=symbols,
        db_dir=str(test_dir),
    )
    bridge.engine.convergence_db_path = str(test_dir / "test_convergence.db")
    bridge.engine._init_convergence_db()

    final = bridge.apply(test_tes_winner, 1.25, "XAUUSD")
    print(f"  Bridge.apply({test_tes_winner}, 1.25) = {final:.3f}")

    print("\n  --- SUMMARY ---")
    summary = engine.get_summary()
    print(f"  Lineages:            {summary['n_lineages']}")
    print(f"  Total patterns:      {summary['n_total_patterns']}")
    print(f"  Electrocytes:        {summary['n_electrocytes']}")
    print(f"  Partial convergent:  {summary['n_partial_convergent']}")
    print(f"  Convergent losers:   {summary['n_convergent_losers']}")

    # Write signal file
    engine.write_signal_file()
    print(f"\n  Signal file: {engine.signal_file}")

    # Cleanup
    import shutil
    try:
        shutil.rmtree(str(test_dir))
        print(f"  Cleaned up test dir: {test_dir}")
    except Exception:
        pass

    print("\n" + "=" * 76)
    print("  Electric Organs engine test complete.")
    print("  Convergent evolution is the strongest evidence against overfitting.")
    print("=" * 76)
