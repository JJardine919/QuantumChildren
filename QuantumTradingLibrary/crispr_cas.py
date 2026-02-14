"""
CRISPR-Cas ADAPTIVE IMMUNE MEMORY -- Algorithm #3
===================================================
Pattern-matching trade blocker that remembers exact market conditions
preceding losses and blocks future trades matching those patterns.

Biological basis:
    CRISPR-Cas9 is a ~3 billion-year-old bacterial immune system. When a
    bacteriophage attacks, the bacterium captures a snippet of viral DNA
    (a "spacer") and stores it in the CRISPR array. On re-infection, the
    spacer is transcribed into guide RNA that directs the Cas9 nuclease
    to cut matching viral DNA with surgical precision.

Trading translation:
    Spacer = exact market fingerprint captured after a losing trade
    CRISPR array = ordered memory of loss patterns (newest first)
    Guide RNA = fingerprint comparison against current conditions
    PAM sequence = broad context check (volatility regime + session)
    Cas9 cut = trade entry blocked
    Anti-CRISPR = domestication boost overriding the block
    Spacer decay = old unused spacers expire after N days

Integration:
    - Hooks into TradeOutcomePoller: on loss, acquire spacer
    - Hooks into TEQABridge: before trade entry, run Cas9 gate check
    - Reads domestication boost from TEDomesticationTracker for anti-CRISPR
    - Persists everything to SQLite (crispr_cas.db)
    - Becomes Gate G12 in the extended Jardine's Gate system

Authors: DooDoo + Claude
Date:    2026-02-08
Version: CRISPR-CAS-1.0
"""

import json
import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config_loader import CONFIDENCE_THRESHOLD, MAX_LOSS_DOLLARS

log = logging.getLogger(__name__)

VERSION = "CRISPR-CAS-1.0"

# ============================================================
# CONSTANTS
# ============================================================

# CRISPR array limits
CRISPR_ARRAY_MAX_SPACERS = 200          # Max active spacers per symbol
SPACER_DECAY_DAYS = 30                  # Expire spacers unused for this many days
SPACER_PURGE_DAYS = 90                  # Delete expired spacers older than this

# Spacer acquisition
SPACER_MIN_LOSS_DOLLARS = 0.0           # Minimum loss to trigger acquisition (0 = any loss)
ACQUISITION_LOOKBACK_BARS = 20          # Bars of price history captured in fingerprint
SPACER_DEDUP_SIMILARITY = 0.95          # If new spacer is this similar to existing, merge instead

# Guide RNA matching
GUIDE_RNA_MATCH_THRESHOLD = 0.80        # Cosine similarity threshold for Cas9 cut
MATCH_WEIGHT_RECENCY = 0.85            # Exponential decay for older spacers
MATCH_WEIGHT_LOSS_SEVERITY_CAP = 2.0   # Max multiplier from loss severity

# PAM sequence requirements
PAM_REGIME_REQUIRED = True              # Must match volatility regime
PAM_SESSION_REQUIRED = True             # Must match trading session

# Anti-CRISPR
ANTI_CRISPR_BOOST_THRESHOLD = 1.20     # Domestication boost above this overrides CRISPR

# Cas9 cooldown
CAS9_COOLDOWN_SECONDS = 300            # After a cut, cooldown before re-scanning same spacer

# Maintenance
MAINTENANCE_INTERVAL_CYCLES = 100      # Run maintenance every N trading cycles


# ============================================================
# MARKET FINGERPRINT (the "spacer DNA")
# ============================================================

@dataclass
class MarketFingerprint:
    """
    A snapshot of market conditions at the time of trade entry.
    This is the "spacer" -- the captured pattern that will be stored
    in the CRISPR array and matched against future conditions.

    The fingerprint has two parts:
      1. A numeric vector (for cosine similarity matching)
      2. Categorical context fields (for PAM matching)
    """
    # Numeric vector components
    price_returns: List[float]     # Normalized returns of last N bars
    atr_normalized: float          # Current ATR / 20-bar ATR mean
    rsi_14: float                  # RSI(14) normalized to [0, 1]
    bb_position: float             # Price position in BB (-1 to +1)
    momentum_10: float             # 10-bar momentum (normalized)
    volume_ratio: float            # Current volume / 20-bar average
    spread_normalized: float       # Spread / ATR (normalized)

    # Categorical context (PAM sequence)
    volatility_regime: str         # "LOW" | "MEDIUM" | "HIGH" | "EXTREME"
    session: str                   # "ASIAN" | "LONDON" | "NEWYORK" | "OVERLAP"
    hour_of_day: int               # 0-23 UTC
    day_of_week: int               # 0=Mon .. 4=Fri

    # Metadata
    symbol: str = ""
    direction: int = 0             # 1=LONG, -1=SHORT
    active_te_hash: str = ""       # MD5 of sorted active TEs
    active_tes: List[str] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Convert numeric components to a flat numpy vector for similarity matching."""
        components = list(self.price_returns)
        components.extend([
            self.atr_normalized,
            self.rsi_14,
            self.bb_position,
            self.momentum_10,
            self.volume_ratio,
            self.spread_normalized,
        ])
        return np.array(components, dtype=np.float64)

    def vector_id(self) -> str:
        """Compute a unique hash from the fingerprint vector."""
        vec = self.to_vector()
        raw = vec.tobytes()
        return hashlib.md5(raw).hexdigest()[:16]


# ============================================================
# FINGERPRINT COMPUTATION
# ============================================================

class FingerprintEngine:
    """
    Computes MarketFingerprint from raw bar data.
    This is the spacer acquisition machinery -- it captures the exact
    market conditions that the guide RNA will later match against.
    """

    @staticmethod
    def compute(
        bars: np.ndarray,
        spread: float = 0.0,
        active_tes: List[str] = None,
        direction: int = 0,
        symbol: str = "",
        hour_utc: int = None,
        day_of_week: int = None,
    ) -> MarketFingerprint:
        """
        Compute a MarketFingerprint from OHLCV bars.

        Args:
            bars: numpy array shape (N, 5) with [open, high, low, close, volume]
                  Minimum 21 bars required (20 for returns + 1 for current).
            spread: current spread in price units
            active_tes: list of active TE family names
            direction: trade direction (1=LONG, -1=SHORT)
            symbol: trading symbol
            hour_utc: hour in UTC (0-23). If None, uses current time.
            day_of_week: 0=Monday..4=Friday. If None, uses current day.

        Returns:
            MarketFingerprint
        """
        if active_tes is None:
            active_tes = []

        n = len(bars)
        lookback = min(ACQUISITION_LOOKBACK_BARS, n - 1)

        close = bars[:, 3]
        high = bars[:, 1]
        low = bars[:, 2]
        open_p = bars[:, 0]
        volume = bars[:, 4] if bars.shape[1] > 4 else np.ones(n)

        # -- Price returns (normalized) --
        # Take the last `lookback` returns, normalize to zero-mean unit-variance
        if n > 1:
            raw_returns = np.diff(close[-(lookback + 1):]) / (close[-(lookback + 1):-1] + 1e-10)
            if len(raw_returns) < lookback:
                # Pad with zeros if not enough bars
                raw_returns = np.pad(raw_returns, (lookback - len(raw_returns), 0))
            std_ret = np.std(raw_returns)
            if std_ret > 1e-10:
                price_returns = ((raw_returns - np.mean(raw_returns)) / std_ret).tolist()
            else:
                price_returns = raw_returns.tolist()
        else:
            price_returns = [0.0] * lookback

        # -- ATR normalized --
        atr_vals = FingerprintEngine._compute_atr_series(high, low, close, 14)
        current_atr = atr_vals[-1] if len(atr_vals) > 0 else 0.0
        mean_atr_20 = np.mean(atr_vals[-20:]) if len(atr_vals) >= 20 else (
            np.mean(atr_vals) if len(atr_vals) > 0 else 1e-10
        )
        atr_normalized = current_atr / (mean_atr_20 + 1e-10)

        # -- Volatility regime --
        volatility_regime = FingerprintEngine._classify_volatility(atr_normalized)

        # -- RSI(14) normalized to [0, 1] --
        rsi_raw = FingerprintEngine._rsi(close, 14)
        rsi_14 = rsi_raw / 100.0  # Normalize to [0, 1]

        # -- Bollinger Band position --
        bb_period = min(20, n)
        if bb_period >= 2:
            bb_sma = np.mean(close[-bb_period:])
            bb_std = np.std(close[-bb_period:])
            if bb_std > 1e-10:
                bb_position = (close[-1] - bb_sma) / (2.0 * bb_std)
                bb_position = max(-1.0, min(1.0, bb_position))
            else:
                bb_position = 0.0
        else:
            bb_position = 0.0

        # -- 10-bar momentum (normalized by ATR) --
        if n >= 11:
            mom_raw = (close[-1] - close[-11])
            momentum_10 = mom_raw / (current_atr + 1e-10)
            momentum_10 = max(-5.0, min(5.0, momentum_10))  # Clip extremes
        else:
            momentum_10 = 0.0

        # -- Volume ratio --
        if n >= 20:
            avg_vol = np.mean(volume[-20:])
            volume_ratio = volume[-1] / (avg_vol + 1e-10)
            volume_ratio = min(5.0, volume_ratio)  # Clip extremes
        else:
            volume_ratio = 1.0

        # -- Spread normalized --
        spread_normalized = spread / (current_atr + 1e-10) if current_atr > 1e-10 else 0.0

        # -- Session classification --
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        h = hour_utc if hour_utc is not None else now_utc.hour
        dow = day_of_week if day_of_week is not None else now_utc.weekday()
        session = FingerprintEngine._classify_session(h)

        # -- TE hash --
        te_hash = ""
        if active_tes:
            combo = "+".join(sorted(active_tes))
            te_hash = hashlib.md5(combo.encode()).hexdigest()[:8]

        return MarketFingerprint(
            price_returns=price_returns,
            atr_normalized=float(atr_normalized),
            rsi_14=float(rsi_14),
            bb_position=float(bb_position),
            momentum_10=float(momentum_10),
            volume_ratio=float(volume_ratio),
            spread_normalized=float(spread_normalized),
            volatility_regime=volatility_regime,
            session=session,
            hour_of_day=h,
            day_of_week=dow,
            symbol=symbol,
            direction=direction,
            active_te_hash=te_hash,
            active_tes=list(active_tes),
        )

    @staticmethod
    def _classify_volatility(atr_ratio: float) -> str:
        """Classify volatility regime from ATR ratio."""
        if atr_ratio < 0.7:
            return "LOW"
        elif atr_ratio < 1.3:
            return "MEDIUM"
        elif atr_ratio < 2.0:
            return "HIGH"
        else:
            return "EXTREME"

    @staticmethod
    def _classify_session(hour_utc: int) -> str:
        """Classify trading session from UTC hour."""
        # Approximate session times (UTC):
        # Asian:   00:00 - 08:00
        # London:  08:00 - 12:00
        # Overlap: 12:00 - 17:00
        # NewYork: 17:00 - 22:00
        # Off:     22:00 - 00:00 (mapped to ASIAN for simplicity)
        if 0 <= hour_utc < 8:
            return "ASIAN"
        elif 8 <= hour_utc < 12:
            return "LONDON"
        elif 12 <= hour_utc < 17:
            return "OVERLAP"
        elif 17 <= hour_utc < 22:
            return "NEWYORK"
        else:
            return "ASIAN"

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> float:
        """Compute RSI value for the most recent bar."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _compute_atr_series(high, low, close, period=14) -> np.ndarray:
        """Compute ATR series."""
        n = len(close)
        if n < 2:
            return np.array([0.0])
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


# ============================================================
# CRISPR ARRAY (SQLite-backed spacer storage)
# ============================================================

class CRISPRArray:
    """
    The CRISPR array: an ordered, persistent collection of spacers.

    Each spacer is a captured market fingerprint from a losing trade.
    Spacers are ordered by acquisition time (newest first, like the
    biological CRISPR array where new spacers are added at the leader end).

    Backed by SQLite for persistence across restarts.
    """

    def __init__(self, db_path: str = None, max_spacers: int = CRISPR_ARRAY_MAX_SPACERS):
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "crispr_cas.db")
        else:
            self.db_path = db_path
        self.max_spacers = max_spacers
        self._init_db()

    def _init_db(self):
        """Initialize the CRISPR database with all required tables."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")

                # Spacers table -- the core CRISPR array
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS spacers (
                        spacer_id TEXT PRIMARY KEY,
                        fingerprint_json TEXT NOT NULL,
                        fingerprint_vector BLOB NOT NULL,
                        symbol TEXT NOT NULL,
                        direction INTEGER NOT NULL,
                        loss_amount REAL DEFAULT 0.0,
                        active_tes TEXT DEFAULT '[]',
                        volatility_regime TEXT DEFAULT 'MEDIUM',
                        session TEXT DEFAULT 'OVERLAP',
                        hour_of_day INTEGER DEFAULT 12,
                        day_of_week INTEGER DEFAULT 2,
                        acquired_at TEXT NOT NULL,
                        last_matched TEXT,
                        match_count INTEGER DEFAULT 0,
                        expired INTEGER DEFAULT 0,
                        merge_count INTEGER DEFAULT 1
                    )
                """)

                # Indexes for fast lookup
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_spacers_symbol_dir_active
                    ON spacers (symbol, direction, expired)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_spacers_acquired
                    ON spacers (acquired_at DESC)
                """)

                # Cas9 events table -- log of every blocked trade
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cas9_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        direction INTEGER NOT NULL,
                        spacer_id TEXT NOT NULL,
                        match_score REAL NOT NULL,
                        blocked_confidence REAL DEFAULT 0.0,
                        volatility_regime TEXT,
                        session TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cas9_symbol_ts
                    ON cas9_events (symbol, timestamp)
                """)

                # Anti-CRISPR events table -- log of every override
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS anti_crispr_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        direction INTEGER NOT NULL,
                        spacer_id TEXT NOT NULL,
                        match_score REAL NOT NULL,
                        domestication_boost REAL NOT NULL,
                        reason TEXT
                    )
                """)

                conn.commit()
                log.info("[CRISPR] Database initialized: %s", self.db_path)

        except Exception as e:
            log.warning("[CRISPR] DB init failed: %s", e)

    # ----------------------------------------------------------
    # SPACER OPERATIONS
    # ----------------------------------------------------------

    def insert_spacer(
        self,
        fingerprint: MarketFingerprint,
        loss_amount: float,
    ) -> str:
        """
        Insert a new spacer into the CRISPR array (at the leader end).

        If a very similar spacer already exists (cosine similarity > SPACER_DEDUP_SIMILARITY),
        merge instead of creating a duplicate.

        Returns the spacer_id of the inserted or merged spacer.
        """
        spacer_id = fingerprint.vector_id()
        vec = fingerprint.to_vector()
        vec_bytes = vec.tobytes()
        now = datetime.now().isoformat()

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()

                # Check for exact duplicate
                cursor.execute(
                    "SELECT spacer_id, match_count, loss_amount, merge_count "
                    "FROM spacers WHERE spacer_id = ?",
                    (spacer_id,)
                )
                existing = cursor.fetchone()

                if existing:
                    # Exact duplicate: reinforce the spacer
                    new_loss_avg = (existing[2] * existing[3] + loss_amount) / (existing[3] + 1)
                    cursor.execute("""
                        UPDATE spacers
                        SET match_count = match_count + 1,
                            loss_amount = ?,
                            last_matched = ?,
                            merge_count = merge_count + 1,
                            expired = 0
                        WHERE spacer_id = ?
                    """, (new_loss_avg, now, spacer_id))
                    conn.commit()
                    log.info(
                        "[CRISPR] Reinforced existing spacer %s (%s %s) merge_count=%d",
                        spacer_id[:8], fingerprint.symbol,
                        fingerprint.volatility_regime, existing[3] + 1,
                    )
                    return spacer_id

                # Check for near-duplicate (cosine similarity > threshold)
                near_dup = self._find_near_duplicate(
                    conn, vec, fingerprint.symbol, fingerprint.direction,
                )
                if near_dup is not None:
                    # Merge into the existing similar spacer
                    dup_id, dup_loss, dup_merge = near_dup
                    new_loss_avg = (dup_loss * dup_merge + loss_amount) / (dup_merge + 1)
                    cursor.execute("""
                        UPDATE spacers
                        SET loss_amount = ?,
                            last_matched = ?,
                            merge_count = merge_count + 1,
                            expired = 0
                        WHERE spacer_id = ?
                    """, (new_loss_avg, now, dup_id))
                    conn.commit()
                    log.info(
                        "[CRISPR] Merged new spacer into existing %s (near-duplicate)",
                        dup_id[:8],
                    )
                    return dup_id

                # Insert new spacer
                fp_json = json.dumps({
                    "price_returns": fingerprint.price_returns,
                    "atr_normalized": fingerprint.atr_normalized,
                    "rsi_14": fingerprint.rsi_14,
                    "bb_position": fingerprint.bb_position,
                    "momentum_10": fingerprint.momentum_10,
                    "volume_ratio": fingerprint.volume_ratio,
                    "spread_normalized": fingerprint.spread_normalized,
                    "volatility_regime": fingerprint.volatility_regime,
                    "session": fingerprint.session,
                    "hour_of_day": fingerprint.hour_of_day,
                    "day_of_week": fingerprint.day_of_week,
                    "active_te_hash": fingerprint.active_te_hash,
                }, sort_keys=True)

                cursor.execute("""
                    INSERT INTO spacers (
                        spacer_id, fingerprint_json, fingerprint_vector,
                        symbol, direction, loss_amount, active_tes,
                        volatility_regime, session, hour_of_day, day_of_week,
                        acquired_at, last_matched, match_count, expired, merge_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 0, 1)
                """, (
                    spacer_id, fp_json, vec_bytes,
                    fingerprint.symbol, fingerprint.direction, loss_amount,
                    json.dumps(fingerprint.active_tes),
                    fingerprint.volatility_regime, fingerprint.session,
                    fingerprint.hour_of_day, fingerprint.day_of_week,
                    now,
                ))
                conn.commit()

                # Enforce array size limit for this symbol
                self._enforce_array_limit(conn, fingerprint.symbol)

                log.info(
                    "[CRISPR] Acquired spacer %s | %s %s | loss=$%.2f | "
                    "regime=%s session=%s | TEs=%s",
                    spacer_id[:8], fingerprint.symbol,
                    "LONG" if fingerprint.direction > 0 else "SHORT",
                    loss_amount, fingerprint.volatility_regime,
                    fingerprint.session, fingerprint.active_te_hash[:6],
                )
                return spacer_id

        except Exception as e:
            log.warning("[CRISPR] Spacer insertion failed: %s", e)
            return ""

    def _find_near_duplicate(
        self, conn, vec: np.ndarray, symbol: str, direction: int,
    ) -> Optional[Tuple[str, float, int]]:
        """
        Check if any existing active spacer is very similar to the new one.
        Returns (spacer_id, loss_amount, merge_count) or None.
        """
        cursor = conn.cursor()
        cursor.execute(
            "SELECT spacer_id, fingerprint_vector, loss_amount, merge_count "
            "FROM spacers "
            "WHERE symbol = ? AND direction = ? AND expired = 0 "
            "ORDER BY acquired_at DESC LIMIT 50",
            (symbol, direction),
        )
        rows = cursor.fetchall()

        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-10:
            return None

        for row in rows:
            existing_vec = np.frombuffer(row[1], dtype=np.float64)
            if len(existing_vec) != len(vec):
                continue
            existing_norm = np.linalg.norm(existing_vec)
            if existing_norm < 1e-10:
                continue
            sim = float(np.dot(vec, existing_vec) / (vec_norm * existing_norm))
            if sim >= SPACER_DEDUP_SIMILARITY:
                return (row[0], row[2], row[3])

        return None

    def get_active_spacers(
        self, symbol: str, direction: int,
    ) -> List[Tuple[str, np.ndarray, float, str, str, int]]:
        """
        Get all active (non-expired) spacers for a symbol + direction.

        Returns list of tuples:
            (spacer_id, fingerprint_vector, loss_amount,
             volatility_regime, session, match_count)

        Ordered by acquired_at DESC (newest first, like the biological array).
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT spacer_id, fingerprint_vector, loss_amount,
                           volatility_regime, session, match_count
                    FROM spacers
                    WHERE symbol = ? AND direction = ? AND expired = 0
                    ORDER BY acquired_at DESC
                """, (symbol, direction))

                results = []
                for row in cursor.fetchall():
                    vec = np.frombuffer(row[1], dtype=np.float64)
                    results.append((
                        row[0],   # spacer_id
                        vec,      # fingerprint_vector
                        row[2],   # loss_amount
                        row[3],   # volatility_regime
                        row[4],   # session
                        row[5],   # match_count
                    ))
                return results

        except Exception as e:
            log.warning("[CRISPR] Failed to load spacers: %s", e)
            return []

    def record_cas9_event(
        self, spacer_id: str, symbol: str, direction: int,
        match_score: float, confidence: float,
        volatility_regime: str, session: str,
    ):
        """Log a Cas9 cut event and update the spacer's match count."""
        now = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")

                # Update the spacer
                conn.execute("""
                    UPDATE spacers
                    SET match_count = match_count + 1,
                        last_matched = ?
                    WHERE spacer_id = ?
                """, (now, spacer_id))

                # Log the event
                conn.execute("""
                    INSERT INTO cas9_events
                    (timestamp, symbol, direction, spacer_id, match_score,
                     blocked_confidence, volatility_regime, session)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (now, symbol, direction, spacer_id, match_score,
                      confidence, volatility_regime, session))

                conn.commit()

        except Exception as e:
            log.warning("[CRISPR] Failed to record Cas9 event: %s", e)

    def record_anti_crispr_event(
        self, spacer_id: str, symbol: str, direction: int,
        match_score: float, domestication_boost: float, reason: str,
    ):
        """Log an anti-CRISPR override event."""
        now = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    INSERT INTO anti_crispr_events
                    (timestamp, symbol, direction, spacer_id, match_score,
                     domestication_boost, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (now, symbol, direction, spacer_id, match_score,
                      domestication_boost, reason))
                conn.commit()

        except Exception as e:
            log.warning("[CRISPR] Failed to record anti-CRISPR event: %s", e)

    def _enforce_array_limit(self, conn, symbol: str):
        """Expire oldest spacers if array exceeds max size for a symbol."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM spacers WHERE symbol = ? AND expired = 0",
            (symbol,)
        )
        count = cursor.fetchone()[0]

        if count > self.max_spacers:
            excess = count - self.max_spacers
            cursor.execute("""
                UPDATE spacers SET expired = 1
                WHERE spacer_id IN (
                    SELECT spacer_id FROM spacers
                    WHERE symbol = ? AND expired = 0
                    ORDER BY acquired_at ASC
                    LIMIT ?
                )
            """, (symbol, excess))
            conn.commit()
            log.info(
                "[CRISPR] Array limit enforced for %s: expired %d oldest spacers",
                symbol, excess,
            )

    def run_maintenance(self):
        """
        Periodic maintenance: expire stale spacers and purge ancient ones.

        Stale: active spacers that haven't matched anything in SPACER_DECAY_DAYS.
        Ancient: expired spacers older than SPACER_PURGE_DAYS.
        """
        now = datetime.now()
        decay_cutoff = (now - timedelta(days=SPACER_DECAY_DAYS)).isoformat()
        purge_cutoff = (now - timedelta(days=SPACER_PURGE_DAYS)).isoformat()

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()

                # Expire stale spacers (never matched, or last match too old)
                cursor.execute("""
                    UPDATE spacers SET expired = 1
                    WHERE expired = 0
                      AND acquired_at < ?
                      AND (last_matched IS NULL OR last_matched < ?)
                """, (decay_cutoff, decay_cutoff))
                expired_count = cursor.rowcount

                # Purge ancient expired spacers
                cursor.execute("""
                    DELETE FROM spacers
                    WHERE expired = 1 AND acquired_at < ?
                """, (purge_cutoff,))
                purged_count = cursor.rowcount

                conn.commit()

                if expired_count > 0 or purged_count > 0:
                    log.info(
                        "[CRISPR] Maintenance: expired %d stale spacers, "
                        "purged %d ancient spacers",
                        expired_count, purged_count,
                    )

                # Report array health per symbol
                cursor.execute("""
                    SELECT symbol,
                           COUNT(*) as active_count,
                           COALESCE(SUM(match_count), 0) as total_cuts
                    FROM spacers
                    WHERE expired = 0
                    GROUP BY symbol
                """)
                for row in cursor.fetchall():
                    log.info(
                        "[CRISPR] Array [%s]: %d active spacers, %d total Cas9 cuts",
                        row[0], row[1], row[2],
                    )

        except Exception as e:
            log.warning("[CRISPR] Maintenance failed: %s", e)

    def get_stats(self, symbol: str = None) -> Dict:
        """Get CRISPR array statistics for dashboard/monitoring."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()

                where = "WHERE symbol = ?" if symbol else ""
                params = (symbol,) if symbol else ()

                # Active spacers
                cursor.execute(
                    f"SELECT COUNT(*) FROM spacers WHERE expired = 0 "
                    + (f"AND symbol = ?" if symbol else ""),
                    params
                )
                active_count = cursor.fetchone()[0]

                # Total spacers (including expired)
                cursor.execute(
                    f"SELECT COUNT(*) FROM spacers "
                    + (f"WHERE symbol = ?" if symbol else ""),
                    params
                )
                total_count = cursor.fetchone()[0]

                # Total Cas9 cuts
                cursor.execute(
                    f"SELECT COUNT(*) FROM cas9_events "
                    + (f"WHERE symbol = ?" if symbol else ""),
                    params
                )
                cas9_count = cursor.fetchone()[0]

                # Anti-CRISPR overrides
                cursor.execute(
                    f"SELECT COUNT(*) FROM anti_crispr_events "
                    + (f"WHERE symbol = ?" if symbol else ""),
                    params
                )
                override_count = cursor.fetchone()[0]

                # Per-symbol breakdown
                cursor.execute("""
                    SELECT symbol,
                           SUM(CASE WHEN expired = 0 THEN 1 ELSE 0 END) as active,
                           SUM(CASE WHEN expired = 1 THEN 1 ELSE 0 END) as expired,
                           COALESCE(SUM(match_count), 0) as total_matches
                    FROM spacers
                    GROUP BY symbol
                    ORDER BY symbol
                """)
                per_symbol = [
                    {
                        "symbol": row[0],
                        "active_spacers": row[1],
                        "expired_spacers": row[2],
                        "total_matches": row[3],
                    }
                    for row in cursor.fetchall()
                ]

                return {
                    "active_spacers": active_count,
                    "total_spacers": total_count,
                    "cas9_cuts": cas9_count,
                    "anti_crispr_overrides": override_count,
                    "per_symbol": per_symbol,
                }

        except Exception as e:
            log.warning("[CRISPR] Stats query failed: %s", e)
            return {
                "active_spacers": 0,
                "total_spacers": 0,
                "cas9_cuts": 0,
                "anti_crispr_overrides": 0,
                "per_symbol": [],
            }


# ============================================================
# GUIDE RNA MATCHER (similarity scoring)
# ============================================================

class GuideRNAMatcher:
    """
    The guide RNA matching engine.

    In biology: crRNA (CRISPR RNA) transcribed from a spacer forms a
    complex with Cas9 and scans incoming DNA for complementary sequences.

    In trading: the matcher computes cosine similarity between the current
    market fingerprint and each stored spacer, applying recency weighting
    and loss severity scaling.
    """

    def __init__(
        self,
        match_threshold: float = GUIDE_RNA_MATCH_THRESHOLD,
        recency_weight: float = MATCH_WEIGHT_RECENCY,
        loss_severity_cap: float = MATCH_WEIGHT_LOSS_SEVERITY_CAP,
        pam_regime_required: bool = PAM_REGIME_REQUIRED,
        pam_session_required: bool = PAM_SESSION_REQUIRED,
    ):
        self.match_threshold = match_threshold
        self.recency_weight = recency_weight
        self.loss_severity_cap = loss_severity_cap
        self.pam_regime_required = pam_regime_required
        self.pam_session_required = pam_session_required

    def scan(
        self,
        current_fp: MarketFingerprint,
        spacers: List[Tuple[str, np.ndarray, float, str, str, int]],
    ) -> Dict:
        """
        Scan current market conditions against all spacers.

        This is the guide RNA scanning the genome: for each spacer,
        check PAM first, then compute similarity.

        Args:
            current_fp: current market fingerprint (the "protospacer")
            spacers: list of (spacer_id, vector, loss_amount,
                     volatility_regime, session, match_count)
                     from CRISPRArray.get_active_spacers()

        Returns:
            {
                "blocked": bool,
                "best_score": float,
                "best_spacer_id": str or None,
                "pam_filtered": int,       # spacers skipped due to PAM
                "scanned": int,            # spacers actually compared
                "reason": str,
            }
        """
        if not spacers:
            return {
                "blocked": False,
                "best_score": 0.0,
                "best_spacer_id": None,
                "pam_filtered": 0,
                "scanned": 0,
                "reason": "no spacers in array",
            }

        current_vec = current_fp.to_vector()
        current_norm = np.linalg.norm(current_vec)

        if current_norm < 1e-10:
            return {
                "blocked": False,
                "best_score": 0.0,
                "best_spacer_id": None,
                "pam_filtered": 0,
                "scanned": 0,
                "reason": "current fingerprint is zero-vector",
            }

        best_score = 0.0
        best_spacer_id = None
        pam_filtered = 0
        scanned = 0

        for i, (spacer_id, spacer_vec, loss_amount,
                vol_regime, sess, match_count) in enumerate(spacers):

            # PAM CHECK -- broad context must match first
            if self.pam_regime_required:
                if vol_regime != current_fp.volatility_regime:
                    pam_filtered += 1
                    continue

            if self.pam_session_required:
                if sess != current_fp.session:
                    pam_filtered += 1
                    continue

            # Vector length check
            if len(spacer_vec) != len(current_vec):
                continue

            spacer_norm = np.linalg.norm(spacer_vec)
            if spacer_norm < 1e-10:
                continue

            scanned += 1

            # GUIDE RNA MATCHING -- cosine similarity
            raw_similarity = float(
                np.dot(current_vec, spacer_vec) / (current_norm * spacer_norm)
            )

            # Recency weighting: newer spacers (lower index) are weighted higher
            recency_factor = self.recency_weight ** i

            # Loss severity weighting: bigger losses make spacers more sensitive
            loss_weight = min(
                self.loss_severity_cap,
                1.0 + loss_amount / (MAX_LOSS_DOLLARS + 1e-10),
            )

            # Composite score
            weighted_score = raw_similarity * recency_factor * loss_weight

            if weighted_score > best_score:
                best_score = weighted_score
                best_spacer_id = spacer_id

        # Cas9 decision
        blocked = best_score >= self.match_threshold and best_spacer_id is not None

        if blocked:
            reason = (
                f"Cas9 CUT: spacer {best_spacer_id[:8]} "
                f"score={best_score:.3f} >= threshold={self.match_threshold}"
            )
        else:
            reason = (
                f"no match above threshold "
                f"(best={best_score:.3f}, threshold={self.match_threshold})"
            )

        return {
            "blocked": blocked,
            "best_score": float(best_score),
            "best_spacer_id": best_spacer_id,
            "pam_filtered": pam_filtered,
            "scanned": scanned,
            "reason": reason,
        }


# ============================================================
# CAS9 GATE (trade blocking decision engine)
# ============================================================

class Cas9Gate:
    """
    The Cas9 nuclease gate: decides whether to block (cut) a trade.

    This integrates:
      1. Guide RNA matching (similarity check)
      2. PAM verification (context check)
      3. Anti-CRISPR override (domestication boost check)

    Used as Gate G12 in the Jardine's Gate system.
    """

    def __init__(
        self,
        crispr_array: CRISPRArray,
        matcher: GuideRNAMatcher = None,
        anti_crispr_threshold: float = ANTI_CRISPR_BOOST_THRESHOLD,
    ):
        self.array = crispr_array
        self.matcher = matcher or GuideRNAMatcher()
        self.anti_crispr_threshold = anti_crispr_threshold

        # Cooldown tracking: spacer_id -> last_cut_timestamp
        self._cooldowns: Dict[str, float] = {}

    def check(
        self,
        symbol: str,
        direction: int,
        bars: np.ndarray,
        spread: float = 0.0,
        active_tes: List[str] = None,
        domestication_boost: float = 1.0,
        confidence: float = 0.0,
        hour_utc: int = None,
        day_of_week: int = None,
    ) -> Dict:
        """
        Run the full Cas9 gate check.

        Args:
            symbol: trading symbol
            direction: 1=LONG, -1=SHORT
            bars: OHLCV numpy array (N x 5)
            spread: current spread in price units
            active_tes: list of active TE family names
            domestication_boost: current boost from TEDomesticationTracker
            confidence: signal confidence (for logging)
            hour_utc: override hour (for testing)
            day_of_week: override day (for testing)

        Returns:
            {
                "gate_pass": bool,
                "blocked": bool,
                "match_score": float,
                "spacer_id": str or None,
                "anti_crispr_override": bool,
                "reason": str,
                "pam_filtered": int,
                "scanned": int,
            }
        """
        if active_tes is None:
            active_tes = []

        # Step 1: Compute current market fingerprint
        fp = FingerprintEngine.compute(
            bars=bars,
            spread=spread,
            active_tes=active_tes,
            direction=direction,
            symbol=symbol,
            hour_utc=hour_utc,
            day_of_week=day_of_week,
        )

        # Step 2: Load spacers for this symbol + direction
        spacers = self.array.get_active_spacers(symbol, direction)

        if not spacers:
            return {
                "gate_pass": True,
                "blocked": False,
                "match_score": 0.0,
                "spacer_id": None,
                "anti_crispr_override": False,
                "reason": "no spacers for this symbol/direction",
                "pam_filtered": 0,
                "scanned": 0,
            }

        # Step 3: Run guide RNA matching
        match_result = self.matcher.scan(fp, spacers)

        if not match_result["blocked"]:
            return {
                "gate_pass": True,
                "blocked": False,
                "match_score": match_result["best_score"],
                "spacer_id": match_result["best_spacer_id"],
                "anti_crispr_override": False,
                "reason": match_result["reason"],
                "pam_filtered": match_result["pam_filtered"],
                "scanned": match_result["scanned"],
            }

        spacer_id = match_result["best_spacer_id"]
        match_score = match_result["best_score"]

        # Step 4: Check cooldown (don't re-cut same spacer too frequently)
        if spacer_id in self._cooldowns:
            elapsed = time.time() - self._cooldowns[spacer_id]
            if elapsed < CAS9_COOLDOWN_SECONDS:
                # Still in cooldown -- pass through
                return {
                    "gate_pass": True,
                    "blocked": False,
                    "match_score": match_score,
                    "spacer_id": spacer_id,
                    "anti_crispr_override": False,
                    "reason": f"Cas9 cooldown ({CAS9_COOLDOWN_SECONDS - elapsed:.0f}s remaining)",
                    "pam_filtered": match_result["pam_filtered"],
                    "scanned": match_result["scanned"],
                }

        # Step 5: Anti-CRISPR check
        if domestication_boost >= self.anti_crispr_threshold:
            log.info(
                "[CRISPR] Anti-CRISPR override for %s %s | "
                "domestication_boost=%.2f >= threshold=%.2f | "
                "spacer=%s score=%.3f",
                symbol, "LONG" if direction > 0 else "SHORT",
                domestication_boost, self.anti_crispr_threshold,
                spacer_id[:8], match_score,
            )
            self.array.record_anti_crispr_event(
                spacer_id=spacer_id,
                symbol=symbol,
                direction=direction,
                match_score=match_score,
                domestication_boost=domestication_boost,
                reason=(
                    f"domestication boost {domestication_boost:.2f} "
                    f">= threshold {self.anti_crispr_threshold}"
                ),
            )
            return {
                "gate_pass": True,
                "blocked": True,  # Was matched, but overridden
                "match_score": match_score,
                "spacer_id": spacer_id,
                "anti_crispr_override": True,
                "reason": (
                    f"Anti-CRISPR override: domestication {domestication_boost:.2f} "
                    f">= {self.anti_crispr_threshold}"
                ),
                "pam_filtered": match_result["pam_filtered"],
                "scanned": match_result["scanned"],
            }

        # Step 6: Cas9 CUT confirmed -- block the trade
        log.warning(
            "[CRISPR] Cas9 CUT! Blocked %s %s | spacer=%s score=%.3f | "
            "regime=%s session=%s | confidence_was=%.3f",
            symbol, "LONG" if direction > 0 else "SHORT",
            spacer_id[:8], match_score,
            fp.volatility_regime, fp.session, confidence,
        )

        # Record the event
        self.array.record_cas9_event(
            spacer_id=spacer_id,
            symbol=symbol,
            direction=direction,
            match_score=match_score,
            confidence=confidence,
            volatility_regime=fp.volatility_regime,
            session=fp.session,
        )

        # Set cooldown
        self._cooldowns[spacer_id] = time.time()

        # Prune old cooldowns
        cutoff = time.time() - CAS9_COOLDOWN_SECONDS * 2
        self._cooldowns = {
            k: v for k, v in self._cooldowns.items() if v > cutoff
        }

        return {
            "gate_pass": False,
            "blocked": True,
            "match_score": match_score,
            "spacer_id": spacer_id,
            "anti_crispr_override": False,
            "reason": match_result["reason"],
            "pam_filtered": match_result["pam_filtered"],
            "scanned": match_result["scanned"],
        }


# ============================================================
# SPACER ACQUISITION ENGINE
# ============================================================

class SpacerAcquisition:
    """
    Handles spacer acquisition when a trade loses.

    Called by the TradeOutcomePoller when a losing trade is detected.
    Captures the market fingerprint at the time of entry and stores
    it in the CRISPR array.
    """

    def __init__(
        self,
        crispr_array: CRISPRArray,
        min_loss: float = SPACER_MIN_LOSS_DOLLARS,
    ):
        self.array = crispr_array
        self.min_loss = min_loss

    def acquire(
        self,
        bars: np.ndarray,
        symbol: str,
        direction: int,
        loss_amount: float,
        active_tes: List[str] = None,
        spread: float = 0.0,
        hour_utc: int = None,
        day_of_week: int = None,
    ) -> Optional[str]:
        """
        Capture a spacer from a losing trade.

        Args:
            bars: OHLCV bars at the time of trade entry (lookback window)
            symbol: trading symbol
            direction: trade direction that lost (1=LONG, -1=SHORT)
            loss_amount: absolute loss amount in dollars
            active_tes: TE families that were active at entry time
            spread: spread at entry time
            hour_utc: hour in UTC at entry time
            day_of_week: day of week at entry time

        Returns:
            spacer_id if acquired, None if skipped.
        """
        if active_tes is None:
            active_tes = []

        # Check minimum loss threshold
        if loss_amount < self.min_loss:
            log.debug(
                "[CRISPR] Skipping spacer acquisition: loss $%.2f < min $%.2f",
                loss_amount, self.min_loss,
            )
            return None

        # Compute fingerprint
        fp = FingerprintEngine.compute(
            bars=bars,
            spread=spread,
            active_tes=active_tes,
            direction=direction,
            symbol=symbol,
            hour_utc=hour_utc,
            day_of_week=day_of_week,
        )

        # Insert into CRISPR array
        spacer_id = self.array.insert_spacer(fp, loss_amount)
        return spacer_id if spacer_id else None


# ============================================================
# TEQA BRIDGE INTEGRATION
# ============================================================

class CRISPRTEQABridge:
    """
    Bridges the CRISPR-Cas system with the existing TEQA pipeline.

    This is the integration point that connects CRISPR to:
      1. The BRAIN trading loop (gate check before entry)
      2. The TradeOutcomePoller (spacer acquisition on loss)
      3. The TEDomesticationTracker (anti-CRISPR via boost)

    Usage in BRAIN scripts:
        from crispr_cas import CRISPRTEQABridge

        crispr_bridge = CRISPRTEQABridge()

        # Before trade entry:
        cas9_result = crispr_bridge.gate_check(
            symbol='BTCUSD',
            direction=1,
            bars=bars,
            spread=current_spread,
            active_tes=['L1_Neuronal', 'Alu_Exonization'],
            domestication_boost=1.15,
            confidence=0.65,
        )
        if not cas9_result['gate_pass']:
            # BLOCKED by CRISPR -- do not trade
            log.warning("Trade blocked: %s", cas9_result['reason'])

        # After trade loss (in feedback loop):
        crispr_bridge.on_trade_loss(
            bars=entry_bars,
            symbol='BTCUSD',
            direction=1,
            loss_amount=0.85,
            active_tes=['L1_Neuronal', 'Alu_Exonization'],
            spread=entry_spread,
        )
    """

    def __init__(
        self,
        db_path: str = None,
        max_spacers: int = CRISPR_ARRAY_MAX_SPACERS,
        match_threshold: float = GUIDE_RNA_MATCH_THRESHOLD,
        anti_crispr_threshold: float = ANTI_CRISPR_BOOST_THRESHOLD,
    ):
        self.array = CRISPRArray(db_path=db_path, max_spacers=max_spacers)
        self.matcher = GuideRNAMatcher(match_threshold=match_threshold)
        self.gate = Cas9Gate(
            crispr_array=self.array,
            matcher=self.matcher,
            anti_crispr_threshold=anti_crispr_threshold,
        )
        self.acquisition = SpacerAcquisition(crispr_array=self.array)

        # Maintenance counter
        self._cycle_count = 0

        log.info(
            "[CRISPR] Bridge initialized | db=%s | max_spacers=%d | "
            "match_threshold=%.2f | anti_crispr=%.2f",
            self.array.db_path, max_spacers,
            match_threshold, anti_crispr_threshold,
        )

    def gate_check(
        self,
        symbol: str,
        direction: int,
        bars: np.ndarray,
        spread: float = 0.0,
        active_tes: List[str] = None,
        domestication_boost: float = 1.0,
        confidence: float = 0.0,
        hour_utc: int = None,
        day_of_week: int = None,
    ) -> Dict:
        """
        Run the Cas9 gate check. This is Gate G12 in Jardine's Gate.

        Returns the same dict as Cas9Gate.check().
        """
        result = self.gate.check(
            symbol=symbol,
            direction=direction,
            bars=bars,
            spread=spread,
            active_tes=active_tes,
            domestication_boost=domestication_boost,
            confidence=confidence,
            hour_utc=hour_utc,
            day_of_week=day_of_week,
        )

        # Periodic maintenance
        self._cycle_count += 1
        if self._cycle_count % MAINTENANCE_INTERVAL_CYCLES == 0:
            self.array.run_maintenance()

        return result

    def on_trade_loss(
        self,
        bars: np.ndarray,
        symbol: str,
        direction: int,
        loss_amount: float,
        active_tes: List[str] = None,
        spread: float = 0.0,
        hour_utc: int = None,
        day_of_week: int = None,
    ) -> Optional[str]:
        """
        Called by the feedback loop when a trade loses.
        Acquires a new spacer from the losing conditions.

        Returns spacer_id if acquired, None otherwise.
        """
        return self.acquisition.acquire(
            bars=bars,
            symbol=symbol,
            direction=direction,
            loss_amount=loss_amount,
            active_tes=active_tes,
            spread=spread,
            hour_utc=hour_utc,
            day_of_week=day_of_week,
        )

    def on_trade_win(self, symbol: str, direction: int):
        """
        Called when a trade WINS. Currently no action needed --
        CRISPR only learns from losses. Domestication handles wins.

        Reserved for future use (e.g., removing spacers that match
        conditions where we now consistently win).
        """
        pass

    def get_status_line(self, symbol: str = None) -> str:
        """One-line status for dashboard display."""
        stats = self.array.get_stats(symbol)
        return (
            f"[CRISPR] spacers={stats['active_spacers']} "
            f"| cuts={stats['cas9_cuts']} "
            f"| overrides={stats['anti_crispr_overrides']}"
        )

    def get_full_stats(self, symbol: str = None) -> Dict:
        """Full stats dict for monitoring and diagnostics."""
        return self.array.get_stats(symbol)


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
    print("  CRISPR-Cas ADAPTIVE IMMUNE MEMORY -- Algorithm #3")
    print("  Bacterial Immune System -> Pattern-Matching Trade Blocker")
    print("=" * 76)

    # Generate synthetic OHLCV data
    np.random.seed(42)
    n_bars = 200

    returns = np.random.randn(n_bars) * 0.008 + 0.0001
    for i in range(1, len(returns)):
        returns[i] -= 0.2 * returns[i - 1]
    close = 50000 + np.cumsum(returns) * 50000
    close = np.maximum(close, 100)

    high = close + np.abs(np.random.randn(n_bars) * close * 0.003)
    low = close - np.abs(np.random.randn(n_bars) * close * 0.003)
    open_p = close + np.random.randn(n_bars) * close * 0.001
    volume = np.abs(np.random.randn(n_bars) * 100 + 500)
    bars = np.column_stack([open_p, high, low, close, volume])

    # Use temp DB for test
    test_db = str(Path(__file__).parent / "test_crispr_cas.db")

    print(f"\n  Synthetic data: {n_bars} bars (BTC-like)")
    print(f"  Test database: {test_db}")

    # Initialize the CRISPR bridge
    bridge = CRISPRTEQABridge(db_path=test_db)

    # --- PHASE 1: SPACER ACQUISITION ---
    print("\n  --- PHASE 1: SPACER ACQUISITION (simulating losses) ---")
    print()

    test_tes = [
        ["L1_Neuronal", "Alu_Exonization", "Ty3_gypsy"],
        ["HERV_Synapse", "LINE", "Mariner_Tc1"],
        ["BEL_Pao", "CACTA", "RTE"],
        ["Crypton", "SVA_Regulatory", "DIRS1"],
        ["L1_Neuronal", "Alu_Exonization", "Ty3_gypsy"],  # Duplicate of first
    ]

    for i, tes in enumerate(test_tes):
        # Use different bar windows to simulate different market conditions
        bar_window = bars[i * 25:(i * 25) + 50]
        if len(bar_window) < 21:
            bar_window = bars[:50]

        loss = 0.50 + np.random.rand() * 0.50
        direction = 1 if i % 2 == 0 else -1

        spacer_id = bridge.on_trade_loss(
            bars=bar_window,
            symbol="BTCUSD",
            direction=direction,
            loss_amount=loss,
            active_tes=tes,
            spread=2.5,
            hour_utc=10 + i,
            day_of_week=i % 5,
        )
        dir_str = "LONG" if direction > 0 else "SHORT"
        print(f"    Loss #{i + 1}: {dir_str} loss=${loss:.2f} -> spacer={spacer_id[:8] if spacer_id else 'merged'}")

    stats = bridge.get_full_stats("BTCUSD")
    print(f"\n    Array status: {stats['active_spacers']} active spacers")

    # --- PHASE 2: CAS9 GATE CHECK ---
    print("\n  --- PHASE 2: CAS9 GATE CHECK (pre-trade scanning) ---")
    print()

    # Test with conditions similar to first loss (should match)
    print("    Test 1: Conditions similar to a known loss pattern...")
    result = bridge.gate_check(
        symbol="BTCUSD",
        direction=1,
        bars=bars[:50],  # Same window as first loss
        spread=2.5,
        active_tes=["L1_Neuronal", "Alu_Exonization", "Ty3_gypsy"],
        domestication_boost=1.0,
        confidence=0.55,
        hour_utc=10,
        day_of_week=0,
    )
    status = "BLOCKED" if not result["gate_pass"] else "PASSED"
    print(f"    Result: {status} | score={result['match_score']:.3f} | reason={result['reason']}")
    print(f"    PAM filtered: {result['pam_filtered']} | Scanned: {result['scanned']}")

    # Test with different conditions (should pass)
    print("\n    Test 2: Different market conditions (novel pattern)...")
    result2 = bridge.gate_check(
        symbol="BTCUSD",
        direction=1,
        bars=bars[150:200],  # Different time window
        spread=1.0,
        active_tes=["pogo", "hobo", "Helitron"],
        domestication_boost=1.0,
        confidence=0.60,
        hour_utc=3,      # Different session
        day_of_week=4,
    )
    status2 = "BLOCKED" if not result2["gate_pass"] else "PASSED"
    print(f"    Result: {status2} | score={result2['match_score']:.3f} | reason={result2['reason']}")

    # Test anti-CRISPR override
    print("\n    Test 3: Anti-CRISPR override (strong domestication boost)...")
    result3 = bridge.gate_check(
        symbol="BTCUSD",
        direction=1,
        bars=bars[:50],  # Same conditions as first loss
        spread=2.5,
        active_tes=["L1_Neuronal", "Alu_Exonization", "Ty3_gypsy"],
        domestication_boost=1.25,  # Above anti-CRISPR threshold
        confidence=0.70,
        hour_utc=10,
        day_of_week=0,
    )
    status3 = "BLOCKED" if not result3["gate_pass"] else "PASSED"
    override = " (anti-CRISPR)" if result3.get("anti_crispr_override") else ""
    print(f"    Result: {status3}{override} | score={result3['match_score']:.3f} | reason={result3['reason']}")

    # --- PHASE 3: MAINTENANCE ---
    print("\n  --- PHASE 3: MAINTENANCE ---")
    bridge.array.run_maintenance()

    # --- PHASE 4: STATISTICS ---
    print("\n  --- PHASE 4: STATISTICS ---")
    final_stats = bridge.get_full_stats()
    print(f"    Active spacers:      {final_stats['active_spacers']}")
    print(f"    Total spacers:       {final_stats['total_spacers']}")
    print(f"    Cas9 cuts:           {final_stats['cas9_cuts']}")
    print(f"    Anti-CRISPR events:  {final_stats['anti_crispr_overrides']}")

    if final_stats["per_symbol"]:
        print("\n    Per-symbol breakdown:")
        for ps in final_stats["per_symbol"]:
            print(f"      {ps['symbol']}: "
                  f"active={ps['active_spacers']} "
                  f"expired={ps['expired_spacers']} "
                  f"matches={ps['total_matches']}")

    print(f"\n    Status line: {bridge.get_status_line('BTCUSD')}")

    # --- CLEANUP ---
    import os
    try:
        os.remove(test_db)
    except OSError:
        pass

    print("\n" + "=" * 76)
    print("  CRISPR-Cas Adaptive Immune Memory test complete.")
    print("  Algorithm #3 in the Quantum Children biological algorithm series.")
    print("=" * 76)
