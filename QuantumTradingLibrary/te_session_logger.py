"""
TE Session Activation Logger
=============================
Logs which TE families fire by hour/session/regime.
Builds prediction maps so we can identify which TEs are
consistently active vs dormant per trading session.

Used by FocusedQuantumEngine to build right-sized circuits.

Authors: DooDoo + Claude
Date:    2026-02-13
Version: 1.0
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ============================================================
# SESSION DEFINITIONS
# ============================================================

SESSION_RANGES = {
    "TOKYO":      (0, 8),
    "LONDON":     (8, 13),
    "NY_OVERLAP": (13, 17),
    "NEW_YORK":   (17, 21),
    "OFF_SESSION": (21, 24),
}

# Minimum samples before we trust historical profiles
MIN_PROFILE_SAMPLES = 20

# TE must fire in >50% of cycles to be "predicted active"
ACTIVE_THRESHOLD = 0.50


class TESessionLogger:
    """
    Logs TE activation patterns by time/session/regime.
    Builds historical profiles to predict which TEs will be active
    in a given session, enabling focused quantum circuits.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "te_session_log.db")
        else:
            self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the session activations table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS te_session_activations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        te_name TEXT NOT NULL,
                        strength REAL NOT NULL,
                        direction INTEGER NOT NULL,
                        hour_utc INTEGER NOT NULL,
                        session_name TEXT NOT NULL,
                        shock_label TEXT DEFAULT 'UNKNOWN',
                        symbol TEXT DEFAULT 'UNKNOWN',
                        timestamp TEXT NOT NULL
                    )
                """)
                # Index for fast session profile queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_te
                    ON te_session_activations (session_name, te_name)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_hour
                    ON te_session_activations (session_name, hour_utc)
                """)
                # Cycle counter table to track total cycles per session
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS te_session_cycles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_name TEXT NOT NULL,
                        hour_utc INTEGER NOT NULL,
                        symbol TEXT DEFAULT 'UNKNOWN',
                        timestamp TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cycles_session
                    ON te_session_cycles (session_name)
                """)
                conn.commit()
            log.info("TESessionLogger: DB ready at %s", self.db_path)
        except Exception as e:
            log.error("TESessionLogger: DB init failed: %s", e)

    @staticmethod
    def get_session_name(hour_utc: int) -> str:
        """
        Map UTC hour to trading session name.

        TOKYO:       0-8 UTC
        LONDON:      8-13 UTC
        NY_OVERLAP:  13-17 UTC (London/NY overlap -- highest volume)
        NEW_YORK:    17-21 UTC
        OFF_SESSION: 21-24 UTC
        """
        hour_utc = hour_utc % 24
        for name, (start, end) in SESSION_RANGES.items():
            if start <= hour_utc < end:
                return name
        return "OFF_SESSION"

    def log_activations(
        self,
        te_activations: List[Dict],
        hour_utc: int,
        session_name: str,
        shock_label: str = "UNKNOWN",
        symbol: str = "UNKNOWN",
    ) -> int:
        """
        Log each active TE (strength > 0.3) to the session database.
        Also records a cycle entry to track total cycles per session.

        Args:
            te_activations: list of dicts with {te, strength, direction, ...}
            hour_utc: current UTC hour (0-23)
            session_name: trading session name
            shock_label: genomic shock label (CALM, NORMAL, etc.)
            symbol: trading symbol

        Returns:
            Number of active TEs logged this cycle.
        """
        now = datetime.now(timezone.utc).isoformat()
        active_count = 0

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                # Record this cycle
                conn.execute(
                    "INSERT INTO te_session_cycles (session_name, hour_utc, symbol, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (session_name, hour_utc, symbol, now)
                )

                # Log each active TE
                for act in te_activations:
                    strength = act.get("strength", 0.0)
                    if strength > 0.3:
                        conn.execute(
                            "INSERT INTO te_session_activations "
                            "(te_name, strength, direction, hour_utc, session_name, "
                            "shock_label, symbol, timestamp) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                act.get("te", "UNKNOWN"),
                                strength,
                                act.get("direction", 0),
                                hour_utc,
                                session_name,
                                shock_label,
                                symbol,
                                now,
                            )
                        )
                        active_count += 1

                conn.commit()

        except Exception as e:
            log.warning("TESessionLogger: log_activations failed: %s", e)

        return active_count

    def get_session_profile(
        self,
        session_name: str,
        hour_utc: Optional[int] = None,
        min_samples: int = MIN_PROFILE_SAMPLES,
        symbol: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Query historical activation frequency per TE for a given session.

        Returns:
            {te_name: activation_probability} -- e.g. {"LINE": 0.92, "Alu": 0.14}
            Only TEs that have appeared at all are included.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                # Count total cycles for this session
                if hour_utc is not None and symbol:
                    total_cycles = conn.execute(
                        "SELECT COUNT(*) FROM te_session_cycles "
                        "WHERE session_name = ? AND hour_utc = ? AND symbol = ?",
                        (session_name, hour_utc, symbol)
                    ).fetchone()[0]
                elif hour_utc is not None:
                    total_cycles = conn.execute(
                        "SELECT COUNT(*) FROM te_session_cycles "
                        "WHERE session_name = ? AND hour_utc = ?",
                        (session_name, hour_utc)
                    ).fetchone()[0]
                elif symbol:
                    total_cycles = conn.execute(
                        "SELECT COUNT(*) FROM te_session_cycles "
                        "WHERE session_name = ? AND symbol = ?",
                        (session_name, symbol)
                    ).fetchone()[0]
                else:
                    total_cycles = conn.execute(
                        "SELECT COUNT(*) FROM te_session_cycles "
                        "WHERE session_name = ?",
                        (session_name,)
                    ).fetchone()[0]

                if total_cycles < min_samples:
                    return {}  # Not enough data

                # Count how many cycles each TE fired in
                if hour_utc is not None and symbol:
                    rows = conn.execute(
                        "SELECT te_name, COUNT(*) as fires, AVG(strength) as avg_str "
                        "FROM te_session_activations "
                        "WHERE session_name = ? AND hour_utc = ? AND symbol = ? "
                        "GROUP BY te_name",
                        (session_name, hour_utc, symbol)
                    ).fetchall()
                elif hour_utc is not None:
                    rows = conn.execute(
                        "SELECT te_name, COUNT(*) as fires, AVG(strength) as avg_str "
                        "FROM te_session_activations "
                        "WHERE session_name = ? AND hour_utc = ? "
                        "GROUP BY te_name",
                        (session_name, hour_utc)
                    ).fetchall()
                elif symbol:
                    rows = conn.execute(
                        "SELECT te_name, COUNT(*) as fires, AVG(strength) as avg_str "
                        "FROM te_session_activations "
                        "WHERE session_name = ? AND symbol = ? "
                        "GROUP BY te_name",
                        (session_name, symbol)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT te_name, COUNT(*) as fires, AVG(strength) as avg_str "
                        "FROM te_session_activations "
                        "WHERE session_name = ? "
                        "GROUP BY te_name",
                        (session_name,)
                    ).fetchall()

                profile = {}
                for te_name, fires, avg_str in rows:
                    profile[te_name] = fires / total_cycles

                return profile

        except Exception as e:
            log.warning("TESessionLogger: get_session_profile failed: %s", e)
            return {}

    def predict_active_tes(
        self,
        hour_utc: int,
        session_name: str,
        all_te_names: Optional[List[str]] = None,
        current_activations: Optional[List[Dict]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Predict which TEs will be active vs dormant for this session.

        Uses historical profiles. Falls back to current-cycle activations
        if fewer than MIN_PROFILE_SAMPLES samples exist.

        Args:
            hour_utc: current UTC hour
            session_name: trading session name
            all_te_names: list of all TE family names (for dormant calculation)
            current_activations: current cycle activations (fallback)

        Returns:
            (predicted_active, predicted_dormant) -- both are lists of TE names
        """
        profile = self.get_session_profile(session_name, hour_utc=hour_utc)

        if profile:
            # We have enough historical data
            predicted_active = [
                te_name for te_name, prob in profile.items()
                if prob >= ACTIVE_THRESHOLD
            ]
            if all_te_names:
                predicted_dormant = [
                    te_name for te_name in all_te_names
                    if te_name not in predicted_active
                ]
            else:
                predicted_dormant = [
                    te_name for te_name, prob in profile.items()
                    if prob < ACTIVE_THRESHOLD
                ]
        elif current_activations:
            # Fallback: use current cycle data
            predicted_active = [
                a["te"] for a in current_activations
                if a.get("strength", 0) > 0.3
            ]
            predicted_dormant = [
                a["te"] for a in current_activations
                if a.get("strength", 0) <= 0.3
            ]
        else:
            # No data at all
            predicted_active = []
            predicted_dormant = list(all_te_names) if all_te_names else []

        return predicted_active, predicted_dormant

    def get_stats(self) -> Dict:
        """Return summary statistics about the session log."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                total_cycles = conn.execute(
                    "SELECT COUNT(*) FROM te_session_cycles"
                ).fetchone()[0]

                total_activations = conn.execute(
                    "SELECT COUNT(*) FROM te_session_activations"
                ).fetchone()[0]

                sessions = conn.execute(
                    "SELECT session_name, COUNT(*) FROM te_session_cycles "
                    "GROUP BY session_name ORDER BY COUNT(*) DESC"
                ).fetchall()

                top_tes = conn.execute(
                    "SELECT te_name, COUNT(*) as fires, ROUND(AVG(strength), 3) "
                    "FROM te_session_activations "
                    "GROUP BY te_name ORDER BY fires DESC LIMIT 10"
                ).fetchall()

                return {
                    "total_cycles": total_cycles,
                    "total_activations": total_activations,
                    "sessions": {s: c for s, c in sessions},
                    "top_tes": [
                        {"te": name, "fires": fires, "avg_strength": avg}
                        for name, fires, avg in top_tes
                    ],
                }
        except Exception as e:
            log.warning("TESessionLogger: get_stats failed: %s", e)
            return {"total_cycles": 0, "total_activations": 0}
