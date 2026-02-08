"""
TEQA Signal History Database
============================
Logs every TEQA signal to SQLite for pattern analysis over time.

Usage:
    from teqa_signal_history import SignalHistoryDB

    db = SignalHistoryDB()  # creates teqa_signal_history.db
    db.log_signal(signal_json)  # call after each cycle

Author: DooDoo + Claude
Date: 2026-02-07
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent / "teqa_signal_history.db"


class SignalHistoryDB:
    """Persistent store for every TEQA signal emitted."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DEFAULT_DB_PATH)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                symbol          TEXT NOT NULL,
                direction       INTEGER NOT NULL,
                direction_str   TEXT NOT NULL,
                confidence      REAL NOT NULL,
                consensus_score REAL,
                consensus_pass  INTEGER,
                n_neurons       INTEGER,
                vote_long_n     INTEGER,
                vote_short_n    INTEGER,
                vote_neutral_n  INTEGER,
                shock_score     REAL,
                shock_label     TEXT,
                novelty         REAL,
                entropy         REAL,
                n_states        INTEGER,
                n_active_qubits INTEGER,
                vote_long       REAL,
                vote_short      REAL,
                gates_pass      INTEGER NOT NULL,
                blocked_gate    TEXT,
                n_active_class_i  INTEGER,
                n_active_class_ii INTEGER,
                n_active_neural   INTEGER,
                active_tes      TEXT,
                lot_scale       REAL,
                amplitude_sq    REAL,
                elapsed_ms      REAL,
                version         TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts
            ON signals (symbol, timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_gates
            ON signals (gates_pass, symbol)
        """)
        conn.commit()
        conn.close()
        logger.info(f"Signal history DB ready: {self.db_path}")

    def log_signal(self, signal: dict):
        """Log a signal JSON dict to the database."""
        try:
            jg = signal.get("jardines_gate", {})
            pos = signal.get("position", {})
            q = signal.get("quantum", {})
            neural = signal.get("neural", {})
            shock = signal.get("genomic_shock", {})
            gates = signal.get("gates", {})
            te_sum = signal.get("te_summary", {})
            dom = signal.get("domestication", {})

            direction = jg.get("direction", 0)
            direction_str = "LONG" if direction == 1 else ("SHORT" if direction == -1 else "NEUTRAL")

            all_gates_pass = all(gates.values()) if gates else False

            # Find which gate blocked (if any)
            blocked_gate = None
            if not all_gates_pass and gates:
                for gate_name, passed in gates.items():
                    if not passed:
                        blocked_gate = gate_name
                        break

            vote_counts = neural.get("vote_counts", {})

            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                INSERT INTO signals (
                    timestamp, symbol, direction, direction_str, confidence,
                    consensus_score, consensus_pass, n_neurons,
                    vote_long_n, vote_short_n, vote_neutral_n,
                    shock_score, shock_label,
                    novelty, entropy, n_states, n_active_qubits,
                    vote_long, vote_short,
                    gates_pass, blocked_gate,
                    n_active_class_i, n_active_class_ii, n_active_neural,
                    active_tes, lot_scale, amplitude_sq, elapsed_ms, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.get("timestamp", datetime.now().isoformat()),
                signal.get("symbol", "UNKNOWN"),
                direction,
                direction_str,
                jg.get("confidence", 0.0),
                neural.get("consensus_score", 0.0),
                1 if neural.get("consensus_pass", False) else 0,
                neural.get("n_neurons", 0),
                vote_counts.get("long", 0),
                vote_counts.get("short", 0),
                vote_counts.get("neutral", 0),
                shock.get("score", 0.0),
                shock.get("label", "UNKNOWN"),
                q.get("novelty", 0.0),
                q.get("measurement_entropy", 0.0),
                q.get("n_states", 0),
                q.get("n_active_qubits", 0),
                q.get("vote_long", 0.0),
                q.get("vote_short", 0.0),
                1 if all_gates_pass else 0,
                blocked_gate,
                te_sum.get("n_active_class_i", 0),
                te_sum.get("n_active_class_ii", 0),
                te_sum.get("n_active_neural", 0),
                json.dumps(dom.get("active_tes", [])),
                pos.get("lot_scale", 1.0),
                jg.get("amplitude_sq", 0.0),
                q.get("elapsed_ms", 0.0),
                signal.get("version", ""),
            ))
            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to log signal to history DB: {e}")

    def get_stats(self, symbol: str = None, last_n: int = None) -> dict:
        """Quick stats for analysis."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        where = ""
        params = []
        if symbol:
            where = "WHERE symbol = ?"
            params.append(symbol)

        order_limit = ""
        if last_n:
            order_limit = f"ORDER BY id DESC LIMIT {last_n}"

        # Total counts
        row = conn.execute(
            f"SELECT COUNT(*) as total, "
            f"SUM(gates_pass) as passed, "
            f"COUNT(*) - SUM(gates_pass) as failed "
            f"FROM signals {where}", params
        ).fetchone()

        total = row["total"]
        passed = row["passed"] or 0
        failed = row["failed"] or 0

        # Avg confidence by pass/fail
        pass_conf = conn.execute(
            f"SELECT AVG(confidence) as avg_conf FROM signals {where} "
            f"{'AND' if where else 'WHERE'} gates_pass = 1", params
        ).fetchone()

        fail_conf = conn.execute(
            f"SELECT AVG(confidence) as avg_conf FROM signals {where} "
            f"{'AND' if where else 'WHERE'} gates_pass = 0", params
        ).fetchone()

        # Shock distribution
        shocks = conn.execute(
            f"SELECT shock_label, COUNT(*) as cnt FROM signals {where} "
            f"GROUP BY shock_label ORDER BY cnt DESC", params
        ).fetchall()

        # Direction stability
        directions = conn.execute(
            f"SELECT direction_str, COUNT(*) as cnt FROM signals {where} "
            f"GROUP BY direction_str ORDER BY cnt DESC", params
        ).fetchall()

        # Novelty range
        novelty = conn.execute(
            f"SELECT MIN(novelty) as min_n, MAX(novelty) as max_n, "
            f"AVG(novelty) as avg_n FROM signals {where}", params
        ).fetchone()

        # Per-symbol breakdown
        per_symbol = conn.execute(
            "SELECT symbol, COUNT(*) as total, SUM(gates_pass) as passed, "
            "AVG(confidence) as avg_conf, AVG(consensus_score) as avg_consensus "
            "FROM signals GROUP BY symbol ORDER BY symbol"
        ).fetchall()

        conn.close()

        return {
            "total_signals": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_confidence_pass": pass_conf["avg_conf"] if pass_conf["avg_conf"] else 0,
            "avg_confidence_fail": fail_conf["avg_conf"] if fail_conf["avg_conf"] else 0,
            "shock_distribution": {r["shock_label"]: r["cnt"] for r in shocks},
            "direction_distribution": {r["direction_str"]: r["cnt"] for r in directions},
            "novelty_min": novelty["min_n"] if novelty["min_n"] else 0,
            "novelty_max": novelty["max_n"] if novelty["max_n"] else 0,
            "novelty_avg": novelty["avg_n"] if novelty["avg_n"] else 0,
            "per_symbol": [
                {
                    "symbol": r["symbol"],
                    "total": r["total"],
                    "passed": r["passed"] or 0,
                    "pass_rate": (r["passed"] or 0) / r["total"] if r["total"] > 0 else 0,
                    "avg_confidence": r["avg_conf"] or 0,
                    "avg_consensus": r["avg_consensus"] or 0,
                }
                for r in per_symbol
            ],
        }
