"""
AUTO TE FEEDER - Automated TE Domestication Feeding
=====================================================
Part of the auto training loop. Polls MT5 for recently closed trades,
matches them to TEQA signals, and feeds win/loss outcomes to the
TEDomesticationTracker.

This is the closed-loop learning mechanism:
    Trade closes -> match to signal -> update domestication DB
    -> next TEQA cycle uses updated boost factors

This script ONLY reads trade history. It NEVER places or modifies trades.

RULE #1: DO NOT TOUCH THE LIVE TRADING SYSTEM.

Usage:
    python auto_te_feeder.py
    python auto_te_feeder.py --lookback 600   (look back 10 minutes)
    python auto_te_feeder.py --all-accounts    (scan all enabled accounts)

Authors: DooDoo + Claude
Date: 2026-02-12
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][TE_FEEDER] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("TE_FEEDER")


def get_magic_numbers() -> list:
    """Get all magic numbers from MASTER_CONFIG for deal matching."""
    try:
        with open(SCRIPT_DIR / "MASTER_CONFIG.json") as f:
            config = json.load(f)
        magics = []
        for key, acc in config.get("ACCOUNTS", {}).items():
            if acc.get("enabled", False) and "magic" in acc:
                magics.append(acc["magic"])
        return magics
    except Exception as e:
        log.warning(f"Could not load magic numbers: {e}")
        return [212001, 366001, 365001, 152001, 113001, 107001]


def feed_domestication(lookback_seconds: int = 600) -> dict:
    """
    Main feeding function.

    1. Connect to MT5 (read-only, no login switch)
    2. Pull recently closed deals
    3. Match to TEQA signals
    4. Feed outcomes to domestication tracker

    Returns stats dict.
    """
    stats = {"deals_scanned": 0, "matches_found": 0, "wins_fed": 0, "losses_fed": 0, "errors": 0}

    # Check required DBs exist
    dom_db_path = SCRIPT_DIR / "teqa_domestication.db"
    signal_db_path = SCRIPT_DIR / "teqa_signal_history.db"

    if not signal_db_path.exists():
        log.info("teqa_signal_history.db not found -- no signals to match against")
        return stats

    # Initialize MT5
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.error("MetaTrader5 not installed")
        return stats

    if not mt5.initialize():
        log.error("MT5 initialization failed")
        return stats

    try:
        magic_numbers = set(get_magic_numbers())
        log.info(f"Scanning for deals with magic numbers: {magic_numbers}")

        # Get recently closed deals
        now = datetime.now()
        from_date = now - timedelta(seconds=lookback_seconds)

        deals = mt5.history_deals_get(from_date, now)
        if deals is None:
            log.info("No deals in lookback window")
            return stats

        # Filter to our closing deals
        closing_deals = []
        for deal in deals:
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
            if deal.magic not in magic_numbers:
                continue
            closing_deals.append(deal)

        stats["deals_scanned"] = len(closing_deals)
        log.info(f"Found {len(closing_deals)} closing deals in last {lookback_seconds}s")

        if not closing_deals:
            return stats

        # Match deals to TEQA signals and feed outcomes
        try:
            conn = sqlite3.connect(str(signal_db_path), timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row

            # Check if processed_tickets table exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_tickets (
                    ticket INTEGER PRIMARY KEY,
                    processed_at TEXT NOT NULL
                )
            """)

            for deal in closing_deals:
                ticket = deal.ticket
                # Skip already processed
                row = conn.execute(
                    "SELECT 1 FROM processed_tickets WHERE ticket=?", (ticket,)
                ).fetchone()
                if row:
                    continue

                # Match to signal
                deal_time = datetime.fromtimestamp(deal.time)
                # Closing BUY = was SHORT, closing SELL = was LONG
                original_direction = -1 if deal.type == 0 else 1

                # Search for matching signal
                window_start = (deal_time - timedelta(seconds=300)).isoformat()
                window_end = deal_time.isoformat()

                try:
                    signal_row = conn.execute("""
                        SELECT active_tes, confidence
                        FROM signals
                        WHERE symbol = ?
                          AND direction = ?
                          AND timestamp BETWEEN ? AND ?
                          AND gates_pass = 1
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (deal.symbol, original_direction, window_start, window_end)).fetchone()
                except Exception:
                    signal_row = None

                if signal_row and signal_row["active_tes"]:
                    try:
                        active_tes = json.loads(signal_row["active_tes"])
                    except (json.JSONDecodeError, TypeError):
                        active_tes = None
                else:
                    active_tes = None

                if active_tes and len(active_tes) > 0:
                    net_pnl = deal.profit + deal.commission + deal.swap
                    won = net_pnl > 0

                    # Feed to domestication DB
                    _record_pattern(dom_db_path, active_tes, won, net_pnl)

                    stats["matches_found"] += 1
                    if won:
                        stats["wins_fed"] += 1
                    else:
                        stats["losses_fed"] += 1

                    te_combo = "+".join(sorted(active_tes))
                    outcome = "WIN" if won else "LOSS"
                    log.info(
                        f"  {outcome} ticket={ticket} {deal.symbol} "
                        f"profit=${deal.profit:.2f} TEs={te_combo}"
                    )

                # Mark as processed
                conn.execute(
                    "INSERT OR IGNORE INTO processed_tickets (ticket, processed_at) VALUES (?, ?)",
                    (ticket, datetime.now().isoformat())
                )

            conn.commit()
            conn.close()

        except Exception as e:
            log.error(f"Signal matching error: {e}")
            stats["errors"] += 1

    finally:
        mt5.shutdown()

    log.info(
        f"Feeding complete: {stats['matches_found']} matches "
        f"({stats['wins_fed']}W / {stats['losses_fed']}L), "
        f"{stats['deals_scanned']} deals scanned"
    )
    return stats


def _record_pattern(db_path: Path, active_tes: list, won: bool, profit: float):
    """
    Record a TE pattern outcome in the domestication database.
    Mirrors the logic from ALGORITHM_TE_DOMESTICATION.py.
    """
    import hashlib

    combo = "+".join(sorted(active_tes))
    pattern_hash = hashlib.md5(combo.encode()).hexdigest()[:16]

    DOMESTICATION_MIN_TRADES = 20
    DOMESTICATION_MIN_WR = 0.70
    DOMESTICATION_DE_MIN_WR = 0.60

    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")

        # Ensure table exists
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
                total_profit REAL DEFAULT 0.0
            )
        """)

        row = conn.execute(
            "SELECT * FROM domesticated_patterns WHERE pattern_hash=?",
            (pattern_hash,)
        ).fetchone()

        now = datetime.now().isoformat()

        if row:
            win_count = row[2] + (1 if won else 0)
            loss_count = row[3] + (0 if won else 1)
            total = win_count + loss_count
            win_rate = win_count / total if total > 0 else 0.0
            was_domesticated = bool(row[5])
            total_profit = (row[9] if len(row) > 9 else 0.0) + profit

            # Hysteresis domestication check
            if was_domesticated:
                domesticated = win_rate >= DOMESTICATION_DE_MIN_WR
            else:
                domesticated = (total >= DOMESTICATION_MIN_TRADES and
                                win_rate >= DOMESTICATION_MIN_WR)

            # Sigmoid boost
            import math
            if domesticated:
                boost = 1.0 + 0.30 / (1.0 + math.exp(-15 * (win_rate - 0.65)))
            else:
                boost = 1.0

            conn.execute("""
                UPDATE domesticated_patterns
                SET win_count=?, loss_count=?, win_rate=?,
                    domesticated=?, boost_factor=?, last_seen=?,
                    total_profit=?
                WHERE pattern_hash=?
            """, (win_count, loss_count, win_rate,
                  1 if domesticated else 0, boost, now,
                  total_profit, pattern_hash))
        else:
            # New pattern
            conn.execute("""
                INSERT INTO domesticated_patterns
                (pattern_hash, te_combo, win_count, loss_count, win_rate,
                 domesticated, boost_factor, first_seen, last_seen, total_profit)
                VALUES (?, ?, ?, ?, ?, 0, 1.0, ?, ?, ?)
            """, (pattern_hash, combo,
                  1 if won else 0,
                  0 if won else 1,
                  1.0 if won else 0.0,
                  now, now, profit))

        conn.commit()
        conn.close()

    except Exception as e:
        log.error(f"Failed to record pattern {combo}: {e}")


def show_domestication_stats():
    """Display current domestication database statistics."""
    dom_db = SCRIPT_DIR / "teqa_domestication.db"
    if not dom_db.exists():
        print("teqa_domestication.db not found")
        return

    conn = sqlite3.connect(str(dom_db), timeout=5)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM domesticated_patterns").fetchone()[0]
    domesticated = conn.execute(
        "SELECT COUNT(*) FROM domesticated_patterns WHERE domesticated=1"
    ).fetchone()[0]

    print(f"\nTE Domestication Database:")
    print(f"  Total patterns:        {total}")
    print(f"  Domesticated:          {domesticated}")
    print(f"  Wild/neutral:          {total - domesticated}")

    if domesticated > 0:
        print(f"\n  Top domesticated patterns:")
        rows = conn.execute("""
            SELECT te_combo, win_rate, boost_factor, win_count, loss_count
            FROM domesticated_patterns
            WHERE domesticated=1
            ORDER BY boost_factor DESC
            LIMIT 10
        """).fetchall()
        for row in rows:
            total_trades = row["win_count"] + row["loss_count"]
            print(
                f"    {row['te_combo']:<50} "
                f"WR={row['win_rate']*100:.0f}% "
                f"boost={row['boost_factor']:.3f} "
                f"({row['win_count']}W/{row['loss_count']}L={total_trades})"
            )

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto TE Domestication Feeder")
    parser.add_argument("--lookback", type=int, default=600,
                        help="Lookback window in seconds (default: 600)")
    parser.add_argument("--stats", action="store_true",
                        help="Show domestication DB stats")
    args = parser.parse_args()

    if args.stats:
        show_domestication_stats()
    else:
        stats = feed_domestication(lookback_seconds=args.lookback)
        print(json.dumps(stats, indent=2))
