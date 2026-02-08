"""
TEQA FEEDBACK - Closes the Domestication Learning Loop
=======================================================
Polls MT5 for recently closed trades, matches them to the TEQA signal
that triggered the trade (via signal history DB), and feeds win/loss
outcomes into the TEDomesticationTracker.

Pipeline:
    Trade closes (SL/TP hit) → MT5 deal history
    → match deal to signal (timestamp + symbol + direction + magic)
    → extract active_tes from that signal
    → TEDomesticationTracker.record_pattern(active_tes, won)
    → next TEQA cycle's get_boost() returns updated weights

This is the missing feedback loop that turns TEQA from open-loop
to closed-loop learning (TE domestication = STDP for transposons).

Author: DooDoo + Claude
Date: 2026-02-08
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

# How far back to scan for closed deals each poll (seconds)
DEFAULT_LOOKBACK_SECONDS = 300  # 5 minutes

# Tolerance for matching signal timestamp to trade open time (seconds)
SIGNAL_MATCH_WINDOW_SECONDS = 120  # 2 minutes


class TradeOutcomePoller:
    """
    Polls MT5 for closed trades and feeds outcomes to TEDomesticationTracker.

    Usage in BRAIN scripts:
        from teqa_feedback import TradeOutcomePoller
        from teqa_v3_neural_te import TEQAv3Engine

        engine = TEQAv3Engine(...)
        poller = TradeOutcomePoller(
            magic_numbers=[212001],
            domestication_tracker=engine.domestication,
        )

        # In your run loop, after run_cycle():
        poller.poll()
    """

    def __init__(
        self,
        magic_numbers: List[int],
        domestication_tracker,
        signal_history_db_path: str = None,
        lookback_seconds: int = DEFAULT_LOOKBACK_SECONDS,
    ):
        """
        Args:
            magic_numbers: MT5 magic numbers to track (filters our trades from manual ones)
            domestication_tracker: TEDomesticationTracker instance (has record_pattern method)
            signal_history_db_path: path to teqa_signal_history.db
            lookback_seconds: how far back to scan MT5 history each poll
        """
        self.magic_numbers = set(magic_numbers)
        self.tracker = domestication_tracker
        self.lookback_seconds = lookback_seconds

        if signal_history_db_path is None:
            self.signal_db_path = str(Path(__file__).parent / "teqa_signal_history.db")
        else:
            self.signal_db_path = signal_history_db_path

        # Track which deal tickets we've already processed (avoid double-counting)
        self._processed_tickets: set = set()
        # Cap the set size to prevent unbounded memory growth
        self._max_processed = 10000

        logger.info(f"[FEEDBACK] Initialized | magic={list(self.magic_numbers)} | "
                     f"lookback={lookback_seconds}s | signal_db={self.signal_db_path}")

    def poll(self) -> List[Dict]:
        """
        Check for newly closed trades and feed outcomes to domestication tracker.

        Returns list of processed outcomes for logging:
            [{"ticket": 123, "symbol": "BTCUSD", "profit": 2.50, "won": True, "active_tes": [...]}]
        """
        closed_deals = self._get_recent_closed_deals()
        if not closed_deals:
            return []

        outcomes = []
        for deal in closed_deals:
            ticket = deal["ticket"]

            # Skip already-processed deals
            if ticket in self._processed_tickets:
                continue

            # Match deal to the TEQA signal that triggered it
            active_tes = self._match_signal_to_deal(deal)

            if active_tes is not None and len(active_tes) > 0:
                won = deal["profit"] > 0
                self.tracker.record_pattern(active_tes, won)

                outcome = {
                    "ticket": ticket,
                    "symbol": deal["symbol"],
                    "profit": deal["profit"],
                    "won": won,
                    "active_tes": active_tes,
                    "te_combo": "+".join(sorted(active_tes)),
                }
                outcomes.append(outcome)

                logger.info(
                    f"[FEEDBACK] {'WIN' if won else 'LOSS'} ticket={ticket} "
                    f"{deal['symbol']} profit=${deal['profit']:.2f} | "
                    f"TEs: {outcome['te_combo']}"
                )
            else:
                logger.debug(
                    f"[FEEDBACK] No signal match for ticket={ticket} "
                    f"{deal['symbol']} (manual trade or signal expired)"
                )

            # Mark as processed
            self._processed_tickets.add(ticket)

            # Prevent unbounded growth
            if len(self._processed_tickets) > self._max_processed:
                # Keep only the most recent half
                sorted_tickets = sorted(self._processed_tickets)
                self._processed_tickets = set(sorted_tickets[len(sorted_tickets) // 2:])

        if outcomes:
            wins = sum(1 for o in outcomes if o["won"])
            losses = len(outcomes) - wins
            logger.info(f"[FEEDBACK] Cycle: {wins}W / {losses}L fed to domestication tracker")

        return outcomes

    def _get_recent_closed_deals(self) -> List[Dict]:
        """Get closing deals from MT5 history within lookback window."""
        now = datetime.now()
        from_date = now - timedelta(seconds=self.lookback_seconds)

        try:
            deals = mt5.history_deals_get(from_date, now)
        except Exception as e:
            logger.warning(f"[FEEDBACK] MT5 history_deals_get failed: {e}")
            return []

        if deals is None:
            return []

        result = []
        for deal in deals:
            # Only closing deals (DEAL_ENTRY_OUT) from our magic numbers
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
            if deal.magic not in self.magic_numbers:
                continue

            result.append({
                "ticket": deal.ticket,
                "order": deal.order,
                "symbol": deal.symbol,
                "type": "BUY" if deal.type == 0 else "SELL",
                "volume": deal.volume,
                "price": deal.price,
                "profit": deal.profit,
                "commission": deal.commission,
                "swap": deal.swap,
                "magic": deal.magic,
                "time": datetime.fromtimestamp(deal.time),
                "time_iso": datetime.fromtimestamp(deal.time).isoformat(),
            })

        return result

    def _match_signal_to_deal(self, deal: Dict) -> Optional[List[str]]:
        """
        Find the TEQA signal that most likely triggered this trade.

        Matching strategy:
            1. Same symbol
            2. Signal timestamp within SIGNAL_MATCH_WINDOW_SECONDS before trade open time
            3. Direction matches (LONG→BUY, SHORT→SELL)
            4. Gates passed (wouldn't have traded if blocked)
            5. Take the closest match by timestamp

        Returns the active_tes list from the matched signal, or None.
        """
        deal_time = deal["time"]
        deal_symbol = deal["symbol"]
        # Deal type is the CLOSING side, so the original trade direction is opposite
        # BUY close = was SHORT, SELL close = was LONG
        original_direction = -1 if deal["type"] == "BUY" else 1

        # Search signal history DB for matching signals
        window_start = (deal_time - timedelta(seconds=SIGNAL_MATCH_WINDOW_SECONDS)).isoformat()
        window_end = deal_time.isoformat()

        try:
            conn = sqlite3.connect(self.signal_db_path, timeout=5)
            conn.row_factory = sqlite3.Row

            rows = conn.execute("""
                SELECT timestamp, direction, active_tes, confidence
                FROM signals
                WHERE symbol = ?
                  AND timestamp BETWEEN ? AND ?
                  AND direction = ?
                  AND gates_pass = 1
                ORDER BY timestamp DESC
                LIMIT 1
            """, (deal_symbol, window_start, window_end, original_direction)).fetchall()

            conn.close()

            if not rows:
                return None

            row = rows[0]
            active_tes_json = row["active_tes"]
            if active_tes_json:
                active_tes = json.loads(active_tes_json)
                if isinstance(active_tes, list) and len(active_tes) > 0:
                    return active_tes

            return None

        except Exception as e:
            logger.warning(f"[FEEDBACK] Signal DB query failed: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get feedback loop stats for dashboard display."""
        return {
            "processed_tickets": len(self._processed_tickets),
            "magic_numbers": list(self.magic_numbers),
            "lookback_seconds": self.lookback_seconds,
        }
