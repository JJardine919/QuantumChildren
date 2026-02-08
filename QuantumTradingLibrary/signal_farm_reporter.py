"""
SIGNAL FARM REPORTER - Dual Reporting
=======================================
Reports signal farm data to two destinations:
  1. Collection server (http://203.161.61.61:8888) via entropy_collector.py
  2. Base44 webhook (when configured)

Reuses the existing entropy_collector infrastructure for server reporting.
"""

import json
import logging
import time
import threading
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from signal_farm_config import COLLECTION_SERVER_URL, BASE44_WEBHOOK_URL

logger = logging.getLogger("signal_farm")

# Import existing entropy_collector (reuse its node ID and functions)
try:
    from entropy_collector import (
        collect_signal,
        collect_outcome,
        collect_entropy_snapshot,
        NODE_ID,
    )
    COLLECTOR_AVAILABLE = True
except ImportError:
    COLLECTOR_AVAILABLE = False
    NODE_ID = "FARM_LOCAL"


class SignalFarmReporter:
    """Dual-destination reporter for signal farm events."""

    def __init__(self, base44_url: str = ""):
        self.base44_url = base44_url or BASE44_WEBHOOK_URL
        self.collection_url = COLLECTION_SERVER_URL
        self.node_id = NODE_ID if COLLECTOR_AVAILABLE else "FARM_LOCAL"

        # Queue for async reporting
        self._queue: List[dict] = []
        self._lock = threading.Lock()

        # Stats
        self.signals_sent = 0
        self.outcomes_sent = 0
        self.errors = 0

        # Local log
        self.log_dir = Path("signal_farm_logs")
        self.log_dir.mkdir(exist_ok=True)

    def report_signal(
        self,
        account_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        entropy_state: str,
        price: float,
        ema_alignment: float = 0.0,
        ema_separation: float = 0.0,
        rsi_range: float = 0.0,
        atr_stability: float = 0.0,
        raw_score: float = 0.0,
    ):
        """Report a trading signal (entry decision)."""
        data = {
            "type": "signal",
            "source": f"FARM_{account_id}",
            "account_id": account_id,
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "entropy_state": entropy_state,
            "price": price,
            "components": {
                "ema_alignment": ema_alignment,
                "ema_separation": ema_separation,
                "rsi_range": rsi_range,
                "atr_stability": atr_stability,
                "raw_score": raw_score,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Send to collection server via entropy_collector
        if COLLECTOR_AVAILABLE:
            try:
                collect_signal({
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": confidence,
                    "quantum_entropy": raw_score,
                    "dominant_state": confidence,
                    "price": price,
                    "regime": entropy_state,
                    "source": f"FARM_{account_id}",
                })
            except Exception as e:
                self.errors += 1

        # Send to Base44
        self._send_base44(data)

        # Local log
        self._log_local(data)
        self.signals_sent += 1

    def report_outcome(
        self,
        account_id: str,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str,
        volume: float = 0.0,
        hold_bars: int = 0,
    ):
        """Report a trade outcome (close)."""
        outcome = "WIN" if pnl > 0 else ("BREAKEVEN" if pnl == 0 else "LOSS")

        data = {
            "type": "outcome",
            "source": f"FARM_{account_id}",
            "account_id": account_id,
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "outcome": outcome,
            "pnl": round(pnl, 4),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "reason": reason,
            "volume": volume,
            "hold_bars": hold_bars,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Send to collection server
        if COLLECTOR_AVAILABLE:
            try:
                collect_outcome(
                    ticket=ticket,
                    symbol=symbol,
                    outcome=outcome,
                    pnl=pnl,
                    entry_price=entry_price,
                    exit_price=exit_price,
                )
            except Exception as e:
                self.errors += 1

        # Send to Base44
        self._send_base44(data)

        # Local log
        self._log_local(data)
        self.outcomes_sent += 1

    def report_challenge_event(
        self,
        account_id: str,
        label: str,
        event: str,
        attempt: int,
        balance: float,
        profit_pct: float,
        total_trades: int,
        win_rate: float,
        trading_days: int,
        max_dd_pct: float,
    ):
        """Report a challenge lifecycle event (pass/fail/restart)."""
        data = {
            "type": "challenge_event",
            "source": f"FARM_{account_id}",
            "account_id": account_id,
            "label": label,
            "event": event,
            "attempt": attempt,
            "balance": round(balance, 2),
            "profit_pct": round(profit_pct, 2),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "trading_days": trading_days,
            "max_dd_pct": round(max_dd_pct, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._send_base44(data)
        self._log_local(data)

    def report_status(self, accounts_summary: List[Dict]):
        """Report periodic status of all accounts (for dashboard)."""
        data = {
            "type": "farm_status",
            "node_id": self.node_id,
            "accounts": accounts_summary,
            "stats": {
                "signals_sent": self.signals_sent,
                "outcomes_sent": self.outcomes_sent,
                "errors": self.errors,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._send_base44(data)
        self._log_local(data)

    def _send_base44(self, data: dict):
        """Send data to Base44 webhook (non-blocking)."""
        if not self.base44_url:
            return

        def _send():
            try:
                requests.post(
                    self.base44_url,
                    json=data,
                    timeout=5,
                    headers={"Content-Type": "application/json"},
                )
            except Exception:
                pass  # Silently fail - Base44 is optional

        thread = threading.Thread(target=_send, daemon=True)
        thread.start()

    def _log_local(self, data: dict):
        """Write event to local JSONL log."""
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = self.log_dir / f"farm_{date_str}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass
