"""
VIRTUAL ACCOUNT - Challenge State Machine
===========================================
Each virtual account runs a prop-firm-style challenge:
  - Tracks balance, equity, daily DD, max DD
  - Enforces profit target, DD limits, time limits
  - Auto-restarts on pass or fail with fresh $5,000
  - Counts trading days and enforces minimums

This is the "brain" that decides whether to trade and
tracks the challenge lifecycle.
"""

import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from signal_farm_config import FarmParams, ChallengeRules, SymbolSpec, CHALLENGE_RULES
from indicator_engine import IndicatorSnapshot, SymbolIndicators
from entropy_confidence import score_and_classify, ConfidenceResult, EntropyState
from virtual_position_manager import VirtualPositionManager, VirtualPosition

logger = logging.getLogger("signal_farm")


class ChallengeStatus:
    ACTIVE = "ACTIVE"
    PASSED = "PASSED"
    FAILED_DAILY_DD = "FAILED_DAILY_DD"
    FAILED_MAX_DD = "FAILED_MAX_DD"
    FAILED_TIME = "FAILED_TIME"


@dataclass
class ChallengeState:
    """Tracks one challenge attempt."""
    attempt_number: int = 1
    status: str = ChallengeStatus.ACTIVE
    balance: float = 5000.0
    starting_balance: float = 5000.0
    high_water_mark: float = 5000.0
    daily_start_balance: float = 5000.0
    current_day: int = 0               # Calendar day (for daily reset)
    trading_days_elapsed: int = 0
    days_with_trades: int = 0          # Days that had at least one trade
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    last_bar_day: int = -1             # Last bar's day-of-year for daily reset


@dataclass
class ProcessResult:
    """Result of processing one bar for one account."""
    account_id: str
    symbol: str
    confidence: Optional[ConfidenceResult] = None
    positions_opened: List[VirtualPosition] = field(default_factory=list)
    positions_closed: List[Tuple[VirtualPosition, float, str]] = field(default_factory=list)
    challenge_event: str = ""          # "PASS", "FAIL_DAILY_DD", "FAIL_MAX_DD", "RESTART", ""
    balance: float = 0.0
    equity: float = 0.0
    daily_dd_pct: float = 0.0
    max_dd_pct: float = 0.0


class VirtualAccount:
    """
    One virtual challenge account with its own parameters.
    Manages multiple symbols, each with its own position manager.
    """

    def __init__(
        self,
        params: FarmParams,
        symbol_specs: Dict[str, SymbolSpec],
        rules: ChallengeRules = CHALLENGE_RULES,
    ):
        self.params = params
        self.rules = rules
        self.symbol_specs = symbol_specs

        # Position managers per symbol
        self.position_managers: Dict[str, VirtualPositionManager] = {
            sym: VirtualPositionManager(params, spec)
            for sym, spec in symbol_specs.items()
        }

        # Per-account balance override (0 = use rules default)
        bal = params.starting_balance if params.starting_balance > 0 else rules.starting_balance
        self._effective_starting_balance = bal

        # Challenge state
        self.state = ChallengeState(
            balance=bal,
            starting_balance=bal,
            high_water_mark=bal,
            daily_start_balance=bal,
            start_time=time.time(),
        )

        # Stats across all challenge attempts
        self.total_attempts = 0
        self.total_passes = 0
        self.total_fails = 0
        self._traded_today = False

    @property
    def account_id(self) -> str:
        return self.params.account_id

    @property
    def label(self) -> str:
        return self.params.label

    @property
    def balance(self) -> float:
        return self.state.balance

    def get_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate current equity (balance + unrealized PnL)."""
        unrealized = 0.0
        for sym, pm in self.position_managers.items():
            if sym in current_prices:
                unrealized += pm.get_unrealized_pnl(current_prices[sym])
        return self.state.balance + unrealized

    def process_bar(
        self,
        symbol: str,
        snapshot: IndicatorSnapshot,
        engine: SymbolIndicators,
        bar_time: int,
        current_prices: Dict[str, float],
    ) -> ProcessResult:
        """
        Process a single new M5 bar for one symbol.
        This is the main entry point called by the engine.

        Steps:
          1. Daily reset check
          2. High water mark update
          3. Drawdown check
          4. Position management (SL/TP/partial/breakeven/trail)
          5. Entry evaluation (confidence + grid spacing)
          6. Challenge status check (pass/fail/time)
        """
        result = ProcessResult(
            account_id=self.account_id,
            symbol=symbol,
            balance=self.state.balance,
        )

        # Skip if challenge already ended (shouldn't happen, but safety)
        if self.state.status != ChallengeStatus.ACTIVE:
            self._restart_challenge()
            result.challenge_event = "RESTART"
            return result

        pm = self.position_managers.get(symbol)
        if pm is None:
            return result

        price = snapshot.price
        bar_high = snapshot.bar_high
        bar_low = snapshot.bar_low
        atr = snapshot.atr_14

        # --- 1. Daily Reset ---
        bar_dt = datetime.fromtimestamp(bar_time, tz=timezone.utc)
        bar_day = bar_dt.timetuple().tm_yday + bar_dt.year * 1000  # Unique day ID

        if self.state.last_bar_day == -1:
            self.state.last_bar_day = bar_day

        if bar_day != self.state.last_bar_day:
            # New trading day
            self.state.last_bar_day = bar_day
            self.state.daily_start_balance = self.state.balance
            self.state.trading_days_elapsed += 1
            if self._traded_today:
                self.state.days_with_trades += 1
            self._traded_today = False

        # --- 2. High Water Mark ---
        equity = self.get_equity(current_prices)
        result.equity = equity

        if self.state.balance > self.state.high_water_mark:
            self.state.high_water_mark = self.state.balance

        # --- 3. Drawdown Check ---
        # Daily DD: (daily_start - min(balance, equity)) / starting_balance
        daily_low = min(self.state.balance, equity)
        daily_dd = (self.state.daily_start_balance - daily_low) / self.state.starting_balance
        result.daily_dd_pct = daily_dd

        # Max DD: (high_water_mark - min(balance, equity)) / starting_balance
        # NOTE: Using starting_balance as denominator (standard prop firm calculation)
        max_dd = (self.state.high_water_mark - daily_low) / self.state.starting_balance
        result.max_dd_pct = max_dd

        if daily_dd >= self.rules.daily_dd_limit_pct:
            self._end_challenge(ChallengeStatus.FAILED_DAILY_DD)
            result.challenge_event = "FAIL_DAILY_DD"
            return result

        if max_dd >= self.rules.max_dd_limit_pct:
            self._end_challenge(ChallengeStatus.FAILED_MAX_DD)
            result.challenge_event = "FAIL_MAX_DD"
            return result

        # --- 4. Position Management ---
        closed = pm.manage_positions(bar_high, bar_low, price, atr)
        for pos, pnl, reason in closed:
            if reason != "PARTIAL_TP":
                # Full close
                self.state.balance += pnl
                self.state.total_pnl += pnl
                self.state.total_trades += 1
                if pnl > 0:
                    self.state.winning_trades += 1
                else:
                    self.state.losing_trades += 1
            else:
                # Partial close - add partial PnL to balance
                self.state.balance += pnl
                self.state.total_pnl += pnl

        result.positions_closed = closed
        result.balance = self.state.balance

        # --- 5. Check Profit Target ---
        profit_pct = (self.state.balance - self.state.starting_balance) / self.state.starting_balance
        if profit_pct >= self.rules.profit_target_pct:
            if self.state.days_with_trades >= self.rules.min_trading_days:
                self._end_challenge(ChallengeStatus.PASSED)
                result.challenge_event = "PASS"
                return result

        # --- 6. Check Time Limit ---
        if self.state.trading_days_elapsed >= self.rules.time_limit_days:
            self._end_challenge(ChallengeStatus.FAILED_TIME)
            result.challenge_event = "FAIL_TIME"
            return result

        # --- 7. Entry Evaluation ---
        if not snapshot.ready:
            return result

        # Don't open if DD is close to limit (safety margin)
        if daily_dd >= self.rules.daily_dd_limit_pct * 0.85:
            return result
        if max_dd >= self.rules.max_dd_limit_pct * 0.85:
            return result

        # Score confidence
        confidence = score_and_classify(
            snapshot, engine,
            self.params.confidence_threshold,
            self.params.compression_boost,
        )
        result.confidence = confidence

        # Skip if entropy too high
        if confidence.entropy_state == EntropyState.HIGH:
            return result

        # Skip if no direction
        if confidence.direction == "NONE":
            return result

        # Skip if below threshold
        if confidence.final_score < self.params.confidence_threshold:
            return result

        # Check grid spacing
        if not pm.check_grid_spacing(price, confidence.direction, atr):
            return result

        # Check position limit
        if not pm.can_open():
            return result

        # Open position
        pos = pm.open_position(
            direction=confidence.direction,
            price=price,
            atr=atr,
            entropy_state=confidence.entropy_state,
            balance=self.state.balance,
        )

        if pos is not None:
            result.positions_opened.append(pos)
            self._traded_today = True

        return result

    def _end_challenge(self, status: str):
        """End the current challenge with given status."""
        self.state.status = status
        self.state.end_time = time.time()
        self.total_attempts += 1

        if status == ChallengeStatus.PASSED:
            self.total_passes += 1
        else:
            self.total_fails += 1

        # Force-close all positions
        for pm in self.position_managers.values():
            # Get last known price from the position entries if available
            if pm.positions:
                # Use entry price as rough estimate (better than nothing)
                pm.close_all(pm.positions[0].entry_price)
            pm.clear_closed_history()

        duration = self.state.end_time - self.state.start_time
        profit_pct = (self.state.balance - self.state.starting_balance) / self.state.starting_balance * 100

        logger.info(
            f"[{self.account_id}] CHALLENGE {status} | "
            f"Attempt #{self.state.attempt_number} | "
            f"Balance: ${self.state.balance:.2f} ({profit_pct:+.1f}%) | "
            f"Trades: {self.state.total_trades} (W:{self.state.winning_trades} L:{self.state.losing_trades}) | "
            f"Days: {self.state.trading_days_elapsed} | "
            f"Duration: {duration/3600:.1f}h"
        )

    def _restart_challenge(self):
        """Restart challenge with fresh balance and same parameters."""
        attempt = self.state.attempt_number + 1
        bal = self._effective_starting_balance
        self.state = ChallengeState(
            attempt_number=attempt,
            balance=bal,
            starting_balance=bal,
            high_water_mark=bal,
            daily_start_balance=bal,
            start_time=time.time(),
        )

        # Reset position managers
        for pm in self.position_managers.values():
            pm.positions.clear()
            pm.clear_closed_history()

        self._traded_today = False

        logger.info(
            f"[{self.account_id}] RESTART Challenge #{attempt} | "
            f"Record: {self.total_passes}P / {self.total_fails}F / {self.total_attempts} total"
        )

    def get_status_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """Get a status summary dict for reporting."""
        equity = self.get_equity(current_prices or {})
        profit_pct = (self.state.balance - self.state.starting_balance) / self.state.starting_balance * 100

        total_positions = sum(pm.position_count for pm in self.position_managers.values())

        return {
            "account_id": self.account_id,
            "label": self.label,
            "status": self.state.status,
            "attempt": self.state.attempt_number,
            "balance": round(self.state.balance, 2),
            "equity": round(equity, 2),
            "profit_pct": round(profit_pct, 2),
            "high_water_mark": round(self.state.high_water_mark, 2),
            "daily_dd_pct": round(
                (self.state.daily_start_balance - min(self.state.balance, equity))
                / self.state.starting_balance * 100, 2
            ),
            "max_dd_pct": round(
                (self.state.high_water_mark - min(self.state.balance, equity))
                / self.state.starting_balance * 100, 2
            ),
            "total_trades": self.state.total_trades,
            "win_rate": round(
                self.state.winning_trades / max(self.state.total_trades, 1) * 100, 1
            ),
            "trading_days": self.state.trading_days_elapsed,
            "days_traded": self.state.days_with_trades,
            "open_positions": total_positions,
            "total_attempts": self.total_attempts,
            "total_passes": self.total_passes,
            "total_fails": self.total_fails,
        }
