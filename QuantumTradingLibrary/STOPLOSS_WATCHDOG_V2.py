"""
STOP LOSS WATCHDOG V2 - ENHANCED POSITION MONITOR
===================================================
CRITICAL SAFETY NET for all MT5 accounts.

Capabilities:
  1. Detects positions with MISSING stop losses (SL=0.0) and auto-applies them
  2. Force-closes positions exceeding configurable loss limit
  3. Logs EVERY position open/modify/close with timestamps
  4. Alerts when positions exceed loss threshold
  5. Tracks OrderModify success/failure rates
  6. Monitors for rogue trades (magic=0 or unknown magic numbers)
  7. Configurable max loss per position AND max total drawdown

Run:
  python STOPLOSS_WATCHDOG_V2.py --account ATLAS
  python STOPLOSS_WATCHDOG_V2.py --account ATLAS --limit 1.50 --drawdown 500
  python STOPLOSS_WATCHDOG_V2.py --account ATLAS --force-sl --emergency-sl-dollars 2.00

Author: DooDoo + Claude (QuantumChildren)
Date: 2026-02-05
"""

import time
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

import MetaTrader5 as mt5

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

# Process lock to prevent duplicate watchdog instances
from process_lock import ProcessLock

# Load config
try:
    from config_loader import MAX_LOSS_DOLLARS, AGENT_SL_MAX, ACCOUNTS
    from credential_manager import get_credentials, CredentialError
    CONFIG_LOADED = True
except ImportError:
    MAX_LOSS_DOLLARS = 1.00
    AGENT_SL_MAX = 1.00
    ACCOUNTS = {}
    CONFIG_LOADED = False
    CredentialError = Exception


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PositionSnapshot:
    """Snapshot of a position at a point in time"""
    ticket: int
    symbol: str
    direction: str  # BUY or SELL
    volume: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    magic: int
    comment: str
    open_time: str
    snapshot_time: str


@dataclass
class WatchdogEvent:
    """An event logged by the watchdog"""
    timestamp: str
    event_type: str  # POSITION_OPENED, SL_MISSING, SL_APPLIED, SL_MODIFY_FAILED,
                     # LOSS_EXCEEDED, POSITION_CLOSED, ROGUE_TRADE, DRAWDOWN_ALERT
    ticket: int
    symbol: str
    details: str
    profit: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0


@dataclass
class WatchdogStats:
    """Running statistics"""
    positions_monitored: int = 0
    sl_missing_detected: int = 0
    sl_auto_applied: int = 0
    sl_apply_failed: int = 0
    positions_force_closed: int = 0
    rogue_trades_detected: int = 0
    drawdown_alerts: int = 0
    total_loss_prevented: float = 0.0
    uptime_seconds: int = 0
    cycles_completed: int = 0


# ============================================================
# ENHANCED WATCHDOG
# ============================================================

class StopLossWatchdogV2:
    """
    Enhanced watchdog with:
    - Missing SL detection and auto-apply
    - Force-close on loss limit
    - Comprehensive logging
    - Rogue trade detection
    - Total drawdown monitoring
    """

    # System magic numbers (non-account, used by infrastructure)
    SYSTEM_MAGIC_NUMBERS: Set[int] = {
        888888,   # MCP close
        999999,   # Watchdog close
        20251222, # LSTM live trading
        20251227, # Adaptation/SEAL
    }

    @classmethod
    def _build_known_magic_numbers(cls) -> Set[int]:
        """Build known magic numbers from MASTER_CONFIG.json accounts + system magics."""
        magics = set(cls.SYSTEM_MAGIC_NUMBERS)
        for acc in ACCOUNTS.values():
            magic = acc.get('magic')
            if magic:
                magics.add(magic)
        return magics

    def __init__(
        self,
        account_config: dict,
        loss_limit: float = 1.50,
        max_drawdown: float = 500.0,
        force_sl: bool = True,
        emergency_sl_dollars: float = 2.00,
        log_dir: str = None,
    ):
        self.account = account_config
        self.loss_limit = loss_limit
        self.max_drawdown = max_drawdown
        self.force_sl = force_sl
        self.emergency_sl_dollars = emergency_sl_dollars
        self.connected = False
        self.stats = WatchdogStats()
        self.start_time = datetime.now()
        self.starting_balance = 0.0
        self.events: List[WatchdogEvent] = []
        self.known_tickets: Set[int] = set()  # Track positions we've already seen
        self.sl_applied_tickets: Set[int] = set()  # Track positions we've already fixed
        self.KNOWN_MAGIC_NUMBERS = self._build_known_magic_numbers()

        # Logging setup
        if log_dir is None:
            log_dir = str(Path(__file__).parent)
        self.log_dir = log_dir

        # Create separate log files
        self.event_log_path = os.path.join(log_dir, "watchdog_events.jsonl")
        self.stats_log_path = os.path.join(log_dir, "watchdog_stats.json")

        # Configure logging
        self.logger = logging.getLogger("WatchdogV2")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        formatter = logging.Formatter('[%(asctime)s][WATCHDOG-V2] %(message)s', '%H:%M:%S')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, "watchdog_v2.log"))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    # --------------------------------------------------------
    # CONNECTION
    # --------------------------------------------------------

    def connect(self) -> bool:
        """Connect to MT5 terminal - does NOT call mt5.login() to avoid killing trades"""
        if self.connected:
            try:
                acc = mt5.account_info()
                if acc and acc.login == self.account.get('account'):
                    return True
            except Exception:
                pass
            self.connected = False

        # Initialize MT5
        terminal_path = self.account.get('terminal_path')
        if terminal_path:
            if not mt5.initialize(path=terminal_path):
                self.logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
        else:
            if not mt5.initialize():
                self.logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False

        # Verify correct account WITHOUT calling mt5.login()
        # mt5.login() can kill open trades - the terminal should already be logged in
        acc = mt5.account_info()
        if acc is None:
            self.logger.error("Could not get account info after init")
            return False

        expected = self.account.get('account')
        if expected and acc.login != expected:
            self.logger.error(f"WRONG ACCOUNT! Expected {expected}, got {acc.login}")
            self.logger.error("Make sure the correct MT5 terminal is open and logged in")
            return False

        self.starting_balance = acc.balance
        self.connected = True
        self.logger.info(f"CONNECTED: Account {acc.login} | Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f}")
        return True

    # --------------------------------------------------------
    # CORE: POSITION CHECKS
    # --------------------------------------------------------

    def check_positions(self):
        """Main check cycle - runs all safety checks"""
        positions = mt5.positions_get()
        if positions is None:
            return

        current_tickets = set()
        total_floating = 0.0

        for pos in positions:
            current_tickets.add(pos.ticket)
            total_floating += pos.profit
            direction = "BUY" if pos.type == 0 else "SELL"

            # 1. Detect NEW positions
            if pos.ticket not in self.known_tickets:
                self.known_tickets.add(pos.ticket)
                self._log_event(WatchdogEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="POSITION_OPENED",
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    details=f"NEW: {direction} {pos.volume} @ {pos.price_open} | SL={pos.sl} TP={pos.tp} | magic={pos.magic}",
                    profit=pos.profit,
                    sl=pos.sl,
                    tp=pos.tp,
                    magic=pos.magic,
                ))

                # Check if it's a rogue trade
                if pos.magic not in self.KNOWN_MAGIC_NUMBERS:
                    self.stats.rogue_trades_detected += 1
                    self._log_event(WatchdogEvent(
                        timestamp=datetime.now().isoformat(),
                        event_type="ROGUE_TRADE",
                        ticket=pos.ticket,
                        symbol=pos.symbol,
                        details=f"UNKNOWN SOURCE: magic={pos.magic} comment='{pos.comment}' | {direction} {pos.volume} @ {pos.price_open}",
                        magic=pos.magic,
                    ))
                    self.logger.warning(
                        f"*** ROGUE TRADE DETECTED *** #{pos.ticket} {pos.symbol} {direction} {pos.volume} "
                        f"| magic={pos.magic} | NOT from known trading system!"
                    )

            # 2. Check for MISSING stop loss
            if pos.sl == 0.0 and pos.ticket not in self.sl_applied_tickets:
                self.stats.sl_missing_detected += 1
                self._log_event(WatchdogEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="SL_MISSING",
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    details=f"NO STOP LOSS! {direction} {pos.volume} @ {pos.price_open} | P/L=${pos.profit:.2f}",
                    profit=pos.profit,
                    sl=0.0,
                    tp=pos.tp,
                    magic=pos.magic,
                ))
                self.logger.warning(
                    f"*** MISSING SL *** #{pos.ticket} {pos.symbol} {direction} {pos.volume} "
                    f"| SL=0.0 | P/L=${pos.profit:.2f}"
                )

                # Auto-apply emergency SL if enabled
                if self.force_sl:
                    self._apply_emergency_sl(pos)

            # 3. Check loss limit per position
            if pos.profit < -self.loss_limit:
                self.logger.warning(
                    f"LOSS EXCEEDED: #{pos.ticket} {pos.symbol} ${pos.profit:.2f} > limit ${self.loss_limit}"
                )
                self._force_close(pos)

        # 4. Detect CLOSED positions (were known, no longer present)
        closed_tickets = self.known_tickets - current_tickets
        for ticket in closed_tickets:
            self.known_tickets.discard(ticket)
            self.sl_applied_tickets.discard(ticket)
            self._log_event(WatchdogEvent(
                timestamp=datetime.now().isoformat(),
                event_type="POSITION_CLOSED",
                ticket=ticket,
                symbol="",
                details=f"Position #{ticket} no longer open",
            ))

        # 5. Check total drawdown
        if total_floating < -self.max_drawdown:
            self.stats.drawdown_alerts += 1
            self.logger.warning(
                f"*** DRAWDOWN ALERT *** Total floating: ${total_floating:.2f} exceeds limit ${self.max_drawdown}"
            )
            self._log_event(WatchdogEvent(
                timestamp=datetime.now().isoformat(),
                event_type="DRAWDOWN_ALERT",
                ticket=0,
                symbol="ALL",
                details=f"Total floating P/L: ${total_floating:.2f} exceeds max drawdown ${self.max_drawdown}",
                profit=total_floating,
            ))

        self.stats.positions_monitored = len(positions)

    # --------------------------------------------------------
    # SL APPLICATION
    # --------------------------------------------------------

    def _apply_emergency_sl(self, position) -> bool:
        """Calculate and apply emergency stop loss to a position missing SL"""
        symbol = position.symbol
        ticket = position.ticket
        volume = position.volume
        pos_type = position.type
        open_price = position.price_open

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Cannot get symbol info for {symbol}")
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.error(f"Cannot get tick for {symbol}")
            return False

        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        point = symbol_info.point
        digits = symbol_info.digits
        stops_level = symbol_info.trade_stops_level
        spread = symbol_info.spread

        # Calculate SL distance for emergency_sl_dollars
        if tick_value > 0 and volume > 0:
            sl_ticks = self.emergency_sl_dollars / (tick_value * volume)
            sl_distance = sl_ticks * tick_size
        else:
            # Fallback: use a fixed number of points
            sl_distance = 100 * point

        # Ensure minimum stop level is respected
        min_sl_distance = (stops_level + spread + 20) * point
        if sl_distance < min_sl_distance:
            sl_distance = min_sl_distance
            self.logger.info(
                f"#{ticket} SL distance adjusted to minimum: {sl_distance:.{digits}f}"
            )

        # Calculate SL price based on CURRENT price (not open price)
        # This limits FURTHER loss from current level
        if pos_type == mt5.POSITION_TYPE_BUY:
            current_price = tick.bid
            sl_price = current_price - sl_distance
        else:
            current_price = tick.ask
            sl_price = current_price + sl_distance

        sl_price = round(sl_price, digits)

        # Log what we're about to do
        self.logger.info(
            f"APPLYING EMERGENCY SL: #{ticket} {symbol} | "
            f"Current: {current_price} | SL: {sl_price} | "
            f"Distance: {sl_distance:.{digits}f} | Max loss: ${self.emergency_sl_dollars}"
        )

        # Send the modification
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": sl_price,
            "tp": position.tp,  # Keep existing TP (even if 0)
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.stats.sl_auto_applied += 1
            self.sl_applied_tickets.add(ticket)
            self._log_event(WatchdogEvent(
                timestamp=datetime.now().isoformat(),
                event_type="SL_APPLIED",
                ticket=ticket,
                symbol=symbol,
                details=f"Emergency SL set to {sl_price} (${self.emergency_sl_dollars} max further loss from current price)",
                sl=sl_price,
                magic=position.magic,
            ))
            self.logger.info(f"SUCCESS: SL applied to #{ticket} at {sl_price}")
            return True
        else:
            self.stats.sl_apply_failed += 1
            error_msg = f"{result.comment if result else 'None'} (retcode={result.retcode if result else 'N/A'})"
            self._log_event(WatchdogEvent(
                timestamp=datetime.now().isoformat(),
                event_type="SL_MODIFY_FAILED",
                ticket=ticket,
                symbol=symbol,
                details=f"Failed to set SL: {error_msg} | Attempted SL={sl_price}",
                sl=sl_price,
                magic=position.magic,
            ))
            self.logger.error(f"FAILED to set SL on #{ticket}: {error_msg}")

            # If SL modification fails, force close as last resort
            if position.profit < -self.loss_limit:
                self.logger.warning(f"SL modify failed AND loss exceeded -- force closing #{ticket}")
                self._force_close(position)

            return False

    # --------------------------------------------------------
    # FORCE CLOSE
    # --------------------------------------------------------

    def _force_close(self, position) -> bool:
        """Force close a position that exceeds loss limit.

        After the initial close attempt, re-checks for residual volume
        from partial fills and retries up to MAX_FORCE_CLOSE_RETRIES times.
        """
        MAX_FORCE_CLOSE_RETRIES = 3
        symbol = position.symbol
        ticket = position.ticket
        volume = position.volume
        pos_type = position.type
        profit = position.profit

        symbol_info = mt5.symbol_info(symbol)
        filling_mode = mt5.ORDER_FILLING_IOC
        if symbol_info and symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK

        remaining_volume = volume
        attempt = 0

        while remaining_volume > 0 and attempt <= MAX_FORCE_CLOSE_RETRIES:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"Cannot get tick for {symbol}")
                return False

            if pos_type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": remaining_volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 50,
                "magic": 999999,
                "comment": "WATCHDOG_V2_CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            if attempt > 0:
                self.logger.warning(
                    f"FORCE CLOSE RETRY #{ticket} {symbol} | "
                    f"attempt {attempt}/{MAX_FORCE_CLOSE_RETRIES} | "
                    f"residual volume: {remaining_volume}"
                )

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Check if position still exists with residual volume
                time.sleep(0.5)  # Brief pause for server to update
                residual_pos = mt5.positions_get(ticket=ticket)
                if residual_pos and len(residual_pos) > 0:
                    remaining_volume = residual_pos[0].volume
                    if remaining_volume > 0:
                        attempt += 1
                        self.logger.warning(
                            f"PARTIAL FILL on #{ticket} {symbol} | "
                            f"residual: {remaining_volume} lots still open"
                        )
                        continue
                # Fully closed
                self.stats.positions_force_closed += 1
                self.stats.total_loss_prevented += abs(profit) - self.loss_limit
                self._log_event(WatchdogEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="FORCE_CLOSED",
                    ticket=ticket,
                    symbol=symbol,
                    details=f"Force closed at ${profit:.2f} (limit: ${self.loss_limit})"
                           + (f" [after {attempt} retries]" if attempt > 0 else ""),
                    profit=profit,
                    magic=position.magic,
                ))
                self.logger.info(f"FORCE CLOSED #{ticket} {symbol} @ ${profit:.2f}")
                return True
            else:
                error_msg = f"{result.comment if result else 'None'} ({result.retcode if result else 'N/A'})"
                self.logger.error(f"FORCE CLOSE FAILED #{ticket}: {error_msg}")
                attempt += 1

        if remaining_volume > 0:
            self.logger.error(
                f"FORCE CLOSE EXHAUSTED all {MAX_FORCE_CLOSE_RETRIES} retries for "
                f"#{ticket} {symbol} | {remaining_volume} lots STILL OPEN with no SL!"
            )
            self._log_event(WatchdogEvent(
                timestamp=datetime.now().isoformat(),
                event_type="FORCE_CLOSE_FAILED",
                ticket=ticket,
                symbol=symbol,
                details=f"Failed after {MAX_FORCE_CLOSE_RETRIES} retries, {remaining_volume} lots remain",
                profit=profit,
                magic=position.magic,
            ))
        return False

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    def _log_event(self, event: WatchdogEvent):
        """Append event to JSONL log file"""
        self.events.append(event)
        try:
            with open(self.event_log_path, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write event log: {e}")

    def _save_stats(self):
        """Save current stats to JSON file"""
        self.stats.uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
        try:
            stats_data = {
                "last_updated": datetime.now().isoformat(),
                "account": self.account.get("account", "unknown"),
                "loss_limit": self.loss_limit,
                "max_drawdown": self.max_drawdown,
                "force_sl_enabled": self.force_sl,
                "emergency_sl_dollars": self.emergency_sl_dollars,
                "stats": asdict(self.stats),
            }
            with open(self.stats_log_path, "w") as f:
                json.dump(stats_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save stats: {e}")

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    def show_status(self):
        """Display comprehensive status"""
        positions = mt5.positions_get()
        acc = mt5.account_info()

        if not acc:
            return

        uptime = datetime.now() - self.start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)

        print(f"\n{'='*70}")
        print(f"  WATCHDOG V2 STATUS | Uptime: {hours}h {minutes}m | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")
        print(f"  Account: {acc.login} | Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f}")
        print(f"  Loss Limit: ${self.loss_limit}/position | Max Drawdown: ${self.max_drawdown}")
        print(f"  Force SL: {'ON' if self.force_sl else 'OFF'} | Emergency SL: ${self.emergency_sl_dollars}")
        print(f"{'='*70}")

        # Stats
        s = self.stats
        print(f"  STATS:")
        print(f"    Positions monitored:  {s.positions_monitored}")
        print(f"    Missing SL detected:  {s.sl_missing_detected}")
        print(f"    SL auto-applied:      {s.sl_auto_applied}")
        print(f"    SL apply failures:    {s.sl_apply_failed}")
        print(f"    Force-closed:         {s.positions_force_closed}")
        print(f"    Rogue trades:         {s.rogue_trades_detected}")
        print(f"    Drawdown alerts:      {s.drawdown_alerts}")
        print(f"    Cycles completed:     {s.cycles_completed}")
        print(f"{'='*70}")

        # Position table
        if positions:
            total_profit = 0
            print(f"  OPEN POSITIONS:")
            print(f"  {'Ticket':<12} {'Symbol':<10} {'Dir':<5} {'Vol':<6} {'P/L':>10} {'SL':>12} {'TP':>12} {'Magic':>8} {'Status'}")
            print(f"  {'-'*95}")

            for pos in positions:
                total_profit += pos.profit
                direction = "BUY" if pos.type == 0 else "SELL"
                sl_status = "NO SL!" if pos.sl == 0.0 else f"{pos.sl:.2f}"
                tp_status = "NO TP" if pos.tp == 0.0 else f"{pos.tp:.2f}"

                # Status flags
                flags = []
                if pos.sl == 0.0:
                    flags.append("DANGER:NO_SL")
                if pos.profit < -self.loss_limit:
                    flags.append("OVER_LIMIT")
                if pos.magic not in self.KNOWN_MAGIC_NUMBERS:
                    flags.append("ROGUE")
                status = " | ".join(flags) if flags else "OK"

                print(
                    f"  #{pos.ticket:<11} {pos.symbol:<10} {direction:<5} {pos.volume:<6.2f} "
                    f"${pos.profit:>9.2f} {sl_status:>12} {tp_status:>12} {pos.magic:>8} {status}"
                )

            print(f"  {'-'*95}")
            print(f"  TOTAL P/L: ${total_profit:+.2f}")
        else:
            print(f"  No open positions.")

        print(f"{'='*70}\n")

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    def run(self, check_interval: int = 3):
        """Main watchdog loop"""
        print("=" * 70)
        print("  STOP LOSS WATCHDOG V2 - ENHANCED POSITION MONITOR")
        print("=" * 70)
        print(f"  Loss Limit:       ${self.loss_limit} per position")
        print(f"  Max Drawdown:     ${self.max_drawdown} total")
        print(f"  Force SL:         {'ENABLED' if self.force_sl else 'DISABLED'}")
        print(f"  Emergency SL:     ${self.emergency_sl_dollars} (max further loss from current price)")
        print(f"  Check Interval:   {check_interval}s")
        print(f"  Event Log:        {self.event_log_path}")
        print(f"  Stats Log:        {self.stats_log_path}")
        print("=" * 70)
        print("  This script monitors ALL positions and:")
        print("  - Auto-applies SL to positions missing them")
        print("  - Force-closes positions exceeding loss limit")
        print("  - Detects rogue trades from unknown sources")
        print("  - Tracks total drawdown")
        print("=" * 70)

        if not self.connect():
            self.logger.error("Failed to connect. Exiting.")
            return

        self.logger.info("Watchdog V2 started successfully")
        self._log_event(WatchdogEvent(
            timestamp=datetime.now().isoformat(),
            event_type="WATCHDOG_STARTED",
            ticket=0,
            symbol="",
            details=f"Watchdog V2 started | limit=${self.loss_limit} | drawdown=${self.max_drawdown} | force_sl={self.force_sl}",
        ))

        try:
            while True:
                self.stats.cycles_completed += 1

                # Reconnect check
                if not self.connect():
                    time.sleep(10)
                    continue

                # Core position check
                self.check_positions()

                # Show full status every 20 cycles (~1 minute at 3s interval)
                if self.stats.cycles_completed % 20 == 0:
                    self.show_status()
                    self._save_stats()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print(f"\n{'='*70}")
            print(f"  WATCHDOG V2 STOPPED")
            print(f"  Positions force-closed:  {self.stats.positions_force_closed}")
            print(f"  Stop losses auto-applied: {self.stats.sl_auto_applied}")
            print(f"  Rogue trades detected:    {self.stats.rogue_trades_detected}")
            print(f"{'='*70}")
            self._save_stats()
        finally:
            mt5.shutdown()


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Stop Loss Watchdog V2 - Enhanced Position Monitor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python STOPLOSS_WATCHDOG_V2.py --account ATLAS
  python STOPLOSS_WATCHDOG_V2.py --account ATLAS --limit 1.50 --drawdown 500
  python STOPLOSS_WATCHDOG_V2.py --account ATLAS --force-sl --emergency-sl-dollars 2.00
  python STOPLOSS_WATCHDOG_V2.py --account BG_INSTANT --limit 1.00 --no-force-sl
        """
    )
    parser.add_argument('--account', '-a', default='ATLAS',
                        help='Account key (ATLAS, BG_INSTANT, BG_CHALLENGE, GL_1, GL_2, GL_3)')
    parser.add_argument('--limit', '-l', type=float, default=1.50,
                        help='Max loss per position in dollars (default: $1.50)')
    parser.add_argument('--drawdown', '-d', type=float, default=500.0,
                        help='Max total floating drawdown in dollars (default: $500)')
    parser.add_argument('--interval', '-i', type=int, default=3,
                        help='Check interval in seconds (default: 3)')
    parser.add_argument('--force-sl', action='store_true', default=True,
                        help='Auto-apply emergency SL to positions missing them (default: ON)')
    parser.add_argument('--no-force-sl', action='store_true',
                        help='Disable auto-apply of emergency SL')
    parser.add_argument('--emergency-sl-dollars', type=float, default=2.00,
                        help='Emergency SL: max further loss from current price (default: $2.00)')

    args = parser.parse_args()

    force_sl = True
    if args.no_force_sl:
        force_sl = False

    # Get account config
    account_config = None

    if CONFIG_LOADED and args.account in ACCOUNTS:
        account_config = ACCOUNTS[args.account].copy()
        # Load password from credential_manager
        if not account_config.get('password'):
            try:
                creds = get_credentials(args.account)
                account_config['password'] = creds['password']
            except Exception as e:
                print(f"Warning: Could not load password: {e}")
    else:
        # Try credential_manager directly
        try:
            from credential_manager import get_credentials as gc
            creds = gc(args.account)
            account_config = creds
            account_config['name'] = args.account
        except Exception as e:
            print(f"Error: Could not load account config for {args.account}: {e}")
            print("Make sure MASTER_CONFIG.json and .env are configured")
            sys.exit(1)

    if account_config is None:
        print(f"Unknown account: {args.account}")
        if CONFIG_LOADED:
            print(f"Available: {list(ACCOUNTS.keys())}")
        sys.exit(1)

    # CRITICAL: Acquire process lock to prevent duplicate watchdog instances
    lock = ProcessLock(f"WATCHDOG_{args.account}", account=str(account_config.get('account')))

    try:
        with lock:
            print("=" * 60)
            print(f"WATCHDOG_{args.account} - PROCESS LOCK ACQUIRED")
            print("=" * 60)

            watchdog = StopLossWatchdogV2(
                account_config=account_config,
                loss_limit=args.limit,
                max_drawdown=args.drawdown,
                force_sl=force_sl,
                emergency_sl_dollars=args.emergency_sl_dollars,
            )
            watchdog.run(check_interval=args.interval)

    except RuntimeError as e:
        print("=" * 60)
        print("WATCHDOG PROCESS LOCK FAILURE")
        print("=" * 60)
        print(str(e))
        print("")
        print(f"Another watchdog for {args.account} is already running.")
        print("This prevents duplicate monitoring of the same account.")
        print("")
        print("To stop all processes safely:")
        print("  Run: SAFE_SHUTDOWN.bat")
        print("")
        print("To check running processes:")
        print("  Run: python process_lock.py --list")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
