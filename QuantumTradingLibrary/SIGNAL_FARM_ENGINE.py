"""
SIGNAL FARM ENGINE - Main Orchestrator
========================================
Single Python process that:
  1. Connects to ONE MT5 terminal (read-only)
  2. Fetches M5 bars for BTCUSD, XAUUSD, ETHUSD
  3. Runs 30 virtual challenge accounts against live data
  4. Reports signals/outcomes to collection server + Base44

NO REAL TRADES ARE PLACED. This is a simulation engine fed by live data.

Run: python SIGNAL_FARM_ENGINE.py
  or: START_SIGNAL_FARM.bat
"""

import sys
import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# Ensure we're running from the right directory
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

import MetaTrader5 as mt5

from signal_farm_config import (
    FARM_SYMBOLS,
    FARM_ACCOUNTS,
    CHALLENGE_RULES,
    ENGINE_CHECK_INTERVAL,
    INDICATOR_BUFFER_SIZE,
    MT5_TERMINAL_PATHS,
    SymbolSpec,
)
from indicator_engine import IndicatorEngine, BarData
from virtual_account import VirtualAccount
from signal_farm_reporter import SignalFarmReporter

# ============================================================
# LOGGING
# ============================================================

LOG_DIR = Path("signal_farm_logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][FARM] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "signal_farm.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("signal_farm")


# ============================================================
# MT5 CONNECTION (READ-ONLY SINGLETON)
# ============================================================

class MT5DataFeed:
    """Read-only MT5 connection. Fetches bars and symbol info only."""

    def __init__(self):
        self.connected = False
        self._last_bar_times: Dict[str, int] = {}

    def connect(self) -> bool:
        """Connect to the first available MT5 terminal."""
        if self.connected:
            if mt5.terminal_info() is not None:
                return True
            self.connected = False

        # Try each terminal path
        for path in MT5_TERMINAL_PATHS:
            if Path(path).exists():
                if mt5.initialize(path=path):
                    terminal = mt5.terminal_info()
                    if terminal:
                        logger.info(f"MT5 connected: {terminal.name}")
                        self.connected = True
                        return True
                    mt5.shutdown()

        # Try default (no path)
        if mt5.initialize():
            terminal = mt5.terminal_info()
            if terminal:
                logger.info(f"MT5 connected (default): {terminal.name}")
                self.connected = True
                return True

        logger.error("MT5 connection FAILED - no terminal available")
        return False

    def get_symbol_spec(self, symbol: str) -> Optional[SymbolSpec]:
        """Get symbol specification from MT5."""
        info = mt5.symbol_info(symbol)
        if info is None:
            # Try to enable the symbol
            mt5.symbol_select(symbol, True)
            time.sleep(0.5)
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"Symbol {symbol} not available in MT5")
                return None

        # dollar_per_point = tick_value / tick_size (per 1.0 lot, per 1 price unit)
        if info.trade_tick_size > 0:
            dollar_per_point = info.trade_tick_value / info.trade_tick_size
        else:
            dollar_per_point = 0.0

        return SymbolSpec(
            name=symbol,
            dollar_per_point=dollar_per_point,
            tick_size=info.trade_tick_size,
            tick_value=info.trade_tick_value,
            point=info.point,
            digits=info.digits,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
        )

    def fetch_history(self, symbol: str, bars: int = 250) -> list:
        """Fetch historical M5 bars for initial buffer fill."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to fetch history for {symbol}")
            return []
        return rates

    def fetch_latest(self, symbol: str) -> Optional[tuple]:
        """Fetch the most recent M5 bar. Returns None if no new bar."""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 2)
        if rates is None or len(rates) < 2:
            return None

        # Use the second-to-last bar (the last COMPLETED bar)
        bar = rates[-2]
        bar_time = int(bar[0])  # bar['time'] from structured array

        last_time = self._last_bar_times.get(symbol, 0)
        if bar_time <= last_time:
            return None  # Not a new bar

        self._last_bar_times[symbol] = bar_time
        return bar

    def shutdown(self):
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")


# ============================================================
# MAIN ENGINE
# ============================================================

class SignalFarmEngine:
    """Orchestrates the entire signal farm."""

    def __init__(self):
        self.data_feed = MT5DataFeed()
        self.indicator_engine: Optional[IndicatorEngine] = None
        self.accounts: Dict[str, VirtualAccount] = {}
        self.reporter = SignalFarmReporter()
        self.symbol_specs: Dict[str, SymbolSpec] = {}
        self.running = False
        self.cycle_count = 0
        self._status_interval = 60  # Print status every N seconds
        self._last_status_time = 0

    def initialize(self) -> bool:
        """Initialize MT5 connection, symbols, indicators, and accounts."""
        print("=" * 70)
        print("  SIGNAL FARM ENGINE - 30 Virtual Challenge Accounts")
        print("  NO REAL TRADES - Simulation Only (Read-Only MT5)")
        print("=" * 70)
        print()

        # Step 1: Connect to MT5
        logger.info("Connecting to MT5...")
        if not self.data_feed.connect():
            return False

        # Step 2: Resolve symbol specs
        active_symbols = []
        for symbol in FARM_SYMBOLS:
            spec = self.data_feed.get_symbol_spec(symbol)
            if spec:
                self.symbol_specs[symbol] = spec
                active_symbols.append(symbol)
                logger.info(
                    f"  {symbol}: $/pt={spec.dollar_per_point:.4f} | "
                    f"tick={spec.tick_size} | "
                    f"lot={spec.volume_min}-{spec.volume_max} step={spec.volume_step}"
                )
            else:
                logger.warning(f"  {symbol}: NOT AVAILABLE - skipping")

        if not active_symbols:
            logger.error("No symbols available. Exiting.")
            return False

        # Step 3: Initialize indicator engine
        self.indicator_engine = IndicatorEngine(active_symbols, INDICATOR_BUFFER_SIZE)

        # Step 4: Load historical bars
        logger.info(f"Loading {INDICATOR_BUFFER_SIZE} bars of history...")
        for symbol in active_symbols:
            history = self.data_feed.fetch_history(symbol, INDICATOR_BUFFER_SIZE)
            if history is not None and len(history) > 0:
                self.indicator_engine.load_history(symbol, history)
                # Set last bar time so we don't re-process historical bars
                self.data_feed._last_bar_times[symbol] = int(history[-1][0])
                logger.info(f"  {symbol}: {len(history)} bars loaded")
            else:
                logger.warning(f"  {symbol}: No history loaded")

        # Check readiness
        status = self.indicator_engine.status()
        logger.info(f"Indicator status: {status}")

        # Step 5: Create virtual accounts
        for acc_id, params in FARM_ACCOUNTS.items():
            account = VirtualAccount(
                params=params,
                symbol_specs=self.symbol_specs,
                rules=CHALLENGE_RULES,
            )
            self.accounts[acc_id] = account
            logger.info(
                f"  {acc_id} ({params.label}): "
                f"conf={params.confidence_threshold} | "
                f"TP={params.tp_atr_multiplier}x ATR | "
                f"SL={params.sl_atr_multiplier}x ATR | "
                f"risk=${params.max_loss_dollars}"
            )

        print()
        print(f"  Active symbols: {active_symbols}")
        print(f"  Virtual accounts: {list(self.accounts.keys())}")
        print(f"  Challenge: {CHALLENGE_RULES.profit_target_pct*100:.0f}% target | "
              f"{CHALLENGE_RULES.daily_dd_limit_pct*100:.1f}% daily DD | "
              f"{CHALLENGE_RULES.max_dd_limit_pct*100:.0f}% max DD | "
              f"{CHALLENGE_RULES.time_limit_days} days")
        print(f"  Check interval: {ENGINE_CHECK_INTERVAL}s")
        print()
        print("=" * 70)
        print("  FARM RUNNING - Press Ctrl+C to stop")
        print("=" * 70)

        return True

    def run(self):
        """Main event loop."""
        self.running = True

        try:
            while self.running:
                self._tick()
                time.sleep(ENGINE_CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Shutting down (Ctrl+C)")
        finally:
            self._shutdown()

    def _tick(self):
        """One engine cycle: check for new bars, process all accounts."""
        self.cycle_count += 1

        # Verify MT5 connection
        if not self.data_feed.connected:
            if not self.data_feed.connect():
                logger.warning("MT5 reconnection failed, will retry...")
                return

        # Check each symbol for new bars
        new_bars = {}
        for symbol in self.symbol_specs:
            bar = self.data_feed.fetch_latest(symbol)
            if bar is not None:
                new_bars[symbol] = bar

        if not new_bars:
            # No new bars this cycle
            if self.cycle_count % 10 == 0:
                self._print_heartbeat()
            return

        # Process new bars through indicator engine
        for symbol, bar in new_bars.items():
            bar_data = BarData(
                time=int(bar[0]),
                open=float(bar[1]),
                high=float(bar[2]),
                low=float(bar[3]),
                close=float(bar[4]),
                tick_volume=float(bar[5]),
            )
            self.indicator_engine.update(symbol, bar_data)

        # Get current prices for equity calculation
        current_prices = {}
        for symbol in self.symbol_specs:
            snapshot = self.indicator_engine.get_snapshot(symbol)
            if snapshot:
                current_prices[symbol] = snapshot.price

        # Process each new bar through all accounts
        for symbol, bar in new_bars.items():
            snapshot = self.indicator_engine.get_snapshot(symbol)
            engine = self.indicator_engine.get_engine(symbol)

            if snapshot is None or engine is None:
                continue

            bar_time = int(bar[0])

            for acc_id, account in self.accounts.items():
                result = account.process_bar(
                    symbol=symbol,
                    snapshot=snapshot,
                    engine=engine,
                    bar_time=bar_time,
                    current_prices=current_prices,
                )

                # Report signals
                if result.confidence and result.positions_opened:
                    for pos in result.positions_opened:
                        self.reporter.report_signal(
                            account_id=acc_id,
                            symbol=symbol,
                            direction=result.confidence.direction,
                            confidence=result.confidence.final_score,
                            entropy_state=result.confidence.entropy_state.value,
                            price=snapshot.price,
                            ema_alignment=result.confidence.ema_alignment,
                            ema_separation=result.confidence.ema_separation,
                            rsi_range=result.confidence.rsi_range,
                            atr_stability=result.confidence.atr_stability,
                            raw_score=result.confidence.raw_score,
                        )
                        logger.info(
                            f"[{acc_id}] OPEN {pos.direction} {symbol} @ {pos.entry_price:.2f} | "
                            f"SL={pos.virtual_sl:.2f} TP={pos.virtual_tp:.2f} | "
                            f"Vol={pos.volume:.3f} | Conf={result.confidence.final_score:.2f}"
                        )

                # Report outcomes
                for pos, pnl, reason in result.positions_closed:
                    if reason == "PARTIAL_TP":
                        continue  # Don't report partials as full outcomes
                    self.reporter.report_outcome(
                        account_id=acc_id,
                        ticket=pos.ticket,
                        symbol=symbol,
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=pos.virtual_sl if reason == "SL" else pos.virtual_tp,
                        pnl=pnl,
                        reason=reason,
                        volume=pos.original_volume,
                    )
                    logger.info(
                        f"[{acc_id}] CLOSE {reason} {symbol} #{pos.ticket} | "
                        f"PnL=${pnl:+.2f} | Balance=${result.balance:.2f}"
                    )

                # Report challenge events
                if result.challenge_event:
                    summary = account.get_status_summary(current_prices)
                    self.reporter.report_challenge_event(
                        account_id=acc_id,
                        label=account.label,
                        event=result.challenge_event,
                        attempt=summary["attempt"],
                        balance=summary["balance"],
                        profit_pct=summary["profit_pct"],
                        total_trades=summary["total_trades"],
                        win_rate=summary["win_rate"],
                        trading_days=summary["trading_days"],
                        max_dd_pct=summary["max_dd_pct"],
                    )

        # Periodic status report
        now = time.time()
        if now - self._last_status_time >= self._status_interval:
            self._print_status(current_prices)
            self._last_status_time = now

    def _print_heartbeat(self):
        """Print a heartbeat to show the engine is alive."""
        now = datetime.now().strftime("%H:%M:%S")
        total_pos = sum(
            sum(pm.position_count for pm in acc.position_managers.values())
            for acc in self.accounts.values()
        )
        logger.debug(f"Heartbeat | Cycle #{self.cycle_count} | {total_pos} open positions")

    def _print_status(self, current_prices: Dict[str, float]):
        """Print a status table for all accounts."""
        print()
        print(f"{'='*90}")
        print(f"  SIGNAL FARM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Cycle #{self.cycle_count} | Signals: {self.reporter.signals_sent} | Outcomes: {self.reporter.outcomes_sent}")
        print(f"{'='*90}")
        print(f"  {'Account':<12} {'Label':<14} {'Status':<10} {'Balance':>10} {'P/L%':>8} {'DD%':>6} {'Trades':>7} {'WR%':>6} {'Pos':>4} {'Att':>4}")
        print(f"  {'-'*12} {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*4} {'-'*4}")

        summaries = []
        for acc_id, account in self.accounts.items():
            s = account.get_status_summary(current_prices)
            summaries.append(s)
            print(
                f"  {s['account_id']:<12} {s['label']:<14} {s['status']:<10} "
                f"${s['balance']:>8,.2f} {s['profit_pct']:>+7.1f}% "
                f"{s['max_dd_pct']:>5.1f}% {s['total_trades']:>7} "
                f"{s['win_rate']:>5.1f}% {s['open_positions']:>4} "
                f"{s['total_attempts']:>4}"
            )

        print(f"{'='*90}")
        print()

        # Send status to reporter
        self.reporter.report_status(summaries)

    def _shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down Signal Farm Engine...")
        self.running = False
        self.data_feed.shutdown()
        logger.info("Shutdown complete.")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    engine = SignalFarmEngine()

    if not engine.initialize():
        logger.error("Initialization failed. Exiting.")
        sys.exit(1)

    engine.run()


if __name__ == "__main__":
    main()
