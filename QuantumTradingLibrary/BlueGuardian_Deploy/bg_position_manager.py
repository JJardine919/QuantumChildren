"""
Blue Guardian Dynamic Position Manager
=======================================
Manages open positions with:
- Trailing Stop Loss
- Dynamic Take Profit (scales with SL)
- Partial Take Profit
- Maintains 1:3 R:R ratio

HARDCODED VALUES - DO NOT CHANGE
These are locked for consistency across all account sizes.

Author: Quantum Library
Version: 3.0
"""

import MetaTrader5 as mt5
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

# =============================================================================
# HARDCODED DYNAMIC SL/TP SETTINGS - DO NOT CHANGE
# =============================================================================
SL_ATR_MULT = 1.5           # SL = ATR x 1.5
RR_RATIO = 3.0              # Risk:Reward = 1:3 (LOCKED)
USE_TRAILING_SL = True      # Trailing Stop ALWAYS ON
USE_DYNAMIC_TP = True       # Dynamic TP ALWAYS ON
BREAKEVEN_TRIGGER = 0.5     # Move to BE at 50% of SL distance
BREAKEVEN_BUFFER = 5.0      # Buffer above breakeven (price units)
TRAIL_START_MULT = 1.0      # Start trail at 100% of SL profit
TRAIL_DISTANCE = 25.0       # Trail distance behind price
USE_PARTIAL_TP = True       # Partial TP ALWAYS ON
PARTIAL_TP_PCT = 50.0       # Close 50% at partial TP
PARTIAL_TP_TRIGGER = 0.5    # Partial at 50% of full TP distance
# =============================================================================

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | POSITION_MGR | %(message)s',
    handlers=[
        logging.FileHandler('bg_position_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Tracks state for dynamic management of a position"""
    ticket: int
    symbol: str
    magic: int
    entry_price: float
    initial_sl: float
    initial_tp: float
    volume: float
    position_type: int  # 0=BUY, 1=SELL
    breakeven_hit: bool = False
    partial_closed: bool = False
    trailing_active: bool = False
    last_update: datetime = field(default_factory=datetime.now)


class DynamicPositionManager:
    """
    Manages open positions with dynamic SL/TP.

    Features (ALL HARDCODED - LOCKED):
    - Trailing Stop: Moves SL to lock in profits
    - Breakeven: Moves SL to entry + buffer at 50% profit
    - Dynamic TP: Scales TP as SL trails, maintaining 1:3 R:R
    - Partial TP: Closes 50% at halfway to target
    """

    def __init__(self, config_path: str = "accounts_config.json"):
        self.config_path = Path(config_path)
        self.accounts: Dict[str, Dict] = {}
        self.position_states: Dict[int, PositionState] = {}
        self.running = False

        self._load_config()
        self._print_settings()

    def _load_config(self):
        """Load account configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.accounts = {
                    acc['name']: acc
                    for acc in config.get('accounts', [])
                }
            log.info(f"Loaded {len(self.accounts)} accounts")
        else:
            log.warning(f"Config not found: {self.config_path}")

    def _print_settings(self):
        """Print hardcoded settings"""
        print("=" * 60)
        print("BLUE GUARDIAN DYNAMIC POSITION MANAGER v3.0")
        print("=" * 60)
        print("HARDCODED SETTINGS (LOCKED):")
        print(f"  SL: ATR x {SL_ATR_MULT}")
        print(f"  R:R Ratio: 1:{RR_RATIO}")
        print(f"  Trailing SL: {'ENABLED' if USE_TRAILING_SL else 'DISABLED'}")
        print(f"  Dynamic TP: {'ENABLED' if USE_DYNAMIC_TP else 'DISABLED'}")
        print(f"  Partial TP: {PARTIAL_TP_PCT}%")
        print(f"  Breakeven Trigger: {BREAKEVEN_TRIGGER * 100}% of SL")
        print(f"  Trail Start: {TRAIL_START_MULT * 100}% of SL")
        print(f"  Trail Distance: {TRAIL_DISTANCE}")
        print("=" * 60)

    def connect(self) -> bool:
        """Connect to MT5"""
        if not mt5.initialize():
            log.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        account_info = mt5.account_info()
        if account_info:
            log.info(f"Connected to MT5: Account {account_info.login}")
            return True
        return False

    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        log.info("Disconnected from MT5")

    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate ATR for symbol"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            # Fallback
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return (tick.ask - tick.bid) * 10
            return 50.0  # Default for Gold

        tr_sum = 0
        for i in range(1, len(rates)):
            high = rates[i]['high']
            low = rates[i]['low']
            prev_close = rates[i-1]['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_sum += tr

        return tr_sum / period

    def get_current_price(self, symbol: str, position_type: int) -> Optional[float]:
        """Get current price for position type"""
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        # BUY positions close at bid, SELL at ask
        return tick.bid if position_type == mt5.POSITION_TYPE_BUY else tick.ask

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify position stop loss"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        symbol_info = mt5.symbol_info(pos.symbol)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": round(new_sl, symbol_info.digits),
            "tp": pos.tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True

        log.error(f"Modify SL failed: {result.comment if result else 'No result'}")
        return False

    def modify_tp(self, ticket: int, new_tp: float) -> bool:
        """Modify position take profit"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        symbol_info = mt5.symbol_info(pos.symbol)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": pos.sl,
            "tp": round(new_tp, symbol_info.digits),
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True

        log.error(f"Modify TP failed: {result.comment if result else 'No result'}")
        return False

    def close_partial(self, ticket: int, volume: float) -> bool:
        """Close partial position"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False

        pos = position[0]
        symbol_info = mt5.symbol_info(pos.symbol)

        # Normalize volume
        volume = max(volume, symbol_info.volume_min)
        volume = min(volume, pos.volume)
        volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step

        # Close type is opposite of position type
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": volume,
            "type": close_type,
            "price": price,
            "deviation": 50,
            "magic": pos.magic,
            "comment": "BG_PartialTP",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True

        log.error(f"Partial close failed: {result.comment if result else 'No result'}")
        return False

    def manage_position(self, position) -> None:
        """
        Manage a single position with dynamic SL/TP.

        Logic:
        1. Breakeven at 50% of SL profit
        2. Trail SL at 100% of SL profit
        3. Scale TP to maintain 1:3 R:R
        4. Partial close at 50% of TP
        """
        ticket = position.ticket
        symbol = position.symbol
        pos_type = position.type
        entry_price = position.price_open
        current_sl = position.sl
        current_tp = position.tp
        volume = position.volume
        magic = position.magic

        # Get or create position state
        if ticket not in self.position_states:
            self.position_states[ticket] = PositionState(
                ticket=ticket,
                symbol=symbol,
                magic=magic,
                entry_price=entry_price,
                initial_sl=current_sl,
                initial_tp=current_tp,
                volume=volume,
                position_type=pos_type
            )
            log.info(f"[NEW] Tracking position {ticket}: {symbol} @ {entry_price}")

        state = self.position_states[ticket]

        # Get current price
        current_price = self.get_current_price(symbol, pos_type)
        if not current_price:
            return

        # Calculate profit distance
        if pos_type == mt5.POSITION_TYPE_BUY:
            profit_dist = current_price - entry_price
            risk_dist = entry_price - current_sl if current_sl > 0 else self.calculate_atr(symbol) * SL_ATR_MULT
        else:  # SELL
            profit_dist = entry_price - current_price
            risk_dist = current_sl - entry_price if current_sl > 0 else self.calculate_atr(symbol) * SL_ATR_MULT

        # Ensure valid risk distance
        if risk_dist <= 0:
            risk_dist = self.calculate_atr(symbol) * SL_ATR_MULT

        # ===== 1. BREAKEVEN LOGIC =====
        if USE_TRAILING_SL and not state.breakeven_hit:
            be_threshold = risk_dist * BREAKEVEN_TRIGGER

            if profit_dist >= be_threshold:
                if pos_type == mt5.POSITION_TYPE_BUY:
                    new_sl = entry_price + BREAKEVEN_BUFFER
                    if new_sl > current_sl:
                        if self.modify_sl(ticket, new_sl):
                            state.breakeven_hit = True
                            log.info(f"[BREAKEVEN] {ticket}: SL -> {new_sl:.2f} (+{BREAKEVEN_BUFFER} buffer)")
                else:  # SELL
                    new_sl = entry_price - BREAKEVEN_BUFFER
                    if new_sl < current_sl or current_sl == 0:
                        if self.modify_sl(ticket, new_sl):
                            state.breakeven_hit = True
                            log.info(f"[BREAKEVEN] {ticket}: SL -> {new_sl:.2f} (-{BREAKEVEN_BUFFER} buffer)")

        # ===== 2. TRAILING STOP LOGIC =====
        if USE_TRAILING_SL and state.breakeven_hit:
            trail_threshold = risk_dist * TRAIL_START_MULT

            if profit_dist >= trail_threshold:
                state.trailing_active = True

                if pos_type == mt5.POSITION_TYPE_BUY:
                    new_sl = current_price - TRAIL_DISTANCE
                    if new_sl > current_sl:
                        if self.modify_sl(ticket, new_sl):
                            log.info(f"[TRAILING] {ticket}: SL -> {new_sl:.2f} | Profit: {profit_dist:.2f}")
                else:  # SELL
                    new_sl = current_price + TRAIL_DISTANCE
                    if new_sl < current_sl:
                        if self.modify_sl(ticket, new_sl):
                            log.info(f"[TRAILING] {ticket}: SL -> {new_sl:.2f} | Profit: {profit_dist:.2f}")

        # ===== 3. DYNAMIC TP - Maintain R:R Ratio =====
        if USE_DYNAMIC_TP and state.trailing_active:
            # Recalculate current risk from SL
            if pos_type == mt5.POSITION_TYPE_BUY:
                current_risk = current_price - current_sl
            else:
                current_risk = current_sl - current_price

            if current_risk > 0:
                target_profit = current_risk * RR_RATIO

                if pos_type == mt5.POSITION_TYPE_BUY:
                    new_tp = current_price + target_profit
                    if new_tp > current_tp:
                        if self.modify_tp(ticket, new_tp):
                            log.info(f"[DYNAMIC TP] {ticket}: TP -> {new_tp:.2f} | 1:{RR_RATIO} R:R maintained")
                else:  # SELL
                    new_tp = current_price - target_profit
                    if new_tp < current_tp:
                        if self.modify_tp(ticket, new_tp):
                            log.info(f"[DYNAMIC TP] {ticket}: TP -> {new_tp:.2f} | 1:{RR_RATIO} R:R maintained")

        # ===== 4. PARTIAL TAKE PROFIT =====
        if USE_PARTIAL_TP and not state.partial_closed:
            # Calculate TP distance
            if pos_type == mt5.POSITION_TYPE_BUY:
                tp_dist = current_tp - entry_price if current_tp > 0 else risk_dist * RR_RATIO
            else:
                tp_dist = entry_price - current_tp if current_tp > 0 else risk_dist * RR_RATIO

            partial_trigger = tp_dist * PARTIAL_TP_TRIGGER

            symbol_info = mt5.symbol_info(symbol)
            min_volume = symbol_info.volume_min if symbol_info else 0.01

            if profit_dist >= partial_trigger and volume > min_volume:
                close_volume = volume * (PARTIAL_TP_PCT / 100.0)

                if close_volume >= min_volume:
                    if self.close_partial(ticket, close_volume):
                        state.partial_closed = True
                        log.info(f"[PARTIAL TP] {ticket}: Closed {PARTIAL_TP_PCT}% ({close_volume:.2f} lots)")

        state.last_update = datetime.now()

    def manage_all_positions(self, magic_numbers: list = None) -> None:
        """Manage all open positions"""
        positions = mt5.positions_get()
        if not positions:
            return

        for pos in positions:
            # Filter by magic number if specified
            if magic_numbers and pos.magic not in magic_numbers:
                continue

            try:
                self.manage_position(pos)
            except Exception as e:
                log.error(f"Error managing position {pos.ticket}: {e}")

        # Clean up closed positions from state
        open_tickets = {pos.ticket for pos in positions}
        closed = [t for t in self.position_states if t not in open_tickets]
        for ticket in closed:
            del self.position_states[ticket]
            log.info(f"Position {ticket} closed - removed from tracking")

    def run(self, interval_seconds: float = 1.0):
        """Main loop - manage positions continuously"""
        if not self.connect():
            return

        # Get magic numbers from config
        magic_numbers = [
            acc.get('magic_number')
            for acc in self.accounts.values()
            if acc.get('enabled', True)
        ]

        log.info(f"Managing positions for magic numbers: {magic_numbers}")
        self.running = True

        try:
            while self.running:
                self.manage_all_positions(magic_numbers)
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            log.info("Shutdown requested")
        finally:
            self.disconnect()

    def stop(self):
        """Stop the manager"""
        self.running = False


# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Blue Guardian Dynamic Position Manager")
    parser.add_argument('--config', default='accounts_config.json', help='Config file path')
    parser.add_argument('--interval', type=float, default=1.0, help='Check interval in seconds')

    args = parser.parse_args()

    manager = DynamicPositionManager(args.config)
    manager.run(args.interval)


if __name__ == "__main__":
    main()
