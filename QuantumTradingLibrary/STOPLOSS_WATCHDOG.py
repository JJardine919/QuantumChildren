"""
STOP LOSS WATCHDOG - EMERGENCY POSITION MONITOR
================================================
This script monitors ALL open positions and force-closes
any position that exceeds the loss limit.

Works INDEPENDENTLY of trading scripts.
Closes trades from ANY source (Python, MQL5 EA, manual).

RUN THIS ALONGSIDE YOUR TRADING SCRIPTS.

Author: QuantumChildren
"""

import time
import logging
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5

# Load config - WATCHDOG_LIMIT from MASTER_CONFIG.json
try:
    from config_loader import MAX_LOSS_DOLLARS, AGENT_SL_MAX, ACCOUNTS
    from credential_manager import get_credentials, CredentialError
    LOSS_LIMIT = AGENT_SL_MAX  # Use AGENT_SL_MAX from config ($1.00)
except ImportError:
    LOSS_LIMIT = 1.00  # Fallback only if config unavailable
    CredentialError = Exception  # Fallback

# Use config value - do NOT hardcode
MAX_LOSS_PER_TRADE = LOSS_LIMIT

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][WATCHDOG] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('watchdog_stoploss.log'),
        logging.StreamHandler()
    ]
)


class StopLossWatchdog:
    """
    Monitors all positions and force-closes any that exceed loss limit.
    This is a SAFETY NET - independent of trading logic.
    """

    def __init__(self, account_config: dict, loss_limit: float = MAX_LOSS_PER_TRADE):
        self.account = account_config
        self.loss_limit = loss_limit
        self.connected = False
        self.positions_closed = 0
        self.total_saved = 0.0

    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        if self.connected:
            # Check if still connected
            try:
                acc = mt5.account_info()
                if acc and acc.login == self.account['account']:
                    return True
            except:
                pass
            self.connected = False

        # Initialize MT5
        terminal_path = self.account.get('terminal_path')
        if terminal_path:
            if not mt5.initialize(path=terminal_path):
                logging.error(f"MT5 init failed: {mt5.last_error()}")
                return False
        else:
            if not mt5.initialize():
                logging.error(f"MT5 init failed: {mt5.last_error()}")
                return False

        # Login
        if not mt5.login(
            self.account['account'],
            password=self.account['password'],
            server=self.account['server']
        ):
            logging.error(f"Login failed: {mt5.last_error()}")
            return False

        acc = mt5.account_info()
        if acc:
            logging.info(f"CONNECTED: {self.account.get('name', self.account['account'])} | Balance: ${acc.balance:,.2f}")
            self.connected = True
            return True

        return False

    def close_position(self, position) -> bool:
        """Force close a position"""
        symbol = position.symbol
        ticket = position.ticket
        volume = position.volume
        pos_type = position.type
        profit = position.profit

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error(f"Cannot get tick for {symbol}")
            return False

        # Close in opposite direction
        if pos_type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        # Get filling mode
        symbol_info = mt5.symbol_info(symbol)
        filling_mode = mt5.ORDER_FILLING_IOC
        if symbol_info and symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 50,
            "magic": 999999,  # Watchdog magic number
            "comment": "WATCHDOG_SL",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            saved = abs(profit) - self.loss_limit  # How much worse could it have gotten
            self.positions_closed += 1
            logging.info(f"CLOSED #{ticket} {symbol} @ ${profit:.2f} loss | Prevented further damage")
            return True
        else:
            logging.error(f"CLOSE FAILED #{ticket}: {result.comment if result else 'None'} ({result.retcode if result else 'N/A'})")
            return False

    def check_positions(self):
        """Check all positions and close any exceeding loss limit"""
        positions = mt5.positions_get()
        if positions is None:
            return

        for pos in positions:
            profit = pos.profit

            # Check if loss exceeds limit (profit is negative for losses)
            if profit < -self.loss_limit:
                logging.warning(f"LOSS LIMIT EXCEEDED: #{pos.ticket} {pos.symbol} = ${profit:.2f}")
                self.close_position(pos)

    def show_positions(self):
        """Display current positions"""
        positions = mt5.positions_get()
        if not positions:
            return

        total_profit = 0
        print(f"\n  OPEN POSITIONS (Loss Limit: ${self.loss_limit})")
        print(f"  {'-'*50}")

        for pos in positions:
            profit = pos.profit
            total_profit += profit

            # Color indicator
            status = "OK" if profit > -self.loss_limit else "DANGER"
            direction = "BUY" if pos.type == 0 else "SELL"

            print(f"  #{pos.ticket} {pos.symbol} {direction} {pos.volume} | P/L: ${profit:+.2f} [{status}]")

        print(f"  {'-'*50}")
        print(f"  TOTAL: ${total_profit:+.2f}")

    def run(self, check_interval: int = 5):
        """Main watchdog loop"""
        print("=" * 60)
        print("  STOP LOSS WATCHDOG")
        print(f"  Loss Limit: ${self.loss_limit} per position")
        print(f"  Check Interval: {check_interval}s")
        print("=" * 60)
        print("  This script monitors ALL positions and force-closes")
        print("  any position that exceeds the loss limit.")
        print("=" * 60)

        if not self.connect():
            logging.error("Failed to connect. Exiting.")
            return

        try:
            cycle = 0
            while True:
                cycle += 1

                # Reconnect check
                if not self.connect():
                    time.sleep(10)
                    continue

                # Check positions
                self.check_positions()

                # Show status every 12 cycles (1 minute at 5s interval)
                if cycle % 12 == 0:
                    self.show_positions()
                    acc = mt5.account_info()
                    if acc:
                        logging.info(f"Balance: ${acc.balance:,.2f} | Positions Closed: {self.positions_closed}")

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print(f"\nStopped. Positions closed by watchdog: {self.positions_closed}")
        finally:
            mt5.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stop Loss Watchdog')
    parser.add_argument('--account', '-a', default='ATLAS',
                       help='Account key (ATLAS, BG_INSTANT, BG_CHALLENGE, GL_1, GL_2, GL_3)')
    parser.add_argument('--limit', '-l', type=float, default=MAX_LOSS_PER_TRADE,
                       help=f'Max loss per position in dollars (default: ${MAX_LOSS_PER_TRADE})')
    parser.add_argument('--interval', '-i', type=int, default=5,
                       help='Check interval in seconds (default: 5)')

    args = parser.parse_args()

    # Get account config
    try:
        from config_loader import ACCOUNTS
        from credential_manager import get_credentials, CredentialError
        if args.account not in ACCOUNTS:
            print(f"Unknown account: {args.account}")
            print(f"Available: {list(ACCOUNTS.keys())}")
            return
        account_config = ACCOUNTS[args.account].copy()
        # Ensure password is loaded from credential_manager
        if not account_config.get('password'):
            try:
                creds = get_credentials(args.account)
                account_config['password'] = creds['password']
            except CredentialError as e:
                print(f"Error loading credentials: {e}")
                return
    except ImportError:
        # Fallback - try to load from credential_manager directly
        try:
            from credential_manager import get_credentials, CredentialError
            creds = get_credentials(args.account)
            account_config = creds
            account_config['name'] = args.account
        except Exception as e:
            print(f"Failed to load credentials: {e}")
            return

    watchdog = StopLossWatchdog(account_config, loss_limit=args.limit)
    watchdog.run(check_interval=args.interval)


if __name__ == "__main__":
    main()
