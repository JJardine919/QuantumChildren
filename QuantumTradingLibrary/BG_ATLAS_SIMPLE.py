"""
SIMPLE ATLAS-STYLE TRADER FOR BLUE GUARDIAN
============================================
Matches EXACTLY what's shown on Atlas screenshot:
- Multiple BUY positions on BTCUSD
- Lot: 0.06-0.07
- TP: ~450 points above entry (no visible SL)
- Grid scaling on dips

Run: python BG_ATLAS_SIMPLE.py

Just works. No BS.
"""

import MetaTrader5 as mt5
import time
import random
from datetime import datetime

# ==============================================================================
# BLUE GUARDIAN ACCOUNTS - PICK ONE OR BOTH
# ==============================================================================

ACCOUNTS = [
    {
        'account': 366604,
        'password': 'YF^oHH&4Nm',
        'server': 'BlueGuardian-Server',
        'terminal': r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
        'magic': 366001,
        'name': 'BG $5K Instant',
    },
    {
        'account': 365060,
        'password': ')8xaE(gAuU',
        'server': 'BlueGuardian-Server',
        'terminal': r"C:\Program Files\Blue Guardian MT5 Terminal 2\terminal64.exe",
        'magic': 365001,
        'name': 'BG $100K Challenge',
    },
]

# Trading settings - ATLAS STYLE
SYMBOL = 'BTCUSD'
LOT_MIN = 0.06
LOT_MAX = 0.07
TP_POINTS = 450          # ~$450 on BTC
MAX_POSITIONS = 5        # Max grid positions
GRID_SPACING = 100       # Points between entries
CHECK_INTERVAL = 30      # Seconds between checks


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def connect(acc):
    """Connect to MT5 account"""
    if not mt5.initialize(path=acc['terminal']):
        log(f"MT5 init failed: {mt5.last_error()}")
        return False

    if not mt5.login(acc['account'], password=acc['password'], server=acc['server']):
        log(f"Login failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    log(f"Connected: {acc['name']} | Balance: ${info.balance:,.2f}")
    return True


def get_positions(magic):
    """Get our positions"""
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return []
    return [p for p in positions if p.magic == magic]


def get_last_entry_price(magic):
    """Get most recent entry price"""
    positions = get_positions(magic)
    if not positions:
        return 0

    latest = max(positions, key=lambda p: p.time)
    return latest.price_open


def should_open_position(magic, current_price):
    """Check if we should open a new grid position"""
    positions = get_positions(magic)
    count = len(positions)

    # Not at max positions
    if count >= MAX_POSITIONS:
        return False

    # First position - always open
    if count == 0:
        return True

    # Check grid spacing - only add if price dropped enough from last entry
    last_entry = get_last_entry_price(magic)
    if last_entry == 0:
        return True

    symbol_info = mt5.symbol_info(SYMBOL)
    spacing = GRID_SPACING * symbol_info.point

    # Open if price is BELOW last entry by grid spacing (scaling into dip)
    if current_price <= last_entry - spacing:
        return True

    return False


def open_buy(magic, level):
    """Open a BUY position - ATLAS STYLE"""
    symbol_info = mt5.symbol_info(SYMBOL)
    if not symbol_info:
        log(f"Symbol {SYMBOL} not found")
        return False

    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)
        time.sleep(0.5)

    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        log("No tick data")
        return False

    # Random lot between 0.06 and 0.07
    lot = LOT_MIN if random.random() < 0.5 else LOT_MAX

    price = tick.ask
    tp = price + (TP_POINTS * symbol_info.point)

    # Determine filling mode
    filling = mt5.ORDER_FILLING_IOC
    if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
        filling = mt5.ORDER_FILLING_FOK

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": 0,  # NO VISIBLE SL - ATLAS STYLE
        "tp": tp,
        "magic": magic,
        "comment": f"BUY L{level} placed by expert",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    result = mt5.order_send(request)

    if result is None:
        log(f"Order failed: {mt5.last_error()}")
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"OPENED: BUY L{level} @ {price:.2f} | Lot: {lot} | TP: {tp:.2f}")
        return True
    else:
        log(f"Order rejected: {result.comment} ({result.retcode})")
        return False


def run_account(acc):
    """Run trading cycle for one account"""
    if not connect(acc):
        return

    magic = acc['magic']

    while True:
        try:
            # Get current positions and price
            positions = get_positions(magic)
            tick = mt5.symbol_info_tick(SYMBOL)

            if not tick:
                time.sleep(5)
                continue

            current_price = tick.ask
            pos_count = len(positions)

            # Show status
            total_profit = sum(p.profit for p in positions) if positions else 0
            log(f"{acc['name']} | Positions: {pos_count}/{MAX_POSITIONS} | Price: {current_price:.2f} | Profit: ${total_profit:.2f}")

            # Check if we should open new position
            if should_open_position(magic, current_price):
                level = pos_count + 1
                log(f"Opening grid level {level}...")
                open_buy(magic, level)

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log("Stopping...")
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(10)

    mt5.shutdown()


def main():
    print("=" * 60)
    print("  BLUE GUARDIAN - ATLAS STYLE TRADER")
    print("  Simple grid BUY system matching Atlas performance")
    print("=" * 60)
    print(f"  Symbol: {SYMBOL}")
    print(f"  Lots: {LOT_MIN}-{LOT_MAX}")
    print(f"  TP: {TP_POINTS} points")
    print(f"  Max Positions: {MAX_POSITIONS}")
    print(f"  Grid Spacing: {GRID_SPACING} points")
    print("=" * 60)

    # Ask which account
    print("\nSelect account:")
    print("  1. BG $5K Instant (366604)")
    print("  2. BG $100K Challenge (365060)")
    print("  3. Both (alternating)")

    choice = input("\nChoice [1/2/3]: ").strip()

    if choice == '1':
        run_account(ACCOUNTS[0])
    elif choice == '2':
        run_account(ACCOUNTS[1])
    elif choice == '3':
        # Alternate between accounts
        idx = 0
        while True:
            acc = ACCOUNTS[idx]
            log(f"\n--- Switching to {acc['name']} ---")

            if connect(acc):
                magic = acc['magic']
                positions = get_positions(magic)
                tick = mt5.symbol_info_tick(SYMBOL)

                if tick:
                    current_price = tick.ask
                    pos_count = len(positions)
                    total_profit = sum(p.profit for p in positions) if positions else 0

                    log(f"Positions: {pos_count}/{MAX_POSITIONS} | Profit: ${total_profit:.2f}")

                    if should_open_position(magic, current_price):
                        open_buy(magic, pos_count + 1)

            idx = (idx + 1) % 2
            time.sleep(CHECK_INTERVAL)
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
