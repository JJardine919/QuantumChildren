import MetaTrader5 as mt5
import time
import sys

# --- CONFIGURATION (NO PASSWORDS HERE) ---
SYMBOL = "BTCUSD"
VOLUME = 0.05
MAGIC = 777
DEVIATION = 50

def print_line(msg):
    print(f"PASSIVE_KAMIKAZE >> {msg}")

def main():
    # 1. Initialize (Connect to whatever is open)
    if not mt5.initialize():
        print_line(f"FAILED to connect to Terminal: {mt5.last_error()}")
        print_line("Please open your MetaTrader 5 Terminal first.")
        return

    # 2. Check Connection
    info = mt5.terminal_info()
    if not info:
        print_line("Could not get terminal info.")
        return

    print_line(f"CONNECTED TO: {info.name}")
    print_line(f"PATH: {info.path}")
    
    # 3. Check Account (Read-Only)
    account = mt5.account_info()
    if not account:
        print_line("No account logged in! Please log in manually.")
        return
        
    print_line(f"ACCOUNT: {account.login} (Server: {account.server})")
    print_line(f"BALANCE: {account.balance} {account.currency}")

    # 4. Check Algo Trading Permissions
    if not info.trade_allowed:
        print_line("CRITICAL WARNING: 'Algo Trading' is DISABLED in the terminal.")
        print_line("Please click the 'Algo Trading' button in MT5 toolbar to enable it.")
        # We don't exit, we let the loop run so it starts working when you click the button

    # 5. Symbol Setup
    if not mt5.symbol_select(SYMBOL, True):
        print_line(f"Failed to select {SYMBOL}. Is it in your Market Watch?")
    
    print_line("Strategy Started. Waiting for setup...")

    while True:
        # Heartbeat check
        if not mt5.terminal_info().connected:
            print_line("Terminal disconnected. Waiting...")
            time.sleep(5)
            continue

        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            time.sleep(1)
            continue

        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 1)
        if not rates:
            time.sleep(1)
            continue

        # Logic: Simple 1-minute Trend Follow
        current_open = rates[0]['open']
        
        if tick.bid > current_open:
            action = mt5.ORDER_TYPE_BUY
            price = tick.ask
            label = "BUY"
            sl_dist = 50000 * mt5.symbol_info(SYMBOL).point # Wide SL
        else:
            action = mt5.ORDER_TYPE_SELL
            price = tick.bid
            label = "SELL"
            sl_dist = 50000 * mt5.symbol_info(SYMBOL).point

        # Check existing positions
        pos = mt5.positions_get(symbol=SYMBOL)
        if pos is None: pos = []
        
        # Only check OUR trades (Magic Number)
        my_pos = [p for p in pos if p.magic == MAGIC]

        if len(my_pos) == 0:
            print_line(f"SIGNAL: {label} @ {price}")
            
            # Determine best filling mode (Auto-detect)
            fill_mode = mt5.ORDER_FILLING_FOK
            s_info = mt5.symbol_info(SYMBOL)
            if (s_info.filling_mode & mt5.SYMBOL_FILLING_IOC):
                fill_mode = mt5.ORDER_FILLING_IOC
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": VOLUME,
                "type": action,
                "price": price,
                "magic": MAGIC,
                "comment": "PASSIVE_V1",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode,
                "deviation": DEVIATION,
            }
            
            res = mt5.order_send(request)
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                print_line(f"ORDER FAILED: {res.comment} ({res.retcode})")
            else:
                print_line(f"ORDER FILLED: {label} Volume: {res.volume}")
        
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPASSIVE_KAMIKAZE >> Stopped by user.")
