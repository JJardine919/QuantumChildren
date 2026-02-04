import MetaTrader5 as mt5
import time
import sys

# --- GETLEVERAGED ACCOUNTS ---
GL_ACCOUNTS = {
    1: {'account': 113326, 'password': '%bwN)IvJ5F', 'server': 'GetLeveraged-Trade'},
    2: {'account': 113328, 'password': 'H*M5c7jpR7', 'server': 'GetLeveraged-Trade'},
    3: {'account': 107245, 'password': '$86eCmFbXR', 'server': 'GetLeveraged-Trade'},
}

# --- CONFIG ---
SYMBOL = "BTCUSD"
VOLUME = 0.01
MAGIC = 777

def log(msg):
    print(f"GL_BOT >> {msg}")
    try:
        with open("passive_gl.log", "a") as f:
            f.write(f"{msg}\n")
    except:
        pass

def main():
    # Select account from command line or default to 1
    acct_num = 1
    if len(sys.argv) > 1:
        try:
            acct_num = int(sys.argv[1])
        except:
            pass

    if acct_num not in GL_ACCOUNTS:
        log(f"Invalid account number. Use 1, 2, or 3")
        return

    acct = GL_ACCOUNTS[acct_num]
    log(f"Using GetLeveraged Account {acct_num}: {acct['account']}")

    # Initialize MT5 (works if already running)
    if not mt5.initialize():
        log(f"Failed to connect: {mt5.last_error()}")
        log("Make sure MT5 is running")
        return

    # Login to specific account
    if not mt5.login(acct['account'], password=acct['password'], server=acct['server']):
        log(f"Login failed: {mt5.last_error()}")
        return

    # Verify connection
    info = mt5.terminal_info()
    log(f"Connected to: {info.name}")

    acc = mt5.account_info()
    if not acc:
        log("Could not get account info")
        return
    log(f"Trading on Account: {acc.login}")

    # 4. Trading Loop
    log("Strategy Active. Monitoring BTCUSD...")
    while True:
        if not mt5.terminal_info().connected:
            time.sleep(5); continue
        
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            time.sleep(1); continue

        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 1)
        if not rates or len(rates) == 0:
            time.sleep(1); continue

        # --- KAMIKAZE LOGIC ---
        current_open = rates[0]['open']
        if tick.bid > current_open:
            action = mt5.ORDER_TYPE_BUY
            price = tick.ask
            label = "BUY"
        else:
            action = mt5.ORDER_TYPE_SELL
            price = tick.bid
            label = "SELL"

        # Check existing positions for THIS bot (Magic 777)
        pos = mt5.positions_get(symbol=SYMBOL)
        if pos is None: pos = []
        my_pos = [p for p in pos if p.magic == MAGIC]

        if len(my_pos) == 0:
            log(f"FIRE: {label} @ {price} (Open: {current_open})")
            
            # Auto-detect filling (Prefer IOC for GetLeveraged)
            fill_mode = mt5.ORDER_FILLING_IOC 
            # Fallback to FOK only if IOC is explicitly missing (rare)
            s_info = mt5.symbol_info(SYMBOL)
            if s_info and not (s_info.filling_mode & mt5.SYMBOL_FILLING_IOC): 
                fill_mode = mt5.ORDER_FILLING_FOK
            
            # SL/TP (BTC Scale)
            # User requested 50 cents risk @ 0.01 vol => $50 price distance.
            # Assuming point is 0.01, 5000 points = $50.00
            # TP set to 15000 points ($150.00) for a 1:3 ratio.
            sl_points = 5000 
            tp_points = 15000
            
            point = s_info.point
            sl = price - (sl_points * point) if action == mt5.ORDER_TYPE_BUY else price + (sl_points * point)
            tp = price + (tp_points * point) if action == mt5.ORDER_TYPE_BUY else price - (tp_points * point)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": VOLUME,
                "type": action,
                "price": price,
                "sl": sl,
                "tp": tp,
                "magic": MAGIC,
                "comment": "KAMIKAZE_P",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode,
                "deviation": 50,
            }
            
            res = mt5.order_send(request)
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                log(f"ERROR: {res.comment} ({res.retcode})")
            else:
                log(f"SUCCESS: {label} Opened.")
        
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
