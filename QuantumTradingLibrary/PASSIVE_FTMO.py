import MetaTrader5 as mt5
import time

# --- CONFIG FOR FTMO ---
SYMBOL = "BTCUSD"
VOLUME = 0.05 
MAGIC = 888 
TERMINAL_PATH = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"

def log(msg): print(f"FTMO_BOT >> {msg}")

def main():
    # 1. Initialize strictly to FTMO path
    if not mt5.initialize(path=TERMINAL_PATH):
        log(f"Failed to connect to FTMO Terminal: {mt5.last_error()}")
        return

    # 2. Verify we are on the right terminal
    info = mt5.terminal_info()
    log(f"Connected to: {info.name}")
    
    # 3. Passive Account Check
    acc = mt5.account_info()
    if not acc:
        log("Please log into your FTMO account manually.")
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

        # Check existing positions for THIS bot (Magic 888)
        pos = mt5.positions_get(symbol=SYMBOL)
        if pos is None: pos = []
        my_pos = [p for p in pos if p.magic == MAGIC]

        if len(my_pos) == 0:
            log(f"FIRE: {label} @ {price} (Open: {current_open})")
            
            # Auto-detect filling
            fill_mode = mt5.ORDER_FILLING_FOK
            s_info = mt5.symbol_info(SYMBOL)
            if s_info and (s_info.filling_mode & 2): fill_mode = mt5.ORDER_FILLING_IOC
            
            # SL/TP (BTC Scale)
            point = s_info.point
            sl = price - (5000 * point) if action == mt5.ORDER_TYPE_BUY else price + (5000 * point)
            tp = price + (10000 * point) if action == mt5.ORDER_TYPE_BUY else price - (10000 * point)

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
