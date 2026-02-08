import MetaTrader5 as mt5
import time
from credential_manager import get_credentials, CredentialError

# --- CONFIG FOR GETLEVERAGED #3 (107245) ---
creds = get_credentials('GL_3')
ACCOUNT = creds['account']
PASSWORD = creds['password']
SERVER = creds['server']
SYMBOL = "BTCUSD"
VOLUME = 0.01
MAGIC = 7773  # Unique Magic

def log(msg):
    print(f"GL_BOT_3 >> {msg}")
    try:
        with open("passive_gl_3.log", "a") as f:
            f.write(f"{msg}\n")
    except:
        pass

def main():
    # Try to initialize MT5 (works if already running)
    if not mt5.initialize():
        log(f"Failed to connect: {mt5.last_error()}")
        log("Make sure MT5 is running")
        return

    # Login to specific account
    if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
        log(f"Login failed: {mt5.last_error()}")
        return

    acc = mt5.account_info()
    if not acc:
        log("Could not get account info")
        return
    log(f"Trading on Account: {acc.login} (Magic {MAGIC})")

    log("Strategy Active. Monitoring BTCUSD...")
    while True:
        if not mt5.terminal_info().connected:
            time.sleep(5); continue
        
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick: time.sleep(1); continue

        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 1)
        if not rates: time.sleep(1); continue

        current_open = rates[0]['open']
        if tick.bid > current_open:
            action = mt5.ORDER_TYPE_BUY; price = tick.ask; label = "BUY"
        else:
            action = mt5.ORDER_TYPE_SELL; price = tick.bid; label = "SELL"

        pos = mt5.positions_get(symbol=SYMBOL)
        if pos is None: pos = []
        my_pos = [p for p in pos if p.magic == MAGIC]

        if len(my_pos) == 0:
            log(f"FIRE: {label} @ {price}")
            
            fill_mode = mt5.ORDER_FILLING_IOC 
            s_info = mt5.symbol_info(SYMBOL)
            if s_info and not (s_info.filling_mode & mt5.SYMBOL_FILLING_IOC):
                fill_mode = mt5.ORDER_FILLING_FOK
            
            sl_points = 5000; tp_points = 15000
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
                "comment": "KAMIKAZE_P_3",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode,
                "deviation": 50,
            }
            
            res = mt5.order_send(request)
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                log(f"ERROR: {res.comment}")
            else:
                log(f"SUCCESS: {label} Opened.")
        
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
