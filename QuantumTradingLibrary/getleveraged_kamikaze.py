
import MetaTrader5 as mt5
import time
import sys
from credential_manager import get_credentials, CredentialError

# --- GETLEVERAGED CONFIG ---
# Load credentials from credential_manager
try:
    gl1_creds = get_credentials('GL_1')
    gl2_creds = get_credentials('GL_2')
except CredentialError as e:
    print(f"Error loading credentials: {e}")
    sys.exit(1)

ACCOUNTS = {
    gl1_creds['account']: {
        "password": gl1_creds['password'],
        "server": gl1_creds['server'],
        "volume": 0.01,
        "magic": gl1_creds['magic']
    },
    gl2_creds['account']: {
        "password": gl2_creds['password'],
        "server": gl2_creds['server'],
        "volume": 0.05,
        "magic": gl2_creds['magic']
    }
}

# Defaults
DEFAULT_LOGIN = gl1_creds['account']
SYMBOL = "BTCUSD"

def print_red(msg):
    print(f"GETLEVERAGED_KAMIKAZE >> {msg}")
    try:
        with open("kamikaze.log", "a") as f:
            f.write(f"{msg}\n")
    except:
        pass

def ensure_login():
    """Smart login that avoids re-logging if already connected."""
    current_account = mt5.account_info()

    if current_account:
        print_red(f"Current Terminal Login: {current_account.login}")
        if current_account.login in ACCOUNTS:
            print_red(f"Already logged into valid account {current_account.login}. Skipping login to prevent lag.")
            return current_account.login
        else:
            print_red(f"Account {current_account.login} is not in our GetLeveraged list. Switching...")

    # Needs login
    target = DEFAULT_LOGIN
    creds = ACCOUNTS[target]

    print_red(f"Logging into GetLeveraged Account {target}...")
    if not mt5.login(target, password=creds["password"], server=creds["server"]):
        print_red(f"Login Failed: {mt5.last_error()}")
        return None

    return target

def main():
    # FORCE PATH to GetLeveraged Terminal
    GL_PATH = r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe"

    if not mt5.initialize(path=GL_PATH):
        print_red(f"Init Failed (Path: {GL_PATH}): {mt5.last_error()}")
        # Try default init if path fails
        if not mt5.initialize():
            print_red("Fatal: MT5 Initialize failed completely.")
            return

    # 1. Smart Login
    login_id = ensure_login()
    if not login_id:
        return

    # 2. Verify Algo Trading
    if not mt5.terminal_info().trade_allowed:
        print_red("WARNING: AutoTrading is DISABLED in Terminal! Please enable it.")
        # We don't exit, just warn, in case user enables it while running

    # 3. Get Account Config
    config = ACCOUNTS[login_id]
    VOLUME = config["volume"]
    MAGIC = config["magic"]

    print_red(f"RUNNING ON ACCOUNT {login_id} | Magic: {MAGIC} | Vol: {VOLUME}")

    # Force symbol selection
    if not mt5.symbol_select(SYMBOL, True):
        print_red(f"Failed to select {SYMBOL}. Attempting to continue anyway...")

    print_red("monitoring market...")

    iters = 0
    while True:
        iters += 1
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            time.sleep(1)
            continue

        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 1)
        if not rates:
            time.sleep(1)
            continue

        # Simple Trend Following
        current_open = rates[0]['open']
        if tick.bid > current_open:
            action = mt5.ORDER_TYPE_BUY
            price = tick.ask
            label = "BUY"
        else:
            action = mt5.ORDER_TYPE_SELL
            price = tick.bid
            label = "SELL"

        # Heartbeat every ~10s
        if iters % 10 == 0:
            print_red(f"Alive | {SYMBOL} | Bid: {tick.bid} | Open: {current_open} | Action: {label}")

        # Check existing
        pos = mt5.positions_get(symbol=SYMBOL)
        if pos is None: pos = [] # Safety

        # Filter positions by magic number to avoid messing with other strats/manual trades
        my_pos = [p for p in pos if p.magic == MAGIC]

        if len(my_pos) == 0:
            print_red(f"FIRE: {label} @ {price} (Bid: {tick.bid}, Open: {current_open})")

            # Determine filling mode
            fill_mode = mt5.ORDER_FILLING_FOK # Default safer
            s_info = mt5.symbol_info(SYMBOL)
            if s_info:
                # If specifically allows IOC, use it (often better for market)
                if (s_info.filling_mode & 2) != 0:
                    fill_mode = mt5.ORDER_FILLING_IOC
                elif (s_info.filling_mode & 1) != 0:
                    fill_mode = mt5.ORDER_FILLING_FOK

            # --- RISK MANAGEMENT ---
            # User Goal: 50 cents risk per trade.
            # Volume: 0.01
            # Math: $0.50 / 0.01 = $50.00 Price Distance
            SL_DISTANCE = 50.0 # BTCUSD Price Points

            if action == mt5.ORDER_TYPE_BUY:
                sl = price - SL_DISTANCE
            else:
                sl = price + SL_DISTANCE

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": VOLUME,
                "type": action,
                "price": price,
                "sl": sl,
                "magic": MAGIC,
                "comment": "REVENGE_V1_SL",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode,
                "deviation": 50, # Slippage tolerance
            }
            res = mt5.order_send(request)
            if res.retcode != mt5.TRADE_RETCODE_DONE:
                print_red(f"FAILED: {res.comment} ({res.retcode})")
            else:
                print_red(f"ORDER FILLED: {res.order} | SL: {sl}")
        else:
            # print_red(f"Holding {label}. PnL: {my_pos[0].profit}")
            pass

        time.sleep(1) # Reduced sleep for better response

if __name__ == "__main__":
    main()
