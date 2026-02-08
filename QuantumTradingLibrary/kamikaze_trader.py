
import MetaTrader5 as mt5
import time
import sys
from credential_manager import get_credentials

# Import trading settings from config - DO NOT HARDCODE
from config_loader import MAX_LOSS_DOLLARS, TP_MULTIPLIER

# --- CONFIG ---
_creds = get_credentials('FTMO')
LOGIN = _creds['account']
PASSWORD = _creds['password']
SERVER = _creds['server']
SYMBOL = "BTCUSD"
VOLUME = 0.10
DEVIATION = 20
MAGIC = 666  # The Mark of the Beast (Kamikaze)

def print_red(msg): print(f"KAMIKAZE >> {msg}")

def main():
    print_red("INITIALIZING KAMIKAZE PROTOCOL...")

    # 1. Initialize
    if not mt5.initialize():
        print_red(f"FAILED to init MT5: {mt5.last_error()}")
        return

    # 2. Use Active Account (Do not login explicitly, it crashes the bridge)
    info = mt5.account_info()
    if info is None:
        print_red(f"FAILED to get account info: {mt5.last_error()}")
        return
    
    print_red(f"CONNECTED TO ACCOUNT: {info.login} ({info.server})")
    print_red("READY TO BURN.")

    # 3. Enable Symbol
    selected = mt5.symbol_select(SYMBOL, True)
    if not selected:
        print_red(f"FAILED to select {SYMBOL}. Trying XAUUSD...")
        SYMBOL = "XAUUSD"
        mt5.symbol_select(SYMBOL, True)

    # 4. TRADING LOOP
    while True:
        # Get Price
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            print_red("No Tick Data...")
            time.sleep(1)
            continue

        # Get Bar Data (for direction)
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 1)
        if rates is None:
            time.sleep(1)
            continue
            
        current_open = rates[0]['open']
        current_close = rates[0]['close'] # approximates current price
        
        # DECISION
        signal = "HOLD"
        direction = 0
        
        # Calculate SL/TP from config (MAX_LOSS_DOLLARS, TP_MULTIPLIER)
        symbol_info = mt5.symbol_info(SYMBOL)
        tick_value = symbol_info.trade_tick_value if symbol_info else 1.0
        tick_size = symbol_info.trade_tick_size if symbol_info else 0.01
        if tick_value > 0 and VOLUME > 0:
            sl_ticks = MAX_LOSS_DOLLARS / (tick_value * VOLUME)
            sl_distance = sl_ticks * tick_size
        else:
            sl_distance = 50.0
        tp_distance = sl_distance * TP_MULTIPLIER

        if tick.bid > current_open:
            signal = "BUY"
            price = tick.ask
            type_op = mt5.ORDER_TYPE_BUY
            sl = price - sl_distance
            tp = price + tp_distance
        else:
            signal = "SELL"
            price = tick.bid
            type_op = mt5.ORDER_TYPE_SELL
            sl = price + sl_distance
            tp = price - tp_distance

        print_red(f"{SYMBOL} | Open: {current_open} | Current: {tick.bid} | SIGNAL: {signal}")

        # EXECUTE (If no position exists)
        positions = mt5.positions_get(symbol=SYMBOL)
        if len(positions) == 0:
            print_red(f"EXECUTING {signal} FIRE...")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": VOLUME,
                "type": type_op,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": DEVIATION,
                "magic": MAGIC,
                "comment": "KAMIKAZE_V1",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print_red(f"ORDER FAILED: {result.comment} ({result.retcode})")
            else:
                print_red(f"ORDER FILLED: {result.order}")
        else:
            print_red(f"Position held. PnL: {positions[0].profit}")

        time.sleep(1) # Check every second

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKAMIKAZE ABORTED.")
        mt5.shutdown()
