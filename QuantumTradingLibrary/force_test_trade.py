import MetaTrader5 as mt5
import time

# GetLeveraged Path
PATH = r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe"

if not mt5.initialize(path=PATH):
    print(f"FAILED to init: {mt5.last_error()}")
    quit()

print(f"Connected to: {mt5.terminal_info().name}")
print(f"Account: {mt5.account_info().login}")

# Check Algo Enabled
if not mt5.terminal_info().trade_allowed:
    print("!!! ALGO TRADING IS DISABLED IN TERMINAL !!!")
    print("Please click the 'Algo Trading' button at the top of MT5.")
    mt5.shutdown()
    quit()

symbol = "BTCUSD"
if not mt5.symbol_select(symbol, True):
    print(f"Failed to select {symbol}")

# Prepare TEST Order
tick = mt5.symbol_info_tick(symbol)
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.05,
    "type": mt5.ORDER_TYPE_BUY,
    "price": tick.ask,
    "magic": 123456,
    "comment": "CONNECTION_TEST",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,
    "deviation": 50
}

print("Sending TEST Order...")
res = mt5.order_send(request)

if res.retcode != mt5.TRADE_RETCODE_DONE:
    print(f"FAILED: {res.comment} ({res.retcode})")
    # Try IOC if FOK failed
    if res.retcode == 10030: # Unsupported filling
        print("Retrying with IOC...")
        request["type_filling"] = mt5.ORDER_FILLING_IOC
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
             print(f"SUCCESS! Ticket: {res.order}")
else:
    print(f"SUCCESS! Ticket: {res.order}")

mt5.shutdown()
