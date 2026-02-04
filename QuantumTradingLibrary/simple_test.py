
import MetaTrader5 as mt5
import time

print("INIT...")
time.sleep(2)
if not mt5.initialize():
    print(f"FAILED: {mt5.last_error()}")
    quit()

print(f"ACCOUNT: {mt5.account_info().login}")
symbol = "BTCUSD"
mt5.symbol_select(symbol, True)
tick = mt5.symbol_info_tick(symbol)

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_BUY,
    "price": tick.ask,
    "magic": 999,
    "comment": "SIMPLE_TEST",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

result = mt5.order_send(request)
print(f"RESULT: {result.retcode} - {result.comment}")
mt5.shutdown()
