import MetaTrader5 as mt5
from datetime import datetime
import time

if not mt5.initialize():
    print(f"MT5 Init Failed: {mt5.last_error()}")
    quit()

print(f"Connected to: {mt5.terminal_info().name}")
print(f"Server Time: {datetime.fromtimestamp(mt5.symbol_info_tick('BTCUSD').time)}")
print(f"Local Time:  {datetime.now()}")

symbol = "BTCUSD"
tick = mt5.symbol_info_tick(symbol)

if tick:
    print(f"\nTick Data for {symbol}:")
    print(f"  Time: {datetime.fromtimestamp(tick.time)}")
    print(f"  Bid:  {tick.bid}")
    print(f"  Ask:  {tick.ask}")
    
    # Check if data is old (stale)
    lag = time.time() - tick.time
    print(f"  Lag:  {lag:.1f} seconds")
    
    if lag > 60:
        print("  >>> DATA IS STALE (Market Closed or No Connection)")
    else:
        print("  >>> DATA IS FRESH")
else:
    print("No tick data available.")

mt5.shutdown()