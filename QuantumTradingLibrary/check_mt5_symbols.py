import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize():
    print("Failed to initialize MT5")
    quit()

symbols = mt5.symbols_get()
print(f"Total symbols found: {len(symbols)}")

# Check some likely symbols
target_symbols = ["EURUSD", "XAUUSD", "ETHUSD", "BTCUSD"]
for ts in target_symbols:
    # Try with and without .ecn
    found = False
    for s in symbols:
        if ts in s.name:
            print(f"Found symbol: {s.name}")
            rates = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_M1, 0, 10)
            if rates is not None:
                print(f"  - History available: Yes ({len(rates)} bars)")
            else:
                print(f"  - History available: No")
            found = True
    if not found:
        print(f"Symbol {ts} not found in any form.")

mt5.shutdown()
