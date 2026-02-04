
import MetaTrader5 as mt5

def find_btc_symbol():
    if not mt5.initialize():
        print("MT5 Init Failed")
        return

    print("Searching for BTC symbols...")
    symbols = mt5.symbols_get()
    btc_symbols = [s.name for s in symbols if "BTC" in s.name]
    
    print(f"Found {len(btc_symbols)} BTC symbols:")
    for s in btc_symbols:
        print(f" - {s}")

    mt5.shutdown()

if __name__ == "__main__":
    find_btc_symbol()
