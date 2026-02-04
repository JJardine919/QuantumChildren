
import MetaTrader5 as mt5
import sys

def test():
    if not mt5.initialize():
        print("MT5 Init Failed")
        return

    accs = [
        {"login": 1512338719, "password": "l6SxHm$@", "server": "FTMO-Demo"},
        {"login": 1512287880, "password": "1a3Q@fT24@LEw", "server": "FTMO-Demo"}
    ]

    for acc in accs:
        res = mt5.login(acc["login"], password=acc["password"], server=acc["server"])
        if res:
            print(f"Login OK: {acc['login']}")
            # Check BTCUSD
            info = mt5.symbol_info("BTCUSD")
            if info:
                print(f"  BTCUSD found for {acc['login']}")
            else:
                # Try suffix
                symbols = mt5.symbols_get()
                matches = [s.name for s in symbols if "BTCUSD" in s.name]
                print(f"  BTCUSD NOT found. Matches: {matches}")
        else:
            print(f"Login FAILED: {acc['login']} - Error: {mt5.last_error()}")

    mt5.shutdown()

if __name__ == "__main__":
    test()
