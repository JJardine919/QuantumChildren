
import MetaTrader5 as mt5
import sys
from credential_manager import get_credentials

# FTMO Credentials from .env
_creds = get_credentials('FTMO')
FTMO_ACCOUNTS = [
    {"login": _creds['account'], "password": _creds['password'], "server": _creds['server']}
]

if not mt5.initialize():
    print("MT5 Init Failed")
    sys.exit()

for acc in FTMO_ACCOUNTS:
    print(f"\nChecking Account {acc['login']}...")
    if mt5.login(acc['login'], password=acc['password'], server=acc['server']):
        print("  [OK] Login Successful")
        
        # Check Trade Permissions
        info = mt5.account_info()
        print(f"  Trade Allowed: {info.trade_allowed}")
        print(f"  Mode: {'REAL' if info.trade_mode == 2 else 'DEMO' if info.trade_mode == 0 else 'CONTEST'}")
        
        # Check Symbol Visibility for BTCUSD
        symbol = "BTCUSD"
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"  {symbol} Price: {tick.ask}")
        else:
            print(f"  {symbol} NOT VISIBLE or Market Closed")
            
            # Check what symbols ARE available
            all_symbols = mt5.symbols_get()
            names = [s.name for s in all_symbols if "BTC" in s.name]
            print(f"  Available BTC pairs: {names[:5]}")

    else:
        print(f"  [FAIL] Login Error: {mt5.last_error()}")

mt5.shutdown()
