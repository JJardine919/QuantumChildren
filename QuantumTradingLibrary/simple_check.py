import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"MT5 Init Failed: {mt5.last_error()}")
    quit()

print(f"Connected to: {mt5.terminal_info().name}")
print(f"Path: {mt5.terminal_info().path}")

account = mt5.account_info()
if account:
    print(f"\n--- ACCOUNT ---")
    print(f"Login: {account.login}")
    print(f"Server: {account.server}")
    print(f"Trade Allowed: {account.trade_allowed}")
    print(f"Algo Trading Allowed: {account.trade_expert}")
else:
    print("No account info.")

print(f"\n--- SYMBOLS MATCHING 'BTC' ---")
symbols = mt5.symbols_get()
btc_symbols = [s for s in symbols if "BTC" in s.name]

if not btc_symbols:
    print("No symbols with 'BTC' found in Market Watch.")
    print("Attempting to find all...")
    # Try to find common ones
    for s in ["BTCUSD", "Bitcoin", "BTCUSD.pro", "BTCUSD.r"]:
        info = mt5.symbol_info(s)
        if info:
            print(f"Found specific check: {s}")
else:
    for s in btc_symbols:
        print(f"Symbol: {s.name}")
        print(f"  - Visible: {s.visible}")
        print(f"  - Min Volume: {s.volume_min}")
        print(f"  - Max Volume: {s.volume_max}")
        print(f"  - Volume Step: {s.volume_step}")
        print(f"  - Trade Mode: {s.trade_mode} (4=Full)")
        
        # Check specific config match
        if s.name == "BTCUSD":
            print("  >>> MATCHES CONFIG 'BTCUSD'")
            
            # Filling Modes
            fill_flags = s.filling_mode
            print(f"  - Filling Flags: {fill_flags}")
            print(f"    - FOK: {(fill_flags & mt5.SYMBOL_FILLING_FOK) != 0}")
            print(f"    - IOC: {(fill_flags & mt5.SYMBOL_FILLING_IOC) != 0}")
            # print(f"    - RETURN: {(fill_flags & mt5.SYMBOL_FILLING_RETURN) != 0}") # Some versions might differ
            
            # Execution Mode
            # 0=Request, 1=Instant, 2=Market, 3=Exchange
            print(f"  - Execution Mode: {s.trade_mode} (Wait, trade_mode is access level. execution_mode is separate)")
            # Actually execution_mode is not directly on symbol_info in Python API usually, it's inferred or implicit. 
            # But we can check trade_exemode if available in newer bindings, or just rely on filling.
            
        else:
            print(f"  >>> DOES NOT MATCH CONFIG 'BTCUSD' (Config needs update?)")

mt5.shutdown()