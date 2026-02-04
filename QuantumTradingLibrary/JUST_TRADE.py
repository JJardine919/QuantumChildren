"""
JUST PLACE A TRADE - NO BS
Run this and tell me what it says.
"""
import MetaTrader5 as mt5

print("="*50)
print("JUST TRADE - Diagnostic")
print("="*50)

# Try Blue Guardian Terminal 1
TERMINAL = r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe"

print(f"\n1. Init MT5...")
if not mt5.initialize(path=TERMINAL):
    print(f"   FAIL: {mt5.last_error()}")
    # Try without path
    print("   Trying default init...")
    if not mt5.initialize():
        print(f"   FAIL: {mt5.last_error()}")
        exit()

print("   OK")

# Get account info
info = mt5.account_info()
if info:
    print(f"\n2. Account: {info.login}")
    print(f"   Server: {info.server}")
    print(f"   Balance: ${info.balance:.2f}")
    print(f"   Trade allowed: {info.trade_allowed}")
    print(f"   Trade expert: {info.trade_expert}")
else:
    print("   No account info!")
    mt5.shutdown()
    exit()

# Check terminal
term = mt5.terminal_info()
print(f"\n3. Terminal:")
print(f"   Connected: {term.connected}")
print(f"   Trade allowed: {term.trade_allowed}")

if not term.trade_allowed:
    print("\n   *** AUTOTRADING DISABLED - Enable it in MT5! ***")

# Find a symbol
print(f"\n4. Finding symbol...")
symbols = mt5.symbols_get()
if not symbols:
    print("   No symbols!")
    mt5.shutdown()
    exit()

# Look for BTC or XAU
symbol = None
for s in symbols:
    if 'BTC' in s.name and s.visible:
        symbol = s.name
        break
    elif 'XAU' in s.name and s.visible:
        symbol = s.name

if not symbol:
    # Just use first visible
    for s in symbols:
        if s.visible:
            symbol = s.name
            break

if not symbol:
    symbol = symbols[0].name
    mt5.symbol_select(symbol, True)

print(f"   Using: {symbol}")

# Get tick
tick = mt5.symbol_info_tick(symbol)
sym_info = mt5.symbol_info(symbol)
if not tick or not sym_info:
    print(f"   Can't get tick for {symbol}")
    mt5.shutdown()
    exit()

print(f"   Bid: {tick.bid}, Ask: {tick.ask}")
print(f"   Min lot: {sym_info.volume_min}")

# Try to place trade
print(f"\n5. Placing BUY order...")

lot = sym_info.volume_min
price = tick.ask

# Filling
filling = mt5.ORDER_FILLING_IOC
if sym_info.filling_mode & mt5.ORDER_FILLING_FOK:
    filling = mt5.ORDER_FILLING_FOK

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "magic": 999999,
    "comment": "TEST",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": filling,
}

print(f"   BUY {lot} {symbol} @ {price}")

result = mt5.order_send(request)

print(f"\n6. Result:")
if result is None:
    print(f"   NULL - {mt5.last_error()}")
else:
    print(f"   Retcode: {result.retcode}")
    print(f"   Comment: {result.comment}")

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("\n   *** SUCCESS! Trade placed! ***")
    else:
        print(f"\n   *** FAILED: {result.comment} ***")

        # Decode error
        codes = {
            10004: "Requote",
            10006: "Request rejected",
            10007: "Canceled by trader",
            10010: "Partial fill",
            10011: "Processing error",
            10012: "Timeout",
            10013: "Invalid request",
            10014: "Invalid volume",
            10015: "Invalid price",
            10016: "Invalid stops",
            10017: "Trade disabled",
            10018: "Market closed",
            10019: "Not enough money",
            10020: "Prices changed",
            10021: "No quotes",
            10022: "Invalid expiration",
            10023: "Order changed",
            10024: "Too many requests",
            10025: "No changes",
            10026: "Autotrading disabled by server",
            10027: "Autotrading disabled by client",
            10028: "Order locked",
            10029: "Long only",
            10030: "Short only",
            10031: "Close only",
            10032: "FIFO close only",
        }
        if result.retcode in codes:
            print(f"   Meaning: {codes[result.retcode]}")

mt5.shutdown()
print("\nDone. Copy this output and show me.")
