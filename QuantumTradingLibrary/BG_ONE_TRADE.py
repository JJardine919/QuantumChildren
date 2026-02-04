"""
ONE TRADE TEST - Blue Guardian
Just connects and places ONE BUY. That's it.
"""

import MetaTrader5 as mt5

# Blue Guardian $5K
ACCOUNT = 366604
PASSWORD = 'YF^oHH&4Nm'
SERVER = 'BlueGuardian-Server'
TERMINAL = r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe"

print("=" * 50)
print("BLUE GUARDIAN - ONE TRADE TEST")
print("=" * 50)

# Step 1: Initialize
print("\n1. Initializing MT5...")
if not mt5.initialize(path=TERMINAL):
    print(f"   FAILED: {mt5.last_error()}")
    exit()
print("   OK")

# Step 2: Login
print("\n2. Logging in...")
if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
    print(f"   FAILED: {mt5.last_error()}")
    mt5.shutdown()
    exit()

info = mt5.account_info()
print(f"   OK - Balance: ${info.balance:.2f}")

# Step 3: Check symbol
print("\n3. Checking BTCUSD...")
symbol_info = mt5.symbol_info("BTCUSD")
if not symbol_info:
    print("   BTCUSD not found!")
    # Try to find what symbols exist
    symbols = mt5.symbols_get()
    btc_symbols = [s.name for s in symbols if 'BTC' in s.name][:10]
    print(f"   Available BTC symbols: {btc_symbols}")
    mt5.shutdown()
    exit()

if not symbol_info.visible:
    print("   Selecting symbol...")
    mt5.symbol_select("BTCUSD", True)

print(f"   OK - Min lot: {symbol_info.volume_min}, Point: {symbol_info.point}")

# Step 4: Get price
print("\n4. Getting price...")
tick = mt5.symbol_info_tick("BTCUSD")
if not tick:
    print(f"   FAILED: {mt5.last_error()}")
    mt5.shutdown()
    exit()
print(f"   Bid: {tick.bid}, Ask: {tick.ask}")

# Step 5: Check if trading allowed
print("\n5. Checking trading permissions...")
terminal_info = mt5.terminal_info()
print(f"   Trade allowed: {terminal_info.trade_allowed}")
print(f"   Expert enabled: {terminal_info.trade_expert}")

if not terminal_info.trade_allowed:
    print("\n   *** AUTOTRADING IS DISABLED IN MT5 ***")
    print("   Click the AutoTrading button in MT5 toolbar!")
    mt5.shutdown()
    exit()

# Step 6: Place the trade
print("\n6. Placing BUY order...")

price = tick.ask
tp = price + (450 * symbol_info.point)
lot = 0.06

# Filling mode
filling = mt5.ORDER_FILLING_IOC
if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
    filling = mt5.ORDER_FILLING_FOK

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": "BTCUSD",
    "volume": lot,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "sl": 0,
    "tp": tp,
    "magic": 366001,
    "comment": "BG_TEST",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": filling,
}

print(f"   Request: BUY {lot} BTCUSD @ {price:.2f}, TP: {tp:.2f}")

result = mt5.order_send(request)

if result is None:
    print(f"   FAILED: {mt5.last_error()}")
else:
    print(f"   Return code: {result.retcode}")
    print(f"   Comment: {result.comment}")

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("\n   *** SUCCESS - TRADE PLACED! ***")
    else:
        print(f"\n   *** FAILED - {result.comment} ***")

        # Common error codes
        if result.retcode == 10004:
            print("   Error 10004: Requote - price changed")
        elif result.retcode == 10006:
            print("   Error 10006: Request rejected")
        elif result.retcode == 10007:
            print("   Error 10007: Request canceled by trader")
        elif result.retcode == 10010:
            print("   Error 10010: Only part of request completed")
        elif result.retcode == 10011:
            print("   Error 10011: Processing error")
        elif result.retcode == 10012:
            print("   Error 10012: Request canceled due to timeout")
        elif result.retcode == 10013:
            print("   Error 10013: Invalid request")
        elif result.retcode == 10014:
            print("   Error 10014: Invalid volume")
        elif result.retcode == 10015:
            print("   Error 10015: Invalid price")
        elif result.retcode == 10016:
            print("   Error 10016: Invalid stops")
        elif result.retcode == 10017:
            print("   Error 10017: Trade disabled")
        elif result.retcode == 10018:
            print("   Error 10018: Market closed")
        elif result.retcode == 10019:
            print("   Error 10019: Not enough money")
        elif result.retcode == 10020:
            print("   Error 10020: Prices changed")
        elif result.retcode == 10021:
            print("   Error 10021: No quotes")
        elif result.retcode == 10022:
            print("   Error 10022: Invalid order expiration")
        elif result.retcode == 10023:
            print("   Error 10023: Order state changed")
        elif result.retcode == 10024:
            print("   Error 10024: Too frequent requests")
        elif result.retcode == 10025:
            print("   Error 10025: No changes in request")
        elif result.retcode == 10026:
            print("   Error 10026: Autotrading disabled by server")
        elif result.retcode == 10027:
            print("   Error 10027: Autotrading disabled by client")
        elif result.retcode == 10028:
            print("   Error 10028: Order locked for processing")
        elif result.retcode == 10029:
            print("   Error 10029: Long only")
        elif result.retcode == 10030:
            print("   Error 10030: Short only")

mt5.shutdown()
print("\nDone.")
