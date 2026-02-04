
# Blue Guardian Login Script
import MetaTrader5 as mt5
import sys

# Login details
LOGIN = 365060
PASSWORD = ")8xaE(gAuU"
SERVER = "BlueGuardian-Server"

if not mt5.initialize():
    print(f"FAILED to init MT5: {mt5.last_error()}")
    sys.exit(1)

print(f"Attempting login to {LOGIN} on {SERVER}...")
authorized = mt5.login(login=LOGIN, password=PASSWORD, server=SERVER)

if authorized:
    print("LOGIN SUCCESSFUL")
    info = mt5.account_info()
    print(f"Balance: {info.balance}")
    print(f"Equity: {info.equity}")
else:
    print(f"LOGIN FAILED: {mt5.last_error()}")

mt5.shutdown()
