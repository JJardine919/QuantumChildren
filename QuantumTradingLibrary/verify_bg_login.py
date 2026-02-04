import MetaTrader5 as mt5
import sys

# Credentials
LOGIN = 365060
PASSWORD = ")8xaE(gAuU"
SERVER = "BlueGuardian-Server"

if not mt5.initialize():
    print(f"MT5 Init Failed: {mt5.last_error()}")
    sys.exit(1)

print(f"Attempting login to {LOGIN}...")
if mt5.login(LOGIN, password=PASSWORD, server=SERVER):
    print("SUCCESS: Logged into Blue Guardian")
    acc = mt5.account_info()
    print(f"Balance: {acc.balance}")
else:
    print(f"FAILED: {mt5.last_error()}")

mt5.shutdown()
