
# Blue Guardian Login Script
import MetaTrader5 as mt5
import sys
from credential_manager import get_credentials, CredentialError

# Login details - load from credential_manager
try:
    _creds = get_credentials('BG_CHALLENGE')
    LOGIN = _creds['account']
    PASSWORD = _creds['password']
    SERVER = _creds['server']
except CredentialError as e:
    print(f"ERROR: Could not load BG_CHALLENGE credentials: {e}")
    print("Set BG_CHALLENGE_PASSWORD in .env file or environment")
    sys.exit(1)

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
