
import MetaTrader5 as mt5
from credential_manager import get_credentials

def test_connection():
    account_key = 'QNIF_FTMO'
    print(f"Testing connection to {account_key}...")
    creds = get_credentials(account_key)
    
    if not mt5.initialize(path=creds.get('terminal_path')):
        print(f"MT5 init failed: {mt5.last_error()}")
        return
    
    login_ok = mt5.login(creds['account'], password=creds['password'], server=creds['server'])
    if not login_ok:
        print(f"MT5 login failed: {mt5.last_error()}")
    else:
        print("Successfully connected!")
        info = mt5.account_info()
        print(f"Account: {info.login}")
        print(f"Balance: {info.balance}")
        print(f"Server: {info.server}")
    
    mt5.shutdown()

if __name__ == "__main__":
    test_connection()
