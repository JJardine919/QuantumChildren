
import MetaTrader5 as mt5
from credential_manager import get_credentials

def test_connection():
    account_key = 'QNIF_GL_3'
    print(f"Testing connection to {account_key}...")
    try:
        creds = get_credentials(account_key)
        
        # Initialize (GetLeveraged often uses the generic MT5 terminal)
        if not mt5.initialize():
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
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_connection()
