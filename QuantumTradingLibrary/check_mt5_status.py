import MetaTrader5 as mt5
import sys

def check_mt5_connection():
    """
    Initializes and checks the connection to the MetaTrader 5 terminal.
    """
    print("--- Attempting to connect to MetaTrader 5 Terminal ---")
    
    # Attempt to initialize the connection
    if not mt5.initialize():
        print("\n[FATAL ERROR] mt5.initialize() failed.")
        print(f"Error code: {mt5.last_error()}")
        print("\n--- TROUBLESHOOTING ---")
        print("1. Is the MetaTrader 5 terminal running on this machine?")
        print("2. Are you logged into a trading account within the terminal?")
        print("3. Is the 'Algo Trading' button in the MT5 toolbar enabled (green)?")
        print("------------------------")
        mt5.shutdown()
        sys.exit(1) # Exit with an error code

    print("\n[SUCCESS] MetaTrader 5 terminal connected successfully!")
    
    # Print some account info to be sure
    account_info = mt5.account_info()
    if account_info:
        print(f"  - Account: {account_info.login}")
        print(f"  - Server: {account_info.server}")
        print(f"  - Balance: {account_info.balance} {account_info.currency}")
    else:
         print("[WARNING] Could not retrieve account information.")
         
    # Properly shut down the connection
    mt5.shutdown()
    print("\nConnection closed. MT5 status is OK.")

if __name__ == "__main__":
    check_mt5_connection()