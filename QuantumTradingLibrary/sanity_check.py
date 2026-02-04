
import MetaTrader5 as mt5

def main():
    print("--- SYSTEM SANITY CHECK ---")
    if not mt5.initialize():
        print(f"MT5 INIT: FAILED ({mt5.last_error()})")
        return
    
    info = mt5.account_info()
    if info:
        print(f"ACCOUNT: {info.login}")
        print(f"SERVER:  {info.server}")
        print(f"BALANCE: ${info.balance}")
    else:
        print("ACCOUNT INFO: FAILED")

    symbol = "BTCUSD"
    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"PRICE:   {symbol} Bid={tick.bid}")
    else:
        print(f"PRICE:   {symbol} NO DATA")

    mt5.shutdown()
    print("---------------------------")

if __name__ == "__main__":
    main()
