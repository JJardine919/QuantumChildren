import MetaTrader5 as mt5

def fix_stops():
    # Force path to GetLeveraged terminal
    GL_PATH = r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe"
    
    if not mt5.initialize(path=GL_PATH):
        print(f"Failed to initialize: {mt5.last_error()}")
        return

    # Targeting the specific BTCUSD trades
    positions = mt5.positions_get(symbol="BTCUSD")
    if positions:
        for p in positions:
            # We target the specific account 113326 just in case
            acc = mt5.account_info()
            if acc.login != 113326:
                print(f"Currently on account {acc.login}. Please switch to 113326 first.")
                break

            # Calculate the new tight SL ($50.00 away from entry = 50 cents risk @ 0.01)
            if p.type == mt5.ORDER_TYPE_BUY:
                new_sl = p.price_open - 50.0
            else:
                new_sl = p.price_open + 50.0
            
            # Print current state
            print(f"Position {p.ticket} | Open: {p.price_open} | Current SL: {p.sl} | Target SL: {new_sl}")

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "sl": new_sl,
                "tp": p.tp # Keep existing TP
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"==> SUCCESS: Fixed Position {p.ticket}. New SL: {new_sl}")
            else:
                print(f"==> FAILED: {p.ticket}: {result.comment} ({result.retcode})")
    else:
        print("No open BTCUSD positions found to fix.")
    
    mt5.shutdown()

if __name__ == "__main__":
    fix_stops()
