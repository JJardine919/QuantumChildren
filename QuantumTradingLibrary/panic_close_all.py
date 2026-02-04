import MetaTrader5 as mt5
import time

def close_all_positions():
    # Force path to GetLeveraged terminal
    GL_PATH = r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe"
    
    if not mt5.initialize(path=GL_PATH):
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return

    # Check account
    acc = mt5.account_info()
    if not acc or acc.login != 113326:
        print(f"ERROR: Not logged into 113326 (Current: {acc.login if acc else 'None'})")
        mt5.shutdown()
        return

    positions = mt5.positions_get()
    if positions is None:
        print("No positions found or error occurred.")
        mt5.shutdown()
        return

    if len(positions) == 0:
        print("No open positions to close.")
        mt5.shutdown()
        return

    print(f"Found {len(positions)} positions. Closing all...")

    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        type_dict = {
            mt5.ORDER_TYPE_BUY: mt5.ORDER_TYPE_SELL,
            mt5.ORDER_TYPE_SELL: mt5.ORDER_TYPE_BUY
        }
        price_dict = {
            mt5.ORDER_TYPE_BUY: tick.bid,
            mt5.ORDER_TYPE_SELL: tick.ask
        }

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": type_dict[pos.type],
            "price": price_dict[pos.type],
            "deviation": 20,
            "magic": pos.magic,
            "comment": "PANIC CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close {pos.ticket}: {result.comment} ({result.retcode})")
        else:
            print(f"Closed Position {pos.ticket} at {result.price}")

    print("--- Finished ---")
    mt5.shutdown()

if __name__ == "__main__":
    close_all_positions()
