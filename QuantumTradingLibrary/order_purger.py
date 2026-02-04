"""
QUANTUM ORDER PURGER
====================
Cleans up stale pending orders every 3 minutes to prevent terminal clogging.
"""

import MetaTrader5 as mt5
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [PURGER] - %(message)s")

def purge_stale_orders(max_age_seconds=180):
    orders = mt5.orders_get()
    if orders is None:
        return

    now = time.time()
    purged_count = 0

    for order in orders:
        # Check if it's a pending order (not a live position)
        if order.type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT, 
                          mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP]:
            
            age = now - order.time_setup
            if age > max_age_seconds:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"Purged stale order {order.ticket} ({order.symbol}) - Age: {int(age)}s")
                    purged_count += 1
                else:
                    logging.error(f"Failed to purge {order.ticket}: {result.comment}")

    if purged_count > 0:
        logging.info(f"Cleanup complete. Total purged: {purged_count}")

def main():
    if not mt5.initialize():
        print("MT5 Initialization failed")
        return

    print("="*60)
    print("QUANTUM STALE ORDER PURGER ACTIVE")
    print("Interval: 180 seconds")
    print("="*60)

    try:
        while True:
            purge_stale_orders()
            time.sleep(180) # 3 minutes
    except KeyboardInterrupt:
        print("Purger stopped.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
