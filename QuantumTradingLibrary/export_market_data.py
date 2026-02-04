
import MetaTrader5 as mt5
import json
import time
import os
from datetime import datetime

# Config
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 1000
OUTPUT_FILE = r"C:\Program Files\MetaTrader 5\MQL5\Files\market_data.json"

def main():
    if not mt5.initialize():
        print(f"MT5 Init Failed: {mt5.last_error()}")
        return

    print(f"Starting Data Exporter for {SYMBOL}...")
    
    while True:
        # Check connection
        if not mt5.terminal_info().connected:
            print("Terminal disconnected. Waiting...")
            time.sleep(5)
            continue

        # Get Data
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
        if rates is None:
            print(f"Failed to get rates for {SYMBOL}")
            time.sleep(5)
            continue

        # Convert to list of dicts
        data_list = []
        for rate in rates:
            data_list.append({
                "time": int(rate['time']),
                "open": float(rate['open']),
                "high": float(rate['high']),
                "low": float(rate['low']),
                "close": float(rate['close']),
                "tick_volume": int(rate['tick_volume'])
            })

        # Save to JSON
        export_data = {SYMBOL: data_list}
        try:
            # We write to a temp file then rename to ensure atomicity
            temp_file = OUTPUT_FILE + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(export_data, f)
            os.replace(temp_file, OUTPUT_FILE)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Exported {len(data_list)} bars for {SYMBOL}")
        except Exception as e:
            print(f"Error writing file: {e}")

        # Sleep 1 minute (M5 candles don't update that fast)
        time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        mt5.shutdown()
