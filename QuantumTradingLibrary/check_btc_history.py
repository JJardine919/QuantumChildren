
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz

def check_data_availability():
    if not mt5.initialize():
        print("MT5 Init Failed")
        return

    symbol = "BTCUSD"
    # Check if symbol exists, if not try variations
    if not mt5.symbol_info(symbol):
        for s in ["BTCUSD.ecn", "Bitcoin", "BTCUSDT"]:
            if mt5.symbol_info(s):
                symbol = s
                break
    
    print(f"Checking data for {symbol}...")
    
    # Target: 60 months ~ 5 years
    now = datetime.now().replace(tzinfo=pytz.utc)
    start_time = now - timedelta(days=60*30) 
    
    # Try to fetch M5 bars
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_time, now)
    
    if rates is None:
        print("❌ No data returned. History might not be downloaded.")
        print(f"Error: {mt5.last_error()}")
    else:
        print(f"✅ Downloaded {len(rates)} bars")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(f"Start: {df['time'].iloc[0]}")
        print(f"End:   {df['time'].iloc[-1]}")
        print(f"Total Time: {df['time'].iloc[-1] - df['time'].iloc[0]}")
        
    mt5.shutdown()

if __name__ == "__main__":
    check_data_availability()
