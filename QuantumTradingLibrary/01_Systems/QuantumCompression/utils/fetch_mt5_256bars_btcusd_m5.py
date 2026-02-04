import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

# Initialize MT5 connection
if not mt5.initialize():
    print("Failed to initialize MT5")
    mt5.shutdown()
    exit()

# Define function to fetch M5 bars
def fetch_m5_bars(symbol, start_date, end_date, num_bars=256):
    utc_from = datetime.strptime(start_date, '%Y-%m-%d')
    utc_from = pytz.utc.localize(utc_from)
    utc_to = datetime.strptime(end_date, '%Y-%m-%d')
    utc_to = pytz.utc.localize(utc_to)
    
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, utc_from, utc_to)
    
    if rates is None or len(rates) == 0:
        print(f"No data returned for {symbol} between {start_date} and {end_date}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['time', 'open', 'high', 'low', 'close']]
    
    # Take the most recent num_bars if more data is available
    if len(df) > num_bars:
        df = df.tail(num_bars)
    elif len(df) < num_bars:
        print(f"Insufficient bars ({len(df)}) for {symbol} between {start_date} and {end_date}; wanted {num_bars}")
        return None
    
    return df

# Main execution - adjust these dates based on your chart observations
symbol = "BTCUSD"

# Period 1: Attempted strong uptrend example (mid-January momentum)
df_uptrend = fetch_m5_bars(symbol, '2026-01-12', '2026-01-14')

# Period 2: Attempted choppy/consolidation example (early-mid January range)
df_choppy = fetch_m5_bars(symbol, '2026-01-07', '2026-01-09')

# Period 3: Attempted pullback/correction example (recent action)
df_pullback = fetch_m5_bars(symbol, '2026-01-15', '2026-01-17')

import os

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "04_Data", "MarketData")
os.makedirs(output_dir, exist_ok=True)

# Save and report
if df_uptrend is not None:
    print(f"Uptrend period: {len(df_uptrend)} bars fetched")
    df_uptrend.to_csv(os.path.join(output_dir, 'btc_uptrend_256bars.csv'), index=False)
else:
    print("Uptrend data fetch failed")

if df_choppy is not None:
    print(f"Choppy period: {len(df_choppy)} bars fetched")
    df_choppy.to_csv(os.path.join(output_dir, 'btc_choppy_256bars.csv'), index=False)
else:
    print("Choppy data fetch failed")

if df_pullback is not None:
    print(f"Pullback period: {len(df_pullback)} bars fetched")
    df_pullback.to_csv(os.path.join(output_dir, 'btc_pullback_256bars.csv'), index=False)
else:
    print("Pullback data fetch failed")

# Clean up
mt5.shutdown()