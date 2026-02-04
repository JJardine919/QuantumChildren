"""
Fetch 60 months of BTC data from Binance
M1, M5, M15 timeframes
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

def fetch_binance_klines(symbol, interval, start_time, end_time, limit=1000):
    """Fetch klines from Binance API"""
    url = "https://api.binance.com/api/v3/klines"

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(end_time.timestamp() * 1000),
        'limit': limit
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    return response.json()

def fetch_all_data(symbol, interval, months=60):
    """Fetch all historical data for given months"""

    end_time = datetime.now()
    start_time = end_time - timedelta(days=months * 30)

    print(f"Fetching {symbol} {interval} from {start_time.date()} to {end_time.date()}")

    all_data = []
    current_start = start_time

    # Interval to timedelta mapping
    interval_minutes = {
        '1m': 1,
        '5m': 5,
        '15m': 15
    }

    minutes = interval_minutes[interval]
    chunk_size = 1000  # Binance limit
    chunk_duration = timedelta(minutes=minutes * chunk_size)

    request_count = 0

    while current_start < end_time:
        chunk_end = min(current_start + chunk_duration, end_time)

        data = fetch_binance_klines(symbol, interval, current_start, chunk_end)

        if data:
            all_data.extend(data)
            request_count += 1

            if request_count % 10 == 0:
                print(f"  Fetched {len(all_data):,} bars... ({current_start.date()})")

        current_start = chunk_end

        # Rate limiting - Binance allows 1200 requests/minute
        time.sleep(0.05)

    print(f"  Total: {len(all_data):,} bars")
    return all_data

def convert_to_dataframe(data):
    """Convert Binance klines to DataFrame"""
    df = pd.DataFrame(data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['tick_volume'] = df['volume'].astype(float)

    # Keep only needed columns
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    return df

# Create data directory
os.makedirs('binance_data', exist_ok=True)

print("="*60)
print("FETCHING 60 MONTHS OF BTCUSDT DATA FROM BINANCE")
print("="*60)

# Fetch each timeframe
for interval in ['1m', '5m', '15m']:
    print(f"\n--- {interval.upper()} ---")

    data = fetch_all_data('BTCUSDT', interval, months=60)

    if data:
        df = convert_to_dataframe(data)
        filename = f'binance_data/BTCUSDT_{interval}_60months.csv'
        df.to_csv(filename, index=False)
        print(f"  Saved to {filename}")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

print("\n" + "="*60)
print("DATA FETCH COMPLETE")
print("="*60)
