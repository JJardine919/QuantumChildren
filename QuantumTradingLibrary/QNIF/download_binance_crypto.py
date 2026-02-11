"""
Binance Crypto Historical Data Downloader
Downloads BTCUSDT and ETHUSDT data in the exact CSV format as XAU files.

Author: Biskits
Built for: Jim's Quantum Trading Library
"""

import os
import time
import json
import csv
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import requests

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required for resampling custom timeframes")
    print("Install with: pip install pandas")
    exit(1)


# Configuration
BASE_URL = "https://api.binance.com/api/v3/klines"
OUTPUT_DIR = r"C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\QNIF\HistoricalData\Full"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Standard Binance timeframes (API native)
STANDARD_TIMEFRAMES = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d"
}

# Custom timeframes to generate by resampling 1m data
CUSTOM_TIMEFRAMES = ["M2", "M4", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "M20"]

# Rate limiting
RATE_LIMIT_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# CSV Header matching XAU format exactly
CSV_HEADER = [
    "Open time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close time",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
    "Ignore"
]


def get_months_ago_timestamp(months: int) -> int:
    """Get Unix timestamp in milliseconds for N months ago."""
    now = datetime.now()
    target_date = now - timedelta(days=months * 30)  # Approximate
    return int(target_date.timestamp() * 1000)


def fetch_binance_klines(symbol: str, interval: str, start_time: int, limit: int = 1000) -> List[List]:
    """
    Fetch kline data from Binance API with retry logic.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Timeframe (e.g., 1m, 5m, 1h)
        start_time: Start timestamp in milliseconds
        limit: Number of candles to fetch (max 1000)

    Returns:
        List of kline data
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": limit
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"    Rate limited (429). Waiting {RETRY_DELAY * (attempt + 1)}s...")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"    Error {response.status_code}: {response.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"    Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return []


def download_standard_timeframe(symbol: str, interval: str, months: int = 6) -> List[List]:
    """
    Download all available data for a standard timeframe with pagination.

    Args:
        symbol: Trading pair
        interval: Binance interval string
        months: Minimum months of data to fetch

    Returns:
        Combined list of all klines
    """
    print(f"  Downloading {symbol} {interval}...")

    all_klines = []
    start_time = get_months_ago_timestamp(months)

    page = 1
    while True:
        klines = fetch_binance_klines(symbol, interval, start_time, limit=1000)

        if not klines:
            break

        all_klines.extend(klines)
        print(f"    Page {page}: Retrieved {len(klines)} candles (Total: {len(all_klines)})")

        # Get the last close time and add 1ms for next request
        last_close_time = klines[-1][6]
        start_time = last_close_time + 1

        # If we got less than 1000, we've reached the end
        if len(klines) < 1000:
            break

        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    print(f"  [OK] Total candles downloaded: {len(all_klines)}")
    return all_klines


def save_klines_to_csv(klines: List[List], filepath: str):
    """
    Save klines data to CSV in the exact XAU format.

    Binance kline format:
    [
      Open time, Open, High, Low, Close, Volume, Close time,
      Quote asset volume, Number of trades, Taker buy base volume,
      Taker buy quote volume, Ignore
    ]
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(klines)

    print(f"  [OK] Saved to: {filepath}")


def resample_1m_to_custom(klines_1m: List[List], minutes: int) -> List[List]:
    """
    Resample 1-minute klines to custom timeframe using pandas.

    Args:
        klines_1m: List of 1-minute klines from Binance
        minutes: Target timeframe in minutes

    Returns:
        Resampled klines in same format
    """
    if not klines_1m:
        return []

    # Convert to DataFrame
    df = pd.DataFrame(klines_1m, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'taker_buy_base', 'taker_buy_quote']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['trades'] = pd.to_numeric(df['trades'], errors='coerce').astype(int)
    df['ignore'] = pd.to_numeric(df['ignore'], errors='coerce').astype(int)

    # Set index to timestamp
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Resample with proper OHLCV aggregation
    resampled = df.resample(f'{minutes}min', label='left', closed='left').agg({
        'open_time': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'close_time': 'last',
        'quote_volume': 'sum',
        'trades': 'sum',
        'taker_buy_base': 'sum',
        'taker_buy_quote': 'sum',
        'ignore': 'first'
    })

    # Drop any NaN rows (incomplete periods)
    resampled = resampled.dropna()

    # Convert back to list format
    result = []
    for _, row in resampled.iterrows():
        result.append([
            int(row['open_time']),
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume']),
            int(row['close_time']),
            float(row['quote_volume']),
            int(row['trades']),
            float(row['taker_buy_base']),
            float(row['taker_buy_quote']),
            int(row['ignore'])
        ])

    return result


def generate_custom_timeframe(symbol: str, custom_tf: str, klines_1m: List[List]) -> bool:
    """
    Generate custom timeframe by resampling 1m data.

    Args:
        symbol: Trading pair
        custom_tf: Custom timeframe string (e.g., M2, M10)
        klines_1m: 1-minute klines data

    Returns:
        True if successful
    """
    # Extract minutes from custom timeframe (e.g., M2 -> 2)
    minutes = int(custom_tf[1:])

    print(f"  Generating {symbol} {custom_tf} (resampling {minutes}m from 1m data)...")

    resampled = resample_1m_to_custom(klines_1m, minutes)

    if not resampled:
        print(f"  [FAIL] Failed to resample {custom_tf}")
        return False

    filepath = os.path.join(OUTPUT_DIR, f"{symbol}_{custom_tf}.csv")
    save_klines_to_csv(resampled, filepath)
    print(f"  [OK] Generated {len(resampled)} candles for {custom_tf}")

    return True


def download_symbol_data(symbol: str, force: bool = False):
    """
    Download all timeframes for a single symbol.

    Args:
        symbol: Trading pair to download
        force: If True, overwrite existing files
    """
    print(f"\n{'='*60}")
    print(f"Processing {symbol}")
    print(f"{'='*60}")

    # Download standard timeframes
    klines_1m = None  # Cache for resampling

    for tf_name, binance_interval in STANDARD_TIMEFRAMES.items():
        filepath = os.path.join(OUTPUT_DIR, f"{symbol}_{tf_name}.csv")

        if os.path.exists(filepath) and not force:
            print(f"  [SKIP] {symbol} {tf_name} already exists (use --force to overwrite)")

            # Load 1m for resampling custom timeframes
            if tf_name == "1m":
                print(f"  Loading existing 1m data for resampling...")
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    klines_1m = [
                        [int(row[0]), float(row[1]), float(row[2]), float(row[3]),
                         float(row[4]), float(row[5]), int(row[6]), float(row[7]),
                         int(row[8]), float(row[9]), float(row[10]), int(row[11])]
                        for row in reader
                    ]
            continue

        klines = download_standard_timeframe(symbol, binance_interval, months=6)

        if klines:
            save_klines_to_csv(klines, filepath)

            # Cache 1m data for resampling
            if tf_name == "1m":
                klines_1m = klines
        else:
            print(f"  [FAIL] Failed to download {symbol} {tf_name}")

    # Generate custom timeframes from 1m data
    if klines_1m:
        print(f"\n  Generating custom timeframes from 1m data...")
        for custom_tf in CUSTOM_TIMEFRAMES:
            filepath = os.path.join(OUTPUT_DIR, f"{symbol}_{custom_tf}.csv")

            if os.path.exists(filepath) and not force:
                print(f"  [SKIP] {symbol} {custom_tf} already exists (use --force to overwrite)")
                continue

            generate_custom_timeframe(symbol, custom_tf, klines_1m)
    else:
        print(f"  [FAIL] No 1m data available for resampling custom timeframes")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download Binance crypto historical data in XAU CSV format"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        choices=SYMBOLS + ["all"],
        default="all",
        help="Specific symbol to download (default: all)"
    )

    args = parser.parse_args()

    print("="*60)
    print("BINANCE CRYPTO DATA DOWNLOADER")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Symbols: {SYMBOLS if args.symbol == 'all' else [args.symbol]}")
    print(f"Standard timeframes: {list(STANDARD_TIMEFRAMES.keys())}")
    print(f"Custom timeframes: {CUSTOM_TIMEFRAMES}")
    print(f"Target: At least 6 months of data per timeframe")
    print(f"Force overwrite: {args.force}")
    print("="*60)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()

    # Process symbols
    symbols_to_process = SYMBOLS if args.symbol == "all" else [args.symbol]

    for symbol in symbols_to_process:
        try:
            download_symbol_data(symbol, force=args.force)
        except Exception as e:
            print(f"\n[FAIL] ERROR processing {symbol}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETED in {elapsed:.1f} seconds")
    print(f"{'='*60}")
    print(f"\nAll data saved to: {OUTPUT_DIR}")
    print("\nYou can now use this data with your quantum trading system.")


if __name__ == "__main__":
    main()
