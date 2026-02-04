import logging
from binance.client import Client
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# It's recommended to use API keys from a configuration file or environment variables
# For this implementation, we will assume they are set as environment variables
# e.g., BINANCE_API_KEY, BINANCE_API_SECRET
client = Client()

def fetch_binance_data(symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
    """
    Fetches historical OHLCV data from Binance.

    Args:
        symbol (str): The financial instrument to fetch (e.g., 'BTCUSDT').
        timeframe (str): The timeframe (e.g., '1m', '5m').
        start_dt (datetime): The start date of the data range.
        end_dt (datetime): The end date of the data range.

    Returns:
        pd.DataFrame | None: A DataFrame with the historical data, or None if it fails.
    """
    # Binance uses a different symbol format (e.g., BTCUSDT)
    binance_symbol = symbol.replace('USD', 'USDT')
    logging.info(f"Attempting to fetch Binance data for {binance_symbol} on {timeframe} from {start_dt} to {end_dt}...")

    # Convert datetimes to string format required by Binance API
    start_str = start_dt.strftime("%d %b, %Y %H:%M:%S")
    end_str = end_dt.strftime("%d %b, %Y %H:%M:%S")

    try:
        # Fetch the data
        klines = client.get_historical_klines(binance_symbol, timeframe, start_str, end_str)

        if not klines:
            logging.warning(f"No data returned from Binance for {binance_symbol} in the specified range.")
            return None

        # Convert to DataFrame
        columns = [
            'time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(klines, columns=columns)

        # --- Data Cleaning and Formatting ---
        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        
        # Select and rename columns to match the project's standard
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        df = df.rename(columns={'volume': 'tick_volume'})

        # Convert columns to numeric types
        for col in ['open', 'high', 'low', 'close', 'tick_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logging.info(f"Successfully fetched {len(df):,} bars of data from Binance for {binance_symbol}.")
        
        # Sanity check for sufficient data
        MINIMUM_BARS_THRESHOLD = 100
        if len(df) < MINIMUM_BARS_THRESHOLD:
            logging.warning(f"Insufficient data fetched from Binance ({len(df)} bars). Skipping.")
            return None

        return df

    except Exception as e:
        logging.error(f"An exception occurred during Binance data fetching: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example Usage
    TEST_SYMBOL = "BTCUSD"
    TEST_TIMEFRAME = "5m" # Use '1m', '5m', '15m' for Binance
    END_DATE = datetime.now()
    START_DATE = END_DATE - pd.Timedelta(days=1)
    
    binance_df = fetch_binance_data(TEST_SYMBOL, TEST_TIMEFRAME, START_DATE, END_DATE)
    
    if binance_df is not None:
        print("\n--- BINANCE FETCHER TEST SUCCESS ---")
        print(f"Shape of DataFrame: {binance_df.shape}")
        print("Columns:", binance_df.columns)
        print(binance_df.head())
        print("------------------------------------")
    else:
        print("\n--- BINANCE FETCHER TEST FAILED ---")