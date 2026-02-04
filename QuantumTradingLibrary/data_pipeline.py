import logging
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Assuming etare_module.py is in the same directory or in PYTHONPATH
from etare_module import HybridTrader 

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}

def get_timeframe_object(timeframe_str):
    """Converts a timeframe string (e.g., 'M15') to an MT5 timeframe object."""
    return TIMEFRAME_MAP.get(timeframe_str.upper())

def fetch_data(symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
    """
    Fetches historical OHLCV data from MetaTrader 5.

    Args:
        symbol (str): The financial instrument to fetch.
        timeframe (str): The timeframe (e.g., 'M1', 'H4').
        start_dt (datetime): The start date of the data range.
        end_dt (datetime): The end date of the data range.

    Returns:
        pd.DataFrame | None: A DataFrame with the historical data, or None if it fails.
    """
    logging.info(f"Attempting to fetch data for {symbol} on {timeframe} from {start_dt} to {end_dt}...")
    
    # Ensure MT5 is initialized
    if not mt5.initialize():
        logging.error("MetaTrader 5 initialization failed.")
        mt5.shutdown()
        return None
    
    mt5_timeframe = get_timeframe_object(timeframe)
    if mt5_timeframe is None:
        logging.error(f"Invalid timeframe provided: {timeframe}")
        mt5.shutdown()
        return None

    # MT5 requires timezone-aware datetime objects
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
        
    try:
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_dt, end_dt)
        
        if rates is None or len(rates) == 0:
            logging.warning(f"No data returned for {symbol} in the specified range.")
            mt5.shutdown()
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        # Convert epoch time to readable datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Add a sanity check for the number of bars
        MINIMUM_BARS_THRESHOLD = 100 # A reasonable minimum for a training period
        if len(df) < MINIMUM_BARS_THRESHOLD:
            logging.warning(
                f"Insufficient data fetched for {symbol} on {timeframe}. "
                f"Expected a large dataset but got only {len(df)} bars. "
                "This might be due to a lack of historical data on the broker's server for this date range. "
                "Skipping this data segment."
            )
            mt5.shutdown()
            return None
            
        logging.info(f"Successfully fetched {len(df):,} bars of data for {symbol}.")
        
        # Ensure standard column names that prepare_features expects
        df = df.rename(columns={
            'time': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'tick_volume',
        })
        
        return df

    except Exception as e:
        logging.error(f"An exception occurred during data fetching: {e}")
        return None
    finally:
        mt5.shutdown()



def create_labels(df: pd.DataFrame, look_forward_period: int = 10, threshold: float = 0.001) -> np.ndarray:
    """
    Creates labels for supervised learning based on future price movement.
    Action.HOLD = 0, Action.OPEN_BUY = 1, Action.OPEN_SELL = 2

    Args:
        df (pd.DataFrame): DataFrame with at least a 'close' column.
        look_forward_period (int): How many bars to look into the future.
        threshold (float): The percentage change required to trigger a buy/sell signal.

    Returns:
        np.ndarray: An array of labels (0, 1, or 2).
    """
    logging.info(f"Generating labels with look-forward={look_forward_period} and threshold={threshold}...")
    
    future_returns = df['close'].pct_change(periods=look_forward_period).shift(-look_forward_period)
    
    labels = np.zeros(len(df), dtype=int)  # Default to HOLD (0)
    
    # Where future return is > threshold, label is BUY (1)
    labels[future_returns > threshold] = 1
    
    # Where future return is < -threshold, label is SELL (2)
    labels[future_returns < -threshold] = 2
    
    return labels

def process_data_for_model(raw_data: pd.DataFrame, label_generation_params: dict = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Processes raw OHLCV data into a feature matrix and labels for the model.
    This version uses a more robust method to align features and labels.
    """
    if raw_data is None or raw_data.empty:
        logging.warning("Cannot process data: input DataFrame is empty or None.")
        return None, None
        
    logging.info(f"Processing {len(raw_data):,} bars of raw data into features and labels...")
    
    try:
        # 1. Calculate features first
        dummy_trader = HybridTrader(symbols=[])
        feature_df = dummy_trader.prepare_features(raw_data)
        
        # 2. Drop NaN rows created by feature engineering (e.g., rolling means)
        # This primarily trims the beginning of the DataFrame.
        feature_df.dropna(inplace=True)
        feature_df.reset_index(drop=True, inplace=True)
        
        if feature_df.empty:
            logging.error("DataFrame is empty after feature calculation and NaN drop. Check feature engineering logic.")
            return None, None
            
        # 3. Generate labels based on the aligned feature DataFrame
        label_params = label_generation_params if label_generation_params else {}
        labels = create_labels(feature_df, **label_params)

        # 4. Align features and labels by removing trailing NaNs from labels
        # The create_labels function introduces NaNs at the end of the series.
        valid_label_indices = ~np.isnan(labels)
        labels = labels[valid_label_indices]
        feature_df = feature_df[valid_label_indices]

        # 5. Extract numpy arrays for the model
        feature_matrix = feature_df.drop(columns=['time'], errors='ignore').values
        
        if feature_matrix.shape[0] != labels.shape[0]:
             raise RuntimeError(f"Shape mismatch! Features: {feature_matrix.shape}, Labels: {labels.shape}")

        logging.info(f"Data processing complete. Final feature matrix shape: {feature_matrix.shape}, Labels shape: {labels.shape}")
        
        return feature_matrix, labels
        
    except Exception as e:
        logging.error(f"An exception occurred during data processing: {e}", exc_info=True)
        return None, None


if __name__ == '__main__':
    # Example Usage for testing the pipeline
    
    # --- Parameters ---
    TEST_SYMBOL = "BTCUSD"
    TEST_TIMEFRAME = "M5"
    END_DATE = datetime.now()
    START_DATE = END_DATE - pd.Timedelta(days=30) # Fetch last 30 days
    
    # 1. Fetch raw data
    raw_ohlcv_data = fetch_data(TEST_SYMBOL, TEST_TIMEFRAME, START_DATE, END_DATE)
    
    # 2. Process data for the model
    if raw_ohlcv_data is not None:
        feature_vectors, target_labels = process_data_for_model(raw_ohlcv_data)
        
        if feature_vectors is not None and target_labels is not None:
            print("\n--- PIPELINE TEST SUCCESS ---")
            print(f"Shape of final feature matrix: {feature_vectors.shape}")
            print(f"Shape of final labels vector: {target_labels.shape}")
            print("This matrix is ready to be used for training or backtesting.")
            print("-----------------------------")
        else:
            print("\n--- PIPELINE TEST FAILED (Processing) ---")
    else:
        print("\n--- PIPELINE TEST FAILED (Fetching) ---")
