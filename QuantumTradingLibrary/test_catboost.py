"""Test the CatBoost quantum model"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostClassifier
import MetaTrader5 as mt5

MODEL_PATH = 'ETARE_QuantumFusion/models/catboost_quantum_3d.cbm'

def compression_ratio(prices):
    """Statistical approximation of quantum compression"""
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    volatility = np.std(returns)
    trend_strength = abs(np.mean(returns)) / (volatility + 1e-8)
    return 1.0 + trend_strength * 2

def calc_features(df):
    """Calculate technical features"""
    df = df.copy()

    # Price changes
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()

    # Bollinger
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))

    # Volatility
    df['volatility'] = df['log_returns'].rolling(20).std()

    # Volume
    df['vol_ma'] = df['tick_volume'].rolling(20).mean()
    df['vol_ratio'] = df['tick_volume'] / df['vol_ma']

    return df.dropna()

print('Loading CatBoost model...')
if not os.path.exists(MODEL_PATH):
    print(f'ERROR: Model not found at {MODEL_PATH}')
    exit(1)

model = CatBoostClassifier()
model.load_model(MODEL_PATH)
print(f'Model loaded. Classes: {model.classes_}')

# Get feature names
feat_names = model.feature_names_
print(f'Features ({len(feat_names)}): {feat_names[:10]}...')

# Load data
print('\nLoading BTCUSD data...')
mt5.initialize()
end = datetime.now()
start = end - timedelta(days=30)
rates = mt5.copy_rates_range('BTCUSD', mt5.TIMEFRAME_M5, start, end)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print('No data')
    exit(1)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = calc_features(df)
print(f'Data: {len(df)} bars')

# Backtest
wins_trending = 0
losses_trending = 0
wins_choppy = 0
losses_choppy = 0
skipped = 0

print('\nRunning backtest...')

for i in range(256, len(df) - 12, 12):
    row = df.iloc[i]
    prices = df['close'].iloc[i-256:i].values
    ratio = compression_ratio(prices)
    is_trending = ratio > 1.3

    # Build feature vector matching model expectations
    try:
        # Try to match feature names
        X = {}
        if 'rsi' in feat_names or 'RSI' in feat_names:
            X['rsi'] = row.get('rsi', 50)
            X['RSI'] = row.get('rsi', 50)
        if 'macd' in feat_names or 'MACD' in feat_names:
            X['macd'] = row.get('macd', 0)
            X['MACD'] = row.get('macd', 0)
        if 'atr' in feat_names or 'ATR' in feat_names:
            X['atr'] = row.get('atr', 0)
            X['ATR'] = row.get('atr', 0)
        if 'vol_ratio' in feat_names:
            X['vol_ratio'] = row.get('vol_ratio', 1)
        if 'bb_position' in feat_names or 'BB_position' in feat_names:
            X['bb_position'] = row.get('bb_position', 0.5)
            X['BB_position'] = row.get('bb_position', 0.5)
        if 'stoch_k' in feat_names or 'Stoch_K' in feat_names:
            X['stoch_k'] = row.get('stoch_k', 50)
            X['Stoch_K'] = row.get('stoch_k', 50)

        # Create DataFrame with correct columns
        X_df = pd.DataFrame([{name: X.get(name, 0) for name in feat_names}])

        proba = model.predict_proba(X_df)[0]
        pred = model.predict(X_df)[0]
        conf = max(proba)

    except Exception as e:
        skipped += 1
        continue

    if conf < 0.55:
        skipped += 1
        continue

    # Check outcome
    future_price = df['close'].iloc[i + 12]
    current_price = row['close']

    # pred is 0 or 1 (DOWN or UP typically)
    if pred == 1:  # UP
        correct = future_price > current_price
    else:  # DOWN
        correct = future_price < current_price

    if is_trending:
        if correct:
            wins_trending += 1
        else:
            losses_trending += 1
    else:
        if correct:
            wins_choppy += 1
        else:
            losses_choppy += 1

total_trending = wins_trending + losses_trending
total_choppy = wins_choppy + losses_choppy
total = total_trending + total_choppy

print()
print('='*60)
print('CATBOOST + COMPRESSION BACKTEST')
print('='*60)

if total_trending > 0:
    wr_trending = wins_trending / total_trending * 100
    print(f'TRENDING: {wins_trending}/{total_trending} = {wr_trending:.1f}%')
else:
    wr_trending = 0
    print('TRENDING: No trades')

if total_choppy > 0:
    wr_choppy = wins_choppy / total_choppy * 100
    print(f'CHOPPY:   {wins_choppy}/{total_choppy} = {wr_choppy:.1f}%')
else:
    wr_choppy = 0
    print('CHOPPY: No trades')

if total > 0:
    wr_total = (wins_trending + wins_choppy) / total * 100
    print(f'OVERALL:  {wins_trending + wins_choppy}/{total} = {wr_total:.1f}%')

print(f'Skipped: {skipped}')

if total_trending > 0 and total_choppy > 0:
    print(f'\nCOMPRESSION EDGE: +{wr_trending - wr_choppy:.1f}%')
