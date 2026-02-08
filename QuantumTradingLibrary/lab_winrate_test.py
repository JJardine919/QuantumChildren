"""
LAB WIN RATE TEST
=================
Before/After: Does the Mushroom Rabies Lab actually improve win rate?

1. Load expert -> test win rate on real data
2. Run Mushroom Rabies Lab -> create mutant
3. Test mutant win rate on SAME data
4. Compare

"One number. That's all that matters."
"""
import sys
import io
import os
import torch
import torch.nn as nn
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Encoding handled by mushroom_rabies_lab import

SCRIPT_DIR = Path(__file__).parent.absolute()

# Import the lab
from mushroom_rabies_lab import MushroomRabiesLab, LSTMModel


# ============================================================
# DATA: Fetch from Binance (no MT5 dependency)
# ============================================================

def fetch_binance(symbol, interval='5m', months=3):
    """Fetch recent candle data from Binance."""
    print(f"  Fetching {symbol} {interval} ({months} months from Binance)...")
    url = "https://api.binance.com/api/v3/klines"
    end_time = datetime.now()
    start_time = end_time - timedelta(days=months * 30)

    all_data = []
    current = start_time
    interval_mins = {'1m': 1, '5m': 5, '15m': 15}[interval]
    chunk = timedelta(minutes=interval_mins * 1000)

    while current < end_time:
        chunk_end = min(current + chunk, end_time)
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': int(current.timestamp() * 1000),
            'endTime': int(chunk_end.timestamp() * 1000),
            'limit': 1000
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                all_data.extend(r.json())
        except Exception as e:
            print(f"    Binance error: {e}")
        current = chunk_end

    df = pd.DataFrame(all_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    print(f"  Got {len(df):,} bars")
    return df


def try_mt5_data(symbol, bars=30000):
    """Try to get data from MT5 if available."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
        mt5.shutdown()
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(f"  Got {len(df):,} bars from MT5")
        return df
    except:
        return None


# ============================================================
# FEATURES: 8 features matching LSTM input
# ============================================================

def prepare_features(df):
    """Prepare 8 normalized features for LSTM."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["macd"] = exp1 - exp2

    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["momentum"] = df["close"] / df["close"].shift(10)
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["price_change"] = df["close"].pct_change()

    df = df.ffill().bfill()

    cols = ['rsi', 'macd', 'bb_middle', 'bb_std', 'ema_10', 'momentum', 'atr', 'price_change']
    for c in cols:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

    return df.dropna(), cols


# ============================================================
# WIN RATE: Signal-based evaluation
# ============================================================

def evaluate_winrate(model, features_tensor, prices, hold_bars=5):
    """Calculate win rate: BUY=0, SELL=1, HOLD=2."""
    model.eval()
    with torch.no_grad():
        # Batch through to avoid memory issues
        batch_size = 1000
        all_actions = []
        for i in range(0, len(features_tensor), batch_size):
            batch = features_tensor[i:i + batch_size]
            result = model(batch)
            output = result[0] if isinstance(result, tuple) else result
            actions = torch.argmax(output, dim=1).cpu().numpy()
            all_actions.extend(actions)

    wins, losses = 0, 0
    position = None
    entry = 0
    entry_idx = 0

    for i in range(len(all_actions) - hold_bars):
        a = all_actions[i]
        curr = prices[i]

        if position is None:
            if a == 0:  # BUY
                position, entry, entry_idx = 'buy', curr, i
            elif a == 1:  # SELL
                position, entry, entry_idx = 'sell', curr, i
        else:
            held = (i - entry_idx) >= hold_bars
            opposite = (position == 'buy' and a == 1) or (position == 'sell' and a == 0)

            if held or opposite:
                exit_price = curr
                if position == 'buy':
                    if exit_price > entry:
                        wins += 1
                    else:
                        losses += 1
                else:
                    if exit_price < entry:
                        wins += 1
                    else:
                        losses += 1
                position = None

                if opposite:
                    if a == 0:
                        position, entry, entry_idx = 'buy', curr, i
                    elif a == 1:
                        position, entry, entry_idx = 'sell', curr, i

    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    return wr, wins, losses, total


# ============================================================
# MAIN: The one-number test
# ============================================================

def test_expert(expert_file, binance_symbol, rabies=1.5, dose=0.15):
    """Full before/after test for one expert."""
    expert_path = SCRIPT_DIR / 'top_50_experts' / expert_file

    print()
    print("=" * 60)
    print(f"  EXPERT: {expert_file}")
    print(f"  PARAMS: rabies={rabies}, dose={dose}")
    print("=" * 60)

    # --- Get test data ---
    print("\n[DATA]")
    # Try MT5 first for the actual symbol
    mt5_symbol = binance_symbol.replace('USDT', 'USD')
    df = try_mt5_data(mt5_symbol)
    if df is None:
        df = fetch_binance(binance_symbol, '5m', 3)

    df, cols = prepare_features(df)
    features_tensor = torch.FloatTensor(df[cols].values)
    prices = df['close'].values
    print(f"  Test set: {len(df):,} bars")

    # --- Load & test ORIGINAL expert ---
    print("\n[ORIGINAL]")
    original_model = LSTMModel(input_size=8, hidden_size=128, output_size=3, num_layers=2)
    state = torch.load(str(expert_path), map_location='cpu', weights_only=False)
    original_model.load_state_dict(state)
    original_model.eval()

    orig_wr, orig_w, orig_l, orig_total = evaluate_winrate(original_model, features_tensor, prices)
    print(f"  Win Rate: {orig_wr:.1f}%  ({orig_w}W / {orig_l}L = {orig_total} trades)")

    # --- Run Mushroom Rabies Lab ---
    print("\n[MUTATING]")
    lab = MushroomRabiesLab(
        rabies_aggression=rabies,
        mushroom_dose=dose,
        teqa_strength=0.2,
        compress_fidelity=0.85,
    )
    report = lab.run_pipeline(str(expert_path))

    # --- Load & test MUTANT expert ---
    print("\n[MUTANT]")
    mutant_path = report['mutant_path']
    mutant_model = LSTMModel(input_size=8, hidden_size=128, output_size=3, num_layers=2)
    mutant_state = torch.load(mutant_path, map_location='cpu', weights_only=False)
    mutant_model.load_state_dict({k: v.float() if v.is_floating_point() else v
                                   for k, v in mutant_state.items()})
    mutant_model.eval()

    mut_wr, mut_w, mut_l, mut_total = evaluate_winrate(mutant_model, features_tensor, prices)
    print(f"  Win Rate: {mut_wr:.1f}%  ({mut_w}W / {mut_l}L = {mut_total} trades)")

    # --- THE NUMBER ---
    delta = mut_wr - orig_wr
    arrow = "^" if delta > 0 else "v" if delta < 0 else "="

    print()
    print("=" * 60)
    print("  THE NUMBER THAT MATTERS")
    print("=" * 60)
    print(f"  BEFORE:  {orig_wr:.1f}%  ({orig_total} trades)")
    print(f"  AFTER:   {mut_wr:.1f}%  ({mut_total} trades)")
    print(f"  DELTA:   {delta:+.1f}%  {arrow}")
    print("=" * 60)

    return {
        'expert': expert_file,
        'rabies': rabies,
        'dose': dose,
        'original_wr': orig_wr,
        'original_trades': orig_total,
        'mutant_wr': mut_wr,
        'mutant_trades': mut_total,
        'delta': delta,
    }


if __name__ == '__main__':
    print()
    print("#" * 60)
    print("#  MUSHROOM RABIES LAB - WIN RATE BEFORE/AFTER TEST")
    print("#" * 60)

    results = []

    # Test 1: BTCUSD Special (heavy trip)
    r1 = test_expert('expert_BTCUSD_special.pth', 'BTCUSDT', rabies=2.0, dose=0.3)
    results.append(r1)

    # Test 2: XAUUSD Rank 11 (standard) - uses BTC data as proxy since no Binance gold
    r2 = test_expert('expert_rank11_XAUUSD.pth', 'BTCUSDT', rabies=1.5, dose=0.15)
    results.append(r2)

    # Test 3: ETHUSD Rank 21 (standard)
    r3 = test_expert('expert_rank21_ETHUSD.pth', 'ETHUSDT', rabies=1.5, dose=0.15)
    results.append(r3)

    # --- FINAL SUMMARY ---
    print()
    print()
    print("#" * 60)
    print("#  FINAL COMPARISON")
    print("#" * 60)
    print()
    print(f"  {'Expert':<30} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        name = r['expert'].replace('.pth', '')
        print(f"  {name:<30} {r['original_wr']:>7.1f}% {r['mutant_wr']:>7.1f}% {r['delta']:>+7.1f}%")
    print()
    print("#" * 60)
