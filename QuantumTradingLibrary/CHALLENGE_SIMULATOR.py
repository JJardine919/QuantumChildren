"""
CHALLENGE SIMULATOR - Monte Carlo Prop Firm Pass Rate
======================================================
Simulates 5000 prop firm challenge attempts using:
- ETARE expert (88.3% WR numpy feedforward)
- ATR_MULTIPLIER 0.0438 for SL distance
- $1 risk per trade, 3:1 TP
- Dynamic TP at 50% with expert consultation
- Rolling SL at 1.5x with TP trail

Uses historical M5 BTCUSD data from MT5.
Each trial = 3 weeks of simulated trading.
Reports pass/fail rate.

Run: .venv311\Scripts\python.exe CHALLENGE_SIMULATOR.py
"""

import sys
import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Load ETARE expert
from etare_expert import load_etare_expert, prepare_etare_features, ETAREExpert

# Config
from config_loader import (
    MAX_LOSS_DOLLARS,
    TP_MULTIPLIER,
    ROLLING_SL_MULTIPLIER,
    DYNAMIC_TP_PERCENT,
    ATR_MULTIPLIER,
    CONFIDENCE_THRESHOLD,
)

# ==============================================================
# CHALLENGE RULES (Blue Guardian style)
# ==============================================================
@dataclass
class ChallengeRules:
    starting_balance: float = 5000.0
    profit_target_pct: float = 0.08       # 8% to pass
    max_daily_loss_pct: float = 0.04      # 4% daily loss limit
    max_total_drawdown_pct: float = 0.08  # 8% total drawdown
    trading_days: int = 15                # 3 weeks = 15 trading days
    bars_per_day: int = 288               # M5 = 288 bars per 24h day


# ==============================================================
# SIMULATED POSITION
# ==============================================================
@dataclass
class SimPosition:
    direction: str  # "BUY" or "SELL"
    entry_price: float
    sl: float
    tp: float
    lot: float
    sl_distance: float  # original SL distance from entry
    bar_opened: int


# ==============================================================
# PRE-COMPUTE ALL DATA
# ==============================================================
def fetch_historical_data(symbol: str = "BTCUSD", bars: int = 60000) -> Optional[pd.DataFrame]:
    """Fetch M5 data from MT5. 60000 bars ~ 7 months."""
    print(f"Fetching {bars} M5 bars for {symbol}...")

    # Try to connect
    if not mt5.initialize():
        # Try BG terminal
        if not mt5.initialize(path=r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe"):
            print("ERROR: Cannot connect to MT5")
            return None

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
    mt5.shutdown()

    if rates is None or len(rates) < 10000:
        print(f"ERROR: Got only {len(rates) if rates is not None else 0} bars, need at least 10000")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Got {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    return df


def precompute_features(df: pd.DataFrame) -> np.ndarray:
    """Compute ETARE 20-feature vectors for every bar using rolling window."""
    print("Pre-computing features for all bars...")

    d = df.copy()
    for c in ["open", "high", "low", "close", "tick_volume"]:
        d[c] = d[c].astype(float)

    # RSI (14)
    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    d["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12-26-9)
    exp12 = d["close"].ewm(span=12, adjust=False).mean()
    exp26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = exp12 - exp26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    d["bb_middle"] = d["close"].rolling(20).mean()
    d["bb_std"] = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_middle"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_middle"] - 2 * d["bb_std"]

    # EMAs
    for p in [5, 10, 20, 50]:
        d[f"ema_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

    # Momentum
    d["momentum"] = d["close"] / d["close"].shift(10)

    # ATR (for features, not for SL - that uses separate ATR)
    d["atr"] = d["high"].rolling(14).max() - d["low"].rolling(14).min()

    # Price changes
    d["price_change"] = d["close"].pct_change()
    d["price_change_abs"] = d["price_change"].abs()

    # Volume
    d["volume_ma"] = d["tick_volume"].rolling(20).mean()
    d["volume_std"] = d["tick_volume"].rolling(20).std()

    # Stochastic K/D
    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = 100 * (d["close"] - low14) / (high14 - low14 + 1e-10)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # ROC
    d["roc"] = d["close"].pct_change(10) * 100

    feature_cols = [
        "rsi", "macd", "macd_signal",
        "bb_middle", "bb_upper", "bb_lower", "bb_std",
        "ema_5", "ema_10", "ema_20", "ema_50",
        "momentum", "atr",
        "price_change", "price_change_abs",
        "volume_ma", "volume_std",
        "stoch_k", "stoch_d",
        "roc",
    ]

    d = d.fillna(0)
    raw_features = d[feature_cols].values.astype(np.float64)

    # Z-score normalize using rolling window (200 bars lookback)
    window = 200
    normalized = np.zeros_like(raw_features)
    for i in range(window, len(raw_features)):
        chunk = raw_features[i-window:i]
        means = chunk.mean(axis=0)
        stds = chunk.std(axis=0) + 1e-8
        normalized[i] = np.clip((raw_features[i] - means) / stds, -4.0, 4.0)

    print(f"Features computed for {len(raw_features)} bars (valid from bar {window}+)")
    return normalized


def precompute_atr_sl(df: pd.DataFrame) -> np.ndarray:
    """Compute ATR-based SL distance for every bar."""
    tr = np.maximum(
        df['high'].values - df['low'].values,
        np.maximum(
            np.abs(df['high'].values - np.roll(df['close'].values, 1)),
            np.abs(df['low'].values - np.roll(df['close'].values, 1))
        )
    )
    tr[0] = df['high'].iloc[0] - df['low'].iloc[0]

    # Rolling ATR(14)
    atr = pd.Series(tr).rolling(14).mean().fillna(tr[0]).values
    sl_distances = atr * ATR_MULTIPLIER
    return sl_distances


def precompute_regime(df: pd.DataFrame) -> np.ndarray:
    """Simple regime detection: CLEAN if recent price has clear trend."""
    close = df['close'].values.astype(float)
    regime = np.zeros(len(close), dtype=bool)  # True = CLEAN

    for i in range(50, len(close)):
        # Use 50-bar lookback
        window = close[i-50:i]
        # Trend strength: correlation with linear regression
        x = np.arange(50)
        corr = np.corrcoef(x, window)[0, 1]
        # Clean if |correlation| > 0.3 (moderate trend)
        regime[i] = abs(corr) > 0.3

    return regime


def precompute_signals(expert: ETAREExpert, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run ETARE expert on all feature vectors. Returns directions and confidences."""
    print("Pre-computing ETARE signals for all bars...")
    n = len(features)
    # 0=HOLD, 1=BUY, 2=SELL
    directions = np.zeros(n, dtype=int)
    confidences = np.zeros(n)

    for i in range(200, n):
        state = features[i]
        if np.all(state == 0):
            continue
        probs = expert.forward(state)
        action_idx = int(np.argmax(probs))
        confidence = float(probs[action_idx])

        # Map ETARE 6-action to 3-action
        if action_idx == 0:  # OPEN_BUY
            directions[i] = 1
        elif action_idx == 1:  # OPEN_SELL
            directions[i] = 2
        else:
            directions[i] = 0

        confidences[i] = confidence

    buys = np.sum(directions == 1)
    sells = np.sum(directions == 2)
    holds = np.sum(directions == 0)
    print(f"Signals: {buys} BUY, {sells} SELL, {holds} HOLD")
    return directions, confidences


# ==============================================================
# SINGLE CHALLENGE SIMULATION
# ==============================================================
def simulate_challenge(
    start_bar: int,
    rules: ChallengeRules,
    ohlc: np.ndarray,         # [N, 4] = open, high, low, close
    sl_distances: np.ndarray,
    directions: np.ndarray,
    confidences: np.ndarray,
    regime_clean: np.ndarray,
    dollar_per_point: float,
    lot_min: float = 0.01,
    cooldown_bars: int = 0,
    max_trades_per_day: int = 0,
) -> dict:
    """
    Simulate one 3-week prop firm challenge.
    Returns dict with pass/fail, final balance, max drawdown, etc.
    """
    total_bars = rules.bars_per_day * rules.trading_days
    end_bar = start_bar + total_bars

    balance = rules.starting_balance
    peak_balance = balance
    max_drawdown = 0.0
    daily_start_balance = balance
    daily_bar_count = 0
    trades_taken = 0
    trades_won = 0
    trades_lost = 0
    daily_loss_breach = False
    drawdown_breach = False
    last_trade_bar = -999999
    daily_trades = 0

    position: Optional[SimPosition] = None

    for bar_idx in range(start_bar, end_bar):
        # Daily reset (every 288 bars = 24h for crypto)
        daily_bar_count += 1
        if daily_bar_count >= rules.bars_per_day:
            daily_bar_count = 0
            daily_start_balance = balance
            daily_trades = 0

        bar_open = ohlc[bar_idx, 0]
        bar_high = ohlc[bar_idx, 1]
        bar_low = ohlc[bar_idx, 2]
        bar_close = ohlc[bar_idx, 3]

        # --- CHECK OPEN POSITION ---
        if position is not None:
            pnl = 0.0
            closed = False

            if position.direction == "BUY":
                # Check SL hit (low touches SL)
                if bar_low <= position.sl:
                    pnl = (position.sl - position.entry_price) * position.lot * dollar_per_point
                    closed = True
                # Check TP hit (high touches TP)
                elif bar_high >= position.tp:
                    pnl = (position.tp - position.entry_price) * position.lot * dollar_per_point
                    closed = True
                else:
                    # Position still open - manage it
                    current_profit_dist = bar_close - position.entry_price
                    tp_target_dist = position.sl_distance * TP_MULTIPLIER
                    dyn_tp_threshold = tp_target_dist * (DYNAMIC_TP_PERCENT / 100.0)

                    # Dynamic TP checkpoint
                    if current_profit_dist >= dyn_tp_threshold:
                        # Consult expert: does it still say BUY?
                        if directions[bar_idx] == 1 and confidences[bar_idx] >= CONFIDENCE_THRESHOLD:
                            # Expert says hold - trail SL and TP
                            rolled_sl_dist = position.sl_distance / ROLLING_SL_MULTIPLIER
                            new_sl = bar_close - rolled_sl_dist
                            new_sl = max(new_sl, position.entry_price)  # At least breakeven
                            if new_sl > position.sl:
                                # Move TP forward maintaining 3:1
                                risk = bar_close - new_sl
                                new_tp = bar_close + (risk * TP_MULTIPLIER)
                                new_tp = max(new_tp, position.tp)
                                position.sl = new_sl
                                position.tp = new_tp
                        else:
                            # Expert says close - take dynamic TP
                            pnl = current_profit_dist * position.lot * dollar_per_point
                            closed = True

                    # Rolling SL (even if not at dyn TP threshold)
                    elif current_profit_dist > 0:
                        rolled_sl_dist = position.sl_distance / ROLLING_SL_MULTIPLIER
                        new_sl = bar_close - rolled_sl_dist
                        new_sl = max(new_sl, position.entry_price)
                        if new_sl > position.sl:
                            risk = bar_close - new_sl
                            new_tp = bar_close + (risk * TP_MULTIPLIER)
                            new_tp = max(new_tp, position.tp)
                            position.sl = new_sl
                            position.tp = new_tp

            else:  # SELL
                if bar_high >= position.sl:
                    pnl = (position.entry_price - position.sl) * position.lot * dollar_per_point
                    closed = True
                elif bar_low <= position.tp:
                    pnl = (position.entry_price - position.tp) * position.lot * dollar_per_point
                    closed = True
                else:
                    current_profit_dist = position.entry_price - bar_close
                    tp_target_dist = position.sl_distance * TP_MULTIPLIER
                    dyn_tp_threshold = tp_target_dist * (DYNAMIC_TP_PERCENT / 100.0)

                    if current_profit_dist >= dyn_tp_threshold:
                        if directions[bar_idx] == 2 and confidences[bar_idx] >= CONFIDENCE_THRESHOLD:
                            rolled_sl_dist = position.sl_distance / ROLLING_SL_MULTIPLIER
                            new_sl = bar_close + rolled_sl_dist
                            new_sl = min(new_sl, position.entry_price)
                            if new_sl < position.sl:
                                risk = new_sl - bar_close
                                new_tp = bar_close - (risk * TP_MULTIPLIER)
                                new_tp = min(new_tp, position.tp)
                                position.sl = new_sl
                                position.tp = new_tp
                        else:
                            pnl = current_profit_dist * position.lot * dollar_per_point
                            closed = True

                    elif current_profit_dist > 0:
                        rolled_sl_dist = position.sl_distance / ROLLING_SL_MULTIPLIER
                        new_sl = bar_close + rolled_sl_dist
                        new_sl = min(new_sl, position.entry_price)
                        if new_sl < position.sl:
                            risk = new_sl - bar_close
                            new_tp = bar_close - (risk * TP_MULTIPLIER)
                            new_tp = min(new_tp, position.tp)
                            position.sl = new_sl
                            position.tp = new_tp

            if closed:
                balance += pnl
                trades_taken += 1
                if pnl > 0:
                    trades_won += 1
                else:
                    trades_lost += 1
                position = None

        # --- CHECK RISK LIMITS ---
        if balance > peak_balance:
            peak_balance = balance

        current_drawdown = (peak_balance - balance) / rules.starting_balance
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown

        daily_loss = (daily_start_balance - balance) / rules.starting_balance
        if daily_loss >= rules.max_daily_loss_pct:
            daily_loss_breach = True
            break

        if current_drawdown >= rules.max_total_drawdown_pct:
            drawdown_breach = True
            break

        # --- OPEN NEW POSITION ---
        if position is None and regime_clean[bar_idx]:
            # Cooldown check
            if cooldown_bars > 0 and (bar_idx - last_trade_bar) < cooldown_bars:
                continue
            # Daily trade cap
            if max_trades_per_day > 0 and daily_trades >= max_trades_per_day:
                continue

            direction = directions[bar_idx]
            confidence = confidences[bar_idx]

            if direction in [1, 2] and confidence >= CONFIDENCE_THRESHOLD:
                sl_dist = sl_distances[bar_idx]
                if sl_dist <= 0:
                    continue

                # Calculate lot for $1 risk
                sl_ticks = sl_dist  # sl_dist is already in price units
                if dollar_per_point > 0 and sl_ticks > 0:
                    lot = MAX_LOSS_DOLLARS / (sl_ticks * dollar_per_point)
                    lot = max(lot_min, lot)
                    lot = min(lot, 0.5)
                    # Round to lot step matching lot_min precision
                    if lot_min < 0.01:
                        lot = round(lot * 1000) / 1000  # 0.001 precision
                    else:
                        lot = round(lot * 100) / 100  # 0.01 precision
                else:
                    continue

                tp_dist = sl_dist * TP_MULTIPLIER

                if direction == 1:  # BUY
                    entry = bar_close
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                    position = SimPosition("BUY", entry, sl, tp, lot, sl_dist, bar_idx)
                    last_trade_bar = bar_idx
                    daily_trades += 1
                else:  # SELL
                    entry = bar_close
                    sl = entry + sl_dist
                    tp = entry - tp_dist
                    position = SimPosition("SELL", entry, sl, tp, lot, sl_dist, bar_idx)
                    last_trade_bar = bar_idx
                    daily_trades += 1

    # --- CLOSE ANY REMAINING POSITION AT MARKET ---
    if position is not None:
        final_close = ohlc[min(end_bar - 1, len(ohlc) - 1), 3]
        if position.direction == "BUY":
            pnl = (final_close - position.entry_price) * position.lot * dollar_per_point
        else:
            pnl = (position.entry_price - final_close) * position.lot * dollar_per_point
        balance += pnl
        trades_taken += 1
        if pnl > 0:
            trades_won += 1
        else:
            trades_lost += 1

    profit_pct = (balance - rules.starting_balance) / rules.starting_balance
    passed = (
        profit_pct >= rules.profit_target_pct
        and not daily_loss_breach
        and not drawdown_breach
    )

    return {
        "passed": passed,
        "final_balance": balance,
        "profit_pct": profit_pct,
        "max_drawdown": max_drawdown,
        "daily_loss_breach": daily_loss_breach,
        "drawdown_breach": drawdown_breach,
        "trades_taken": trades_taken,
        "trades_won": trades_won,
        "trades_lost": trades_lost,
        "win_rate": trades_won / max(trades_taken, 1),
    }


# ==============================================================
# MAIN
# ==============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--atr', type=float, default=None, help='Override ATR_MULTIPLIER')
    parser.add_argument('--trials', type=int, default=5000, help='Number of trials')
    parser.add_argument('--confidence', type=float, default=None, help='Override CONFIDENCE_THRESHOLD')
    parser.add_argument('--tp-mult', type=float, default=None, help='Override TP_MULTIPLIER')
    parser.add_argument('--risk', type=float, default=None, help='Override MAX_LOSS_DOLLARS')
    parser.add_argument('--rolling-sl', type=float, default=None, help='Override ROLLING_SL_MULTIPLIER')
    parser.add_argument('--dyn-tp-pct', type=float, default=None, help='Override DYNAMIC_TP_PERCENT')
    parser.add_argument('--lot-min', type=float, default=0.01, help='Minimum lot size (default 0.01)')
    parser.add_argument('--cooldown', type=int, default=0, help='Min bars between trades (0=disabled)')
    parser.add_argument('--max-daily', type=int, default=0, help='Max trades per day (0=unlimited)')
    parser.add_argument('--balance', type=float, default=5000.0, help='Starting balance (default $5000)')
    args = parser.parse_args()

    # Apply overrides to module-level config references
    global ATR_MULTIPLIER, CONFIDENCE_THRESHOLD, TP_MULTIPLIER, MAX_LOSS_DOLLARS
    global ROLLING_SL_MULTIPLIER, DYNAMIC_TP_PERCENT
    if args.atr is not None:
        ATR_MULTIPLIER = args.atr
    if args.confidence is not None:
        CONFIDENCE_THRESHOLD = args.confidence
    if args.tp_mult is not None:
        TP_MULTIPLIER = args.tp_mult
    if args.risk is not None:
        MAX_LOSS_DOLLARS = args.risk
    if args.rolling_sl is not None:
        ROLLING_SL_MULTIPLIER = args.rolling_sl
    if args.dyn_tp_pct is not None:
        DYNAMIC_TP_PERCENT = args.dyn_tp_pct

    NUM_TRIALS = args.trials
    rules = ChallengeRules(starting_balance=args.balance)

    print("=" * 60)
    print("  CHALLENGE SIMULATOR - Monte Carlo")
    print(f"  {NUM_TRIALS} trials | 3 weeks each")
    print(f"  Starting balance: ${rules.starting_balance:,.0f}")
    print(f"  Profit target: {rules.profit_target_pct*100:.0f}%")
    print(f"  Max daily loss: {rules.max_daily_loss_pct*100:.0f}%")
    print(f"  Max drawdown: {rules.max_total_drawdown_pct*100:.0f}%")
    print(f"  ATR_MULTIPLIER: {ATR_MULTIPLIER}")
    print(f"  Risk per trade: ${MAX_LOSS_DOLLARS}")
    print(f"  TP: {TP_MULTIPLIER}x | Dyn TP: {DYNAMIC_TP_PERCENT}%")
    print(f"  Rolling SL: {ROLLING_SL_MULTIPLIER}x")
    print(f"  Min lot: {args.lot_min}")
    if args.cooldown > 0:
        print(f"  Cooldown: {args.cooldown} bars ({args.cooldown*5}min)")
    if args.max_daily > 0:
        print(f"  Max trades/day: {args.max_daily}")
    print("=" * 60)

    # Step 1: Fetch data
    df = fetch_historical_data("BTCUSD", bars=60000)
    if df is None:
        sys.exit(1)

    # Step 2: Load ETARE expert
    expert = load_etare_expert("BTCUSD")
    if expert is None:
        print("ERROR: No ETARE expert found")
        sys.exit(1)
    print(f"ETARE expert: fitness={expert.fitness:.4f}, WR={expert.win_rate*100:.1f}%")

    # Step 3: Pre-compute everything
    ohlc = df[['open', 'high', 'low', 'close']].values.astype(np.float64)
    features = precompute_features(df)
    sl_distances = precompute_atr_sl(df)
    regime_clean = precompute_regime(df)
    directions, confidences = precompute_signals(expert, features)

    # Dollar per point for BTCUSD at 0.01 lot
    # From real trade data: $3.00 profit on 8.98 point move at 0.01 lot
    # = $0.334 per point per 0.01 lot = $33.41 per point per 1.0 lot
    dollar_per_point = 33.41  # per 1.0 lot, per 1 point of price movement

    # Step 4: Determine valid starting range
    total_bars_per_trial = rules.bars_per_day * rules.trading_days
    min_start = 200  # Need lookback for features
    max_start = len(ohlc) - total_bars_per_trial - 1

    if max_start <= min_start:
        print(f"ERROR: Not enough data. Need {total_bars_per_trial} bars per trial, have {len(ohlc)}")
        sys.exit(1)

    print(f"\nValid start range: bar {min_start} to {max_start} ({max_start - min_start} options)")
    print(f"Bars per trial: {total_bars_per_trial}")

    # Step 5: Run simulations
    print(f"\nRunning {NUM_TRIALS} simulations...")
    np.random.seed(42)
    start_bars = np.random.randint(min_start, max_start, size=NUM_TRIALS)

    results = []
    t0 = time.time()

    for i, start in enumerate(start_bars):
        result = simulate_challenge(
            start_bar=start,
            rules=rules,
            ohlc=ohlc,
            sl_distances=sl_distances,
            directions=directions,
            confidences=confidences,
            regime_clean=regime_clean,
            dollar_per_point=dollar_per_point,
            lot_min=args.lot_min,
            cooldown_bars=args.cooldown,
            max_trades_per_day=args.max_daily,
        )
        results.append(result)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            passed_so_far = sum(1 for r in results if r['passed'])
            rate = passed_so_far / len(results) * 100
            print(f"  [{i+1}/{NUM_TRIALS}] Pass rate: {rate:.1f}% | {elapsed:.1f}s")

    elapsed = time.time() - t0

    # Step 6: Analyze results
    passed = sum(1 for r in results if r['passed'])
    failed_dd = sum(1 for r in results if r['drawdown_breach'])
    failed_daily = sum(1 for r in results if r['daily_loss_breach'])
    failed_target = sum(1 for r in results if not r['passed'] and not r['drawdown_breach'] and not r['daily_loss_breach'])

    avg_profit = np.mean([r['profit_pct'] for r in results]) * 100
    avg_trades = np.mean([r['trades_taken'] for r in results])
    avg_wr = np.mean([r['win_rate'] for r in results]) * 100
    avg_dd = np.mean([r['max_drawdown'] for r in results]) * 100
    median_profit = np.median([r['profit_pct'] for r in results]) * 100

    best = max(results, key=lambda r: r['profit_pct'])
    worst = min(results, key=lambda r: r['profit_pct'])

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Total trials:        {NUM_TRIALS}")
    print(f"  PASSED:              {passed} ({passed/NUM_TRIALS*100:.1f}%)")
    print(f"  FAILED:              {NUM_TRIALS - passed} ({(NUM_TRIALS-passed)/NUM_TRIALS*100:.1f}%)")
    print(f"    - Drawdown breach: {failed_dd}")
    print(f"    - Daily loss:      {failed_daily}")
    print(f"    - Missed target:   {failed_target}")
    print()
    print(f"  Avg profit:          {avg_profit:.2f}%")
    print(f"  Median profit:       {median_profit:.2f}%")
    print(f"  Avg trades/trial:    {avg_trades:.0f}")
    print(f"  Avg win rate:        {avg_wr:.1f}%")
    print(f"  Avg max drawdown:    {avg_dd:.2f}%")
    print()
    print(f"  Best trial:          {best['profit_pct']*100:.2f}% (${best['final_balance']:.2f})")
    print(f"  Worst trial:         {worst['profit_pct']*100:.2f}% (${worst['final_balance']:.2f})")
    print()
    print(f"  Time: {elapsed:.1f}s ({elapsed/NUM_TRIALS*1000:.1f}ms per trial)")
    print("=" * 60)

    # Distribution
    profits = [r['profit_pct'] * 100 for r in results]
    print("\n  PROFIT DISTRIBUTION:")
    brackets = [(-100, -8), (-8, -4), (-4, 0), (0, 4), (4, 8), (8, 20), (20, 100)]
    for lo, hi in brackets:
        count = sum(1 for p in profits if lo <= p < hi)
        bar = "#" * (count // 20)
        label = f"  {lo:>4}% to {hi:>4}%"
        print(f"  {label}: {count:>5} {bar}")

    # Print actual risk per trade info
    avg_sl_dist = np.mean(sl_distances[200:])
    actual_lot = max(args.lot_min, MAX_LOSS_DOLLARS / (avg_sl_dist * dollar_per_point))
    actual_risk = avg_sl_dist * dollar_per_point * actual_lot
    print(f"\n  RISK ANALYSIS:")
    print(f"    Avg SL distance:    {avg_sl_dist:.2f} points")
    print(f"    Calculated lot:     {MAX_LOSS_DOLLARS / (avg_sl_dist * dollar_per_point):.6f}")
    print(f"    Actual lot (after min): {actual_lot:.4f}")
    print(f"    ACTUAL risk/trade:  ${actual_risk:.2f}")
    print(f"    Target risk/trade:  ${MAX_LOSS_DOLLARS:.2f}")
    print(f"    Risk inflation:     {actual_risk/MAX_LOSS_DOLLARS:.1f}x")


def sweep_atr():
    """Sweep ATR multiplier values to find optimal."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lot-min', type=float, default=0.01)
    parser.add_argument('--trials', type=int, default=5000)
    args = parser.parse_args()

    rules = ChallengeRules()

    print("Fetching data...")
    df = fetch_historical_data("BTCUSD", bars=60000)
    if df is None:
        sys.exit(1)

    expert = load_etare_expert("BTCUSD")
    if expert is None:
        sys.exit(1)

    ohlc = df[['open', 'high', 'low', 'close']].values.astype(np.float64)
    features = precompute_features(df)
    regime_clean = precompute_regime(df)
    directions, confidences = precompute_signals(expert, features)
    dollar_per_point = 33.41

    total_bars_per_trial = rules.bars_per_day * rules.trading_days
    min_start = 200
    max_start = len(ohlc) - total_bars_per_trial - 1
    np.random.seed(42)
    start_bars = np.random.randint(min_start, max_start, size=args.trials)

    # Sweep ATR values
    atr_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]

    print(f"\n{'='*80}")
    print(f"  ATR MULTIPLIER SWEEP | {args.trials} trials | lot_min={args.lot_min}")
    print(f"{'='*80}")
    print(f"  {'ATR':>6} | {'Pass%':>6} | {'AvgProfit':>9} | {'WR':>5} | {'AvgDD':>6} | {'Trades':>6} | {'$/Trade':>8} | {'DDFail':>6} | {'DLFail':>6} | {'MissTgt':>7}")
    print(f"  {'-'*6} | {'-'*6} | {'-'*9} | {'-'*5} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*7}")

    for atr_mult in atr_values:
        # Recompute SL distances for this ATR
        tr = np.maximum(
            df['high'].values - df['low'].values,
            np.maximum(
                np.abs(df['high'].values - np.roll(df['close'].values, 1)),
                np.abs(df['low'].values - np.roll(df['close'].values, 1))
            )
        )
        tr[0] = df['high'].iloc[0] - df['low'].iloc[0]
        atr = pd.Series(tr).rolling(14).mean().fillna(tr[0]).values
        sl_dists = atr * atr_mult

        results = []
        for start in start_bars:
            result = simulate_challenge(
                start_bar=start, rules=rules, ohlc=ohlc,
                sl_distances=sl_dists, directions=directions,
                confidences=confidences, regime_clean=regime_clean,
                dollar_per_point=dollar_per_point, lot_min=args.lot_min,
            )
            results.append(result)

        passed = sum(1 for r in results if r['passed'])
        failed_dd = sum(1 for r in results if r['drawdown_breach'])
        failed_dl = sum(1 for r in results if r['daily_loss_breach'])
        failed_tgt = sum(1 for r in results if not r['passed'] and not r['drawdown_breach'] and not r['daily_loss_breach'])
        avg_profit = np.mean([r['profit_pct'] for r in results]) * 100
        avg_wr = np.mean([r['win_rate'] for r in results]) * 100
        avg_dd = np.mean([r['max_drawdown'] for r in results]) * 100
        avg_trades = np.mean([r['trades_taken'] for r in results])

        avg_sl = np.mean(sl_dists[200:])
        raw_lot = MAX_LOSS_DOLLARS / (avg_sl * dollar_per_point)
        act_lot = max(args.lot_min, raw_lot)
        risk_per = avg_sl * dollar_per_point * act_lot

        pass_pct = passed / args.trials * 100
        print(f"  {atr_mult:>6.3f} | {pass_pct:>5.1f}% | {avg_profit:>+8.2f}% | {avg_wr:>4.1f}% | {avg_dd:>5.2f}% | {avg_trades:>6.0f} | ${risk_per:>6.2f} | {failed_dd:>6} | {failed_dl:>6} | {failed_tgt:>7}")

    print(f"{'='*80}")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sys.argv.remove("--sweep")
        sweep_atr()
    else:
        main()
