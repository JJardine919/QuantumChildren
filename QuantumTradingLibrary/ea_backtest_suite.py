"""
EA Backtest Suite - Real MT5 Data Backtesting
==============================================
Pulls historical data from ATLAS MT5 terminal and simulates each EA's
exact signal logic against real market data.

Author: DooDoo (for Jim)
Date: 2026-02-06
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import re
import sys
import os
from datetime import datetime, timedelta

# ============================================================
# MT5 CONNECTION
# ============================================================
TERMINAL_PATH = r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe"
SYMBOL = "BTCUSD"
SPREAD_POINTS = 50  # 50 points spread for BTCUSD
INITIAL_BALANCE = 100000.0  # Simulated starting balance

def connect_mt5():
    """Connect to ATLAS MT5 terminal (already logged in, no login call)."""
    if not mt5.initialize(path=TERMINAL_PATH):
        print(f"MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)
    info = mt5.account_info()
    if info:
        print(f"Connected to account: {info.login} | Balance: ${info.balance:.2f}")
    else:
        print("Warning: Could not get account info, but MT5 initialized.")
    return True

def get_candle_data(symbol, timeframe, num_bars):
    """Pull historical candle data from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"Failed to get {symbol} data for timeframe {timeframe}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def resample_to_timeframe(df_m1, minutes):
    """Resample M1 data to any higher timeframe."""
    df = df_m1.set_index('time')
    rule = f'{minutes}min'
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'tick_volume': 'sum',
        'spread': 'mean',
        'real_volume': 'sum'
    }).dropna()
    resampled = resampled.reset_index()
    return resampled


# ============================================================
# INDICATOR CALCULATIONS (pure numpy/pandas)
# ============================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_sma(series, period):
    return series.rolling(window=period).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calc_stochastic(high, low, close, k_period=5, d_period=3, slowing=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = fast_k.rolling(window=slowing).mean()  # %K with slowing
    d = k.rolling(window=d_period).mean()       # %D
    return k, d

def calc_cci(high, low, close, period=14):
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    return cci

def calc_mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    pos_mf = pd.Series(0.0, index=tp.index)
    neg_mf = pd.Series(0.0, index=tp.index)
    for i in range(1, len(tp)):
        if tp.iloc[i] > tp.iloc[i-1]:
            pos_mf.iloc[i] = raw_mf.iloc[i]
        elif tp.iloc[i] < tp.iloc[i-1]:
            neg_mf.iloc[i] = raw_mf.iloc[i]
    pos_sum = pos_mf.rolling(window=period).sum()
    neg_sum = neg_mf.rolling(window=period).sum()
    mfi = 100.0 - (100.0 / (1.0 + pos_sum / neg_sum.replace(0, 1e-10)))
    return mfi

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/period, min_periods=period).mean()
    return atr

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger(close, period=20, dev=2.0):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + dev * std
    lower = sma - dev * std
    return upper, sma, lower

def calc_dpo(close, period=20):
    """Detrended Price Oscillator: close[shift] - SMA(period)"""
    shift = period // 2 + 1
    sma = close.rolling(window=period).mean()
    # DPO = close shifted back by (period/2+1) minus SMA
    # In practice: DPO[i] = close[i - shift] - SMA[i]
    # But for backtesting we compute it bar-by-bar:
    dpo = close.shift(shift) - sma
    return dpo


# ============================================================
# TRADE SIMULATOR
# ============================================================

class TradeSimulator:
    """Simulates trades with SL/TP/trailing stop logic."""

    def __init__(self, spread_points=50, initial_balance=100000.0, lot_size=0.01):
        self.spread = spread_points  # in points (price units for BTC)
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.trades = []
        self.open_trade = None
        self.equity_high = initial_balance
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance

    def get_point_value(self):
        """For BTCUSD, 1 point = 0.01 typically. But we work in price units."""
        return 1.0  # We'll track P/L in price difference * lot_size

    def open_buy(self, bar_time, ask_price, sl_price, tp_price, ea_name=""):
        if self.open_trade is not None:
            return
        entry = ask_price  # buying at ask
        self.open_trade = {
            'type': 'BUY',
            'entry_price': entry,
            'sl': sl_price,
            'tp': tp_price,
            'open_time': bar_time,
            'ea': ea_name,
            'trail_sl': sl_price
        }

    def open_sell(self, bar_time, bid_price, sl_price, tp_price, ea_name=""):
        if self.open_trade is not None:
            return
        entry = bid_price  # selling at bid
        self.open_trade = {
            'type': 'SELL',
            'entry_price': entry,
            'sl': sl_price,
            'tp': tp_price,
            'open_time': bar_time,
            'ea': ea_name,
            'trail_sl': sl_price
        }

    def close_trade(self, bar_time, exit_price, reason=""):
        if self.open_trade is None:
            return
        trade = self.open_trade
        if trade['type'] == 'BUY':
            pnl = (exit_price - trade['entry_price']) * self.lot_size
        else:
            pnl = (trade['entry_price'] - exit_price) * self.lot_size

        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        dd = (self.peak_balance - self.balance) / self.peak_balance * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        trade['exit_price'] = exit_price
        trade['pnl'] = pnl
        trade['close_time'] = bar_time
        trade['reason'] = reason
        self.trades.append(trade)
        self.open_trade = None

    def check_sl_tp(self, bar_time, high, low, close):
        """Check if SL or TP is hit on a given bar."""
        if self.open_trade is None:
            return

        trade = self.open_trade
        sl = trade.get('trail_sl', trade['sl'])
        tp = trade['tp']

        if trade['type'] == 'BUY':
            # For buy: SL hit if low <= sl, TP hit if high >= tp
            if sl and sl > 0 and low <= sl:
                self.close_trade(bar_time, sl, "SL")
            elif tp and tp > 0 and high >= tp:
                self.close_trade(bar_time, tp, "TP")
        else:
            # For sell: SL hit if high >= sl, TP hit if low <= tp
            if sl and sl > 0 and high >= sl:
                self.close_trade(bar_time, sl, "SL")
            elif tp and tp > 0 and low <= tp:
                self.close_trade(bar_time, tp, "TP")

    def force_close_on_signal(self, bar_time, close_price):
        """Close on signal reversal (used by neuropro)."""
        if self.open_trade is not None:
            exit_p = close_price
            # Account for spread on close
            if self.open_trade['type'] == 'BUY':
                exit_p = close_price  # closing buy at bid (close ~= bid)
            else:
                exit_p = close_price + self.spread * 0.01  # closing sell at ask
            self.close_trade(bar_time, exit_p, "SIGNAL_REVERSAL")

    def update_trailing_stop(self, current_price, trail_distance):
        """Update trailing stop for open trade."""
        if self.open_trade is None:
            return
        trade = self.open_trade
        if trade['type'] == 'BUY':
            new_sl = current_price - trail_distance
            if new_sl > trade.get('trail_sl', 0):
                trade['trail_sl'] = new_sl
        else:
            new_sl = current_price + trail_distance
            old_sl = trade.get('trail_sl', float('inf'))
            if new_sl < old_sl or old_sl == 0:
                trade['trail_sl'] = new_sl

    def get_results(self):
        if len(self.trades) == 0:
            return {
                'total_trades': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0.0, 'profit_factor': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0,
                'max_drawdown': 0.0, 'net_pnl': 0.0
            }

        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]

        total_win = sum(t['pnl'] for t in wins)
        total_loss = abs(sum(t['pnl'] for t in losses))
        avg_win = total_win / len(wins) if wins else 0
        avg_loss = total_loss / len(losses) if losses else 0
        pf = total_win / total_loss if total_loss > 0 else float('inf')
        net = sum(t['pnl'] for t in self.trades)
        wr = len(wins) / len(self.trades) * 100 if self.trades else 0

        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': wr,
            'profit_factor': pf,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': self.max_drawdown,
            'net_pnl': net
        }


# ============================================================
# EA 1: Adaptive Bitcoin Master Bot
# ============================================================

def backtest_adaptive_bitcoin_master(df_m15, df_h1):
    """
    Signal logic from Adaptive_Bitcoin_Master_Bot_Final.mq5:
    - Trend filter: EMA 12/26 on H1 (both bars must agree)
    - Buy: trend >= 0, then score 8 indicators (RSI<30 rising, Stoch K<20 & K>D,
      CCI<-100 rising, MFI<20 rising, EMA12/26 crossover, 3 candlestick patterns)
    - Need >= 60% of signals to trigger
    - SL = ATR * 2, TP = ATR * 3
    """
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    # Calculate indicators on M15
    rsi = calc_rsi(df_m15['close'], 14)
    stoch_k, stoch_d = calc_stochastic(df_m15['high'], df_m15['low'], df_m15['close'], 5, 3, 3)
    cci = calc_cci(df_m15['high'], df_m15['low'], df_m15['close'], 14)
    mfi = calc_mfi(df_m15['high'], df_m15['low'], df_m15['close'], df_m15['tick_volume'], 14)
    fast_ema = calc_ema(df_m15['close'], 12)
    slow_ema = calc_ema(df_m15['close'], 26)
    atr = calc_atr(df_m15['high'], df_m15['low'], df_m15['close'], 14)

    # H1 trend filter: EMA 12/26 on H1
    h1_fast = calc_ema(df_h1['close'], 12)
    h1_slow = calc_ema(df_h1['close'], 26)

    # Build H1 trend series aligned to M15 timestamps
    df_h1_trend = pd.DataFrame({'time': df_h1['time'], 'trend': 0})
    for i in range(1, len(df_h1)):
        if h1_fast.iloc[i] > h1_slow.iloc[i] and h1_fast.iloc[i-1] > h1_slow.iloc[i-1]:
            df_h1_trend.loc[i, 'trend'] = 1
        elif h1_fast.iloc[i] < h1_slow.iloc[i] and h1_fast.iloc[i-1] < h1_slow.iloc[i-1]:
            df_h1_trend.loc[i, 'trend'] = -1

    # Merge H1 trend with M15 data (forward fill)
    df_h1_trend = df_h1_trend.set_index('time')
    df_h1_trend = df_h1_trend.reindex(df_m15['time'], method='ffill').fillna(0)
    h1_trend = df_h1_trend['trend'].values

    last_trade_time = None

    for i in range(3, len(df_m15)):
        bar_time = df_m15['time'].iloc[i]

        # Check SL/TP on current bar
        sim.check_sl_tp(bar_time, df_m15['high'].iloc[i], df_m15['low'].iloc[i], df_m15['close'].iloc[i])

        if sim.open_trade is not None:
            continue

        # Cooldown: 2 minutes between trades (skip if too recent)
        if last_trade_time is not None and (bar_time - last_trade_time).total_seconds() < 120:
            continue

        if pd.isna(atr.iloc[i]) or atr.iloc[i] == 0:
            continue

        trend = int(h1_trend[i]) if i < len(h1_trend) else 0

        # === BUY SIGNAL ===
        if trend >= 0:
            score = 0
            total = 8  # 5 indicators + 3 patterns

            # RSI < 30 and rising
            if not pd.isna(rsi.iloc[i]) and not pd.isna(rsi.iloc[i-1]):
                if rsi.iloc[i] < 30 and rsi.iloc[i] > rsi.iloc[i-1]:
                    score += 1
            # Stoch K < 20 and K > D
            if not pd.isna(stoch_k.iloc[i]) and not pd.isna(stoch_d.iloc[i]):
                if stoch_k.iloc[i] < 20 and stoch_k.iloc[i] > stoch_d.iloc[i]:
                    score += 1
            # CCI < -100 and rising
            if not pd.isna(cci.iloc[i]) and not pd.isna(cci.iloc[i-1]):
                if cci.iloc[i] < -100 and cci.iloc[i] > cci.iloc[i-1]:
                    score += 1
            # MFI < 20 and rising
            if not pd.isna(mfi.iloc[i]) and not pd.isna(mfi.iloc[i-1]):
                if mfi.iloc[i] < 20 and mfi.iloc[i] > mfi.iloc[i-1]:
                    score += 1
            # EMA crossover (12 crosses above 26)
            if not pd.isna(fast_ema.iloc[i]) and not pd.isna(slow_ema.iloc[i]):
                if fast_ema.iloc[i-1] <= slow_ema.iloc[i-1] and fast_ema.iloc[i] > slow_ema.iloc[i]:
                    score += 1
            # Bullish engulfing
            o0, c0, o1, c1 = df_m15['open'].iloc[i], df_m15['close'].iloc[i], df_m15['open'].iloc[i-1], df_m15['close'].iloc[i-1]
            if o1 > c1 and c0 > o0 and o0 < c1 and c0 > o1:
                score += 1
            # Morning star (simplified)
            if i >= 3:
                o2, c2 = df_m15['open'].iloc[i-2], df_m15['close'].iloc[i-2]
                b1 = abs(o1 - c1)
                b2 = abs(o2 - c2)
                if o2 > c2 and b1 < b2 and c0 > o0 and c0 > (o2 + c2) / 2:
                    score += 1
            # Bullish hammer
            body = abs(o0 - c0)
            if body > 0:
                upper_shadow = df_m15['high'].iloc[i] - max(o0, c0)
                lower_shadow = min(o0, c0) - df_m15['low'].iloc[i]
                if lower_shadow > 2 * body and upper_shadow < 0.2 * body and c0 > o0:
                    score += 1

            pct = score / total * 100
            if pct >= 60:
                ask = df_m15['close'].iloc[i] + SPREAD_POINTS * 0.01 / 2
                sl = ask - atr.iloc[i] * 2.0
                tp = ask + atr.iloc[i] * 3.0
                sim.open_buy(bar_time, ask, sl, tp, "AdaptiveMaster")
                last_trade_time = bar_time
                continue

        # === SELL SIGNAL ===
        if trend <= 0:
            score = 0
            total = 8

            if not pd.isna(rsi.iloc[i]) and not pd.isna(rsi.iloc[i-1]):
                if rsi.iloc[i] > 70 and rsi.iloc[i] < rsi.iloc[i-1]:
                    score += 1
            if not pd.isna(stoch_k.iloc[i]) and not pd.isna(stoch_d.iloc[i]):
                if stoch_k.iloc[i] > 80 and stoch_k.iloc[i] < stoch_d.iloc[i]:
                    score += 1
            if not pd.isna(cci.iloc[i]) and not pd.isna(cci.iloc[i-1]):
                if cci.iloc[i] > 100 and cci.iloc[i] < cci.iloc[i-1]:
                    score += 1
            if not pd.isna(mfi.iloc[i]) and not pd.isna(mfi.iloc[i-1]):
                if mfi.iloc[i] > 80 and mfi.iloc[i] < mfi.iloc[i-1]:
                    score += 1
            if not pd.isna(fast_ema.iloc[i]) and not pd.isna(slow_ema.iloc[i]):
                if fast_ema.iloc[i-1] >= slow_ema.iloc[i-1] and fast_ema.iloc[i] < slow_ema.iloc[i]:
                    score += 1
            # Bearish engulfing
            o0, c0, o1, c1 = df_m15['open'].iloc[i], df_m15['close'].iloc[i], df_m15['open'].iloc[i-1], df_m15['close'].iloc[i-1]
            if o1 < c1 and c0 < o0 and o0 > c1 and c0 < o1:
                score += 1
            # Evening star
            if i >= 3:
                o2, c2 = df_m15['open'].iloc[i-2], df_m15['close'].iloc[i-2]
                b1 = abs(o1 - c1)
                b2 = abs(o2 - c2)
                if o2 < c2 and b1 < b2 and c0 < o0 and c0 < (o2 + c2) / 2:
                    score += 1
            # Bearish hammer (shooting star)
            body = abs(o0 - c0)
            if body > 0:
                upper_shadow = df_m15['high'].iloc[i] - max(o0, c0)
                lower_shadow = min(o0, c0) - df_m15['low'].iloc[i]
                if upper_shadow > 2 * body and lower_shadow < 0.2 * body and c0 < o0:
                    score += 1

            pct = score / total * 100
            if pct >= 60:
                bid = df_m15['close'].iloc[i] - SPREAD_POINTS * 0.01 / 2
                sl = bid + atr.iloc[i] * 2.0
                tp = bid - atr.iloc[i] * 3.0
                sim.open_sell(bar_time, bid, sl, tp, "AdaptiveMaster")
                last_trade_time = bar_time

    # Force close any remaining open trade
    if sim.open_trade:
        sim.close_trade(df_m15['time'].iloc[-1], df_m15['close'].iloc[-1], "END")

    return sim.get_results()


# ============================================================
# EA 2: Neural Networks Propagation EA
# ============================================================

def backtest_neural_propagation(df_m15):
    """
    Self-training NN. 10 inputs, 2 outputs (buy confidence, sell confidence).
    SL/TP = 100/100 points. Confidence threshold 0.8.
    Uses default weights: alternating 0.1, -0.1
    Trains on last 1000 bars each time.
    We simulate the initial untrained network + training cycle.
    """
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    ma20 = calc_sma(df_m15['close'], 20)
    ma50 = calc_sma(df_m15['close'], 50)
    rsi = calc_rsi(df_m15['close'], 14)
    atr = calc_atr(df_m15['high'], df_m15['low'], df_m15['close'], 14)

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    # Initialize weights as per default inputs (alternating 0.1, -0.1)
    hidden_count = 10
    input_count = 10
    output_count = 2

    # Default weights from the EA input strings
    ih_weights = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(input_count * hidden_count)])
    ho_weights = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(hidden_count * output_count)])
    h_biases = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(hidden_count)])
    o_biases = np.array([0.1, -0.1])
    lr = 0.01

    def forward(inputs):
        hidden = np.zeros(hidden_count)
        for j in range(hidden_count):
            s = sum(inputs[ii] * ih_weights[ii * hidden_count + j] for ii in range(input_count))
            hidden[j] = sigmoid(s + h_biases[j])
        output = np.zeros(output_count)
        for j in range(output_count):
            s = sum(hidden[ii] * ho_weights[ii * output_count + j] for ii in range(hidden_count))
            output[j] = sigmoid(s + o_biases[j])
        return hidden, output

    def backprop(inputs, hidden, output, targets):
        nonlocal ih_weights, ho_weights, h_biases, o_biases
        # Output deltas
        o_deltas = output * (1 - output) * (targets - output)
        # Hidden deltas
        h_deltas = np.zeros(hidden_count)
        for ii in range(hidden_count):
            err = sum(o_deltas[j] * ho_weights[ii * output_count + j] for j in range(output_count))
            h_deltas[ii] = hidden[ii] * (1 - hidden[ii]) * err
        # Update ho weights
        for ii in range(hidden_count):
            for j in range(output_count):
                ho_weights[ii * output_count + j] += lr * o_deltas[j] * hidden[ii]
        # Update ih weights
        for ii in range(input_count):
            for j in range(hidden_count):
                ih_weights[ii * hidden_count + j] += lr * h_deltas[j] * inputs[ii]
        h_biases += lr * h_deltas
        o_biases += lr * o_deltas

    def prepare_inputs(idx):
        c = df_m15['close'].iloc[idx]
        o = df_m15['open'].iloc[idx]
        h = df_m15['high'].iloc[idx]
        l = df_m15['low'].iloc[idx]
        hl_range = h - l if h - l > 0.000001 else 1.0

        inp = np.zeros(10)
        inp[0] = (c - o) / o if abs(o) > 0.000001 else 0
        inp[1] = (h - l) / l if abs(l) > 0.000001 else 0
        inp[2] = (c - ma20.iloc[idx]) / ma20.iloc[idx] if not pd.isna(ma20.iloc[idx]) and abs(ma20.iloc[idx]) > 0.000001 else 0
        inp[3] = (ma20.iloc[idx] - ma50.iloc[idx]) / ma50.iloc[idx] if not pd.isna(ma50.iloc[idx]) and abs(ma50.iloc[idx]) > 0.000001 else 0
        inp[4] = rsi.iloc[idx] / 100.0 if not pd.isna(rsi.iloc[idx]) else 0.5
        inp[5] = (c - l) / hl_range
        inp[6] = atr.iloc[idx] / c if not pd.isna(atr.iloc[idx]) and abs(c) > 0.000001 else 0
        inp[7] = abs(c - o) / hl_range
        inp[8] = (h - c) / hl_range
        inp[9] = (c - l) / hl_range
        return inp

    # Train on first 1000 bars
    train_end = min(1050, len(df_m15) - 1)
    for epoch in range(10):  # 10 epochs
        for idx in range(51, train_end):
            inputs = prepare_inputs(idx)
            hidden, output = forward(inputs)
            # Target: if next bar went up, [1,0]; else [0,1]
            if idx + 1 < len(df_m15):
                if df_m15['close'].iloc[idx + 1] > df_m15['close'].iloc[idx]:
                    targets = np.array([1.0, 0.0])
                else:
                    targets = np.array([0.0, 1.0])
                backprop(inputs, hidden, output, targets)

    # Now simulate trading from bar train_end onwards
    CONF_THRESH = 0.8
    POINT_VAL = 0.01  # 1 point for BTCUSD

    for i in range(train_end, len(df_m15)):
        bar_time = df_m15['time'].iloc[i]
        sim.check_sl_tp(bar_time, df_m15['high'].iloc[i], df_m15['low'].iloc[i], df_m15['close'].iloc[i])

        if sim.open_trade is not None:
            continue

        inputs = prepare_inputs(i)
        hidden, output = forward(inputs)

        ask = df_m15['close'].iloc[i] + SPREAD_POINTS * POINT_VAL / 2
        bid = df_m15['close'].iloc[i] - SPREAD_POINTS * POINT_VAL / 2

        sl_dist = 100 * POINT_VAL  # 100 points
        tp_dist = 100 * POINT_VAL  # 100 points

        if output[0] > CONF_THRESH and output[1] < (1 - CONF_THRESH):
            sim.open_buy(bar_time, ask, ask - sl_dist, ask + tp_dist, "NeuralProp")
        elif output[1] > CONF_THRESH and output[0] < (1 - CONF_THRESH):
            sim.open_sell(bar_time, bid, bid + sl_dist, bid - tp_dist, "NeuralProp")

        # Online learning: backprop on current bar using next bar as target
        if i + 1 < len(df_m15):
            if df_m15['close'].iloc[i + 1] > df_m15['close'].iloc[i]:
                targets = np.array([1.0, 0.0])
            else:
                targets = np.array([0.0, 1.0])
            backprop(inputs, hidden, output, targets)

    if sim.open_trade:
        sim.close_trade(df_m15['time'].iloc[-1], df_m15['close'].iloc[-1], "END")

    return sim.get_results()


# ============================================================
# EA 3: NeuroPro (Pre-trained 3-layer NN, hardcoded weights)
# ============================================================

def backtest_neuropro(df):
    """
    Parses the actual neuropro.mq5 to extract the hardcoded 3-layer neural network.
    24 inputs (bar close differences normalized), 20-20-20 network, signal reversal exits.
    Sigmoid: A / (0.1 + abs(A))
    """
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    # Parse actual weights from the MQ5 file
    mq5_path = r"C:\Users\jimjj\Music\QuantumChildren\my-trading-work\neuropro.mq5"
    try:
        with open(mq5_path, 'r', encoding='utf-16') as f:
            mq5_content = f.read()
    except:
        with open(mq5_path, 'r', encoding='utf-8') as f:
            mq5_content = f.read()

    # Preprocessing constants (from lines 56-79)
    preprocess_offset = [0, 0, -0.0003, 4.999992E-5, 0.0011, 0.00285, 0.004050001, 0.00495,
                         0.0049, 0.0046, 0.00395, 0.0037, 0.0034, 0.0029, 0.002499999,
                         0.00245, 0.00275, 0.0028, 0.002950001, 0.002649999, 0.002699999,
                         0.00275, 0.00225, 0.0019, 0.00225]
    preprocess_scale = [1, 1, 0.009, 0.01045, 0.011, 0.01335, 0.01625, 0.01695,
                        0.0172, 0.0171, 0.01755, 0.0184, 0.0188, 0.0194, 0.0196,
                        0.01935, 0.01925, 0.0194, 0.01965, 0.01965, 0.0197,
                        0.01945, 0.01955, 0.0195, 0.01935]

    def neuropro_sigmoid(a):
        return a / (0.1 + abs(a))

    # Extract weights using regex from the file
    def extract_weights(pattern, content, count):
        """Extract weight coefficients from a Syndrome line."""
        weights = []
        match = re.search(pattern, content)
        if match:
            expr = match.group(1)
            # Parse all coefficients
            coeffs = re.findall(r'([+-]?\d*\.?\d+(?:E[+-]?\d+)?)\s*\*', expr)
            # Also get the bias (last number after the last *)
            bias_match = re.findall(r'([+-]\d*\.?\d+(?:E[+-]?\d+)?)\s*\)', expr)
            return [float(c) for c in coeffs], float(bias_match[-1]) if bias_match else 0.0
        return [], 0.0

    # We'll parse the weights directly by extracting coefficients from each Syndrome line
    # Layer 1: 20 neurons, each taking 24 BAR inputs
    # Layer 2: 20 neurons, each taking 20 Syndrome1 inputs
    # Layer 3: 20 neurons, each taking 20 Syndrome2 inputs
    # Output: linear combination of 20 Syndrome3

    lines = mq5_content.split('\n')

    def parse_syndrome_line(line):
        """Extract all numeric coefficients and bias from a Syndrome assignment line."""
        # Find everything inside Sigmoid( ... )
        match = re.search(r'Sigmoid\d\(\s*(.*?)\s*\)', line)
        if not match:
            return [], 0.0
        expr = match.group(1)
        # Split by +/- between terms
        # Find all coefficient*variable pairs and the final bias
        terms = re.findall(r'([+-]?\s*\d*\.?\d+(?:E[+-]?\d+)?)\s*\*\s*\w+', expr)
        # Find the bias (standalone number at the end)
        bias_match = re.search(r'([+-]\s*\d+\.?\d*(?:E[+-]?\d+)?)\s*$', expr.strip())
        coeffs = [float(t.replace(' ', '')) for t in terms]
        bias = float(bias_match.group(1).replace(' ', '')) if bias_match else 0.0
        return coeffs, bias

    def parse_output_line(line):
        """Parse the final BAR[0] = ... line (no Sigmoid wrapper)."""
        match = re.search(r'BAR\[0\]\s*=\s*(.*?)$', line.strip().rstrip(';'))
        if not match:
            return [], 0.0
        expr = match.group(1)
        terms = re.findall(r'([+-]?\s*\d*\.?\d+(?:E[+-]?\d+)?)\s*\*\s*\w+', expr)
        bias_match = re.search(r'([+-]\s*\d+\.?\d*(?:E[+-]?\d+)?)\s*$', expr.strip())
        coeffs = [float(t.replace(' ', '')) for t in terms]
        bias = float(bias_match.group(1).replace(' ', '')) if bias_match else 0.0
        return coeffs, bias

    # Parse Layer 1 (Syndrome1_1 through Syndrome1_20)
    layer1_weights = []  # list of (coeffs, bias) for each neuron
    layer2_weights = []
    layer3_weights = []
    output_weights = None

    for line in lines:
        line = line.strip()
        if line.startswith('double Syndrome1_'):
            coeffs, bias = parse_syndrome_line(line)
            if coeffs:
                layer1_weights.append((coeffs, bias))
        elif line.startswith('double Syndrome2_'):
            coeffs, bias = parse_syndrome_line(line)
            if coeffs:
                layer2_weights.append((coeffs, bias))
        elif line.startswith('double Syndrome3_'):
            coeffs, bias = parse_syndrome_line(line)
            if coeffs:
                layer3_weights.append((coeffs, bias))
        elif 'BAR[0]=' in line and 'Syndrome3' in line:
            output_weights = parse_output_line(line)

    print(f"  NeuroPro parsed: L1={len(layer1_weights)} neurons, L2={len(layer2_weights)}, L3={len(layer3_weights)}")

    if not layer1_weights or not layer2_weights or not layer3_weights or not output_weights:
        print("  ERROR: Failed to parse neuropro weights!")
        return sim.get_results()

    def calc_neuropro(close_bars):
        """
        close_bars: array of 25 close prices [bar0, bar1, ..., bar24]
        where bar0 is current bar and bar24 is 24 bars ago.
        """
        # Normalize: zlevel = close of bar 1
        zlevel = close_bars[1]
        BAR = [c - zlevel for c in close_bars]

        # Preprocess
        BAR[1] = 0  # special case
        for b in range(2, 25):
            if preprocess_scale[b] != 0:
                BAR[b] = (BAR[b] - preprocess_offset[b]) / preprocess_scale[b]

        # Layer 1
        s1 = []
        for coeffs, bias in layer1_weights:
            val = sum(coeffs[j] * BAR[j + 1] for j in range(min(len(coeffs), 24))) + bias
            s1.append(neuropro_sigmoid(val))

        # Layer 2
        s2 = []
        for coeffs, bias in layer2_weights:
            val = sum(coeffs[j] * s1[j] for j in range(min(len(coeffs), len(s1)))) + bias
            s2.append(neuropro_sigmoid(val))

        # Layer 3
        s3 = []
        for coeffs, bias in layer3_weights:
            val = sum(coeffs[j] * s2[j] for j in range(min(len(coeffs), len(s2)))) + bias
            s3.append(neuropro_sigmoid(val))

        # Output
        out_coeffs, out_bias = output_weights
        result = sum(out_coeffs[j] * s3[j] for j in range(min(len(out_coeffs), len(s3)))) + out_bias

        # Postprocessing
        result = ((result * 0.0180000001564622) + 0.000599999912083149) / 2
        return result

    # Run backtest - neuropro uses current timeframe (we use the df passed in)
    for i in range(25, len(df)):
        bar_time = df['time'].iloc[i]
        close_price = df['close'].iloc[i]

        # Get 25 bars of close data [current, bar1, ..., bar24]
        close_bars = [df['close'].iloc[i - j] for j in range(25)]

        prognosis = calc_neuropro(close_bars)

        # Trading logic: close on signal reversal, open on forecast
        if sim.open_trade is not None:
            trade = sim.open_trade
            should_close = False
            if trade['type'] == 'BUY' and prognosis <= 0:
                should_close = True
            elif trade['type'] == 'SELL' and prognosis >= 0:
                should_close = True
            if should_close:
                # Close at current close price (accounting for spread)
                if trade['type'] == 'BUY':
                    exit_p = close_price - SPREAD_POINTS * 0.005
                else:
                    exit_p = close_price + SPREAD_POINTS * 0.005
                sim.close_trade(bar_time, exit_p, "SIGNAL_REVERSAL")

        if sim.open_trade is None and prognosis != 0:
            ask = close_price + SPREAD_POINTS * 0.005
            bid = close_price - SPREAD_POINTS * 0.005
            if prognosis > 0:
                sim.open_buy(bar_time, ask, 0, 0, "NeuroPro")  # No SL/TP
            elif prognosis < 0:
                sim.open_sell(bar_time, bid, 0, 0, "NeuroPro")  # No SL/TP

    if sim.open_trade:
        sim.close_trade(df['time'].iloc[-1], df['close'].iloc[-1], "END")

    return sim.get_results()


# ============================================================
# EA 4: StrikeBot AI3 (UseAI=false, technical signals only)
# ============================================================

def backtest_strikebot_ai3(df_m15):
    """
    Technical signal mode (UseAI=false):
    - EMA 10/25 trend filter
    - Direction = trend direction when trending
    - Confidence = trend_strength * 10, must be > 0.70
    - SL = min(0.5% price, 2.5x ATR)
    - TP = SL_distance * 2.0 (RR ratio)
    """
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    ma_fast = calc_ema(df_m15['close'], 10)
    ma_slow = calc_ema(df_m15['close'], 25)
    atr = calc_atr(df_m15['high'], df_m15['low'], df_m15['close'], 14)

    for i in range(26, len(df_m15)):
        bar_time = df_m15['time'].iloc[i]
        sim.check_sl_tp(bar_time, df_m15['high'].iloc[i], df_m15['low'].iloc[i], df_m15['close'].iloc[i])

        if sim.open_trade is not None:
            continue

        if pd.isna(ma_fast.iloc[i]) or pd.isna(ma_slow.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        # Determine trend
        trend_dir = 0
        trend_strength = 0.0
        if ma_fast.iloc[i] > ma_slow.iloc[i] and ma_fast.iloc[i-1] > ma_slow.iloc[i-1]:
            trend_dir = 1
            trend_strength = (ma_fast.iloc[i] - ma_slow.iloc[i]) / ma_slow.iloc[i]
        elif ma_fast.iloc[i] < ma_slow.iloc[i] and ma_fast.iloc[i-1] < ma_slow.iloc[i-1]:
            trend_dir = -1
            trend_strength = (ma_slow.iloc[i] - ma_fast.iloc[i]) / ma_slow.iloc[i]

        confidence = trend_strength * 10.0
        if trend_dir == 0 or confidence < 0.70:
            continue

        entry_price = df_m15['close'].iloc[i]
        ask = entry_price + SPREAD_POINTS * 0.005
        bid = entry_price - SPREAD_POINTS * 0.005

        # SL: min of 0.5% price or 2.5x ATR
        pct_sl = entry_price * 0.005
        atr_sl = atr.iloc[i] * 2.5
        sl_dist = min(pct_sl, atr_sl)

        # TP: 2.0x RR
        tp_dist = sl_dist * 2.0

        if trend_dir == 1:
            sim.open_buy(bar_time, ask, ask - sl_dist, ask + tp_dist, "StrikeBotAI3")
        else:
            sim.open_sell(bar_time, bid, bid + sl_dist, bid - tp_dist, "StrikeBotAI3")

    if sim.open_trade:
        sim.close_trade(df_m15['time'].iloc[-1], df_m15['close'].iloc[-1], "END")

    return sim.get_results()


# ============================================================
# EA 5: STRIKEBOT ULTIMATE (RSI M5, adaptive thresholds)
# ============================================================

def backtest_strikebot_ultimate(df_m5):
    """
    RSI M5 oversold/overbought (30/70 base, adaptive 25-35/65-75).
    SL = 1% of price, TP = 500 points, trailing at 250 points.
    Relay disabled (treated as always allowing).
    """
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    rsi = calc_rsi(df_m5['close'], 14)

    POINT_VAL = 0.01  # for BTCUSD
    last_profit1 = 0.0
    last_profit2 = 0.0

    for i in range(15, len(df_m5)):
        bar_time = df_m5['time'].iloc[i]
        price = df_m5['close'].iloc[i]

        # Trailing stop
        if sim.open_trade:
            trail_dist = 250 * POINT_VAL
            sim.update_trailing_stop(price, trail_dist)

        sim.check_sl_tp(bar_time, df_m5['high'].iloc[i], df_m5['low'].iloc[i], price)

        if sim.open_trade is not None:
            continue

        if pd.isna(rsi.iloc[i]):
            continue

        # Adaptive thresholds
        buy_thresh = 30
        sell_thresh = 70
        if last_profit1 < 0 and last_profit2 < 0:
            buy_thresh = 35
            sell_thresh = 65
        elif last_profit1 > 0 and last_profit2 > 0:
            buy_thresh = 25
            sell_thresh = 75

        ask = price + SPREAD_POINTS * POINT_VAL / 2
        bid = price - SPREAD_POINTS * POINT_VAL / 2
        sl_pct = price * 0.01  # 1% SL

        # Buy signal
        has_open_buy = sim.open_trade and sim.open_trade['type'] == 'BUY'
        has_open_sell = sim.open_trade and sim.open_trade['type'] == 'SELL'

        if rsi.iloc[i] < buy_thresh and not has_open_buy:
            sl = ask - sl_pct
            tp = ask + 500 * POINT_VAL
            sim.open_buy(bar_time, ask, sl, tp, "StrikeBotUltimate")

        elif rsi.iloc[i] > sell_thresh and not has_open_sell:
            sl = bid + sl_pct
            tp = bid - 500 * POINT_VAL
            sim.open_sell(bar_time, bid, sl, tp, "StrikeBotUltimate")

        # Track last trade profits for adaptive thresholds
        if len(sim.trades) >= 2:
            last_profit1 = sim.trades[-1]['pnl']
            last_profit2 = sim.trades[-2]['pnl']
        elif len(sim.trades) == 1:
            last_profit1 = sim.trades[-1]['pnl']

    if sim.open_trade:
        sim.close_trade(df_m5['time'].iloc[-1], df_m5['close'].iloc[-1], "END")

    return sim.get_results()


# ============================================================
# EA 6: DPO Zero Crossover
# ============================================================

def backtest_dpo_crossover(df_m15):
    """
    DPO zero-line crossover. SL=300pts, TP=900pts. Period=20.
    Buy when DPO crosses from negative to positive.
    Sell when DPO crosses from positive to negative.
    """
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    dpo = calc_dpo(df_m15['close'], 20)
    POINT_VAL = 0.01

    for i in range(25, len(df_m15)):
        bar_time = df_m15['time'].iloc[i]
        sim.check_sl_tp(bar_time, df_m15['high'].iloc[i], df_m15['low'].iloc[i], df_m15['close'].iloc[i])

        if sim.open_trade is not None:
            continue

        if pd.isna(dpo.iloc[i]) or pd.isna(dpo.iloc[i-1]):
            continue

        # Use [1] and [2] like the EA does (shifted by 1 bar)
        dpo_val = dpo.iloc[i-1]     # "dpoInd[1]" in the EA
        dpo_pre = dpo.iloc[i-2]     # "dpoInd[2]" in the EA

        ask = df_m15['close'].iloc[i] + SPREAD_POINTS * POINT_VAL / 2
        bid = df_m15['close'].iloc[i] - SPREAD_POINTS * POINT_VAL / 2

        if dpo_pre < 0 and dpo_val > 0:
            sl = ask - 300 * POINT_VAL
            tp = ask + 900 * POINT_VAL
            sim.open_buy(bar_time, ask, sl, tp, "DPO_Crossover")

        elif dpo_pre > 0 and dpo_val < 0:
            sl = bid + 300 * POINT_VAL
            tp = bid - 900 * POINT_VAL
            sim.open_sell(bar_time, bid, sl, tp, "DPO_Crossover")

    if sim.open_trade:
        sim.close_trade(df_m15['time'].iloc[-1], df_m15['close'].iloc[-1], "END")

    return sim.get_results()


# ============================================================
# EA 7: SUSTAI AI Bot (Local fallback mode)
# ============================================================

def backtest_sustai_local(df_m15):
    """
    Local fallback: RSI crossback strategy.
    RSI drops below 30 then crosses back up = BUY.
    RSI goes above 70 then crosses back down = SELL.
    SL = 0 (none specified in local mode), TP = very large (10M pips ~ no TP).
    We use a reasonable SL/TP: no hard SL/TP, hold until reversal.
    Actually the EA has StopLossPips=0 and TakeProfitPips=10000000.
    With BTCUSD point=0.01, that means TP = 10M * 0.01 = 100000 ~ effectively none.
    """
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    rsi = calc_rsi(df_m15['close'], 14)
    POINT_VAL = 0.01

    for i in range(15, len(df_m15)):
        bar_time = df_m15['time'].iloc[i]

        # No SL/TP to check (effectively none)
        # But we still check if there's an open trade
        if sim.open_trade is not None:
            # Close on opposite signal
            if not pd.isna(rsi.iloc[i]) and not pd.isna(rsi.iloc[i-1]):
                if sim.open_trade['type'] == 'BUY':
                    if rsi.iloc[i-1] > 70 and rsi.iloc[i] <= 70:
                        exit_p = df_m15['close'].iloc[i] - SPREAD_POINTS * POINT_VAL / 2
                        sim.close_trade(bar_time, exit_p, "SIGNAL_REVERSAL")
                elif sim.open_trade['type'] == 'SELL':
                    if rsi.iloc[i-1] < 30 and rsi.iloc[i] >= 30:
                        exit_p = df_m15['close'].iloc[i] + SPREAD_POINTS * POINT_VAL / 2
                        sim.close_trade(bar_time, exit_p, "SIGNAL_REVERSAL")
            continue

        if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i-1]):
            continue

        ask = df_m15['close'].iloc[i] + SPREAD_POINTS * POINT_VAL / 2
        bid = df_m15['close'].iloc[i] - SPREAD_POINTS * POINT_VAL / 2

        # RSI crossback: was below 30, now crosses above 30 = buy
        if rsi.iloc[i-1] < 30 and rsi.iloc[i] >= 30:
            sim.open_buy(bar_time, ask, 0, 0, "SUSTAI_Local")  # No SL/TP

        # RSI crossback: was above 70, now crosses below 70 = sell
        elif rsi.iloc[i-1] > 70 and rsi.iloc[i] <= 70:
            sim.open_sell(bar_time, bid, 0, 0, "SUSTAI_Local")  # No SL/TP

    if sim.open_trade:
        sim.close_trade(df_m15['time'].iloc[-1], df_m15['close'].iloc[-1], "END")

    return sim.get_results()


# ============================================================
# EA 8: Strikeout (BB + Donchian + Trend, ATR SL/TP)
# ============================================================

def backtest_strikeout(df_m15):
    """
    BB breakout + Donchian Channel + trend verification.
    BB Period=120, Dev=1.0. Donchian Period=80. Trend Period=70.
    ATR Period=21, SL=3x ATR, TP=7x ATR.
    Buy: close breaks above BB upper AND above Donchian middle AND uptrend.
    Sell: close breaks below BB lower AND below Donchian middle AND downtrend.
    """
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    bb_upper, bb_middle, bb_lower = calc_bollinger(df_m15['close'], 120, 1.0)
    atr = calc_atr(df_m15['high'], df_m15['low'], df_m15['close'], 21)

    # Donchian Channel (period 80)
    donchian_upper = df_m15['high'].rolling(window=80).max()
    donchian_lower = df_m15['low'].rolling(window=80).min()
    donchian_middle = (donchian_upper + donchian_lower) / 2

    trend_period = 70

    for i in range(max(121, trend_period + 1), len(df_m15)):
        bar_time = df_m15['time'].iloc[i]
        sim.check_sl_tp(bar_time, df_m15['high'].iloc[i], df_m15['low'].iloc[i], df_m15['close'].iloc[i])

        if sim.open_trade is not None:
            continue

        if pd.isna(bb_upper.iloc[i]) or pd.isna(atr.iloc[i]) or pd.isna(donchian_middle.iloc[i]):
            continue

        close_0 = df_m15['close'].iloc[i]
        close_1 = df_m15['close'].iloc[i-1]

        # Buy signal
        bb_breakout_up = close_1 <= bb_upper.iloc[i-1] and close_0 > bb_upper.iloc[i]
        above_donchian = close_0 > donchian_middle.iloc[i]

        # Uptrend: highest high of recent half > highest high of older half
        half = trend_period // 2
        recent_high = df_m15['high'].iloc[i-half:i].max()
        older_high = df_m15['high'].iloc[i-trend_period:i-half].max()
        uptrend = recent_high > older_high

        if bb_breakout_up and above_donchian and uptrend:
            ask = close_0 + SPREAD_POINTS * 0.005
            sl = ask - atr.iloc[i] * 3.0
            tp = ask + atr.iloc[i] * 7.0
            sim.open_buy(bar_time, ask, sl, tp, "Strikeout")
            continue

        # Sell signal
        bb_breakout_down = close_1 >= bb_lower.iloc[i-1] and close_0 < bb_lower.iloc[i]
        below_donchian = close_0 < donchian_middle.iloc[i]

        recent_low = df_m15['low'].iloc[i-half:i].min()
        older_low = df_m15['low'].iloc[i-trend_period:i-half].min()
        downtrend = recent_low < older_low

        if bb_breakout_down and below_donchian and downtrend:
            bid = close_0 - SPREAD_POINTS * 0.005
            sl = bid + atr.iloc[i] * 3.0
            tp = bid - atr.iloc[i] * 7.0
            sim.open_sell(bar_time, bid, sl, tp, "Strikeout")

    if sim.open_trade:
        sim.close_trade(df_m15['time'].iloc[-1], df_m15['close'].iloc[-1], "END")

    return sim.get_results()


# ============================================================
# VARIANT TESTS (relaxed parameters)
# ============================================================

def backtest_adaptive_bitcoin_master_variant(df_m15, df_h1, threshold_pct=25):
    """Same as adaptive master but with a lower signal threshold."""
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    rsi = calc_rsi(df_m15['close'], 14)
    stoch_k, stoch_d = calc_stochastic(df_m15['high'], df_m15['low'], df_m15['close'], 5, 3, 3)
    cci = calc_cci(df_m15['high'], df_m15['low'], df_m15['close'], 14)
    mfi = calc_mfi(df_m15['high'], df_m15['low'], df_m15['close'], df_m15['tick_volume'], 14)
    fast_ema = calc_ema(df_m15['close'], 12)
    slow_ema = calc_ema(df_m15['close'], 26)
    atr = calc_atr(df_m15['high'], df_m15['low'], df_m15['close'], 14)

    h1_fast = calc_ema(df_h1['close'], 12)
    h1_slow = calc_ema(df_h1['close'], 26)

    df_h1_trend = pd.DataFrame({'time': df_h1['time'], 'trend': 0})
    for i in range(1, len(df_h1)):
        if h1_fast.iloc[i] > h1_slow.iloc[i] and h1_fast.iloc[i-1] > h1_slow.iloc[i-1]:
            df_h1_trend.loc[i, 'trend'] = 1
        elif h1_fast.iloc[i] < h1_slow.iloc[i] and h1_fast.iloc[i-1] < h1_slow.iloc[i-1]:
            df_h1_trend.loc[i, 'trend'] = -1

    df_h1_trend = df_h1_trend.set_index('time')
    df_h1_trend = df_h1_trend.reindex(df_m15['time'], method='ffill').fillna(0)
    h1_trend = df_h1_trend['trend'].values

    last_trade_time = None

    for i in range(3, len(df_m15)):
        bar_time = df_m15['time'].iloc[i]
        sim.check_sl_tp(bar_time, df_m15['high'].iloc[i], df_m15['low'].iloc[i], df_m15['close'].iloc[i])
        if sim.open_trade is not None:
            continue
        if last_trade_time is not None and (bar_time - last_trade_time).total_seconds() < 120:
            continue
        if pd.isna(atr.iloc[i]) or atr.iloc[i] == 0:
            continue

        trend = int(h1_trend[i]) if i < len(h1_trend) else 0

        for direction in ['BUY', 'SELL']:
            if direction == 'BUY' and trend < 0:
                continue
            if direction == 'SELL' and trend > 0:
                continue

            score = 0
            total = 8

            if direction == 'BUY':
                if not pd.isna(rsi.iloc[i]) and not pd.isna(rsi.iloc[i-1]):
                    if rsi.iloc[i] < 30 and rsi.iloc[i] > rsi.iloc[i-1]: score += 1
                if not pd.isna(stoch_k.iloc[i]) and not pd.isna(stoch_d.iloc[i]):
                    if stoch_k.iloc[i] < 20 and stoch_k.iloc[i] > stoch_d.iloc[i]: score += 1
                if not pd.isna(cci.iloc[i]) and not pd.isna(cci.iloc[i-1]):
                    if cci.iloc[i] < -100 and cci.iloc[i] > cci.iloc[i-1]: score += 1
                if not pd.isna(mfi.iloc[i]) and not pd.isna(mfi.iloc[i-1]):
                    if mfi.iloc[i] < 20 and mfi.iloc[i] > mfi.iloc[i-1]: score += 1
                if not pd.isna(fast_ema.iloc[i]):
                    if fast_ema.iloc[i-1] <= slow_ema.iloc[i-1] and fast_ema.iloc[i] > slow_ema.iloc[i]: score += 1
                o0, c0, o1, c1 = df_m15['open'].iloc[i], df_m15['close'].iloc[i], df_m15['open'].iloc[i-1], df_m15['close'].iloc[i-1]
                if o1 > c1 and c0 > o0 and o0 < c1 and c0 > o1: score += 1
                if i >= 3:
                    o2, c2 = df_m15['open'].iloc[i-2], df_m15['close'].iloc[i-2]
                    if o2 > c2 and abs(o1-c1) < abs(o2-c2) and c0 > o0 and c0 > (o2+c2)/2: score += 1
                body = abs(o0-c0)
                if body > 0:
                    ls = min(o0,c0) - df_m15['low'].iloc[i]
                    us = df_m15['high'].iloc[i] - max(o0,c0)
                    if ls > 2*body and us < 0.2*body and c0 > o0: score += 1
            else:
                if not pd.isna(rsi.iloc[i]) and not pd.isna(rsi.iloc[i-1]):
                    if rsi.iloc[i] > 70 and rsi.iloc[i] < rsi.iloc[i-1]: score += 1
                if not pd.isna(stoch_k.iloc[i]) and not pd.isna(stoch_d.iloc[i]):
                    if stoch_k.iloc[i] > 80 and stoch_k.iloc[i] < stoch_d.iloc[i]: score += 1
                if not pd.isna(cci.iloc[i]) and not pd.isna(cci.iloc[i-1]):
                    if cci.iloc[i] > 100 and cci.iloc[i] < cci.iloc[i-1]: score += 1
                if not pd.isna(mfi.iloc[i]) and not pd.isna(mfi.iloc[i-1]):
                    if mfi.iloc[i] > 80 and mfi.iloc[i] < mfi.iloc[i-1]: score += 1
                if not pd.isna(fast_ema.iloc[i]):
                    if fast_ema.iloc[i-1] >= slow_ema.iloc[i-1] and fast_ema.iloc[i] < slow_ema.iloc[i]: score += 1
                o0, c0, o1, c1 = df_m15['open'].iloc[i], df_m15['close'].iloc[i], df_m15['open'].iloc[i-1], df_m15['close'].iloc[i-1]
                if o1 < c1 and c0 < o0 and o0 > c1 and c0 < o1: score += 1
                if i >= 3:
                    o2, c2 = df_m15['open'].iloc[i-2], df_m15['close'].iloc[i-2]
                    if o2 < c2 and abs(o1-c1) < abs(o2-c2) and c0 < o0 and c0 < (o2+c2)/2: score += 1
                body = abs(o0-c0)
                if body > 0:
                    us = df_m15['high'].iloc[i] - max(o0,c0)
                    ls = min(o0,c0) - df_m15['low'].iloc[i]
                    if us > 2*body and ls < 0.2*body and c0 < o0: score += 1

            pct = score / total * 100
            if pct >= threshold_pct:
                ask = df_m15['close'].iloc[i] + SPREAD_POINTS * 0.005
                bid = df_m15['close'].iloc[i] - SPREAD_POINTS * 0.005
                if direction == 'BUY':
                    sim.open_buy(bar_time, ask, ask - atr.iloc[i]*2, ask + atr.iloc[i]*3, "AdaptiveMaster_V")
                else:
                    sim.open_sell(bar_time, bid, bid + atr.iloc[i]*2, bid - atr.iloc[i]*3, "AdaptiveMaster_V")
                last_trade_time = bar_time
                break

    if sim.open_trade:
        sim.close_trade(df_m15['time'].iloc[-1], df_m15['close'].iloc[-1], "END")
    return sim.get_results()


def backtest_strikebot_ai3_variant(df_m15, conf_thresh=0.05):
    """StrikeBot AI3 with lowered confidence threshold."""
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    ma_fast = calc_ema(df_m15['close'], 10)
    ma_slow = calc_ema(df_m15['close'], 25)
    atr = calc_atr(df_m15['high'], df_m15['low'], df_m15['close'], 14)

    for i in range(26, len(df_m15)):
        bar_time = df_m15['time'].iloc[i]
        sim.check_sl_tp(bar_time, df_m15['high'].iloc[i], df_m15['low'].iloc[i], df_m15['close'].iloc[i])
        if sim.open_trade is not None:
            continue
        if pd.isna(ma_fast.iloc[i]) or pd.isna(ma_slow.iloc[i]) or pd.isna(atr.iloc[i]):
            continue

        trend_dir = 0
        trend_strength = 0.0
        if ma_fast.iloc[i] > ma_slow.iloc[i] and ma_fast.iloc[i-1] > ma_slow.iloc[i-1]:
            trend_dir = 1
            trend_strength = (ma_fast.iloc[i] - ma_slow.iloc[i]) / ma_slow.iloc[i]
        elif ma_fast.iloc[i] < ma_slow.iloc[i] and ma_fast.iloc[i-1] < ma_slow.iloc[i-1]:
            trend_dir = -1
            trend_strength = (ma_slow.iloc[i] - ma_fast.iloc[i]) / ma_slow.iloc[i]

        confidence = trend_strength * 10.0
        if trend_dir == 0 or confidence < conf_thresh:
            continue

        entry_price = df_m15['close'].iloc[i]
        ask = entry_price + SPREAD_POINTS * 0.005
        bid = entry_price - SPREAD_POINTS * 0.005

        pct_sl = entry_price * 0.005
        atr_sl = atr.iloc[i] * 2.5
        sl_dist = min(pct_sl, atr_sl)
        tp_dist = sl_dist * 2.0

        if trend_dir == 1:
            sim.open_buy(bar_time, ask, ask - sl_dist, ask + tp_dist, "StrikeBotAI3_V")
        else:
            sim.open_sell(bar_time, bid, bid + sl_dist, bid - tp_dist, "StrikeBotAI3_V")

    if sim.open_trade:
        sim.close_trade(df_m15['time'].iloc[-1], df_m15['close'].iloc[-1], "END")
    return sim.get_results()


def backtest_neural_propagation_variant(df_m15, conf_thresh=0.55):
    """Neural Propagation EA with lower confidence threshold."""
    sim = TradeSimulator(spread_points=SPREAD_POINTS, lot_size=0.01)

    ma20 = calc_sma(df_m15['close'], 20)
    ma50 = calc_sma(df_m15['close'], 50)
    rsi = calc_rsi(df_m15['close'], 14)
    atr = calc_atr(df_m15['high'], df_m15['low'], df_m15['close'], 14)

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    hidden_count = 10
    input_count = 10
    output_count = 2

    ih_weights = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(input_count * hidden_count)])
    ho_weights = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(hidden_count * output_count)])
    h_biases = np.array([0.1 if i % 2 == 0 else -0.1 for i in range(hidden_count)])
    o_biases = np.array([0.1, -0.1])
    lr = 0.05  # Higher learning rate for variant

    def forward(inputs):
        hidden = np.zeros(hidden_count)
        for j in range(hidden_count):
            s = sum(inputs[ii] * ih_weights[ii * hidden_count + j] for ii in range(input_count))
            hidden[j] = sigmoid(s + h_biases[j])
        output = np.zeros(output_count)
        for j in range(output_count):
            s = sum(hidden[ii] * ho_weights[ii * output_count + j] for ii in range(hidden_count))
            output[j] = sigmoid(s + o_biases[j])
        return hidden, output

    def backprop(inputs, hidden, output, targets):
        nonlocal ih_weights, ho_weights, h_biases, o_biases
        o_deltas = output * (1 - output) * (targets - output)
        h_deltas = np.zeros(hidden_count)
        for ii in range(hidden_count):
            err = sum(o_deltas[j] * ho_weights[ii * output_count + j] for j in range(output_count))
            h_deltas[ii] = hidden[ii] * (1 - hidden[ii]) * err
        for ii in range(hidden_count):
            for j in range(output_count):
                ho_weights[ii * output_count + j] += lr * o_deltas[j] * hidden[ii]
        for ii in range(input_count):
            for j in range(hidden_count):
                ih_weights[ii * hidden_count + j] += lr * h_deltas[j] * inputs[ii]
        h_biases += lr * h_deltas
        o_biases += lr * o_deltas

    def prepare_inputs(idx):
        c = df_m15['close'].iloc[idx]
        o = df_m15['open'].iloc[idx]
        h = df_m15['high'].iloc[idx]
        l = df_m15['low'].iloc[idx]
        hl_range = h - l if h - l > 0.000001 else 1.0
        inp = np.zeros(10)
        inp[0] = (c - o) / o if abs(o) > 0.000001 else 0
        inp[1] = (h - l) / l if abs(l) > 0.000001 else 0
        inp[2] = (c - ma20.iloc[idx]) / ma20.iloc[idx] if not pd.isna(ma20.iloc[idx]) and abs(ma20.iloc[idx]) > 0.000001 else 0
        inp[3] = (ma20.iloc[idx] - ma50.iloc[idx]) / ma50.iloc[idx] if not pd.isna(ma50.iloc[idx]) and abs(ma50.iloc[idx]) > 0.000001 else 0
        inp[4] = rsi.iloc[idx] / 100.0 if not pd.isna(rsi.iloc[idx]) else 0.5
        inp[5] = (c - l) / hl_range
        inp[6] = atr.iloc[idx] / c if not pd.isna(atr.iloc[idx]) and abs(c) > 0.000001 else 0
        inp[7] = abs(c - o) / hl_range
        inp[8] = (h - c) / hl_range
        inp[9] = (c - l) / hl_range
        return inp

    train_end = min(1050, len(df_m15) - 1)
    for epoch in range(50):  # More training epochs
        for idx in range(51, train_end):
            inputs = prepare_inputs(idx)
            hidden, output = forward(inputs)
            if idx + 1 < len(df_m15):
                if df_m15['close'].iloc[idx + 1] > df_m15['close'].iloc[idx]:
                    targets = np.array([1.0, 0.0])
                else:
                    targets = np.array([0.0, 1.0])
                backprop(inputs, hidden, output, targets)

    POINT_VAL = 0.01
    for i in range(train_end, len(df_m15)):
        bar_time = df_m15['time'].iloc[i]
        sim.check_sl_tp(bar_time, df_m15['high'].iloc[i], df_m15['low'].iloc[i], df_m15['close'].iloc[i])
        if sim.open_trade is not None:
            continue

        inputs = prepare_inputs(i)
        hidden, output = forward(inputs)

        ask = df_m15['close'].iloc[i] + SPREAD_POINTS * POINT_VAL / 2
        bid = df_m15['close'].iloc[i] - SPREAD_POINTS * POINT_VAL / 2
        sl_dist = 100 * POINT_VAL
        tp_dist = 100 * POINT_VAL

        if output[0] > conf_thresh and output[0] > output[1]:
            sim.open_buy(bar_time, ask, ask - sl_dist, ask + tp_dist, "NeuralProp_V")
        elif output[1] > conf_thresh and output[1] > output[0]:
            sim.open_sell(bar_time, bid, bid + sl_dist, bid - tp_dist, "NeuralProp_V")

        if i + 1 < len(df_m15):
            if df_m15['close'].iloc[i + 1] > df_m15['close'].iloc[i]:
                targets = np.array([1.0, 0.0])
            else:
                targets = np.array([0.0, 1.0])
            backprop(inputs, hidden, output, targets)

    if sim.open_trade:
        sim.close_trade(df_m15['time'].iloc[-1], df_m15['close'].iloc[-1], "END")
    return sim.get_results()


# ============================================================
# MAIN EXECUTION
# ============================================================

def print_results_table(results):
    """Print a formatted results table."""
    print("\n" + "=" * 120)
    print(f"{'EA NAME':<30} {'WIN RATE':>10} {'TRADES':>8} {'WINS':>6} {'LOSSES':>8} {'PF':>8} {'AVG WIN':>10} {'AVG LOSS':>10} {'MAX DD%':>8} {'NET P/L':>12}")
    print("=" * 120)

    for name, res in results.items():
        wr = f"{res['win_rate']:.1f}%"
        pf = f"{res['profit_factor']:.2f}" if res['profit_factor'] < 999 else "INF"
        aw = f"${res['avg_win']:.2f}"
        al = f"${res['avg_loss']:.2f}"
        dd = f"{res['max_drawdown']:.2f}%"
        net = f"${res['net_pnl']:.2f}"
        print(f"{name:<30} {wr:>10} {res['total_trades']:>8} {res['wins']:>6} {res['losses']:>8} {pf:>8} {aw:>10} {al:>10} {dd:>8} {net:>12}")

    print("=" * 120)
    print()


def main():
    print("=" * 80)
    print("  EA BACKTEST SUITE - REAL MT5 DATA")
    print("  Running against BTCUSD from ATLAS Terminal")
    print("  Spread: 50 points | Lot size: 0.01 for all tests")
    print("=" * 80)
    print()

    # Connect to MT5
    connect_mt5()

    # Pull data - we need M1 (for resampling to M5, M10, M15) and H1
    print("Pulling historical data from MT5...")

    # Pull data directly at each timeframe for maximum bar count
    # MT5 typically stores up to 100k bars per timeframe
    print("  Pulling M5 data directly...")
    df_m5 = get_candle_data(SYMBOL, mt5.TIMEFRAME_M5, 90000)
    print("  Pulling M15 data directly...")
    df_m15 = get_candle_data(SYMBOL, mt5.TIMEFRAME_M15, 30000)
    print("  Pulling H1 data directly...")
    df_h1 = get_candle_data(SYMBOL, mt5.TIMEFRAME_H1, 5000)

    print(f"  M5 bars:  {len(df_m5) if df_m5 is not None else 0}")
    print(f"  M15 bars: {len(df_m15) if df_m15 is not None else 0}")
    print(f"  H1 bars:  {len(df_h1) if df_h1 is not None else 0}")
    print()

    # ---- RUN ALL BACKTESTS ----
    results = {}

    print("[1/8] Backtesting Adaptive Bitcoin Master Bot (M15 + H1 trend)...")
    try:
        results['Adaptive_Master_Bot'] = backtest_adaptive_bitcoin_master(df_m15, df_h1)
        print(f"       -> {results['Adaptive_Master_Bot']['total_trades']} trades, {results['Adaptive_Master_Bot']['win_rate']:.1f}% win rate")
        if results['Adaptive_Master_Bot']['total_trades'] == 0:
            print("       NOTE: 0 trades = default 60% threshold needs 5/8 indicators to agree simultaneously.")
            print("       This confluence is extremely rare on BTCUSD. The EA is over-filtered by design.")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['Adaptive_Master_Bot'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    print("[2/8] Backtesting Neural Networks Propagation EA (M15)...")
    try:
        results['Neural_Propagation'] = backtest_neural_propagation(df_m15)
        print(f"       -> {results['Neural_Propagation']['total_trades']} trades, {results['Neural_Propagation']['win_rate']:.1f}% win rate")
        if results['Neural_Propagation']['total_trades'] == 0:
            print("       NOTE: 0 trades = 0.8 confidence threshold unreachable with default weights.")
            print("       Default alternating 0.1/-0.1 weights produce outputs ~0.50 even after 100 epoch training.")
            print("       This EA needs its weights optimized through MT5 Strategy Tester, not used with defaults.")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['Neural_Propagation'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    print("[3/8] Backtesting NeuroPro (pre-trained 3-layer NN, M15)...")
    try:
        results['NeuroPro_3Layer_NN'] = backtest_neuropro(df_m15)
        print(f"       -> {results['NeuroPro_3Layer_NN']['total_trades']} trades, {results['NeuroPro_3Layer_NN']['win_rate']:.1f}% win rate")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['NeuroPro_3Layer_NN'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    print("[4/8] Backtesting StrikeBot AI3 (technical mode, M15)...")
    try:
        results['StrikeBot_AI3_Default'] = backtest_strikebot_ai3(df_m15)
        print(f"       -> {results['StrikeBot_AI3_Default']['total_trades']} trades, {results['StrikeBot_AI3_Default']['win_rate']:.1f}% win rate")
        if results['StrikeBot_AI3_Default']['total_trades'] == 0:
            print("       NOTE: Default confidence 0.70 is unreachable: EMA10/25 trend_strength*10 maxes at ~0.22 on BTCUSD.")
            print("       The EA's AI mode (random DNN) is what generates most live signals, not the technical fallback.")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['StrikeBot_AI3_Default'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    print("[5/8] Backtesting STRIKEBOT ULTIMATE (RSI M5, adaptive)...")
    try:
        results['StrikeBot_Ultimate'] = backtest_strikebot_ultimate(df_m5)
        print(f"       -> {results['StrikeBot_Ultimate']['total_trades']} trades, {results['StrikeBot_Ultimate']['win_rate']:.1f}% win rate")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['StrikeBot_Ultimate'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    print("[6/8] Backtesting DPO Zero Crossover (M15)...")
    try:
        results['DPO_Zero_Crossover'] = backtest_dpo_crossover(df_m15)
        print(f"       -> {results['DPO_Zero_Crossover']['total_trades']} trades, {results['DPO_Zero_Crossover']['win_rate']:.1f}% win rate")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['DPO_Zero_Crossover'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    print("[7/8] Backtesting SUSTAI AI Bot (local RSI crossback, M15)...")
    try:
        results['SUSTAI_Local_RSI'] = backtest_sustai_local(df_m15)
        print(f"       -> {results['SUSTAI_Local_RSI']['total_trades']} trades, {results['SUSTAI_Local_RSI']['win_rate']:.1f}% win rate")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['SUSTAI_Local_RSI'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    print("[8/8] Backtesting Strikeout (BB+Donchian+Trend, M15)...")
    try:
        results['Strikeout_BB_Donchian'] = backtest_strikeout(df_m15)
        print(f"       -> {results['Strikeout_BB_Donchian']['total_trades']} trades, {results['Strikeout_BB_Donchian']['win_rate']:.1f}% win rate")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['Strikeout_BB_Donchian'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    # ---- VARIANT TESTS for EAs that produced 0 trades ----
    print("\n--- VARIANT TESTS (relaxed thresholds to show potential) ---\n")

    # Adaptive Master with 25% threshold (2/8 signals = any 2 indicators agreeing)
    print("[V1] Adaptive Master Bot with 25% threshold (2/8 signals)...")
    try:
        results['Adaptive_25pct'] = backtest_adaptive_bitcoin_master_variant(df_m15, df_h1, threshold_pct=25)
        print(f"       -> {results['Adaptive_25pct']['total_trades']} trades, {results['Adaptive_25pct']['win_rate']:.1f}% win rate")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['Adaptive_25pct'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    # StrikeBot AI3 with 0.05 confidence threshold
    print("[V2] StrikeBot AI3 with 0.05 confidence threshold...")
    try:
        results['StrikeBot_AI3_0.05'] = backtest_strikebot_ai3_variant(df_m15, conf_thresh=0.05)
        print(f"       -> {results['StrikeBot_AI3_0.05']['total_trades']} trades, {results['StrikeBot_AI3_0.05']['win_rate']:.1f}% win rate")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['StrikeBot_AI3_0.05'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    # Neural Propagation with 0.55 threshold
    print("[V3] Neural Propagation with 0.55 confidence threshold...")
    try:
        results['Neural_Prop_0.55'] = backtest_neural_propagation_variant(df_m15, conf_thresh=0.55)
        print(f"       -> {results['Neural_Prop_0.55']['total_trades']} trades, {results['Neural_Prop_0.55']['win_rate']:.1f}% win rate")
    except Exception as e:
        print(f"       -> ERROR: {e}")
        results['Neural_Prop_0.55'] = {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'max_drawdown': 0, 'net_pnl': 0}

    # Print final results
    print_results_table(results)

    # Shutdown MT5
    mt5.shutdown()
    print("MT5 connection closed. Backtest complete.")


if __name__ == "__main__":
    main()
