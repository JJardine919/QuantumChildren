"""
BlueGuardian Strategy Simulation
Tests the current settings against historical data
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize MT5
if not mt5.initialize():
    print("MT5 init failed")
    exit()

# Settings (matching BlueGuardian)
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOT_SIZE = 0.06
MAX_LOSS_DOLLARS = 1.00  # $1 max loss per trade
TP_MULTIPLIER = 3.0      # TP = 3x SL
TRAILING_TRIGGER = 0.8   # Trail at 80% to TP

# EMA settings
EMA_FAST = 8
EMA_SLOW = 21
CONFIDENCE_THRESHOLD = 0.48

# Get historical data (last 24 hours of M1 data = 1440 bars)
print(f"\n{'='*60}")
print("BLUEGUARDIAN STRATEGY SIMULATION")
print(f"{'='*60}")
print(f"Symbol: {SYMBOL}")
print(f"Timeframe: M1")
print(f"Lot Size: {LOT_SIZE}")
print(f"Max Loss: ${MAX_LOSS_DOLLARS}")
print(f"TP Target: ${MAX_LOSS_DOLLARS * TP_MULTIPLIER}")
print(f"{'='*60}\n")

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1440)
if rates is None or len(rates) < 100:
    print("Failed to get historical data")
    mt5.shutdown()
    exit()

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Calculate EMAs
df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

# Calculate signal strength (normalized EMA difference)
df['ema_diff'] = df['ema_fast'] - df['ema_slow']
df['ema_diff_pct'] = df['ema_diff'] / df['close'] * 100

# Get symbol info for tick calculations
symbol_info = mt5.symbol_info(SYMBOL)
tick_value = symbol_info.trade_tick_value
tick_size = symbol_info.trade_tick_size

# Calculate SL distance for $1 max loss
sl_ticks = MAX_LOSS_DOLLARS / (tick_value * LOT_SIZE)
sl_distance = sl_ticks * tick_size

print(f"Calculated SL Distance: ${sl_distance:.2f}")
print(f"TP Distance: ${sl_distance * TP_MULTIPLIER:.2f}")
print(f"\nSimulating last 24 hours ({len(df)} bars)...\n")

# Simulation variables
trades = []
position = None  # {'type': 'BUY'/'SELL', 'entry': price, 'sl': price, 'tp': price, 'trailing': False}
balance = 5000.00  # Starting balance
starting_balance = balance

# Run simulation
for i in range(50, len(df)):  # Start at 50 to have enough EMA history
    row = df.iloc[i]
    price = row['close']
    high = row['high']
    low = row['low']
    ema_fast = row['ema_fast']
    ema_slow = row['ema_slow']

    # Calculate confidence (normalized EMA difference)
    ema_diff = abs(ema_fast - ema_slow)
    confidence = min(ema_diff / 100, 1.0)  # Normalize to 0-1

    # Determine signal
    if ema_fast > ema_slow:
        signal = 'BUY'
    elif ema_fast < ema_slow:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    # If we have a position, check SL/TP
    if position:
        pnl = 0
        closed = False
        close_reason = ""

        if position['type'] == 'BUY':
            # Check SL hit
            if low <= position['sl']:
                pnl = -MAX_LOSS_DOLLARS  # Fixed $1 loss
                closed = True
                close_reason = "SL HIT"
            # Check TP hit
            elif high >= position['tp']:
                pnl = MAX_LOSS_DOLLARS * TP_MULTIPLIER  # $3 profit (3x risk)
                closed = True
                close_reason = "TP HIT"
            # Check trailing
            else:
                progress = (price - position['entry']) / (position['tp'] - position['entry'])
                if progress >= TRAILING_TRIGGER and not position['trailing']:
                    position['trailing'] = True
                    # Move SL up
                    trail_sl = price - (sl_distance * 0.2)
                    if trail_sl > position['sl']:
                        position['sl'] = trail_sl
                elif position['trailing']:
                    # Update trailing SL
                    trail_sl = price - (sl_distance * 0.2)
                    if trail_sl > position['sl']:
                        position['sl'] = trail_sl

        else:  # SELL position
            # Check SL hit
            if high >= position['sl']:
                pnl = -MAX_LOSS_DOLLARS  # Fixed $1 loss
                closed = True
                close_reason = "SL HIT"
            # Check TP hit
            elif low <= position['tp']:
                pnl = MAX_LOSS_DOLLARS * TP_MULTIPLIER  # $3 profit
                closed = True
                close_reason = "TP HIT"
            # Check trailing
            else:
                progress = (position['entry'] - price) / (position['entry'] - position['tp'])
                if progress >= TRAILING_TRIGGER and not position['trailing']:
                    position['trailing'] = True
                    trail_sl = price + (sl_distance * 0.2)
                    if trail_sl < position['sl']:
                        position['sl'] = trail_sl
                elif position['trailing']:
                    trail_sl = price + (sl_distance * 0.2)
                    if trail_sl < position['sl']:
                        position['sl'] = trail_sl

        if closed:
            balance += pnl
            trades.append({
                'time': row['time'],
                'type': position['type'],
                'entry': position['entry'],
                'exit': position['sl'] if 'SL' in close_reason else position['tp'],
                'pnl': pnl,
                'reason': close_reason,
                'balance': balance
            })
            position = None

    # If no position and we have a signal with confidence
    if position is None and confidence >= CONFIDENCE_THRESHOLD:
        if signal == 'BUY':
            position = {
                'type': 'BUY',
                'entry': price,
                'sl': price - sl_distance,
                'tp': price + (sl_distance * TP_MULTIPLIER),
                'trailing': False
            }
        elif signal == 'SELL':
            position = {
                'type': 'SELL',
                'entry': price,
                'sl': price + sl_distance,
                'tp': price - (sl_distance * TP_MULTIPLIER),
                'trailing': False
            }

# Close any open position at end (estimate based on current price)
if position:
    final_price = df.iloc[-1]['close']
    if position['type'] == 'BUY':
        price_diff = final_price - position['entry']
        pnl = (price_diff / sl_distance) * MAX_LOSS_DOLLARS  # Proportional to SL distance
    else:
        price_diff = position['entry'] - final_price
        pnl = (price_diff / sl_distance) * MAX_LOSS_DOLLARS
    pnl = max(pnl, -MAX_LOSS_DOLLARS)  # Cap loss
    balance += pnl
    trades.append({
        'time': df.iloc[-1]['time'],
        'type': position['type'],
        'entry': position['entry'],
        'exit': final_price,
        'pnl': pnl,
        'reason': 'END OF SIM',
        'balance': balance
    })

# Results
print(f"{'='*60}")
print("SIMULATION RESULTS")
print(f"{'='*60}")
print(f"Total Trades: {len(trades)}")

if trades:
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]

    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0

    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Win: ${avg_win:.2f}")
    print(f"Avg Loss: ${avg_loss:.2f}")
    print(f"")
    print(f"Starting Balance: ${starting_balance:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Return: {(total_pnl/starting_balance)*100:.2f}%")

    print(f"\n{'='*60}")
    print("RECENT TRADES (last 10)")
    print(f"{'='*60}")
    for t in trades[-10:]:
        emoji = "+" if t['pnl'] > 0 else ""
        print(f"{t['time'].strftime('%H:%M')} | {t['type']:4} | Entry: {t['entry']:.2f} | Exit: {t['exit']:.2f} | {emoji}${t['pnl']:.2f} | {t['reason']}")

    # Calculate if it would pass prop firm rules
    max_drawdown = 0
    peak = starting_balance
    for t in trades:
        if t['balance'] > peak:
            peak = t['balance']
        dd = (peak - t['balance']) / starting_balance * 100
        if dd > max_drawdown:
            max_drawdown = dd

    print(f"\n{'='*60}")
    print("PROP FIRM CHECK")
    print(f"{'='*60}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Daily Loss Limit (5%): {'PASS' if max_drawdown < 5 else 'FAIL'}")
    print(f"Max Drawdown Limit (10%): {'PASS' if max_drawdown < 10 else 'FAIL'}")
    print(f"Profit Target (10%): {'PASS' if total_pnl/starting_balance >= 0.10 else f'Need ${starting_balance*0.10 - total_pnl:.2f} more'}")

else:
    print("No trades executed in simulation period")

mt5.shutdown()
print(f"\n{'='*60}")
