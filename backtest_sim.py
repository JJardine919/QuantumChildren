"""
BG_AggressiveCompetition v2.0 - Quick Backtest Simulation
Mirrors the MQ5 EA logic: EMA crossover + ATR SL/TP + Dynamic TP + Rolling SL
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# ============================================================
# PARAMETERS (match EA exactly)
# ============================================================
SYMBOL          = "BTCUSD"
TIMEFRAME       = mt5.TIMEFRAME_M1
LOT_SIZE        = 0.03
MAX_POSITIONS   = 20
GRID_SPACING_PTS = 200
BOTH_DIRECTIONS = True

# Risk Management
SL_ATR_MULT     = 1.0       # HARDCODED - SL = ATR x 1.0
TP_MULTIPLIER   = 3.0       # TP = SL x 3
DYN_TP_PERCENT  = 50.0      # Partial close at 50%
ROLLING_SL_MULT = 1.5       # Rolling SL multiplier

# Signal
FAST_EMA        = 5
SLOW_EMA        = 13
ATR_PERIOD      = 14
MIN_CONFIDENCE  = 0.3

# Backtest range
DAYS_BACK       = 30

# ============================================================

class Position:
    def __init__(self, ticket, direction, entry, lot, sl, tp, dyn_tp):
        self.ticket = ticket
        self.direction = direction  # 1=BUY, -1=SELL
        self.entry = entry
        self.lot = lot
        self.sl = sl
        self.tp = tp
        self.dyn_tp = dyn_tp
        self.dyn_tp_taken = False
        self.rolling_sl = sl
        self.remaining_lot = lot
        self.pnl = 0.0
        self.closed = False
        self.close_reason = ""
        self.close_price = 0.0

def run_backtest():
    if not mt5.initialize():
        print("ERROR: MT5 failed to initialize")
        return

    print("=" * 60)
    print("  BG_AggressiveCompetition v2.0 - BACKTEST SIMULATION")
    print("=" * 60)
    print(f"  Symbol:         {SYMBOL}")
    print(f"  Timeframe:      M1")
    print(f"  Period:         Last {DAYS_BACK} days")
    print(f"  SL:             ATR x {SL_ATR_MULT} (hardcoded)")
    print(f"  TP:             SL x {TP_MULTIPLIER} (param)")
    print(f"  Dynamic TP:     {DYN_TP_PERCENT}% partial close")
    print(f"  Rolling SL:     {ROLLING_SL_MULT}x")
    print(f"  Fast EMA:       {FAST_EMA}")
    print(f"  Slow EMA:       {SLOW_EMA}")
    print(f"  Lot:            {LOT_SIZE}")
    print("=" * 60)

    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_BACK)

    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"ERROR: No data for {SYMBOL}")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"  Bars loaded:    {len(df)}")
    print(f"  Date range:     {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    # Calculate indicators
    df['ema_fast'] = df['close'].ewm(span=FAST_EMA, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=SLOW_EMA, adjust=False).mean()

    # ATR calculation
    df['hl'] = df['high'] - df['low']
    df['hc'] = abs(df['high'] - df['close'].shift(1))
    df['lc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()

    # Get symbol info for point value
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"ERROR: Symbol {SYMBOL} not found")
        mt5.shutdown()
        return

    point = symbol_info.point
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    contract_size = symbol_info.trade_contract_size

    print(f"  Point:          {point}")
    print(f"  Tick value:     ${tick_value}")
    print(f"  Tick size:      {tick_size}")
    print("=" * 60)

    # Simulation state
    positions = []
    closed_trades = []
    next_ticket = 1000
    initial_balance = 100000.0
    balance = initial_balance
    peak_balance = initial_balance
    max_dd = 0.0
    max_dd_pct = 0.0

    # Skip warmup period
    start_bar = max(SLOW_EMA, ATR_PERIOD) + 5

    for i in range(start_bar, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']
        atr = row['atr']

        if pd.isna(atr) or atr == 0:
            continue

        # --- MANAGE EXISTING POSITIONS ---
        for pos in positions:
            if pos.closed:
                continue

            current = price

            # Check SL hit
            if pos.direction == 1 and current <= pos.rolling_sl:
                pos.pnl = (pos.rolling_sl - pos.entry) * pos.remaining_lot * contract_size
                pos.closed = True
                pos.close_price = pos.rolling_sl
                pos.close_reason = "SL" if pos.rolling_sl == pos.sl else "ROLLING_SL"
                balance += pos.pnl
                closed_trades.append(pos)
                continue

            if pos.direction == -1 and current >= pos.rolling_sl:
                pos.pnl = (pos.entry - pos.rolling_sl) * pos.remaining_lot * contract_size
                pos.closed = True
                pos.close_price = pos.rolling_sl
                pos.close_reason = "SL" if pos.rolling_sl == pos.sl else "ROLLING_SL"
                balance += pos.pnl
                closed_trades.append(pos)
                continue

            # Check full TP hit
            if pos.direction == 1 and current >= pos.tp:
                pos.pnl = (pos.tp - pos.entry) * pos.remaining_lot * contract_size
                pos.closed = True
                pos.close_price = pos.tp
                pos.close_reason = "TP"
                balance += pos.pnl
                closed_trades.append(pos)
                continue

            if pos.direction == -1 and current <= pos.tp:
                pos.pnl = (pos.entry - pos.tp) * pos.remaining_lot * contract_size
                pos.closed = True
                pos.close_price = pos.tp
                pos.close_reason = "TP"
                balance += pos.pnl
                closed_trades.append(pos)
                continue

            # Dynamic TP - partial close at 50%
            if not pos.dyn_tp_taken:
                hit_dyn = False
                if pos.direction == 1 and current >= pos.dyn_tp:
                    hit_dyn = True
                if pos.direction == -1 and current <= pos.dyn_tp:
                    hit_dyn = True

                if hit_dyn:
                    # Close half
                    half_lot = pos.remaining_lot * 0.5
                    if pos.direction == 1:
                        partial_pnl = (current - pos.entry) * half_lot * contract_size
                    else:
                        partial_pnl = (pos.entry - current) * half_lot * contract_size

                    balance += partial_pnl
                    pos.remaining_lot -= half_lot
                    pos.dyn_tp_taken = True
                    # Move SL to breakeven
                    pos.rolling_sl = pos.entry

            # Rolling SL after dynamic TP taken
            if pos.dyn_tp_taken:
                sl_dist = abs(pos.entry - pos.sl)
                roll_target = sl_dist * ROLLING_SL_MULT

                if pos.direction == 1:
                    profit = current - pos.entry
                    if profit > roll_target:
                        new_sl = current - sl_dist
                        if new_sl > pos.rolling_sl:
                            pos.rolling_sl = new_sl

                elif pos.direction == -1:
                    profit = pos.entry - current
                    if profit > roll_target:
                        new_sl = current + sl_dist
                        if new_sl < pos.rolling_sl or pos.rolling_sl <= 0:
                            pos.rolling_sl = new_sl

        # Remove closed positions from active list
        positions = [p for p in positions if not p.closed]

        # Track drawdown
        equity = balance
        for pos in positions:
            if pos.direction == 1:
                equity += (price - pos.entry) * pos.remaining_lot * contract_size
            else:
                equity += (pos.entry - price) * pos.remaining_lot * contract_size

        if equity > peak_balance:
            peak_balance = equity
        dd = peak_balance - equity
        dd_pct = (dd / peak_balance) * 100 if peak_balance > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        # --- CHECK FOR NEW ENTRIES ---
        ema_f = row['ema_fast']
        ema_s = row['ema_slow']

        buy_signal = ema_f > ema_s
        sell_signal = ema_f < ema_s

        separation = abs(ema_f - ema_s)
        confidence = min(1.0, (separation / atr) * 0.5)
        if confidence < MIN_CONFIDENCE:
            continue

        buy_count = sum(1 for p in positions if p.direction == 1)
        sell_count = sum(1 for p in positions if p.direction == -1)
        total_pos = buy_count + sell_count

        if total_pos >= MAX_POSITIONS:
            continue

        spacing = GRID_SPACING_PTS * point

        # BUY entry
        if buy_signal and buy_count < MAX_POSITIONS // 2:
            last_buy = max([p.entry for p in positions if p.direction == 1], default=0)
            should_buy = (buy_count == 0) or (price < last_buy - spacing)

            if should_buy:
                sl_dist = atr * SL_ATR_MULT
                tp_dist = sl_dist * TP_MULTIPLIER
                dyn_dist = tp_dist * (DYN_TP_PERCENT / 100.0)

                sl = price - sl_dist
                tp = price + tp_dist
                dyn_tp = price + dyn_dist

                pos = Position(next_ticket, 1, price, LOT_SIZE, sl, tp, dyn_tp)
                positions.append(pos)
                next_ticket += 1

        # SELL entry
        if BOTH_DIRECTIONS and sell_signal and sell_count < MAX_POSITIONS // 2:
            last_sell = min([p.entry for p in positions if p.direction == -1], default=999999999)
            should_sell = (sell_count == 0) or (price > last_sell + spacing)

            if should_sell:
                sl_dist = atr * SL_ATR_MULT
                tp_dist = sl_dist * TP_MULTIPLIER
                dyn_dist = tp_dist * (DYN_TP_PERCENT / 100.0)

                sl = price + sl_dist
                tp = price - tp_dist
                dyn_tp = price - dyn_dist

                pos = Position(next_ticket, -1, price, LOT_SIZE, sl, tp, dyn_tp)
                positions.append(pos)
                next_ticket += 1

    # Close remaining positions at last price
    last_price = df.iloc[-1]['close']
    for pos in positions:
        if not pos.closed:
            if pos.direction == 1:
                pos.pnl = (last_price - pos.entry) * pos.remaining_lot * contract_size
            else:
                pos.pnl = (pos.entry - last_price) * pos.remaining_lot * contract_size
            pos.closed = True
            pos.close_price = last_price
            pos.close_reason = "END"
            balance += pos.pnl
            closed_trades.append(pos)

    # ============================================================
    # RESULTS
    # ============================================================
    total_trades = len(closed_trades)
    winners = [t for t in closed_trades if t.pnl > 0]
    losers = [t for t in closed_trades if t.pnl <= 0]
    win_count = len(winners)
    loss_count = len(losers)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    net_profit = balance - initial_balance
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    avg_win = (gross_profit / win_count) if win_count > 0 else 0
    avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0

    # Close reasons
    sl_closes = sum(1 for t in closed_trades if t.close_reason == "SL")
    tp_closes = sum(1 for t in closed_trades if t.close_reason == "TP")
    rolling_closes = sum(1 for t in closed_trades if t.close_reason == "ROLLING_SL")
    end_closes = sum(1 for t in closed_trades if t.close_reason == "END")

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Initial Balance:   ${initial_balance:,.2f}")
    print(f"  Final Balance:     ${balance:,.2f}")
    print(f"  Net Profit:        ${net_profit:,.2f}")
    print(f"  Return:            {(net_profit/initial_balance)*100:.2f}%")
    print("-" * 60)
    print(f"  Total Trades:      {total_trades}")
    print(f"  Winners:           {win_count}")
    print(f"  Losers:            {loss_count}")
    print(f"  Win Rate:          {win_rate:.1f}%")
    print("-" * 60)
    print(f"  Gross Profit:      ${gross_profit:,.2f}")
    print(f"  Gross Loss:        ${gross_loss:,.2f}")
    print(f"  Profit Factor:     {profit_factor:.2f}")
    print(f"  Avg Win:           ${avg_win:,.2f}")
    print(f"  Avg Loss:          ${avg_loss:,.2f}")
    print(f"  Avg Win/Loss:      {(avg_win/avg_loss):.2f}x" if avg_loss > 0 else "  Avg Win/Loss:      N/A")
    print("-" * 60)
    print(f"  Max Drawdown:      ${max_dd:,.2f}")
    print(f"  Max Drawdown %:    {max_dd_pct:.2f}%")
    print("-" * 60)
    print(f"  Closed by SL:      {sl_closes}")
    print(f"  Closed by TP:      {tp_closes}")
    print(f"  Closed by Roll SL: {rolling_closes}")
    print(f"  Open at end:       {end_closes}")
    print("=" * 60)

    mt5.shutdown()

if __name__ == "__main__":
    run_backtest()
