"""
FTMO Challenge Backtest Simulation
===================================
Runs BG_AggressiveCompetition v2.0 logic against BTCUSD M1 data
with FTMO challenge rules at both $25K and $100K sizes.

Tests if the risk:reward ratio holds across account sizes.
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# ============================================================
# EA PARAMETERS (match BG_AggressiveCompetition v2.0)
# ============================================================
SYMBOL          = "BTCUSD"
TIMEFRAME       = mt5.TIMEFRAME_M1
MAX_POSITIONS   = 20
GRID_SPACING_PTS = 200
BOTH_DIRECTIONS = True

# Risk Management - THE SPEC
SL_ATR_MULT     = 1.0       # HARDCODED - SL = ATR x 1.0
TP_MULTIPLIER   = 3.0       # TP = SL x 3
DYN_TP_PERCENT  = 50.0      # Partial close at 50%
ROLLING_SL_MULT = 1.5       # Rolling SL multiplier

# Signal
FAST_EMA        = 5
SLOW_EMA        = 13
ATR_PERIOD      = 14
MIN_CONFIDENCE  = 0.3

# Out-of-sample: test on DIFFERENT months than training
# Training was recent 30 days - test on 60-90 days ago
OOS_START_DAYS_AGO = 90   # Start 90 days back
OOS_END_DAYS_AGO   = 60   # End 60 days back (30-day window)

# ============================================================
# FTMO CHALLENGE RULES
# ============================================================
FTMO_RULES = {
    # --- $25K Dynamic (0.01 to 0.08) ---
    "25K": {
        "name": "$25K FTMO (0.01-0.08 dynamic)",
        "balance": 25000.0,
        "lot_min": 0.01,
        "lot_max": 0.08,
        "max_daily_dd": 1250.0,    # 5% of 25K
        "max_total_dd": 2500.0,    # 10% of 25K
        "profit_target": 2500.0,   # 10% of 25K
    },
    # --- $100K Dynamic (0.03 to 0.24) ---
    "100K": {
        "name": "$100K FTMO (0.03-0.24 dynamic)",
        "balance": 100000.0,
        "lot_min": 0.03,
        "lot_max": 0.24,
        "max_daily_dd": 5000.0,    # 5% of 100K
        "max_total_dd": 10000.0,   # 10% of 100K
        "profit_target": 10000.0,  # 10% of 100K
    },
}


class Position:
    def __init__(self, ticket, direction, entry, lot, sl, tp, dyn_tp):
        self.ticket = ticket
        self.direction = direction
        self.entry = entry
        self.lot = lot
        self.sl = sl
        self.tp = tp
        self.dyn_tp = dyn_tp
        self.dyn_tp_taken = False
        self.rolling_sl = sl
        self.remaining_lot = lot
        self.partial_pnl = 0.0
        self.pnl = 0.0
        self.closed = False
        self.close_reason = ""
        self.close_price = 0.0
        self.open_time = None


def calc_dynamic_lot(rules, confidence, dd_remaining_pct):
    """
    Dynamic lot sizing: scales between lot_min and lot_max.

    Factors:
      - confidence: higher signal confidence = bigger lot
      - dd_remaining_pct: % of DD budget left (1.0 = full, 0.0 = blown)
        Throttles down when DD budget is getting tight
    """
    lot_min = rules["lot_min"]
    lot_max = rules["lot_max"]

    # Confidence drives 70% of lot decision
    conf_factor = min(1.0, max(0.0, (confidence - MIN_CONFIDENCE) / (1.0 - MIN_CONFIDENCE)))

    # DD budget drives 30% - throttle when below 50% of DD remaining
    if dd_remaining_pct < 0.3:
        dd_factor = 0.0  # Near limit, go minimum
    elif dd_remaining_pct < 0.5:
        dd_factor = (dd_remaining_pct - 0.3) / 0.2  # Scale 0->1 between 30-50%
    else:
        dd_factor = 1.0  # Plenty of room

    combined = conf_factor * 0.7 + dd_factor * 0.3
    lot = lot_min + (lot_max - lot_min) * combined

    # Round to 0.01
    lot = round(lot, 2)
    lot = max(lot_min, min(lot_max, lot))
    return lot


def run_ftmo_sim(rule_key):
    rules = FTMO_RULES[rule_key]

    # Get data
    end_date = datetime.now() - timedelta(days=OOS_END_DAYS_AGO)
    start_date = datetime.now() - timedelta(days=OOS_START_DAYS_AGO)
    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)

    if rates is None or len(rates) == 0:
        print(f"ERROR: No data for {SYMBOL}")
        mt5.shutdown()
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['date'] = df['time'].dt.date

    # Indicators
    df['ema_fast'] = df['close'].ewm(span=FAST_EMA, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=SLOW_EMA, adjust=False).mean()
    df['hl'] = df['high'] - df['low']
    df['hc'] = abs(df['high'] - df['close'].shift(1))
    df['lc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"ERROR: Symbol {SYMBOL} not found")
        mt5.shutdown()
        return None

    point = symbol_info.point
    contract_size = symbol_info.trade_contract_size

    # State
    positions = []
    closed_trades = []
    next_ticket = 1000
    initial_balance = rules["balance"]
    balance = initial_balance
    peak_balance = initial_balance
    max_dd = 0.0
    max_dd_pct = 0.0
    lot_sizes_used = []  # Track dynamic lot decisions

    # FTMO tracking
    challenge_passed = False
    challenge_failed = False
    fail_reason = ""
    day_start_equity = initial_balance
    current_day = None
    daily_pnl = 0.0
    days_traded = 0
    target_hit_bar = None

    start_bar = max(SLOW_EMA, ATR_PERIOD) + 5

    for i in range(start_bar, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = row['close']
        atr = row['atr']
        bar_date = row['date']

        if pd.isna(atr) or atr == 0:
            continue

        # --- NEW DAY CHECK ---
        if bar_date != current_day:
            if current_day is not None:
                days_traded += 1
            current_day = bar_date
            floating = 0.0
            for pos in positions:
                if not pos.closed:
                    if pos.direction == 1:
                        floating += (price - pos.entry) * pos.remaining_lot * contract_size
                    else:
                        floating += (pos.entry - price) * pos.remaining_lot * contract_size
            day_start_equity = balance + floating
            daily_pnl = 0.0

        # --- FTMO RULE CHECKS ---
        floating = 0.0
        for pos in positions:
            if not pos.closed:
                if pos.direction == 1:
                    floating += (price - pos.entry) * pos.remaining_lot * contract_size
                else:
                    floating += (pos.entry - price) * pos.remaining_lot * contract_size
        equity = balance + floating

        # Daily drawdown check (from start of day equity)
        daily_dd = day_start_equity - equity
        if daily_dd >= rules["max_daily_dd"]:
            if not challenge_failed:
                challenge_failed = True
                fail_reason = f"Daily DD exceeded: ${daily_dd:,.2f} >= ${rules['max_daily_dd']:,.2f}"
            continue

        # Total drawdown check (from initial balance)
        total_dd = initial_balance - equity
        if total_dd >= rules["max_total_dd"]:
            if not challenge_failed:
                challenge_failed = True
                fail_reason = f"Total DD exceeded: ${total_dd:,.2f} >= ${rules['max_total_dd']:,.2f}"
            break

        # Profit target check
        net_profit = equity - initial_balance
        if net_profit >= rules["profit_target"] and not challenge_passed:
            challenge_passed = True
            target_hit_bar = i

        # DD budget remaining for dynamic lot calc
        dd_remaining = rules["max_total_dd"] - max(0, total_dd)
        dd_remaining_pct = dd_remaining / rules["max_total_dd"]

        # --- MANAGE POSITIONS ---
        for pos in positions:
            if pos.closed:
                continue

            current = price

            # SL hit
            if pos.direction == 1 and current <= pos.rolling_sl:
                pos.pnl = (pos.rolling_sl - pos.entry) * pos.remaining_lot * contract_size + pos.partial_pnl
                pos.closed = True
                pos.close_price = pos.rolling_sl
                pos.close_reason = "SL" if pos.rolling_sl == pos.sl else "ROLLING_SL"
                balance += (pos.rolling_sl - pos.entry) * pos.remaining_lot * contract_size
                daily_pnl += (pos.rolling_sl - pos.entry) * pos.remaining_lot * contract_size
                closed_trades.append(pos)
                continue

            if pos.direction == -1 and current >= pos.rolling_sl:
                pos.pnl = (pos.entry - pos.rolling_sl) * pos.remaining_lot * contract_size + pos.partial_pnl
                pos.closed = True
                pos.close_price = pos.rolling_sl
                pos.close_reason = "SL" if pos.rolling_sl == pos.sl else "ROLLING_SL"
                balance += (pos.entry - pos.rolling_sl) * pos.remaining_lot * contract_size
                daily_pnl += (pos.entry - pos.rolling_sl) * pos.remaining_lot * contract_size
                closed_trades.append(pos)
                continue

            # TP hit
            if pos.direction == 1 and current >= pos.tp:
                pos.pnl = (pos.tp - pos.entry) * pos.remaining_lot * contract_size + pos.partial_pnl
                pos.closed = True
                pos.close_price = pos.tp
                pos.close_reason = "TP"
                balance += (pos.tp - pos.entry) * pos.remaining_lot * contract_size
                daily_pnl += (pos.tp - pos.entry) * pos.remaining_lot * contract_size
                closed_trades.append(pos)
                continue

            if pos.direction == -1 and current <= pos.tp:
                pos.pnl = (pos.entry - pos.tp) * pos.remaining_lot * contract_size + pos.partial_pnl
                pos.closed = True
                pos.close_price = pos.tp
                pos.close_reason = "TP"
                balance += (pos.entry - pos.tp) * pos.remaining_lot * contract_size
                daily_pnl += (pos.entry - pos.tp) * pos.remaining_lot * contract_size
                closed_trades.append(pos)
                continue

            # Dynamic TP - 50% partial close
            if not pos.dyn_tp_taken:
                hit_dyn = False
                if pos.direction == 1 and current >= pos.dyn_tp:
                    hit_dyn = True
                if pos.direction == -1 and current <= pos.dyn_tp:
                    hit_dyn = True

                if hit_dyn:
                    half_lot = pos.remaining_lot * 0.5
                    if pos.direction == 1:
                        partial = (current - pos.entry) * half_lot * contract_size
                    else:
                        partial = (pos.entry - current) * half_lot * contract_size
                    balance += partial
                    daily_pnl += partial
                    pos.partial_pnl += partial
                    pos.remaining_lot -= half_lot
                    pos.dyn_tp_taken = True
                    pos.rolling_sl = pos.entry  # Move to breakeven

            # Rolling SL
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

        positions = [p for p in positions if not p.closed]

        # Track peak/drawdown
        if equity > peak_balance:
            peak_balance = equity
        dd = peak_balance - equity
        dd_pct = (dd / peak_balance) * 100 if peak_balance > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        # --- CHECK ENTRIES ---
        if challenge_failed:
            continue

        ema_f = row['ema_fast']
        ema_s = row['ema_slow']

        buy_signal = ema_f > ema_s
        sell_signal = ema_f < ema_s

        separation = abs(ema_f - ema_s)
        confidence = min(1.0, (separation / atr) * 0.5)
        if confidence < MIN_CONFIDENCE:
            continue

        # DYNAMIC LOT SIZING - let it decide based on risk
        lot_size = calc_dynamic_lot(rules, confidence, dd_remaining_pct)

        buy_count = sum(1 for p in positions if p.direction == 1)
        sell_count = sum(1 for p in positions if p.direction == -1)
        total_pos = buy_count + sell_count
        if total_pos >= MAX_POSITIONS:
            continue

        spacing = GRID_SPACING_PTS * point

        if buy_signal and buy_count < MAX_POSITIONS // 2:
            last_buy = max([p.entry for p in positions if p.direction == 1], default=0)
            if (buy_count == 0) or (price < last_buy - spacing):
                sl_dist = atr * SL_ATR_MULT
                tp_dist = sl_dist * TP_MULTIPLIER
                dyn_dist = tp_dist * (DYN_TP_PERCENT / 100.0)
                sl = price - sl_dist
                tp = price + tp_dist
                dyn_tp = price + dyn_dist
                pos = Position(next_ticket, 1, price, lot_size, sl, tp, dyn_tp)
                pos.open_time = row['time']
                positions.append(pos)
                lot_sizes_used.append(lot_size)
                next_ticket += 1

        if BOTH_DIRECTIONS and sell_signal and sell_count < MAX_POSITIONS // 2:
            last_sell = min([p.entry for p in positions if p.direction == -1], default=999999999)
            if (sell_count == 0) or (price > last_sell + spacing):
                sl_dist = atr * SL_ATR_MULT
                tp_dist = sl_dist * TP_MULTIPLIER
                dyn_dist = tp_dist * (DYN_TP_PERCENT / 100.0)
                sl = price + sl_dist
                tp = price - tp_dist
                dyn_tp = price - dyn_dist
                pos = Position(next_ticket, -1, price, lot_size, sl, tp, dyn_tp)
                pos.open_time = row['time']
                positions.append(pos)
                lot_sizes_used.append(lot_size)
                next_ticket += 1

    # Close remaining at last price
    last_price = df.iloc[-1]['close']
    for pos in positions:
        if not pos.closed:
            if pos.direction == 1:
                pnl = (last_price - pos.entry) * pos.remaining_lot * contract_size
            else:
                pnl = (pos.entry - last_price) * pos.remaining_lot * contract_size
            pos.pnl = pnl + pos.partial_pnl
            pos.closed = True
            pos.close_price = last_price
            pos.close_reason = "END"
            balance += pnl
            closed_trades.append(pos)

    days_traded += 1  # Count last day

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

    sl_closes = sum(1 for t in closed_trades if t.close_reason == "SL")
    tp_closes = sum(1 for t in closed_trades if t.close_reason == "TP")
    rolling_closes = sum(1 for t in closed_trades if t.close_reason == "ROLLING_SL")
    end_closes = sum(1 for t in closed_trades if t.close_reason == "END")

    # Lot size stats
    avg_lot = np.mean(lot_sizes_used) if lot_sizes_used else 0
    min_lot_used = min(lot_sizes_used) if lot_sizes_used else 0
    max_lot_used = max(lot_sizes_used) if lot_sizes_used else 0

    return {
        "name": rules["name"],
        "initial_balance": initial_balance,
        "final_balance": balance,
        "net_profit": net_profit,
        "return_pct": (net_profit / initial_balance) * 100,
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_rr": (avg_win / avg_loss) if avg_loss > 0 else 0,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "sl_closes": sl_closes,
        "tp_closes": tp_closes,
        "rolling_closes": rolling_closes,
        "end_closes": end_closes,
        "days_traded": days_traded,
        "challenge_passed": challenge_passed,
        "challenge_failed": challenge_failed,
        "fail_reason": fail_reason,
        "profit_target": rules["profit_target"],
        "max_daily_dd_rule": rules["max_daily_dd"],
        "max_total_dd_rule": rules["max_total_dd"],
        "target_progress": (net_profit / rules["profit_target"]) * 100,
        "bars": len(df),
        "date_range": f"{df['time'].iloc[0]} -> {df['time'].iloc[-1]}",
        "lot_min": rules["lot_min"],
        "lot_max": rules["lot_max"],
        "avg_lot": avg_lot,
        "min_lot_used": min_lot_used,
        "max_lot_used": max_lot_used,
    }


def print_results(r):
    status = "PASSED" if r["challenge_passed"] else ("FAILED" if r["challenge_failed"] else "IN PROGRESS")

    print("\n" + "=" * 60)
    print(f"  {r['name']}")
    print(f"  CHALLENGE STATUS: {status}")
    if r["challenge_failed"]:
        print(f"  FAIL REASON: {r['fail_reason']}")
    print("=" * 60)
    print(f"  Initial Balance:   ${r['initial_balance']:>12,.2f}")
    print(f"  Final Balance:     ${r['final_balance']:>12,.2f}")
    print(f"  Net Profit:        ${r['net_profit']:>12,.2f}")
    print(f"  Return:            {r['return_pct']:>11.2f}%")
    print(f"  Lot Range:          {r['lot_min']} - {r['lot_max']}")
    print(f"  Avg Lot Used:       {r['avg_lot']:.3f}")
    print(f"  Lot Range Used:     {r['min_lot_used']:.2f} - {r['max_lot_used']:.2f}")
    print("-" * 60)
    print(f"  FTMO RULES:")
    print(f"    Profit Target:   ${r['profit_target']:>12,.2f}  ({r['target_progress']:.1f}% achieved)")
    print(f"    Max Daily DD:    ${r['max_daily_dd_rule']:>12,.2f}")
    print(f"    Max Total DD:    ${r['max_total_dd_rule']:>12,.2f}")
    print(f"    Actual Max DD:   ${r['max_dd']:>12,.2f}  ({r['max_dd_pct']:.2f}%)")
    print(f"    Days Traded:      {r['days_traded']}")
    print("-" * 60)
    print(f"  TRADE STATS:")
    print(f"    Total Trades:     {r['total_trades']}")
    print(f"    Winners:          {r['win_count']}")
    print(f"    Losers:           {r['loss_count']}")
    print(f"    Win Rate:         {r['win_rate']:.1f}%")
    print(f"    Profit Factor:    {r['profit_factor']:.2f}")
    print(f"    Avg Win:         ${r['avg_win']:>12,.2f}")
    print(f"    Avg Loss:        ${r['avg_loss']:>12,.2f}")
    print(f"    Avg R:R:          {r['avg_rr']:.2f}x")
    print("-" * 60)
    print(f"  CLOSE REASONS:")
    print(f"    Stop Loss:        {r['sl_closes']}")
    print(f"    Take Profit:      {r['tp_closes']}")
    print(f"    Rolling SL:       {r['rolling_closes']}")
    print(f"    Open at End:      {r['end_closes']}")
    print("=" * 60)


def run_period(label, start_days_ago, end_days_ago, rule_keys):
    """Run a set of scenarios on a specific date range."""
    global OOS_START_DAYS_AGO, OOS_END_DAYS_AGO
    OOS_START_DAYS_AGO = start_days_ago
    OOS_END_DAYS_AGO = end_days_ago

    print("\n" + "#" * 60)
    print(f"  {label}")
    print(f"  Period: {start_days_ago}-{end_days_ago} days ago")
    print("#" * 60)

    results = {}
    for key in rule_keys:
        print(f"\n>>> Running {FTMO_RULES[key]['name']}...")
        r = run_ftmo_sim(key)
        if r:
            results[key] = r
            print_results(r)

    return results


def print_comparison(results, keys, col_labels):
    """Print side-by-side comparison for any set of results."""
    if len(results) < 2:
        return

    header = f"  {'':30s}"
    divider = f"  {'-'*30}"
    for lbl in col_labels:
        header += f" {lbl:>14s}"
        divider += f" {'-'*14}"

    print("\n" + "=" * (32 + 15 * len(col_labels)))
    print("  COMPARISON")
    print("=" * (32 + 15 * len(col_labels)))
    print(header)
    print(divider)

    rs = [results[k] for k in keys if k in results]

    rows = [
        ("Net Profit", lambda r: f"${r['net_profit']:>13,.2f}"),
        ("Return %", lambda r: f"{r['return_pct']:>13.2f}%"),
        ("Win Rate", lambda r: f"{r['win_rate']:>13.1f}%"),
        ("Profit Factor", lambda r: f"{r['profit_factor']:>14.2f}"),
        ("Avg R:R", lambda r: f"{r['avg_rr']:>13.2f}x"),
        ("Max DD %", lambda r: f"{r['max_dd_pct']:>13.2f}%"),
        ("Max DD $", lambda r: f"${r['max_dd']:>13,.2f}"),
        ("Total Trades", lambda r: f"{r['total_trades']:>14d}"),
        ("Target Progress", lambda r: f"{r['target_progress']:>13.1f}%"),
        ("Days Traded", lambda r: f"{r['days_traded']:>14d}"),
        ("Challenge", lambda r: f"{'PASSED' if r['challenge_passed'] else 'FAILED' if r['challenge_failed'] else 'IN PROGRESS':>14s}"),
    ]

    for label, fmt in rows:
        line = f"  {label:30s}"
        for r in rs:
            line += f" {fmt(r)}"
        print(line)

    print(divider)


def main():
    if not mt5.initialize():
        print("ERROR: MT5 failed to initialize")
        print("Make sure MetaTrader 5 is running!")
        return

    print("=" * 60)
    print("  FTMO CHALLENGE - DYNAMIC LOT SIZING")
    print("  BG_AggressiveCompetition v2.0 Logic")
    print("=" * 60)
    print(f"  SL: ATR x {SL_ATR_MULT} (hardcoded)")
    print(f"  TP: SL x {TP_MULTIPLIER}")
    print(f"  Dynamic TP: {DYN_TP_PERCENT}%")
    print(f"  Rolling SL: {ROLLING_SL_MULT}x")
    print(f"  $25K lots: 0.01 - 0.08 (dynamic)")
    print(f"  $100K lots: 0.03 - 0.24 (dynamic)")
    print(f"  Lot scales with: confidence (70%) + DD budget (30%)")
    print("=" * 60)

    all_results = {}

    # TEST 1: Out-of-sample (Nov/Dec) - Dynamic lots
    r1 = run_period("TEST 1: OUT-OF-SAMPLE (Nov/Dec) - Dynamic Lots",
                     90, 60, ["25K", "100K"])
    all_results.update({k + "_oos": v for k, v in r1.items()})

    # TEST 2: Recent 30 days - Dynamic lots (PROVE IT)
    r2 = run_period("TEST 2: RECENT 30 DAYS - Dynamic Lots (Prove It)",
                     30, 0, ["25K", "100K"])
    all_results.update({k + "_recent": v for k, v in r2.items()})

    mt5.shutdown()

    # === COMPARISON: Same dynamic lots, different months ===
    if "25K_oos" in all_results and "25K_recent" in all_results:
        print("\n" + "#" * 60)
        print("  PROOF: DYNAMIC LOTS, DIFFERENT MONTHS")
        print("#" * 60)
        print_comparison(
            {"oos": all_results["25K_oos"],
             "recent": all_results["25K_recent"]},
            ["oos", "recent"],
            ["$25K Nov/Dec", "$25K Recent"]
        )
        print_comparison(
            {"oos": all_results["100K_oos"],
             "recent": all_results["100K_recent"]},
            ["oos", "recent"],
            ["$100K Nov/Dec", "$100K Recent"]
        )

    # === COMPARISON: $25K vs $100K on recent data ===
    if "25K_recent" in all_results and "100K_recent" in all_results:
        print("\n" + "#" * 60)
        print("  SCALING: $25K vs $100K (Recent Period)")
        print("#" * 60)
        print_comparison(
            {"k25": all_results["25K_recent"],
             "k100": all_results["100K_recent"]},
            ["k25", "k100"],
            ["$25K Dynamic", "$100K Dynamic"]
        )

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
