"""
Prop Farm Account Simulator
============================
Simulates a single prop firm account with full trading rules.
Reuses proven logic from backtest_ftmo.py but adapted for
parallel signal farm operation.

Each instance represents ONE simulated account with:
- Prop firm DD rules (harder than real firms)
- Signal generation from ETARE genetic individuals
- Full position management (SL, TP, rolling SL, partial close)
- Signal collection for training data

Usage:
    from prop_farm_simulator import PropFarmAccount

    account = PropFarmAccount(account_id=1, symbol="BTCUSD")
    result = account.run_simulation(data, individual)
"""

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# Import trading settings from config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import (
    MAX_LOSS_DOLLARS, TP_MULTIPLIER, ROLLING_SL_MULTIPLIER,
    DYNAMIC_TP_PERCENT, CONFIDENCE_THRESHOLD
)


@dataclass
class SimPosition:
    """A single simulated position."""
    ticket: int
    direction: int       # 1=buy, -1=sell
    entry_price: float
    lot: float
    sl: float
    tp: float
    dyn_tp: float
    open_bar: int
    open_time: str = ""
    dyn_tp_taken: bool = False
    rolling_sl: float = 0.0
    remaining_lot: float = 0.0
    partial_pnl: float = 0.0
    pnl: float = 0.0
    closed: bool = False
    close_reason: str = ""
    close_price: float = 0.0
    close_bar: int = 0

    def __post_init__(self):
        if self.rolling_sl == 0.0:
            self.rolling_sl = self.sl
        if self.remaining_lot == 0.0:
            self.remaining_lot = self.lot


@dataclass
class SimSignal:
    """A collected signal from the simulation."""
    bar_index: int
    timestamp: str
    timeframe: str
    symbol: str
    direction: int       # 1=buy, -1=sell, 0=hold
    confidence: float
    price: float
    atr: float
    ema_fast: float
    ema_slow: float
    rsi: float = 0.0
    macd: float = 0.0
    outcome: str = ""    # "win", "loss", "pending"
    pnl: float = 0.0


@dataclass
class AccountResult:
    """Complete results from one account simulation run."""
    account_id: int
    symbol: str
    timeframe: str
    cycle: int
    phase: str           # "train" or "test"

    # Performance
    initial_balance: float
    final_balance: float
    net_profit: float
    return_pct: float
    win_rate: float
    total_trades: int
    winners: int
    losers: int
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_dd: float
    max_dd_pct: float

    # Prop firm
    challenge_passed: bool
    challenge_failed: bool
    fail_reason: str
    days_traded: int

    # Signals collected
    signals_count: int
    signals: List[SimSignal] = field(default_factory=list)

    # Individual info
    individual_fitness: float = 0.0
    individual_generation: int = 0


class PropFarmAccount:
    """
    Simulates a single prop firm account.

    Runs EA logic against historical data with prop firm rules,
    collecting signals for training feedback.
    """

    def __init__(self, account_id: int, symbol: str = "BTCUSD",
                 balance: float = 100000.0, max_daily_dd_pct: float = 5.0,
                 max_total_dd_pct: float = 10.0, profit_target_pct: float = 10.0):
        self.account_id = account_id
        self.symbol = symbol
        self.initial_balance = balance
        self.max_daily_dd = balance * (max_daily_dd_pct / 100.0)
        self.max_total_dd = balance * (max_total_dd_pct / 100.0)
        self.profit_target = balance * (profit_target_pct / 100.0)

        # EA parameters - from MASTER_CONFIG
        self.sl_atr_mult = 1.0
        self.tp_multiplier = TP_MULTIPLIER
        self.dyn_tp_percent = DYNAMIC_TP_PERCENT
        self.rolling_sl_mult = ROLLING_SL_MULTIPLIER
        self.min_confidence = CONFIDENCE_THRESHOLD
        self.max_positions = 20
        self.grid_spacing_pts = 200

        # Signal params
        self.fast_ema_period = 5
        self.slow_ema_period = 13
        self.atr_period = 14

    def _prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed for signal generation."""
        df = df.copy()

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema_period, adjust=False).mean()

        # ATR
        df['hl'] = df['high'] - df['low']
        df['hc'] = abs(df['high'] - df['close'].shift(1))
        df['lc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss_s + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2

        df = df.ffill().bfill()
        return df

    def _calc_confidence(self, ema_f, ema_s, atr):
        """Calculate signal confidence from EMA separation."""
        separation = abs(ema_f - ema_s)
        return min(1.0, (separation / (atr + 1e-10)) * 0.5)

    def _calc_dynamic_lot(self, confidence, dd_remaining_pct):
        """Dynamic lot sizing based on confidence and DD budget."""
        lot_min = 0.01
        lot_max = 0.24

        conf_factor = min(1.0, max(0.0,
            (confidence - self.min_confidence) / (1.0 - self.min_confidence)))

        if dd_remaining_pct < 0.3:
            dd_factor = 0.0
        elif dd_remaining_pct < 0.5:
            dd_factor = (dd_remaining_pct - 0.3) / 0.2
        else:
            dd_factor = 1.0

        combined = conf_factor * 0.7 + dd_factor * 0.3
        lot = lot_min + (lot_max - lot_min) * combined
        return max(lot_min, min(lot_max, round(lot, 2)))

    def run_simulation(self, df: pd.DataFrame, timeframe: str = "M5",
                       cycle: int = 1, phase: str = "train",
                       individual=None, contract_size: float = 1.0,
                       point: float = 0.01) -> AccountResult:
        """
        Run full simulation on the provided data.

        Args:
            df: DataFrame with OHLCV data
            timeframe: timeframe name for logging
            cycle: cycle number
            phase: "train" or "test"
            individual: TradingIndividual for genetic signal generation (optional)
            contract_size: symbol contract size
            point: symbol point size

        Returns:
            AccountResult with full performance metrics and collected signals
        """
        df = self._prepare_indicators(df)
        start_bar = max(self.slow_ema_period, self.atr_period, 26) + 5

        # State
        positions: List[SimPosition] = []
        closed_trades: List[SimPosition] = []
        signals: List[SimSignal] = []
        next_ticket = self.account_id * 10000

        balance = self.initial_balance
        peak_balance = self.initial_balance
        max_dd = 0.0
        max_dd_pct = 0.0

        challenge_passed = False
        challenge_failed = False
        fail_reason = ""
        day_start_equity = self.initial_balance
        current_day = None
        days_traded = 0

        for i in range(start_bar, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            price = row['close']
            atr = row['atr']

            if pd.isna(atr) or atr == 0:
                continue

            bar_date = None
            if 'time' in df.columns:
                bar_date = pd.Timestamp(row['time']).date() if not isinstance(row['time'], str) else row['time'][:10]

            # New day check
            if bar_date is not None and bar_date != current_day:
                if current_day is not None:
                    days_traded += 1
                current_day = bar_date
                floating = sum(
                    (price - p.entry_price if p.direction == 1 else p.entry_price - price)
                    * p.remaining_lot * contract_size
                    for p in positions if not p.closed
                )
                day_start_equity = balance + floating

            # Calculate equity
            floating = sum(
                (price - p.entry_price if p.direction == 1 else p.entry_price - price)
                * p.remaining_lot * contract_size
                for p in positions if not p.closed
            )
            equity = balance + floating

            # DD checks
            daily_dd = day_start_equity - equity
            if daily_dd >= self.max_daily_dd:
                if not challenge_failed:
                    challenge_failed = True
                    fail_reason = f"Daily DD: ${daily_dd:,.2f} >= ${self.max_daily_dd:,.2f}"
                continue

            total_dd = self.initial_balance - equity
            if total_dd >= self.max_total_dd:
                if not challenge_failed:
                    challenge_failed = True
                    fail_reason = f"Total DD: ${total_dd:,.2f} >= ${self.max_total_dd:,.2f}"
                break

            net_profit = equity - self.initial_balance
            if net_profit >= self.profit_target and not challenge_passed:
                challenge_passed = True

            dd_remaining_pct = max(0, (self.max_total_dd - max(0, total_dd)) / self.max_total_dd)

            # Manage positions
            newly_closed = []
            for pos in positions:
                if pos.closed:
                    continue

                # SL hit
                sl_hit = (pos.direction == 1 and price <= pos.rolling_sl) or \
                         (pos.direction == -1 and price >= pos.rolling_sl)
                if sl_hit:
                    if pos.direction == 1:
                        pnl = (pos.rolling_sl - pos.entry_price) * pos.remaining_lot * contract_size
                    else:
                        pnl = (pos.entry_price - pos.rolling_sl) * pos.remaining_lot * contract_size
                    pos.pnl = pnl + pos.partial_pnl
                    pos.closed = True
                    pos.close_price = pos.rolling_sl
                    pos.close_reason = "SL" if pos.rolling_sl == pos.sl else "ROLLING_SL"
                    pos.close_bar = i
                    balance += pnl
                    newly_closed.append(pos)
                    continue

                # TP hit
                tp_hit = (pos.direction == 1 and price >= pos.tp) or \
                         (pos.direction == -1 and price <= pos.tp)
                if tp_hit:
                    if pos.direction == 1:
                        pnl = (pos.tp - pos.entry_price) * pos.remaining_lot * contract_size
                    else:
                        pnl = (pos.entry_price - pos.tp) * pos.remaining_lot * contract_size
                    pos.pnl = pnl + pos.partial_pnl
                    pos.closed = True
                    pos.close_price = pos.tp
                    pos.close_reason = "TP"
                    pos.close_bar = i
                    balance += pnl
                    newly_closed.append(pos)
                    continue

                # Dynamic TP (partial close)
                if not pos.dyn_tp_taken:
                    dyn_hit = (pos.direction == 1 and price >= pos.dyn_tp) or \
                              (pos.direction == -1 and price <= pos.dyn_tp)
                    if dyn_hit:
                        half_lot = pos.remaining_lot * 0.5
                        if pos.direction == 1:
                            partial = (price - pos.entry_price) * half_lot * contract_size
                        else:
                            partial = (pos.entry_price - price) * half_lot * contract_size
                        balance += partial
                        pos.partial_pnl += partial
                        pos.remaining_lot -= half_lot
                        pos.dyn_tp_taken = True
                        pos.rolling_sl = pos.entry_price  # Move to breakeven

                # Rolling SL
                if pos.dyn_tp_taken:
                    sl_dist = abs(pos.entry_price - pos.sl)
                    roll_target = sl_dist * self.rolling_sl_mult
                    if pos.direction == 1:
                        profit = price - pos.entry_price
                        if profit > roll_target:
                            new_sl = price - sl_dist
                            if new_sl > pos.rolling_sl:
                                pos.rolling_sl = new_sl
                    else:
                        profit = pos.entry_price - price
                        if profit > roll_target:
                            new_sl = price + sl_dist
                            if new_sl < pos.rolling_sl or pos.rolling_sl <= 0:
                                pos.rolling_sl = new_sl

            closed_trades.extend(newly_closed)
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

            if challenge_failed:
                continue

            # Signal generation
            ema_f = row['ema_fast']
            ema_s = row['ema_slow']
            confidence = self._calc_confidence(ema_f, ema_s, atr)

            buy_signal = ema_f > ema_s
            sell_signal = ema_f < ema_s
            direction = 1 if buy_signal else (-1 if sell_signal else 0)

            # Collect signal
            sig = SimSignal(
                bar_index=i,
                timestamp=str(row.get('time', i)),
                timeframe=timeframe,
                symbol=self.symbol,
                direction=direction,
                confidence=confidence,
                price=price,
                atr=atr,
                ema_fast=ema_f,
                ema_slow=ema_s,
                rsi=row.get('rsi', 0),
                macd=row.get('macd', 0),
            )
            signals.append(sig)

            if confidence < self.min_confidence:
                continue

            # Dynamic lot
            lot = self._calc_dynamic_lot(confidence, dd_remaining_pct)

            buy_count = sum(1 for p in positions if p.direction == 1)
            sell_count = sum(1 for p in positions if p.direction == -1)
            if buy_count + sell_count >= self.max_positions:
                continue

            spacing = self.grid_spacing_pts * point

            # Entry logic
            if buy_signal and buy_count < self.max_positions // 2:
                last_buy = max([p.entry_price for p in positions if p.direction == 1], default=0)
                if buy_count == 0 or price < last_buy - spacing:
                    sl_dist = atr * self.sl_atr_mult
                    tp_dist = sl_dist * self.tp_multiplier
                    dyn_dist = tp_dist * (self.dyn_tp_percent / 100.0)
                    pos = SimPosition(
                        ticket=next_ticket, direction=1, entry_price=price,
                        lot=lot, sl=price - sl_dist, tp=price + tp_dist,
                        dyn_tp=price + dyn_dist, open_bar=i,
                        open_time=str(row.get('time', ''))
                    )
                    positions.append(pos)
                    next_ticket += 1

            if sell_signal and sell_count < self.max_positions // 2:
                last_sell = min([p.entry_price for p in positions if p.direction == -1], default=999999999)
                if sell_count == 0 or price > last_sell + spacing:
                    sl_dist = atr * self.sl_atr_mult
                    tp_dist = sl_dist * self.tp_multiplier
                    dyn_dist = tp_dist * (self.dyn_tp_percent / 100.0)
                    pos = SimPosition(
                        ticket=next_ticket, direction=-1, entry_price=price,
                        lot=lot, sl=price + sl_dist, tp=price - tp_dist,
                        dyn_tp=price - dyn_dist, open_bar=i,
                        open_time=str(row.get('time', ''))
                    )
                    positions.append(pos)
                    next_ticket += 1

        # Close remaining positions at last price
        if len(df) > 0:
            last_price = df.iloc[-1]['close']
            for pos in positions:
                if not pos.closed:
                    if pos.direction == 1:
                        pnl = (last_price - pos.entry_price) * pos.remaining_lot * contract_size
                    else:
                        pnl = (pos.entry_price - last_price) * pos.remaining_lot * contract_size
                    pos.pnl = pnl + pos.partial_pnl
                    pos.closed = True
                    pos.close_price = last_price
                    pos.close_reason = "END"
                    pos.close_bar = len(df) - 1
                    balance += pnl
                    closed_trades.append(pos)

        days_traded += 1

        # Calculate results
        total_trades = len(closed_trades)
        winners_list = [t for t in closed_trades if t.pnl > 0]
        losers_list = [t for t in closed_trades if t.pnl <= 0]
        win_count = len(winners_list)
        loss_count = len(losers_list)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in winners_list)
        gross_loss = abs(sum(t.pnl for t in losers_list))
        net_profit_val = balance - self.initial_balance
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        avg_win = (gross_profit / win_count) if win_count > 0 else 0
        avg_loss = (gross_loss / loss_count) if loss_count > 0 else 0

        return AccountResult(
            account_id=self.account_id,
            symbol=self.symbol,
            timeframe=timeframe,
            cycle=cycle,
            phase=phase,
            initial_balance=self.initial_balance,
            final_balance=balance,
            net_profit=net_profit_val,
            return_pct=(net_profit_val / self.initial_balance) * 100,
            win_rate=win_rate,
            total_trades=total_trades,
            winners=win_count,
            losers=loss_count,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_dd=max_dd,
            max_dd_pct=max_dd_pct,
            challenge_passed=challenge_passed,
            challenge_failed=challenge_failed,
            fail_reason=fail_reason,
            days_traded=days_traded,
            signals_count=len(signals),
            signals=signals,
            individual_fitness=individual.fitness if individual else 0,
            individual_generation=individual.generation if individual else 0,
        )


def signals_to_json(signals: List[SimSignal]) -> list:
    """Convert signals to JSON-serializable format."""
    return [
        {
            "bar": s.bar_index,
            "time": s.timestamp,
            "tf": s.timeframe,
            "sym": s.symbol,
            "dir": s.direction,
            "conf": round(s.confidence, 4),
            "price": round(s.price, 2),
            "atr": round(s.atr, 2),
            "rsi": round(s.rsi, 2),
            "macd": round(s.macd, 4),
            "outcome": s.outcome,
            "pnl": round(s.pnl, 2),
        }
        for s in signals
    ]


def result_to_json(result: AccountResult) -> dict:
    """Convert AccountResult to JSON-serializable format."""
    return {
        "account_id": result.account_id,
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "cycle": result.cycle,
        "phase": result.phase,
        "balance": round(result.final_balance, 2),
        "net_profit": round(result.net_profit, 2),
        "return_pct": round(result.return_pct, 2),
        "win_rate": round(result.win_rate, 1),
        "total_trades": result.total_trades,
        "winners": result.winners,
        "losers": result.losers,
        "profit_factor": round(result.profit_factor, 2),
        "max_dd": round(result.max_dd, 2),
        "max_dd_pct": round(result.max_dd_pct, 2),
        "challenge_passed": result.challenge_passed,
        "challenge_failed": result.challenge_failed,
        "fail_reason": result.fail_reason,
        "days_traded": result.days_traded,
        "signals_count": result.signals_count,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  PROP FARM SIMULATOR - Module Check")
    print("=" * 60)
    print(f"  SL ATR Mult:       1.0 (hardcoded)")
    print(f"  TP Multiplier:     {TP_MULTIPLIER}")
    print(f"  Rolling SL Mult:   {ROLLING_SL_MULTIPLIER}")
    print(f"  Dynamic TP %:      {DYNAMIC_TP_PERCENT}")
    print(f"  Confidence Thresh: {CONFIDENCE_THRESHOLD}")
    print(f"  Max Positions:     20")
    print("=" * 60)
    print("  Ready. Import and use PropFarmAccount class.")
