"""
VIRTUAL POSITION MANAGER - Grid Position Lifecycle
====================================================
Manages virtual (simulated) grid positions with:
  - Hidden SL/TP (never sent to broker)
  - Partial TP (close configurable % at partial distance)
  - Breakeven (move SL to entry when profit hits trigger)
  - Trailing stop (trail at ATR distance)

Faithfully ports the position lifecycle from EntropyGridCore.mqh
ManagePositions(), OpenGridPosition(), and ClosePosition().
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from signal_farm_config import FarmParams, SymbolSpec
from entropy_confidence import ConfidenceResult, EntropyState

logger = logging.getLogger("signal_farm")


@dataclass
class VirtualPosition:
    """One virtual grid position (no real broker order)."""
    ticket: int                  # Unique ID
    symbol: str
    direction: str               # "BUY" or "SELL"
    entry_price: float
    volume: float                # Current remaining volume
    original_volume: float       # Volume at open
    open_time: float             # Unix timestamp
    grid_level: int              # Grid level index

    # Hidden levels (virtual, not sent to broker)
    virtual_sl: float
    virtual_tp: float
    partial_tp: float            # Price level for partial close

    # Lifecycle flags
    partial_closed: bool = False
    breakeven_set: bool = False
    trailing_active: bool = False

    # Partial volume (amount to close at partial TP)
    partial_volume: float = 0.0

    # ATR at open (for trailing calculations)
    atr_at_open: float = 0.0

    # PnL tracking
    realized_pnl: float = 0.0
    close_reason: str = ""


_next_ticket = 100000


def _get_ticket() -> int:
    global _next_ticket
    _next_ticket += 1
    return _next_ticket


class VirtualPositionManager:
    """Manages all virtual positions for one account + one symbol."""

    def __init__(self, params: FarmParams, symbol_spec: SymbolSpec):
        self.params = params
        self.spec = symbol_spec
        self.positions: List[VirtualPosition] = []
        self._closed_positions: List[VirtualPosition] = []

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def closed_positions(self) -> List[VirtualPosition]:
        return self._closed_positions

    def clear_closed_history(self):
        self._closed_positions.clear()

    def drain_closed(self) -> List[VirtualPosition]:
        """Return and clear closed positions (for reporting)."""
        result = list(self._closed_positions)
        self._closed_positions.clear()
        return result

    def can_open(self) -> bool:
        """Check if we can open another position."""
        return self.position_count < self.params.max_positions_per_symbol

    def check_grid_spacing(self, current_price: float, direction: str, atr: float) -> bool:
        """Check if price has moved enough from last entry for grid spacing.

        For buys: price should have dipped (moved down) from last buy entry
        For sells: price should have rallied (moved up) from last sell entry
        """
        if not self.positions:
            return True

        min_spacing = atr * self.params.grid_spacing_atr
        if min_spacing <= 0:
            return True

        # Find last entry in same direction
        same_dir = [p for p in self.positions if p.direction == direction]
        if not same_dir:
            return True

        last_entry = same_dir[-1].entry_price

        if direction == "BUY":
            # Price should have dipped below last entry minus spacing
            return current_price <= last_entry - min_spacing
        else:
            # Price should have rallied above last entry plus spacing
            return current_price >= last_entry + min_spacing

    def open_position(
        self,
        direction: str,
        price: float,
        atr: float,
        entropy_state: EntropyState,
        balance: float,
    ) -> Optional[VirtualPosition]:
        """
        Open a new virtual grid position.
        Ports EntropyGridCore.mqh OpenGridPosition().

        Returns the new position or None if blocked.
        """
        if not self.can_open():
            return None

        if entropy_state == EntropyState.HIGH:
            return None

        # SL/TP distances from ATR
        sl_distance = atr * self.params.sl_atr_multiplier
        tp_distance = atr * self.params.tp_atr_multiplier

        if sl_distance <= 0 or tp_distance <= 0:
            return None

        # Lot sizing: risk amount / (sl_distance / tick_size * tick_value)
        if self.spec.tick_size > 0 and self.spec.tick_value > 0:
            sl_ticks = sl_distance / self.spec.tick_size
            lot = self.params.max_loss_dollars / (sl_ticks * self.spec.tick_value)
        else:
            # Fallback: use dollar_per_point
            if self.spec.dollar_per_point > 0:
                lot = self.params.max_loss_dollars / (sl_distance * self.spec.dollar_per_point)
            else:
                lot = self.params.base_lot_size

        # ENTROPY_MEDIUM halves the lot
        if entropy_state == EntropyState.MEDIUM:
            lot *= 0.5

        # Clamp lot
        lot = max(self.params.base_lot_size, lot)
        lot = min(lot, self.params.max_lot_size)
        # Round to step
        if self.spec.volume_step > 0:
            lot = round(lot / self.spec.volume_step) * self.spec.volume_step
        lot = max(self.params.base_lot_size, lot)

        # Partial TP distance = partial_tp_ratio of full TP distance
        partial_tp_distance = tp_distance * self.params.partial_tp_ratio

        # Calculate levels
        if direction == "BUY":
            sl = price - sl_distance
            tp = price + tp_distance
            partial_tp = price + partial_tp_distance
        else:
            sl = price + sl_distance
            tp = price - tp_distance
            partial_tp = price - partial_tp_distance

        pos = VirtualPosition(
            ticket=_get_ticket(),
            symbol=self.spec.name,
            direction=direction,
            entry_price=price,
            volume=lot,
            original_volume=lot,
            open_time=time.time(),
            grid_level=self.position_count,
            virtual_sl=sl,
            virtual_tp=tp,
            partial_tp=partial_tp,
            partial_volume=round(lot * 0.5 / self.spec.volume_step) * self.spec.volume_step if self.spec.volume_step > 0 else round(lot * 0.5, 2),
            atr_at_open=atr,
        )
        # Ensure partial volume is at least min lot
        pos.partial_volume = max(pos.partial_volume, self.spec.volume_min)
        # Don't let partial exceed current volume
        if pos.partial_volume >= pos.volume:
            pos.partial_volume = round(pos.volume * 0.5 / self.spec.volume_step) * self.spec.volume_step if self.spec.volume_step > 0 else round(pos.volume * 0.5, 2)
            pos.partial_volume = max(pos.partial_volume, self.spec.volume_min)

        self.positions.append(pos)
        return pos

    def manage_positions(self, bar_high: float, bar_low: float, current_price: float, atr: float) -> List[Tuple[VirtualPosition, float, str]]:
        """
        Run full position lifecycle management on every new bar.
        Ports EntropyGridCore.mqh ManagePositions().

        Checks in order:
          1. Virtual SL/TP (using bar high/low for within-bar detection)
          2. Partial TP
          3. Breakeven
          4. Trailing stop

        Returns: list of (position, pnl, reason) for positions that closed this bar.
        """
        closed_this_bar: List[Tuple[VirtualPosition, float, str]] = []

        # Iterate in reverse so we can remove safely
        for i in range(len(self.positions) - 1, -1, -1):
            pos = self.positions[i]

            # --- 1. Virtual SL/TP Check ---
            sl_hit = False
            tp_hit = False

            if pos.direction == "BUY":
                if bar_low <= pos.virtual_sl:
                    sl_hit = True
                elif bar_high >= pos.virtual_tp:
                    tp_hit = True
            else:  # SELL
                if bar_high >= pos.virtual_sl:
                    sl_hit = True
                elif bar_low <= pos.virtual_tp:
                    tp_hit = True

            if sl_hit:
                pnl = self._calc_pnl(pos, pos.virtual_sl)
                pos.realized_pnl = pnl
                pos.close_reason = "SL"
                closed_this_bar.append((pos, pnl, "SL"))
                self._closed_positions.append(self.positions.pop(i))
                continue

            if tp_hit:
                pnl = self._calc_pnl(pos, pos.virtual_tp)
                pos.realized_pnl = pnl
                pos.close_reason = "TP"
                closed_this_bar.append((pos, pnl, "TP"))
                self._closed_positions.append(self.positions.pop(i))
                continue

            # --- 2. Partial Take Profit ---
            if not pos.partial_closed:
                partial_hit = False
                if pos.direction == "BUY" and bar_high >= pos.partial_tp:
                    partial_hit = True
                elif pos.direction == "SELL" and bar_low <= pos.partial_tp:
                    partial_hit = True

                if partial_hit:
                    # Close partial_volume worth
                    close_vol = pos.partial_volume
                    if close_vol >= pos.volume:
                        close_vol = max(
                            round(pos.volume * 0.5 / self.spec.volume_step) * self.spec.volume_step if self.spec.volume_step > 0 else round(pos.volume * 0.5, 2),
                            self.spec.volume_min,
                        )

                    partial_pnl = self._calc_pnl_at_volume(pos, pos.partial_tp, close_vol)
                    pos.volume -= close_vol
                    pos.volume = max(pos.volume, self.spec.volume_min)
                    pos.partial_closed = True
                    pos.realized_pnl += partial_pnl
                    # Report partial close but don't remove position
                    closed_this_bar.append((pos, partial_pnl, "PARTIAL_TP"))

            # --- 3. Break Even ---
            if not pos.breakeven_set:
                tp_distance = abs(pos.virtual_tp - pos.entry_price)
                trigger_distance = tp_distance * self.params.breakeven_trigger

                if pos.direction == "BUY":
                    profit_progress = current_price - pos.entry_price
                else:
                    profit_progress = pos.entry_price - current_price

                if profit_progress >= trigger_distance:
                    # Move SL to entry + small buffer (10 points)
                    buffer = 10 * self.spec.point if self.spec.point > 0 else 0.1
                    if pos.direction == "BUY":
                        new_sl = pos.entry_price + buffer
                        if new_sl > pos.virtual_sl:
                            pos.virtual_sl = new_sl
                    else:
                        new_sl = pos.entry_price - buffer
                        if new_sl < pos.virtual_sl:
                            pos.virtual_sl = new_sl
                    pos.breakeven_set = True

            # --- 4. Trailing Stop ---
            if pos.breakeven_set:
                tp_distance = abs(pos.virtual_tp - pos.entry_price)
                trail_trigger_distance = tp_distance * self.params.trail_start_trigger

                if pos.direction == "BUY":
                    profit_progress = current_price - pos.entry_price
                else:
                    profit_progress = pos.entry_price - current_price

                if profit_progress >= trail_trigger_distance:
                    trail_distance = atr * self.params.trail_distance_atr
                    pos.trailing_active = True

                    if pos.direction == "BUY":
                        new_sl = current_price - trail_distance
                        if new_sl > pos.virtual_sl:
                            pos.virtual_sl = new_sl
                    else:
                        new_sl = current_price + trail_distance
                        if new_sl < pos.virtual_sl:
                            pos.virtual_sl = new_sl

        return closed_this_bar

    def close_all(self, current_price: float) -> List[Tuple[VirtualPosition, float, str]]:
        """Force-close all positions at current price (for challenge reset)."""
        closed = []
        for pos in self.positions:
            pnl = self._calc_pnl(pos, current_price)
            pos.realized_pnl += pnl
            pos.close_reason = "FORCE_CLOSE"
            closed.append((pos, pnl, "FORCE_CLOSE"))
            self._closed_positions.append(pos)
        self.positions.clear()
        return closed

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate total unrealized PnL across all open positions."""
        total = 0.0
        for pos in self.positions:
            total += self._calc_pnl(pos, current_price)
        return total

    def _calc_pnl(self, pos: VirtualPosition, exit_price: float) -> float:
        """Calculate PnL for full remaining volume at given exit price."""
        return self._calc_pnl_at_volume(pos, exit_price, pos.volume)

    def _calc_pnl_at_volume(self, pos: VirtualPosition, exit_price: float, volume: float) -> float:
        """Calculate PnL for a specific volume at given exit price."""
        if pos.direction == "BUY":
            price_diff = exit_price - pos.entry_price
        else:
            price_diff = pos.entry_price - exit_price

        # PnL = price_diff * volume * dollar_per_point
        if self.spec.dollar_per_point > 0:
            return price_diff * volume * self.spec.dollar_per_point
        elif self.spec.tick_size > 0 and self.spec.tick_value > 0:
            ticks = price_diff / self.spec.tick_size
            return ticks * self.spec.tick_value * volume
        else:
            return 0.0
