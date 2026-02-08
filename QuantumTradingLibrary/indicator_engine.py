"""
INDICATOR ENGINE - Rolling Indicator Calculations
===================================================
Maintains rolling buffers of ATR(14), EMA(8/21/200), RSI(14)
per symbol. Fed by M5 OHLCV bars from MT5.

Faithfully replicates the indicator set from EntropyGridCore.mqh.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class BarData:
    """Single M5 OHLCV bar."""
    time: int       # Unix timestamp
    open: float
    high: float
    low: float
    close: float
    tick_volume: float = 0.0


@dataclass
class IndicatorSnapshot:
    """Current indicator values for one symbol at one point in time."""
    atr_14: float = 0.0
    ema_8: float = 0.0
    ema_21: float = 0.0
    ema_200: float = 0.0
    rsi_14: float = 50.0
    # Previous bar values (for change detection)
    prev_atr_14: float = 0.0
    prev_ema_8: float = 0.0
    prev_ema_21: float = 0.0
    prev_ema_200: float = 0.0
    prev_rsi_14: float = 50.0
    # Raw price
    price: float = 0.0
    bar_high: float = 0.0
    bar_low: float = 0.0
    ready: bool = False


class SymbolIndicators:
    """Rolling indicator engine for a single symbol."""

    def __init__(self, buffer_size: int = 250):
        self.buffer_size = buffer_size
        self.bars_loaded = 0

        # Raw price arrays (ring buffer)
        self._high = np.zeros(buffer_size)
        self._low = np.zeros(buffer_size)
        self._close = np.zeros(buffer_size)

        # EMA state (exponential, no buffer needed)
        self._ema_8 = 0.0
        self._ema_21 = 0.0
        self._ema_200 = 0.0
        self._ema_8_prev = 0.0
        self._ema_21_prev = 0.0
        self._ema_200_prev = 0.0

        # EMA multipliers
        self._ema_8_k = 2.0 / (8 + 1)
        self._ema_21_k = 2.0 / (21 + 1)
        self._ema_200_k = 2.0 / (200 + 1)

        # RSI state
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi = 50.0
        self._rsi_prev = 50.0
        self._prev_close = 0.0

        # ATR state (Wilder's smoothing)
        self._atr = 0.0
        self._atr_prev = 0.0
        self._atr_initialized = False

        # EMA separation tracking
        self._prev_ema_sep = 0.0

        # Write index into ring buffer
        self._idx = 0
        self._last_bar_time = 0

    def _store_bar(self, bar: BarData):
        """Store bar data in ring buffer."""
        idx = self._idx % self.buffer_size
        self._high[idx] = bar.high
        self._low[idx] = bar.low
        self._close[idx] = bar.close
        self._idx += 1

    def load_history(self, bars: list):
        """Initialize indicators from historical bars (list of BarData or tuples).

        Bars must be in chronological order (oldest first).
        """
        for bar in bars:
            if isinstance(bar, BarData):
                self._process_bar(bar)
            else:
                # Assume numpy structured array row or tuple
                self._process_bar(BarData(
                    time=int(bar[0]) if hasattr(bar, '__getitem__') else int(bar.time),
                    open=float(bar[1]) if hasattr(bar, '__getitem__') else float(bar.open),
                    high=float(bar[2]) if hasattr(bar, '__getitem__') else float(bar.high),
                    low=float(bar[3]) if hasattr(bar, '__getitem__') else float(bar.low),
                    close=float(bar[4]) if hasattr(bar, '__getitem__') else float(bar.close),
                    tick_volume=float(bar[5]) if hasattr(bar, '__getitem__') else float(getattr(bar, 'tick_volume', 0)),
                ))

    def _process_bar(self, bar: BarData):
        """Process a single new bar, updating all indicators."""
        close = bar.close
        high = bar.high
        low = bar.low

        self._store_bar(bar)
        self.bars_loaded += 1

        # --- EMA updates ---
        if self.bars_loaded == 1:
            self._ema_8 = close
            self._ema_21 = close
            self._ema_200 = close
        else:
            self._ema_8_prev = self._ema_8
            self._ema_21_prev = self._ema_21
            self._ema_200_prev = self._ema_200

            self._ema_8 = close * self._ema_8_k + self._ema_8 * (1 - self._ema_8_k)
            self._ema_21 = close * self._ema_21_k + self._ema_21 * (1 - self._ema_21_k)
            self._ema_200 = close * self._ema_200_k + self._ema_200 * (1 - self._ema_200_k)

        # --- RSI update (Wilder's smoothing) ---
        if self.bars_loaded == 1:
            self._prev_close = close
        else:
            delta = close - self._prev_close
            gain = max(delta, 0)
            loss = max(-delta, 0)

            self._rsi_prev = self._rsi

            if self.bars_loaded <= 14:
                # Accumulation phase
                self._rsi_avg_gain += gain
                self._rsi_avg_loss += loss
                if self.bars_loaded == 14:
                    self._rsi_avg_gain /= 14
                    self._rsi_avg_loss /= 14
                    if self._rsi_avg_loss == 0:
                        self._rsi = 100.0
                    else:
                        rs = self._rsi_avg_gain / self._rsi_avg_loss
                        self._rsi = 100 - (100 / (1 + rs))
            else:
                # Wilder's smoothing
                self._rsi_avg_gain = (self._rsi_avg_gain * 13 + gain) / 14
                self._rsi_avg_loss = (self._rsi_avg_loss * 13 + loss) / 14
                if self._rsi_avg_loss == 0:
                    self._rsi = 100.0
                else:
                    rs = self._rsi_avg_gain / self._rsi_avg_loss
                    self._rsi = 100 - (100 / (1 + rs))

            self._prev_close = close

        # --- ATR update (Wilder's smoothing, period 14) ---
        if self.bars_loaded == 1:
            tr = high - low
            self._atr = tr
        else:
            prev_idx = (self._idx - 2) % self.buffer_size
            prev_close = self._close[prev_idx]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )

            self._atr_prev = self._atr

            if self.bars_loaded <= 14:
                # Simple average for first 14 bars
                if self.bars_loaded == 14:
                    # Recalculate from buffer
                    n = min(self.bars_loaded, self.buffer_size)
                    trs = []
                    for i in range(1, n):
                        ci = (self._idx - n + i) % self.buffer_size
                        pi = (self._idx - n + i - 1) % self.buffer_size
                        h, l, pc = self._high[ci], self._low[ci], self._close[pi]
                        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
                    trs.insert(0, self._high[(self._idx - n) % self.buffer_size] - self._low[(self._idx - n) % self.buffer_size])
                    self._atr = sum(trs) / len(trs)
                    self._atr_initialized = True
                else:
                    self._atr = tr  # Placeholder
            else:
                # Wilder's smoothing
                self._atr = (self._atr * 13 + tr) / 14

        self._last_bar_time = bar.time

    def update(self, bar: BarData) -> bool:
        """Process a new bar. Returns True if this was a new bar (not duplicate)."""
        if bar.time <= self._last_bar_time:
            return False
        self._process_bar(bar)
        return True

    def get_snapshot(self) -> IndicatorSnapshot:
        """Get current indicator values."""
        return IndicatorSnapshot(
            atr_14=self._atr,
            ema_8=self._ema_8,
            ema_21=self._ema_21,
            ema_200=self._ema_200,
            rsi_14=self._rsi,
            prev_atr_14=self._atr_prev,
            prev_ema_8=self._ema_8_prev,
            prev_ema_21=self._ema_21_prev,
            prev_ema_200=self._ema_200_prev,
            prev_rsi_14=self._rsi_prev,
            price=self._close[(self._idx - 1) % self.buffer_size] if self.bars_loaded > 0 else 0.0,
            bar_high=self._high[(self._idx - 1) % self.buffer_size] if self.bars_loaded > 0 else 0.0,
            bar_low=self._low[(self._idx - 1) % self.buffer_size] if self.bars_loaded > 0 else 0.0,
            ready=self.bars_loaded >= 200,
        )

    def get_ema_separation(self) -> float:
        """Get EMA8-EMA21 separation as a fraction of price."""
        if self._ema_21 == 0:
            return 0.0
        return abs(self._ema_8 - self._ema_21) / self._ema_21

    def get_prev_ema_separation(self) -> float:
        """Get previous bar's EMA separation."""
        if self._ema_21_prev == 0:
            return 0.0
        return abs(self._ema_8_prev - self._ema_21_prev) / self._ema_21_prev


class IndicatorEngine:
    """Manages indicator engines for multiple symbols."""

    def __init__(self, symbols: list, buffer_size: int = 250):
        self.engines: Dict[str, SymbolIndicators] = {
            sym: SymbolIndicators(buffer_size) for sym in symbols
        }

    def load_history(self, symbol: str, bars: list):
        """Load historical bars for a symbol."""
        if symbol in self.engines:
            self.engines[symbol].load_history(bars)

    def update(self, symbol: str, bar: BarData) -> bool:
        """Update indicators with a new bar. Returns True if new bar processed."""
        if symbol in self.engines:
            return self.engines[symbol].update(bar)
        return False

    def get_snapshot(self, symbol: str) -> Optional[IndicatorSnapshot]:
        """Get current indicator snapshot for a symbol."""
        if symbol in self.engines:
            return self.engines[symbol].get_snapshot()
        return None

    def get_engine(self, symbol: str) -> Optional[SymbolIndicators]:
        """Get the raw engine for a symbol."""
        return self.engines.get(symbol)

    def all_ready(self) -> bool:
        """Check if all symbols have enough data."""
        return all(e.bars_loaded >= 200 for e in self.engines.values())

    def status(self) -> Dict[str, int]:
        """Get bars loaded per symbol."""
        return {sym: eng.bars_loaded for sym, eng in self.engines.items()}
