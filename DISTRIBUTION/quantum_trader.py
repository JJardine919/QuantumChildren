"""
⚡ QUANTUM CHILDREN - FREE TRADING SYSTEM
=========================================

The secret is in the compression.

Run: python quantum_trader.py

This system:
1. Analyzes market data using quantum-inspired compression
2. Detects tradeable regimes (CLEAN vs VOLATILE)
3. Generates signals when confidence is high
4. Contributes to the QuantumChildren network

Your signals help improve the models for everyone.
"""

import json
import time
import zlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Try to import optional components
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARNING] MetaTrader5 not installed. Running in simulation mode.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. Using rule-based signals only.")

# Import entropy collector
from entropy_collector import collect_signal, collect_entropy_snapshot, collect_outcome, NODE_ID

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_FILE = Path("config.json")

DEFAULT_CONFIG = {
    "symbols": ["BTCUSD", "XAUUSD"],
    "timeframe": "M5",
    "lot_size": 0.01,
    "confidence_threshold": 0.55,
    "entropy_threshold": 4.5,
    "max_positions": 3,
    "check_interval": 60,
    "enable_trading": False,
    "magic_number": 777777
}

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    else:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG

CONFIG = load_config()

# ============================================================
# ENUMS
# ============================================================

class Regime(Enum):
    CLEAN = "CLEAN"
    VOLATILE = "VOLATILE"
    CHOPPY = "CHOPPY"

class Direction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

# ============================================================
# COMPRESSION-BASED REGIME DETECTION
# ============================================================

class RegimeDetector:
    """
    Detects market regime using compression ratio.
    Higher compression = more predictable = CLEAN regime
    """

    def analyze(self, prices: np.ndarray) -> Tuple[Regime, float, float]:
        """
        Analyze price data and return regime with entropy.

        Returns:
            Tuple of (Regime, fidelity, entropy)
        """
        # Convert to bytes and compress
        data_bytes = prices.astype(np.float32).tobytes()
        compressed = zlib.compress(data_bytes, level=9)

        # Compression ratio
        ratio = len(data_bytes) / len(compressed)

        # Calculate entropy (Shannon)
        # Normalize prices to distribution
        price_changes = np.diff(prices)
        if len(price_changes) == 0:
            return Regime.CHOPPY, 0.5, 8.0

        # Bin the changes
        hist, _ = np.histogram(price_changes, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zeros

        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(len(hist) + 1)
        entropy = min(8.0, max(0.0, entropy * 8))  # Scale to 0-8

        # Determine regime
        if ratio >= 1.3 and entropy < CONFIG['entropy_threshold']:
            regime = Regime.CLEAN
            fidelity = 0.96
        elif ratio >= 1.1:
            regime = Regime.VOLATILE
            fidelity = 0.88
        else:
            regime = Regime.CHOPPY
            fidelity = 0.75

        return regime, fidelity, entropy

# ============================================================
# SIGNAL GENERATOR
# ============================================================

class SignalGenerator:
    """Generate trading signals using technical analysis + compression"""

    def __init__(self):
        self.regime_detector = RegimeDetector()

    def analyze(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        Analyze market data and generate signal.

        Returns dict with:
            - direction: BUY/SELL/HOLD
            - confidence: 0-1
            - regime: CLEAN/VOLATILE/CHOPPY
            - entropy: 0-8
            - reason: explanation
        """
        if len(df) < 50:
            return self._hold("Insufficient data")

        prices = df['close'].values
        regime, fidelity, entropy = self.regime_detector.analyze(prices)

        # Only trade in CLEAN regime
        if regime != Regime.CLEAN:
            return {
                'direction': Direction.HOLD.value,
                'confidence': 0.0,
                'regime': regime.value,
                'entropy': entropy,
                'fidelity': fidelity,
                'reason': f"Regime is {regime.value}, waiting for CLEAN"
            }

        # Calculate indicators
        rsi = self._rsi(prices)
        macd, signal = self._macd(prices)
        momentum = self._momentum(prices)

        # Generate direction
        bullish_signals = 0
        bearish_signals = 0

        # RSI
        if rsi < 30:
            bullish_signals += 1
        elif rsi > 70:
            bearish_signals += 1

        # MACD
        if macd > signal:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # Momentum
        if momentum > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # Determine direction
        if bullish_signals > bearish_signals:
            direction = Direction.BUY
            confidence = (bullish_signals / 3) * fidelity
        elif bearish_signals > bullish_signals:
            direction = Direction.SELL
            confidence = (bearish_signals / 3) * fidelity
        else:
            direction = Direction.HOLD
            confidence = 0.5

        # Apply confidence threshold
        if confidence < CONFIG['confidence_threshold']:
            direction = Direction.HOLD
            reason = f"Confidence {confidence:.2f} below threshold {CONFIG['confidence_threshold']}"
        else:
            reason = f"{'Bullish' if direction == Direction.BUY else 'Bearish'} signals: RSI={rsi:.1f}, MACD={'above' if macd > signal else 'below'} signal"

        return {
            'direction': direction.value,
            'confidence': confidence,
            'regime': regime.value,
            'entropy': entropy,
            'fidelity': fidelity,
            'rsi': rsi,
            'macd': macd,
            'reason': reason
        }

    def _hold(self, reason: str) -> dict:
        return {
            'direction': Direction.HOLD.value,
            'confidence': 0.0,
            'regime': Regime.CHOPPY.value,
            'entropy': 8.0,
            'fidelity': 0.5,
            'reason': reason
        }

    def _rsi(self, prices: np.ndarray, period: int = 14) -> float:
        delta = np.diff(prices)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _macd(self, prices: np.ndarray) -> Tuple[float, float]:
        exp1 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
        exp2 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
        macd = exp1 - exp2
        signal = pd.Series(prices).ewm(span=9).mean().iloc[-1]
        return macd, signal

    def _momentum(self, prices: np.ndarray, period: int = 10) -> float:
        if len(prices) < period:
            return 0.0
        return prices[-1] - prices[-period]

# ============================================================
# MT5 INTERFACE
# ============================================================

class MT5Interface:
    """Interface with MetaTrader 5"""

    def __init__(self):
        self.connected = False

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            return False

        if not mt5.initialize():
            print(f"[MT5] Failed to initialize: {mt5.last_error()}")
            return False

        account = mt5.account_info()
        if account:
            print(f"[MT5] Connected: {account.login} | Balance: ${account.balance:,.2f}")
            self.connected = True
            return True
        return False

    def get_data(self, symbol: str, timeframe: str = "M5", bars: int = 200) -> Optional[pd.DataFrame]:
        if not self.connected:
            return None

        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def get_price(self, symbol: str) -> Optional[float]:
        if not self.connected:
            return None
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid if tick else None

    def has_position(self, symbol: str, magic: int) -> bool:
        if not self.connected:
            return False
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                if pos.magic == magic:
                    return True
        return False

    def open_trade(self, symbol: str, direction: str, lot: float, magic: int) -> bool:
        if not self.connected or not CONFIG['enable_trading']:
            return False

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False

        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "magic": magic,
            "comment": f"QC_{NODE_ID[:8]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[TRADE] {direction} {symbol} @ {price} | Ticket: {result.order}")
            return True
        else:
            print(f"[TRADE] Failed: {result.comment}")
            return False

    def shutdown(self):
        if MT5_AVAILABLE:
            mt5.shutdown()

# ============================================================
# MAIN TRADER
# ============================================================

class QuantumTrader:
    """Main trading system"""

    def __init__(self):
        self.mt5 = MT5Interface()
        self.signal_gen = SignalGenerator()
        self.running = False

    def run(self):
        print("=" * 60)
        print("  ⚡ QUANTUM CHILDREN - Trading System")
        print(f"  Node ID: {NODE_ID}")
        print(f"  Symbols: {CONFIG['symbols']}")
        print(f"  Trading: {'ENABLED' if CONFIG['enable_trading'] else 'DISABLED (watching only)'}")
        print("=" * 60)

        # Connect to MT5
        if MT5_AVAILABLE:
            if not self.mt5.connect():
                print("[ERROR] Could not connect to MT5. Make sure it's running.")
                return

        self.running = True

        try:
            while self.running:
                self._cycle()
                time.sleep(CONFIG['check_interval'])

        except KeyboardInterrupt:
            print("\n[STOP] Shutting down...")

        finally:
            self.mt5.shutdown()

    def _cycle(self):
        """Run one analysis cycle"""
        print(f"\n{'='*60}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        for symbol in CONFIG['symbols']:
            self._analyze_symbol(symbol)

    def _analyze_symbol(self, symbol: str):
        """Analyze a single symbol"""
        # Get data
        df = self.mt5.get_data(symbol, CONFIG['timeframe'])

        if df is None or len(df) < 50:
            print(f"  [{symbol}] No data available")
            return

        # Generate signal
        signal = self.signal_gen.analyze(df, symbol)
        price = self.mt5.get_price(symbol) or df['close'].iloc[-1]

        # Display
        regime_icon = "+" if signal['regime'] == "CLEAN" else "-"
        print(f"  [{regime_icon}] {symbol}: {signal['regime']} | "
              f"{signal['direction']} ({signal['confidence']:.2f}) | "
              f"Entropy: {signal['entropy']:.2f}")

        if signal.get('reason'):
            print(f"      {signal['reason']}")

        # Send to QuantumChildren network
        collect_signal({
            'symbol': symbol,
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'quantum_entropy': signal['entropy'],
            'dominant_state': signal['fidelity'],
            'price': price,
            'regime': signal['regime']
        })

        # Send entropy snapshot
        collect_entropy_snapshot(
            symbol=symbol,
            timeframe=CONFIG['timeframe'],
            entropy=signal['entropy'],
            dominant=signal['fidelity'],
            significant=int(signal['entropy'] * 10),
            variance=signal.get('rsi', 50),
            regime=signal['regime'],
            price=price
        )

        # Execute trade if conditions met
        if (signal['direction'] in ['BUY', 'SELL'] and
            signal['confidence'] >= CONFIG['confidence_threshold'] and
            signal['regime'] == 'CLEAN'):

            if not self.mt5.has_position(symbol, CONFIG['magic_number']):
                if CONFIG['enable_trading']:
                    self.mt5.open_trade(
                        symbol,
                        signal['direction'],
                        CONFIG['lot_size'],
                        CONFIG['magic_number']
                    )
                else:
                    print(f"      [SIGNAL] Would {signal['direction']} - enable_trading is False")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    trader = QuantumTrader()
    trader.run()
