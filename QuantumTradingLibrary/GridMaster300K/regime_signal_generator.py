"""
GridMaster 300K - Regime Signal Generator
==========================================
Generates regime signals for the 3 specialized grid experts.
Uses compression filtering with 80%+ confidence threshold.

Each expert has +12 compression boost.
"""

import json
import time
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

# Configuration
CONFIDENCE_THRESHOLD = 0.80
COMPRESSION_BOOST_PER_EXPERT = 12
CHECK_INTERVAL_SECONDS = 30

# File paths
SCRIPT_DIR = Path(__file__).parent
SIGNAL_OUTPUT_DIR = SCRIPT_DIR
CONFIG_FILE = SCRIPT_DIR / "account_300k_config.json"


class RegimeDetector:
    """
    Detects market regime (BULLISH, BEARISH, NEUTRAL) with confidence scoring.
    Uses EMA-based methodology with compression boost.
    """

    def __init__(self, compression_boost: int = 12):
        self.compression_boost = compression_boost
        self.ema_fast_period = 8
        self.ema_slow_period = 21
        self.ema_trend_period = 200

    def calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA for given prices."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def detect_regime(self, prices: np.ndarray) -> Tuple[str, float]:
        """
        Detect current market regime.

        Returns:
            Tuple of (regime_name, confidence)
        """
        if len(prices) < self.ema_trend_period:
            return "NEUTRAL", 0.0

        current_price = prices[-1]
        ema_fast = self.calculate_ema(prices, self.ema_fast_period)
        ema_slow = self.calculate_ema(prices, self.ema_slow_period)
        ema_200 = self.calculate_ema(prices, self.ema_trend_period)

        # Calculate regime confidences
        bullish_conf = self._calc_bullish_confidence(current_price, ema_fast, ema_slow, ema_200)
        bearish_conf = self._calc_bearish_confidence(current_price, ema_fast, ema_slow, ema_200)
        neutral_conf = self._calc_neutral_confidence(current_price, ema_fast, ema_slow, ema_200)

        # Determine dominant regime
        regimes = {
            "BULLISH": bullish_conf,
            "BEARISH": bearish_conf,
            "NEUTRAL": neutral_conf
        }

        dominant = max(regimes, key=regimes.get)
        confidence = regimes[dominant]

        return dominant, confidence

    def _calc_bullish_confidence(self, price: float, ema_fast: float,
                                  ema_slow: float, ema_200: float) -> float:
        """Calculate bullish regime confidence."""
        confidence = 0.0

        # Price above EMA200
        if price > ema_200:
            confidence += 0.35

        # Fast EMA above Slow EMA
        if ema_fast > ema_slow:
            confidence += 0.35

        # Slow EMA above EMA200
        if ema_slow > ema_200:
            confidence += 0.30

        # Apply compression boost
        confidence += self.compression_boost / 100.0

        return min(confidence, 1.0)

    def _calc_bearish_confidence(self, price: float, ema_fast: float,
                                  ema_slow: float, ema_200: float) -> float:
        """Calculate bearish regime confidence."""
        confidence = 0.0

        # Price below EMA200
        if price < ema_200:
            confidence += 0.35

        # Fast EMA below Slow EMA
        if ema_fast < ema_slow:
            confidence += 0.35

        # Slow EMA below EMA200
        if ema_slow < ema_200:
            confidence += 0.30

        # Apply compression boost
        confidence += self.compression_boost / 100.0

        return min(confidence, 1.0)

    def _calc_neutral_confidence(self, price: float, ema_fast: float,
                                  ema_slow: float, ema_200: float) -> float:
        """Calculate neutral/ranging regime confidence."""
        confidence = 0.0

        # Distance from EMA200
        dist_from_200 = abs(price - ema_200) / ema_200 if ema_200 > 0 else 0

        if dist_from_200 < 0.01:
            confidence += 0.40
        elif dist_from_200 < 0.02:
            confidence += 0.25
        else:
            confidence += 0.10

        # EMA convergence
        ema_diff = abs(ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0

        if ema_diff < 0.005:
            confidence += 0.40
        elif ema_diff < 0.01:
            confidence += 0.25
        else:
            confidence += 0.10

        # Apply compression boost
        confidence += self.compression_boost / 100.0

        return min(confidence, 1.0)


class GridSignalGenerator:
    """
    Generates signals for the 3 grid experts based on regime detection.
    """

    def __init__(self):
        self.bearish_detector = RegimeDetector(COMPRESSION_BOOST_PER_EXPERT)
        self.bullish_detector = RegimeDetector(COMPRESSION_BOOST_PER_EXPERT)
        self.neutral_detector = RegimeDetector(COMPRESSION_BOOST_PER_EXPERT)

        self.load_config()

    def load_config(self):
        """Load account configuration."""
        try:
            with open(CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {CONFIG_FILE}")
            self.config = {}

    def generate_signals(self, symbol: str, prices: np.ndarray) -> Dict:
        """
        Generate signals for all three experts.

        Returns:
            Dictionary with signals for each expert
        """
        timestamp = datetime.datetime.now().isoformat()

        # Detect regime for each expert perspective
        bearish_regime, bearish_conf = self.bearish_detector.detect_regime(prices)
        bullish_regime, bullish_conf = self.bullish_detector.detect_regime(prices)
        neutral_regime, neutral_conf = self.neutral_detector.detect_regime(prices)

        signals = {
            "timestamp": timestamp,
            "symbol": symbol,
            "current_price": float(prices[-1]) if len(prices) > 0 else 0,
            "experts": {
                "BEARISH": {
                    "expert_id": "GRID_EXPERT_BEARISH",
                    "magic_number": 300001,
                    "detected_regime": bearish_regime,
                    "confidence": round(bearish_conf, 4),
                    "compression_boost": COMPRESSION_BOOST_PER_EXPERT,
                    "meets_threshold": bearish_conf >= CONFIDENCE_THRESHOLD,
                    "action": self._get_bearish_action(bearish_regime, bearish_conf),
                    "direction": "SELL"
                },
                "BULLISH": {
                    "expert_id": "GRID_EXPERT_BULLISH",
                    "magic_number": 300002,
                    "detected_regime": bullish_regime,
                    "confidence": round(bullish_conf, 4),
                    "compression_boost": COMPRESSION_BOOST_PER_EXPERT,
                    "meets_threshold": bullish_conf >= CONFIDENCE_THRESHOLD,
                    "action": self._get_bullish_action(bullish_regime, bullish_conf),
                    "direction": "BUY"
                },
                "NEUTRAL": {
                    "expert_id": "GRID_EXPERT_NEUTRAL",
                    "magic_number": 300003,
                    "detected_regime": neutral_regime,
                    "confidence": round(neutral_conf, 4),
                    "compression_boost": COMPRESSION_BOOST_PER_EXPERT,
                    "meets_threshold": neutral_conf >= CONFIDENCE_THRESHOLD,
                    "action": self._get_neutral_action(neutral_regime, neutral_conf),
                    "direction": "BOTH"
                }
            },
            "compression_filter": {
                "threshold": CONFIDENCE_THRESHOLD,
                "boost_per_expert": COMPRESSION_BOOST_PER_EXPERT,
                "total_active_boost": COMPRESSION_BOOST_PER_EXPERT * 3
            }
        }

        return signals

    def _get_bearish_action(self, regime: str, confidence: float) -> str:
        """Get action for bearish expert."""
        if regime == "BEARISH" and confidence >= CONFIDENCE_THRESHOLD:
            return "ACTIVE_SELL"
        return "STANDBY"

    def _get_bullish_action(self, regime: str, confidence: float) -> str:
        """Get action for bullish expert."""
        if regime == "BULLISH" and confidence >= CONFIDENCE_THRESHOLD:
            return "ACTIVE_BUY"
        return "STANDBY"

    def _get_neutral_action(self, regime: str, confidence: float) -> str:
        """Get action for neutral expert."""
        if regime == "NEUTRAL" and confidence >= CONFIDENCE_THRESHOLD:
            return "ACTIVE_RANGE"
        return "STANDBY"

    def write_signal_file(self, signals: Dict, symbol: str):
        """Write signals to file for MT5 to read."""
        filename = f"signal_GRID_{symbol}.json"
        filepath = SIGNAL_OUTPUT_DIR / filename

        with open(filepath, 'w') as f:
            json.dump(signals, f, indent=2)

        print(f"[{signals['timestamp']}] Signals written to {filepath}")

        # Also print summary
        self._print_summary(signals)

    def _print_summary(self, signals: Dict):
        """Print signal summary."""
        print(f"\n{'='*60}")
        print(f"GRIDMASTER 300K - REGIME SIGNALS")
        print(f"{'='*60}")
        print(f"Symbol: {signals['symbol']}")
        print(f"Price: {signals['current_price']}")
        print(f"{'='*60}")

        for expert_name, expert_data in signals['experts'].items():
            status = "ACTIVE" if expert_data['meets_threshold'] else "STANDBY"
            print(f"{expert_name}: {expert_data['detected_regime']} "
                  f"({expert_data['confidence']*100:.1f}%) -> {expert_data['action']}")

        print(f"{'='*60}\n")


def main():
    """Main loop for signal generation."""
    print("GridMaster 300K - Regime Signal Generator Started")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD*100}%")
    print(f"Compression Boost: +{COMPRESSION_BOOST_PER_EXPERT} per expert")
    print(f"Check Interval: {CHECK_INTERVAL_SECONDS} seconds")
    print("="*60)

    generator = GridSignalGenerator()

    # For demonstration, generate sample signals
    # In production, this would connect to MT5 or data feed

    symbols = ["BTCUSD", "ETHUSD", "XAUUSD"]

    print("\nDemo mode: Generating sample signals...")
    print("In production, connect this to live market data.\n")

    for symbol in symbols:
        # Generate sample price data (in production, get from MT5)
        np.random.seed(hash(symbol) % 2**32)
        base_price = {"BTCUSD": 45000, "ETHUSD": 2500, "XAUUSD": 2000}.get(symbol, 1000)
        prices = np.cumsum(np.random.randn(300) * base_price * 0.001) + base_price

        signals = generator.generate_signals(symbol, prices)
        generator.write_signal_file(signals, symbol)


if __name__ == "__main__":
    main()
