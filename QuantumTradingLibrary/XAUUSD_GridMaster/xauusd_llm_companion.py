"""
XAUUSD Grid Trading System - LLM Companion Script
==================================================
Integrates with Ollama (local LLM) for dynamic SL/TP adjustments.

Features:
- Real-time volatility regime detection
- Dynamic SL/TP multiplier adjustments
- Trade signal confidence scoring
- Integration with MT5 via file-based communication

Supported LLMs (via Ollama):
- gemma3:12b (recommended for production)
- gemma2:2b (faster, lighter)
- llama3:8b
- mistral:7b

Usage:
    python xauusd_llm_companion.py [--account ACCOUNT_ID]

Author: DooDoo - Quantum Trading Library
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("WARNING: ollama not installed - pip install ollama")

# Try to import MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("WARNING: MetaTrader5 not installed - pip install MetaTrader5")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | XAUUSD_LLM | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("xauusd_llm_companion.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class MarketAnalysis:
    """Market analysis result from LLM"""
    volatility_regime: VolatilityRegime
    sl_adjustment: float      # -0.5 to +0.5
    tp_adjustment: float      # -1.0 to +1.0
    confidence_boost: float   # 0 to 0.15
    tighten_sltp: bool
    llm_confidence: float     # 0 to 1
    reasoning: str
    timestamp: datetime


class XAUUSDLLMCompanion:
    """
    LLM-based companion for XAUUSD Grid Trading System.

    Analyzes market conditions and provides dynamic SL/TP adjustments
    via Ollama (local LLM).
    """

    # Default LLM model
    DEFAULT_MODEL = "gemma3:12b"
    FALLBACK_MODEL = "gemma2:2b"

    # MT5 Common folder path (Windows)
    MT5_COMMON_FOLDER = Path(os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal\Common\Files"))

    # Analysis interval
    ANALYSIS_INTERVAL = 30  # seconds

    # Hard-coded base multipliers (from MQL5)
    BASE_SL_MULTIPLIER = 1.5
    BASE_TP_MULTIPLIER = 3.0

    def __init__(self,
                 account_id: str = "113328",
                 model_name: str = None,
                 signal_file: str = "xauusd_llm_signal.txt"):
        """
        Initialize the LLM companion.

        Args:
            account_id: GetLeveraged account ID
            model_name: Ollama model name
            signal_file: Output signal file for MT5
        """
        self.account_id = account_id
        self.model_name = model_name or self.DEFAULT_MODEL
        self.signal_file = self.MT5_COMMON_FOLDER / signal_file
        self.running = False

        # Market data cache
        self.price_history: List[Dict] = []
        self.atr_history: List[float] = []
        self.volatility_history: List[float] = []

        # Last analysis
        self.last_analysis: Optional[MarketAnalysis] = None
        self.last_analysis_time = None

        self._verify_model()
        self._ensure_signal_file()

        log.info(f"XAUUSD LLM Companion initialized")
        log.info(f"Account: {self.account_id}")
        log.info(f"Model: {self.model_name}")
        log.info(f"Signal File: {self.signal_file}")

    def _verify_model(self):
        """Verify the LLM model is available"""
        if not OLLAMA_AVAILABLE:
            log.warning("Ollama not available - using rule-based fallback")
            return

        try:
            models = ollama.list()
            model_names = [m.get('name', '') for m in models.get('models', [])]

            if self.model_name not in str(model_names):
                log.warning(f"Model {self.model_name} not found")
                # Try fallback
                if self.FALLBACK_MODEL in str(model_names):
                    log.info(f"Using fallback model: {self.FALLBACK_MODEL}")
                    self.model_name = self.FALLBACK_MODEL
                else:
                    log.info(f"Attempting to pull {self.model_name}...")
                    try:
                        ollama.pull(self.model_name)
                        log.info("Model pulled successfully")
                    except Exception as e:
                        log.error(f"Failed to pull model: {e}")
            else:
                log.info(f"Model {self.model_name} verified")

        except Exception as e:
            log.error(f"Error verifying model: {e}")

    def _ensure_signal_file(self):
        """Ensure signal file directory exists"""
        self.signal_file.parent.mkdir(parents=True, exist_ok=True)

    def get_market_data(self) -> Optional[pd.DataFrame]:
        """
        Get XAUUSD market data from MT5.

        Returns:
            DataFrame with OHLCV data or None
        """
        if not MT5_AVAILABLE:
            log.warning("MT5 not available - cannot get market data")
            return None

        # Initialize MT5 if needed
        if not mt5.terminal_info():
            if not mt5.initialize():
                log.error(f"MT5 init failed: {mt5.last_error()}")
                return None

        # Get XAUUSD data
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 100)

        if rates is None or len(rates) == 0:
            log.warning("No XAUUSD data received")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        return df

    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate volatility metrics for the market.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dict with volatility metrics
        """
        if df is None or len(df) < 20:
            return {}

        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        current_atr = df['atr'].iloc[-1]

        # Calculate volatility (standard deviation of returns)
        df['returns'] = df['close'].pct_change()
        volatility = df['returns'].rolling(20).std().iloc[-1]

        # Historical ATR comparison
        avg_atr = df['atr'].mean()
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1

        # Trend strength (using EMAs)
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()

        current_price = df['close'].iloc[-1]
        ema8 = df['ema8'].iloc[-1]
        ema21 = df['ema21'].iloc[-1]
        ema200 = df['ema200'].iloc[-1]

        # Trend direction
        trend_up = ema8 > ema21 > ema200
        trend_down = ema8 < ema21 < ema200
        trend = "BULLISH" if trend_up else ("BEARISH" if trend_down else "NEUTRAL")

        # Price momentum
        momentum = (current_price - df['close'].iloc[-10]) / df['close'].iloc[-10]

        # Bollinger Band width (volatility indicator)
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_width'] = (df['bb_std'] * 4) / df['bb_mid']
        bb_width = df['bb_width'].iloc[-1]

        return {
            'current_price': current_price,
            'current_atr': current_atr,
            'avg_atr': avg_atr,
            'atr_ratio': atr_ratio,
            'volatility': volatility,
            'trend': trend,
            'momentum': momentum,
            'bb_width': bb_width,
            'ema8': ema8,
            'ema21': ema21,
            'ema200': ema200,
        }

    def classify_volatility_regime(self, metrics: Dict) -> VolatilityRegime:
        """
        Classify the current volatility regime.

        Args:
            metrics: Volatility metrics dict

        Returns:
            VolatilityRegime enum
        """
        if not metrics:
            return VolatilityRegime.NORMAL

        atr_ratio = metrics.get('atr_ratio', 1)
        volatility = metrics.get('volatility', 0)
        bb_width = metrics.get('bb_width', 0)

        # EXTREME: Very high volatility (e.g., news events)
        if atr_ratio > 2.0 or volatility > 0.015 or bb_width > 0.05:
            return VolatilityRegime.EXTREME

        # HIGH: Above average volatility
        if atr_ratio > 1.3 or volatility > 0.008 or bb_width > 0.03:
            return VolatilityRegime.HIGH

        # LOW: Below average volatility
        if atr_ratio < 0.7 or volatility < 0.003 or bb_width < 0.015:
            return VolatilityRegime.LOW

        # NORMAL: Average volatility
        return VolatilityRegime.NORMAL

    def rule_based_adjustment(self, metrics: Dict, regime: VolatilityRegime) -> MarketAnalysis:
        """
        Rule-based SL/TP adjustment (fallback when LLM unavailable).

        Args:
            metrics: Volatility metrics
            regime: Volatility regime

        Returns:
            MarketAnalysis with adjustments
        """
        sl_adj = 0.0
        tp_adj = 0.0
        conf_boost = 0.0
        tighten = False

        if regime == VolatilityRegime.EXTREME:
            # EXTREME: Widen SL to avoid whipsaws, reduce TP expectations
            sl_adj = 0.5   # 1.5 -> 2.0x ATR
            tp_adj = -0.5  # 3.0 -> 2.5x ATR
            conf_boost = -0.1  # Reduce confidence
            tighten = False

        elif regime == VolatilityRegime.HIGH:
            # HIGH: Slightly wider SL, maintain TP
            sl_adj = 0.25  # 1.5 -> 1.75x ATR
            tp_adj = 0.0   # Keep 3.0x ATR
            conf_boost = 0.0
            tighten = False

        elif regime == VolatilityRegime.LOW:
            # LOW: Tighter SL, wider TP (trending environment)
            sl_adj = -0.25  # 1.5 -> 1.25x ATR
            tp_adj = 0.5    # 3.0 -> 3.5x ATR
            conf_boost = 0.1  # Increase confidence (trending)
            tighten = True

        else:  # NORMAL
            # NORMAL: No adjustment
            sl_adj = 0.0
            tp_adj = 0.0
            conf_boost = 0.05
            tighten = False

        return MarketAnalysis(
            volatility_regime=regime,
            sl_adjustment=sl_adj,
            tp_adjustment=tp_adj,
            confidence_boost=conf_boost,
            tighten_sltp=tighten,
            llm_confidence=0.8,
            reasoning=f"Rule-based: {regime.value} volatility detected",
            timestamp=datetime.now()
        )

    def llm_analysis(self, metrics: Dict) -> Optional[MarketAnalysis]:
        """
        Use LLM for intelligent market analysis.

        Args:
            metrics: Volatility metrics

        Returns:
            MarketAnalysis or None if failed
        """
        if not OLLAMA_AVAILABLE:
            return None

        prompt = f"""XAUUSD GRID TRADING - VOLATILITY ANALYSIS

CURRENT MARKET CONDITIONS:
- Price: ${metrics.get('current_price', 0):.2f}
- Current ATR (14): ${metrics.get('current_atr', 0):.2f}
- Average ATR: ${metrics.get('avg_atr', 0):.2f}
- ATR Ratio: {metrics.get('atr_ratio', 1):.2f}x average
- Volatility (20-period): {metrics.get('volatility', 0)*100:.3f}%
- Trend: {metrics.get('trend', 'UNKNOWN')}
- Momentum (10-bar): {metrics.get('momentum', 0)*100:.2f}%
- Bollinger Band Width: {metrics.get('bb_width', 0)*100:.2f}%

GRID SYSTEM BASE PARAMETERS:
- Base SL: 1.5x ATR
- Base TP: 3.0x ATR
- Partial TP: 50% at first target
- Break-Even: 30% of TP
- Trailing Stop: 50% of TP

TASK:
Analyze the market conditions and recommend SL/TP adjustments.

Respond in this EXACT format:
VOLATILITY_REGIME: [LOW/NORMAL/HIGH/EXTREME]
SL_ADJUSTMENT: [number between -0.5 and 0.5]
TP_ADJUSTMENT: [number between -1.0 and 1.0]
CONFIDENCE_BOOST: [number between -0.1 and 0.15]
TIGHTEN: [true/false]
CONFIDENCE: [0-100]%

REASONING:
[1-2 sentences explaining your recommendation]

IMPORTANT:
- Positive SL_ADJUSTMENT = WIDER stop loss (more room)
- Negative SL_ADJUSTMENT = TIGHTER stop loss (less room)
- Consider current volatility vs historical
- In low volatility (trending): tighter SL, wider TP
- In high volatility (ranging): wider SL, tighter TP
- In extreme volatility: caution, wider SL, reduced TP
"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 400,
                }
            )

            return self._parse_llm_response(response['response'], metrics)

        except Exception as e:
            log.error(f"LLM analysis failed: {e}")
            return None

    def _parse_llm_response(self, response: str, metrics: Dict) -> MarketAnalysis:
        """Parse LLM response into MarketAnalysis"""
        import re

        # Defaults
        regime = VolatilityRegime.NORMAL
        sl_adj = 0.0
        tp_adj = 0.0
        conf_boost = 0.05
        tighten = False
        confidence = 0.7
        reasoning = "Parsed from LLM response"

        try:
            # Parse volatility regime
            regime_match = re.search(r'VOLATILITY_REGIME:\s*(\w+)', response, re.I)
            if regime_match:
                regime_str = regime_match.group(1).upper()
                if regime_str in [r.value for r in VolatilityRegime]:
                    regime = VolatilityRegime(regime_str)

            # Parse SL adjustment
            sl_match = re.search(r'SL_ADJUSTMENT:\s*([-\d.]+)', response, re.I)
            if sl_match:
                sl_adj = float(sl_match.group(1))
                sl_adj = max(-0.5, min(0.5, sl_adj))

            # Parse TP adjustment
            tp_match = re.search(r'TP_ADJUSTMENT:\s*([-\d.]+)', response, re.I)
            if tp_match:
                tp_adj = float(tp_match.group(1))
                tp_adj = max(-1.0, min(1.0, tp_adj))

            # Parse confidence boost
            cb_match = re.search(r'CONFIDENCE_BOOST:\s*([-\d.]+)', response, re.I)
            if cb_match:
                conf_boost = float(cb_match.group(1))
                conf_boost = max(-0.1, min(0.15, conf_boost))

            # Parse tighten
            tighten_match = re.search(r'TIGHTEN:\s*(true|false)', response, re.I)
            if tighten_match:
                tighten = tighten_match.group(1).lower() == 'true'

            # Parse confidence
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response, re.I)
            if conf_match:
                confidence = int(conf_match.group(1)) / 100.0

            # Parse reasoning
            reason_match = re.search(r'REASONING:\s*(.+?)(?=$|\n\n)', response, re.I | re.S)
            if reason_match:
                reasoning = reason_match.group(1).strip()[:200]

        except Exception as e:
            log.error(f"Error parsing LLM response: {e}")

        return MarketAnalysis(
            volatility_regime=regime,
            sl_adjustment=sl_adj,
            tp_adjustment=tp_adj,
            confidence_boost=conf_boost,
            tighten_sltp=tighten,
            llm_confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now()
        )

    def analyze_market(self) -> Optional[MarketAnalysis]:
        """
        Perform complete market analysis.

        Returns:
            MarketAnalysis with recommendations
        """
        # Get market data
        df = self.get_market_data()
        if df is None:
            log.warning("Cannot get market data - using last analysis")
            return self.last_analysis

        # Calculate metrics
        metrics = self.calculate_volatility_metrics(df)
        if not metrics:
            log.warning("Cannot calculate metrics")
            return self.last_analysis

        # Classify volatility regime
        regime = self.classify_volatility_regime(metrics)

        # Try LLM analysis first
        analysis = None
        if OLLAMA_AVAILABLE:
            analysis = self.llm_analysis(metrics)

        # Fall back to rule-based if LLM fails
        if analysis is None:
            analysis = self.rule_based_adjustment(metrics, regime)

        self.last_analysis = analysis
        self.last_analysis_time = datetime.now()

        return analysis

    def write_signal_file(self, analysis: MarketAnalysis):
        """
        Write analysis to signal file for MT5 EA.

        Args:
            analysis: MarketAnalysis to write
        """
        content = f"""# XAUUSD LLM Signal File
# Generated: {analysis.timestamp.isoformat()}
# Account: {self.account_id}

sl_adj={analysis.sl_adjustment:.3f}
tp_adj={analysis.tp_adjustment:.3f}
confidence_boost={analysis.confidence_boost:.3f}
volatility_regime={analysis.volatility_regime.value}
llm_confidence={analysis.llm_confidence:.3f}
tighten={'true' if analysis.tighten_sltp else 'false'}

# Effective multipliers:
# SL: {self.BASE_SL_MULTIPLIER + analysis.sl_adjustment:.2f}x ATR
# TP: {self.BASE_TP_MULTIPLIER + analysis.tp_adjustment:.2f}x ATR

# Reasoning: {analysis.reasoning[:100]}
"""

        try:
            with open(self.signal_file, 'w') as f:
                f.write(content)
            log.debug(f"Signal file updated: {self.signal_file}")
        except Exception as e:
            log.error(f"Failed to write signal file: {e}")

    def run(self):
        """Main loop - continuously analyze and update signal file"""
        log.info("="*60)
        log.info("XAUUSD LLM COMPANION - STARTED")
        log.info(f"Account: {self.account_id}")
        log.info(f"Model: {self.model_name}")
        log.info(f"Analysis Interval: {self.ANALYSIS_INTERVAL}s")
        log.info("="*60)

        self.running = True

        while self.running:
            try:
                # Perform analysis
                analysis = self.analyze_market()

                if analysis:
                    # Write to signal file
                    self.write_signal_file(analysis)

                    # Log result
                    log.info(f"Analysis: {analysis.volatility_regime.value} | "
                            f"SL_adj={analysis.sl_adjustment:+.2f} | "
                            f"TP_adj={analysis.tp_adjustment:+.2f} | "
                            f"Conf={analysis.llm_confidence:.0%}")

                # Wait for next interval
                time.sleep(self.ANALYSIS_INTERVAL)

            except KeyboardInterrupt:
                log.info("Shutdown requested...")
                self.running = False
            except Exception as e:
                log.error(f"Analysis cycle error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)

        log.info("XAUUSD LLM Companion stopped")

    def stop(self):
        """Stop the companion"""
        self.running = False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="XAUUSD Grid Trading LLM Companion"
    )
    parser.add_argument(
        "--account",
        default="113328",
        help="GetLeveraged account ID"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name (default: gemma3:12b)"
    )
    parser.add_argument(
        "--signal-file",
        default="xauusd_llm_signal.txt",
        help="Signal file name for MT5"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Analysis interval in seconds"
    )

    args = parser.parse_args()

    # Create companion
    companion = XAUUSDLLMCompanion(
        account_id=args.account,
        model_name=args.model,
        signal_file=args.signal_file
    )
    companion.ANALYSIS_INTERVAL = args.interval

    # Run
    try:
        companion.run()
    except KeyboardInterrupt:
        companion.stop()


if __name__ == "__main__":
    main()
