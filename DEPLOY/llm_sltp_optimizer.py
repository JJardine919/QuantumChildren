"""
LLM-Driven Dynamic SL/TP Optimizer for GetLeveraged Grid Traders
Uses Ollama (local LLM) to analyze market conditions and suggest SL/TP adjustments
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('llm_sltp_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - HARD-CODED ATR MULTIPLIERS (BASE VALUES)
# =============================================================================
BASE_SL_ATR_MULT = 1.5      # Stop Loss = 1.5x ATR
BASE_TP_ATR_MULT = 3.0      # Take Profit = 3.0x ATR
PARTIAL_TP_RATIO = 0.50     # 50% partial take profit
ATR_PERIOD = 14             # ATR calculation period

# LLM adjustment ranges (prevents extreme values)
MIN_SL_MULT = 1.0
MAX_SL_MULT = 2.5
MIN_TP_MULT = 2.0
MAX_TP_MULT = 5.0

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"  # or "llama2", "codellama", etc.

# Symbols to monitor
SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD"]

# GetLeveraged accounts
ACCOUNTS = [
    {"login": 113328, "password": "H*M5c7jpR7", "server": "GetLeveraged-Trade"},
    {"login": 113326, "password": "%bwN)IvJ5F", "server": "GetLeveraged-Trade"},
    {"login": 107245, "password": "$86eCmFbXR", "server": "GetLeveraged-Trade"},
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class MarketAnalysis:
    symbol: str
    timestamp: datetime
    atr: float
    volatility_regime: str      # "low", "normal", "high", "extreme"
    trend_strength: float       # 0-1 scale
    trend_direction: str        # "bullish", "bearish", "neutral"
    entropy_score: float        # 0-1 scale (lower = more predictable)
    suggested_sl_mult: float
    suggested_tp_mult: float
    reasoning: str

@dataclass
class PositionUpdate:
    ticket: int
    symbol: str
    new_sl: float
    new_tp: float
    reason: str

# =============================================================================
# MARKET DATA COLLECTION
# =============================================================================
def get_market_data(symbol: str, timeframe=mt5.TIMEFRAME_M5, bars: int = 200) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from MT5"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        logger.warning(f"No data for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr)

def calculate_volatility_regime(df: pd.DataFrame, atr: float) -> str:
    """Determine volatility regime based on ATR percentile"""
    # Calculate historical ATR
    atrs = []
    for i in range(20, len(df)):
        subset = df.iloc[i-20:i]
        tr = (subset['high'] - subset['low']).mean()
        atrs.append(tr)

    if len(atrs) < 50:
        return "normal"

    percentile = np.percentile(atrs, [25, 50, 75, 95])

    if atr < percentile[0]:
        return "low"
    elif atr < percentile[2]:
        return "normal"
    elif atr < percentile[3]:
        return "high"
    else:
        return "extreme"

def calculate_trend_metrics(df: pd.DataFrame) -> Tuple[float, str]:
    """Calculate trend strength (0-1) and direction"""
    close = df['close']

    # EMAs
    ema8 = close.ewm(span=8).mean()
    ema21 = close.ewm(span=21).mean()
    ema200 = close.ewm(span=200).mean()

    current_price = close.iloc[-1]
    ema8_val = ema8.iloc[-1]
    ema21_val = ema21.iloc[-1]
    ema200_val = ema200.iloc[-1]

    # Direction
    if current_price > ema200_val and ema8_val > ema21_val:
        direction = "bullish"
    elif current_price < ema200_val and ema8_val < ema21_val:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength (0-1 based on alignment)
    strength = 0.0
    if (ema8_val > ema21_val > ema200_val) or (ema8_val < ema21_val < ema200_val):
        strength = 0.8  # Strong alignment
    elif (ema8_val > ema21_val) or (ema8_val < ema21_val):
        strength = 0.5  # Partial alignment
    else:
        strength = 0.2  # No clear trend

    # ADX-like measure
    high_low_range = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
    directional_move = abs(close.diff()).rolling(14).mean().iloc[-1]
    if high_low_range > 0:
        strength *= min(directional_move / high_low_range, 1.0)

    return float(strength), direction

def calculate_entropy(df: pd.DataFrame) -> float:
    """Calculate Shannon entropy of returns (lower = more predictable)"""
    returns = df['close'].pct_change().dropna()

    # Bin returns
    hist, _ = np.histogram(returns, bins=20, density=True)
    hist = hist / (hist.sum() + 1e-10)

    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    # Normalize to 0-1 (max entropy for 20 bins is ~4.3)
    normalized = min(entropy / 4.3, 1.0)
    return float(normalized)

# =============================================================================
# LLM INTEGRATION
# =============================================================================
def query_llm(prompt: str) -> Optional[str]:
    """Query local Ollama LLM"""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 500
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            logger.error(f"LLM error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"LLM connection failed: {e}")
        return None

def get_llm_sltp_suggestion(analysis: Dict) -> Tuple[float, float, str]:
    """Get SL/TP multiplier suggestions from LLM"""

    prompt = f"""You are a trading risk manager. Analyze these market conditions and suggest ATR multipliers for stop-loss and take-profit.

MARKET CONDITIONS:
- Symbol: {analysis['symbol']}
- Volatility Regime: {analysis['volatility_regime']}
- Trend Direction: {analysis['trend_direction']}
- Trend Strength: {analysis['trend_strength']:.2f} (0-1 scale)
- Entropy Score: {analysis['entropy']:.2f} (lower = more predictable)
- Current ATR: {analysis['atr']:.5f}

BASE SETTINGS:
- Base SL Multiplier: 1.5x ATR
- Base TP Multiplier: 3.0x ATR

CONSTRAINTS:
- SL must be between 1.0x and 2.5x ATR
- TP must be between 2.0x and 5.0x ATR
- Maintain at least 1.5:1 reward-to-risk ratio

RULES:
- HIGH volatility -> WIDER stops (higher multipliers)
- EXTREME volatility -> Much wider stops or skip trading
- LOW entropy (predictable) -> Tighter stops OK
- HIGH entropy (chaotic) -> Wider stops needed
- STRONG trend -> Can use tighter stops in trend direction
- WEAK/NEUTRAL trend -> Wider stops needed

Respond ONLY in this exact JSON format:
{{"sl_mult": 1.5, "tp_mult": 3.0, "reasoning": "brief explanation"}}
"""

    response = query_llm(prompt)

    if response:
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                sl = float(data.get('sl_mult', BASE_SL_ATR_MULT))
                tp = float(data.get('tp_mult', BASE_TP_ATR_MULT))
                reasoning = data.get('reasoning', 'LLM suggestion')

                # Clamp to valid ranges
                sl = max(MIN_SL_MULT, min(MAX_SL_MULT, sl))
                tp = max(MIN_TP_MULT, min(MAX_TP_MULT, tp))

                # Ensure minimum R:R
                if tp / sl < 1.5:
                    tp = sl * 1.5

                return sl, tp, reasoning
        except Exception as e:
            logger.warning(f"LLM parse error: {e}")

    # Fallback to rule-based if LLM fails
    return get_rule_based_suggestion(analysis)

def get_rule_based_suggestion(analysis: Dict) -> Tuple[float, float, str]:
    """Fallback rule-based SL/TP suggestion when LLM unavailable"""

    sl_mult = BASE_SL_ATR_MULT
    tp_mult = BASE_TP_ATR_MULT
    reasons = []

    vol_regime = analysis['volatility_regime']
    entropy = analysis['entropy']
    trend_strength = analysis['trend_strength']

    # Volatility adjustments
    if vol_regime == "extreme":
        sl_mult += 0.5
        tp_mult += 1.0
        reasons.append("extreme volatility: widened stops")
    elif vol_regime == "high":
        sl_mult += 0.25
        tp_mult += 0.5
        reasons.append("high volatility: slightly widened")
    elif vol_regime == "low":
        sl_mult -= 0.2
        tp_mult -= 0.3
        reasons.append("low volatility: tightened stops")

    # Entropy adjustments
    if entropy > 0.7:
        sl_mult += 0.3
        reasons.append("high entropy: unpredictable market")
    elif entropy < 0.3:
        sl_mult -= 0.15
        reasons.append("low entropy: predictable patterns")

    # Trend strength adjustments
    if trend_strength > 0.7:
        tp_mult += 0.5
        reasons.append("strong trend: extended TP target")
    elif trend_strength < 0.3:
        sl_mult += 0.2
        reasons.append("weak trend: protective SL")

    # Clamp values
    sl_mult = max(MIN_SL_MULT, min(MAX_SL_MULT, sl_mult))
    tp_mult = max(MIN_TP_MULT, min(MAX_TP_MULT, tp_mult))

    # Ensure R:R
    if tp_mult / sl_mult < 1.5:
        tp_mult = sl_mult * 1.5

    return sl_mult, tp_mult, "; ".join(reasons) if reasons else "using base values"

# =============================================================================
# POSITION MANAGEMENT
# =============================================================================
def analyze_symbol(symbol: str) -> Optional[MarketAnalysis]:
    """Complete market analysis for a symbol"""
    df = get_market_data(symbol)
    if df is None:
        return None

    atr = calculate_atr(df, ATR_PERIOD)
    vol_regime = calculate_volatility_regime(df, atr)
    trend_strength, trend_dir = calculate_trend_metrics(df)
    entropy = calculate_entropy(df)

    analysis_dict = {
        'symbol': symbol,
        'atr': atr,
        'volatility_regime': vol_regime,
        'trend_direction': trend_dir,
        'trend_strength': trend_strength,
        'entropy': entropy
    }

    # Get LLM suggestion (falls back to rules if unavailable)
    sl_mult, tp_mult, reasoning = get_llm_sltp_suggestion(analysis_dict)

    return MarketAnalysis(
        symbol=symbol,
        timestamp=datetime.now(),
        atr=atr,
        volatility_regime=vol_regime,
        trend_strength=trend_strength,
        trend_direction=trend_dir,
        entropy_score=entropy,
        suggested_sl_mult=sl_mult,
        suggested_tp_mult=tp_mult,
        reasoning=reasoning
    )

def write_sltp_file(analyses: List[MarketAnalysis]):
    """Write SL/TP suggestions to file for EA to read"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "symbols": {}
    }

    for analysis in analyses:
        output["symbols"][analysis.symbol] = {
            "sl_mult": round(analysis.suggested_sl_mult, 2),
            "tp_mult": round(analysis.suggested_tp_mult, 2),
            "atr": round(analysis.atr, 6),
            "volatility": analysis.volatility_regime,
            "trend": analysis.trend_direction,
            "entropy": round(analysis.entropy_score, 3),
            "reasoning": analysis.reasoning
        }

    # Write to MQL5 common files folder
    mt5_data = mt5.terminal_info().data_path
    filepath = f"{mt5_data}\\MQL5\\Files\\llm_sltp_config.json"

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Updated SL/TP config: {filepath}")
    return filepath

# =============================================================================
# MAIN LOOP
# =============================================================================
def initialize_mt5(account: Dict) -> bool:
    """Initialize MT5 connection to specific account"""
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return False

    authorized = mt5.login(
        account["login"],
        password=account["password"],
        server=account["server"]
    )

    if not authorized:
        logger.error(f"Login failed for {account['login']}")
        return False

    logger.info(f"Connected to account {account['login']}")
    return True

def run_optimizer(interval_seconds: int = 60):
    """Main optimization loop"""
    logger.info("="*60)
    logger.info("LLM SL/TP Optimizer Starting")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Base SL: {BASE_SL_ATR_MULT}x ATR | Base TP: {BASE_TP_ATR_MULT}x ATR")
    logger.info(f"LLM Model: {OLLAMA_MODEL}")
    logger.info("="*60)

    # Use first account for data (all should have same market data)
    if not initialize_mt5(ACCOUNTS[0]):
        return

    try:
        while True:
            analyses = []

            for symbol in SYMBOLS:
                logger.info(f"Analyzing {symbol}...")
                analysis = analyze_symbol(symbol)

                if analysis:
                    analyses.append(analysis)
                    logger.info(
                        f"  {symbol}: Vol={analysis.volatility_regime}, "
                        f"Trend={analysis.trend_direction} ({analysis.trend_strength:.2f}), "
                        f"Entropy={analysis.entropy_score:.2f}"
                    )
                    logger.info(
                        f"  -> SL: {analysis.suggested_sl_mult:.2f}x ATR, "
                        f"TP: {analysis.suggested_tp_mult:.2f}x ATR"
                    )
                    logger.info(f"  -> Reason: {analysis.reasoning}")

            if analyses:
                write_sltp_file(analyses)

            logger.info(f"Next update in {interval_seconds}s")
            logger.info("-"*40)
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Optimizer stopped by user")
    finally:
        mt5.shutdown()

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import sys

    # Optional: specify update interval
    interval = 60
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except:
            pass

    run_optimizer(interval)
