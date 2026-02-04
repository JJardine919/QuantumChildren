"""
QuantumChildren Trading Bridge
MT5 → Archiver (fidelity filter) → Ollama quantumchild → JSON signal → MT5 execution
"""

import json
import subprocess
import time
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
import MetaTrader5 as mt5
import atexit

# Register cleanup on script exit
atexit.register(mt5.shutdown)

CONFIG = {
    "symbols": ["BTCUSD", "ETHUSD"],
    "timeframe": mt5.TIMEFRAME_M5,
    "bars_to_fetch": 100,
    "ollama_model": "quantumchild",
    "min_fidelity": 0.60,
    "max_entropy": 4.5,
    "high_confidence_fidelity": 0.95,
    "high_confidence_entropy": 2.5,
    "min_confidence": 70,
    "max_positions_per_symbol": 1,
    "risk_percent": 1.0,
}

def connect_mt5() -> bool:
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    print(f"MT5 connected: {mt5.terminal_info().name}")
    return True

def get_market_data(symbol: str, timeframe: int, bars: int) -> Optional[np.ndarray]:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"❌ Failed to get rates for {symbol}")
        return None
    return rates

def _ema(data: np.ndarray, period: int) -> float:
    if len(data) < period:
        return float(np.mean(data))
    multiplier = 2 / (period + 1)
    ema = float(data[0])
    for price in data[1:]:
        ema = (float(price) * multiplier) + (ema * (1 - multiplier))
    return ema

def calculate_indicators(rates: np.ndarray) -> Dict[str, float]:
    close = np.array([r['close'] for r in rates])
    high = np.array([r['high'] for r in rates])
    low = np.array([r['low'] for r in rates])
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    
    # Build MACD history for signal line
    macd_history = []
    for i in range(26, len(close)):
        e12 = _ema(close[:i+1], 12)
        e26 = _ema(close[:i+1], 26)
        macd_history.append(e12 - e26)
    macd_signal = _ema(np.array(macd_history[-9:]), 9) if len(macd_history) >= 9 else macd_line
    
    sma20 = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    bb_upper = sma20 + (2 * std20)
    bb_lower = sma20 - (2 * std20)
    
    momentum = close[-1] - close[-11] if len(close) > 10 else 0
    roc = ((close[-1] - close[-11]) / close[-11]) * 100 if len(close) > 10 else 0
    
    tr_list = []
    for i in range(1, min(15, len(close))):
        tr = max(high[-i] - low[-i], abs(high[-i] - close[-i-1]), abs(low[-i] - close[-i-1]))
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else 0
    
    return {
        "rsi": round(float(rsi), 2),
        "macd": round(float(macd_line), 4),
        "macd_signal": round(float(macd_signal), 4),
        "bb_upper": round(float(bb_upper), 2),
        "bb_lower": round(float(bb_lower), 2),
        "momentum": round(float(momentum), 2),
        "roc": round(float(roc), 2),
        "atr": round(float(atr), 2),
        "current_price": round(float(close[-1]), 2)
    }

def calculate_fidelity(rates: np.ndarray) -> float:
    close = np.array([r['close'] for r in rates])
    normalized = (close - np.min(close)) / (np.max(close) - np.min(close) + 1e-10)
    autocorr = np.correlate(normalized - np.mean(normalized), normalized - np.mean(normalized), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)
    structure_score = np.mean(np.abs(autocorr[1:20]))
    diff = np.diff(normalized)
    noise_ratio = np.std(diff) / (np.std(normalized) + 1e-10)
    noise_penalty = max(0, 1 - noise_ratio)
    fidelity = (structure_score * 0.6 + noise_penalty * 0.4)
    fidelity = min(1.0, max(0.0, fidelity * 1.2))
    return round(fidelity, 3)

def calculate_entropy(rates: np.ndarray) -> float:
    close = np.array([r['close'] for r in rates])
    returns = np.diff(close) / close[:-1]
    hist, _ = np.histogram(returns, bins=50, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10)) / 10
    return round(max(0, min(8, entropy)), 2)

def archiver_filter(rates: np.ndarray) -> Dict[str, Any]:
    fidelity = calculate_fidelity(rates)
    entropy = calculate_entropy(rates)
    passes_filter = (fidelity >= CONFIG["min_fidelity"] and entropy <= CONFIG["max_entropy"])
    high_confidence = (fidelity >= CONFIG["high_confidence_fidelity"] and entropy <= CONFIG["high_confidence_entropy"])
    return {"fidelity": fidelity, "entropy": entropy, "passes_filter": passes_filter, "high_confidence": high_confidence}

def query_quantumchild(symbol: str, timeframe: str, indicators: Dict, fidelity: float, entropy: float) -> Optional[Dict]:
    input_str = f"{symbol}, {timeframe}, RSI={indicators['rsi']}, MACD={indicators['macd']}, MACD_signal={indicators['macd_signal']}, BB_upper={indicators['bb_upper']}, BB_lower={indicators['bb_lower']}, momentum={indicators['momentum']}, ROC={indicators['roc']}, ATR={indicators['atr']}, fidelity={fidelity}, entropy={entropy}"
    print(f"\n📊 Querying quantumchild...")
    try:
        result = subprocess.run(["ollama", "run", CONFIG["ollama_model"], input_str], capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        json_start = output.rfind('{')
        json_end = output.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            signal = json.loads(output[json_start:json_end])
            print(f"✅ Signal: {signal}")
            return signal
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def analyze_symbol(symbol: str) -> Optional[Dict]:
    print(f"\n{'='*50}\n📈 Analyzing {symbol}\n{'='*50}")
    rates = get_market_data(symbol, CONFIG["timeframe"], CONFIG["bars_to_fetch"])
    if rates is None:
        return None
    indicators = calculate_indicators(rates)
    print(f"📊 RSI={indicators['rsi']}, MACD={indicators['macd']}, ATR={indicators['atr']}")
    archiver = archiver_filter(rates)
    print(f"🔍 Fidelity={archiver['fidelity']}, Entropy={archiver['entropy']}")
    if not archiver["passes_filter"]:
        print(f"🚫 Filtered out")
        return None
    signal = query_quantumchild(symbol, "M5", indicators, archiver["fidelity"], archiver["entropy"])
    return signal

if __name__ == "__main__":
    print("\n🚀 QUANTUMCHILDREN BRIDGE - TEST MODE\n")
    if connect_mt5():
        for symbol in CONFIG["symbols"]:
            signal = analyze_symbol(symbol)
            if signal:
                print(f"\n📋 Signal: {json.dumps(signal, indent=2)}")
        mt5.shutdown()