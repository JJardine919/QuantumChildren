"""
QuantumChildren 3-Week Backtest Simulation
Runs historical M5 data through Archiver → Ollama → Signal pipeline
"""

import json
import subprocess
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import MetaTrader5 as mt5

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "symbols": ["BTCUSD", "ETHUSD"],
    "timeframe": mt5.TIMEFRAME_M5,
    "lookback_bars": 100,  # bars needed for indicators
    "ollama_model": "quantumchild",
    "min_fidelity": 0.85,
    "max_entropy": 4.5,
    "min_confidence": 70,
    "initial_balance": 10000,
    "risk_percent": 1.0,
}

# =============================================================================
# INDICATOR FUNCTIONS
# =============================================================================

def _ema(data: np.ndarray, period: int) -> float:
    if len(data) < period:
        return float(np.mean(data))
    multiplier = 2 / (period + 1)
    ema = float(data[0])
    for price in data[1:]:
        ema = (float(price) * multiplier) + (ema * (1 - multiplier))
    return ema

def calculate_indicators(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
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

def calculate_fidelity(close: np.ndarray) -> float:
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

def calculate_entropy(close: np.ndarray) -> float:
    returns = np.diff(close) / (close[:-1] + 1e-10)
    hist, _ = np.histogram(returns, bins=30, density=True)
    hist = hist + 1e-10
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist))
    return round(entropy, 2)

# =============================================================================
# OLLAMA QUERY (FAST MODE - SKIP THINKING OUTPUT)
# =============================================================================

def query_quantumchild_fast(symbol: str, indicators: Dict, fidelity: float, entropy: float) -> Optional[Dict]:
    """Query Ollama - returns signal dict or None"""
    input_str = f"{symbol}, M5, RSI={indicators['rsi']}, MACD={indicators['macd']}, MACD_signal={indicators['macd_signal']}, BB_upper={indicators['bb_upper']}, BB_lower={indicators['bb_lower']}, momentum={indicators['momentum']}, ROC={indicators['roc']}, ATR={indicators['atr']}, fidelity={fidelity}, entropy={entropy}"
    
    try:
        result = subprocess.run(
            ["ollama", "run", CONFIG["ollama_model"], input_str],
            capture_output=True,
            text=True,
            timeout=90,
            encoding='utf-8',
            errors='ignore'
        )
        output = result.stdout.strip()
        json_start = output.rfind('{')
        json_end = output.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(output[json_start:json_end])
        return None
    except:
        return None

# =============================================================================
# SIMULATED TRADE EXECUTION
# =============================================================================

class BacktestEngine:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades: List[Dict] = []
        self.open_positions: Dict[str, Dict] = {}
        
    def open_trade(self, symbol: str, signal: Dict, entry_price: float, timestamp: datetime, atr: float):
        if symbol in self.open_positions:
            return  # Already have position
        
        direction = signal["signal"]
        sl_pips = signal["sl_pips"]
        tp_pips = signal["tp_pips"]
        
        # Calculate SL/TP prices
        pip_value = 1.0 if "BTC" in symbol else 0.1  # Simplified
        
        if direction == "BUY":
            sl_price = entry_price - (sl_pips * pip_value)
            tp_price = entry_price + (tp_pips * pip_value)
        else:
            sl_price = entry_price + (sl_pips * pip_value)
            tp_price = entry_price - (tp_pips * pip_value)
        
        # Position sizing (1% risk)
        risk_amount = self.balance * (CONFIG["risk_percent"] / 100)
        position_size = risk_amount / (sl_pips * pip_value) if sl_pips > 0 else 0.01
        
        self.open_positions[symbol] = {
            "direction": direction,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "position_size": position_size,
            "entry_time": timestamp,
            "confidence": signal["confidence"],
            "reason": signal["reason"]
        }
        
    def check_exits(self, symbol: str, high: float, low: float, close: float, timestamp: datetime):
        if symbol not in self.open_positions:
            return
        
        pos = self.open_positions[symbol]
        exit_price = None
        exit_reason = None
        
        if pos["direction"] == "BUY":
            if low <= pos["sl_price"]:
                exit_price = pos["sl_price"]
                exit_reason = "SL"
            elif high >= pos["tp_price"]:
                exit_price = pos["tp_price"]
                exit_reason = "TP"
        else:  # SELL
            if high >= pos["sl_price"]:
                exit_price = pos["sl_price"]
                exit_reason = "SL"
            elif low <= pos["tp_price"]:
                exit_price = pos["tp_price"]
                exit_reason = "TP"
        
        if exit_price:
            # Calculate P&L
            pip_value = 1.0 if "BTC" in symbol else 0.1
            if pos["direction"] == "BUY":
                pips = (exit_price - pos["entry_price"]) / pip_value
            else:
                pips = (pos["entry_price"] - exit_price) / pip_value
            
            pnl = pips * pos["position_size"] * pip_value
            self.balance += pnl
            
            self.trades.append({
                "symbol": symbol,
                "direction": pos["direction"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "entry_time": pos["entry_time"],
                "exit_time": timestamp,
                "pips": round(pips, 1),
                "pnl": round(pnl, 2),
                "exit_reason": exit_reason,
                "confidence": pos["confidence"]
            })
            
            del self.open_positions[symbol]
    
    def get_stats(self) -> Dict:
        if not self.trades:
            return {"total_trades": 0}
        
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        
        total_pnl = sum(t["pnl"] for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        profit_factor = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else 0
        
        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "final_balance": round(self.balance, 2),
            "return_pct": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2)
        }

# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_backtest(weeks: int = 3, check_interval: int = 12):
    """
    Run backtest over historical data
    check_interval: check for signals every N bars (12 = hourly on M5)
    """
    print("\n" + "="*60)
    print("🧪 QUANTUMCHILDREN 3-WEEK BACKTEST")
    print("="*60)
    
    if not mt5.initialize():
        print("❌ MT5 init failed")
        return
    
    print(f"✅ MT5 connected")
    
    # Calculate bars needed (3 weeks of M5 = 3 * 7 * 24 * 12 = 6048 bars)
    bars_needed = weeks * 7 * 24 * 12 + CONFIG["lookback_bars"]
    
    engine = BacktestEngine(CONFIG["initial_balance"])
    signals_generated = 0
    signals_filtered = 0
    
    for symbol in CONFIG["symbols"]:
        print(f"\n📊 Loading {symbol} data...")
        
        rates = mt5.copy_rates_from_pos(symbol, CONFIG["timeframe"], 0, bars_needed)
        if rates is None or len(rates) < bars_needed:
            print(f"❌ Not enough data for {symbol}")
            continue
        
        print(f"✅ Loaded {len(rates)} bars")
        
        # Convert to arrays
        times = [datetime.fromtimestamp(r['time']) for r in rates]
        opens = np.array([r['open'] for r in rates])
        highs = np.array([r['high'] for r in rates])
        lows = np.array([r['low'] for r in rates])
        closes = np.array([r['close'] for r in rates])
        
        print(f"📅 Period: {times[CONFIG['lookback_bars']]} to {times[-1]}")
        
        # Walk forward through data
        total_checks = (len(rates) - CONFIG["lookback_bars"]) // check_interval
        print(f"🔄 Running {total_checks} signal checks...")
        
        for i in range(CONFIG["lookback_bars"], len(rates), check_interval):
            # Check open positions for exits on every bar
            for j in range(max(CONFIG["lookback_bars"], i - check_interval), i):
                engine.check_exits(symbol, highs[j], lows[j], closes[j], times[j])
            
            # Get window of data for analysis
            window_close = closes[i-CONFIG["lookback_bars"]:i]
            window_high = highs[i-CONFIG["lookback_bars"]:i]
            window_low = lows[i-CONFIG["lookback_bars"]:i]
            
            # Calculate archiver metrics
            fidelity = calculate_fidelity(window_close)
            entropy = calculate_entropy(window_close)
            
            # Filter check
            if fidelity < CONFIG["min_fidelity"] or entropy > CONFIG["max_entropy"]:
                signals_filtered += 1
                continue
            
            # Calculate indicators
            indicators = calculate_indicators(window_close, window_high, window_low)
            
            # Query Ollama
            signal = query_quantumchild_fast(symbol, indicators, fidelity, entropy)
            
            if signal and signal.get("signal") in ["BUY", "SELL"]:
                if signal.get("confidence", 0) >= CONFIG["min_confidence"]:
                    signals_generated += 1
                    engine.open_trade(
                        symbol=symbol,
                        signal=signal,
                        entry_price=closes[i],
                        timestamp=times[i],
                        atr=indicators["atr"]
                    )
                    print(f"  📈 {times[i].strftime('%m/%d %H:%M')} {signal['signal']} {symbol} @ {closes[i]:.2f} (conf:{signal['confidence']})")
        
        # Close any remaining positions at end
        if symbol in engine.open_positions:
            engine.check_exits(symbol, highs[-1], lows[-1], closes[-1], times[-1])
    
    mt5.shutdown()
    
    # Print results
    stats = engine.get_stats()
    
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    print(f"Period: 3 weeks")
    print(f"Signals filtered (noise): {signals_filtered}")
    print(f"Signals generated: {signals_generated}")
    print(f"Total trades: {stats['total_trades']}")
    print(f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}")
    print(f"Win rate: {stats.get('win_rate', 0)}%")
    print(f"Profit factor: {stats.get('profit_factor', 0)}")
    print(f"Avg win: ${stats.get('avg_win', 0)} | Avg loss: ${stats.get('avg_loss', 0)}")
    print(f"Total P&L: ${stats.get('total_pnl', 0)}")
    print(f"Return: {stats.get('return_pct', 0)}%")
    print(f"Final balance: ${stats.get('final_balance', CONFIG['initial_balance'])}")
    print("="*60)
    
    # Print trade log
    if engine.trades:
        print("\n📋 TRADE LOG:")
        for t in engine.trades:
            emoji = "✅" if t["pnl"] > 0 else "❌"
            print(f"  {emoji} {t['entry_time'].strftime('%m/%d %H:%M')} {t['direction']} {t['symbol']} | {t['exit_reason']} | {t['pips']} pips | ${t['pnl']}")

if __name__ == "__main__":
    run_backtest(weeks=3, check_interval=12)  # Check hourly
