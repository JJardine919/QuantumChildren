"""
QUANTUM BRAIN VPS - SIMPLE VERSION
===================================
No torch/LSTM - uses regime detection + momentum signals.
Runs on VPS via Wine Python.
"""

import sys
import os
import time
import zlib
import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum

import MetaTrader5 as mt5

# Configuration
CHECK_INTERVAL = 60
CLEAN_THRESHOLD = 0.95
CONFIDENCE_MIN = 0.6
BASE_LOT = 0.01

ACCOUNT_CONFIG = {
    'magic_number': 366001,
    'daily_loss_limit': 0.05,
    'max_drawdown': 0.10,
    'symbols': ['BTCUSD', 'ETHUSD', 'XAUUSD'],
}

class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class Regime(Enum):
    CLEAN = "CLEAN"
    VOLATILE = "VOLATILE"
    CHOPPY = "CHOPPY"

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")
    try:
        with open('quantum_brain_vps.log', 'a') as f:
            f.write(f"[{timestamp}] {msg}\n")
    except:
        pass

def analyze_regime(prices):
    """Regime detection via compression ratio"""
    data_bytes = prices.astype(np.float32).tobytes()
    compressed = zlib.compress(data_bytes, level=9)
    ratio = len(data_bytes) / len(compressed)

    if ratio >= 3.5:
        return Regime.CLEAN, 0.96
    elif ratio >= 2.5:
        return Regime.VOLATILE, 0.88
    else:
        return Regime.CHOPPY, 0.75

def get_signal(df):
    """Simple momentum signal"""
    data = df.copy()

    # EMAs
    data['ema_fast'] = data['close'].ewm(span=8).mean()
    data['ema_slow'] = data['close'].ewm(span=21).mean()

    # RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    data['rsi'] = 100 - (100 / (1 + rs))

    current = data.iloc[-1]
    prev = data.iloc[-2]

    # Crossover + RSI confirmation
    if prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']:
        if current['rsi'] < 70:  # Not overbought
            return Action.BUY, 0.7
    elif prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']:
        if current['rsi'] > 30:  # Not oversold
            return Action.SELL, 0.7

    return Action.HOLD, 0.5

def has_position(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            if pos.magic == magic:
                return True
    return False

def execute_trade(symbol, action):
    if action not in [Action.BUY, Action.SELL]:
        return False

    magic = ACCOUNT_CONFIG['magic_number']
    if has_position(symbol, magic):
        log(f"Already have position in {symbol}")
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    account = mt5.account_info()
    lot = max(0.01, round((account.balance / 100000) * 0.01, 2))
    lot = min(lot, 0.03)  # Hard cap - small lots for tighter stops

    # ATR for SL/TP
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 15)
    if rates is not None:
        df = pd.DataFrame(rates)
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
    else:
        atr = 100 * symbol_info.point

    if action == Action.BUY:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - (atr * 1.5)
        tp = price + (atr * 3.0)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + (atr * 1.5)
        tp = price - (atr * 3.0)

    filling = mt5.ORDER_FILLING_IOC

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": magic,
        "comment": "QB_VPS",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"TRADE: {action.name} {symbol} @ {price:.2f} SL:{sl:.2f} TP:{tp:.2f}")
        return True
    else:
        log(f"FAILED: {result.comment}")
        return False

def main():
    log("=" * 50)
    log("QUANTUM BRAIN VPS - SIMPLE")
    log("=" * 50)

    # VPS terminal path
    terminal_path = '/root/.wine/drive_c/Program Files/Blue Guardian MT5 Terminal/terminal64.exe'

    if not mt5.initialize(path=terminal_path):
        log(f"MT5 init failed: {mt5.last_error()}")
        return

    account = mt5.account_info()
    if account is None:
        log("Please login to MT5 first")
        mt5.shutdown()
        return

    starting_balance = account.balance
    log(f"Account: {account.login}, Balance: ${account.balance:,.2f}")
    log(f"Symbols: {ACCOUNT_CONFIG['symbols']}")
    log("=" * 50)

    try:
        while True:
            account = mt5.account_info()
            if account:
                loss = (starting_balance - account.balance) / starting_balance
                if loss >= ACCOUNT_CONFIG['daily_loss_limit']:
                    log(f"DAILY LOSS LIMIT: {loss*100:.1f}%")
                    break

            for symbol in ACCOUNT_CONFIG['symbols']:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 256)
                if rates is None or len(rates) < 100:
                    continue

                df = pd.DataFrame(rates)
                prices = df['close'].values

                regime, fidelity = analyze_regime(prices)

                if regime != Regime.CLEAN:
                    log(f"[{symbol}] {regime.value} ({fidelity:.2f}) - SKIP")
                    continue

                action, conf = get_signal(df)
                log(f"[{symbol}] CLEAN ({fidelity:.2f}) | {action.name} ({conf:.2f})")

                if action in [Action.BUY, Action.SELL] and conf >= CONFIDENCE_MIN:
                    execute_trade(symbol, action)

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        log("Stopped")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
