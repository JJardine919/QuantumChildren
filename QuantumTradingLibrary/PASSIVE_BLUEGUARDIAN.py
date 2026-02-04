import sys
import os
import json
import numpy as np
import MetaTrader5 as mt5
import time
from datetime import datetime
import pandas as pd

# Add the project modules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'ETARE_QuantumFusion', 'modules'))

try:
    from signal_fusion import SignalFusionEngine
except ImportError:
    print("Warning: SignalFusionEngine not available, using simple strategy")
    SignalFusionEngine = None

# ==============================================================================
# CONFIGURATION - BLUE GUARDIAN $100K CHALLENGE
# ==============================================================================

CONFIG = {
    'account': 365060,
    'password': ')8xaE(gAuU',
    'server': 'BlueGuardian-Server',
    'terminal_path': r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",

    'symbol': 'BTCUSD',
    'timeframe': mt5.TIMEFRAME_M5,
    'magic_number': 365001,

    'volume': 0.01,
    'risk_multiplier': 1.5,
    'tp_ratio': 3.0,
    'be_percent': 0.5,

    'fusion_config': os.path.join(script_dir, 'ETARE_QuantumFusion', 'config', 'config.yaml'),
    'veto_threshold': 0.15,
}

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"BG_365060 >> [{timestamp}] {msg}")

def manage_existing_positions():
    """Handles Break-Even logic for open positions"""
    positions = mt5.positions_get(symbol=CONFIG['symbol'])
    if not positions:
        return

    for pos in positions:
        if pos.magic != CONFIG['magic_number']:
            continue

        entry = pos.price_open
        current = pos.price_current
        tp = pos.tp
        sl = pos.sl

        if tp == 0:
            continue

        total_dist = abs(tp - entry)
        current_dist = abs(current - entry)
        progress = current_dist / total_dist if total_dist > 0 else 0

        if progress >= CONFIG['be_percent']:
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            is_at_be = (is_buy and sl >= entry) or (not is_buy and sl <= entry)

            if not is_at_be:
                log(f"BE TRIGGER: Moving SL to entry for Ticket {pos.ticket} ({progress:.1%} progress)")
                point = mt5.symbol_info(CONFIG['symbol']).point
                new_sl = entry + (50 * point if is_buy else -50 * point)

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": tp
                }
                res = mt5.order_send(request)
                if res.retcode != mt5.TRADE_RETCODE_DONE:
                    log(f"BE Failed: {res.comment}")

def get_atr(symbol, timeframe, period=14):
    """Calculate ATR"""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
    if rates is None or len(rates) < period:
        return None

    df = pd.DataFrame(rates)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    return df['tr'].rolling(period).mean().iloc[-1]

def check_for_signal():
    """Check for trading signals"""
    rates = mt5.copy_rates_from_pos(CONFIG['symbol'], CONFIG['timeframe'], 0, 50)
    if rates is None or len(rates) < 50:
        return None

    df = pd.DataFrame(rates)

    # Simple momentum strategy
    df['ema_fast'] = df['close'].ewm(span=8).mean()
    df['ema_slow'] = df['close'].ewm(span=21).mean()

    current = df.iloc[-1]
    prev = df.iloc[-2]

    # Crossover signals
    if prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']:
        return 'BUY'
    elif prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']:
        return 'SELL'

    return None

def execute_trade(signal):
    """Execute a trade"""
    symbol_info = mt5.symbol_info(CONFIG['symbol'])
    if symbol_info is None:
        log(f"Symbol {CONFIG['symbol']} not found")
        return False

    if not symbol_info.visible:
        mt5.symbol_select(CONFIG['symbol'], True)

    atr = get_atr(CONFIG['symbol'], CONFIG['timeframe'])
    if atr is None:
        log("Could not calculate ATR")
        return False

    price = mt5.symbol_info_tick(CONFIG['symbol'])
    point = symbol_info.point

    if signal == 'BUY':
        order_type = mt5.ORDER_TYPE_BUY
        entry_price = price.ask
        sl = entry_price - (atr * CONFIG['risk_multiplier'])
        tp = entry_price + (atr * CONFIG['risk_multiplier'] * CONFIG['tp_ratio'])
    else:
        order_type = mt5.ORDER_TYPE_SELL
        entry_price = price.bid
        sl = entry_price + (atr * CONFIG['risk_multiplier'])
        tp = entry_price - (atr * CONFIG['risk_multiplier'] * CONFIG['tp_ratio'])

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": CONFIG['symbol'],
        "volume": CONFIG['volume'],
        "type": order_type,
        "price": entry_price,
        "sl": sl,
        "tp": tp,
        "magic": CONFIG['magic_number'],
        "comment": "BG_Quantum",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"TRADE OPENED: {signal} @ {entry_price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        return True
    else:
        log(f"TRADE FAILED: {result.comment}")
        return False

def has_open_position():
    """Check if we already have an open position"""
    positions = mt5.positions_get(symbol=CONFIG['symbol'])
    if positions:
        for pos in positions:
            if pos.magic == CONFIG['magic_number']:
                return True
    return False

def main():
    log("=" * 50)
    log("BLUE GUARDIAN $100K CHALLENGE - QUANTUM FUSION")
    log("=" * 50)

    if not mt5.initialize(path=CONFIG['terminal_path']):
        log(f"MT5 Init Failed: {mt5.last_error()}")
        return

    if not mt5.login(CONFIG['account'], password=CONFIG['password'], server=CONFIG['server']):
        log(f"Login Failed: {mt5.last_error()}")
        mt5.shutdown()
        return

    acc = mt5.account_info()
    log(f"CONNECTED: Account {acc.login} | Balance: ${acc.balance:.2f}")
    log(f"Target Symbol: {CONFIG['symbol']} | Magic: {CONFIG['magic_number']}")
    log("=" * 50)

    while True:
        try:
            # Manage existing positions (break-even)
            manage_existing_positions()

            # Check for new signals if no open position
            if not has_open_position():
                signal = check_for_signal()
                if signal:
                    log(f"SIGNAL: {signal}")
                    execute_trade(signal)

            time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            log("Shutting down...")
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(60)

    mt5.shutdown()
    log("Disconnected")

if __name__ == '__main__':
    main()
