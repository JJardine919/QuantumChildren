#!/usr/bin/env python3
"""
BLUE GUARDIAN Account 365060 - VPS Trading Bot
Direct rpyc connection to Wine MT5 (bypasses mt5linux wrapper issues)
"""
import sys
import os
import numpy as np
import time
from datetime import datetime
import pandas as pd
import rpyc

# For VPS/Wine environment
os.environ['DISPLAY'] = ':99'

# Import credentials securely
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from credential_manager import get_credentials, CredentialError

# ==============================================================================
# CONFIGURATION - BLUE GUARDIAN $100K CHALLENGE (VPS)
# ==============================================================================

def _load_config():
    """Load config with credentials from .env file."""
    try:
        creds = get_credentials('BG_CHALLENGE')
        return {
            'account': creds['account'],
            'password': creds['password'],
            'server': creds['server'],
            'terminal_path': r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
            'symbol': 'BTCUSD',
            'magic_number': 365001,
            'volume': 0.01,
            'risk_multiplier': 1.5,
            'tp_ratio': 3.0,
            'be_percent': 0.5,
        }
    except CredentialError as e:
        print(f"Failed to load BG_CHALLENGE credentials: {e}")
        print("Please configure BG_CHALLENGE_PASSWORD in .env file")
        sys.exit(1)

CONFIG = _load_config()

# Global MT5 connection
mt5 = None
conn = None

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] BG_365060 >> {msg}", flush=True)

def connect_mt5():
    """Connect to MT5 via rpyc"""
    global mt5, conn

    log("Connecting to rpyc server...")
    # Increase timeout for slow MT5 operations
    rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout'] = 120
    conn = rpyc.classic.connect('localhost', 18812)
    mt5 = conn.modules.MetaTrader5
    log("Connected to rpyc server")

    log(f"Initializing MT5 with path: {CONFIG['terminal_path']}")
    if not mt5.initialize(path=CONFIG['terminal_path']):
        log(f"MT5 Init Failed: {mt5.last_error()}")
        log("Trying without path...")
        if not mt5.initialize():
            log(f"MT5 Init Failed again: {mt5.last_error()}")
            return False

    log(f"Logging into account {CONFIG['account']}...")
    if not mt5.login(CONFIG['account'], password=CONFIG['password'], server=CONFIG['server']):
        log(f"Login Failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    return True

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
                log(f"BE TRIGGER: Moving SL to entry for Ticket {pos.ticket}")
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

def get_atr(symbol, period=14):
    """Calculate ATR"""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, period + 1)
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
    """Check for trading signals - EMA crossover"""
    rates = mt5.copy_rates_from_pos(CONFIG['symbol'], mt5.TIMEFRAME_M5, 0, 50)
    if rates is None or len(rates) < 50:
        return None

    df = pd.DataFrame(rates)
    df['ema_fast'] = df['close'].ewm(span=8).mean()
    df['ema_slow'] = df['close'].ewm(span=21).mean()

    current = df.iloc[-1]
    prev = df.iloc[-2]

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

    atr = get_atr(CONFIG['symbol'])
    if atr is None:
        log("Could not calculate ATR")
        return False

    price = mt5.symbol_info_tick(CONFIG['symbol'])

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
        "comment": "BG_VPS",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"TRADE: {signal} @ {entry_price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
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
    log("=" * 60)
    log("BLUE GUARDIAN $100K - VPS MODE (Direct rpyc)")
    log("=" * 60)

    if not connect_mt5():
        log("Failed to connect to MT5. Exiting.")
        return

    acc = mt5.account_info()
    log(f"CONNECTED: Account {acc.login} | Balance: ${acc.balance:.2f}")
    log(f"Symbol: {CONFIG['symbol']} | Magic: {CONFIG['magic_number']}")
    log("=" * 60)

    # Main loop
    while True:
        try:
            manage_existing_positions()

            if not has_open_position():
                signal = check_for_signal()
                if signal:
                    log(f"SIGNAL: {signal}")
                    execute_trade(signal)

            time.sleep(60)

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
