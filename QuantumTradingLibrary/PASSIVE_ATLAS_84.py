import sys
import os
import json
import numpy as np
import MetaTrader5 as mt5
import time
from datetime import datetime
import pandas as pd
from credential_manager import get_credentials, CredentialError

# Add the project modules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'ETARE_QuantumFusion', 'modules'))

try:
    from signal_fusion import SignalFusionEngine
except ImportError:
    print("Error: Could not find SignalFusionEngine. Ensure modules are in ETARE_QuantumFusion/modules")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION - ATLAS FUNDED #1 (84)
# ==============================================================================

# Load credentials from credential_manager
atlas_creds = get_credentials('ATLAS')

CONFIG = {
    'account': atlas_creds['account'],
    'password': atlas_creds['password'],
    'server': atlas_creds['server'],
    'terminal_path': atlas_creds['terminal_path'],

    'symbol': 'BTCUSD',
    'timeframe': mt5.TIMEFRAME_M5,
    'magic_number': 84001,

    'volume': 0.01,
    'risk_multiplier': 1.5, # SL = 1.5x ATR
    'tp_ratio': 3.0,        # TP = 3x SL (1:3 RR)
    'be_percent': 0.5,      # Move to Break-Even at 50% of TP progress

    # Fusion Veto
    'fusion_config': os.path.join(script_dir, 'ETARE_QuantumFusion', 'config', 'config.yaml'),
    'veto_threshold': 0.15,
}

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"ATLAS_84 >> [{timestamp}] {msg}")

def manage_existing_positions():
    """Handles Break-Even logic for open positions"""
    positions = mt5.positions_get(symbol=CONFIG['symbol'], magic=CONFIG['magic_number'])
    if not positions:
        return

    for pos in positions:
        # Calculate current progress
        entry = pos.price_open
        current = pos.price_current
        tp = pos.tp
        sl = pos.sl
        
        if tp == 0: continue # Skip if no TP set
        
        total_dist = abs(tp - entry)
        current_dist = abs(current - entry)
        progress = current_dist / total_dist if total_dist > 0 else 0
        
        # If price moved 50% towards TP and SL is not already at/past entry
        if progress >= CONFIG['be_percent']:
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            is_at_be = (is_buy and sl >= entry) or (not is_buy and sl <= entry)
            
            if not is_at_be:
                log(f"BE TRIGGER: Moving SL to entry for Ticket {pos.ticket} ({progress:.1%} progress)")
                new_sl = entry + (50 * mt5.symbol_info(CONFIG['symbol']).point if is_buy else -50 * mt5.symbol_info(CONFIG['symbol']).point)
                
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": tp
                }
                res = mt5.order_send(request)
                if res.retcode != mt5.TRADE_RETCODE_DONE:
                    log(f"BE Failed: {res.comment}")

def main():
    log("Initializing Atlas Fusion Agent (V2.1 - BE Enabled)...")

    if not mt5.initialize(path=CONFIG['terminal_path']):
        log(f"MT5 Init Failed: {mt5.last_error()}")
        return

    if not mt5.login(CONFIG['account'], password=CONFIG['password'], server=CONFIG['server']):
        log(f"Login Failed: {mt5.last_error()}")
        mt5.shutdown()
        return

    acc = mt5.account_info()
    log(f"CONNECTED: Account {acc.login} | Balance: ${acc.balance:.2f}")

    fusion_engine = SignalFusionEngine(CONFIG['fusion_config'])
    log("Quantum Council Online. Standardized SL/TP Configured.")

    while True:
        try:
            if not mt5.terminal_info().connected:
                time.sleep(5); continue

            # --- MANAGE OPEN TRADES (BREAK-EVEN) ---
            manage_existing_positions()

            # Get latest price and ATR
            rates_m1 = mt5.copy_rates_from_pos(CONFIG['symbol'], mt5.TIMEFRAME_M1, 0, 20)
            tick = mt5.symbol_info_tick(CONFIG['symbol'])
            s_info = mt5.symbol_info(CONFIG['symbol'])
            
            if not rates_m1 or not tick or not s_info:
                time.sleep(1); continue

            # Calculate ATR for Dynamic Stops
            df_m1 = pd.DataFrame(rates_m1)
            high_low = df_m1['high'] - df_m1['low']
            atr = high_low.mean()
            sl_dist = max(atr * CONFIG['risk_multiplier'], 100.0 * s_info.point)
            
            # Basic Signal (M1 Price Position)
            current_open = rates_m1[-1]['open']
            raw_action = "BUY" if tick.bid > current_open else "SELL"
            
            # --- THE FUSION VETO ---
            fused = fusion_engine.get_fused_signal(CONFIG['symbol'], CONFIG['timeframe'])
            score = fused['composite_score']
            regime = fused['regime']['regime']
            
            log(f"Scan: {raw_action} | Score: {score:.4f} | ATR_SL: {sl_dist:.2f}")

            # Veto Logic
            trade_allowed = False
            if raw_action == "BUY" and score > -CONFIG['veto_threshold']:
                trade_allowed = True
            elif raw_action == "SELL" and score < CONFIG['veto_threshold']:
                trade_allowed = True

            # Check existing positions
            pos = mt5.positions_get(symbol=CONFIG['symbol'], magic=CONFIG['magic_number'])
            if pos is None: pos = []

            if len(pos) == 0 and trade_allowed:
                if not mt5.terminal_info().trade_allowed:
                    log("WARNING: AutoTrading is DISABLED in MT5.")
                    time.sleep(10); continue

                price = tick.ask if raw_action == "BUY" else tick.bid
                sl = price - sl_dist if raw_action == "BUY" else price + sl_dist
                tp = price + (sl_dist * CONFIG['tp_ratio']) if raw_action == "BUY" else price - (sl_dist * CONFIG['tp_ratio'])

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": CONFIG['symbol'],
                    "volume": CONFIG['volume'],
                    "type": mt5.ORDER_TYPE_BUY if raw_action == "BUY" else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "magic": CONFIG['magic_number'],
                    "comment": f"FUSION_V2.1_{regime[:3]}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                res = mt5.order_send(request)
                if res.retcode != mt5.TRADE_RETCODE_DONE:
                    log(f"Trade Failed: {res.comment}")
                else:
                    log(f"TRADE PLACED: {raw_action} @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")

            time.sleep(10) # Faster loop for Break-Even management

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Critical Loop Error: {e}")
            time.sleep(10)

    mt5.shutdown()

if __name__ == '__main__':
    main()