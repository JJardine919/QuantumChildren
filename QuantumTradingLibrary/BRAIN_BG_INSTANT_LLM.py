"""
QUANTUM BRAIN - BLUEGUARDIAN INSTANT (LLM-GATED)
==================================================
Claude API is the MANDATORY trade gatekeeper.
No LLM response = ZERO trades. Period.

Account: 366604 (BlueGuardian $5K Instant - PAYS REAL MONEY)

Architecture:
  Every 30s -> gather BTCUSD market data from MT5
           -> send to Claude API with trading rules
           -> Claude decides BUY / SELL / HOLD
           -> execute Claude's decision via MT5
           -> broker-side SL/TP on every position at entry

Run: python BRAIN_BG_INSTANT_LLM.py
"""

import sys
import os
import time
import json
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Tuple, Dict

import MetaTrader5 as mt5

# ============================================================
# CONFIG FROM MASTER_CONFIG.json - DO NOT HARDCODE
# ============================================================
from config_loader import (
    MAX_LOSS_DOLLARS,
    TP_MULTIPLIER,
    ROLLING_SL_MULTIPLIER,
    DYNAMIC_TP_PERCENT,
    SET_DYNAMIC_TP,
    ROLLING_SL_ENABLED,
    ATR_MULTIPLIER,
    CHECK_INTERVAL_SECONDS as CHECK_INTERVAL,
    CONFIDENCE_THRESHOLD,
)

from credential_manager import get_credentials, CredentialError

# ============================================================
# ANTHROPIC SDK
# ============================================================
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Load API key from .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][BG_LLM] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('brain_bg_instant_llm.log'),
        logging.StreamHandler()
    ]
)

# ============================================================
# ACCOUNT
# ============================================================
def _load_account():
    try:
        creds = get_credentials('BG_INSTANT')
        return {
            'account': creds['account'],
            'password': creds['password'],
            'server': creds['server'],
            'terminal_path': creds.get('terminal_path') or r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
            'name': 'BlueGuardian $5K Instant',
            'symbols': creds.get('symbols', ['BTCUSD']),
            'magic_number': creds.get('magic', 366001),
        }
    except CredentialError as e:
        logging.error(f"Failed to load BG_INSTANT credentials: {e}")
        sys.exit(1)

ACCOUNT = _load_account()


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


# ============================================================
# MARKET DATA GATHERING
# ============================================================

def gather_market_snapshot(symbol: str) -> Optional[Dict]:
    """Pull all market data from MT5 for Claude to analyze."""
    # M5 candles (last 100 = ~8 hours)
    rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
    if rates_m5 is None or len(rates_m5) < 30:
        return None

    df = pd.DataFrame(rates_m5)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Current tick
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    # ATR (14-period on M5)
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr_14 = tr.rolling(14).mean().iloc[-1]

    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss_s + 1e-10)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = (exp12 - exp26).iloc[-1]
    macd_signal = (exp12 - exp26).ewm(span=9, adjust=False).mean().iloc[-1]

    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean().iloc[-1]
    bb_std = df['close'].rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # EMA crossover state
    ema_8 = df['close'].ewm(span=8, adjust=False).mean().iloc[-1]
    ema_21 = df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
    ema_50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]

    # Volume analysis
    avg_vol = df['tick_volume'].rolling(20).mean().iloc[-1]
    current_vol = df['tick_volume'].iloc[-1]
    vol_ratio = current_vol / (avg_vol + 1e-10)

    # Recent price action (last 10 candles summary)
    recent = df.tail(10)
    recent_highs = recent['high'].values.tolist()
    recent_lows = recent['low'].values.tolist()
    recent_closes = recent['close'].values.tolist()
    recent_times = [t.strftime('%H:%M') for t in recent['time']]

    # M15 candles for bigger picture (last 20 = 5 hours)
    rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 20)
    m15_summary = []
    if rates_m15 is not None and len(rates_m15) > 0:
        df15 = pd.DataFrame(rates_m15)
        df15['time'] = pd.to_datetime(df15['time'], unit='s')
        for _, row in df15.tail(10).iterrows():
            m15_summary.append({
                'time': row['time'].strftime('%H:%M'),
                'open': round(float(row['open']), 2),
                'high': round(float(row['high']), 2),
                'low': round(float(row['low']), 2),
                'close': round(float(row['close']), 2),
            })

    # H1 candles for trend (last 24)
    rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 24)
    h1_trend = "unknown"
    if rates_h1 is not None and len(rates_h1) >= 10:
        df_h1 = pd.DataFrame(rates_h1)
        h1_close = df_h1['close'].values
        if h1_close[-1] > h1_close[-5] > h1_close[-10]:
            h1_trend = "bullish"
        elif h1_close[-1] < h1_close[-5] < h1_close[-10]:
            h1_trend = "bearish"
        else:
            h1_trend = "sideways"

    # Account info
    account_info = mt5.account_info()
    balance = account_info.balance if account_info else 0
    equity = account_info.equity if account_info else 0
    profit = account_info.profit if account_info else 0

    # Open positions for this account
    positions = mt5.positions_get(symbol=symbol)
    our_positions = []
    if positions:
        for p in positions:
            if p.magic == ACCOUNT['magic_number']:
                our_positions.append({
                    'ticket': p.ticket,
                    'type': 'BUY' if p.type == 0 else 'SELL',
                    'volume': p.volume,
                    'open_price': p.price_open,
                    'sl': p.sl,
                    'tp': p.tp,
                    'profit': round(p.profit, 2),
                })

    return {
        'symbol': symbol,
        'bid': tick.bid,
        'ask': tick.ask,
        'spread': round(tick.ask - tick.bid, 2),
        'atr_14': round(float(atr_14), 2),
        'rsi_14': round(float(rsi), 2),
        'macd': round(float(macd_line), 2),
        'macd_signal': round(float(macd_signal), 2),
        'bb_upper': round(float(bb_upper), 2),
        'bb_mid': round(float(bb_mid), 2),
        'bb_lower': round(float(bb_lower), 2),
        'ema_8': round(float(ema_8), 2),
        'ema_21': round(float(ema_21), 2),
        'ema_50': round(float(ema_50), 2),
        'volume_ratio': round(float(vol_ratio), 2),
        'h1_trend': h1_trend,
        'recent_m5': [
            {'time': t, 'high': round(h, 2), 'low': round(l, 2), 'close': round(c, 2)}
            for t, h, l, c in zip(recent_times, recent_highs, recent_lows, recent_closes)
        ],
        'm15_candles': m15_summary,
        'account_balance': round(balance, 2),
        'account_equity': round(equity, 2),
        'account_profit': round(profit, 2),
        'open_positions': our_positions,
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
    }


# ============================================================
# CLAUDE API - MANDATORY TRADE GATEKEEPER
# ============================================================

SYSTEM_PROMPT = f"""You are a professional BTCUSD trading analyst for a $5,000 funded account.
Your job: analyze market data and decide whether to BUY, SELL, or HOLD.

CRITICAL RULES:
- Risk per trade: ${MAX_LOSS_DOLLARS} max loss (broker-side SL set at entry)
- TP target: {TP_MULTIPLIER}x the SL distance (${MAX_LOSS_DOLLARS * TP_MULTIPLIER} reward)
- You need at least 33% win rate to be profitable at 3:1 R:R
- HOLD is the default. Only trade when you see a CLEAR setup.
- If price is chopping sideways with no clear direction: HOLD
- If volume is low and range is tight: HOLD
- If indicators conflict (RSI says one thing, MACD another): HOLD
- Only take trades where multiple signals align
- Consider the H1 trend - trading WITH the trend is safer

WHAT MAKES A GOOD TRADE:
- Clear trend on H1 (not sideways)
- Price breaking out of a range with volume
- RSI not at extremes (not overbought/oversold against the trade direction)
- MACD confirming the direction
- EMAs aligned (8 > 21 > 50 for buys, reversed for sells)
- Price near a Bollinger Band edge in trend direction
- Volume above average (vol_ratio > 1.2)

WHAT TO AVOID:
- Trading in chop/consolidation (this destroyed 36 out of 38 trades yesterday)
- Fading strong trends
- Trading when spread is abnormally wide
- Trading against H1 trend
- Low volume environments

RESPOND IN EXACTLY THIS JSON FORMAT:
{{"action": "HOLD", "confidence": 0.0, "reasoning": "brief explanation"}}

Where:
- action: "BUY", "SELL", or "HOLD"
- confidence: 0.0 to 1.0 (only trade above {CONFIDENCE_THRESHOLD})
- reasoning: 1-2 sentence explanation

NO OTHER TEXT. JUST THE JSON."""


def ask_claude(market_data: Dict) -> Tuple[Action, float, str]:
    """Ask Claude API for a trade decision. Returns (action, confidence, reasoning).
    If Claude is unavailable or doesn't respond: returns HOLD with 0 confidence.
    This is the MANDATORY gatekeeper - no Claude = no trades."""

    if not ANTHROPIC_AVAILABLE:
        logging.error("GATEKEEPER: anthropic package not installed. NO TRADES ALLOWED.")
        return Action.HOLD, 0.0, "anthropic SDK not installed"

    if not ANTHROPIC_API_KEY:
        logging.error("GATEKEEPER: ANTHROPIC_API_KEY not set. NO TRADES ALLOWED.")
        return Action.HOLD, 0.0, "API key not configured"

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        user_msg = f"Current market data:\n{json.dumps(market_data, indent=2)}"

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()

        # Parse JSON from response (handle markdown code blocks)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        decision = json.loads(raw)

        action_str = decision.get("action", "HOLD").upper()
        confidence = float(decision.get("confidence", 0.0))
        reasoning = decision.get("reasoning", "")

        if action_str == "BUY":
            action = Action.BUY
        elif action_str == "SELL":
            action = Action.SELL
        else:
            action = Action.HOLD

        return action, confidence, reasoning

    except json.JSONDecodeError as e:
        logging.warning(f"GATEKEEPER: Claude response not valid JSON: {e}")
        logging.warning(f"Raw response: {raw[:200]}")
        return Action.HOLD, 0.0, f"JSON parse error: {e}"
    except anthropic.APIError as e:
        logging.error(f"GATEKEEPER: Claude API error: {e}")
        return Action.HOLD, 0.0, f"API error: {e}"
    except Exception as e:
        logging.error(f"GATEKEEPER: Unexpected error calling Claude: {e}")
        return Action.HOLD, 0.0, f"Error: {e}"


# ============================================================
# TRADE EXECUTION
# ============================================================

def execute_trade(symbol: str, action: Action, confidence: float, reasoning: str) -> bool:
    """Execute a trade with broker-side SL/TP. Returns True if filled."""
    if action not in [Action.BUY, Action.SELL]:
        return False

    # Check if we already have a position
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for p in positions:
            if p.magic == ACCOUNT['magic_number']:
                logging.info(f"Already have {symbol} position #{p.ticket} - skipping")
                return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    point = symbol_info.point
    stops_level = symbol_info.trade_stops_level

    # ATR-based SL distance, lot sized for $1 max loss
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
    if rates is not None and len(rates) >= 14:
        df_atr = pd.DataFrame(rates)
        tr = np.maximum(
            df_atr['high'] - df_atr['low'],
            np.maximum(
                abs(df_atr['high'] - df_atr['close'].shift(1)),
                abs(df_atr['low'] - df_atr['close'].shift(1))
            )
        )
        atr = tr.rolling(14).mean().iloc[-1]
        sl_distance = atr * ATR_MULTIPLIER
    else:
        sl_distance = 50 * point

    # Respect broker minimum stop level
    min_sl_distance = (stops_level + 10) * point
    if sl_distance < min_sl_distance:
        sl_distance = min_sl_distance

    # Lot size: MAX_LOSS_DOLLARS / (sl_ticks * tick_value)
    sl_ticks = sl_distance / tick_size
    if tick_value > 0 and sl_ticks > 0:
        lot = MAX_LOSS_DOLLARS / (sl_ticks * tick_value)
    else:
        lot = symbol_info.volume_min

    lot = max(symbol_info.volume_min, lot)
    lot = min(lot, symbol_info.volume_max)
    lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
    lot = max(symbol_info.volume_min, lot)

    tp_distance = sl_distance * TP_MULTIPLIER

    if action == Action.BUY:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - sl_distance
        tp = price + tp_distance
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + sl_distance
        tp = price - tp_distance

    digits = symbol_info.digits
    sl = round(sl, digits)
    tp = round(tp, digits)
    price = round(price, digits)

    filling_mode = mt5.ORDER_FILLING_IOC
    if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
        filling_mode = mt5.ORDER_FILLING_FOK

    # Truncate reasoning for MT5 comment (max 31 chars)
    comment = f"LLM_{action.name}_{confidence:.0%}"[:31]

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": ACCOUNT['magic_number'],
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logging.info(
            f"TRADE EXECUTED: {action.name} {symbol} @ {price:.2f} | "
            f"Lot: {lot} | SL: {sl:.2f} (${MAX_LOSS_DOLLARS}) | TP: {tp:.2f} (${MAX_LOSS_DOLLARS * TP_MULTIPLIER}) | "
            f"Confidence: {confidence:.2f} | Reason: {reasoning}"
        )
        return True
    else:
        rc = result.retcode if result else 'None'
        comment_err = result.comment if result else 'None'
        logging.error(f"TRADE FAILED: retcode={rc} | {comment_err}")
        return False


# ============================================================
# POSITION MANAGEMENT (rolling SL + dynamic TP)
# ============================================================

def manage_positions():
    """Active position management: rolling SL + dynamic TP.
    This runs EVERY cycle regardless of Claude - protects existing trades."""
    positions = mt5.positions_get()
    if not positions:
        return

    for pos in positions:
        if pos.magic != ACCOUNT['magic_number']:
            continue

        symbol_info = mt5.symbol_info(pos.symbol)
        if symbol_info is None:
            continue

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            continue

        digits = symbol_info.digits
        point = symbol_info.point
        stops_level = symbol_info.trade_stops_level
        min_distance = (stops_level + 10) * point

        if pos.type == 0:  # BUY
            current_price = tick.bid
            entry_sl_distance = pos.price_open - pos.sl if pos.sl > 0 else 0
            current_profit_distance = current_price - pos.price_open
        else:  # SELL
            current_price = tick.ask
            entry_sl_distance = pos.sl - pos.price_open if pos.sl > 0 else 0
            current_profit_distance = pos.price_open - current_price

        if entry_sl_distance <= 0:
            continue

        tp_target_distance = entry_sl_distance * TP_MULTIPLIER
        dynamic_tp_threshold = tp_target_distance * (DYNAMIC_TP_PERCENT / 100.0)

        # Dynamic TP: close at threshold
        if SET_DYNAMIC_TP and current_profit_distance >= dynamic_tp_threshold:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if pos.type == 0 else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": round(close_price, digits),
                "magic": ACCOUNT['magic_number'],
                "comment": "DYN_TP",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"DYNAMIC TP: Closed {pos.symbol} #{pos.ticket} | Profit: ${pos.profit:.2f}")
            continue

        # Rolling SL
        if ROLLING_SL_ENABLED and current_profit_distance > 0:
            rolled_sl_distance = entry_sl_distance / ROLLING_SL_MULTIPLIER

            if pos.type == 0:  # BUY
                new_sl = current_price - rolled_sl_distance
                new_sl = max(new_sl, pos.price_open)
                if new_sl - tick.bid > -min_distance:
                    new_sl = tick.bid - min_distance
                if pos.sl > 0 and new_sl <= pos.sl:
                    continue
                risk_from_sl = current_price - new_sl
                new_tp = current_price + (risk_from_sl * TP_MULTIPLIER)
                new_tp = max(new_tp, pos.tp) if pos.tp > 0 else new_tp
            else:  # SELL
                new_sl = current_price + rolled_sl_distance
                new_sl = min(new_sl, pos.price_open)
                if tick.ask - new_sl > -min_distance:
                    new_sl = tick.ask + min_distance
                if pos.sl > 0 and new_sl >= pos.sl:
                    continue
                risk_from_sl = new_sl - current_price
                new_tp = current_price - (risk_from_sl * TP_MULTIPLIER)
                new_tp = min(new_tp, pos.tp) if pos.tp > 0 else new_tp

            new_sl = round(new_sl, digits)
            new_tp = round(new_tp, digits)

            if new_sl != round(pos.sl, digits) or new_tp != round(pos.tp, digits):
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": pos.symbol,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": new_tp,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"TRAIL: {pos.symbol} #{pos.ticket} SL -> {new_sl:.2f} | TP -> {new_tp:.2f}")


# ============================================================
# TRADE JOURNAL (local JSON log of every decision)
# ============================================================

JOURNAL_PATH = Path(__file__).parent / 'llm_trade_journal.jsonl'

def log_decision(market_data: Dict, action: Action, confidence: float, reasoning: str, executed: bool):
    """Log every Claude decision to a JSONL file for review."""
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': market_data.get('symbol', 'BTCUSD'),
        'bid': market_data.get('bid', 0),
        'ask': market_data.get('ask', 0),
        'atr': market_data.get('atr_14', 0),
        'rsi': market_data.get('rsi_14', 0),
        'h1_trend': market_data.get('h1_trend', ''),
        'action': action.name,
        'confidence': confidence,
        'reasoning': reasoning,
        'executed': executed,
        'balance': market_data.get('account_balance', 0),
    }
    try:
        with open(JOURNAL_PATH, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass


# ============================================================
# MAIN LOOP
# ============================================================

class LLMTrader:
    def __init__(self):
        self.connected = False
        self.cycle_count = 0
        self.trades_today = 0
        self.holds_today = 0

    def connect(self) -> bool:
        if self.connected:
            if mt5.terminal_info() is not None:
                acc = mt5.account_info()
                if acc and acc.login == ACCOUNT['account']:
                    return True
            self.connected = False

        if not mt5.initialize(path=ACCOUNT['terminal_path']):
            logging.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        if not mt5.login(ACCOUNT['account'], password=ACCOUNT['password'], server=ACCOUNT['server']):
            logging.error(f"Login failed: {mt5.last_error()}")
            return False

        acc = mt5.account_info()
        if acc:
            logging.info(f"CONNECTED: {ACCOUNT['name']} | Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f}")
            self.connected = True
            return True

        return False

    def run_cycle(self):
        self.cycle_count += 1

        if not self.connect():
            logging.error("Connection lost - will retry next cycle")
            return

        # Always manage existing positions (rolling SL, dynamic TP)
        manage_positions()

        for symbol in ACCOUNT['symbols']:
            # Gather market data
            snapshot = gather_market_snapshot(symbol)
            if snapshot is None:
                logging.warning(f"[{symbol}] Could not gather market data")
                continue

            # Ask Claude - MANDATORY gatekeeper
            action, confidence, reasoning = ask_claude(snapshot)

            # Log the decision
            executed = False

            if action == Action.HOLD:
                self.holds_today += 1
                logging.info(
                    f"[{symbol}] HOLD ({confidence:.2f}) | "
                    f"RSI: {snapshot['rsi_14']} | H1: {snapshot['h1_trend']} | "
                    f"Reason: {reasoning}"
                )
            elif confidence < CONFIDENCE_THRESHOLD:
                logging.info(
                    f"[{symbol}] {action.name} rejected - confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD} | "
                    f"Reason: {reasoning}"
                )
                action = Action.HOLD
            else:
                logging.info(
                    f"[{symbol}] Claude says {action.name} ({confidence:.2f}) | Reason: {reasoning}"
                )
                executed = execute_trade(symbol, action, confidence, reasoning)
                if executed:
                    self.trades_today += 1

            log_decision(snapshot, action, confidence, reasoning, executed)

        # Periodic status
        if self.cycle_count % 20 == 0:
            acc = mt5.account_info()
            if acc:
                logging.info(
                    f"STATUS: Cycle #{self.cycle_count} | "
                    f"Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f} | "
                    f"Trades today: {self.trades_today} | Holds: {self.holds_today}"
                )


def main():
    # Preflight checks
    if not ANTHROPIC_AVAILABLE:
        print("ERROR: 'anthropic' package not installed.")
        print("Run: pip install anthropic")
        sys.exit(1)

    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Add to .env file: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    print("=" * 60)
    print("  BLUEGUARDIAN INSTANT - LLM-GATED TRADING")
    print("=" * 60)
    print(f"  Account:    {ACCOUNT['account']} ({ACCOUNT['name']})")
    print(f"  Magic:      {ACCOUNT['magic_number']}")
    print(f"  Symbols:    {', '.join(ACCOUNT['symbols'])}")
    print(f"  SL:         ${MAX_LOSS_DOLLARS} per trade")
    print(f"  TP:         {TP_MULTIPLIER}x SL (${MAX_LOSS_DOLLARS * TP_MULTIPLIER})")
    print(f"  Dynamic TP: {DYNAMIC_TP_PERCENT}% partial close")
    print(f"  Rolling SL: {ROLLING_SL_MULTIPLIER}x")
    print(f"  Confidence: >= {CONFIDENCE_THRESHOLD}")
    print(f"  Interval:   {CHECK_INTERVAL}s")
    print(f"  Gatekeeper: Claude (sonnet-4-5)")
    print()
    print("  NO LLM RESPONSE = NO TRADES")
    print("=" * 60)

    trader = LLMTrader()

    try:
        while True:
            try:
                trader.run_cycle()
            except Exception as e:
                logging.error(f"Cycle error: {e}")
                logging.error(traceback.format_exc())
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown complete")


if __name__ == "__main__":
    main()
