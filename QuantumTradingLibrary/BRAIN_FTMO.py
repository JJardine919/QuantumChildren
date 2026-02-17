"""
QUANTUM BRAIN - FTMO CHALLENGE ONLY
====================================
DEDICATED script for FTMO $200K Challenge account ONLY.
Runs independently - does NOT touch other accounts.

Account: 1521063483 (FTMO $200K Challenge)

FTMO-SPECIFIC OVERRIDES:
  - Risk per trade: $150 (not $1 - scaled for $200K account)
  - ATR multiplier: 0.5 (wider stops, lot adjusts to keep $150 risk)
  These are NOT in MASTER_CONFIG.json because they only apply to FTMO.

Run: python BRAIN_FTMO.py
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import MetaTrader5 as mt5

# ETARE numpy expert (preferred for BTCUSD)
try:
    from etare_expert import load_etare_expert, prepare_etare_features
    ETARE_AVAILABLE = True
except ImportError:
    ETARE_AVAILABLE = False

# QuantumChildren network collection
try:
    from entropy_collector import collect_signal, NODE_ID
    COLLECTION_ENABLED = True
except ImportError:
    COLLECTION_ENABLED = False
    NODE_ID = "LOCAL"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][FTMO] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('brain_ftmo.log'),
        logging.StreamHandler()
    ]
)

# ============================================================
# CONFIG FROM MASTER_CONFIG.json - DO NOT HARDCODE
# (except FTMO-specific overrides below)
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
    LSTM_MAX_AGE_DAYS,
    AGENT_SL_MIN,
    FTMO_DAILY_DD_PCT,
    FTMO_TOTAL_DD_PCT,
    FTMO_RISK_PER_TRADE as FTMO_RISK_PER_TRADE_CFG,
    FTMO_ATR_MULTIPLIER_CFG,
)

# ============================================================
# FTMO-SPECIFIC OVERRIDES
# The $200K FTMO account needs different risk sizing than the
# $5K Blue Guardian accounts. These values were determined by
# Monte Carlo simulation (CHALLENGE_SIMULATOR.py):
#   $150 risk + ATR=0.5 -> 99.4% pass rate, zero breaches
# ============================================================
FTMO_RISK_PER_TRADE = FTMO_RISK_PER_TRADE_CFG   # From config, default $150
FTMO_ATR_MULTIPLIER = FTMO_ATR_MULTIPLIER_CFG   # From config, default 0.5

# Import credentials securely - passwords from .env file
from credential_manager import get_credentials, CredentialError

# Process lock to prevent duplicate launches
from process_lock import ProcessLock

# Pre-launch validation
from prelaunch_validator import validate_prelaunch

# TEQA quantum signal bridge
try:
    from teqa_bridge import TEQABridge
    TEQA_ENABLED = True
except ImportError:
    TEQA_ENABLED = False

# QNIF quantum-immune signal bridge (Hybrid Veto authority)
try:
    from qnif_bridge import QNIFBridge
    QNIF_ENABLED = True
except ImportError:
    QNIF_ENABLED = False

# ============================================================
# SINGLE ACCOUNT - Credentials from .env via credential_manager
# ============================================================

def _load_account():
    """Load FTMO account with credentials from .env file."""
    try:
        creds = get_credentials('FTMO')
        return {
            'account': creds['account'],
            'password': creds['password'],
            'server': creds['server'],
            'terminal_path': creds.get('terminal_path') or r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
            'name': 'FTMO $200K Challenge',
            'symbols': creds.get('symbols', ['BTCUSD', 'XAUUSD', 'ETHUSD']),
            'magic_number': creds.get('magic', 152001),
        }
    except CredentialError as e:
        logging.error(f"Failed to load FTMO credentials: {e}")
        logging.error("Please configure FTMO_PASSWORD in .env file")
        sys.exit(1)

ACCOUNT = _load_account()


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


# Regime enum and detect_regime from quantum bridge
from quantum_regime_bridge import detect_regime, Regime


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(df: pd.DataFrame) -> np.ndarray:
    data = df.copy()
    for c in ['open', 'high', 'low', 'close', 'tick_volume']:
        if c in data.columns:
            data[c] = data[c].astype(float)

    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    data['bb_middle'] = data['close'].rolling(20).mean()
    data['bb_std'] = data['close'].rolling(20).std()
    data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
    data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']

    data['momentum'] = data['close'] / data['close'].shift(10)
    data['roc'] = data['close'].pct_change(10) * 100

    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr'] = data['tr'].rolling(14).mean()

    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
    data = data.dropna()

    if len(data) == 0:
        return np.zeros((1, len(feature_cols)))

    for col in feature_cols:
        col_mean = data[col].mean()
        col_std = data[col].std() + 1e-8
        data[col] = (data[col] - col_mean) / col_std

    data = data.fillna(0)
    return data[feature_cols].values


# ============================================================
# EXPERT LOADER
# ============================================================

class ExpertLoader:
    def __init__(self):
        script_dir = Path(__file__).parent.absolute()
        self.experts_dir = script_dir / "top_50_experts"
        self.experts = {}
        self.etare_experts = {}
        self._etare_mtimes = {}
        self._manifest_mtime = 0.0
        self._load_etare_experts()
        self._load_experts()
        self._record_initial_mtimes()

    def _load_etare_experts(self):
        """Load ETARE numpy experts (preferred over LSTM when available)."""
        if not ETARE_AVAILABLE:
            return
        for symbol in ['BTCUSD']:
            expert = load_etare_expert(symbol)
            if expert:
                self.etare_experts[symbol] = expert
                logging.info(f"ETARE expert loaded for {symbol} (WR={expert.win_rate*100:.1f}%)")

    def _check_model_staleness(self, model_path: Path):
        """H-8: Warn if LSTM model file is older than LSTM_MAX_AGE_DAYS."""
        try:
            import os
            mtime = os.path.getmtime(str(model_path))
            age_days = (time.time() - mtime) / 86400.0
            if age_days > LSTM_MAX_AGE_DAYS:
                logging.warning(
                    f"STALE MODEL: {model_path.name} is {age_days:.1f} days old "
                    f"(limit: {LSTM_MAX_AGE_DAYS} days). Consider retraining."
                )
        except Exception as e:
            logging.debug(f"Could not check model staleness: {e}")

    def _load_experts(self):
        manifest_path = self.experts_dir / "top_50_manifest.json"
        if not manifest_path.exists():
            logging.warning("No expert manifest found")
            return

        import json
        with open(manifest_path) as f:
            manifest = json.load(f)

        for entry in manifest.get('experts', []):
            symbol = entry['symbol'].upper()
            model_file = self.experts_dir / entry['filename']
            if model_file.exists():
                # H-8: Check model staleness before loading
                self._check_model_staleness(model_file)
                model = LSTMModel(
                    input_size=entry.get('input_size', 8),
                    hidden_size=entry.get('hidden_size', 128),
                    output_size=3,
                    num_layers=2
                )
                model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=False))
                model.eval()
                self.experts[symbol] = model
                logging.info(f"Loaded LSTM expert for {symbol}")

    def get_expert(self, symbol: str):
        return self.experts.get(symbol.upper())

    def get_etare_expert(self, symbol: str):
        return self.etare_experts.get(symbol.upper())

    def _etare_json_path(self, symbol: str) -> Path:
        script_dir = Path(__file__).parent.absolute()
        return script_dir / "ETARE_QuantumFusion" / "models" / f"{symbol.lower()}_etare_expert.json"

    def _record_initial_mtimes(self):
        """Record file modification times after initial load."""
        for symbol in ['BTCUSD']:
            p = self._etare_json_path(symbol)
            if p.exists():
                self._etare_mtimes[symbol] = p.stat().st_mtime
        manifest_path = self.experts_dir / "top_50_manifest.json"
        if manifest_path.exists():
            self._manifest_mtime = manifest_path.stat().st_mtime

    def check_for_updates(self):
        """Check if expert files have been modified and hot-reload if needed."""
        # Check ETARE experts
        for symbol in list(self._etare_mtimes.keys()) + ['BTCUSD']:
            p = self._etare_json_path(symbol)
            if not p.exists():
                continue
            current_mtime = p.stat().st_mtime
            old_mtime = self._etare_mtimes.get(symbol, 0.0)
            if current_mtime > old_mtime:
                try:
                    if ETARE_AVAILABLE:
                        expert = load_etare_expert(symbol)
                        if expert:
                            self.etare_experts[symbol] = expert
                            self._etare_mtimes[symbol] = current_mtime
                            logging.info(f"HOT-RELOAD: ETARE expert for {symbol} reloaded (WR={expert.win_rate*100:.1f}%)")
                except Exception as e:
                    logging.warning(f"HOT-RELOAD: Failed to reload ETARE {symbol}: {e}")

        # Check LSTM manifest
        manifest_path = self.experts_dir / "top_50_manifest.json"
        if manifest_path.exists():
            current_mtime = manifest_path.stat().st_mtime
            if current_mtime > self._manifest_mtime:
                try:
                    self.experts.clear()
                    self._load_experts()
                    self._manifest_mtime = current_mtime
                    logging.info("HOT-RELOAD: LSTM experts reloaded from updated manifest")
                except Exception as e:
                    logging.warning(f"HOT-RELOAD: Failed to reload LSTM experts: {e}")


# ============================================================
# MAIN TRADING LOGIC
# ============================================================

class FTMOTrader:
    def __init__(self):
        self.expert_loader = ExpertLoader()
        self.teqa_bridge = TEQABridge() if TEQA_ENABLED else None
        self.qnif_bridge = QNIFBridge() if QNIF_ENABLED else None
        self.connected = False
        # FTMO drawdown protection: recorded once on first connect each day
        self.initial_balance = 0.0       # Balance at account start (for total DD)
        self.starting_balance = 0.0      # Balance at session start (for daily DD)
        self._dd_blocked = False

    def connect(self) -> bool:
        """Connect ONCE and stay connected"""
        if self.connected:
            if mt5.terminal_info() is not None:
                acc = mt5.account_info()
                if acc and acc.login == ACCOUNT['account']:
                    return True
            self.connected = False

        if not mt5.initialize(path=ACCOUNT['terminal_path']):
            logging.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        # Check if already logged into the correct account (skip login to avoid killing trades)
        pre_check = mt5.account_info()
        if pre_check and pre_check.login == ACCOUNT['account']:
            logging.info(f"Already logged in as {ACCOUNT['account']}, skipping mt5.login()")
        elif pre_check and pre_check.login != 0:
            # SAFETY: Another account is logged in — don't kick it out
            logging.error(
                f"TERMINAL CONFLICT: Account {pre_check.login} is already logged in on this terminal. "
                f"Logging in as {ACCOUNT['account']} would KILL its open trades. "
                f"Install a second FTMO terminal to run both accounts simultaneously."
            )
            mt5.shutdown()
            return False
        else:
            # No account logged in — safe to login
            if not mt5.login(ACCOUNT['account'], password=ACCOUNT['password'], server=ACCOUNT['server']):
                logging.error(f"Login failed: {mt5.last_error()}")
                return False

        acc = mt5.account_info()
        if acc:
            if self.initial_balance == 0.0:
                self.initial_balance = acc.balance  # Set once, never reset
            if self.starting_balance == 0.0:
                self.starting_balance = acc.balance  # Daily starting balance
            logging.info(f"CONNECTED: {ACCOUNT['name']} | Balance: ${acc.balance:,.2f}")
            self.connected = True
            return True

        return False

    def check_ftmo_drawdown(self) -> bool:
        """Check FTMO daily and total drawdown limits using equity.
        Returns True if trading is allowed, False if DD limit breached."""
        acc = mt5.account_info()
        if acc is None:
            return False

        equity = acc.equity

        # Daily DD check: (starting_balance - equity) / starting_balance
        if self.starting_balance > 0:
            daily_dd_pct = (self.starting_balance - equity) / self.starting_balance * 100.0
            if daily_dd_pct >= FTMO_DAILY_DD_PCT:
                logging.critical(
                    f"FTMO DAILY DD BREACH: {daily_dd_pct:.2f}% >= {FTMO_DAILY_DD_PCT}% | "
                    f"Equity: ${equity:,.2f} | Start: ${self.starting_balance:,.2f} | "
                    f"BLOCKING NEW TRADES"
                )
                self._dd_blocked = True
                return False

        # Total DD check: (initial_balance - equity) / initial_balance
        if self.initial_balance > 0:
            total_dd_pct = (self.initial_balance - equity) / self.initial_balance * 100.0
            if total_dd_pct >= FTMO_TOTAL_DD_PCT:
                logging.critical(
                    f"FTMO TOTAL DD BREACH: {total_dd_pct:.2f}% >= {FTMO_TOTAL_DD_PCT}% | "
                    f"Equity: ${equity:,.2f} | Initial: ${self.initial_balance:,.2f} | "
                    f"BLOCKING NEW TRADES"
                )
                self._dd_blocked = True
                return False

        self._dd_blocked = False
        return True

    def has_position(self, symbol: str) -> bool:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                if pos.magic == ACCOUNT['magic_number']:
                    return True
        return False

    def analyze(self, symbol: str) -> Tuple[Regime, float, Action, float]:
        # Regime detection uses M1 data
        rates_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        if rates_m1 is None or len(rates_m1) < 50:
            return Regime.CHOPPY, 0.0, Action.HOLD, 0.0

        df_m1 = pd.DataFrame(rates_m1)
        df_m1['time'] = pd.to_datetime(df_m1['time'], unit='s')
        prices = df_m1['close'].values

        regime, fidelity = detect_regime(prices, symbol=symbol)

        if regime != Regime.CLEAN:
            return regime, fidelity, Action.HOLD, 0.0

        # Try ETARE expert first (preferred for BTCUSD)
        etare = self.expert_loader.get_etare_expert(symbol)
        if etare and ETARE_AVAILABLE:
            rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 200)
            if rates_m5 is not None and len(rates_m5) >= 60:
                df_m5 = pd.DataFrame(rates_m5)
                df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
                state = prepare_etare_features(df_m5)
                if state is not None:
                    direction, confidence = etare.predict(state)
                    if confidence >= CONFIDENCE_THRESHOLD:
                        if direction == "BUY":
                            return regime, fidelity, Action.BUY, confidence
                        elif direction == "SELL":
                            return regime, fidelity, Action.SELL, confidence
                    return regime, fidelity, Action.HOLD, confidence

        # Fallback to LSTM expert
        expert = self.expert_loader.get_expert(symbol)
        if expert is None:
            return regime, fidelity, Action.HOLD, 0.0

        features = prepare_features(df_m1)
        if len(features) < 30:
            return regime, fidelity, Action.HOLD, 0.0

        sequence = features[-30:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)

        with torch.no_grad():
            output = expert(sequence_tensor)
            probs = torch.softmax(output, dim=1)
            action_idx = torch.argmax(probs).item()
            confidence = probs[0, action_idx].item()

        action = Action(action_idx)
        if confidence < CONFIDENCE_THRESHOLD:
            action = Action.HOLD

        return regime, fidelity, action, confidence

    def _apply_lipolysis(self, symbol: str, tighten_factor: float):
        """Tighten SL on losing positions (growth amplifier lipolysis).
        Only tightens, never loosens. Respects AGENT_SL_MIN."""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        for pos in positions:
            if pos.magic != ACCOUNT['magic_number']:
                continue
            if pos.profit >= 0:
                continue
            if pos.sl == 0.0:
                continue
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                continue
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            current_sl_dist = abs(pos.price_open - pos.sl)
            new_sl_dist = current_sl_dist * tighten_factor
            if tick_value > 0 and pos.volume > 0:
                min_sl_dist = AGENT_SL_MIN / (tick_value * pos.volume) * tick_size
            else:
                min_sl_dist = 0
            new_sl_dist = max(new_sl_dist, min_sl_dist)
            if new_sl_dist >= current_sl_dist:
                continue
            digits = symbol_info.digits
            if pos.type == 0:
                new_sl = round(pos.price_open - new_sl_dist, digits)
            else:
                new_sl = round(pos.price_open + new_sl_dist, digits)
            try:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(
                        f"[FTMO][{symbol}] LIPOLYSIS: "
                        f"tightened SL on #{pos.ticket} "
                        f"{current_sl_dist:.5f} -> {new_sl_dist:.5f} "
                        f"({(1-tighten_factor)*100:.1f}% tighter)")
            except Exception as e:
                logging.debug(f"[FTMO] Lipolysis error: {e}")

    def execute_trade(self, symbol: str, action: Action) -> bool:
        if action not in [Action.BUY, Action.SELL]:
            return False

        if self.has_position(symbol):
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

        # ATR-based SL distance using FTMO-specific ATR multiplier
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
            sl_distance = atr * FTMO_ATR_MULTIPLIER
        else:
            sl_distance = 50 * point

        # Respect broker's minimum stop level
        min_sl_distance = (stops_level + 10) * point
        if sl_distance < min_sl_distance:
            sl_distance = min_sl_distance

        # Calculate lot size to keep loss at exactly FTMO_RISK_PER_TRADE
        sl_ticks = sl_distance / tick_size if tick_size > 0 else 0
        if tick_value > 0 and sl_ticks > 0:
            lot = FTMO_RISK_PER_TRADE / (sl_ticks * tick_value)
        else:
            lot = symbol_info.volume_min

        # Clamp lot to broker limits and round to step
        lot = max(symbol_info.volume_min, lot)
        lot = min(lot, symbol_info.volume_max)
        lot = round(lot / symbol_info.volume_step) * symbol_info.volume_step
        lot = max(symbol_info.volume_min, lot)  # Ensure not zero after rounding

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

        # Normalize SL/TP to symbol's price precision
        digits = symbol_info.digits
        sl = round(sl, digits)
        tp = round(tp, digits)
        price = round(price, digits)

        filling_mode = mt5.ORDER_FILLING_IOC
        if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": ACCOUNT['magic_number'],
            "comment": "FTMO_BRAIN",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        try:
            result = mt5.order_send(request)
        except Exception as e:
            logging.error(f"order_send exception: {e}")
            return False
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"TRADE: {action.name} {symbol} @ {price:.2f} | Lot: {lot:.3f} | SL: ${FTMO_RISK_PER_TRADE} | TP: ${FTMO_RISK_PER_TRADE * TP_MULTIPLIER} | Dyn TP at {DYNAMIC_TP_PERCENT}%")
            return True
        else:
            logging.error(f"TRADE FAILED: {result.comment if result else 'None'}")
            return False

    def manage_positions(self):
        """Active position management: rolling SL + dynamic TP."""
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

            # --- DYNAMIC TP: consult expert before closing ---
            if SET_DYNAMIC_TP and current_profit_distance >= dynamic_tp_threshold:
                # Ask ETARE expert if the trend supports holding
                should_hold = False
                etare = self.expert_loader.get_etare_expert(pos.symbol)
                if etare and ETARE_AVAILABLE:
                    rates_m5 = mt5.copy_rates_from_pos(pos.symbol, mt5.TIMEFRAME_M5, 0, 200)
                    if rates_m5 is not None and len(rates_m5) >= 60:
                        df_m5 = pd.DataFrame(rates_m5)
                        df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
                        state = prepare_etare_features(df_m5)
                        if state is not None:
                            direction, confidence = etare.predict(state)
                            pos_direction = "BUY" if pos.type == 0 else "SELL"
                            if direction == pos_direction and confidence >= CONFIDENCE_THRESHOLD:
                                should_hold = True
                                logging.info(
                                    f"EXPERT HOLD: {pos.symbol} #{pos.ticket} | "
                                    f"Expert says {direction} ({confidence:.2f}) - riding the trend"
                                )

                if not should_hold:
                    # Expert says close or no expert - take the dynamic TP
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
                    try:
                        result = mt5.order_send(request)
                    except Exception as e:
                        logging.error(f"DYN_TP order_send exception for {pos.symbol} #{pos.ticket}: {e}")
                        continue
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"DYNAMIC TP: Closed {pos.symbol} #{pos.ticket} | Profit: ${pos.profit:.2f}")
                    continue
                # Expert says hold - fall through to rolling SL to trail and protect

            # --- ROLLING SL + TP TRAIL: maintain 3:1 ratio as price moves ---
            if ROLLING_SL_ENABLED and current_profit_distance > 0:
                rolled_sl_distance = entry_sl_distance / ROLLING_SL_MULTIPLIER

                if pos.type == 0:  # BUY
                    new_sl = current_price - rolled_sl_distance
                    new_sl = max(new_sl, pos.price_open)
                    if new_sl - tick.bid > -min_distance:
                        new_sl = tick.bid - min_distance
                    if pos.sl > 0 and new_sl <= pos.sl:
                        continue
                    # Move TP forward to maintain 3:1 from new SL
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
                    # Move TP forward to maintain 3:1 from new SL
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
                    try:
                        result = mt5.order_send(request)
                    except Exception as e:
                        logging.error(f"TRAIL order_send exception for {pos.symbol} #{pos.ticket}: {e}")
                        continue
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"TRAIL: {pos.symbol} #{pos.ticket} SL -> {new_sl:.2f} | TP -> {new_tp:.2f}")

    def run_cycle(self):
        self.expert_loader.check_for_updates()

        if not self.connect():
            logging.error("Connection lost - will retry next cycle")
            return

        # Manage existing positions first (dynamic TP + rolling SL) -- always runs
        self.manage_positions()

        # FTMO drawdown protection: block new trades if DD limits breached
        if not self.check_ftmo_drawdown():
            logging.warning("FTMO DD limit breached - managing positions only, no new trades")
            return

        for symbol in ACCOUNT['symbols']:
            regime, fidelity, action, confidence = self.analyze(symbol)

            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if tick else 0

            logging.info(f"[{symbol}] {regime.value} | {action.name} ({confidence:.2f})")

            if COLLECTION_ENABLED:
                try:
                    collect_signal({
                        'symbol': symbol,
                        'direction': action.name,
                        'confidence': confidence,
                        'quantum_entropy': fidelity,
                        'dominant_state': fidelity,
                        'price': price,
                        'regime': regime.value,
                        'source': 'FTMO'
                    })
                except Exception:
                    pass

            # Apply TEQA quantum signal
            if self.teqa_bridge is not None and action is not None:
                final_act, final_conf, lot_mult, teqa_reason = \
                    self.teqa_bridge.apply_to_lstm(action.name, confidence, symbol=symbol)
                action_map = {'BUY': Action.BUY, 'SELL': Action.SELL, 'HOLD': Action.HOLD}
                action = action_map.get(final_act, Action.HOLD)
                confidence = final_conf
                logging.info(f"[{symbol}] {teqa_reason}")

            # Apply QNIF Hybrid Veto (has override authority over TEQA/LSTM)
            if self.qnif_bridge is not None and action is not None:
                final_act, final_conf, lot_mult, qnif_reason = \
                    self.qnif_bridge.apply_hybrid_veto(action.name, confidence, symbol=symbol)
                action_map = {'BUY': Action.BUY, 'SELL': Action.SELL, 'HOLD': Action.HOLD}
                action = action_map.get(final_act, Action.HOLD)
                confidence = final_conf
                logging.info(f"[{symbol}] {qnif_reason}")

            trade_executed = False
            if regime == Regime.CLEAN and action in [Action.BUY, Action.SELL]:
                trade_executed = self.execute_trade(symbol, action)

            # Growth Amplifier: Hyperplasia (second position on strong growth)
            if trade_executed and self.teqa_bridge is not None:
                try:
                    signal = self.teqa_bridge.read_signal(symbol=symbol)
                    if signal and signal.growth_active and signal.growth_hyperplasia:
                        ratio = signal.growth_second_lot_ratio
                        if ratio > 0:
                            self.execute_trade(symbol, action)
                            logging.info(
                                f"[FTMO][{symbol}] GROWTH: hyperplasia={ratio:.0%} lot "
                                f"| signal={signal.growth_signal:.3f} "
                                f"| variant={signal.growth_variant}")
                except Exception as ge:
                    logging.debug(f"[FTMO] Growth hyperplasia check: {ge}")

            # Growth Amplifier: Lipolysis (tighten SL on losing positions)
            if self.teqa_bridge is not None:
                try:
                    signal = self.teqa_bridge.read_signal(symbol=symbol)
                    if signal and signal.growth_active and signal.growth_lipolysis_factor < 1.0:
                        self._apply_lipolysis(symbol, signal.growth_lipolysis_factor)
                except Exception as le:
                    logging.debug(f"[FTMO] Growth lipolysis check: {le}")


def main():
    print("=" * 60)
    print("  FTMO CHALLENGE - DEDICATED")
    print(f"  Account: {ACCOUNT['account']} ({ACCOUNT['name']})")
    print(f"  Risk: ${FTMO_RISK_PER_TRADE}/trade | ATR mult: {FTMO_ATR_MULTIPLIER}")
    print(f"  TP: {TP_MULTIPLIER}x | Dyn TP: {DYNAMIC_TP_PERCENT}% | Rolling SL: {ROLLING_SL_MULTIPLIER}x")
    print("=" * 60)

    trader = FTMOTrader()

    try:
        while True:
            # Verify correct account before each cycle
            acct = mt5.account_info()
            if acct is None or acct.login != ACCOUNT['account']:
                logging.critical(
                    f"ACCOUNT MISMATCH! Expected {ACCOUNT['account']}, "
                    f"got {acct.login if acct else 'None'}. Stopping to protect trades."
                )
                break
            trader.run_cycle()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    # CRITICAL: Acquire process lock BEFORE doing anything
    lock = ProcessLock("BRAIN_FTMO", account="1521063483")

    try:
        with lock:
            logging.info("=" * 60)
            logging.info("BRAIN_FTMO - PROCESS LOCK ACQUIRED")
            logging.info("=" * 60)

            # Pre-launch validation - ensures experts are trained before trading
            TRADING_SYMBOLS = ACCOUNT['symbols']
            if not validate_prelaunch(symbols=TRADING_SYMBOLS):
                logging.error("Pre-launch validation failed. Exiting.")
                sys.exit(1)

            main()

    except RuntimeError as e:
        logging.error("=" * 60)
        logging.error("PROCESS LOCK FAILURE")
        logging.error("=" * 60)
        logging.error(str(e))
        logging.error("")
        logging.error("Another instance of BRAIN_FTMO is already running.")
        logging.error("This prevents duplicate trades on account 1521063483.")
        logging.error("")
        logging.error("To stop all processes safely:")
        logging.error("  Run: SAFE_SHUTDOWN.bat")
        logging.error("")
        logging.error("To check running processes:")
        logging.error("  Run: python process_lock.py --list")
        logging.error("=" * 60)
        sys.exit(1)
