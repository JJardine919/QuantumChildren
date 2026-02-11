"""
QUANTUM BRAIN - GETLEVERAGED DEDICATED
=======================================
ONE script per account. Run separate instances:

  python BRAIN_GETLEVERAGED.py --account GL_1
  python BRAIN_GETLEVERAGED.py --account GL_2
  python BRAIN_GETLEVERAGED.py --account GL_3

Accounts:
  GL_1 = 113326
  GL_2 = 113328
  GL_3 = 107245

Author: DooDoo + Claude
Date: 2026-01-30
"""

import sys
import os
import json
import time
import logging
import argparse
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

# QuantumChildren network collection
try:
    from entropy_collector import collect_signal, collect_entropy_snapshot, NODE_ID
    COLLECTION_ENABLED = True
except ImportError:
    COLLECTION_ENABLED = False
    NODE_ID = "LOCAL"

# Import trading settings from config - DO NOT HARDCODE
from config_loader import (
    MAX_LOSS_DOLLARS,
    TP_MULTIPLIER,
    ROLLING_SL_MULTIPLIER,
    DYNAMIC_TP_PERCENT,
    SET_DYNAMIC_TP,
    ROLLING_SL_ENABLED,
    CONFIDENCE_THRESHOLD,
    CHECK_INTERVAL_SECONDS,
    LSTM_MAX_AGE_DAYS,
)

# Secure credential loading (H-1: no plaintext passwords)
from credential_manager import get_credentials, CredentialError

# Pre-launch validation (H-7)
from prelaunch_validator import validate_prelaunch

# TEQA quantum signal bridge (H-5)
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][GL] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('brain_getleveraged.log'),
        logging.StreamHandler()
    ]
)

# ============================================================
# GETLEVERAGED ACCOUNTS ONLY
# ============================================================

def _load_account(account_key: str) -> dict:
    """Load account config from credential_manager. No plaintext passwords."""
    creds = get_credentials(account_key)
    return {
        'account': creds['account'],
        'password': creds['password'],
        'server': creds['server'],
        'terminal_path': creds.get('terminal_path'),
        'name': f'GetLeveraged {account_key}',
        'initial_balance': 10000,
        'profit_target': 0.10,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
        'symbols': creds.get('symbols', ['BTCUSD', 'XAUUSD', 'ETHUSD']),
        'magic_number': creds.get('magic', int(account_key.split('_')[1]) * 1000 + 1),
    }


# ============================================================
# CONFIG
# ============================================================

@dataclass
class BrainConfig:
    CLEAN_REGIME_THRESHOLD: float = 0.95
    VOLATILE_REGIME_THRESHOLD: float = 0.85
    CONFIDENCE_THRESHOLD: float = CONFIDENCE_THRESHOLD  # From config_loader, NOT hardcoded
    RISK_PER_TRADE_PCT: float = 0.005
    BARS_FOR_ANALYSIS: int = 256
    SEQUENCE_LENGTH: int = 30
    CHECK_INTERVAL_SECONDS: int = CHECK_INTERVAL_SECONDS  # From config_loader
    BASE_LOT: float = 0.01


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class Regime(Enum):
    CLEAN = "CLEAN"
    VOLATILE = "VOLATILE"
    CHOPPY = "CHOPPY"


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
# REGIME DETECTOR (Compression-based)
# ============================================================

class RegimeDetector:
    def __init__(self, config: BrainConfig):
        self.config = config

    def analyze_regime(self, prices: np.ndarray) -> Tuple[Regime, float]:
        import zlib
        data_bytes = prices.astype(np.float32).tobytes()
        compressed = zlib.compress(data_bytes, level=9)
        ratio = len(data_bytes) / len(compressed)

        # Log the actual ratio for debugging
        logging.debug(f"Compression ratio: {ratio:.3f} (need >= 1.3 for CLEAN)")

        if ratio >= 1.1:
            fidelity = 0.96
            regime = Regime.CLEAN
        elif ratio >= 0.9:
            fidelity = 0.88
            regime = Regime.VOLATILE
        else:
            fidelity = 0.75
            regime = Regime.CHOPPY

        return regime, fidelity


# ============================================================
# EXPERT LOADER
# ============================================================

class ExpertLoader:
    def __init__(self, experts_dir: str = None):
        # Use absolute path relative to this script's location
        if experts_dir is None:
            script_dir = Path(__file__).parent.absolute()
            self.experts_dir = script_dir / "top_50_experts"
        else:
            self.experts_dir = Path(experts_dir)
        self.manifest = None
        self.loaded_experts: Dict[str, nn.Module] = {}
        self._load_manifest()

    def _load_manifest(self):
        manifest_path = self.experts_dir / "top_50_manifest.json"
        logging.info(f"Looking for experts at: {manifest_path}")
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            logging.info(f"Loaded {len(self.manifest['experts'])} experts")
        else:
            logging.error(f"Expert manifest NOT FOUND at {manifest_path}")

    def get_best_expert_for_symbol(self, symbol: str) -> Optional[nn.Module]:
        if not self.manifest:
            return None
        for expert in self.manifest['experts']:
            if expert['symbol'] == symbol:
                return self._load_expert(expert)
        return None

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

    def _load_expert(self, expert_info: dict) -> Optional[nn.Module]:
        filename = expert_info['filename']
        if filename in self.loaded_experts:
            return self.loaded_experts[filename]

        expert_path = self.experts_dir / filename
        if not expert_path.exists():
            return None

        # H-8: Check model staleness before loading
        self._check_model_staleness(expert_path)

        try:
            model = LSTMModel(
                input_size=expert_info['input_size'],
                hidden_size=expert_info['hidden_size'],
                output_size=3,
                num_layers=2
            )
            state_dict = torch.load(str(expert_path), map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            self.loaded_experts[filename] = model
            return model
        except Exception as e:
            logging.error(f"Failed to load expert: {e}")
            return None


# ============================================================
# FEATURE ENGINEER (MUST match training exactly)
# ============================================================

class FeatureEngineer:
    """Features: rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr"""

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        data = df.copy()
        for c in ['open', 'high', 'low', 'close', 'tick_volume']:
            if c in data.columns:
                data[c] = data[c].astype(float)

        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(20).mean()
        data['bb_std'] = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']

        # Momentum & ROC
        data['momentum'] = data['close'] / data['close'].shift(10)
        data['roc'] = data['close'].pct_change(10) * 100

        # ATR
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

        # Global Z-score normalization
        for col in feature_cols:
            col_mean = data[col].mean()
            col_std = data[col].std() + 1e-8
            data[col] = (data[col] - col_mean) / col_std

        data = data.fillna(0)
        return data[feature_cols].values


# ============================================================
# ACCOUNT TRADER
# ============================================================

class AccountTrader:
    def __init__(self, account_key: str, account_config: dict, config: BrainConfig):
        self.account_key = account_key
        self.account_config = account_config
        self.config = config
        self.regime_detector = RegimeDetector(config)
        self.expert_loader = ExpertLoader()
        self.feature_engineer = FeatureEngineer()
        self.starting_balance = 0.0
        # H-5: TEQA bridge
        self.teqa_bridge = TEQABridge() if TEQA_ENABLED else None
        self.qnif_bridge = QNIFBridge() if QNIF_ENABLED else None

    def connect(self) -> bool:
        """Connect ONCE at startup. Called only once — never re-logins."""
        terminal_path = self.account_config.get('terminal_path')
        if terminal_path:
            init_ok = mt5.initialize(path=terminal_path)
        else:
            init_ok = mt5.initialize()

        if not init_ok:
            logging.error(f"[{self.account_key}] MT5 init failed: {mt5.last_error()}")
            logging.info(f"[{self.account_key}] Make sure MT5 is running")
            return False

        # Login ONCE
        login_ok = mt5.login(
            self.account_config['account'],
            password=self.account_config['password'],
            server=self.account_config['server']
        )
        if not login_ok:
            err = mt5.last_error()
            logging.error(f"[{self.account_key}] Login failed: {err}")
            return False

        account_info = mt5.account_info()
        if account_info is None:
            logging.error(f"[{self.account_key}] Could not get account info")
            return False

        self.starting_balance = account_info.balance
        logging.info(f"[{self.account_key}] Connected - Account: {account_info.login} "
                     f"Balance: ${account_info.balance:,.2f}")
        return True

    def check_risk_limits(self) -> bool:
        account_info = mt5.account_info()
        if account_info is None:
            return False

        if self.starting_balance > 0:
            daily_loss = (self.starting_balance - account_info.balance) / self.starting_balance
            if daily_loss >= self.account_config['daily_loss_limit']:
                logging.warning(f"[{self.account_key}] DAILY LOSS LIMIT: {daily_loss*100:.2f}%")
                return False
        return True

    def get_lot_size(self, symbol: str) -> float:
        # Lot range 0.01-0.03 - small lots = less swing = tighter stops
        account_info = mt5.account_info()
        symbol_info = mt5.symbol_info(symbol)

        if account_info is None or symbol_info is None:
            return 0.01

        vol_step = symbol_info.volume_step

        # Scale slightly but cap at 0.03
        desired_lot = (account_info.balance / 100000) * 0.01
        lot = round(desired_lot / vol_step) * vol_step
        lot = max(0.01, min(lot, 0.03))
        return lot

    def analyze_symbol(self, symbol: str) -> Tuple[Regime, float, Optional[Action], float]:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, self.config.BARS_FOR_ANALYSIS)
        if rates is None or len(rates) < self.config.BARS_FOR_ANALYSIS:
            return Regime.CHOPPY, 0.0, None, 0.0

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        prices = df['close'].values

        regime, fidelity = self.regime_detector.analyze_regime(prices)
        logging.info(f"[{self.account_key}][{symbol}] Regime: {regime.value} ({fidelity:.3f})")

        if regime != Regime.CLEAN:
            return regime, fidelity, Action.HOLD, 0.0

        expert = self.expert_loader.get_best_expert_for_symbol(symbol)
        if expert is None:
            logging.warning(f"[{self.account_key}][{symbol}] NO EXPERT FOUND - cannot trade!")
            return regime, fidelity, None, 0.0

        features = self.feature_engineer.prepare_features(df)
        seq_len = self.config.SEQUENCE_LENGTH
        if len(features) < seq_len:
            return regime, fidelity, Action.HOLD, 0.0

        sequence = features[-seq_len:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)

        with torch.no_grad():
            output = expert(sequence_tensor)
            probs = torch.softmax(output, dim=1)
            action_idx = torch.argmax(probs).item()
            confidence = probs[0, action_idx].item()

        action = Action(action_idx)
        if confidence < self.config.CONFIDENCE_THRESHOLD:
            action = Action.HOLD

        logging.info(f"[{self.account_key}][{symbol}] Signal: {action.name} ({confidence:.2f})")
        return regime, fidelity, action, confidence

    def has_position(self, symbol: str) -> bool:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                if pos.magic == self.account_config['magic_number']:
                    return True
        return False

    def execute_trade(self, symbol: str, action: Action, confidence: float) -> bool:
        if action not in [Action.BUY, Action.SELL]:
            return False

        if self.has_position(symbol):
            logging.info(f"[{self.account_key}] Already have position in {symbol}")
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        lot = self.get_lot_size(symbol)

        # All values from config_loader (MASTER_CONFIG.json)
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        point = symbol_info.point
        stops_level = symbol_info.trade_stops_level

        # Calculate SL distance for MAX_LOSS_DOLLARS from config
        if tick_value > 0 and lot > 0:
            sl_ticks = MAX_LOSS_DOLLARS / (tick_value * lot)
            sl_distance = sl_ticks * tick_size
        else:
            sl_distance = 50 * point

        # Respect broker's minimum stop level
        min_sl_distance = (stops_level + 10) * point
        if sl_distance < min_sl_distance:
            sl_distance = min_sl_distance

        # TP from config
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

        logging.info(f"[{self.account_key}] SL Distance: {sl_distance:.2f} | Max Loss: ${MAX_LOSS_DOLLARS}")

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
            "magic": self.account_config['magic_number'],
            "comment": f"GL_{self.account_key}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        try:
            result = mt5.order_send(request)
        except Exception as e:
            logging.error(f"[{self.account_key}] order_send exception: {e}")
            return False
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"[{self.account_key}] TRADE: {action.name} {symbol} @ {price:.2f} SL:{sl:.2f} TP:{tp:.2f}")
            return True
        else:
            logging.error(f"[{self.account_key}] FAILED: {result.comment if result else 'None'} ({result.retcode if result else 'N/A'})")
            return False

    def manage_positions(self):
        """H-6: Active position management — rolling SL + dynamic TP.
        Ported from BRAIN_BG_INSTANT.py."""
        positions = mt5.positions_get()
        if not positions:
            return

        for pos in positions:
            if pos.magic != self.account_config['magic_number']:
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

            # --- DYNAMIC TP: close at threshold ---
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
                    "magic": self.account_config['magic_number'],
                    "comment": "DYN_TP",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                try:
                    result = mt5.order_send(request)
                except Exception as e:
                    logging.error(f"[{self.account_key}] DYN_TP order_send exception: {e}")
                    continue
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"[{self.account_key}] DYNAMIC TP: {pos.symbol} #{pos.ticket} | Profit: ${pos.profit:.2f}")
                continue

            # --- ROLLING SL: trail stop to lock in profit ---
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
                    try:
                        result = mt5.order_send(request)
                    except Exception as e:
                        logging.error(f"[{self.account_key}] TRAIL order_send exception for {pos.symbol} #{pos.ticket}: {e}")
                        continue
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"[{self.account_key}] TRAIL: {pos.symbol} #{pos.ticket} SL -> {new_sl} | TP -> {new_tp}")

    def run_cycle(self) -> Dict[str, dict]:
        results = {}
        if not self.check_risk_limits():
            return results

        # H-6: Manage existing positions first (rolling SL + dynamic TP)
        self.manage_positions()

        for symbol in self.account_config['symbols']:
            regime, fidelity, action, confidence = self.analyze_symbol(symbol)

            # H-5: Apply TEQA bridge if available
            if self.teqa_bridge is not None and action is not None:
                lstm_action_str = action.name
                final_action_str, final_conf, lot_mult, teqa_reason = \
                    self.teqa_bridge.apply_to_lstm(lstm_action_str, confidence, symbol=symbol)
                action_map = {'BUY': Action.BUY, 'SELL': Action.SELL, 'HOLD': Action.HOLD}
                action = action_map.get(final_action_str, Action.HOLD)
                confidence = final_conf
                if teqa_reason:
                    logging.info(f"[{self.account_key}][{symbol}] TEQA: {teqa_reason}")

            # Apply QNIF Hybrid Veto (has override authority over TEQA/LSTM)
            qnif_reason = ""
            if self.qnif_bridge is not None and action is not None:
                legacy_action_str = action.name
                final_action_str, final_conf, lot_mult, qnif_reason = \
                    self.qnif_bridge.apply_hybrid_veto(legacy_action_str, confidence, symbol=symbol)
                action_map = {'BUY': Action.BUY, 'SELL': Action.SELL, 'HOLD': Action.HOLD}
                action = action_map.get(final_action_str, Action.HOLD)
                confidence = final_conf
                logging.info(f"[{self.account_key}][{symbol}] {qnif_reason}")

            trade_executed = False
            if regime == Regime.CLEAN and action in [Action.BUY, Action.SELL]:
                trade_executed = self.execute_trade(symbol, action, confidence)

            results[symbol] = {
                'regime': regime.value,
                'fidelity': fidelity,
                'action': action.name if action else 'NONE',
                'confidence': confidence,
                'trade_executed': trade_executed,
            }

            # Send to QuantumChildren network
            if COLLECTION_ENABLED:
                try:
                    tick = mt5.symbol_info_tick(symbol)
                    price = tick.bid if tick else 0
                    collect_signal({
                        'symbol': symbol,
                        'direction': action.name if action else 'HOLD',
                        'confidence': confidence,
                        'quantum_entropy': fidelity,
                        'dominant_state': fidelity,
                        'price': price,
                        'regime': regime.value,
                        'source': 'GETLEVERAGED'
                    })
                except Exception as e:
                    pass  # Don't let collection errors affect trading

        return results


# ============================================================
# MAIN BRAIN
# ============================================================

VALID_ACCOUNTS = ['GL_1', 'GL_2', 'GL_3']


def run_single_account(account_key: str):
    """Run brain for ONE account only. Never cycles. Never re-logins."""
    if account_key not in VALID_ACCOUNTS:
        logging.error(f"Unknown account: {account_key}")
        logging.error(f"Valid accounts: {', '.join(VALID_ACCOUNTS)}")
        sys.exit(1)

    # H-1: Load credentials securely
    try:
        account_config = _load_account(account_key)
    except CredentialError as e:
        logging.error(f"Credential error for {account_key}: {e}")
        sys.exit(1)

    # H-7: Pre-launch validation
    if not validate_prelaunch(symbols=account_config['symbols']):
        logging.error("Pre-launch validation failed. Exiting.")
        sys.exit(1)

    config = BrainConfig()
    trader = AccountTrader(account_key, account_config, config)

    print("=" * 60)
    print(f"  GETLEVERAGED QUANTUM BRAIN — {account_config['name']}")
    print(f"  Account: {account_config['account']} (ONE account, ONE script)")
    print(f"  SL: ${MAX_LOSS_DOLLARS} | TP: {TP_MULTIPLIER}x | Dynamic TP: {DYNAMIC_TP_PERCENT}%")
    print(f"  Rolling SL: {ROLLING_SL_MULTIPLIER}x | TEQA: {'ON' if TEQA_ENABLED else 'OFF'}")
    print("=" * 60)

    # Connect ONCE at startup — never again
    if not trader.connect():
        logging.error(f"Failed to connect to {account_key}. Is MT5 running?")
        sys.exit(1)

    logging.info(f"Connected to {account_key}. Running single-account loop.")

    try:
        while True:
            # Verify we're still on the right account (detect if another script stole the session)
            acct = mt5.account_info()
            if acct is None or acct.login != account_config['account']:
                logging.error(f"ACCOUNT MISMATCH! Expected {account_config['account']}, "
                              f"got {acct.login if acct else 'None'}. Another script may have switched.")
                logging.error("Stopping to protect trades. Fix: run each GL account in separate MT5 terminal.")
                break

            print(f"\n{'='*60}")
            print(f"  {account_config['name']} | {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")

            results = trader.run_cycle()
            for symbol, data in results.items():
                icon = "+" if data['regime'] == 'CLEAN' else "-"
                status = "TRADED" if data['trade_executed'] else ""
                print(f"  [{icon}] {symbol}: {data['regime']} | "
                      f"{data['action']} ({data['confidence']:.2f}) {status}")

            if TEQA_ENABLED and trader.teqa_bridge:
                print(f"  {trader.teqa_bridge.get_status_line()}")
                if QNIF_ENABLED and trader.qnif_bridge:
                    print(f"  {trader.qnif_bridge.get_status_line()}")

            wait_time = config.CHECK_INTERVAL_SECONDS
            print(f"\n  Waiting {wait_time}s...")
            time.sleep(wait_time)

    except KeyboardInterrupt:
        logging.info("Stopped")
    finally:
        mt5.shutdown()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='GetLeveraged Quantum Brain — ONE account per instance',
        epilog='Run 3 separate windows:\n'
               '  python BRAIN_GETLEVERAGED.py --account GL_1\n'
               '  python BRAIN_GETLEVERAGED.py --account GL_2\n'
               '  python BRAIN_GETLEVERAGED.py --account GL_3',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--account', required=True, choices=['GL_1', 'GL_2', 'GL_3'],
                        help='Which GetLeveraged account to trade')
    args = parser.parse_args()

    run_single_account(args.account)
