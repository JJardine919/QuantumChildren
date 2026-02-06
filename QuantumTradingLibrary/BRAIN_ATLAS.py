"""
QUANTUM BRAIN - ATLAS DEDICATED
================================
Handles Atlas Funded account ONLY.

Accounts:
  - 212000584 (Atlas Funded)

Run: python BRAIN_ATLAS.py

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
    INITIAL_SL_DOLLARS,
    TP_MULTIPLIER,
    ROLLING_SL_MULTIPLIER,
    DYNAMIC_TP_PERCENT,
    SET_DYNAMIC_TP,
    ROLLING_SL_ENABLED,
    CONFIDENCE_THRESHOLD,
    CHECK_INTERVAL_SECONDS,
    REQUIRE_TRAINED_EXPERT
)

# Import credentials securely - passwords from .env file
from credential_manager import get_credentials, CredentialError

# Pre-launch validation
from prelaunch_validator import validate_prelaunch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][ATLAS] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('brain_atlas.log'),
        logging.StreamHandler()
    ]
)

# ============================================================
# ATLAS ACCOUNT ONLY - Credentials from .env via credential_manager
# ============================================================

def _load_atlas_account():
    """Load Atlas account with credentials from .env file."""
    try:
        creds = get_credentials('ATLAS')
        return {
            'ATLAS': {
                'account': creds['account'],
                'password': creds['password'],
                'server': creds['server'],
                'terminal_path': creds.get('terminal_path') or r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe",
                'name': 'Atlas Funded',
                'initial_balance': 50000,
                'profit_target': 0.10,
                'daily_loss_limit': 0.05,
                'max_drawdown': 0.10,
                'locked': False,
                'symbols': creds.get('symbols', ['BTCUSD', 'ETHUSD']),
                'magic_number': creds.get('magic', 212001),
            },
        }
    except CredentialError as e:
        logging.error(f"Failed to load ATLAS credentials: {e}")
        logging.error("Please configure ATLAS_PASSWORD in .env file")
        sys.exit(1)

ACCOUNTS = _load_atlas_account()


# ============================================================
# CONFIG
# ============================================================

@dataclass
class BrainConfig:
    CLEAN_REGIME_THRESHOLD: float = 0.95
    VOLATILE_REGIME_THRESHOLD: float = 0.85
    CONFIDENCE_THRESHOLD: float = 0.48
    RISK_PER_TRADE_PCT: float = 0.005
    BARS_FOR_ANALYSIS: int = 256
    SEQUENCE_LENGTH: int = 30
    CHECK_INTERVAL_SECONDS: int = 60
    BASE_LOT: float = 0.01


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


# Regime enum and bridge from quantum_regime_bridge
from quantum_regime_bridge import QuantumRegimeBridge, Regime


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
# REGIME DETECTOR (delegates to QuantumRegimeBridge)
# ============================================================

class RegimeDetector:
    def __init__(self, config: BrainConfig):
        self.config = config
        self._bridge = QuantumRegimeBridge(config)

    def analyze_regime(self, prices: np.ndarray, symbol: str = "BTCUSD") -> Tuple[Regime, float]:
        return self._bridge.analyze_regime(prices, symbol=symbol)


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

    def _load_expert(self, expert_info: dict) -> Optional[nn.Module]:
        filename = expert_info['filename']
        if filename in self.loaded_experts:
            return self.loaded_experts[filename]

        expert_path = self.experts_dir / filename
        if not expert_path.exists():
            return None

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

    def connect(self) -> bool:
        """Connect to MT5 - ONLY accept correct account"""
        expected_account = self.account_config['account']

        # Check if already connected to the right account
        try:
            current_account = mt5.account_info()
            if current_account:
                if current_account.login == expected_account:
                    self.starting_balance = current_account.balance
                    return True
                else:
                    # WRONG ACCOUNT - do NOT trade!
                    logging.error(f"[{self.account_key}] WRONG ACCOUNT! Expected {expected_account}, got {current_account.login}")
                    logging.error(f"[{self.account_key}] Will NOT trade - refusing to switch accounts")
                    return False
        except:
            pass

        # Not connected - try to initialize
        terminal_path = self.account_config.get('terminal_path')
        if terminal_path:
            init_ok = mt5.initialize(path=terminal_path)
        else:
            init_ok = mt5.initialize()

        if not init_ok:
            logging.error(f"[{self.account_key}] MT5 init failed: {mt5.last_error()}")
            logging.info(f"[{self.account_key}] Make sure MT5 is running")
            return False

        # Check what account we connected to
        account_info = mt5.account_info()
        if account_info is None:
            logging.error(f"[{self.account_key}] Could not get account info")
            return False

        # Verify it's the RIGHT account
        if account_info.login != expected_account:
            logging.error(f"[{self.account_key}] WRONG ACCOUNT! Expected {expected_account}, got {account_info.login}")
            logging.error(f"[{self.account_key}] Will NOT trade - open the correct MT5 terminal!")
            return False

        self.starting_balance = account_info.balance
        logging.info(f"[{self.account_key}] Connected - Account: {account_info.login} Balance: ${account_info.balance:,.2f}")
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
        # Lot range based on symbol minimums - some symbols require 0.1+ minimum
        account_info = mt5.account_info()
        symbol_info = mt5.symbol_info(symbol)

        if account_info is None or symbol_info is None:
            return 0.01

        vol_step = symbol_info.volume_step
        vol_min = symbol_info.volume_min  # CRITICAL: Get actual minimum
        vol_max = symbol_info.volume_max

        # Scale slightly with balance
        desired_lot = (account_info.balance / 100000) * 0.01
        lot = round(desired_lot / vol_step) * vol_step

        # Respect symbol's actual min/max, then cap at our max (0.05)
        lot = max(vol_min, lot)  # Use symbol's minimum, not hardcoded 0.01
        lot = min(lot, min(vol_max, 0.05))  # Cap at 0.05 or symbol max

        logging.info(f"[{self.account_key}][{symbol}] Lot: {lot} (min={vol_min}, step={vol_step})")
        return lot

    def analyze_symbol(self, symbol: str) -> Tuple[Regime, float, Optional[Action], float]:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, self.config.BARS_FOR_ANALYSIS)
        if rates is None or len(rates) < self.config.BARS_FOR_ANALYSIS:
            return Regime.CHOPPY, 0.0, None, 0.0

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        prices = df['close'].values

        regime, fidelity = self.regime_detector.analyze_regime(prices, symbol=symbol)
        logging.info(f"[{self.account_key}][{symbol}] Regime: {regime.value} ({fidelity:.3f})")

        if regime != Regime.CLEAN:
            return regime, fidelity, Action.HOLD, 0.0

        expert = self.expert_loader.get_best_expert_for_symbol(symbol)
        if expert is None:
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
        if lot <= 0:
            logging.error(f"[{self.account_key}] Invalid lot size for {symbol}")
            return False

        # All values from config_loader (MASTER_CONFIG.json)
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        point = symbol_info.point
        stops_level = symbol_info.trade_stops_level
        spread = symbol_info.spread

        logging.info(f"[{self.account_key}][{symbol}] Symbol specs: tick_val={tick_value}, tick_size={tick_size}, point={point}, stops_level={stops_level}, spread={spread}")

        # Calculate SL distance for MAX_LOSS_DOLLARS
        if tick_value > 0 and lot > 0:
            sl_ticks = MAX_LOSS_DOLLARS / (tick_value * lot)
            sl_distance = sl_ticks * tick_size
        else:
            sl_distance = 100 * point

        # Respect broker's minimum stop level - use LARGER buffer
        # stops_level is in points, add buffer for spread and safety
        min_sl_distance = (stops_level + spread + 20) * point
        if sl_distance < min_sl_distance:
            logging.warning(f"[{self.account_key}][{symbol}] SL distance {sl_distance:.5f} < min {min_sl_distance:.5f}, adjusting")
            sl_distance = min_sl_distance
            # Recalculate lot to maintain max loss limit with wider SL
            new_sl_ticks = sl_distance / tick_size
            if tick_value > 0 and new_sl_ticks > 0:
                new_lot = MAX_LOSS_DOLLARS / (tick_value * new_sl_ticks)
                new_lot = max(symbol_info.volume_min, new_lot)
                new_lot = round(new_lot / symbol_info.volume_step) * symbol_info.volume_step
                if new_lot < lot:
                    lot = new_lot
                    logging.info(f"[{self.account_key}][{symbol}] Reduced lot to {lot} to maintain ${MAX_LOSS_DOLLARS} max loss")

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

        # Normalize SL/TP to symbol's price precision
        digits = symbol_info.digits
        sl = round(sl, digits)
        tp = round(tp, digits)
        price = round(price, digits)

        # Final validation: ensure stops are far enough from price
        actual_sl_distance = abs(price - sl)
        min_required = (stops_level + spread) * point
        if actual_sl_distance < min_required:
            logging.error(f"[{self.account_key}][{symbol}] ABORT: SL too close! Distance={actual_sl_distance:.5f}, Required={min_required:.5f}")
            return False

        logging.info(f"[{self.account_key}][{symbol}] Price: {price}, SL: {sl}, TP: {tp}, Lot: {lot}")
        logging.info(f"[{self.account_key}][{symbol}] SL Distance: {sl_distance:.5f} | Max Loss: ${MAX_LOSS_DOLLARS}")

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
            "comment": f"ATLAS_{self.account_key}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"[{self.account_key}] TRADE: {action.name} {symbol} @ {price:.2f} SL:{sl:.2f} TP:{tp:.2f}")
            return True
        else:
            logging.error(f"[{self.account_key}] FAILED: {result.comment} ({result.retcode})")
            return False

    def run_cycle(self) -> Dict[str, dict]:
        results = {}
        if not self.check_risk_limits():
            return results

        for symbol in self.account_config['symbols']:
            regime, fidelity, action, confidence = self.analyze_symbol(symbol)
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
                        'source': 'ATLAS'
                    })
                except Exception as e:
                    pass  # Don't let collection errors affect trading

        return results


# ============================================================
# MAIN BRAIN
# ============================================================

class AtlasBrain:
    def __init__(self, config: BrainConfig = None):
        self.config = config or BrainConfig()
        self.traders: Dict[str, AccountTrader] = {}

    def initialize(self) -> bool:
        logging.info("Initializing Atlas brain")
        for key in ACCOUNTS:
            self.traders[key] = AccountTrader(key, ACCOUNTS[key], self.config)
        return True

    def run_loop(self):
        print("=" * 60)
        print("  ATLAS QUANTUM BRAIN")
        print("  Dedicated brain for Atlas Funded account")
        print("=" * 60)

        if not self.initialize():
            return

        account_keys = list(self.traders.keys())
        idx = 0

        try:
            while True:
                key = account_keys[idx]
                trader = self.traders[key]

                print(f"\n{'='*60}")
                print(f"  {ACCOUNTS[key]['name']} | {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*60}")

                if trader.connect():
                    results = trader.run_cycle()
                    for symbol, data in results.items():
                        icon = "+" if data['regime'] == 'CLEAN' else "-"
                        status = "TRADED" if data['trade_executed'] else ""
                        print(f"  [{icon}] {symbol}: {data['regime']} | "
                              f"{data['action']} ({data['confidence']:.2f}) {status}")

                idx = (idx + 1) % len(account_keys)
                time.sleep(self.config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logging.info("Stopped")
        finally:
            mt5.shutdown()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Pre-launch validation - ensures experts are trained before trading
    TRADING_SYMBOLS = ['BTCUSD', 'ETHUSD']
    if not validate_prelaunch(symbols=TRADING_SYMBOLS):
        logging.error("Pre-launch validation failed. Exiting.")
        sys.exit(1)

    brain = AtlasBrain()
    brain.run_loop()
