"""
QUANTUM BRAIN - BLUEGUARDIAN INSTANT ONLY
==========================================
DEDICATED script for $5K Instant account ONLY.
Runs independently - does NOT touch other accounts.

Account: 366604 (BlueGuardian $5K Instant - PAYS REAL MONEY)

Run: python BRAIN_BG_INSTANT.py

DO NOT run this alongside BRAIN_BLUEGUARDIAN.py
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

# QuantumChildren network collection
try:
    from entropy_collector import collect_signal, NODE_ID
    COLLECTION_ENABLED = True
except ImportError:
    COLLECTION_ENABLED = False
    NODE_ID = "LOCAL"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][BG_INSTANT] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('brain_bg_instant.log'),
        logging.StreamHandler()
    ]
)

# ============================================================
# SINGLE ACCOUNT - DO NOT ADD OTHERS
# ============================================================

ACCOUNT = {
    'account': 366604,
    'password': 'YF^oHH&4Nm',
    'server': 'BlueGuardian-Server',
    'terminal_path': r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
    'name': 'BlueGuardian $5K Instant',
    'symbols': ['BTCUSD'],
    'magic_number': 366001,
}

# ============================================================
# CONFIG FROM MASTER_CONFIG.json - DO NOT HARDCODE
# ============================================================
from config_loader import (
    MAX_LOSS_DOLLARS,
    TP_MULTIPLIER,
    CHECK_INTERVAL_SECONDS as CHECK_INTERVAL,
    CONFIDENCE_THRESHOLD
)


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
# REGIME DETECTOR
# ============================================================

def detect_regime(prices: np.ndarray) -> Tuple[Regime, float]:
    import zlib
    data_bytes = prices.astype(np.float32).tobytes()
    compressed = zlib.compress(data_bytes, level=9)
    ratio = len(data_bytes) / len(compressed)

    if ratio >= 1.1:
        return Regime.CLEAN, 0.96
    elif ratio >= 0.9:
        return Regime.VOLATILE, 0.88
    else:
        return Regime.CHOPPY, 0.75


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
        self._load_experts()

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
                model = LSTMModel(input_size=8)
                model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=False))
                model.eval()
                self.experts[symbol] = model
                logging.info(f"Loaded expert for {symbol}")

    def get_expert(self, symbol: str):
        return self.experts.get(symbol.upper())


# ============================================================
# MAIN TRADING LOGIC
# ============================================================

class InstantTrader:
    def __init__(self):
        self.expert_loader = ExpertLoader()
        self.connected = False

    def connect(self) -> bool:
        """Connect ONCE and stay connected"""
        if self.connected:
            # Verify still connected
            if mt5.terminal_info() is not None:
                acc = mt5.account_info()
                if acc and acc.login == ACCOUNT['account']:
                    return True
            self.connected = False

        # Initialize
        if not mt5.initialize(path=ACCOUNT['terminal_path']):
            logging.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        # Login
        if not mt5.login(ACCOUNT['account'], password=ACCOUNT['password'], server=ACCOUNT['server']):
            logging.error(f"Login failed: {mt5.last_error()}")
            return False

        acc = mt5.account_info()
        if acc:
            logging.info(f"CONNECTED: {ACCOUNT['name']} | Balance: ${acc.balance:,.2f}")
            self.connected = True
            return True

        return False

    def has_position(self, symbol: str) -> bool:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                if pos.magic == ACCOUNT['magic_number']:
                    return True
        return False

    def analyze(self, symbol: str) -> Tuple[Regime, float, Action, float]:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        if rates is None or len(rates) < 50:
            return Regime.CHOPPY, 0.0, Action.HOLD, 0.0

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        prices = df['close'].values

        regime, fidelity = detect_regime(prices)

        if regime != Regime.CLEAN:
            return regime, fidelity, Action.HOLD, 0.0

        expert = self.expert_loader.get_expert(symbol)
        if expert is None:
            return regime, fidelity, Action.HOLD, 0.0

        features = prepare_features(df)
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

        # Fixed lot
        lot = 0.01

        # FIXED DOLLAR SL
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        point = symbol_info.point
        stops_level = symbol_info.trade_stops_level

        if tick_value > 0 and lot > 0:
            sl_ticks = MAX_LOSS_DOLLARS / (tick_value * lot)
            sl_distance = sl_ticks * tick_size
        else:
            sl_distance = 50 * point

        # Respect broker's minimum stop level
        min_sl_distance = (stops_level + 10) * point
        if sl_distance < min_sl_distance:
            sl_distance = min_sl_distance

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
            "comment": "BG_INSTANT",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"TRADE: {action.name} {symbol} @ {price:.2f} | SL: ${MAX_LOSS_DOLLARS} | TP: ${MAX_LOSS_DOLLARS * TP_MULTIPLIER}")
            return True
        else:
            logging.error(f"TRADE FAILED: {result.comment if result else 'None'}")
            return False

    def run_cycle(self):
        if not self.connect():
            logging.error("Connection lost - will retry next cycle")
            return

        for symbol in ACCOUNT['symbols']:
            regime, fidelity, action, confidence = self.analyze(symbol)

            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if tick else 0

            logging.info(f"[{symbol}] {regime.value} | {action.name} ({confidence:.2f})")

            # Send to network
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
                        'source': 'BG_INSTANT'
                    })
                except:
                    pass

            if regime == Regime.CLEAN and action in [Action.BUY, Action.SELL]:
                self.execute_trade(symbol, action)


def main():
    print("=" * 60)
    print("  BLUEGUARDIAN INSTANT - DEDICATED")
    print(f"  Account: {ACCOUNT['account']} ({ACCOUNT['name']})")
    print(f"  SL: ${MAX_LOSS_DOLLARS} | TP: ${MAX_LOSS_DOLLARS * TP_MULTIPLIER}")
    print("=" * 60)
    print()
    print("  THIS SCRIPT RUNS ALONE - DO NOT RUN BRAIN_BLUEGUARDIAN.py")
    print()
    print("=" * 60)

    trader = InstantTrader()

    try:
        while True:
            trader.run_cycle()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
