"""
QUANTUM BRAIN - VPS VERSION
============================
Simplified version for running on VPS via Wine.
Only runs BlueGuardian accounts since that's what's on the VPS.

This runs via Wine Python to access MT5.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import MetaTrader5 as mt5

# Configuration
@dataclass
class BrainConfig:
    CLEAN_REGIME_THRESHOLD: float = 0.95
    VOLATILE_REGIME_THRESHOLD: float = 0.85
    CONFIDENCE_THRESHOLD: float = 0.55
    RISK_PER_TRADE_PCT: float = 0.005
    BARS_FOR_ANALYSIS: int = 256
    SEQUENCE_LENGTH: int = 30
    CHECK_INTERVAL_SECONDS: int = 60
    BASE_LOT: float = 0.01

# BlueGuardian VPS Account
ACCOUNT_CONFIG = {
    'account': 366592,  # $5K instant
    'server': 'BlueGuardian-Server',
    'terminal_path': '/root/.wine/drive_c/Program Files/Blue Guardian MT5 Terminal/terminal64.exe',
    'name': 'BlueGuardian $5K Instant',
    'daily_loss_limit': 0.05,
    'max_drawdown': 0.10,
    'symbols': ['BTCUSD', 'ETHUSD', 'XAUUSD'],
    'magic_number': 366001,
}

class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class Regime(Enum):
    CLEAN = "CLEAN"
    VOLATILE = "VOLATILE"
    CHOPPY = "CHOPPY"

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

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")
    with open('quantum_brain_vps.log', 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")

def analyze_regime(prices: np.ndarray, config: BrainConfig) -> Tuple[Regime, float]:
    """Simplified regime detection using compression ratio"""
    import zlib
    data_bytes = prices.astype(np.float32).tobytes()
    compressed = zlib.compress(data_bytes, level=9)
    ratio = len(data_bytes) / len(compressed)

    if ratio >= 3.5:
        fidelity = 0.96
        regime = Regime.CLEAN
    elif ratio >= 2.5:
        fidelity = 0.88
        regime = Regime.VOLATILE
    else:
        fidelity = 0.75
        regime = Regime.CHOPPY

    return regime, fidelity

def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Prepare features for LSTM"""
    data = df.copy()

    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    data['rsi'] = 100 - (100 / (1 + rs))

    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    data['bb_middle'] = data['close'].rolling(20).mean()
    data['bb_std'] = data['close'].rolling(20).std()
    data['bb_position'] = (data['close'] - data['bb_middle']) / (data['bb_std'] + 1e-8)

    data['momentum'] = data['close'] / data['close'].shift(10)
    data['atr'] = data['high'].rolling(14).max() - data['low'].rolling(14).min()
    data['price_change'] = data['close'].pct_change()

    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_position', 'momentum', 'atr', 'price_change', 'close']

    for col in feature_cols:
        if col in data.columns:
            mean = data[col].rolling(100, min_periods=1).mean()
            std = data[col].rolling(100, min_periods=1).std()
            data[col] = (data[col] - mean) / (std + 1e-8)
            data[col] = data[col].clip(-4, 4)

    data = data.fillna(0)
    return data[feature_cols].values

class ExpertLoader:
    def __init__(self, experts_dir: str = "top_50_experts"):
        self.experts_dir = Path(experts_dir)
        self.manifest = None
        self.loaded_experts: Dict[str, nn.Module] = {}
        self._load_manifest()

    def _load_manifest(self):
        manifest_path = self.experts_dir / "top_50_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            log(f"Loaded manifest with {len(self.manifest['experts'])} experts")
        else:
            log(f"No manifest found at {manifest_path}")
            self.manifest = {'experts': []}

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
            log(f"Expert file not found: {expert_path}")
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
            log(f"Loaded expert: {filename}")
            return model
        except Exception as e:
            log(f"Failed to load expert {filename}: {e}")
            return None

def has_position(symbol: str, magic: int) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            if pos.magic == magic:
                return True
    return False

def execute_trade(symbol: str, action: Action, config: BrainConfig) -> bool:
    if action not in [Action.BUY, Action.SELL]:
        return False

    if has_position(symbol, ACCOUNT_CONFIG['magic_number']):
        log(f"Already have position in {symbol}")
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        log(f"Symbol {symbol} not found")
        return False

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    # Get account balance for lot sizing - capped at 0.03
    account = mt5.account_info()
    lot = max(0.01, round((account.balance / 100000) * 0.01, 2))
    lot = min(lot, 0.03)  # Hard cap - small lots = tighter stops work

    point = symbol_info.point

    # Calculate ATR for SL/TP
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 15)
    if rates is not None:
        df = pd.DataFrame(rates)
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
    else:
        atr = 100 * point

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

    filling_mode = mt5.ORDER_FILLING_IOC
    if symbol_info.filling_mode & mt5.SYMBOL_FILLING_FOK:
        filling_mode = mt5.ORDER_FILLING_FOK

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": ACCOUNT_CONFIG['magic_number'],
        "comment": "QB_VPS",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"TRADE OPENED: {action.name} {symbol} @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        return True
    else:
        log(f"TRADE FAILED: {result.comment} ({result.retcode})")
        return False

def main():
    config = BrainConfig()
    expert_loader = ExpertLoader()

    log("=" * 60)
    log("QUANTUM BRAIN VPS - STARTING")
    log("=" * 60)

    # Initialize MT5
    terminal_path = ACCOUNT_CONFIG['terminal_path']
    if not mt5.initialize(path=terminal_path):
        log(f"MT5 init failed: {mt5.last_error()}")
        return

    account = mt5.account_info()
    if account is None:
        log("Failed to get account info - please login to MT5 first")
        mt5.shutdown()
        return

    starting_balance = account.balance
    log(f"Connected: Account {account.login}, Balance: ${account.balance:,.2f}")
    log(f"Symbols: {ACCOUNT_CONFIG['symbols']}")
    log("=" * 60)

    try:
        while True:
            # Risk check
            account = mt5.account_info()
            if account:
                loss_pct = (starting_balance - account.balance) / starting_balance
                if loss_pct >= ACCOUNT_CONFIG['daily_loss_limit']:
                    log(f"DAILY LOSS LIMIT HIT: {loss_pct*100:.2f}%")
                    break

            log("-" * 40)

            for symbol in ACCOUNT_CONFIG['symbols']:
                # Get data
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, config.BARS_FOR_ANALYSIS)
                if rates is None or len(rates) < config.BARS_FOR_ANALYSIS:
                    log(f"[{symbol}] Insufficient data")
                    continue

                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                prices = df['close'].values

                # Analyze regime
                regime, fidelity = analyze_regime(prices, config)
                log(f"[{symbol}] Regime: {regime.value} (Fidelity: {fidelity:.4f})")

                if regime != Regime.CLEAN:
                    continue

                # Get expert prediction
                expert = expert_loader.get_best_expert_for_symbol(symbol)
                if expert is None:
                    log(f"[{symbol}] No expert available")
                    continue

                # Prepare features
                features = prepare_features(df)
                seq_len = config.SEQUENCE_LENGTH
                if len(features) < seq_len:
                    continue

                sequence = features[-seq_len:]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)

                with torch.no_grad():
                    output = expert(sequence_tensor)
                    probs = torch.softmax(output, dim=1)
                    action_idx = torch.argmax(probs).item()
                    confidence = probs[0, action_idx].item()

                action = Action(action_idx)

                if confidence < config.CONFIDENCE_THRESHOLD:
                    action = Action.HOLD

                log(f"[{symbol}] Expert: {action.name} (Confidence: {confidence:.2f})")

                # Execute if BUY/SELL
                if action in [Action.BUY, Action.SELL]:
                    execute_trade(symbol, action, config)

            time.sleep(config.CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        log("Stopped by user")
    finally:
        mt5.shutdown()
        log("Disconnected")

if __name__ == "__main__":
    main()
