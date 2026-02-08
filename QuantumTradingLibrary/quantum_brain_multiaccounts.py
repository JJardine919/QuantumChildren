"""
QUANTUM BRAIN - Multi-Account Trading Orchestrator
====================================================
Manages multiple prop firm accounts with quantum regime detection.

Accounts:
1. BlueGuardian 366592 - $5K Instant (ACTIVE)
2. BlueGuardian 365060 - $100K Challenge (LOCKED)
3. GetLeveraged 113326 - Account 1 (LOCKED)
4. GetLeveraged 113328 - Account 2 (LOCKED)
5. GetLeveraged 107245 - Account 3 (LOCKED)
6. Atlas 212000584 - Atlas Funded (LOCKED)
7. ETARE 1512287880 - QuantumFusion (LOCKED)

Usage:
  python quantum_brain_multiaccounts.py              # Run $5K instant only
  python quantum_brain_multiaccounts.py --unlock-all # Run all accounts
  python quantum_brain_multiaccounts.py --account 365060 # Run specific account

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
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import MetaTrader5 as mt5

# Import credential manager
from credential_manager import get_credentials, get_password, CredentialError

# Add systems path
sys.path.insert(0, str(Path(__file__).parent / "01_Systems" / "QuantumCompression" / "utils"))

try:
    import qutip as qt
    from scipy.optimize import minimize
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    print("[WARNING] QuTiP not available - using simplified regime detection")

try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    print("[WARNING] PyWavelets not available - using raw data")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('quantum_brain_multi.log'),
        logging.StreamHandler()
    ]
)


# ============================================================
# ACCOUNT CONFIGURATIONS
# ============================================================

# ============================================================
# ACCOUNT CONFIGURATIONS - Passwords loaded from credential_manager
# ============================================================

def _load_account_password(key_mapping: str) -> str:
    """Load password from credential_manager, return empty string if not found"""
    try:
        return get_password(key_mapping)
    except CredentialError:
        return ''

ACCOUNTS = {
    # BlueGuardian Accounts
    'BG_5K_INSTANT': {
        'account': 366592,
        'password': _load_account_password('BG_INSTANT'),
        'server': 'BlueGuardian-Server',
        'terminal_path': r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
        'name': 'BlueGuardian $5K Instant',
        'challenge_type': 'instant',
        'initial_balance': 5000,
        'profit_target': 0.10,      # 10%
        'daily_loss_limit': 0.05,   # 5%
        'max_drawdown': 0.10,       # 10%
        'locked': False,            # ACTIVE - Start here
        'symbols': ['BTCUSD', 'ETHUSD'],
        'magic_number': 366001,
    },
    'BG_100K_CHALLENGE': {
        'account': 365060,
        'password': _load_account_password('BG_CHALLENGE'),
        'server': 'BlueGuardian-Server',
        'terminal_path': r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
        'name': 'BlueGuardian $100K Challenge',
        'challenge_type': 'challenge',
        'initial_balance': 100000,
        'profit_target': 0.10,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
        'locked': True,             # LOCKED
        'symbols': ['BTCUSD', 'ETHUSD'],
        'magic_number': 365001,
    },

    # GetLeveraged Accounts
    'GL_ACCOUNT_1': {
        'account': 113326,
        'password': _load_account_password('GL_1'),
        'server': 'GetLeveraged-Server',
        'terminal_path': r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe",
        'name': 'GetLeveraged Account 1',
        'challenge_type': 'instant',
        'initial_balance': 10000,
        'profit_target': 0.10,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
        'locked': True,
        'symbols': ['BTCUSD'],
        'magic_number': 113001,
    },
    'GL_ACCOUNT_2': {
        'account': 113328,
        'password': _load_account_password('GL_2'),
        'server': 'GetLeveraged-Server',
        'terminal_path': r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe",
        'name': 'GetLeveraged Account 2',
        'challenge_type': 'instant',
        'initial_balance': 10000,
        'profit_target': 0.10,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
        'locked': True,
        'symbols': ['BTCUSD'],
        'magic_number': 113002,
    },
    'GL_ACCOUNT_3': {
        'account': 107245,
        'password': _load_account_password('GL_3'),
        'server': 'GetLeveraged-Server',
        'terminal_path': r"C:\Program Files\GetLeveraged MT5 Terminal\terminal64.exe",
        'name': 'GetLeveraged Account 3',
        'challenge_type': 'instant',
        'initial_balance': 10000,
        'profit_target': 0.10,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
        'locked': True,
        'symbols': ['BTCUSD'],
        'magic_number': 107001,
    },

    # Atlas Funded
    'ATLAS': {
        'account': 212000584,
        'password': _load_account_password('ATLAS'),
        'server': 'AtlasFunded-Server',
        'terminal_path': r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe",
        'name': 'Atlas Funded',
        'challenge_type': 'funded',
        'initial_balance': 50000,
        'profit_target': 0.10,
        'daily_loss_limit': 0.05,
        'max_drawdown': 0.10,
        'locked': True,
        'symbols': ['BTCUSD', 'XAUUSD'],
        'magic_number': 212001,
    },

    # ETARE QuantumFusion (no credential_manager key, keep empty)
    'ETARE': {
        'account': 1512287880,
        'password': '',
        'server': 'ETARE-Server',
        'terminal_path': r"C:\Program Files\MetaTrader 5\terminal64.exe",  # Generic
        'name': 'ETARE QuantumFusion',
        'challenge_type': 'research',
        'initial_balance': 10000,
        'profit_target': 0.20,
        'daily_loss_limit': 0.10,
        'max_drawdown': 0.20,
        'locked': True,
        'symbols': ['BTCUSD', 'ETHUSD'],
        'magic_number': 151001,
    },
}


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class BrainConfig:
    """Configuration for the Quantum Brain"""
    # Regime thresholds
    CLEAN_REGIME_THRESHOLD: float = 0.95
    VOLATILE_REGIME_THRESHOLD: float = 0.85

    # Trading parameters
    CONFIDENCE_THRESHOLD: float = 0.55
    RISK_PER_TRADE_PCT: float = 0.005   # 0.5%

    # Data parameters
    BARS_FOR_ANALYSIS: int = 256
    SEQUENCE_LENGTH: int = 30

    # Timing
    CHECK_INTERVAL_SECONDS: int = 60

    # Lot sizing
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
# NOISE REDUCER
# ============================================================

class NoiseReducer:
    def denoise(self, data: np.ndarray, wavelet='db4', level=4) -> np.ndarray:
        if not WAVELET_AVAILABLE:
            return data

        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, threshold, mode='soft'))

        denoised = pywt.waverec(new_coeffs, wavelet)
        return denoised[:len(data)]


# ============================================================
# QUANTUM REGIME DETECTOR
# ============================================================

class QuantumRegimeDetector:
    def __init__(self, config: BrainConfig):
        self.config = config
        self.noise_reducer = NoiseReducer()

    def ry(self, theta):
        return (-1j * theta/2 * qt.sigmay()).expm()

    def cnot(self, N, control, target):
        p0 = qt.ket2dm(qt.basis(2, 0))
        p1 = qt.ket2dm(qt.basis(2, 1))
        I = qt.qeye(2)
        X = qt.sigmax()
        ops = [qt.qeye(2)] * N
        ops[control] = p0
        ops[target] = I
        term1 = qt.tensor(ops)
        ops[control] = p1
        ops[target] = X
        term2 = qt.tensor(ops)
        return term1 + term2

    def get_encoder(self, params, num_qubits, layers=2):
        U = qt.qeye([2]*num_qubits)
        param_idx = 0
        for layer in range(layers):
            ry_ops = [self.ry(params[param_idx + i]) for i in range(num_qubits)]
            param_idx += num_qubits
            U = qt.tensor(ry_ops) * U
            for i in range(num_qubits):
                U = self.cnot(num_qubits, i, (i + 1) % num_qubits) * U
        return U

    def cost(self, params, input_state, num_qubits, num_latent):
        num_trash = num_qubits - num_latent
        U = self.get_encoder(params, num_qubits)
        rho = input_state * input_state.dag() if input_state.type == 'ket' else input_state
        rho_out = U * rho * U.dag()
        rho_trash = rho_out.ptrace(range(num_latent, num_qubits))
        ref = qt.tensor([qt.ket2dm(qt.basis(2, 0)) for _ in range(num_trash)])
        fid = qt.fidelity(rho_trash, ref)
        return 1 - fid

    def analyze_regime(self, prices: np.ndarray) -> Tuple[Regime, float]:
        if not QUTIP_AVAILABLE:
            return self._fallback_regime_detection(prices)

        try:
            denoised = self.noise_reducer.denoise(prices)
            data = denoised - denoised.min() + 1e-6
            norm = np.linalg.norm(data)
            state_vector = (data / norm).astype(complex)

            n = len(state_vector)
            num_qubits = int(np.log2(n))
            if 2**num_qubits != n:
                new_n = 2**num_qubits
                state_vector = state_vector[:new_n]

            input_state = qt.Qobj(state_vector, dims=[[2]*num_qubits, [1]*num_qubits]).unit()

            num_latent = num_qubits - 1
            num_params = 2 * num_qubits
            initial_params = np.random.rand(num_params) * np.pi

            result = minimize(
                self.cost,
                initial_params,
                args=(input_state, num_qubits, num_latent),
                method='COBYLA',
                options={'maxiter': 200}
            )

            fidelity = 1 - result.fun

            if fidelity >= self.config.CLEAN_REGIME_THRESHOLD:
                regime = Regime.CLEAN
            elif fidelity >= self.config.VOLATILE_REGIME_THRESHOLD:
                regime = Regime.VOLATILE
            else:
                regime = Regime.CHOPPY

            return regime, fidelity

        except Exception as e:
            logging.warning(f"Quantum analysis failed: {e}, using fallback")
            return self._fallback_regime_detection(prices)

    def _fallback_regime_detection(self, prices: np.ndarray) -> Tuple[Regime, float]:
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


# ============================================================
# EXPERT LOADER
# ============================================================

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
            logging.info(f"Loaded manifest with {len(self.manifest['experts'])} experts")
        else:
            logging.warning(f"No manifest found at {manifest_path}")
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
        cache_key = filename

        if cache_key in self.loaded_experts:
            return self.loaded_experts[cache_key]

        expert_path = self.experts_dir / filename
        if not expert_path.exists():
            logging.warning(f"Expert file not found: {expert_path}")
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

            self.loaded_experts[cache_key] = model
            logging.info(f"Loaded expert: {filename} (rank {expert_info['rank']})")
            return model

        except Exception as e:
            logging.error(f"Failed to load expert {filename}: {e}")
            return None


# ============================================================
# FEATURE ENGINEER
# ============================================================

class FeatureEngineer:
    """
    CRITICAL FIX 2026-01-30: Feature engineering MUST match training code exactly!
    Training code: 01_Systems/System_03_ETARE/ETARE_Redux.py
    Features: rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr
    Normalization: Global Z-score (NOT rolling window)
    """
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        data = df.copy()

        # Ensure numerical types
        for c in ['open', 'high', 'low', 'close', 'tick_volume']:
            if c in data.columns:
                data[c] = data[c].astype(float)

        # 1. RSI (same as training)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # 2. MACD (same as training)
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp1 - exp2
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

        # 3. Bollinger Bands - bb_upper and bb_lower (TRAINING USED THESE, NOT bb_position!)
        data['bb_middle'] = data['close'].rolling(20).mean()
        data['bb_std'] = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']

        # 4. Momentum & ROC (TRAINING USED ROC, NOT price_change!)
        data['momentum'] = data['close'] / data['close'].shift(10)
        data['roc'] = data['close'].pct_change(10) * 100

        # 5. ATR (true range, same as training)
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr'] = data['tr'].rolling(14).mean()

        # EXACT feature columns from training (ETARE_Redux.py line 372)
        feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']

        # Drop NaN from indicator warmup
        data = data.dropna()

        if len(data) == 0:
            return np.zeros((1, len(feature_cols)))

        # GLOBAL Z-score normalization (MUST match training - NOT rolling window!)
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
    """Handles trading for a single account"""

    def __init__(self, account_key: str, account_config: dict, brain_config: BrainConfig):
        self.account_key = account_key
        self.account_config = account_config
        self.brain_config = brain_config

        self.regime_detector = QuantumRegimeDetector(brain_config)
        self.expert_loader = ExpertLoader()
        self.feature_engineer = FeatureEngineer()

        self.connected = False
        self.starting_balance = 0.0
        self.daily_pnl = 0.0

    def connect(self) -> bool:
        """Connect to this account's terminal"""
        terminal_path = self.account_config['terminal_path']

        # Shutdown any existing connection first
        mt5.shutdown()

        if not mt5.initialize(path=terminal_path):
            logging.error(f"[{self.account_key}] Failed to initialize MT5: {mt5.last_error()}")
            return False

        # Try to login if credentials provided
        if self.account_config.get('password'):
            if not mt5.login(
                self.account_config['account'],
                password=self.account_config['password'],
                server=self.account_config['server']
            ):
                logging.error(f"[{self.account_key}] Login failed: {mt5.last_error()}")
                return False

        # Verify connection
        account_info = mt5.account_info()
        if account_info is None:
            logging.error(f"[{self.account_key}] Failed to get account info")
            return False

        self.starting_balance = account_info.balance
        self.connected = True

        logging.info(f"[{self.account_key}] Connected - Account: {account_info.login}, Balance: ${account_info.balance:,.2f}")
        return True

    def check_risk_limits(self) -> bool:
        """Check if account has hit risk limits"""
        account_info = mt5.account_info()
        if account_info is None:
            return False

        current_balance = account_info.balance

        # Daily loss check
        if self.starting_balance > 0:
            daily_loss = (self.starting_balance - current_balance) / self.starting_balance
            if daily_loss >= self.account_config['daily_loss_limit']:
                logging.warning(f"[{self.account_key}] DAILY LOSS LIMIT HIT: {daily_loss*100:.2f}%")
                return False

            # Max drawdown check
            if daily_loss >= self.account_config['max_drawdown']:
                logging.warning(f"[{self.account_key}] MAX DRAWDOWN HIT: {daily_loss*100:.2f}%")
                return False

        return True

    def get_lot_size(self, symbol: str) -> float:
        """Calculate lot size - capped at 0.03 for tighter stops"""
        account_info = mt5.account_info()
        symbol_info = mt5.symbol_info(symbol)

        if account_info is None or symbol_info is None:
            return 0.01

        vol_step = symbol_info.volume_step

        # Scale slightly but cap at 0.03 - small lots = less swing
        desired_lot = (account_info.balance / 100000) * 0.01
        lot = round(desired_lot / vol_step) * vol_step
        lot = max(0.01, min(lot, 0.03))

        logging.info(f"[{self.account_key}] {symbol} lot={lot:.4f}")
        return lot

    def analyze_symbol(self, symbol: str) -> Tuple[Regime, float, Optional[Action], float]:
        """Analyze a single symbol"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, self.brain_config.BARS_FOR_ANALYSIS)
        if rates is None or len(rates) < self.brain_config.BARS_FOR_ANALYSIS:
            logging.warning(f"[{self.account_key}] Insufficient data for {symbol}")
            return Regime.CHOPPY, 0.0, None, 0.0

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        prices = df['close'].values

        regime, fidelity = self.regime_detector.analyze_regime(prices)

        logging.info(f"[{self.account_key}][{symbol}] Regime: {regime.value} (Fidelity: {fidelity:.4f})")

        if regime != Regime.CLEAN:
            return regime, fidelity, Action.HOLD, 0.0

        expert = self.expert_loader.get_best_expert_for_symbol(symbol)
        if expert is None:
            logging.warning(f"[{self.account_key}] No expert for {symbol}")
            return regime, fidelity, None, 0.0

        features = self.feature_engineer.prepare_features(df)

        seq_len = self.brain_config.SEQUENCE_LENGTH
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

        if confidence < self.brain_config.CONFIDENCE_THRESHOLD:
            action = Action.HOLD

        logging.info(f"[{self.account_key}][{symbol}] Expert: {action.name} (Confidence: {confidence:.2f})")

        return regime, fidelity, action, confidence

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position"""
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            for pos in positions:
                if pos.magic == self.account_config['magic_number']:
                    return True
        return False

    def execute_trade(self, symbol: str, action: Action, confidence: float) -> bool:
        """Execute a trade"""
        if action not in [Action.BUY, Action.SELL]:
            return False

        if self.has_position(symbol):
            logging.info(f"[{self.account_key}] Already have position in {symbol}")
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"[{self.account_key}] Symbol {symbol} not found")
            return False

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        lot = self.get_lot_size(symbol)
        point = symbol_info.point

        # Calculate SL/TP based on ATR
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

        # Determine filling mode
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
            "comment": f"QB_{self.account_key}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"[{self.account_key}] TRADE OPENED: {action.name} {symbol} @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
            return True
        else:
            logging.error(f"[{self.account_key}] TRADE FAILED: {result.comment} ({result.retcode})")
            return False

    def run_cycle(self) -> Dict[str, dict]:
        """Run one trading cycle"""
        results = {}

        if not self.check_risk_limits():
            logging.warning(f"[{self.account_key}] Risk limits breached - skipping cycle")
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

        return results


# ============================================================
# MULTI-ACCOUNT QUANTUM BRAIN
# ============================================================

class MultiAccountQuantumBrain:
    """Orchestrates trading across multiple accounts"""

    def __init__(self, config: BrainConfig = None, unlock_all: bool = False, specific_account: str = None):
        self.config = config or BrainConfig()
        self.unlock_all = unlock_all
        self.specific_account = specific_account
        self.traders: Dict[str, AccountTrader] = {}

    def get_active_accounts(self) -> List[str]:
        """Get list of accounts to trade"""
        active = []

        for key, account in ACCOUNTS.items():
            # If specific account requested
            if self.specific_account:
                if str(account['account']) == self.specific_account or key == self.specific_account:
                    active.append(key)
                continue

            # If unlock_all, add all accounts
            if self.unlock_all:
                active.append(key)
                continue

            # Otherwise, only add unlocked accounts
            if not account.get('locked', True):
                active.append(key)

        return active

    def initialize(self) -> bool:
        """Initialize all active account traders"""
        active_accounts = self.get_active_accounts()

        if not active_accounts:
            logging.error("No active accounts found!")
            return False

        logging.info(f"Initializing {len(active_accounts)} account(s): {active_accounts}")

        for account_key in active_accounts:
            account_config = ACCOUNTS[account_key]
            trader = AccountTrader(account_key, account_config, self.config)
            self.traders[account_key] = trader

        return True

    def run_loop(self):
        """Main trading loop for all accounts"""
        logging.info("=" * 70)
        logging.info("QUANTUM BRAIN - MULTI-ACCOUNT MODE ACTIVATED")
        logging.info("=" * 70)

        if not self.initialize():
            return

        account_keys = list(self.traders.keys())
        current_account_idx = 0

        logging.info(f"Trading {len(account_keys)} account(s): {account_keys}")
        logging.info("=" * 70)

        try:
            while True:
                # Rotate through accounts
                account_key = account_keys[current_account_idx]
                trader = self.traders[account_key]

                print("\n" + "=" * 70)
                print(f"  QUANTUM BRAIN - {ACCOUNTS[account_key]['name']}")
                print(f"  Account: {ACCOUNTS[account_key]['account']} | Time: {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 70)

                # Connect to this account
                if trader.connect():
                    results = trader.run_cycle()

                    # Display results
                    for symbol, data in results.items():
                        status_icon = "+" if data['regime'] == 'CLEAN' else "-"
                        trade_status = "TRADED" if data['trade_executed'] else "no trade"
                        print(f"  [{status_icon}] {symbol}: {data['regime']} | "
                              f"Fidelity: {data['fidelity']:.3f} | "
                              f"{data['action']} ({data['confidence']:.2f}) | "
                              f"{trade_status}")
                else:
                    logging.error(f"Failed to connect to {account_key}")

                print("=" * 70)

                # Move to next account
                current_account_idx = (current_account_idx + 1) % len(account_keys)

                # Wait before next cycle
                time.sleep(self.config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logging.info("Stopped by user")
        finally:
            mt5.shutdown()
            logging.info("All connections closed")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Quantum Brain Multi-Account Trader')
    parser.add_argument('--unlock-all', action='store_true', help='Unlock and trade all accounts')
    parser.add_argument('--account', type=str, help='Trade specific account (by number or key)')
    parser.add_argument('--list', action='store_true', help='List all configured accounts')
    parser.add_argument('--once', action='store_true', help='Run one cycle only')

    args = parser.parse_args()

    if args.list:
        print("\nConfigured Accounts:")
        print("=" * 60)
        for key, account in ACCOUNTS.items():
            status = "ACTIVE" if not account.get('locked', True) else "LOCKED"
            print(f"  [{status:6}] {key}")
            print(f"           Account: {account['account']}")
            print(f"           Name: {account['name']}")
            print(f"           Balance: ${account['initial_balance']:,}")
            print(f"           Symbols: {account['symbols']}")
            print()
        return

    config = BrainConfig()
    brain = MultiAccountQuantumBrain(
        config=config,
        unlock_all=args.unlock_all,
        specific_account=args.account
    )

    if args.once:
        # Single cycle for testing
        if brain.initialize():
            for account_key, trader in brain.traders.items():
                print(f"\n=== {account_key} ===")
                if trader.connect():
                    results = trader.run_cycle()
                    print(json.dumps(results, indent=2))
            mt5.shutdown()
    else:
        brain.run_loop()


if __name__ == "__main__":
    main()
