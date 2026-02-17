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
from datetime import datetime, timedelta
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
    ETARE_CONFIDENCE_THRESHOLD,
    CHECK_INTERVAL_SECONDS,
    REQUIRE_TRAINED_EXPERT,
    LSTM_MAX_AGE_DAYS,
    get_symbol_risk,
    REST_SCHEDULE_ENABLED,
    REST_FOREX_WINDOWS,
    REST_CRYPTO_WINDOWS,
    REST_PERSIST_ON_TRADE_CLOSE,
    get_asset_class,
    LOSS_COOLDOWN_MINUTES,
)

# Import credentials securely - passwords from .env file
from credential_manager import get_credentials, CredentialError

# Process lock to prevent duplicate launches
from process_lock import ProcessLock

# Pre-launch validation
from prelaunch_validator import validate_prelaunch

# State persistence — save/restore ephemeral state across restarts
try:
    from state_persistence import StatePersistence
    STATE_PERSISTENCE_ENABLED = True
except ImportError:
    STATE_PERSISTENCE_ENABLED = False

# ETARE feedforward expert (preferred over Conv1D/LSTM)
try:
    from etare_expert import load_etare_expert, ETAREExpert, prepare_etare_features
    ETARE_AVAILABLE = True
except ImportError:
    ETARE_AVAILABLE = False

# TEQA quantum signal bridge
try:
    from teqa_bridge import TEQABridge
    TEQA_ENABLED = True
except ImportError:
    TEQA_ENABLED = False

# TEQA feedback loop (domestication learning)
try:
    from teqa_feedback import TradeOutcomePoller
    FEEDBACK_ENABLED = True
except ImportError:
    FEEDBACK_ENABLED = False

# QNIF quantum-immune signal bridge (Hybrid Veto authority)
try:
    from qnif_bridge import QNIFBridge
    QNIF_ENABLED = True
except ImportError:
    QNIF_ENABLED = False

# VDJ adaptive immune memory (live feedback loop)
try:
    from vdj_recombination import VDJRecombinationEngine
    VDJ_FEEDBACK_ENABLED = True
except ImportError:
    VDJ_FEEDBACK_ENABLED = False

# CRISPR-Cas9 adaptive immune memory (loss pattern blocking)
try:
    from crispr_cas import CRISPRTEQABridge
    CRISPR_FEEDBACK_ENABLED = True
except ImportError:
    CRISPR_FEEDBACK_ENABLED = False

# Protective Deletion (toxic TE pattern suppression)
try:
    from protective_deletion import ProtectiveDeletionTracker
    PROTECTIVE_DELETION_ENABLED = True
except ImportError:
    PROTECTIVE_DELETION_ENABLED = False

# Toxoplasma regime detector with intraday DD protection
try:
    from toxoplasma import ToxoplasmaEngine, ToxoplasmaTEQABridge
    TOXOPLASMA_ENABLED = True
except ImportError:
    TOXOPLASMA_ENABLED = False

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
    CONFIDENCE_THRESHOLD: float = CONFIDENCE_THRESHOLD  # From config_loader, NOT hardcoded
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


class Conv1DModel(nn.Module):
    """GPU-compatible Conv1D model (DirectML). Same interface as LSTMModel."""
    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super().__init__()
        import torch.nn.functional as F
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, hidden_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        import torch.nn.functional as F
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


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
        self._manifest_mtime = 0.0
        self._load_manifest()
        self._record_initial_mtime()

    def _load_manifest(self):
        manifest_path = self.experts_dir / "top_50_manifest.json"
        logging.info(f"Looking for experts at: {manifest_path}")
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            logging.info(f"Loaded {len(self.manifest['experts'])} experts")
        else:
            logging.error(f"Expert manifest NOT FOUND at {manifest_path}")

    def get_best_expert_for_symbol(self, symbol: str, prefer_mutant: bool = True) -> Optional[nn.Module]:
        if not self.manifest:
            return None
        for expert in self.manifest['experts']:
            if expert['symbol'] == symbol:
                if prefer_mutant:
                    mutant = self._try_load_mutant(expert)
                    if mutant is not None:
                        return mutant
                return self._load_expert(expert)
        return None

    def _try_load_mutant(self, expert_info: dict) -> Optional[nn.Module]:
        """Try to load a _MUTANT.pth variant if it exists."""
        base = expert_info['filename'].replace('.pth', '')
        mutant_filename = f"{base}_MUTANT.pth"
        mutant_path = self.experts_dir / mutant_filename
        if not mutant_path.exists():
            return None
        mutant_info = dict(expert_info)
        mutant_info['filename'] = mutant_filename
        model = self._load_expert(mutant_info)
        if model is not None:
            logging.info(f"Loaded MUTANT expert: {mutant_filename}")
        return model

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
            model_type = expert_info.get('model_type', 'lstm')
            if model_type == 'conv1d':
                model = Conv1DModel(
                    input_size=expert_info['input_size'],
                    hidden_size=expert_info['hidden_size'],
                    output_size=3,
                )
            else:
                model = LSTMModel(
                    input_size=expert_info['input_size'],
                    hidden_size=expert_info['hidden_size'],
                    output_size=3,
                    num_layers=2,
                )
            state_dict = torch.load(str(expert_path), map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            self.loaded_experts[filename] = model
            logging.info(f"Loaded expert: {filename} (type={model_type})")
            return model
        except Exception as e:
            logging.error(f"Failed to load expert: {e}")
            return None

    def _record_initial_mtime(self):
        """Record manifest modification time after initial load."""
        manifest_path = self.experts_dir / "top_50_manifest.json"
        if manifest_path.exists():
            self._manifest_mtime = manifest_path.stat().st_mtime

    def check_for_updates(self):
        """Check if manifest has been modified and hot-reload if needed."""
        manifest_path = self.experts_dir / "top_50_manifest.json"
        if not manifest_path.exists():
            return
        current_mtime = manifest_path.stat().st_mtime
        if current_mtime > self._manifest_mtime:
            try:
                self.loaded_experts.clear()
                self._load_manifest()
                self._manifest_mtime = current_mtime
                logging.info("HOT-RELOAD: LSTM experts cache cleared, manifest reloaded")
            except Exception as e:
                logging.warning(f"HOT-RELOAD: Failed to reload manifest: {e}")


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
        self.etare_experts = {}
        self._load_etare_experts()
        self.teqa_bridge = TEQABridge() if TEQA_ENABLED else None
        self.qnif_bridge = QNIFBridge() if QNIF_ENABLED else None
        self.feedback_poller = None
        if FEEDBACK_ENABLED and TEQA_ENABLED:
            try:
                from teqa_v3_neural_te import TEDomesticationTracker
                # Same DB file the live TEQA engine writes to
                self.feedback_poller = TradeOutcomePoller(
                    magic_numbers=[account_config.get('magic_number', 212001)],
                    domestication_tracker=TEDomesticationTracker(),
                )
            except Exception as e:
                logging.warning(f"[{account_key}] Feedback poller init failed: {e}")
        # VDJ adaptive immune memory -- receives live trade outcomes
        self.vdj_engine = None
        if VDJ_FEEDBACK_ENABLED:
            try:
                self.vdj_engine = VDJRecombinationEngine()
                logging.info(f"[{account_key}] VDJ feedback engine initialized "
                             f"({len(self.vdj_engine.memory_cells)} memory cells)")
            except Exception as e:
                logging.warning(f"[{account_key}] VDJ engine init failed: {e}")
        # CRISPR-Cas9 immune memory -- acquires spacers from losing trades
        self.crispr_bridge = None
        if CRISPR_FEEDBACK_ENABLED:
            try:
                self.crispr_bridge = CRISPRTEQABridge()
                logging.info(f"[{account_key}] CRISPR-Cas9 initialized "
                             f"({self.crispr_bridge.get_status_line()})")
            except Exception as e:
                logging.warning(f"[{account_key}] CRISPR init failed: {e}")
        # Protective Deletion -- suppresses toxic TE signal patterns
        self.protective_deletion = None
        if PROTECTIVE_DELETION_ENABLED:
            try:
                self.protective_deletion = ProtectiveDeletionTracker()
                logging.info(f"[{account_key}] Protective Deletion initialized")
            except Exception as e:
                logging.warning(f"[{account_key}] Protective Deletion init failed: {e}")
        # Toxoplasma regime detector with intraday DD protection
        self.toxoplasma_engine = None
        self.toxoplasma_bridge = None
        if TOXOPLASMA_ENABLED:
            try:
                self.toxoplasma_engine = ToxoplasmaEngine()
                self.toxoplasma_bridge = ToxoplasmaTEQABridge(
                    engine=self.toxoplasma_engine,
                )
                logging.info(
                    f"[{account_key}] Toxoplasma engine initialized "
                    f"(DD protection={'ON' if self.toxoplasma_engine.dd_tracker else 'OFF'})"
                )
            except Exception as e:
                logging.warning(f"[{account_key}] Toxoplasma init failed: {e}")
        # Track closed tickets to avoid double-counting in DD tracker
        self._dd_seen_tickets = set()
        self.starting_balance = 0.0

        # Loss cooldown — prevent re-entry after SL hit
        # Key: symbol, Value: datetime when cooldown expires
        self._loss_cooldown: Dict[str, datetime] = {}
        self.LOSS_COOLDOWN_MINUTES = LOSS_COOLDOWN_MINUTES  # from config

    def _load_etare_experts(self):
        """Load ETARE feedforward experts (preferred over Conv1D)."""
        if not ETARE_AVAILABLE:
            logging.info(f"[{self.account_key}] ETARE not available, using Conv1D fallback")
            return
        for symbol in ['BTCUSD', 'XAUUSD', 'ETHUSD']:
            expert = load_etare_expert(symbol)
            if expert:
                self.etare_experts[symbol] = expert
                logging.info(f"[{self.account_key}] ETARE expert loaded for {symbol} (WR={expert.win_rate*100:.1f}%)")

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
        except Exception:
            pass  # MT5 not initialized yet

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

        # Respect symbol's actual min/max
        lot = max(vol_min, lot)  # Use symbol's minimum, not hardcoded 0.01
        lot = min(lot, vol_max)  # Respect broker max

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

        # Try ETARE feedforward expert first (74% WR vs 54% Conv1D)
        etare = self.etare_experts.get(symbol)
        if etare and ETARE_AVAILABLE:
            rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 200)
            if rates_m5 is not None and len(rates_m5) >= 60:
                df_m5 = pd.DataFrame(rates_m5)
                df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
                state = prepare_etare_features(df_m5, symbol=symbol)
                if state is not None:
                    direction, confidence = etare.predict(state)
                    logging.info(f"[{self.account_key}][{symbol}] ETARE signal: {direction} ({confidence:.2f})")
                    if confidence >= ETARE_CONFIDENCE_THRESHOLD:
                        if direction == "BUY":
                            return regime, fidelity, Action.BUY, confidence
                        elif direction == "SELL":
                            return regime, fidelity, Action.SELL, confidence
                    return regime, fidelity, Action.HOLD, confidence

        # Fallback to Conv1D/LSTM expert
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

    def _apply_lipolysis(self, symbol: str, tighten_factor: float):
        """Tighten SL on losing positions (growth amplifier lipolysis).

        Only tightens, never loosens. Respects AGENT_SL_MIN.
        """
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        magic = self.account_config['magic_number']
        for pos in positions:
            if pos.magic != magic:
                continue
            if pos.profit >= 0:
                continue  # Only tighten losers
            if pos.sl == 0.0:
                continue  # No SL to tighten

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                continue
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            point = symbol_info.point

            # Current SL distance
            current_sl_dist = abs(pos.price_open - pos.sl)
            new_sl_dist = current_sl_dist * tighten_factor

            # Respect AGENT_SL_MIN
            if tick_value > 0 and pos.volume > 0:
                min_sl_dist = AGENT_SL_MIN / (tick_value * pos.volume) * tick_size
            else:
                min_sl_dist = 0
            new_sl_dist = max(new_sl_dist, min_sl_dist)

            if new_sl_dist >= current_sl_dist:
                continue  # Would loosen, skip

            # Calculate new SL price
            digits = symbol_info.digits
            if pos.type == 0:  # BUY
                new_sl = round(pos.price_open - new_sl_dist, digits)
            else:  # SELL
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
                        f"[{self.account_key}][{symbol}] LIPOLYSIS: "
                        f"tightened SL on #{pos.ticket} "
                        f"{current_sl_dist:.5f} -> {new_sl_dist:.5f} "
                        f"({(1-tighten_factor)*100:.1f}% tighter)"
                    )
                else:
                    logging.debug(
                        f"[{self.account_key}] Lipolysis SL modify failed: "
                        f"{result.retcode if result else 'None'}"
                    )
            except Exception as e:
                logging.debug(f"[{self.account_key}] Lipolysis error: {e}")

    def execute_trade(self, symbol: str, action: Action, confidence: float) -> bool:
        if action not in [Action.BUY, Action.SELL]:
            return False

        if self.has_position(symbol):
            logging.info(f"[{self.account_key}] Already have position in {symbol}")
            return False

        # Loss cooldown — don't re-enter a symbol that just hit SL
        cooldown_until = self._loss_cooldown.get(symbol)
        if cooldown_until and datetime.now() < cooldown_until:
            remaining = (cooldown_until - datetime.now()).total_seconds() / 60
            logging.info(f"[{self.account_key}][{symbol}] COOLDOWN: {remaining:.1f}min remaining after loss")
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

        # Calculate SL distance — use per-symbol risk (ETHUSD=$2, others=$1)
        symbol_risk = get_symbol_risk(symbol)
        if tick_value > 0 and lot > 0:
            sl_ticks = symbol_risk / (tick_value * lot)
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
            new_sl_ticks = sl_distance / tick_size if tick_size > 0 else 0
            if tick_value > 0 and new_sl_ticks > 0:
                new_lot = symbol_risk / (tick_value * new_sl_ticks)
                new_lot = max(symbol_info.volume_min, new_lot)
                new_lot = round(new_lot / symbol_info.volume_step) * symbol_info.volume_step
                if new_lot < lot:
                    lot = new_lot
                    logging.info(f"[{self.account_key}][{symbol}] Reduced lot to {lot} to maintain ${symbol_risk} max loss")

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
        logging.info(f"[{self.account_key}][{symbol}] SL Distance: {sl_distance:.5f} | Max Loss: ${symbol_risk} ({'override' if symbol_risk != MAX_LOSS_DOLLARS else 'default'})")

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

    def run_cycle(self) -> Dict[str, dict]:
        self.expert_loader.check_for_updates()

        results = {}
        if not self.check_risk_limits():
            return results

        for symbol in self.account_config['symbols']:
            regime, fidelity, action, confidence = self.analyze_symbol(symbol)
            trade_executed = False
            teqa_reason = ""

            # Apply TEQA quantum signal
            if self.teqa_bridge is not None and action is not None:
                lstm_action_str = action.name  # 'BUY', 'SELL', or 'HOLD'
                final_action_str, final_conf, lot_mult, teqa_reason = \
                    self.teqa_bridge.apply_to_lstm(lstm_action_str, confidence, symbol=symbol)
                # Map back to Action enum
                action_map = {'BUY': Action.BUY, 'SELL': Action.SELL, 'HOLD': Action.HOLD}
                action = action_map.get(final_action_str, Action.HOLD)
                confidence = final_conf
                logging.info(f"[{self.account_key}][{symbol}] {teqa_reason}")

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

            # Apply Toxoplasma DD protection gate (intraday drawdown throttle)
            toxo_reason = ""
            if self.toxoplasma_engine is not None and action in [Action.BUY, Action.SELL]:
                try:
                    # Get current unrealized P/L across all positions
                    unrealized_total = 0.0
                    all_positions = mt5.positions_get()
                    if all_positions:
                        for pos in all_positions:
                            if pos.magic == self.account_config['magic_number']:
                                unrealized_total += pos.profit

                    # Check DD state -- this is fast (in-memory)
                    dd_state = self.toxoplasma_engine.get_dd_state(self.account_key)
                    if dd_state and dd_state.get('dd_blocked', False):
                        action = Action.HOLD
                        confidence = 0.0
                        toxo_reason = (
                            f"[TOXO:DD] BLOCKED - daily DD at "
                            f"{dd_state['dd_pct']:.1f}% "
                            f"(${dd_state['closed_pnl']:.2f} closed)"
                        )
                    elif dd_state and dd_state.get('dd_level') == 'warning':
                        supp = dd_state.get('dopamine_suppression', 1.0)
                        toxo_reason = (
                            f"[TOXO:DD] WARNING - DD at "
                            f"{dd_state['dd_pct']:.1f}% | "
                            f"size_mult={supp:.3f}"
                        )
                    else:
                        toxo_reason = "[TOXO:DD] normal"

                    if toxo_reason:
                        logging.info(f"[{self.account_key}][{symbol}] {toxo_reason}")

                except Exception as te:
                    logging.debug(f"[{self.account_key}] Toxoplasma DD check error: {te}")

            if regime == Regime.CLEAN and action in [Action.BUY, Action.SELL]:
                trade_executed = self.execute_trade(symbol, action, confidence)

            # Growth Amplifier: Hyperplasia (second position on strong growth)
            growth_reason = ""
            hyperplasia_executed = False
            if trade_executed and self.teqa_bridge is not None:
                try:
                    signal = self.teqa_bridge.read_signal(symbol=symbol)
                    if signal and signal.growth_active and signal.growth_hyperplasia:
                        ratio = signal.growth_second_lot_ratio
                        if ratio > 0:
                            hyperplasia_executed = self.execute_trade(
                                symbol, action, confidence
                            )
                            growth_reason = (
                                f"GROWTH: hyperplasia={ratio:.0%} lot "
                                f"| signal={signal.growth_signal:.3f} "
                                f"| variant={signal.growth_variant}"
                            )
                            logging.info(f"[{self.account_key}][{symbol}] {growth_reason}")
                except Exception as ge:
                    logging.debug(f"[{self.account_key}] Growth hyperplasia check: {ge}")

            # Growth Amplifier: Lipolysis (tighten SL on losing positions)
            if self.teqa_bridge is not None:
                try:
                    signal = self.teqa_bridge.read_signal(symbol=symbol)
                    if signal and signal.growth_active and signal.growth_lipolysis_factor < 1.0:
                        self._apply_lipolysis(symbol, signal.growth_lipolysis_factor)
                except Exception as le:
                    logging.debug(f"[{self.account_key}] Growth lipolysis check: {le}")

            results[symbol] = {
                'regime': regime.value,
                'fidelity': fidelity,
                'action': action.name if action else 'NONE',
                'confidence': confidence,
                'trade_executed': trade_executed,
                'teqa': teqa_reason if teqa_reason else 'disabled',
                'qnif': qnif_reason if qnif_reason else 'disabled',
                'toxo_dd': toxo_reason if toxo_reason else 'disabled',
                'growth': growth_reason if growth_reason else 'inactive',
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

        # Feed closed trade outcomes to TE domestication tracker + VDJ immune memory
        if self.feedback_poller:
            try:
                outcomes = self.feedback_poller.poll()
                for o in outcomes:
                    logging.info(f"[{self.account_key}] DOMESTICATION: "
                                 f"{'WIN' if o['won'] else 'LOSS'} {o['symbol']} "
                                 f"${o['profit']:.2f} | TEs: {o['te_combo']}")
                    # Set loss cooldown — prevent rapid re-entry after SL hit
                    if not o['won']:
                        cooldown_until = datetime.now() + timedelta(minutes=self.LOSS_COOLDOWN_MINUTES)
                        self._loss_cooldown[o['symbol']] = cooldown_until
                        logging.warning(f"[{self.account_key}][{o['symbol']}] LOSS COOLDOWN: "
                                        f"no new trades until {cooldown_until.strftime('%H:%M:%S')} "
                                        f"({self.LOSS_COOLDOWN_MINUTES}min)")
                    # Feed to VDJ immune memory
                    if self.vdj_engine and o.get('active_tes'):
                        try:
                            self.vdj_engine.record_live_outcome(
                                symbol=o['symbol'],
                                won=o['won'],
                                profit=o['profit'],
                                active_tes=o['active_tes'],
                            )
                        except Exception as ve:
                            logging.debug(f"[{self.account_key}] VDJ feedback error: {ve}")
                    # Feed to CRISPR-Cas9 (acquire spacers from losses)
                    if self.crispr_bridge and not o['won'] and o.get('active_tes'):
                        try:
                            bars = mt5.copy_rates_from_pos(
                                o['symbol'], mt5.TIMEFRAME_M1, 0, 50)
                            if bars is not None and len(bars) >= 21:
                                bars_np = np.column_stack([
                                    bars['open'], bars['high'],
                                    bars['low'], bars['close'],
                                    bars['tick_volume'],
                                ])
                                self.crispr_bridge.on_trade_loss(
                                    bars=bars_np,
                                    symbol=o['symbol'],
                                    direction=o.get('direction', 0),
                                    loss_amount=abs(o['profit']),
                                    active_tes=o['active_tes'],
                                )
                        except Exception as ce:
                            logging.debug(f"[{self.account_key}] CRISPR feedback error: {ce}")
                    # Feed to Protective Deletion (learn from all outcomes)
                    if self.protective_deletion and o.get('active_tes'):
                        try:
                            acct = mt5.account_info()
                            dd_pct = 0.0
                            if acct and self.starting_balance > 0:
                                dd_pct = max(0, (self.starting_balance - acct.equity) / self.starting_balance)
                            self.protective_deletion.record_outcome(
                                active_tes=o['active_tes'],
                                won=o['won'],
                                profit=o['profit'],
                                account_drawdown_pct=dd_pct,
                            )
                        except Exception as pe:
                            logging.debug(f"[{self.account_key}] Protective Deletion error: {pe}")
                    # Feed to Toxoplasma DD tracker (intraday drawdown accumulation)
                    if self.toxoplasma_engine is not None:
                        try:
                            ticket = o.get('ticket', 0)
                            if ticket and ticket not in self._dd_seen_tickets:
                                self._dd_seen_tickets.add(ticket)
                                self.toxoplasma_engine.feed_dd_closed_trade(
                                    self.account_key, o['profit'],
                                )
                                dd_st = self.toxoplasma_engine.get_dd_state(self.account_key)
                                if dd_st and dd_st['dd_level'] != 'normal':
                                    logging.warning(
                                        f"[{self.account_key}] DD TRACKER: "
                                        f"{dd_st['dd_level'].upper()} "
                                        f"({dd_st['dd_pct']:.1f}%) "
                                        f"closed=${dd_st['closed_pnl']:.2f} "
                                        f"supp={dd_st['dopamine_suppression']:.3f}"
                                    )
                        except Exception as dde:
                            logging.debug(f"[{self.account_key}] Toxoplasma DD feed error: {dde}")
            except Exception as e:
                logging.debug(f"[{self.account_key}] Feedback poll error: {e}")

        return results


# ============================================================
# REST SCHEDULE GATE
# ============================================================

def is_rest_window(symbol: str) -> bool:
    """Check if the given symbol is currently in a rest window.
    Uses UTC time. Forex and crypto have separate rest schedules."""
    if not REST_SCHEDULE_ENABLED:
        return False

    now = datetime.utcnow()
    asset_class = get_asset_class(symbol)
    windows = REST_FOREX_WINDOWS if asset_class == 'forex' else REST_CRYPTO_WINDOWS

    day_names = {0: 'monday', 1: 'tuesday', 2: 'wednesday',
                 3: 'thursday', 4: 'friday', 5: 'saturday', 6: 'sunday'}
    current_day = day_names[now.weekday()]
    current_time = now.strftime('%H:%M')

    for window in windows:
        day_rule = window.get('day', '').lower()
        start = window.get('start', '00:00')
        end = window.get('end', '23:59')

        day_match = (day_rule == 'daily' or day_rule == current_day)
        time_match = (start <= current_time <= end)

        if day_match and time_match:
            return True

    return False


# ============================================================
# MAIN BRAIN
# ============================================================

class AtlasBrain:
    def __init__(self, config: BrainConfig = None):
        self.config = config or BrainConfig()
        self.traders: Dict[str, AccountTrader] = {}
        self._state_managers: Dict[str, StatePersistence] = {}
        self._was_resting: Dict[str, bool] = {}

    def initialize(self) -> bool:
        logging.info("Initializing Atlas brain")
        for key in ACCOUNTS:
            self.traders[key] = AccountTrader(key, ACCOUNTS[key], self.config)
            # Set up state persistence per account
            if STATE_PERSISTENCE_ENABLED:
                sp = StatePersistence(account_key=key)
                self._state_managers[key] = sp
                sp.restore_state(self.traders[key])
                self._was_resting[key] = False
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

                # Check rest schedule for this account's symbols
                trading_symbols = ACCOUNTS[key].get('symbols', [])
                all_resting = all(is_rest_window(s) for s in trading_symbols) if trading_symbols else False
                sp = self._state_managers.get(key)

                # Handle rest/wake transitions
                if all_resting and not self._was_resting.get(key, False):
                    logging.info(f"[{key}] ENTERING REST — persisting state")
                    if sp:
                        sp.save_state(trader)
                    self._was_resting[key] = True
                elif not all_resting and self._was_resting.get(key, False):
                    logging.info(f"[{key}] WAKING FROM REST — restoring state")
                    if sp:
                        sp.restore_state(trader)
                    self._was_resting[key] = False

                if all_resting:
                    print(f"  [REST] All symbols in rest window — skipping trade execution")
                    idx = (idx + 1) % len(account_keys)
                    time.sleep(self.config.CHECK_INTERVAL_SECONDS)
                    continue

                if trader.connect():
                    results = trader.run_cycle()
                    for symbol, data in results.items():
                        icon = "+" if data['regime'] == 'CLEAN' else "-"
                        status = "TRADED" if data['trade_executed'] else ""
                        print(f"  [{icon}] {symbol}: {data['regime']} | "
                              f"{data['action']} ({data['confidence']:.2f}) {status}")
                        if data.get('teqa') and data['teqa'] != 'disabled':
                            print(f"      {data['teqa']}")
                        if data.get('qnif') and data['qnif'] != 'disabled':
                            print(f"      {data['qnif']}")

                    # Show TEQA status line
                    if TEQA_ENABLED and trader.teqa_bridge:
                        print(f"  {trader.teqa_bridge.get_status_line()}")
                    # Show QNIF status line
                    if QNIF_ENABLED and trader.qnif_bridge:
                        print(f"  {trader.qnif_bridge.get_status_line()}")

                    # Show feedback loop status
                    if trader.feedback_poller:
                        stats = trader.feedback_poller.get_stats()
                        print(f"  [FEEDBACK] {stats['processed_tickets']} outcomes tracked")
                    # Show CRISPR status
                    if trader.crispr_bridge:
                        print(f"  {trader.crispr_bridge.get_status_line()}")
                    # Show Protective Deletion status
                    if trader.protective_deletion:
                        try:
                            af = trader.protective_deletion.get_allele_frequency_report()
                            print(f"  [DEL] patterns={af.total_patterns} | "
                                  f"het={af.het_patterns} hom={af.hom_patterns} | "
                                  f"health={af.health}")
                        except Exception as e:
                            logging.debug(f"Protective deletion display error: {e}")
                    # Show Toxoplasma DD protection status
                    if trader.toxoplasma_engine is not None:
                        try:
                            dd_st = trader.toxoplasma_engine.get_dd_state(key)
                            if dd_st:
                                lvl = dd_st['dd_level'].upper()
                                print(
                                    f"  [TOXO:DD] {lvl} | "
                                    f"DD={dd_st['dd_pct']:.1f}% | "
                                    f"closed=${dd_st['closed_pnl']:.2f} / "
                                    f"{dd_st['closed_trades']} trades | "
                                    f"supp={dd_st['dopamine_suppression']:.3f}"
                                    f"{' | BLOCKED' if dd_st['dd_blocked'] else ''}"
                                )
                        except Exception as e:
                            logging.debug(f"Toxoplasma display error: {e}")

                # Periodic state save (every 30 min)
                if sp and sp.should_periodic_save():
                    sp.save_state(trader)

                idx = (idx + 1) % len(account_keys)
                time.sleep(self.config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logging.info("Stopped — saving state before exit")
            for key, trader in self.traders.items():
                sp = self._state_managers.get(key)
                if sp:
                    sp.save_state(trader)
        finally:
            mt5.shutdown()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # CRITICAL: Acquire process lock BEFORE doing anything
    # This prevents multiple instances from fighting over the same MT5 account
    lock = ProcessLock("BRAIN_ATLAS", account="212000584")

    try:
        with lock:
            logging.info("=" * 60)
            logging.info("BRAIN_ATLAS - PROCESS LOCK ACQUIRED")
            logging.info("=" * 60)

            # Pre-launch validation - ensures experts are trained before trading
            TRADING_SYMBOLS = ['BTCUSD', 'ETHUSD']
            if not validate_prelaunch(symbols=TRADING_SYMBOLS):
                logging.error("Pre-launch validation failed. Exiting.")
                sys.exit(1)

            brain = AtlasBrain()
            brain.run_loop()

    except RuntimeError as e:
        logging.error("=" * 60)
        logging.error("PROCESS LOCK FAILURE")
        logging.error("=" * 60)
        logging.error(str(e))
        logging.error("")
        logging.error("Another instance of BRAIN_ATLAS is already running.")
        logging.error("This prevents duplicate trades on account 212000584.")
        logging.error("")
        logging.error("To stop all processes safely:")
        logging.error("  Run: SAFE_SHUTDOWN.bat")
        logging.error("")
        logging.error("To check running processes:")
        logging.error("  Run: python process_lock.py --list")
        logging.error("=" * 60)
        sys.exit(1)
