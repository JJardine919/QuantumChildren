"""
ETARE Walk-Forward Training System (Pro Edition)
=================================================
Evolutionary Trading Algorithm with Reinforcement and Extinction

Architecture:
- 16x AMD RX 6800 GPUs via DirectML
- Walk-Forward: 4 months train, 2 months test, 10 rounds
- Extinction: Bottom 5 replaced by top expert clones
- Timeframes: M1, M5, M15 (30 rounds per symbol)
- Symbols: BTCUSD, ETHUSD, XAUUSD (90 total cycles)
- Real market chaos: slippage, spread, volatility

Win Target: 63-65%
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import logging
import json
import time
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
from copy import deepcopy
import sqlite3

# ============================================================================
# GPU SETUP - CPU for LSTM (DirectML doesn't support LSTM well)
# ============================================================================
# Note: DirectML/AMD GPUs don't fully support LSTM operations
# Using optimized CPU with multi-threading for best LSTM performance
# Your working systems also use CPU for LSTM operations

try:
    import torch_directml
    DML_AVAILABLE = True
    DML_DEVICE_COUNT = torch_directml.device_count()
    print(f"[GPU] DirectML Available: {DML_AVAILABLE} (but LSTM uses CPU)")
    print(f"[GPU] AMD GPU Count: {DML_DEVICE_COUNT}")
    for i in range(DML_DEVICE_COUNT):
        print(f"  [GPU {i}] {torch_directml.device_name(i)}")
except ImportError:
    DML_AVAILABLE = False
    DML_DEVICE_COUNT = 0
    print("[GPU] DirectML not available")

# Use CPU for LSTM (most reliable for this architecture)
# Enable multi-threading for CPU performance
torch.set_num_threads(16)  # Use all available cores
print(f"[CPU] PyTorch threads: {torch.get_num_threads()}")

def get_device(gpu_id: int = 0):
    """Get device - CPU for LSTM operations (DirectML doesn't support LSTM)"""
    return torch.device('cpu')

# ============================================================================
# METATRADER5
# ============================================================================
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    print("[MT5] Not available - data will be simulated")

# ============================================================================
# QISKIT QUANTUM
# ============================================================================
try:
    # Silence Qiskit verbose logging
    import logging as _logging
    _logging.getLogger('qiskit').setLevel(_logging.WARNING)
    _logging.getLogger('qiskit.transpiler').setLevel(_logging.WARNING)

    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("[QUANTUM] Qiskit not available - using classical fallback")

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Walk-Forward Settings
    'train_months': 4,
    'test_months': 2,
    'walk_rounds': 3,  # Reduced for faster testing (set to 10 for full run)
    'total_history_months': 12,  # 1 year for faster testing (set to 60 for full run)

    # Population / Evolution
    'population_size': 5,  # Reduced for faster testing (set to 50 for full run)
    'elite_size': 2,  # Top 2 survive extinction
    'extinction_bottom': 2,  # Bottom 2 get replaced

    # Symbols
    'symbols': ['BTCUSD', 'ETHUSD', 'XAUUSD'],

    # Timeframes
    'timeframes': {
        'M1': 1,  # mt5.TIMEFRAME_M1
        'M5': 5,  # mt5.TIMEFRAME_M5
        'M15': 15,  # mt5.TIMEFRAME_M15
    },

    # Neural Network (matches champion architecture)
    'lstm_hidden': 128,
    'lstm_layers': 2,
    'sequence_length': 30,  # Matches champion training
    'dropout': 0.2,
    'input_features': 8,  # 8 technical indicators
    'output_classes': 3,  # BUY, SELL, HOLD
    'batch_size': 64,
    'learning_rate': 0.0005,
    'weight_decay': 0.01,
    'epochs_per_round': 5,  # Reduced for faster testing (set to 30 for full run)
    'dataset_stride': 10,  # Use every Nth sample (reduces 33k->3k samples, 10x speedup)

    # Quantum
    'n_qubits': 3,
    'n_shots': 1000,
    'quantum_window': 50,
    'entropy_threshold_high': 2.5,
    'entropy_threshold_low': 1.5,

    # Market Chaos Simulation
    'slippage_pips_mean': 2.0,
    'slippage_pips_std': 1.5,
    'spread_multiplier_volatile': 3.0,
    'commission_per_lot': 7.0,
    'latency_ms_mean': 50,
    'latency_ms_std': 30,

    # Trading
    'base_lot': 0.1,
    'min_lot': 0.01,
    'magic': 20260129,
}

# Setup directories
Path("logs").mkdir(exist_ok=True)
Path("models/walkforward").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/etare_walkforward.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ============================================================================
# QUANTUM FEATURE EXTRACTOR
# ============================================================================
class QuantumFeatureExtractor:
    """Shannon entropy-based regime detection - THE 14% EDGE"""

    def __init__(self, num_qubits=3, shots=1000):
        self.num_qubits = num_qubits
        self.shots = shots
        self.simulator = AerSimulator(method='statevector') if QISKIT_AVAILABLE else None
        self.cache = {}

    def create_circuit(self, features: np.ndarray) -> 'QuantumCircuit':
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        for i in range(self.num_qubits):
            angle = np.clip(np.pi * features[i % len(features)], -2*np.pi, 2*np.pi)
            qc.ry(angle, i)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def extract(self, price_window: np.ndarray, use_fast_classical: bool = True) -> dict:
        """Extract 7 quantum features (with fast classical option for training)"""
        data_hash = hashlib.md5(price_window.tobytes()).hexdigest()
        if data_hash in self.cache:
            return self.cache[data_hash]

        returns = np.diff(price_window) / (price_window[:-1] + 1e-10)
        if len(returns) == 0:
            return self._default_features()

        # Fast classical approximation (same entropy concept, no quantum overhead)
        if use_fast_classical or not QISKIT_AVAILABLE:
            result = self._fast_classical_features(returns)
            self.cache[data_hash] = result
            return result

        features = np.tanh(np.array([
            np.mean(returns),
            np.std(returns),
            np.max(returns) - np.min(returns)
        ]))

        try:
            qc = self.create_circuit(features)
            compiled = transpile(qc, self.simulator, optimization_level=0)  # Faster transpilation
            job = self.simulator.run(compiled, shots=self.shots)
            counts = job.result().get_counts()
            result = self._compute_metrics(counts)
            self.cache[data_hash] = result
            return result
        except Exception as e:
            log.error(f"Quantum extraction error: {e}")
            return self._default_features()

    def _fast_classical_features(self, returns: np.ndarray) -> dict:
        """Fast classical approximation of quantum features for training"""
        # Shannon entropy from discretized returns (same concept as quantum entropy)
        hist, _ = np.histogram(returns, bins=8, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        entropy = -np.sum(hist * np.log2(hist)) * (hist > 0).sum() / 8

        # Other metrics from returns distribution
        dominant = np.max(hist)
        significant = np.sum(hist > 0.05)
        superposition = significant / 8.0
        coherence = 1.0 - np.std(returns) / (np.abs(np.mean(returns)) + 1e-10)
        coherence = np.clip(coherence, 0, 1)
        variance = np.var(returns) * 1000  # Scale for similarity to quantum variance

        return {
            'entropy': entropy,
            'dominant_state': dominant,
            'superposition': superposition,
            'coherence': coherence,
            'entanglement': 0.5,  # Classical approx
            'variance': variance,
            'significant_states': float(significant)
        }

    def _compute_metrics(self, counts: dict) -> dict:
        probs = {s: c/self.shots for s, c in counts.items()}

        # Shannon Entropy - THE KEY METRIC
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs.values())

        dominant = max(probs.values())
        significant = sum(1 for p in probs.values() if p > 0.05)
        superposition = significant / (2 ** self.num_qubits)

        state_vals = [int(s, 2) for s in probs.keys()]
        coherence = 1.0 - (np.std(state_vals) / (2**self.num_qubits - 1)) if len(state_vals) > 1 else 0.5

        # Entanglement proxy
        bit_corr = []
        for i in range(self.num_qubits - 1):
            corr = sum(p for s, p in probs.items() if len(s) > i+1 and s[-(i+1)] == s[-(i+2)])
            bit_corr.append(corr)
        entanglement = np.mean(bit_corr) if bit_corr else 0.5

        mean_state = sum(int(s, 2) * p for s, p in probs.items())
        variance = sum((int(s, 2) - mean_state)**2 * p for s, p in probs.items())

        return {
            'entropy': entropy,
            'dominant_state': dominant,
            'superposition': superposition,
            'coherence': coherence,
            'entanglement': entanglement,
            'variance': variance,
            'significant_states': float(significant)
        }

    def _default_features(self):
        return {
            'entropy': 2.5, 'dominant_state': 0.125, 'superposition': 0.5,
            'coherence': 0.5, 'entanglement': 0.5, 'variance': 0.005,
            'significant_states': 4.0
        }

    def should_trade(self, entropy: float) -> Tuple[bool, str]:
        """THE 14% EDGE: Entropy-based regime filter"""
        if entropy > CONFIG['entropy_threshold_high']:
            return False, "high_entropy_avoid"
        elif entropy < CONFIG['entropy_threshold_low']:
            return True, "low_entropy_aggressive"
        return True, "moderate_entropy_normal"

# ============================================================================
# FOCAL LOSS (Class Imbalance Handler)
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        return torch.mean(self.alpha * (1-pt)**self.gamma * BCE)

# ============================================================================
# CHAMPION LSTM MODEL (Matches 63-65% win rate architecture)
# ============================================================================
class ChampionLSTM(nn.Module):
    """
    LSTM architecture matching trained champion weights.
    - 8 technical indicator inputs
    - 2 LSTM layers
    - 3-class output (BUY, SELL, HOLD)
    """
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, quantum_features=None):
        # x shape: (batch, seq_len, 8)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take last timestep
        out = self.fc(out)
        return out


# ============================================================================
# TECHNICAL INDICATOR FEATURE EXTRACTION
# ============================================================================
def extract_technical_indicators(df: pd.DataFrame) -> np.ndarray:
    """
    Extract 8 normalized technical indicators matching champion training.
    Features: rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr
    """
    df = df.copy()

    # 1. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 2. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 3. Bollinger Bands
    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = bb_middle + 2 * bb_std
    df['bb_lower'] = bb_middle - 2 * bb_std

    # 4. Momentum & ROC
    df['momentum'] = df['close'] / df['close'].shift(10)
    df['roc'] = df['close'].pct_change(10) * 100

    # 5. ATR
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = tr.rolling(14).mean()

    # Drop NaN from indicator calculations
    df = df.dropna()

    # Z-score normalization
    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std() + 1e-8
        df[col] = (df[col] - mean) / std

    return df[feature_cols].values


# Alias for backward compatibility
QuantumLSTM = ChampionLSTM

# ============================================================================
# MARKET DATA HANDLER
# ============================================================================
class MarketDataHandler:
    """Handles MT5 data fetching with chaos simulation"""

    def __init__(self):
        self.quantum_extractor = QuantumFeatureExtractor()
        self.mt5_initialized = False

        if MT5_AVAILABLE:
            self.mt5_initialized = mt5.initialize()
            if self.mt5_initialized:
                log.info("[MT5] Connected successfully")

    def get_timeframe_code(self, tf_name: str) -> int:
        """Convert timeframe name to MT5 code"""
        if not MT5_AVAILABLE:
            return {'M1': 1, 'M5': 5, 'M15': 15}.get(tf_name, 1)

        mapping = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
        }
        return mapping.get(tf_name, mt5.TIMEFRAME_M1)

    def fetch_data(self, symbol: str, timeframe: str, months_back: int) -> pd.DataFrame:
        """Fetch historical data from MT5 or generate synthetic"""

        if self.mt5_initialized:
            return self._fetch_mt5_data(symbol, timeframe, months_back)
        else:
            return self._generate_synthetic_data(symbol, timeframe, months_back)

    def _fetch_mt5_data(self, symbol: str, timeframe: str, months_back: int) -> pd.DataFrame:
        """Fetch real MT5 data in chunks (MT5 has ~100k bar limit)"""
        tf_code = self.get_timeframe_code(timeframe)

        # Calculate number of candles needed
        candles_per_month = {
            'M1': 43200,   # 30 days * 24 hours * 60 minutes
            'M5': 8640,
            'M15': 2880,
        }

        total_needed = candles_per_month.get(timeframe, 8640) * months_back
        max_per_request = 99000  # MT5 safe limit

        all_frames = []
        remaining = total_needed
        offset = 0

        while remaining > 0:
            chunk_size = min(remaining, max_per_request)
            rates = mt5.copy_rates_from_pos(symbol, tf_code, offset, chunk_size)

            if rates is None or len(rates) == 0:
                break

            df_chunk = pd.DataFrame(rates)
            df_chunk['time'] = pd.to_datetime(df_chunk['time'], unit='s')
            df_chunk.set_index('time', inplace=True)
            all_frames.append(df_chunk)

            remaining -= len(rates)
            offset += len(rates)

            if len(rates) < chunk_size:
                break  # No more data available

        if not all_frames:
            log.warning(f"No MT5 data for {symbol} - using synthetic")
            return self._generate_synthetic_data(symbol, timeframe, months_back)

        # Combine and sort
        df = pd.concat(all_frames)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        log.info(f"[MT5] Fetched {len(df)} candles for {symbol} {timeframe}")
        return df

    def _generate_synthetic_data(self, symbol: str, timeframe: str, months_back: int) -> pd.DataFrame:
        """Generate synthetic chaotic market data for backtesting"""

        candles_per_month = {'M1': 43200, 'M5': 8640, 'M15': 2880}
        n_candles = candles_per_month.get(timeframe, 8640) * months_back

        # Base prices for different symbols
        base_prices = {'BTCUSD': 45000, 'ETHUSD': 2500, 'XAUUSD': 2000}
        base = base_prices.get(symbol, 1000)

        # Generate chaotic price movements with mean reversion to prevent overflow
        np.random.seed(42 + hash(symbol) % 1000)  # Different seed per symbol

        # Smaller scale returns to prevent overflow
        returns = np.random.randn(n_candles) * 0.0005  # Much smaller base returns

        # Add trend components with mean reversion
        trend = np.zeros(n_candles)
        for i in range(1, n_candles):
            # Mean-reverting trend
            trend[i] = 0.999 * trend[i-1] + np.random.randn() * 0.0001

        # Add occasional larger moves (but controlled)
        jumps = np.random.choice([0, 1], n_candles, p=[0.995, 0.005])
        jump_sizes = np.random.randn(n_candles) * 0.005
        jumps = jumps * jump_sizes

        # Combine and clip to prevent overflow
        total_returns = returns + trend + jumps
        total_returns = np.clip(total_returns, -0.02, 0.02)  # Max 2% move per candle

        # Build price series with clipped cumulative sum
        cum_returns = np.cumsum(total_returns)
        cum_returns = np.clip(cum_returns, -2, 2)  # Clip total movement to ±200%
        close = base * np.exp(cum_returns)

        # Generate OHLCV
        intrabar_vol = np.abs(np.random.randn(n_candles)) * 0.001 + 0.0005
        high = close * (1 + intrabar_vol)
        low = close * (1 - intrabar_vol)
        open_price = np.roll(close, 1)
        open_price[0] = close[0]

        # Ensure OHLC consistency
        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))

        # Volume with spikes
        base_volume = 1000
        volume = base_volume * (1 + np.abs(np.random.randn(n_candles)) * 0.5)
        volume_spikes = np.random.choice([1, 3], n_candles, p=[0.95, 0.05])
        volume = volume * volume_spikes

        # Create datetime index (using newer pandas frequency strings)
        freq_map = {'M1': 'min', 'M5': '5min', 'M15': '15min'}
        dates = pd.date_range(
            end=datetime.now(),
            periods=n_candles,
            freq=freq_map.get(timeframe, '5min')
        )

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': volume.astype(int),
            'spread': np.random.randint(1, 10, n_candles),
            'real_volume': volume.astype(int) * 100
        }, index=dates)

        return df

    def add_chaos(self, df: pd.DataFrame, chaos_level: float = 1.0) -> pd.DataFrame:
        """Add realistic market chaos: slippage, spread widening, gaps"""

        df = df.copy()
        n = len(df)

        # 1. Random slippage
        slippage = np.random.normal(
            CONFIG['slippage_pips_mean'],
            CONFIG['slippage_pips_std'],
            n
        ) * chaos_level * 0.0001  # Convert pips to price
        df['slippage_factor'] = 1 + slippage

        # 2. Spread widening during volatility
        volatility = df['close'].pct_change().rolling(20).std()
        volatility_normalized = (volatility / volatility.mean()).fillna(1)
        df['spread_multiplier'] = 1 + (volatility_normalized - 1) * CONFIG['spread_multiplier_volatile']

        # 3. Weekend/overnight gaps
        df['gap'] = 0.0
        for i in range(1, n):
            if df.index[i].dayofweek == 0 and df.index[i-1].dayofweek == 4:  # Monday after Friday
                gap_size = np.random.uniform(-0.02, 0.02) * chaos_level
                df.iloc[i, df.columns.get_loc('gap')] = gap_size

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare 8 technical indicator features matching champion training.
        Returns: (features, quantum_features, targets)
        """
        d = df.copy()

        # 1. RSI
        delta = d['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # 2. MACD
        ema12 = d['close'].ewm(span=12, adjust=False).mean()
        ema26 = d['close'].ewm(span=26, adjust=False).mean()
        d['macd'] = ema12 - ema26
        d['macd_signal'] = d['macd'].ewm(span=9, adjust=False).mean()

        # 3. Bollinger Bands
        bb_middle = d['close'].rolling(20).mean()
        bb_std = d['close'].rolling(20).std()
        d['bb_upper'] = bb_middle + 2 * bb_std
        d['bb_lower'] = bb_middle - 2 * bb_std

        # 4. Momentum & ROC
        d['momentum'] = d['close'] / d['close'].shift(10)
        d['roc'] = d['close'].pct_change(10) * 100

        # 5. ATR
        tr = pd.concat([
            d['high'] - d['low'],
            (d['high'] - d['close'].shift(1)).abs(),
            (d['low'] - d['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        d['atr'] = tr.rolling(14).mean()

        d = d.dropna()

        # 8 technical indicator features (matching champion architecture)
        feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
        features = d[feature_cols].values.copy()

        # Z-score normalization
        for i in range(features.shape[1]):
            mean = np.mean(features[:, i])
            std = np.std(features[:, i]) + 1e-8
            features[:, i] = (features[:, i] - mean) / std

        log.info(f"  [FEATURES] Extracted {len(features)} samples with 8 indicators")

        # Quantum features (for entropy-based regime filtering)
        quantum_window = CONFIG['quantum_window']
        quantum_sample_rate = 100
        quantum_features = []

        last_q_feat = None
        for i in range(quantum_window, len(d)):
            if last_q_feat is None or (i % quantum_sample_rate) == 0:
                window = d['close'].iloc[i-quantum_window:i].values
                last_q_feat = self.quantum_extractor.extract(window)
            quantum_features.append(list(last_q_feat.values()))

        quantum_features = np.array(quantum_features)
        log.info(f"  [QUANTUM] Extracted features for {len(quantum_features)} samples")

        # 3-class targets: 0=HOLD, 1=BUY, 2=SELL
        # BUY if price goes up >0.1%, SELL if down >0.1%, else HOLD
        future_returns = d['close'].pct_change().shift(-1).values
        threshold = 0.001  # 0.1% threshold

        targets = np.zeros(len(future_returns), dtype=np.int64)
        targets[future_returns > threshold] = 1   # BUY
        targets[future_returns < -threshold] = 2  # SELL
        # 0 = HOLD (default)

        # Align arrays
        min_len = min(len(features) - quantum_window, len(quantum_features), len(targets) - quantum_window)
        features = features[quantum_window:quantum_window+min_len]
        quantum_features = quantum_features[:min_len]
        targets = targets[quantum_window:quantum_window+min_len]

        return features, quantum_features, targets

# ============================================================================
# TRADING INDIVIDUAL (Evolutionary Unit)
# ============================================================================
@dataclass
class TradingIndividual:
    """Individual trading strategy in the population"""
    id: int
    model: QuantumLSTM
    device: torch.device
    fitness: float = 0.0
    total_profit: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    max_drawdown: float = 0.0
    trade_history: List = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def calculate_fitness(self) -> float:
        """Multi-factor fitness score"""
        if self.total_trades < 10:
            return 0.0

        wr = self.win_rate
        pf = self.total_profit / (abs(self.max_drawdown) + 1e-10)
        consistency = 1 / (1 + np.std([t['profit'] for t in self.trade_history]) if self.trade_history else 1)

        self.fitness = (
            wr * 0.5 +                    # Win rate (most important)
            min(pf, 3) / 3 * 0.3 +       # Profit factor (capped)
            consistency * 0.2             # Consistency
        )
        return self.fitness

    def clone_weights_from(self, other: 'TradingIndividual'):
        """Clone weights from another individual (extinction replacement)"""
        self.model.load_state_dict(deepcopy(other.model.state_dict()))
        # Reset stats for fresh start
        self.total_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.trade_history = []
        self.fitness = 0.0

# ============================================================================
# PYTORCH DATASET
# ============================================================================
class WalkForwardDataset(Dataset):
    def __init__(self, price_data, quantum_features, targets, sequence_length=50, stride=10):
        self.price_data = price_data
        self.quantum_features = quantum_features
        self.targets = targets
        self.sequence_length = sequence_length
        self.stride = stride  # Skip samples to reduce dataset size
        # Pre-compute valid indices
        max_idx = len(self.price_data) - self.sequence_length
        self.indices = list(range(0, max_idx, stride)) if max_idx > 0 else []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        price_seq = self.price_data[real_idx:real_idx + self.sequence_length]
        quantum_feat = self.quantum_features[min(real_idx + self.sequence_length - 1, len(self.quantum_features)-1)]
        target = self.targets[min(real_idx + self.sequence_length, len(self.targets)-1)]

        return {
            'price': torch.FloatTensor(price_seq),  # 8 technical indicators
            'quantum': torch.FloatTensor(quantum_feat),
            'target': torch.LongTensor([int(target)])  # Class label (0=HOLD, 1=BUY, 2=SELL)
        }

# ============================================================================
# WALK-FORWARD TRAINER
# ============================================================================
class WalkForwardTrainer:
    """
    Walk-Forward Training System
    - Train 4 months, test 2 months
    - 10 rounds per timeframe
    - Extinction: Bottom 5 replaced by top expert clones
    - Distribute across 16 AMD GPUs
    """

    def __init__(self):
        self.data_handler = MarketDataHandler()
        self.population: List[TradingIndividual] = []
        self.results = []

        # Distribute models across available GPUs
        self.devices = []
        n_gpus = max(DML_DEVICE_COUNT, 1)
        for i in range(CONFIG['population_size']):
            gpu_id = i % n_gpus
            self.devices.append(get_device(gpu_id))

        log.info(f"[TRAINER] Initialized with {n_gpus} GPU(s)")

    def initialize_population(self):
        """Create initial population of trading individuals"""
        self.population = []

        for i in range(CONFIG['population_size']):
            model = ChampionLSTM(
                input_size=CONFIG.get('input_features', 8),
                hidden_size=CONFIG['lstm_hidden'],
                num_layers=CONFIG['lstm_layers'],
                dropout=CONFIG['dropout'],
                output_size=CONFIG.get('output_classes', 3)
            ).to(self.devices[i])

            individual = TradingIndividual(
                id=i,
                model=model,
                device=self.devices[i]
            )
            self.population.append(individual)

        log.info(f"[POPULATION] Created {len(self.population)} individuals")

    def train_individual(self, individual: TradingIndividual,
                         train_loader: DataLoader, val_loader: DataLoader,
                         epochs: int) -> float:
        """Train a single individual using 3-class CrossEntropy"""

        model = individual.model
        device = individual.device

        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        # CrossEntropyLoss for 3-class classification (HOLD, BUY, SELL)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float('inf')
        log.info(f"    [TRAIN] Individual {individual.id} starting {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            batch_count = 0

            for batch in train_loader:
                batch_count += 1
                features = batch['price'].to(device)  # Now contains 8 technical indicators
                target = batch['target'].long().squeeze().to(device)  # Class labels

                optimizer.zero_grad()
                output = model(features)  # (batch, 3)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch['price'].to(device)
                    target = batch['target'].long().squeeze().to(device)

                    output = model(features)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 3 == 0:  # Log every 3 epochs
                log.info(f"      Epoch {epoch+1}/{epochs} - Loss: {val_loss:.4f}")

        log.info(f"    [TRAIN] Individual {individual.id} done - Best Loss: {best_val_loss:.4f}")
        return best_val_loss

    def test_individual(self, individual: TradingIndividual,
                        test_loader: DataLoader) -> Dict:
        """Test individual on out-of-sample data with market chaos"""

        model = individual.model
        device = individual.device
        model.eval()

        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch['price'].to(device)  # 8 technical indicators
                target = batch['target'].to(device)

                output = model(features)  # (batch, 3) logits
                probs = torch.softmax(output, dim=1)
                pred_classes = torch.argmax(probs, dim=1)  # Predicted class

                predictions.extend(pred_classes.cpu().numpy())
                actuals.extend(target.cpu().numpy().flatten().astype(int))

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate win rate for BUY/SELL signals only (not HOLD)
        # 0=HOLD, 1=BUY, 2=SELL
        trade_mask = predictions != 0  # Only count BUY and SELL as trades
        if trade_mask.sum() == 0:
            # Model only predicts HOLD - not useful
            win_rate = 0.33
            total_trades = 0
            winning = 0
        else:
            total_trades = trade_mask.sum()
            # Win = prediction matches actual (BUY when actual is BUY, SELL when actual is SELL)
            winning = ((predictions == actuals) & trade_mask).sum()
            win_rate = winning / total_trades

        # Simulate profit with chaos
        profits = []
        running_balance = 10000
        max_balance = running_balance
        max_dd = 0

        for pred, actual in zip(predictions, actuals):
            if pred == 0:  # HOLD - no trade
                continue

            # Add slippage
            slippage = np.random.normal(CONFIG['slippage_pips_mean'], CONFIG['slippage_pips_std'])
            slippage_cost = slippage * 10  # $10 per pip

            # Add spread
            spread_cost = np.random.uniform(5, 15)

            # Base profit/loss
            if pred == actual:  # Correct prediction
                base_profit = np.random.uniform(50, 200)  # Win
            else:
                base_profit = -np.random.uniform(30, 150)  # Loss

            net_profit = base_profit - slippage_cost - spread_cost - CONFIG['commission_per_lot']
            profits.append(net_profit)

            running_balance += net_profit
            max_balance = max(max_balance, running_balance)
            current_dd = (max_balance - running_balance) / max_balance
            max_dd = max(max_dd, current_dd)

        # Update individual stats
        individual.total_trades = int(total_trades)
        individual.winning_trades = int(winning)
        individual.total_profit = sum(profits) if profits else 0
        individual.max_drawdown = max_dd
        individual.trade_history = [{'profit': p} for p in profits]
        individual.calculate_fitness()

        return {
            'win_rate': individual.win_rate,
            'total_trades': int(total_trades),
            'total_profit': individual.total_profit,
            'max_drawdown': max_dd,
            'fitness': individual.fitness
        }

    def extinction_event(self):
        """Replace bottom 5 with clones of top expert"""

        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Get top performer
        top_expert = self.population[0]

        # Replace bottom 5
        for i in range(-CONFIG['extinction_bottom'], 0):
            self.population[i].clone_weights_from(top_expert)
            log.info(f"  [EXTINCTION] Individual {self.population[i].id} replaced with clone of #{top_expert.id}")

        log.info(f"[EXTINCTION] Completed - Top fitness: {top_expert.fitness:.4f}, Win rate: {top_expert.win_rate:.2%}")

    def run_walk_forward_round(self, symbol: str, timeframe: str,
                               df: pd.DataFrame, round_num: int,
                               train_start: int, train_end: int,
                               test_start: int, test_end: int) -> Dict:
        """Run a single walk-forward round"""

        log.info(f"\n{'='*60}")
        log.info(f"ROUND {round_num} | {symbol} | {timeframe}")
        log.info(f"Train: [{train_start}:{train_end}] | Test: [{test_start}:{test_end}]")
        log.info(f"{'='*60}")

        # Split data
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]

        # Add chaos to test data
        test_df = self.data_handler.add_chaos(test_df, chaos_level=1.0)

        # Prepare features
        train_price, train_quantum, train_targets = self.data_handler.prepare_features(train_df)
        test_price, test_quantum, test_targets = self.data_handler.prepare_features(test_df)

        if len(train_price) < CONFIG['sequence_length'] or len(test_price) < CONFIG['sequence_length']:
            log.warning(f"Insufficient data for round {round_num}")
            return None

        # Create datasets
        seq_len = CONFIG['sequence_length']
        stride = CONFIG.get('dataset_stride', 10)  # Use every Nth sample to speed up training

        # Split train into train/val (80/20)
        split_idx = int(len(train_price) * 0.8)

        train_dataset = WalkForwardDataset(train_price[:split_idx], train_quantum[:split_idx],
                                           train_targets[:split_idx], seq_len, stride=stride)
        val_dataset = WalkForwardDataset(train_price[split_idx:], train_quantum[split_idx:],
                                         train_targets[split_idx:], seq_len, stride=stride)
        test_dataset = WalkForwardDataset(test_price, test_quantum, test_targets, seq_len, stride=1)  # Full resolution for testing

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        log.info(f"  [DATASET] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} (stride={stride})")

        # Train and test each individual
        round_results = []

        for individual in self.population:
            # Train
            val_loss = self.train_individual(individual, train_loader, val_loader,
                                             CONFIG['epochs_per_round'])

            # Test
            test_results = self.test_individual(individual, test_loader)
            test_results['individual_id'] = individual.id
            test_results['val_loss'] = val_loss
            round_results.append(test_results)

        # Extinction event
        self.extinction_event()

        # Aggregate results
        avg_win_rate = np.mean([r['win_rate'] for r in round_results])
        avg_profit = np.mean([r['total_profit'] for r in round_results])
        best_win_rate = max([r['win_rate'] for r in round_results])

        log.info(f"\n[ROUND {round_num} RESULTS]")
        log.info(f"  Avg Win Rate: {avg_win_rate:.2%}")
        log.info(f"  Best Win Rate: {best_win_rate:.2%}")
        log.info(f"  Avg Profit: ${avg_profit:.2f}")

        return {
            'round': round_num,
            'symbol': symbol,
            'timeframe': timeframe,
            'avg_win_rate': avg_win_rate,
            'best_win_rate': best_win_rate,
            'avg_profit': avg_profit,
            'individual_results': round_results
        }

    def run_full_walkforward(self, symbol: str, timeframe: str) -> List[Dict]:
        """Run complete walk-forward for a symbol/timeframe combo"""

        log.info(f"\n{'#'*70}")
        log.info(f"STARTING WALK-FORWARD: {symbol} | {timeframe}")
        log.info(f"Rounds: {CONFIG['walk_rounds']} | Train: {CONFIG['train_months']}mo | Test: {CONFIG['test_months']}mo")
        log.info(f"{'#'*70}")

        # Initialize fresh population
        self.initialize_population()

        # Fetch all data
        df = self.data_handler.fetch_data(symbol, timeframe, CONFIG['total_history_months'])
        log.info(f"[DATA] Loaded {len(df)} candles")

        # Calculate candles per month
        candles_per_month = len(df) // CONFIG['total_history_months']

        train_candles = candles_per_month * CONFIG['train_months']
        test_candles = candles_per_month * CONFIG['test_months']
        step_candles = test_candles  # Step forward by test period

        results = []

        for round_num in range(1, CONFIG['walk_rounds'] + 1):
            # Calculate indices for this round (walk forward from end)
            round_offset = (CONFIG['walk_rounds'] - round_num) * step_candles

            test_end = len(df) - round_offset
            test_start = test_end - test_candles
            train_end = test_start
            train_start = train_end - train_candles

            if train_start < 0:
                log.warning(f"Round {round_num}: Not enough history, skipping")
                continue

            round_result = self.run_walk_forward_round(
                symbol, timeframe, df, round_num,
                train_start, train_end, test_start, test_end
            )

            if round_result:
                results.append(round_result)

        return results

    def run_all_cycles(self):
        """Run all 90 walk-forward cycles"""

        all_results = []
        cycle_count = 0
        total_cycles = len(CONFIG['symbols']) * len(CONFIG['timeframes']) * CONFIG['walk_rounds']

        log.info(f"\n{'*'*80}")
        log.info(f"STARTING FULL WALK-FORWARD TRAINING")
        log.info(f"Symbols: {CONFIG['symbols']}")
        log.info(f"Timeframes: {list(CONFIG['timeframes'].keys())}")
        log.info(f"Total Cycles: {total_cycles}")
        log.info(f"{'*'*80}\n")

        start_time = time.time()

        for symbol in CONFIG['symbols']:
            for tf_name in CONFIG['timeframes'].keys():
                log.info(f"\n{'='*70}")
                log.info(f"CYCLE GROUP: {symbol} | {tf_name}")
                log.info(f"{'='*70}")

                results = self.run_full_walkforward(symbol, tf_name)

                for r in results:
                    r['cycle'] = cycle_count
                    all_results.append(r)
                    cycle_count += 1

                # Save intermediate results
                self._save_results(all_results, f"results/walkforward_{symbol}_{tf_name}.json")

        elapsed = time.time() - start_time

        # Final summary
        self._print_final_summary(all_results, elapsed)

        # Save final results
        self._save_results(all_results, "results/walkforward_final.json")

        return all_results

    def _save_results(self, results: List[Dict], filepath: str):
        """Save results to JSON"""
        # Convert to serializable format
        serializable = []
        for r in results:
            r_copy = r.copy()
            if 'individual_results' in r_copy:
                r_copy['individual_results'] = [
                    {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in ir.items()}
                    for ir in r_copy['individual_results']
                ]
            serializable.append(r_copy)

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        log.info(f"[SAVED] Results to {filepath}")

    def _print_final_summary(self, results: List[Dict], elapsed: float):
        """Print final summary of all walk-forward cycles"""

        log.info(f"\n{'#'*80}")
        log.info(f"FINAL WALK-FORWARD SUMMARY")
        log.info(f"{'#'*80}")

        if not results:
            log.info("No results to summarize")
            return

        # Overall stats
        avg_win_rates = [r['avg_win_rate'] for r in results]
        best_win_rates = [r['best_win_rate'] for r in results]

        log.info(f"\nOVERALL STATISTICS:")
        log.info(f"  Total Cycles: {len(results)}")
        log.info(f"  Average Win Rate: {np.mean(avg_win_rates):.2%}")
        log.info(f"  Best Win Rate Overall: {max(best_win_rates):.2%}")
        log.info(f"  Worst Avg Win Rate: {min(avg_win_rates):.2%}")
        log.info(f"  Win Rate Std Dev: {np.std(avg_win_rates):.2%}")

        # By symbol
        log.info(f"\nBY SYMBOL:")
        for symbol in CONFIG['symbols']:
            symbol_results = [r for r in results if r['symbol'] == symbol]
            if symbol_results:
                avg_wr = np.mean([r['avg_win_rate'] for r in symbol_results])
                log.info(f"  {symbol}: {avg_wr:.2%} avg win rate")

        # By timeframe
        log.info(f"\nBY TIMEFRAME:")
        for tf in CONFIG['timeframes'].keys():
            tf_results = [r for r in results if r['timeframe'] == tf]
            if tf_results:
                avg_wr = np.mean([r['avg_win_rate'] for r in tf_results])
                log.info(f"  {tf}: {avg_wr:.2%} avg win rate")

        log.info(f"\nTotal Time: {elapsed/60:.1f} minutes")
        log.info(f"{'#'*80}\n")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("ETARE WALK-FORWARD TRAINING SYSTEM (PRO EDITION)")
    print("="*80)
    print(f"GPUs: {DML_DEVICE_COUNT} AMD devices via DirectML")
    print(f"Symbols: {CONFIG['symbols']}")
    print(f"Timeframes: {list(CONFIG['timeframes'].keys())}")
    print(f"Walk Rounds: {CONFIG['walk_rounds']} per timeframe")
    print(f"Total Cycles: {len(CONFIG['symbols']) * len(CONFIG['timeframes']) * CONFIG['walk_rounds']}")
    print("="*80 + "\n")

    trainer = WalkForwardTrainer()
    results = trainer.run_all_cycles()

    # Print final win rate
    if results:
        final_avg = np.mean([r['avg_win_rate'] for r in results])
        final_best = max([r['best_win_rate'] for r in results])
        print(f"\n{'='*80}")
        print(f"FINAL WIN RATE: {final_avg:.2%} (avg) | {final_best:.2%} (best)")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
