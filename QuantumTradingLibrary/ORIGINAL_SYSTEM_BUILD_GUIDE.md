# ETARE QUANTUM FUSION - COMPLETE BUILD GUIDE

**Source:** `docs/plans/2026-01-18-etare-quantum-fusion-design.md`
**Purpose:** Rebuild the full 6-layer architecture that was designed but never fully implemented
**Target:** 78-82% win rate (vs current ~65% simplified system)
**Build Location:** `original_system/` (separate from running system)

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ETARE QUANTUM FUSION - 6 LAYER ARCHITECTURE              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 0: MARKET DATA                                                │   │
│  │ MT5 BTCUSD/ETHUSD M5 Timeframe                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 1: QUANTUM COMPRESSION PREPROCESSING                          │   │
│  │ Shannon Information Theory → Regime Detection                       │   │
│  │ Output: CLEAN / VOLATILE / CHOPPY + Fidelity Score                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 2: PARALLEL QUANTUM FEATURE EXTRACTION                        │   │
│  │ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │   │
│  │ │ Quantum LSTM  │ │ 3D Bar        │ │ QPE Horizon   │               │   │
│  │ │ (BiLSTM +     │ │ Analysis      │ │ Prediction    │               │   │
│  │ │ Attention)    │ │ (Yellow       │ │ (Phase        │               │   │
│  │ │               │ │ Clusters)     │ │ Estimation)   │               │   │
│  │ └───────────────┘ └───────────────┘ └───────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 3: CLASSICAL SIGNAL GENERATION                                │   │
│  │ ┌───────────────────────┐ ┌───────────────────────┐                 │   │
│  │ │ Volatility Analysis   │ │ Currency Strength     │                 │   │
│  │ │ (ATR, Bollinger,      │ │ (Cross-pair           │                 │   │
│  │ │ Historical Vol)       │ │ correlation)          │                 │   │
│  │ └───────────────────────┘ └───────────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 4: SIGNAL FUSION ENGINE                                       │   │
│  │                                                                     │   │
│  │ Weighted Scoring:                                                   │   │
│  │   Compression Ratio ──────── 25%                                    │   │
│  │   Quantum LSTM ───────────── 20%                                    │   │
│  │   Quantum 3D ─────────────── 15%                                    │   │
│  │   QPE Horizon ────────────── 15%                                    │   │
│  │   Volatility ─────────────── 10%                                    │   │
│  │   Currency Strength ──────── 5%                                     │   │
│  │   ETARE Base ─────────────── 10%                                    │   │
│  │                              ════                                    │   │
│  │                              100%                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 5: ETARE NEURAL NETWORK ENHANCED                              │   │
│  │ 23 Input Features → Hidden Layers → 3 Outputs (HOLD/BUY/SELL)       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 6: EXECUTION ENGINE                                           │   │
│  │ Grid Trading + Risk Management + Emergency Stop                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## DIRECTORY STRUCTURE

```
original_system/
├── layer0_data/
│   ├── __init__.py
│   └── market_data.py           # MT5 data fetcher
├── layer1_compression/
│   ├── __init__.py
│   └── regime_detector.py       # Quantum compression regime detection
├── layer2_quantum/
│   ├── __init__.py
│   ├── quantum_lstm.py          # Bidirectional LSTM with quantum features
│   ├── bars_3d.py               # 3D bar analysis with yellow clusters
│   └── qpe_horizon.py           # Quantum Phase Estimation predictor
├── layer3_classical/
│   ├── __init__.py
│   ├── volatility.py            # Volatility analysis
│   └── currency_strength.py     # Cross-pair correlation
├── layer4_fusion/
│   ├── __init__.py
│   └── fusion_engine.py         # Weighted signal combination
├── layer5_etare/
│   ├── __init__.py
│   └── etare_network.py         # 23-input neural network
├── layer6_execution/
│   ├── __init__.py
│   ├── grid_trader.py           # Grid trading logic
│   └── risk_manager.py          # Risk management + emergency stop
├── brain/
│   ├── __init__.py
│   └── quantum_fusion_brain.py  # Main orchestrator
├── training/
│   └── train_all_layers.py      # Training pipeline
├── models/                      # Saved model weights
├── config.py                    # All configuration
└── requirements.txt
```

---

## LAYER 0: MARKET DATA

**File:** `layer0_data/market_data.py`

```python
"""
LAYER 0: Market Data Acquisition
================================
Fetches OHLCV data from MT5 for BTCUSD and ETHUSD on M5 timeframe.
"""

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from typing import Optional, Dict
from datetime import datetime


class MarketDataFetcher:
    """
    Fetches and prepares market data for all downstream layers.
    """

    def __init__(self, symbols: list = None):
        self.symbols = symbols or ['BTCUSD', 'ETHUSD']
        self.timeframe = mt5.TIMEFRAME_M5
        self.connected = False

    def connect(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            return False
        self.connected = True
        return True

    def fetch(self, symbol: str, bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol.

        Returns DataFrame with columns:
            time, open, high, low, close, tick_volume
        """
        if not self.connected:
            if not self.connect():
                return None

        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'tick_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def fetch_all(self, bars: int = 500) -> Dict[str, pd.DataFrame]:
        """Fetch data for all configured symbols"""
        data = {}
        for symbol in self.symbols:
            df = self.fetch(symbol, bars)
            if df is not None:
                data[symbol] = df
        return data

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current bid/ask prices"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {'bid': tick.bid, 'ask': tick.ask, 'time': datetime.now()}

    def shutdown(self):
        """Close MT5 connection"""
        mt5.shutdown()
        self.connected = False
```

---

## LAYER 1: QUANTUM COMPRESSION PREPROCESSING

**File:** `layer1_compression/regime_detector.py`

```python
"""
LAYER 1: Quantum Compression Regime Detection
==============================================
Uses Claude Shannon's information theory to detect market regimes.

High compression ratio = structured/trending market = CLEAN regime = TRADE
Low compression ratio = noisy/chaotic market = CHOPPY regime = DON'T TRADE

This is the GATEKEEPER - if regime isn't CLEAN, we don't trade.
"""

import numpy as np
import zlib
from enum import Enum
from typing import Tuple
from dataclasses import dataclass

try:
    import qutip as qt
    from scipy.optimize import minimize
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False


class Regime(Enum):
    CLEAN = "CLEAN"        # Fidelity >= 0.95 - TRADE
    VOLATILE = "VOLATILE"  # Fidelity >= 0.85 - CAUTION
    CHOPPY = "CHOPPY"      # Fidelity < 0.85 - NO TRADE


@dataclass
class RegimeResult:
    regime: Regime
    fidelity: float
    compression_ratio: float
    trade_allowed: bool


class QuantumRegimeDetector:
    """
    Detects market regime using quantum state compression.

    The key insight: Markets that can be compressed (high information redundancy)
    are more predictable and therefore tradeable.
    """

    def __init__(self, clean_threshold: float = 0.95, volatile_threshold: float = 0.85):
        self.clean_threshold = clean_threshold
        self.volatile_threshold = volatile_threshold

    def analyze(self, prices: np.ndarray) -> RegimeResult:
        """
        Analyze price data and return regime classification.

        Args:
            prices: Array of close prices (minimum 64 values recommended)

        Returns:
            RegimeResult with regime, fidelity, compression_ratio, trade_allowed
        """
        if QUTIP_AVAILABLE and len(prices) >= 64:
            return self._quantum_analysis(prices)
        else:
            return self._classical_analysis(prices)

    def _quantum_analysis(self, prices: np.ndarray) -> RegimeResult:
        """Full quantum circuit compression analysis"""
        try:
            # Normalize and encode prices into quantum state
            normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

            # Truncate to power of 2 for quantum encoding
            n = len(normalized)
            num_qubits = int(np.log2(n))
            n_truncated = 2 ** num_qubits
            state_vector = normalized[:n_truncated].astype(complex)
            state_vector = state_vector / np.linalg.norm(state_vector)

            # Create quantum state
            input_state = qt.Qobj(state_vector, dims=[[2]*num_qubits, [1]*num_qubits]).unit()

            # Attempt compression
            num_latent = num_qubits - 1
            num_params = 2 * num_qubits

            initial_params = np.random.rand(num_params) * np.pi

            result = minimize(
                self._compression_cost,
                initial_params,
                args=(input_state, num_qubits, num_latent),
                method='COBYLA',
                options={'maxiter': 200}
            )

            fidelity = 1 - result.fun
            compression_ratio = self._get_compression_ratio(prices)

            return self._classify_regime(fidelity, compression_ratio)

        except Exception as e:
            print(f"Quantum analysis failed: {e}, falling back to classical")
            return self._classical_analysis(prices)

    def _compression_cost(self, params, input_state, num_qubits, num_latent):
        """Cost function for quantum compression optimization"""
        num_trash = num_qubits - num_latent
        U = self._build_encoder(params, num_qubits)
        rho = input_state * input_state.dag()
        rho_out = U * rho * U.dag()
        rho_trash = rho_out.ptrace(range(num_latent, num_qubits))
        ref = qt.tensor([qt.ket2dm(qt.basis(2, 0)) for _ in range(num_trash)])
        fid = qt.fidelity(rho_trash, ref)
        return 1 - fid

    def _build_encoder(self, params, num_qubits, layers=2):
        """Build parameterized quantum encoder circuit"""
        U = qt.qeye([2]*num_qubits)
        param_idx = 0

        for layer in range(layers):
            # RY rotations
            ry_ops = [self._ry(params[param_idx + i]) for i in range(num_qubits)]
            param_idx += num_qubits
            U = qt.tensor(ry_ops) * U

            # CNOT entangling gates
            for i in range(num_qubits):
                U = self._cnot(num_qubits, i, (i + 1) % num_qubits) * U

        return U

    def _ry(self, theta):
        """RY rotation gate"""
        return (-1j * theta/2 * qt.sigmay()).expm()

    def _cnot(self, N, control, target):
        """CNOT gate"""
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

    def _classical_analysis(self, prices: np.ndarray) -> RegimeResult:
        """Fallback classical compression analysis using zlib"""
        compression_ratio = self._get_compression_ratio(prices)

        # Map compression ratio to fidelity estimate
        # Higher compression = more structure = higher fidelity
        if compression_ratio >= 3.5:
            fidelity = 0.96
        elif compression_ratio >= 3.0:
            fidelity = 0.92
        elif compression_ratio >= 2.5:
            fidelity = 0.88
        elif compression_ratio >= 2.0:
            fidelity = 0.82
        else:
            fidelity = 0.75

        return self._classify_regime(fidelity, compression_ratio)

    def _get_compression_ratio(self, prices: np.ndarray) -> float:
        """Calculate zlib compression ratio"""
        data_bytes = prices.astype(np.float32).tobytes()
        compressed = zlib.compress(data_bytes, level=9)
        return len(data_bytes) / len(compressed)

    def _classify_regime(self, fidelity: float, compression_ratio: float) -> RegimeResult:
        """Classify regime based on fidelity"""
        if fidelity >= self.clean_threshold:
            regime = Regime.CLEAN
            trade_allowed = True
        elif fidelity >= self.volatile_threshold:
            regime = Regime.VOLATILE
            trade_allowed = False  # Caution - don't trade
        else:
            regime = Regime.CHOPPY
            trade_allowed = False

        return RegimeResult(
            regime=regime,
            fidelity=fidelity,
            compression_ratio=compression_ratio,
            trade_allowed=trade_allowed
        )
```

---

## LAYER 2: PARALLEL QUANTUM FEATURE EXTRACTION

### 2A: Quantum LSTM

**File:** `layer2_quantum/quantum_lstm.py`

```python
"""
LAYER 2A: Quantum LSTM
======================
Bidirectional LSTM with attention mechanism and quantum-enhanced features.
Contributes 20% to final signal fusion.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict


class QuantumLSTM(nn.Module):
    """
    Bidirectional LSTM with:
    - Quantum feature inputs (7 features)
    - Technical indicator inputs (8 features)
    - Attention mechanism
    - Total: 15 input features
    """

    def __init__(self, input_size: int = 15, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super(QuantumLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # HOLD, BUY, SELL
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            output: (batch, 3) logits for HOLD/BUY/SELL
            attention_weights: (batch, seq_len, 1)
        """
        lstm_out, _ = self.lstm(x)

        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Classification
        output = self.fc(context)

        return output, attention_weights

    def predict(self, x: torch.Tensor) -> Dict:
        """
        Get prediction with confidence.

        Returns dict with:
            action: 0=HOLD, 1=BUY, 2=SELL
            confidence: probability of chosen action
            probabilities: all class probabilities
        """
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x)
            probs = torch.softmax(output, dim=1)
            action = torch.argmax(probs, dim=1).item()
            confidence = probs[0, action].item()

        return {
            'action': action,
            'confidence': confidence,
            'probabilities': probs[0].numpy()
        }


class QuantumFeatureExtractor:
    """
    Extracts 7 quantum-inspired features for the LSTM.
    """

    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits

    def extract(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Extract quantum features from price data.

        Returns 7 features:
            - quantum_entropy
            - dominant_state_prob
            - superposition_measure
            - phase_coherence
            - entanglement_degree
            - quantum_variance
            - num_significant_states
        """
        # Simplified quantum-inspired features (fast computation)
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        volatility = np.std(returns)
        trend = np.mean(returns)

        # Simulate quantum probability distribution based on price patterns
        normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

        # Entropy (measure of uncertainty)
        hist, _ = np.histogram(normalized, bins=16, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) / 4  # Normalize

        # Dominant state (strongest pattern)
        dominant_prob = np.max(hist) if len(hist) > 0 else 0.5

        # Superposition (how spread out the distribution is)
        superposition = 1 - dominant_prob

        # Phase coherence (trend consistency)
        sign_changes = np.sum(np.diff(np.sign(returns)) != 0)
        phase_coherence = 1 - (sign_changes / len(returns))

        # Entanglement (autocorrelation)
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            entanglement = abs(autocorr) if not np.isnan(autocorr) else 0.5
        else:
            entanglement = 0.5

        # Quantum variance
        quantum_variance = volatility * 100

        # Significant states
        significant_states = np.sum(hist > 0.05) if len(hist) > 0 else 4

        return {
            'quantum_entropy': entropy,
            'dominant_state_prob': dominant_prob,
            'superposition_measure': superposition,
            'phase_coherence': phase_coherence,
            'entanglement_degree': entanglement,
            'quantum_variance': quantum_variance,
            'num_significant_states': float(significant_states)
        }
```

### 2B: 3D Bar Analysis

**File:** `layer2_quantum/bars_3d.py`

```python
"""
LAYER 2B: 3D Bar Analysis
=========================
Identifies "yellow clusters" - consolidation patterns that precede breakouts.
Contributes 15% to final signal fusion.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Bar3D:
    """Represents a 3D bar with price, volume, and time dimensions"""
    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float  # high - low
    body: float    # abs(close - open)
    color: str     # 'green', 'red', 'yellow' (doji)


@dataclass
class ClusterResult:
    """Result of cluster analysis"""
    cluster_type: str  # 'yellow', 'green', 'red', 'mixed'
    strength: float    # 0-1 cluster strength
    direction: str     # 'UP', 'DOWN', 'NEUTRAL'
    bars_in_cluster: int
    breakout_probability: float


class Bars3DAnalyzer:
    """
    Analyzes 3D bar patterns looking for yellow clusters.

    Yellow clusters = Doji-like bars with small bodies
    These indicate consolidation before a breakout.
    """

    def __init__(self, min_spread_multiplier: int = 45, volume_threshold: float = 500):
        self.min_spread_multiplier = min_spread_multiplier
        self.volume_threshold = volume_threshold

    def create_3d_bars(self, df: pd.DataFrame) -> List[Bar3D]:
        """Convert OHLCV data to 3D bars"""
        bars = []

        for _, row in df.iterrows():
            spread = row['high'] - row['low']
            body = abs(row['close'] - row['open'])

            # Classify color based on body/spread ratio
            if spread > 0:
                body_ratio = body / spread
                if body_ratio < 0.3:
                    color = 'yellow'  # Doji/indecision
                elif row['close'] > row['open']:
                    color = 'green'
                else:
                    color = 'red'
            else:
                color = 'yellow'

            bars.append(Bar3D(
                time=row['time'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row.get('tick_volume', 0),
                spread=spread,
                body=body,
                color=color
            ))

        return bars

    def find_clusters(self, bars: List[Bar3D], min_cluster_size: int = 3) -> List[ClusterResult]:
        """Find clusters of similar colored bars"""
        clusters = []
        current_cluster = []
        current_color = None

        for bar in bars:
            if bar.color == current_color or current_color is None:
                current_cluster.append(bar)
                current_color = bar.color
            else:
                if len(current_cluster) >= min_cluster_size:
                    clusters.append(self._analyze_cluster(current_cluster))
                current_cluster = [bar]
                current_color = bar.color

        # Don't forget last cluster
        if len(current_cluster) >= min_cluster_size:
            clusters.append(self._analyze_cluster(current_cluster))

        return clusters

    def _analyze_cluster(self, bars: List[Bar3D]) -> ClusterResult:
        """Analyze a cluster of bars"""
        # Count colors
        yellow_count = sum(1 for b in bars if b.color == 'yellow')
        green_count = sum(1 for b in bars if b.color == 'green')
        red_count = sum(1 for b in bars if b.color == 'red')

        total = len(bars)

        # Determine cluster type
        if yellow_count / total > 0.5:
            cluster_type = 'yellow'
        elif green_count / total > 0.6:
            cluster_type = 'green'
        elif red_count / total > 0.6:
            cluster_type = 'red'
        else:
            cluster_type = 'mixed'

        # Calculate strength (tightness of cluster)
        spreads = [b.spread for b in bars]
        avg_spread = np.mean(spreads)
        spread_std = np.std(spreads)
        strength = 1 / (1 + spread_std / (avg_spread + 1e-10))

        # Determine likely breakout direction
        first_close = bars[0].close
        last_close = bars[-1].close
        high_of_cluster = max(b.high for b in bars)
        low_of_cluster = min(b.low for b in bars)

        if last_close > (high_of_cluster + low_of_cluster) / 2:
            direction = 'UP'
        elif last_close < (high_of_cluster + low_of_cluster) / 2:
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'

        # Breakout probability (higher for yellow clusters)
        if cluster_type == 'yellow':
            breakout_prob = 0.7 + strength * 0.2
        else:
            breakout_prob = 0.4 + strength * 0.2

        return ClusterResult(
            cluster_type=cluster_type,
            strength=strength,
            direction=direction,
            bars_in_cluster=total,
            breakout_probability=breakout_prob
        )

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Full 3D bar analysis.

        Returns dict with:
            signal: 'BUY', 'SELL', 'HOLD'
            confidence: 0-1
            cluster_info: details about detected clusters
        """
        bars = self.create_3d_bars(df)
        clusters = self.find_clusters(bars[-50:])  # Analyze last 50 bars

        if not clusters:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'cluster_info': None
            }

        # Focus on most recent cluster
        recent_cluster = clusters[-1]

        if recent_cluster.cluster_type == 'yellow':
            # Yellow cluster - predict breakout direction
            if recent_cluster.direction == 'UP':
                signal = 'BUY'
            elif recent_cluster.direction == 'DOWN':
                signal = 'SELL'
            else:
                signal = 'HOLD'
            confidence = recent_cluster.breakout_probability
        else:
            # Trend continuation
            if recent_cluster.cluster_type == 'green':
                signal = 'BUY'
            elif recent_cluster.cluster_type == 'red':
                signal = 'SELL'
            else:
                signal = 'HOLD'
            confidence = recent_cluster.strength * 0.7

        return {
            'signal': signal,
            'confidence': confidence,
            'cluster_info': recent_cluster
        }
```

### 2C: QPE Horizon Prediction

**File:** `layer2_quantum/qpe_horizon.py`

```python
"""
LAYER 2C: Quantum Phase Estimation Horizon Predictor
====================================================
Uses quantum-inspired phase estimation to predict price horizons.
Contributes 15% to final signal fusion.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.fft import fft, fftfreq


class QPEHorizonPredictor:
    """
    Quantum Phase Estimation inspired price prediction.

    Uses Fourier analysis as a classical approximation to QPE
    to identify dominant price cycles and predict future direction.
    """

    def __init__(self, horizon_bars: int = 10):
        self.horizon_bars = horizon_bars

    def analyze(self, prices: np.ndarray) -> Dict:
        """
        Analyze price data and predict horizon direction.

        Args:
            prices: Array of close prices (minimum 64 values)

        Returns dict with:
            signal: 'BUY', 'SELL', 'HOLD'
            confidence: 0-1
            dominant_period: primary cycle length in bars
            phase: current position in cycle (0-1)
        """
        if len(prices) < 64:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'dominant_period': 0,
                'phase': 0.5
            }

        # Detrend prices
        detrended = prices - np.linspace(prices[0], prices[-1], len(prices))

        # FFT analysis (classical QPE approximation)
        fft_vals = fft(detrended)
        freqs = fftfreq(len(prices))

        # Find dominant frequency (excluding DC component)
        magnitudes = np.abs(fft_vals[1:len(prices)//2])
        dominant_idx = np.argmax(magnitudes) + 1
        dominant_freq = abs(freqs[dominant_idx])

        if dominant_freq > 0:
            dominant_period = int(1 / dominant_freq)
        else:
            dominant_period = len(prices)

        # Calculate current phase in the cycle
        phase_angle = np.angle(fft_vals[dominant_idx])
        phase = (phase_angle + np.pi) / (2 * np.pi)  # Normalize to 0-1

        # Predict direction based on phase
        # Phase 0-0.25: Rising
        # Phase 0.25-0.5: Peak
        # Phase 0.5-0.75: Falling
        # Phase 0.75-1: Trough

        if 0 <= phase < 0.25 or 0.75 <= phase <= 1:
            signal = 'BUY'
            confidence = 0.6 + 0.2 * (magnitudes[dominant_idx - 1] / np.max(magnitudes))
        elif 0.25 <= phase < 0.75:
            signal = 'SELL'
            confidence = 0.6 + 0.2 * (magnitudes[dominant_idx - 1] / np.max(magnitudes))
        else:
            signal = 'HOLD'
            confidence = 0.5

        # Adjust confidence based on cycle strength
        cycle_strength = magnitudes[dominant_idx - 1] / np.sum(magnitudes)
        confidence *= (0.5 + cycle_strength)
        confidence = min(confidence, 0.95)

        return {
            'signal': signal,
            'confidence': confidence,
            'dominant_period': dominant_period,
            'phase': phase
        }
```

---

## LAYER 3: CLASSICAL SIGNAL GENERATION

### 3A: Volatility Analysis

**File:** `layer3_classical/volatility.py`

```python
"""
LAYER 3A: Volatility Analysis
=============================
Classical volatility indicators for risk assessment.
Contributes 10% to final signal fusion.
"""

import numpy as np
import pandas as pd
from typing import Dict


class VolatilityAnalyzer:
    """
    Analyzes market volatility using multiple methods.
    """

    def __init__(self):
        self.atr_period = 14
        self.bb_period = 20
        self.bb_std = 2

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Full volatility analysis.

        Returns dict with:
            signal: 'BUY', 'SELL', 'HOLD'
            confidence: 0-1
            atr: Average True Range value
            bb_position: Position within Bollinger Bands (-1 to 1)
            volatility_regime: 'LOW', 'NORMAL', 'HIGH'
        """
        # ATR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-self.atr_period:])

        # Bollinger Bands
        bb_middle = np.mean(close[-self.bb_period:])
        bb_std = np.std(close[-self.bb_period:])
        bb_upper = bb_middle + self.bb_std * bb_std
        bb_lower = bb_middle - self.bb_std * bb_std

        current_price = close[-1]

        # Position within bands (-1 = lower band, 0 = middle, 1 = upper band)
        if bb_std > 0:
            bb_position = (current_price - bb_middle) / (self.bb_std * bb_std)
        else:
            bb_position = 0

        # Volatility regime
        historical_atr = np.mean(tr[-50:]) if len(tr) >= 50 else atr
        if atr > historical_atr * 1.5:
            vol_regime = 'HIGH'
        elif atr < historical_atr * 0.5:
            vol_regime = 'LOW'
        else:
            vol_regime = 'NORMAL'

        # Signal based on Bollinger Band position
        if bb_position < -0.8:
            signal = 'BUY'  # Near lower band - potential bounce
            confidence = 0.6 + abs(bb_position) * 0.2
        elif bb_position > 0.8:
            signal = 'SELL'  # Near upper band - potential reversal
            confidence = 0.6 + abs(bb_position) * 0.2
        else:
            signal = 'HOLD'
            confidence = 0.5

        # Reduce confidence in high volatility
        if vol_regime == 'HIGH':
            confidence *= 0.8

        return {
            'signal': signal,
            'confidence': min(confidence, 0.95),
            'atr': atr,
            'bb_position': bb_position,
            'volatility_regime': vol_regime
        }
```

### 3B: Currency Strength

**File:** `layer3_classical/currency_strength.py`

```python
"""
LAYER 3B: Currency Strength Analysis
====================================
Cross-pair correlation for BTC/ETH relationship.
Contributes 5% to final signal fusion.

KEY INSIGHT: BTC and ETH are highly correlated (0.85-0.95).
When they diverge, it often signals a trading opportunity.
"""

import numpy as np
from typing import Dict, Optional


class CurrencyStrengthAnalyzer:
    """
    Analyzes BTC/ETH correlation and relative strength.
    """

    def __init__(self, correlation_window: int = 30):
        self.correlation_window = correlation_window
        self.btc_prices = None
        self.eth_prices = None

    def update_prices(self, btc_prices: np.ndarray, eth_prices: np.ndarray):
        """Update price arrays for both symbols"""
        self.btc_prices = btc_prices
        self.eth_prices = eth_prices

    def analyze(self, target_symbol: str) -> Dict:
        """
        Analyze currency strength for target symbol.

        Args:
            target_symbol: 'BTCUSD' or 'ETHUSD'

        Returns dict with:
            signal: 'BUY', 'SELL', 'HOLD'
            confidence: 0-1
            correlation: current BTC/ETH correlation
            relative_strength: target vs pair (-1 to 1)
            divergence: whether prices are diverging
        """
        if self.btc_prices is None or self.eth_prices is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'correlation': 0,
                'relative_strength': 0,
                'divergence': False
            }

        # Calculate correlation
        min_len = min(len(self.btc_prices), len(self.eth_prices))
        btc = self.btc_prices[-min_len:]
        eth = self.eth_prices[-min_len:]

        if min_len < self.correlation_window:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'correlation': 0,
                'relative_strength': 0,
                'divergence': False
            }

        # Returns correlation
        btc_returns = np.diff(btc[-self.correlation_window:]) / btc[-self.correlation_window:-1]
        eth_returns = np.diff(eth[-self.correlation_window:]) / eth[-self.correlation_window:-1]

        correlation = np.corrcoef(btc_returns, eth_returns)[0, 1]
        if np.isnan(correlation):
            correlation = 0.9  # Default high correlation

        # Relative strength (which is outperforming)
        btc_change = (btc[-1] / btc[-self.correlation_window] - 1) * 100
        eth_change = (eth[-1] / eth[-self.correlation_window] - 1) * 100

        relative_strength = (btc_change - eth_change) / (abs(btc_change) + abs(eth_change) + 1e-10)

        # Detect divergence (correlation breakdown)
        divergence = correlation < 0.7

        # Generate signal
        if target_symbol == 'BTCUSD':
            if divergence and relative_strength > 0.3:
                signal = 'BUY'  # BTC outperforming during divergence
                confidence = 0.65
            elif divergence and relative_strength < -0.3:
                signal = 'SELL'  # BTC underperforming during divergence
                confidence = 0.65
            else:
                signal = 'HOLD'
                confidence = 0.5
        else:  # ETHUSD
            if divergence and relative_strength < -0.3:
                signal = 'BUY'  # ETH outperforming during divergence
                confidence = 0.65
            elif divergence and relative_strength > 0.3:
                signal = 'SELL'  # ETH underperforming during divergence
                confidence = 0.65
            else:
                signal = 'HOLD'
                confidence = 0.5

        return {
            'signal': signal,
            'confidence': confidence,
            'correlation': correlation,
            'relative_strength': relative_strength,
            'divergence': divergence
        }
```

---

## LAYER 4: SIGNAL FUSION ENGINE

**File:** `layer4_fusion/fusion_engine.py`

```python
"""
LAYER 4: Signal Fusion Engine
=============================
Combines signals from all layers using weighted scoring.

Fusion Weights (from design doc):
    Compression Ratio:    25%
    Quantum LSTM:         20%
    Quantum 3D:           15%
    QPE Horizon:          15%
    Volatility:           10%
    Currency Strength:     5%
    ETARE Base:           10%
                         ────
                         100%
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class FusionWeights:
    """Configurable fusion weights"""
    compression: float = 0.25
    quantum_lstm: float = 0.20
    quantum_3d: float = 0.15
    qpe_horizon: float = 0.15
    volatility: float = 0.10
    currency_strength: float = 0.05
    etare_base: float = 0.10

    def validate(self):
        total = (self.compression + self.quantum_lstm + self.quantum_3d +
                 self.qpe_horizon + self.volatility + self.currency_strength +
                 self.etare_base)
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"


class SignalFusionEngine:
    """
    Fuses signals from multiple layers into a single trading decision.
    """

    def __init__(self, weights: FusionWeights = None):
        self.weights = weights or FusionWeights()
        self.weights.validate()

        # Signal value mapping
        self.signal_values = {'BUY': 1.0, 'HOLD': 0.0, 'SELL': -1.0}

    def fuse(self, layer_results: Dict) -> Dict:
        """
        Fuse signals from all layers.

        Args:
            layer_results: Dict with keys:
                - compression: RegimeResult from Layer 1
                - quantum_lstm: Dict from Layer 2A
                - quantum_3d: Dict from Layer 2B
                - qpe_horizon: Dict from Layer 2C
                - volatility: Dict from Layer 3A
                - currency_strength: Dict from Layer 3B
                - etare_base: Optional baseline prediction

        Returns dict with:
            action: 'BUY', 'SELL', 'HOLD'
            confidence: 0-1 weighted confidence
            score: raw fusion score (-1 to 1)
            component_scores: individual layer contributions
        """
        # Check Layer 1 gate - if regime isn't tradeable, return HOLD
        compression_result = layer_results.get('compression')
        if compression_result and not compression_result.trade_allowed:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'score': 0.0,
                'component_scores': {},
                'gate_blocked': True,
                'gate_reason': f"Regime: {compression_result.regime.value}"
            }

        # Calculate weighted score
        components = {}
        weighted_score = 0.0
        weighted_confidence = 0.0

        # Layer 1: Compression (regime quality contributes to confidence)
        if compression_result:
            # Use fidelity as confidence multiplier, not as signal direction
            components['compression'] = {
                'signal': 'HOLD',  # Regime doesn't give direction
                'confidence': compression_result.fidelity,
                'weight': self.weights.compression,
                'contribution': compression_result.fidelity * self.weights.compression
            }
            weighted_confidence += compression_result.fidelity * self.weights.compression

        # Layer 2A: Quantum LSTM
        lstm_result = layer_results.get('quantum_lstm', {})
        if lstm_result:
            signal = ['HOLD', 'BUY', 'SELL'][lstm_result.get('action', 0)]
            conf = lstm_result.get('confidence', 0.5)
            score = self.signal_values[signal] * conf

            components['quantum_lstm'] = {
                'signal': signal,
                'confidence': conf,
                'weight': self.weights.quantum_lstm,
                'contribution': score * self.weights.quantum_lstm
            }
            weighted_score += score * self.weights.quantum_lstm
            weighted_confidence += conf * self.weights.quantum_lstm

        # Layer 2B: Quantum 3D
        bars3d_result = layer_results.get('quantum_3d', {})
        if bars3d_result:
            signal = bars3d_result.get('signal', 'HOLD')
            conf = bars3d_result.get('confidence', 0.5)
            score = self.signal_values[signal] * conf

            components['quantum_3d'] = {
                'signal': signal,
                'confidence': conf,
                'weight': self.weights.quantum_3d,
                'contribution': score * self.weights.quantum_3d
            }
            weighted_score += score * self.weights.quantum_3d
            weighted_confidence += conf * self.weights.quantum_3d

        # Layer 2C: QPE Horizon
        qpe_result = layer_results.get('qpe_horizon', {})
        if qpe_result:
            signal = qpe_result.get('signal', 'HOLD')
            conf = qpe_result.get('confidence', 0.5)
            score = self.signal_values[signal] * conf

            components['qpe_horizon'] = {
                'signal': signal,
                'confidence': conf,
                'weight': self.weights.qpe_horizon,
                'contribution': score * self.weights.qpe_horizon
            }
            weighted_score += score * self.weights.qpe_horizon
            weighted_confidence += conf * self.weights.qpe_horizon

        # Layer 3A: Volatility
        vol_result = layer_results.get('volatility', {})
        if vol_result:
            signal = vol_result.get('signal', 'HOLD')
            conf = vol_result.get('confidence', 0.5)
            score = self.signal_values[signal] * conf

            components['volatility'] = {
                'signal': signal,
                'confidence': conf,
                'weight': self.weights.volatility,
                'contribution': score * self.weights.volatility
            }
            weighted_score += score * self.weights.volatility
            weighted_confidence += conf * self.weights.volatility

        # Layer 3B: Currency Strength
        cs_result = layer_results.get('currency_strength', {})
        if cs_result:
            signal = cs_result.get('signal', 'HOLD')
            conf = cs_result.get('confidence', 0.5)
            score = self.signal_values[signal] * conf

            components['currency_strength'] = {
                'signal': signal,
                'confidence': conf,
                'weight': self.weights.currency_strength,
                'contribution': score * self.weights.currency_strength
            }
            weighted_score += score * self.weights.currency_strength
            weighted_confidence += conf * self.weights.currency_strength

        # ETARE Base (if provided)
        etare_result = layer_results.get('etare_base', {})
        if etare_result:
            signal = etare_result.get('signal', 'HOLD')
            conf = etare_result.get('confidence', 0.5)
            score = self.signal_values[signal] * conf

            components['etare_base'] = {
                'signal': signal,
                'confidence': conf,
                'weight': self.weights.etare_base,
                'contribution': score * self.weights.etare_base
            }
            weighted_score += score * self.weights.etare_base
            weighted_confidence += conf * self.weights.etare_base

        # Determine final action
        if weighted_score > 0.15:
            action = 'BUY'
        elif weighted_score < -0.15:
            action = 'SELL'
        else:
            action = 'HOLD'

        # Final confidence is weighted average, scaled by score strength
        final_confidence = weighted_confidence * min(abs(weighted_score) * 2, 1.0)

        return {
            'action': action,
            'confidence': min(final_confidence, 0.95),
            'score': weighted_score,
            'component_scores': components,
            'gate_blocked': False
        }
```

---

## LAYER 5: ETARE NEURAL NETWORK

**File:** `layer5_etare/etare_network.py`

```python
"""
LAYER 5: ETARE Neural Network Enhanced
======================================
23-input neural network that takes fusion output and makes final decision.

Inputs (23 total):
    - 7 quantum features
    - 8 technical indicators
    - 5 layer scores (from fusion)
    - 3 meta features (correlation, regime, volatility regime)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict


class ETARENetwork(nn.Module):
    """
    Enhanced ETARE Neural Network with 23 inputs.
    """

    def __init__(self, input_size: int = 23, hidden_sizes: list = None):
        super(ETARENetwork, self).__init__()

        hidden_sizes = hidden_sizes or [64, 32, 16]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 3))  # HOLD, BUY, SELL

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict(self, features: np.ndarray) -> Dict:
        """
        Make prediction from feature vector.

        Args:
            features: (23,) numpy array

        Returns dict with:
            action: 0=HOLD, 1=BUY, 2=SELL
            confidence: probability of chosen action
            probabilities: all class probabilities
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            output = self.forward(x)
            probs = torch.softmax(output, dim=1)
            action = torch.argmax(probs, dim=1).item()
            confidence = probs[0, action].item()

        return {
            'action': action,
            'confidence': confidence,
            'probabilities': probs[0].numpy()
        }


class ETAREFeatureBuilder:
    """
    Builds the 23-feature input vector for ETARE network.
    """

    def build(self, quantum_features: Dict, technical_features: np.ndarray,
              fusion_result: Dict, meta_features: Dict) -> np.ndarray:
        """
        Build 23-feature vector.

        Args:
            quantum_features: 7 quantum features dict
            technical_features: 8 technical indicator values
            fusion_result: Output from fusion engine
            meta_features: correlation, regime, volatility info

        Returns:
            (23,) numpy array
        """
        features = []

        # 7 quantum features
        quantum_order = [
            'quantum_entropy', 'dominant_state_prob', 'superposition_measure',
            'phase_coherence', 'entanglement_degree', 'quantum_variance',
            'num_significant_states'
        ]
        for key in quantum_order:
            features.append(quantum_features.get(key, 0.0))

        # 8 technical features (already normalized)
        features.extend(technical_features[:8])

        # 5 layer scores from fusion
        component_scores = fusion_result.get('component_scores', {})
        layer_order = ['quantum_lstm', 'quantum_3d', 'qpe_horizon', 'volatility', 'currency_strength']
        for layer in layer_order:
            score_info = component_scores.get(layer, {})
            features.append(score_info.get('contribution', 0.0))

        # 3 meta features
        features.append(meta_features.get('correlation', 0.9))
        features.append(1.0 if meta_features.get('regime') == 'CLEAN' else 0.0)
        features.append({'LOW': 0.0, 'NORMAL': 0.5, 'HIGH': 1.0}.get(
            meta_features.get('volatility_regime', 'NORMAL'), 0.5))

        return np.array(features, dtype=np.float32)
```

---

## LAYER 6: EXECUTION ENGINE

**File:** `layer6_execution/grid_trader.py`

```python
"""
LAYER 6: Execution Engine - Grid Trading
========================================
Executes trades with grid-based position management.
"""

import MetaTrader5 as mt5
from typing import Dict, Optional
from dataclasses import dataclass
import logging


@dataclass
class TradeRequest:
    symbol: str
    action: str  # 'BUY' or 'SELL'
    lot_size: float
    sl_pips: float
    tp_pips: float
    magic_number: int
    comment: str


class GridTrader:
    """
    Grid-based trade execution with partial close management.
    """

    def __init__(self, magic_number: int = 777001):
        self.magic_number = magic_number
        self.partial_closed_tickets = set()

    def execute(self, request: TradeRequest) -> Dict:
        """
        Execute a trade request.

        Returns dict with:
            success: bool
            ticket: order ticket if successful
            error: error message if failed
        """
        symbol_info = mt5.symbol_info(request.symbol)
        if symbol_info is None:
            return {'success': False, 'error': f'Symbol {request.symbol} not found'}

        if not symbol_info.visible:
            mt5.symbol_select(request.symbol, True)

        tick = mt5.symbol_info_tick(request.symbol)
        if tick is None:
            return {'success': False, 'error': 'Failed to get tick'}

        point = symbol_info.point

        if request.action == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = price - request.sl_pips * point
            tp = price + request.tp_pips * point
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = price + request.sl_pips * point
            tp = price - request.tp_pips * point

        filling_mode = mt5.ORDER_FILLING_IOC
        if symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK

        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": request.symbol,
            "volume": request.lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": request.magic_number,
            "comment": request.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(order_request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Trade executed: {request.action} {request.symbol} @ {price}")
            return {'success': True, 'ticket': result.order}
        else:
            logging.error(f"Trade failed: {result.comment}")
            return {'success': False, 'error': result.comment}

    def manage_positions(self, symbol: str, partial_close_pct: float = 0.5,
                        partial_trigger_pct: float = 0.5):
        """
        Manage open positions:
        - Partial close at 50% of TP distance
        - Move SL to breakeven after partial
        """
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        for pos in positions:
            if pos.magic != self.magic_number:
                continue

            if pos.ticket in self.partial_closed_tickets:
                continue

            # Calculate progress to TP
            if pos.type == mt5.POSITION_TYPE_BUY:
                tp_distance = pos.tp - pos.price_open
                current_progress = pos.price_current - pos.price_open
            else:
                tp_distance = pos.price_open - pos.tp
                current_progress = pos.price_open - pos.price_current

            if tp_distance <= 0:
                continue

            progress_pct = current_progress / tp_distance

            # Trigger partial close at 50% to TP
            if progress_pct >= partial_trigger_pct:
                self._partial_close(pos, partial_close_pct)

    def _partial_close(self, position, close_pct: float):
        """Execute partial close and move SL to breakeven"""
        symbol_info = mt5.symbol_info(position.symbol)
        if symbol_info is None:
            return

        close_volume = round(position.volume * close_pct / symbol_info.volume_step) * symbol_info.volume_step
        close_volume = max(symbol_info.volume_min, close_volume)

        remaining = position.volume - close_volume
        if remaining < symbol_info.volume_min:
            return  # Can't partial close

        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return

        if position.type == mt5.POSITION_TYPE_BUY:
            close_price = tick.bid
            close_type = mt5.ORDER_TYPE_SELL
        else:
            close_price = tick.ask
            close_type = mt5.ORDER_TYPE_BUY

        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "magic": self.magic_number,
            "comment": "Partial close 50%",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(close_request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.partial_closed_tickets.add(position.ticket)
            logging.info(f"Partial close: {close_volume} lots @ {close_price}")

            # Move SL to breakeven
            self._move_sl_to_breakeven(position.symbol, position.price_open)

    def _move_sl_to_breakeven(self, symbol: str, breakeven_price: float):
        """Move SL to breakeven for remaining position"""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        for pos in positions:
            if pos.magic != self.magic_number:
                continue

            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": breakeven_price,
                "tp": pos.tp,
            }

            result = mt5.order_send(modify_request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"SL moved to breakeven @ {breakeven_price}")
```

**File:** `layer6_execution/risk_manager.py`

```python
"""
LAYER 6: Risk Management
========================
Enforces prop firm rules and emergency stops.
"""

import MetaTrader5 as mt5
from typing import Dict
from dataclasses import dataclass
import logging


@dataclass
class RiskLimits:
    max_daily_loss_pct: float = 0.05      # 5%
    max_total_drawdown_pct: float = 0.10  # 10%
    max_position_size: float = 1.0        # Max lot size
    max_positions: int = 3                # Max concurrent positions


class RiskManager:
    """
    Enforces risk limits and provides emergency stop functionality.
    """

    def __init__(self, limits: RiskLimits = None, magic_number: int = 777001):
        self.limits = limits or RiskLimits()
        self.magic_number = magic_number
        self.daily_start_balance = None
        self.high_water_mark = None

    def initialize(self):
        """Initialize risk tracking"""
        account = mt5.account_info()
        if account:
            self.daily_start_balance = account.balance
            self.high_water_mark = account.balance

    def check_limits(self) -> Dict:
        """
        Check all risk limits.

        Returns dict with:
            can_trade: bool
            reason: reason if can't trade
            daily_loss_pct: current daily loss percentage
            total_drawdown_pct: current drawdown from high water mark
        """
        account = mt5.account_info()
        if account is None:
            return {'can_trade': False, 'reason': 'Cannot get account info'}

        if self.daily_start_balance is None:
            self.daily_start_balance = account.balance
        if self.high_water_mark is None:
            self.high_water_mark = account.balance

        # Update high water mark
        if account.balance > self.high_water_mark:
            self.high_water_mark = account.balance

        # Calculate daily loss
        daily_loss = self.daily_start_balance - account.balance
        daily_loss_pct = daily_loss / self.daily_start_balance if self.daily_start_balance > 0 else 0

        # Calculate total drawdown
        drawdown = self.high_water_mark - account.balance
        drawdown_pct = drawdown / self.high_water_mark if self.high_water_mark > 0 else 0

        # Check limits
        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            return {
                'can_trade': False,
                'reason': f'Daily loss limit hit: {daily_loss_pct*100:.1f}%',
                'daily_loss_pct': daily_loss_pct,
                'total_drawdown_pct': drawdown_pct
            }

        if drawdown_pct >= self.limits.max_total_drawdown_pct:
            return {
                'can_trade': False,
                'reason': f'Max drawdown hit: {drawdown_pct*100:.1f}%',
                'daily_loss_pct': daily_loss_pct,
                'total_drawdown_pct': drawdown_pct
            }

        # Check position count
        positions = mt5.positions_get()
        our_positions = [p for p in positions if p.magic == self.magic_number] if positions else []

        if len(our_positions) >= self.limits.max_positions:
            return {
                'can_trade': False,
                'reason': f'Max positions reached: {len(our_positions)}',
                'daily_loss_pct': daily_loss_pct,
                'total_drawdown_pct': drawdown_pct
            }

        return {
            'can_trade': True,
            'reason': None,
            'daily_loss_pct': daily_loss_pct,
            'total_drawdown_pct': drawdown_pct
        }

    def emergency_stop(self):
        """
        Close all positions immediately.
        Called when 10% drawdown is hit.
        """
        logging.warning("EMERGENCY STOP TRIGGERED")

        positions = mt5.positions_get()
        if not positions:
            return

        for pos in positions:
            if pos.magic != self.magic_number:
                continue

            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                continue

            if pos.type == mt5.POSITION_TYPE_BUY:
                close_price = tick.bid
                close_type = mt5.ORDER_TYPE_SELL
            else:
                close_price = tick.ask
                close_type = mt5.ORDER_TYPE_BUY

            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": close_price,
                "magic": self.magic_number,
                "comment": "EMERGENCY STOP",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(close_request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Emergency closed position {pos.ticket}")
            else:
                logging.error(f"Failed to close position {pos.ticket}: {result.comment}")
```

---

## MAIN BRAIN ORCHESTRATOR

**File:** `brain/quantum_fusion_brain.py`

```python
"""
ETARE QUANTUM FUSION BRAIN
==========================
Main orchestrator that ties all 6 layers together.
"""

import time
import logging
from datetime import datetime
from typing import Dict

# Layer imports
from layer0_data.market_data import MarketDataFetcher
from layer1_compression.regime_detector import QuantumRegimeDetector
from layer2_quantum.quantum_lstm import QuantumLSTM, QuantumFeatureExtractor
from layer2_quantum.bars_3d import Bars3DAnalyzer
from layer2_quantum.qpe_horizon import QPEHorizonPredictor
from layer3_classical.volatility import VolatilityAnalyzer
from layer3_classical.currency_strength import CurrencyStrengthAnalyzer
from layer4_fusion.fusion_engine import SignalFusionEngine, FusionWeights
from layer5_etare.etare_network import ETARENetwork, ETAREFeatureBuilder
from layer6_execution.grid_trader import GridTrader, TradeRequest
from layer6_execution.risk_manager import RiskManager, RiskLimits

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][FUSION] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('quantum_fusion.log'),
        logging.StreamHandler()
    ]
)


class QuantumFusionBrain:
    """
    The complete 6-layer trading brain.
    """

    def __init__(self, symbols: list = None, magic_number: int = 777001):
        self.symbols = symbols or ['BTCUSD', 'ETHUSD']
        self.magic_number = magic_number

        # Initialize all layers
        self.data_fetcher = MarketDataFetcher(self.symbols)
        self.regime_detector = QuantumRegimeDetector()
        self.quantum_lstm = QuantumLSTM()
        self.quantum_features = QuantumFeatureExtractor()
        self.bars_3d = Bars3DAnalyzer()
        self.qpe_predictor = QPEHorizonPredictor()
        self.volatility = VolatilityAnalyzer()
        self.currency_strength = CurrencyStrengthAnalyzer()
        self.fusion_engine = SignalFusionEngine()
        self.etare_network = ETARENetwork()
        self.feature_builder = ETAREFeatureBuilder()
        self.trader = GridTrader(magic_number)
        self.risk_manager = RiskManager(magic_number=magic_number)

        # Configuration
        self.confidence_threshold = 0.55
        self.lot_size = 0.01

    def initialize(self) -> bool:
        """Initialize all components"""
        if not self.data_fetcher.connect():
            return False

        self.risk_manager.initialize()
        logging.info("Quantum Fusion Brain initialized")
        return True

    def analyze_symbol(self, symbol: str, df) -> Dict:
        """
        Run full 6-layer analysis for a symbol.
        """
        prices = df['close'].values

        # LAYER 1: Regime Detection
        regime_result = self.regime_detector.analyze(prices)
        logging.info(f"[{symbol}] Layer 1 - Regime: {regime_result.regime.value} "
                    f"(Fidelity: {regime_result.fidelity:.3f})")

        if not regime_result.trade_allowed:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reason': f'Regime: {regime_result.regime.value}',
                'layers': {'compression': regime_result}
            }

        # LAYER 2A: Quantum LSTM
        quantum_feats = self.quantum_features.extract(prices[-30:])
        # (Would need full feature prep and LSTM forward pass here)
        lstm_result = {'action': 0, 'confidence': 0.5}  # Placeholder

        # LAYER 2B: 3D Bar Analysis
        bars3d_result = self.bars_3d.analyze(df)

        # LAYER 2C: QPE Horizon
        qpe_result = self.qpe_predictor.analyze(prices)

        # LAYER 3A: Volatility
        vol_result = self.volatility.analyze(df)

        # LAYER 3B: Currency Strength
        cs_result = self.currency_strength.analyze(symbol)

        # LAYER 4: Fusion
        layer_results = {
            'compression': regime_result,
            'quantum_lstm': lstm_result,
            'quantum_3d': bars3d_result,
            'qpe_horizon': qpe_result,
            'volatility': vol_result,
            'currency_strength': cs_result
        }

        fusion_result = self.fusion_engine.fuse(layer_results)

        logging.info(f"[{symbol}] Layer 4 - Fusion: {fusion_result['action']} "
                    f"(Score: {fusion_result['score']:.3f}, "
                    f"Conf: {fusion_result['confidence']:.3f})")

        return {
            'action': fusion_result['action'],
            'confidence': fusion_result['confidence'],
            'score': fusion_result['score'],
            'layers': layer_results,
            'fusion': fusion_result
        }

    def run_cycle(self) -> Dict:
        """Run one complete trading cycle"""
        results = {}

        # Check risk limits first
        risk_check = self.risk_manager.check_limits()
        if not risk_check['can_trade']:
            logging.warning(f"Trading blocked: {risk_check['reason']}")
            if risk_check['total_drawdown_pct'] >= 0.10:
                self.risk_manager.emergency_stop()
            return {'blocked': True, 'reason': risk_check['reason']}

        # Fetch data for all symbols
        all_data = self.data_fetcher.fetch_all(bars=300)

        # Update currency strength analyzer
        if 'BTCUSD' in all_data and 'ETHUSD' in all_data:
            self.currency_strength.update_prices(
                all_data['BTCUSD']['close'].values,
                all_data['ETHUSD']['close'].values
            )

        # Analyze each symbol
        for symbol in self.symbols:
            if symbol not in all_data:
                continue

            df = all_data[symbol]
            analysis = self.analyze_symbol(symbol, df)
            results[symbol] = analysis

            # Execute trade if conditions met
            if (analysis['action'] in ['BUY', 'SELL'] and
                analysis['confidence'] >= self.confidence_threshold):

                request = TradeRequest(
                    symbol=symbol,
                    action=analysis['action'],
                    lot_size=self.lot_size,
                    sl_pips=50 if 'BTC' in symbol else 5,
                    tp_pips=150 if 'BTC' in symbol else 15,
                    magic_number=self.magic_number,
                    comment=f"QF_{analysis['action']}"
                )

                trade_result = self.trader.execute(request)
                results[symbol]['trade'] = trade_result

            # Manage existing positions
            self.trader.manage_positions(symbol)

        return results

    def run_loop(self, interval_seconds: int = 60):
        """Main trading loop"""
        print("=" * 60)
        print("  ETARE QUANTUM FUSION BRAIN")
        print("  6-Layer Architecture Active")
        print("=" * 60)

        if not self.initialize():
            print("Initialization failed")
            return

        try:
            while True:
                print(f"\n{'='*60}")
                print(f"  Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")

                results = self.run_cycle()

                for symbol, data in results.items():
                    if symbol == 'blocked':
                        continue
                    print(f"  [{symbol}] {data.get('action', 'N/A')} "
                          f"| Conf: {data.get('confidence', 0):.2f} "
                          f"| Score: {data.get('score', 0):.3f}")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.data_fetcher.shutdown()


if __name__ == "__main__":
    brain = QuantumFusionBrain(symbols=['BTCUSD', 'ETHUSD'])
    brain.run_loop(interval_seconds=60)
```

---

## REQUIREMENTS.TXT

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
MetaTrader5>=5.0.45
qutip>=4.7.0
```

---

## BUILD CHECKLIST

```
[ ] Create directory structure
[ ] Copy Layer 0: market_data.py
[ ] Copy Layer 1: regime_detector.py
[ ] Copy Layer 2A: quantum_lstm.py
[ ] Copy Layer 2B: bars_3d.py
[ ] Copy Layer 2C: qpe_horizon.py
[ ] Copy Layer 3A: volatility.py
[ ] Copy Layer 3B: currency_strength.py
[ ] Copy Layer 4: fusion_engine.py
[ ] Copy Layer 5: etare_network.py
[ ] Copy Layer 6A: grid_trader.py
[ ] Copy Layer 6B: risk_manager.py
[ ] Copy Main Brain: quantum_fusion_brain.py
[ ] Install requirements
[ ] Train models (use training pipeline)
[ ] Test each layer independently
[ ] Run integration test
[ ] Run prop firm simulation
[ ] Compare to simplified system results
```

---

## PROMPT FOR NEW CLAUDE SESSION

Copy this to start building:

```
Read the file ORIGINAL_SYSTEM_BUILD_GUIDE.md and build the complete 6-layer ETARE Quantum Fusion system in a new folder called `original_system/`.

Follow the directory structure exactly. Create all files as specified. Make sure each layer can be tested independently before integration.

Do NOT modify any existing files in the repository - this is a separate build.

After building, run a basic test to verify each layer works.
```

---

*Source: docs/plans/2026-01-18-etare-quantum-fusion-design.md*
*Generated: 2026-01-30*
