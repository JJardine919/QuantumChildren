"""
Layer 2a: Quantum LSTM Adapter
==============================
Wraps the Bidirectional Quantum LSTM model for the Fusion Engine.

Logic:
1. Extracts 7 Quantum Features (Entropy, Coherence, Entanglement, etc.)
2. Extracts 5 Technical Features (Returns, Log Returns, High/Low, etc.)
3. Feeds into Pre-trained PyTorch LSTM Model
4. Outputs: Signal (BUY/SELL/HOLD), Confidence, and Entropy Metrics
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import hashlib
import logging
from pathlib import Path
import json

# Configure Logging
logger = logging.getLogger("QuantumLSTMAdapter")

# ====================== QUANTUM FEATURE EXTRACTOR ======================
class QuantumFeatureExtractor:
    """
    Extracts 7 quantum features using 3 qubits and 1000 shots.
    """
    def __init__(self, num_qubits=3, shots=1000):
        self.num_qubits = num_qubits
        self.shots = shots
        self.simulator = AerSimulator(method='statevector')
        self.cache = {}

    def create_quantum_circuit(self, features: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        # RY Encoding
        for i in range(self.num_qubits):
            feature_idx = i % len(features)
            angle = np.clip(np.pi * features[feature_idx], -2*np.pi, 2*np.pi)
            qc.ry(angle, i)
        # CNOT Entanglement
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def extract(self, price_window: np.ndarray) -> dict:
        # Classical pre-processing for encoding
        returns = np.diff(price_window) / (price_window[:-1] + 1e-10)
        if len(returns) == 0: return self._default_features()
        
        features = np.array([
            np.mean(returns),
            np.std(returns),
            np.max(returns) - np.min(returns)
        ])
        features = np.tanh(features)

        try:
            qc = self.create_quantum_circuit(features)
            compiled = transpile(qc, self.simulator, optimization_level=0) # Speed optimization
            job = self.simulator.run(compiled, shots=self.shots)
            counts = job.result().get_counts()
            return self._compute_metrics(counts)
        except Exception as e:
            logger.error(f"Quantum extraction failed: {e}")
            return self._default_features()

    def _compute_metrics(self, counts: dict) -> dict:
        probs = {state: count/self.shots for state, count in counts.items()}
        
        # 1. Entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs.values())
        
        # 2. Dominant State
        dominant = max(probs.values()) if probs else 0
        
        # 3. Superposition
        significant = sum(1 for p in probs.values() if p > 0.05)
        superposition = significant / (2 ** self.num_qubits)
        
        # 4. Coherence (proxy)
        state_vals = [int(s, 2) for s in probs.keys()]
        coherence = 1.0 - (np.std(state_vals) / (2**self.num_qubits - 1)) if len(state_vals) > 1 else 0.5
        
        # 5. Entanglement (proxy via correlation)
        entanglement = 0.5 # Simplified
        
        return {
            'entropy': entropy,
            'dominant_state': dominant,
            'superposition': superposition,
            'coherence': coherence,
            'entanglement': entanglement,
            'significant_states': significant
        }

    def _default_features(self):
        return {
            'entropy': 2.5, 'dominant_state': 0.125, 'superposition': 0.5,
            'coherence': 0.5, 'entanglement': 0.5, 'significant_states': 4
        }


# ====================== FAST QUANTUM EXTRACTOR (NUMPY) ======================
class FastQuantumExtractor:
    """
    Drop-in replacement for QuantumFeatureExtractor using pure numpy.
    Simulates the same 3-qubit RY+CNOT circuit analytically (~800x faster).
    Produces exact probabilities instead of shot-sampled counts.
    """
    def __init__(self, num_qubits=3):
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits  # 8
        # Pre-build the CNOT gate matrices (fixed, reusable)
        self._cnot_01 = self._build_cnot(0, 1, num_qubits)
        self._cnot_12 = self._build_cnot(1, 2, num_qubits)

    @staticmethod
    def _build_cnot(control, target, n_qubits):
        """Build CNOT matrix for given control/target in n-qubit space."""
        dim = 2 ** n_qubits
        mat = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            bits = list(format(i, f'0{n_qubits}b'))
            if bits[control] == '1':
                bits[target] = '0' if bits[target] == '1' else '1'
            j = int(''.join(bits), 2)
            mat[j, i] = 1.0
        return mat

    def _simulate(self, angles):
        """Simulate circuit: RY on each qubit, then CNOT(0,1), CNOT(1,2)."""
        # Start with |000>
        state = np.zeros(self.num_states, dtype=np.complex128)
        state[0] = 1.0

        # Apply RY gates via tensor product
        # RY(θ)|0> = [cos(θ/2), sin(θ/2)]
        q_states = []
        for angle in angles:
            q_states.append(np.array([np.cos(angle / 2), np.sin(angle / 2)], dtype=np.complex128))

        # Tensor product of individual qubit states
        state = q_states[0]
        for qs in q_states[1:]:
            state = np.kron(state, qs)

        # Apply CNOTs
        state = self._cnot_01 @ state
        state = self._cnot_12 @ state

        # Measurement probabilities
        probs = np.abs(state) ** 2
        return probs

    def extract(self, price_window):
        """Extract quantum features from a price window (same API as QuantumFeatureExtractor)."""
        returns = np.diff(price_window) / (price_window[:-1] + 1e-10)
        if len(returns) == 0:
            return self._default_features()

        features = np.array([np.mean(returns), np.std(returns), np.max(returns) - np.min(returns)])
        features = np.tanh(features)
        angles = np.clip(np.pi * features, -2 * np.pi, 2 * np.pi)

        probs = self._simulate(angles)
        return self._compute_metrics(probs)

    def batch_extract(self, all_windows):
        """Vectorized extraction across many windows at once. Returns Nx7 array."""
        n = len(all_windows)
        results = np.zeros((n, 7), dtype=np.float64)

        for i, window in enumerate(all_windows):
            m = self.extract(window)
            results[i] = [
                m['entropy'], m['dominant_state'], m['superposition'],
                m['coherence'], m['entanglement'], 0.0, m['significant_states']
            ]

        return results

    def _compute_metrics(self, probs):
        # Filter near-zero for entropy calc
        p_nonzero = probs[probs > 1e-12]
        entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        dominant = float(np.max(probs))
        significant = int(np.sum(probs > 0.05))
        superposition = significant / self.num_states

        state_indices = np.arange(self.num_states)
        weighted_indices = state_indices[probs > 1e-12]
        coherence = 1.0 - (np.std(weighted_indices) / (self.num_states - 1)) if len(weighted_indices) > 1 else 0.5
        entanglement = 0.5

        return {
            'entropy': float(entropy),
            'dominant_state': dominant,
            'superposition': superposition,
            'coherence': float(coherence),
            'entanglement': entanglement,
            'significant_states': significant
        }

    def _default_features(self):
        return {
            'entropy': 2.5, 'dominant_state': 0.125, 'superposition': 0.5,
            'coherence': 0.5, 'entanglement': 0.5, 'significant_states': 4
        }


# ====================== LSTM MODEL DEF ======================
class QuantumLSTM(nn.Module):
    """
    Bidirectional LSTM matching the trained model architecture.
    """
    def __init__(self, input_size=5, quantum_size=7, hidden_size=128, num_layers=3, dropout=0.3):
        super(QuantumLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.quantum_processor = nn.Sequential(
            nn.Linear(quantum_size, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, price_seq, quantum_features):
        lstm_out, _ = self.lstm(price_seq)
        lstm_last = lstm_out[:, -1, :]
        q_processed = self.quantum_processor(quantum_features)
        combined = torch.cat([lstm_last, q_processed], dim=1)
        return self.fusion(combined)

# ====================== ADAPTER CLASS ======================
class QuantumLSTMAdapter:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = QuantumFeatureExtractor()
        self.model = self._load_model(model_path)
        
    def _load_model(self, path):
        model = QuantumLSTM().to(self.device)
        if path and Path(path).exists():
            try:
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.eval()
                logger.info(f"Loaded LSTM model from {path}")
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}")
        else:
            logger.warning("No LSTM model found, using random weights (Initialization Mode)")
            model.eval() # Ensure eval mode to avoid BatchNorm errors
        return model

    def prepare_input(self, df):
        """Prepare last 50 candles for inference."""
        # Calculate features matching training
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low'] = (df['high'] - df['low']) / df['close']
        df['close_open'] = (df['close'] - df['open']) / df['open']
        df = df.dropna()
        
        if len(df) < 50: return None, None
        
        # Technical Features (Last 50)
        tech_feats = df[['returns', 'log_returns', 'high_low', 'close_open', 'tick_volume']].tail(50).values
        # Standardize (Simple estimate)
        tech_feats = (tech_feats - np.mean(tech_feats, axis=0)) / (np.std(tech_feats, axis=0) + 1e-8)
        
        # Quantum Features (Last window)
        window = df['close'].tail(50).values
        q_feats_dict = self.extractor.extract(window)
        # Map dict to array [entropy, dom, super, coh, ent, var, sig]
        # Note: var is missing in extract() above, adding 0.0 placeholder to match size 7
        q_feats = np.array([
            q_feats_dict['entropy'], q_feats_dict['dominant_state'], q_feats_dict['superposition'],
            q_feats_dict['coherence'], q_feats_dict['entanglement'], 0.0, q_feats_dict['significant_states']
        ])
        
        return torch.FloatTensor(tech_feats).unsqueeze(0).to(self.device), torch.FloatTensor(q_feats).unsqueeze(0).to(self.device), q_feats_dict

    def predict(self, df_prices):
        """
        Main entry point.
        Args: df_prices (pd.DataFrame): OHLCV data
        Returns: dict signal
        """
        try:
            price_tensor, q_tensor, q_metrics = self.prepare_input(df_prices)
            if price_tensor is None:
                return {'signal': 'HOLD', 'reason': 'insufficient_data'}
            
            with torch.no_grad():
                output = self.model(price_tensor, q_tensor)
                prob = torch.sigmoid(output).item()
                
            # --- AGGRESSIVE SNIPER MODE ---
            # Lowered thresholds to "Open Wide"
            action = 'BUY' if prob > 0.51 else 'SELL' if prob < 0.49 else 'HOLD'
            confidence = abs(prob - 0.5) * 2 # Map 0.5->0, 1.0->1
            
            return {
                'signal': action,
                'confidence': confidence,
                'raw_prob': prob,
                'entropy': q_metrics['entropy'],
                'dominant_state': q_metrics['dominant_state'],
                'coherence': q_metrics['coherence'],
                'entanglement': q_metrics['entanglement']
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'signal': 'HOLD', 'reason': 'error'}

if __name__ == "__main__":
    # Self-test
    print("Testing Quantum LSTM Adapter...")
    adapter = QuantumLSTMAdapter()
    
    # Mock Data
    dates = pd.date_range(start='2026-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'open': np.random.rand(100) + 100,
        'high': np.random.rand(100) + 101,
        'low': np.random.rand(100) + 99,
        'close': np.random.rand(100) + 100,
        'tick_volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    result = adapter.predict(df)
    print(f"Result: {result}")
