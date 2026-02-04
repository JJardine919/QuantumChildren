# ================================================
# bg_brain_integrated.py
# Blue Guardian Integrated Trading Brain
# ETARE + Quantum Compression + LLM Watchdog
# Version 2.0 - Production Ready
# ================================================
import os
import sys
import json
import time
import logging
import threading
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import sqlite3

# Local imports
from archiver_service import get_archiver, QuantumFeatureArchiver
from llm_watchdog import get_watchdog, LLMWatchdog, AlertLevel, WatchdogDecision

# Optional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed - LSTM features disabled")

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("WARNING: Qiskit not installed - using pseudo-quantum features")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | BG_BRAIN | %(message)s",
    handlers=[
        logging.FileHandler("bg_brain.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ====================== CONFIGURATION ======================

class Config:
    """Global configuration"""
    # ETARE Parameters
    POPULATION_SIZE = 50
    MUTATION_RATE = 0.15
    CROSSOVER_RATE = 0.7
    EXTINCTION_THRESHOLD = 0.3
    GENERATIONS_PER_EXTINCTION = 10

    # Quantum Parameters
    N_QUBITS = 3
    N_SHOTS = 1000
    QUANTUM_WINDOW = 50

    # LSTM Parameters
    SEQUENCE_LENGTH = 50
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT = 0.3

    # Trading Parameters
    MIN_CONFIDENCE = 0.60
    MAX_LOT_SIZE = 0.5
    ATR_SL_MULTIPLIER = 1.5
    ATR_TP_MULTIPLIER = 3.0

    # Signal file paths
    SIGNAL_DIR = "."
    MARKET_DATA_FILE = "market_data.json"


# ====================== QUANTUM FEATURE EXTRACTOR ======================

class QuantumFeatureExtractor:
    """
    Quantum feature extractor using Qiskit.
    Provides 7 quantum metrics for market analysis.
    """

    def __init__(self, num_qubits: int = 3, shots: int = 1000):
        self.num_qubits = num_qubits
        self.shots = shots
        self.cache: Dict[str, Dict] = {}

        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator(method='statevector')
        else:
            self.simulator = None

        log.info(f"Quantum extractor initialized: {num_qubits} qubits, {shots} shots")

    def extract_features(self, price_data: np.ndarray) -> Dict[str, float]:
        """
        Extract quantum features from price data.

        Args:
            price_data: Array of closing prices

        Returns:
            Dict of 7 quantum features
        """
        # Check cache
        data_hash = hashlib.md5(price_data.tobytes()).hexdigest()
        if data_hash in self.cache:
            return self.cache[data_hash]

        if not QISKIT_AVAILABLE:
            return self._pseudo_quantum_features(price_data)

        try:
            # Calculate classical features for encoding
            returns = np.diff(price_data) / (price_data[:-1] + 1e-10)
            features = np.array([
                np.mean(returns),
                np.std(returns),
                np.max(returns) - np.min(returns)
            ])
            features = np.tanh(features)

            # Create quantum circuit
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)

            # RY gates for encoding
            for i in range(self.num_qubits):
                angle = np.clip(np.pi * features[i % len(features)], -2*np.pi, 2*np.pi)
                qc.ry(angle, i)

            # CNOT gates for entanglement
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)

            # Measure
            qc.measure(range(self.num_qubits), range(self.num_qubits))

            # Execute
            compiled = transpile(qc, self.simulator, optimization_level=2)
            job = self.simulator.run(compiled, shots=self.shots)
            counts = job.result().get_counts()

            # Extract quantum metrics
            quantum_features = self._compute_metrics(counts)
            self.cache[data_hash] = quantum_features

            return quantum_features

        except Exception as e:
            log.error(f"Quantum extraction error: {e}")
            return self._pseudo_quantum_features(price_data)

    def _compute_metrics(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Compute 7 quantum metrics from measurement counts"""
        total = sum(counts.values())
        probabilities = {state: count/total for state, count in counts.items()}

        # 1. Quantum entropy
        quantum_entropy = -sum(
            p * np.log2(p) if p > 0 else 0
            for p in probabilities.values()
        )

        # 2. Dominant state probability
        dominant_state_prob = max(probabilities.values())

        # 3. Superposition measure
        significant = sum(1 for p in probabilities.values() if p > 0.05)
        superposition_measure = significant / (2 ** self.num_qubits)

        # 4. Phase coherence
        state_values = [int(state, 2) for state in probabilities.keys()]
        max_value = 2 ** self.num_qubits - 1
        phase_coherence = 1.0 - (np.std(state_values) / max_value) if len(state_values) > 1 else 0.5

        # 5. Entanglement degree
        correlations = []
        for i in range(self.num_qubits - 1):
            corr = sum(
                prob for state, prob in probabilities.items()
                if len(state) > i + 1 and state[-(i+1)] == state[-(i+2)]
            )
            correlations.append(corr)
        entanglement_degree = np.mean(correlations) if correlations else 0.5

        # 6. Quantum variance
        mean_state = sum(int(state, 2) * prob for state, prob in probabilities.items())
        quantum_variance = sum(
            (int(state, 2) - mean_state)**2 * prob
            for state, prob in probabilities.items()
        )

        # 7. Number of significant states
        num_significant_states = float(significant)

        return {
            'quantum_entropy': quantum_entropy,
            'dominant_state_prob': dominant_state_prob,
            'superposition_measure': superposition_measure,
            'phase_coherence': phase_coherence,
            'entanglement_degree': entanglement_degree,
            'quantum_variance': quantum_variance,
            'num_significant_states': num_significant_states
        }

    def _pseudo_quantum_features(self, price_data: np.ndarray) -> Dict[str, float]:
        """Fallback pseudo-quantum features when Qiskit unavailable"""
        returns = np.diff(price_data) / (price_data[:-1] + 1e-10)
        volatility = np.std(returns)

        return {
            'quantum_entropy': 2.0 + volatility * 10,
            'dominant_state_prob': 0.15 + np.random.uniform(-0.05, 0.05),
            'superposition_measure': 0.4 + volatility * 5,
            'phase_coherence': 0.6 + np.random.uniform(-0.1, 0.1),
            'entanglement_degree': 0.5 + np.random.uniform(-0.1, 0.1),
            'quantum_variance': volatility * 0.1,
            'num_significant_states': 4.0 + np.random.randint(-1, 2)
        }


# ====================== ETARE EVOLUTIONARY SYSTEM ======================

class TradingIndividual:
    """
    Individual in ETARE population.
    Represents a trading strategy with neural network weights.
    """

    def __init__(self, input_size: int = 20, hidden_size: int = 32):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Neural network weights
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.bias1 = np.zeros(hidden_size)
        self.weights2 = np.random.randn(hidden_size, 6) * 0.1  # 6 actions
        self.bias2 = np.zeros(6)

        # Grid trading parameters
        self.grid_size = np.random.uniform(0.001, 0.01)
        self.max_positions = np.random.randint(1, 5)

        # Fitness tracking
        self.fitness = 0.0
        self.trades_count = 0
        self.win_count = 0
        self.total_profit = 0.0

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through neural network"""
        hidden = np.tanh(np.dot(features, self.weights1) + self.bias1)
        output = np.dot(hidden, self.weights2) + self.bias2
        return self._softmax(output)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-10)

    def get_action(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Get trading action from features.

        Returns:
            Tuple of (action_index, confidence)
            Actions: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE_BUY, 4=CLOSE_SELL, 5=CLOSE_ALL
        """
        probs = self.forward(features)
        action = np.argmax(probs)
        confidence = probs[action]
        return int(action), float(confidence)

    def mutate(self, rate: float = 0.15):
        """Apply random mutation to weights"""
        if np.random.random() < rate:
            self.weights1 += np.random.randn(*self.weights1.shape) * 0.1
            self.weights2 += np.random.randn(*self.weights2.shape) * 0.1
            self.grid_size *= np.random.uniform(0.9, 1.1)
            self.max_positions = max(1, self.max_positions + np.random.choice([-1, 0, 1]))

    def crossover(self, other: 'TradingIndividual') -> 'TradingIndividual':
        """Create offspring through crossover"""
        child = TradingIndividual(self.input_size, self.hidden_size)

        # Uniform crossover for weights
        mask1 = np.random.random(self.weights1.shape) > 0.5
        child.weights1 = np.where(mask1, self.weights1, other.weights1)

        mask2 = np.random.random(self.weights2.shape) > 0.5
        child.weights2 = np.where(mask2, self.weights2, other.weights2)

        # Average parameters
        child.grid_size = (self.grid_size + other.grid_size) / 2
        child.max_positions = (self.max_positions + other.max_positions) // 2

        return child

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'weights1': self.weights1.tolist(),
            'bias1': self.bias1.tolist(),
            'weights2': self.weights2.tolist(),
            'bias2': self.bias2.tolist(),
            'grid_size': self.grid_size,
            'max_positions': self.max_positions,
            'fitness': self.fitness,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingIndividual':
        """Deserialize from dictionary"""
        ind = cls()
        try:
            ind.weights1 = np.array(data['weights1'])
            ind.bias1 = np.array(data['bias1'])
            ind.weights2 = np.array(data['weights2'])
            ind.bias2 = np.array(data['bias2'])
            ind.grid_size = data.get('grid_size', 0.005)
            ind.max_positions = data.get('max_positions', 3)
            ind.fitness = data.get('fitness', 0.0)
        except KeyError:
            # Incompatible archive format - return fresh individual
            pass
        return ind


class ETARESystem:
    """
    Evolutionary Trading Algorithm with Reinforcement and Extinction.

    Features:
    - Population of trading strategies
    - Genetic evolution (crossover, mutation)
    - Periodic extinction events
    - Fitness-based selection
    """

    ACTION_NAMES = ['HOLD', 'BUY', 'SELL', 'CLOSE_BUY', 'CLOSE_SELL', 'CLOSE_ALL']

    def __init__(self, population_size: int = 50,
                 archiver: QuantumFeatureArchiver = None):
        self.population_size = population_size
        self.archiver = archiver

        self.population: List[TradingIndividual] = []
        self.generation = 0
        self.best_fitness = 0.0
        self.best_individual: Optional[TradingIndividual] = None

        self._initialize_population()
        log.info(f"ETARE system initialized: {population_size} individuals")

    def _initialize_population(self):
        """Create initial population"""
        # Try to load from archive
        if self.archiver:
            best = self.archiver.get_best_etare_individual()
            if best:
                log.info("Loaded best individual from archive")
                self.best_individual = TradingIndividual.from_dict(best)

        # Create population
        self.population = [TradingIndividual() for _ in range(self.population_size)]

        # Inject best individual if available
        if self.best_individual:
            self.population[0] = self.best_individual

    def get_decision(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Get trading decision from ensemble of top individuals.

        Args:
            features: Input feature array

        Returns:
            Tuple of (action_name, confidence)
        """
        # Get votes from top 10 individuals
        top_individuals = sorted(
            self.population,
            key=lambda x: x.fitness,
            reverse=True
        )[:10]

        votes = [0] * 6
        confidences = [0.0] * 6

        for ind in top_individuals:
            action, conf = ind.get_action(features)
            votes[action] += 1
            confidences[action] += conf

        # Weighted voting
        best_action = np.argmax(votes)
        avg_confidence = confidences[best_action] / (votes[best_action] + 1e-10)

        return self.ACTION_NAMES[best_action], avg_confidence

    def update_fitness(self, individual_idx: int, profit: float, won: bool):
        """Update fitness of an individual based on trade result"""
        ind = self.population[individual_idx]
        ind.trades_count += 1
        ind.total_profit += profit
        if won:
            ind.win_count += 1

        # Fitness = Sharpe-like ratio
        if ind.trades_count > 0:
            win_rate = ind.win_count / ind.trades_count
            ind.fitness = (ind.total_profit / (ind.trades_count + 1)) * win_rate

        # Track best
        if ind.fitness > self.best_fitness:
            self.best_fitness = ind.fitness
            self.best_individual = ind

    def evolve(self):
        """Perform one generation of evolution"""
        self.generation += 1

        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Selection: Keep top 20%
        elite_count = self.population_size // 5
        elites = self.population[:elite_count]

        # Crossover to fill rest
        new_population = elites.copy()

        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if np.random.random() < Config.CROSSOVER_RATE:
                child = parent1.crossover(parent2)
            else:
                child = TradingIndividual()

            child.mutate(Config.MUTATION_RATE)
            new_population.append(child)

        self.population = new_population[:self.population_size]

        # Archive population
        if self.archiver:
            self.archiver.store_etare_population(
                self.generation,
                [ind.to_dict() for ind in self.population[:10]]  # Top 10
            )

        log.info(f"ETARE evolved to generation {self.generation}, best fitness: {self.best_fitness:.4f}")

    def _tournament_select(self, tournament_size: int = 5) -> TradingIndividual:
        """Tournament selection"""
        candidates = np.random.choice(
            self.population,
            size=min(tournament_size, len(self.population)),
            replace=False
        )
        return max(candidates, key=lambda x: x.fitness)

    def extinction_event(self):
        """
        Perform extinction event - remove weak individuals.
        Triggered periodically to maintain diversity.
        """
        threshold = self.best_fitness * Config.EXTINCTION_THRESHOLD

        survivors = [ind for ind in self.population if ind.fitness >= threshold]
        extinct_count = len(self.population) - len(survivors)

        # Add new random individuals
        while len(survivors) < self.population_size:
            survivors.append(TradingIndividual())

        self.population = survivors
        log.warning(f"EXTINCTION EVENT: {extinct_count} individuals extinct")


# ====================== BIDIRECTIONAL LSTM MODEL ======================

if TORCH_AVAILABLE:
    class QuantumLSTM(nn.Module):
        """
        Bidirectional LSTM with quantum feature integration.
        """

        def __init__(self, input_size: int = 5, quantum_size: int = 7,
                     hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True
            )

            self.quantum_processor = nn.Sequential(
                nn.Linear(quantum_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU()
            )

            self.fusion = nn.Sequential(
                nn.Linear(hidden_size * 2 + 32, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

        def forward(self, price_seq, quantum_features):
            lstm_out, _ = self.lstm(price_seq)
            lstm_last = lstm_out[:, -1, :]
            quantum_processed = self.quantum_processor(quantum_features)
            combined = torch.cat([lstm_last, quantum_processed], dim=1)
            return self.fusion(combined)


# ====================== INTEGRATED BRAIN ======================

class BlueGuardianBrain:
    """
    Integrated trading brain combining ETARE, Quantum, and LLM watchdog.

    Architecture:
    1. Quantum Feature Extraction - 7 quantum metrics from price data
    2. ETARE Evolutionary System - Adaptive trading strategies
    3. LLM Watchdog - Emergency shutoff and risk management
    4. Multi-Account Support - Isolated trading per account
    """

    def __init__(self, config_path: str = "accounts_config.json"):
        self.config_path = config_path
        self.accounts: Dict[str, Dict] = {}
        self.account_states: Dict[str, Dict] = {}

        # Initialize components
        self.archiver = get_archiver("bg_archive.db")
        self.watchdog = get_watchdog()
        self.quantum = QuantumFeatureExtractor(Config.N_QUBITS, Config.N_SHOTS)
        self.etare = ETARESystem(Config.POPULATION_SIZE, self.archiver)

        # LSTM model (if available)
        self.lstm_model = None
        self._load_lstm_model()

        # Load accounts
        self._load_accounts()

        # Set up watchdog callback
        self.watchdog.set_emergency_callback(self._handle_emergency)

        self.running = False
        log.info("Blue Guardian Brain initialized")

    def _load_accounts(self):
        """Load account configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.accounts = {
                    acc['name']: acc
                    for acc in config.get('accounts', [])
                }
                log.info(f"Loaded {len(self.accounts)} accounts")
        except FileNotFoundError:
            log.warning(f"Config not found: {self.config_path}")
            self.accounts = {}

    def _load_lstm_model(self):
        """Load pre-trained LSTM model"""
        if not TORCH_AVAILABLE:
            return

        model_path = "models/quantum_lstm_best.pth"
        if os.path.exists(model_path):
            try:
                self.lstm_model = QuantumLSTM()
                self.lstm_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.lstm_model.eval()
                log.info("LSTM model loaded")
            except Exception as e:
                log.error(f"Failed to load LSTM: {e}")

    def _handle_emergency(self, account_name: str, decision: WatchdogDecision):
        """Handle emergency from watchdog"""
        log.critical(f"EMERGENCY for {account_name}: {decision.reason}")

        # Write emergency signal
        signal = {
            'timestamp': datetime.now().isoformat(),
            'status': 'BLOCKED',
            'block_reason': decision.reason,
            'action': 'CLOSE_ALL' if decision.close_positions else 'HALT',
            'alert_level': decision.alert_level.value,
        }

        signal_file = f"signal_{account_name}.json"
        with open(signal_file, 'w') as f:
            json.dump(signal, f)

        log.critical(f"Emergency signal written: {signal_file}")

    def _read_market_data(self) -> Optional[Dict]:
        """Read market data from JSON file"""
        try:
            with open(Config.MARKET_DATA_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            return None

    def _prepare_features(self, bars: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Prepare features for ETARE and LSTM.

        Returns:
            Tuple of (feature_array, quantum_features)
        """
        if len(bars) < Config.QUANTUM_WINDOW:
            return None, {}

        # Extract price data
        closes = np.array([b['close'] for b in bars[-Config.QUANTUM_WINDOW:]])
        highs = np.array([b['high'] for b in bars[-Config.QUANTUM_WINDOW:]])
        lows = np.array([b['low'] for b in bars[-Config.QUANTUM_WINDOW:]])
        volumes = np.array([b.get('tick_volume', 0) for b in bars[-Config.QUANTUM_WINDOW:]])

        # Calculate returns
        returns = np.diff(closes) / (closes[:-1] + 1e-10)

        # Technical indicators
        rsi = self._calculate_rsi(closes)
        atr = self._calculate_atr(highs, lows, closes)
        macd, signal = self._calculate_macd(closes)

        # Quantum features
        quantum_features = self.quantum.extract_features(closes)

        # Archive quantum features
        self.archiver.store_quantum_features(
            "BTCUSD",
            datetime.now().isoformat(),
            quantum_features
        )

        # Combine features for ETARE
        features = np.array([
            returns[-1] if len(returns) > 0 else 0,
            np.std(returns) if len(returns) > 1 else 0,
            rsi,
            atr / closes[-1] if closes[-1] > 0 else 0,
            macd,
            signal,
            (closes[-1] - np.mean(closes)) / (np.std(closes) + 1e-10),
            volumes[-1] / (np.mean(volumes) + 1e-10),
            quantum_features['quantum_entropy'],
            quantum_features['dominant_state_prob'],
            quantum_features['superposition_measure'],
            quantum_features['phase_coherence'],
            quantum_features['entanglement_degree'],
            quantum_features['quantum_variance'],
            quantum_features['num_significant_states'],
            # Padding to 20 features
            0, 0, 0, 0, 0
        ])

        return features, quantum_features

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, period: int = 14) -> float:
        """Calculate ATR"""
        if len(closes) < period + 1:
            return highs[-1] - lows[-1]

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        return np.mean(tr[-period:])

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate MACD and signal line"""
        if len(prices) < 26:
            return 0.0, 0.0

        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd = ema12 - ema26
        signal = self._ema(np.array([macd]), 9)  # Simplified

        return macd, signal

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1]

        multiplier = 2 / (period + 1)
        ema = prices[-period]

        for price in prices[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _get_lstm_prediction(self, bars: List[Dict],
                             quantum_features: Dict) -> Optional[Tuple[str, float]]:
        """Get LSTM prediction if model available"""
        if not TORCH_AVAILABLE or self.lstm_model is None:
            return None

        if len(bars) < Config.SEQUENCE_LENGTH:
            return None

        try:
            # Prepare sequence
            closes = np.array([b['close'] for b in bars[-Config.SEQUENCE_LENGTH:]])
            opens = np.array([b['open'] for b in bars[-Config.SEQUENCE_LENGTH:]])
            highs = np.array([b['high'] for b in bars[-Config.SEQUENCE_LENGTH:]])
            lows = np.array([b['low'] for b in bars[-Config.SEQUENCE_LENGTH:]])
            volumes = np.array([b.get('tick_volume', 1) for b in bars[-Config.SEQUENCE_LENGTH:]])

            # Calculate features
            returns = np.diff(closes) / (closes[:-1] + 1e-10)
            log_returns = np.log(closes[1:] / closes[:-1] + 1e-10)
            high_low = (highs[1:] - lows[1:]) / (closes[1:] + 1e-10)
            close_open = (closes[1:] - opens[1:]) / (opens[1:] + 1e-10)

            # Stack features
            features = np.stack([
                returns,
                log_returns,
                high_low,
                close_open,
                volumes[1:] / (np.mean(volumes) + 1e-10)
            ], axis=1)

            # Normalize
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-8
            features = (features - mean) / std

            # Convert to tensors
            price_tensor = torch.FloatTensor(features).unsqueeze(0)
            quantum_tensor = torch.FloatTensor([
                quantum_features['quantum_entropy'],
                quantum_features['dominant_state_prob'],
                quantum_features['superposition_measure'],
                quantum_features['phase_coherence'],
                quantum_features['entanglement_degree'],
                quantum_features['quantum_variance'],
                quantum_features['num_significant_states'],
            ]).unsqueeze(0)

            # Predict
            with torch.no_grad():
                output = self.lstm_model(price_tensor, quantum_tensor)
                prob = torch.sigmoid(output).item()

            direction = "BUY" if prob > 0.5 else "SELL"
            confidence = max(prob, 1 - prob)

            return direction, confidence

        except Exception as e:
            log.error(f"LSTM prediction error: {e}")
            return None

    def generate_signal(self, account_name: str) -> Dict:
        """
        Generate trading signal for an account.

        Returns:
            Signal dict with action, confidence, etc.
        """
        account = self.accounts.get(account_name, {})

        # Read market data
        market_data = self._read_market_data()
        if not market_data:
            return {'status': 'NO_DATA', 'action': 'HOLD'}

        # Get symbol data
        symbol = account.get('symbol', 'BTCUSD')
        bars = market_data.get(symbol, [])

        if not bars or len(bars) < Config.QUANTUM_WINDOW:
            return {'status': 'INSUFFICIENT_DATA', 'action': 'HOLD'}

        # Prepare features
        features, quantum_features = self._prepare_features(bars)
        if features is None:
            return {'status': 'FEATURE_ERROR', 'action': 'HOLD'}

        # Update watchdog
        self.watchdog.update_account_state(account_name, {
            'balance': account.get('balance', 10000),
            'equity': account.get('equity', 10000),
            'daily_drawdown': account.get('daily_drawdown', 0),
            'max_drawdown': account.get('max_drawdown', 0),
            'open_positions': account.get('open_positions', 0),
            'unrealized_pnl': account.get('unrealized_pnl', 0),
        })

        # Check watchdog
        decision = self.watchdog.check_account(
            account_name,
            quantum_features=quantum_features,
            use_llm=True
        )

        if not decision.allow_trading:
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'BLOCKED',
                'block_reason': decision.reason,
                'alert_level': decision.alert_level.value,
                'action': 'CLOSE_ALL' if decision.close_positions else 'HOLD',
            }

        # Get ETARE decision
        etare_action, etare_confidence = self.etare.get_decision(features)

        # Get LSTM prediction (if available)
        lstm_result = self._get_lstm_prediction(bars, quantum_features)

        # Combine signals
        if lstm_result:
            lstm_direction, lstm_confidence = lstm_result

            # Ensemble: ETARE + LSTM
            if etare_action in ['BUY', 'SELL'] and lstm_direction == etare_action:
                # Agreement - boost confidence
                final_confidence = (etare_confidence + lstm_confidence) / 2 + 0.1
                final_action = etare_action
            elif etare_action in ['BUY', 'SELL']:
                # Disagreement - use LSTM if more confident
                if lstm_confidence > etare_confidence:
                    final_action = lstm_direction
                    final_confidence = lstm_confidence
                else:
                    final_action = etare_action
                    final_confidence = etare_confidence * 0.9  # Reduce confidence
            else:
                final_action = etare_action
                final_confidence = etare_confidence
        else:
            final_action = etare_action
            final_confidence = etare_confidence

        # Apply minimum confidence threshold
        if final_confidence < Config.MIN_CONFIDENCE:
            final_action = 'HOLD'

        # Calculate position size
        max_lot = account.get('max_lot_size', Config.MAX_LOT_SIZE)
        current_price = bars[-1]['close']
        atr = self._calculate_atr(
            np.array([b['high'] for b in bars[-14:]]),
            np.array([b['low'] for b in bars[-14:]]),
            np.array([b['close'] for b in bars[-14:]])
        )

        # Signal
        signal = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': final_action,
            'confidence': round(final_confidence, 4),
            'etare_action': etare_action,
            'etare_confidence': round(etare_confidence, 4),
            'lstm_available': lstm_result is not None,
            'current_price': current_price,
            'atr': round(atr, 5),
            'max_lot_size': max_lot,
            'magic_number': account.get('magic_number', 100001),
            'quantum_features': {
                k: round(v, 4) for k, v in quantum_features.items()
            },
            'watchdog_status': decision.alert_level.value,
            'status': 'OK'
        }

        return signal

    def write_signal(self, account_name: str, signal: Dict):
        """Write signal to file for MT5 executor"""
        signal_file = f"signal_{account_name}.json"

        with open(signal_file, 'w') as f:
            json.dump(signal, f, indent=2)

        log.info(f"Signal written: {account_name} -> {signal.get('action', 'HOLD')} "
                f"({signal.get('confidence', 0)*100:.1f}%)")

    def run_cycle(self):
        """Run one trading cycle for all accounts"""
        for account_name in self.accounts:
            try:
                signal = self.generate_signal(account_name)
                self.write_signal(account_name, signal)
            except Exception as e:
                log.error(f"Error processing {account_name}: {e}")

    def start(self, interval_seconds: int = 30):
        """Start continuous trading loop"""
        self.running = True
        self.watchdog.start_monitoring()

        log.info(f"Brain started, checking every {interval_seconds}s")

        while self.running:
            try:
                self.run_cycle()

                # Evolve ETARE periodically
                if self.etare.generation % Config.GENERATIONS_PER_EXTINCTION == 0:
                    self.etare.extinction_event()

                self.etare.evolve()

                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                log.info("Keyboard interrupt received")
                break
            except Exception as e:
                log.error(f"Cycle error: {e}")
                time.sleep(10)

        self.watchdog.stop_monitoring()
        log.info("Brain stopped")

    def stop(self):
        """Stop the brain"""
        self.running = False


# ====================== MAIN ======================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Blue Guardian Integrated Brain")
    parser.add_argument('--config', default='accounts_config.json',
                        help='Path to accounts config file')
    parser.add_argument('--interval', type=int, default=30,
                        help='Signal check interval in seconds')
    parser.add_argument('--once', action='store_true',
                        help='Run single cycle and exit')

    args = parser.parse_args()

    print("=" * 60)
    print("BLUE GUARDIAN INTEGRATED BRAIN v2.0")
    print("ETARE + Quantum Compression + LLM Watchdog")
    print("=" * 60)

    brain = BlueGuardianBrain(args.config)

    if args.once:
        brain.run_cycle()
    else:
        brain.start(args.interval)


if __name__ == "__main__":
    main()
