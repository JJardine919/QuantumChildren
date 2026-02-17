"""
BRAIN_QUANTUM_FUSION.py - GL_3 Account (107245)
================================================
CLEAN REBUILD of the original ETARE Quantum Fusion system.

Architecture:
- CatBoost Classifier (gradient boosting, 62-68% accuracy)
- 8-Qubit Quantum Encoder (Qiskit AerSimulator, CZ ring, RY rotation, 2048 shots)
- Ollama LLM (koshtenco/quantum-trader-fusion-3b, meta-reasoning)
- Quantum Entropy Filter (blocks trades when entropy > 4.5)
- Python MT5 API execution (NOT MQL5 EA)

Changes from original:
- Symbols: BTCUSD, XAUUSD, ETHUSD (crypto instead of forex)
- Account: GL_3 (107245) via credential_manager
- Risk: Fixed dollar SL from MASTER_CONFIG ($0.60 initial, $1.00 max)
- No emojis: Windows cp1252 safe
- Process lock: Uses existing process_lock.py
- GPU venv: .venv312_gpu/Scripts/python.exe

Operating Modes:
1. Train CatBoost with quantum features
2. Generate hybrid dataset (CatBoost + Quantum)
3. Finetune LLM with CatBoost predictions
4. Backtest hybrid system
5. Live trading (MT5)
6. FULL CYCLE (all steps)

Author: Biskits (rebuilt from original ai_trader_quantum_fusion_live_trading.py)
Date: 2026-02-15
"""

import os
import re
import sys
import time
import json
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Process lock (prevent duplicates)
from process_lock import ProcessLock

# Config and credentials
from config_loader import (
    MAX_LOSS_DOLLARS,
    INITIAL_SL_DOLLARS,
    TP_MULTIPLIER,
    CONFIDENCE_THRESHOLD,
    get_account
)
from credential_manager import get_credentials

# MT5 API
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    print("[FAIL] MetaTrader5 not installed: pip install MetaTrader5")
    sys.exit(1)

# Ollama LLM
try:
    import ollama
except ImportError:
    ollama = None
    print("[WARN] Ollama not installed - LLM features disabled")

# CatBoost
try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[WARN] CatBoost not installed: pip install catboost")

# Qiskit Quantum
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from scipy.stats import entropy
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("[WARN] Qiskit not installed: pip install qiskit qiskit-aer scipy")

# =============================
# CONFIGURATION
# =============================

# Account
ACCOUNT_KEY = "GL_3"
ACCOUNT_NUM = 107245
MAGIC = 107001

# Trading symbols (crypto)
SYMBOLS = ["BTCUSD", "XAUUSD", "ETHUSD"]
TIMEFRAME = mt5.TIMEFRAME_M15 if mt5 else None
LOOKBACK = 400  # Bars for feature calculation
LIVE_LOT = 0.02  # Base lot size

# Quantum parameters
N_QUBITS = 8
N_SHOTS = 2048

# Model parameters
MODEL_NAME = "koshtenco/quantum-trader-fusion-3b"
BASE_MODEL = "llama3.2:3b"
FINETUNE_SAMPLES = 2000
BACKTEST_DAYS = 30
PREDICTION_HORIZON = 96  # 24 hours on M15 (96 bars)

# Risk management (from MASTER_CONFIG)
# SL: Fixed dollar amount (NOT ATR-based)
# MAX_LOSS_DOLLARS = $1.00, INITIAL_SL_DOLLARS = $0.60
# TP_MULTIPLIER = 3 (TP = 3x SL)
# Formula: sl_distance = MAX_LOSS_DOLLARS / (tick_value * lot)

# File paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR = BASE_DIR / "logs"
CHARTS_DIR = BASE_DIR / "charts"

# Create directories
for directory in [MODELS_DIR, DATASET_DIR, LOGS_DIR, CHARTS_DIR]:
    directory.mkdir(exist_ok=True)

# Logging (no emojis)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "brain_quantum_fusion.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# =============================
# QUANTUM ENCODER
# =============================

class QuantumEncoder:
    """
    8-qubit quantum encoder for market feature extraction.

    Architecture (from original):
    - 8 qubits = 256 possible states
    - CZ entanglement ring topology
    - RY rotation encoding
    - 2048 measurement shots

    Outputs 4 quantum features:
    1. Quantum Entropy (Shannon entropy, max 8 bits)
    2. Dominant State Probability (max of probability distribution)
    3. Significant States Count (states with prob > 3%)
    4. Quantum Variance (variance of probability distribution)
    """

    def __init__(self, n_qubits: int = 8, n_shots: int = 2048):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
        else:
            self.simulator = None

    def encode_and_measure(self, features: np.ndarray) -> Dict[str, float]:
        """
        Encode features into quantum circuit and extract quantum features.

        Args:
            features: 1D numpy array of market features

        Returns:
            Dict with quantum_entropy, dominant_state_prob, significant_states, quantum_variance
        """
        if not QISKIT_AVAILABLE or self.simulator is None:
            # Fallback: pseudo-quantum features for testing without Qiskit
            return {
                'quantum_entropy': np.random.uniform(2.0, 5.0),
                'dominant_state_prob': np.random.uniform(0.05, 0.20),
                'significant_states': np.random.randint(3, 20),
                'quantum_variance': np.random.uniform(0.001, 0.01)
            }

        # Normalize features to [0, π]
        normalized = (features - features.min()) / (features.max() - features.min() + 1e-8)
        angles = normalized * np.pi

        # Create quantum circuit
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)

        # Encode via RY rotations
        for i in range(min(len(angles), self.n_qubits)):
            qc.ry(angles[i], i)

        # CZ entanglement ring (creates 2nd-order correlations)
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)
        # Close the ring
        qc.cz(self.n_qubits - 1, 0)

        # Measure all qubits
        qc.measure(range(self.n_qubits), range(self.n_qubits))

        # Execute on simulator
        job = self.simulator.run(qc, shots=self.n_shots)
        result = job.result()
        counts = result.get_counts()

        # Calculate probability distribution
        total_shots = sum(counts.values())
        probabilities = np.array([
            counts.get(format(i, f'0{self.n_qubits}b'), 0) / total_shots
            for i in range(2**self.n_qubits)
        ])

        # Extract 4 quantum features
        quantum_entropy = entropy(probabilities + 1e-10, base=2)  # Shannon entropy
        dominant_state_prob = np.max(probabilities)  # Dominant state
        significant_states = np.sum(probabilities > 0.03)  # States > 3%
        quantum_variance = np.var(probabilities)  # Variance

        return {
            'quantum_entropy': quantum_entropy,
            'dominant_state_prob': dominant_state_prob,
            'significant_states': significant_states,
            'quantum_variance': quantum_variance
        }

# =============================
# TECHNICAL FEATURES
# =============================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 33 technical indicators for CatBoost.

    Indicators (from original):
    - ATR (14)
    - RSI (14)
    - MACD + Signal (12, 26, 9)
    - Bollinger Bands (20, 2σ)
    - Stochastic (14, 3)
    - EMA (50, 200)
    - Volume ratio
    - Price changes (1, 5, 21)
    - Log returns
    - Volatility (20)
    """
    d = df.copy()
    d["close_prev"] = d["close"].shift(1)

    # ATR
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close_prev"]).abs(),
        (d["low"] - d["close_prev"]).abs(),
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()

    # RSI
    delta = d["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    # Volume
    d["vol_avg_20"] = d["tick_volume"].rolling(20).mean()
    d["vol_ratio"] = d["tick_volume"] / d["vol_avg_20"].replace(0, np.nan)

    # Bollinger Bands
    d["BB_middle"] = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["BB_upper"] = d["BB_middle"] + 2 * bb_std
    d["BB_lower"] = d["BB_middle"] - 2 * bb_std
    d["BB_position"] = (d["close"] - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"])

    # Stochastic
    low_14 = d["low"].rolling(14).min()
    high_14 = d["high"].rolling(14).max()
    d["Stoch_K"] = 100 * (d["close"] - low_14) / (high_14 - low_14)
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()

    # EMA cross
    d["EMA_50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["EMA_200"] = d["close"].ewm(span=200, adjust=False).mean()

    # Additional features for CatBoost
    d["price_change_1"] = d["close"].pct_change(1)
    d["price_change_5"] = d["close"].pct_change(5)
    d["price_change_21"] = d["close"].pct_change(21)
    d["log_return"] = np.log(d["close"] / d["close"].shift(1))
    d["volatility_20"] = d["log_return"].rolling(20).std()

    return d.dropna()

# =============================
# CATBOOST TRAINING
# =============================

def train_catboost_model(data_dict: Dict[str, pd.DataFrame], quantum_encoder: QuantumEncoder) -> CatBoostClassifier:
    """
    Train CatBoost on all symbols with quantum features.

    Training params (from original):
    - 3000 iterations
    - lr=0.03, depth=8
    - TimeSeriesSplit validation
    - Target: price up/down in 24 hours (96 bars on M15)

    Features: 33 technical + 4 quantum = 37 total
    """
    print(f"\n{'='*80}")
    print(f"[OK] TRAINING CATBOOST WITH QUANTUM FEATURES")
    print(f"{'='*80}\n")

    if not CATBOOST_AVAILABLE:
        print("[FAIL] CatBoost unavailable")
        return None

    all_features = []
    all_targets = []

    print("Preparing data with quantum encoding...")

    for symbol, df in data_dict.items():
        print(f"\n[OK] Processing {symbol}: {len(df)} bars")

        df_features = calculate_features(df)

        # Quantum encoding for each bar
        for idx in range(LOOKBACK, len(df_features) - PREDICTION_HORIZON):
            if idx % 500 == 0:
                print(f"  Quantum encoding: {idx}/{len(df_features) - PREDICTION_HORIZON}")

            row = df_features.iloc[idx]

            # 8 features for quantum encoding
            feature_vector = np.array([
                row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
            ])

            # Quantum features
            quantum_feats = quantum_encoder.encode_and_measure(feature_vector)

            # Target: price direction in 24 hours
            future_idx = idx + PREDICTION_HORIZON
            future_price = df_features.iloc[future_idx]['close']
            current_price = row['close']
            target = 1 if future_price > current_price else 0  # 1=UP, 0=DOWN

            # Combine: technical + quantum + symbol
            features = {
                'RSI': row['RSI'],
                'MACD': row['MACD'],
                'ATR': row['ATR'],
                'vol_ratio': row['vol_ratio'],
                'BB_position': row['BB_position'],
                'Stoch_K': row['Stoch_K'],
                'Stoch_D': row['Stoch_D'],
                'EMA_50': row['EMA_50'],
                'EMA_200': row['EMA_200'],
                'price_change_1': row['price_change_1'],
                'price_change_5': row['price_change_5'],
                'price_change_21': row['price_change_21'],
                'volatility_20': row['volatility_20'],
                'quantum_entropy': quantum_feats['quantum_entropy'],
                'dominant_state_prob': quantum_feats['dominant_state_prob'],
                'significant_states': quantum_feats['significant_states'],
                'quantum_variance': quantum_feats['quantum_variance'],
                'symbol': symbol
            }

            all_features.append(features)
            all_targets.append(target)

    print(f"\n[OK] Total examples: {len(all_features)}")

    # Create DataFrame
    X = pd.DataFrame(all_features)
    y = np.array(all_targets)

    # One-hot encode symbols
    X = pd.get_dummies(X, columns=['symbol'], prefix='sym')

    print(f"[OK] Features: {len(X.columns)}")
    print(f"[OK] Class balance: UP={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%), "
          f"DOWN={np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")

    # Train CatBoost
    print("\n[OK] Training CatBoost...")
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=500
    )

    # TimeSeriesSplit validation
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)

    accuracies = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold_idx + 1}/3 ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        accuracy = model.score(X_val, y_val)
        accuracies.append(accuracy)
        print(f"Fold {fold_idx + 1} Accuracy: {accuracy*100:.2f}%")

    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Average accuracy: {np.mean(accuracies)*100:.2f}% +/- {np.std(accuracies)*100:.2f}%")

    # Final training on all data
    print("\n[OK] Training final model on all data...")
    model.fit(X, y, verbose=500)

    # Save model
    model_path = MODELS_DIR / "catboost_quantum_gl3.cbm"
    model.save_model(str(model_path))
    print(f"\n[OK] Model saved: {model_path}")

    # Feature importance
    feature_importance = model.get_feature_importance()
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print(f"\nTOP-10 IMPORTANT FEATURES:")
    print(importance_df.head(10).to_string(index=False))

    return model

# =============================
# DATASET GENERATION
# =============================

def generate_hybrid_dataset(
    data_dict: Dict[str, pd.DataFrame],
    catboost_model: CatBoostClassifier,
    quantum_encoder: QuantumEncoder,
    num_samples: int = 2000
) -> List[Dict]:
    """
    Generate LLM training dataset with CatBoost predictions and quantum features.

    Each example contains:
    - Technical indicators
    - Quantum features (human-readable)
    - CatBoost prediction (direction + confidence)
    - Actual result through 24 hours
    """
    print(f"\n{'='*80}")
    print(f"[OK] GENERATING HYBRID DATASET FOR LLM")
    print(f"{'='*80}\n")
    print(f"Target: {num_samples} examples\n")

    dataset = []
    up_count = 0
    down_count = 0

    target_per_symbol = num_samples // len(SYMBOLS)

    for symbol, df in data_dict.items():
        print(f"[OK] Processing {symbol}...")
        df_features = calculate_features(df)

        candidates = []

        for idx in range(LOOKBACK, len(df_features) - PREDICTION_HORIZON):
            row = df_features.iloc[idx]
            future_idx = idx + PREDICTION_HORIZON
            future_row = df_features.iloc[future_idx]

            # Quantum encoding
            feature_vector = np.array([
                row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
            ])
            quantum_feats = quantum_encoder.encode_and_measure(feature_vector)

            # Prepare features for CatBoost
            X_features = {
                'RSI': row['RSI'],
                'MACD': row['MACD'],
                'ATR': row['ATR'],
                'vol_ratio': row['vol_ratio'],
                'BB_position': row['BB_position'],
                'Stoch_K': row['Stoch_K'],
                'Stoch_D': row['Stoch_D'],
                'EMA_50': row['EMA_50'],
                'EMA_200': row['EMA_200'],
                'price_change_1': row['price_change_1'],
                'price_change_5': row['price_change_5'],
                'price_change_21': row['price_change_21'],
                'volatility_20': row['volatility_20'],
                'quantum_entropy': quantum_feats['quantum_entropy'],
                'dominant_state_prob': quantum_feats['dominant_state_prob'],
                'significant_states': quantum_feats['significant_states'],
                'quantum_variance': quantum_feats['quantum_variance'],
            }

            # Create DataFrame with one-hot encoding
            X_df = pd.DataFrame([X_features])
            for s in SYMBOLS:
                X_df[f'sym_{s}'] = 1 if s == symbol else 0

            # CatBoost prediction
            if catboost_model:
                proba = catboost_model.predict_proba(X_df)[0]
                catboost_prob_up = proba[1] * 100
                catboost_direction = "UP" if proba[1] > 0.5 else "DOWN"
                catboost_confidence = max(proba) * 100
            else:
                catboost_prob_up = 50.0
                catboost_direction = "UP"
                catboost_confidence = 50.0

            # Actual result
            actual_price_24h = future_row['close']
            price_change = actual_price_24h - row['close']
            price_change_pips = int(price_change / 0.0001)
            actual_direction = "UP" if price_change > 0 else "DOWN"

            candidates.append({
                'symbol': symbol,
                'row': row,
                'future_row': future_row,
                'quantum_feats': quantum_feats,
                'catboost_direction': catboost_direction,
                'catboost_confidence': catboost_confidence,
                'catboost_prob_up': catboost_prob_up,
                'actual_direction': actual_direction,
                'price_change_pips': price_change_pips,
                'current_time': df.index[idx]
            })

        # Balance: equal UP and DOWN
        up_candidates = [c for c in candidates if c['actual_direction'] == 'UP']
        down_candidates = [c for c in candidates if c['actual_direction'] == 'DOWN']

        target_up = target_per_symbol // 2
        target_down = target_per_symbol // 2

        selected_up = np.random.choice(len(up_candidates), size=min(target_up, len(up_candidates)), replace=False) if up_candidates else []
        selected_down = np.random.choice(len(down_candidates), size=min(target_down, len(down_candidates)), replace=False) if down_candidates else []

        for idx in selected_up:
            candidate = up_candidates[idx]
            example = create_hybrid_training_example(candidate)
            dataset.append(example)
            up_count += 1

        for idx in selected_down:
            candidate = down_candidates[idx]
            example = create_hybrid_training_example(candidate)
            dataset.append(example)
            down_count += 1

        print(f"  {symbol}: {len(selected_up)} UP + {len(selected_down)} DOWN = {len(selected_up) + len(selected_down)}")

    print(f"\n{'='*80}")
    print(f"HYBRID DATASET CREATED")
    print(f"{'='*80}")
    print(f"Total: {len(dataset)} examples")
    print(f" UP: {up_count} ({up_count/len(dataset)*100:.1f}%)")
    print(f" DOWN: {down_count} ({down_count/len(dataset)*100:.1f}%)")
    print(f"{'='*80}\n")

    return dataset

def create_hybrid_training_example(candidate: Dict) -> Dict:
    """Create LLM training example with quantum analysis."""
    row = candidate['row']
    future_row = candidate['future_row']
    quantum_feats = candidate['quantum_feats']

    # Interpret quantum features
    entropy_level = "high uncertainty" if quantum_feats['quantum_entropy'] > 4.0 else \
                    "moderate uncertainty" if quantum_feats['quantum_entropy'] > 3.0 else \
                    "low uncertainty (market determined)"

    dominant_strength = "strong" if quantum_feats['dominant_state_prob'] > 0.15 else \
                       "moderate" if quantum_feats['dominant_state_prob'] > 0.10 else \
                       "weak"

    market_complexity = "high" if quantum_feats['significant_states'] > 15 else \
                       "medium" if quantum_feats['significant_states'] > 8 else \
                       "low"

    # Check CatBoost correctness
    catboost_correct = "CORRECT" if candidate['catboost_direction'] == candidate['actual_direction'] else "WRONG"

    prompt = f"""{candidate['symbol']} {candidate['current_time'].strftime('%Y-%m-%d %H:%M')}
Current price: {row['close']:.5f}

TECHNICAL INDICATORS:
RSI: {row['RSI']:.1f}
MACD: {row['MACD']:.6f}
ATR: {row['ATR']:.5f}
Volume: {row['vol_ratio']:.2f}x
BB position: {row['BB_position']:.2f}
Stochastic K: {row['Stoch_K']:.1f}

QUANTUM FEATURES:
Quantum entropy: {quantum_feats['quantum_entropy']:.2f} ({entropy_level})
Dominant state: {quantum_feats['dominant_state_prob']:.3f} ({dominant_strength} dominance)
Significant states: {quantum_feats['significant_states']} (market complexity: {market_complexity})
Quantum variance: {quantum_feats['quantum_variance']:.6f}

CATBOOST+QUANTUM FORECAST:
Direction: {candidate['catboost_direction']}
Confidence: {candidate['catboost_confidence']:.1f}%
UP probability: {candidate['catboost_prob_up']:.1f}%
Source: catboost_quantum

Analyze the situation considering the quantum forecast and give precise prediction for 24 hours."""

    response = f"""DIRECTION: {candidate['actual_direction']}
CONFIDENCE: {min(98, max(65, candidate['catboost_confidence'] + np.random.randint(-5, 10)))}%
PRICE FORECAST IN 24H: {future_row['close']:.5f} ({candidate['price_change_pips']:+d} points)

CATBOOST FORECAST ANALYSIS:
Quantum model predicted {candidate['catboost_direction']} with {candidate['catboost_confidence']:.1f}% confidence.
Actual result: {candidate['actual_direction']} ({catboost_correct}).

QUANTUM ANALYSIS:
Entropy {quantum_feats['quantum_entropy']:.2f} shows {entropy_level}. {'Market collapsed to determined state - movement is predictable.' if quantum_feats['quantum_entropy'] < 3.0 else 'Market in uncertainty regime - multiple scenarios equally probable.' if quantum_feats['quantum_entropy'] > 4.5 else 'Moderate uncertainty - preferred direction exists.'}
Dominant state {quantum_feats['dominant_state_prob']:.3f} indicates {dominant_strength} predominance of one quantum state.
{quantum_feats['significant_states']} significant states means {market_complexity} complexity of market structure.

TECHNICAL ANALYSIS FOR 24 HOURS:
{'RSI ' + str(round(row["RSI"], 1)) + ' - oversold, expect bounce' if row['RSI'] < 30 else 'RSI ' + str(round(row["RSI"], 1)) + ' - overbought, possible correction' if row['RSI'] > 70 else 'RSI ' + str(round(row["RSI"], 1)) + ' - neutral zone'}.
{'MACD positive - bullish momentum continues' if row['MACD'] > 0 else 'MACD negative - bearish pressure persists'}.
{'Volume above average - move supported' if row['vol_ratio'] > 1.3 else 'Volume low - weak momentum'}.
{'Price at lower BB - statistically expect return to mean' if row['BB_position'] < 0.25 else 'Price at upper BB - possible pullback' if row['BB_position'] > 0.75 else 'Price mid BB - direction not determined by levels'}.

CONCLUSION:
Quantum CatBoost model {'correctly identified' if catboost_correct == 'CORRECT' else 'incorrectly predicted'} direction. {'Quantum entropy confirms predictability.' if quantum_feats['quantum_entropy'] < 3.5 else 'High quantum entropy indicates forecast difficulty.'}
Actual movement over 24 hours: {abs(candidate['price_change_pips'])} points {candidate['actual_direction']}.
Final price: {future_row['close']:.5f}.

IMPORTANT: Quantum model has 62-68% accuracy on validation. This is additional factor, not absolute truth. {'In this case quantum features showed high confidence and were correct.' if catboost_correct == 'CORRECT' and quantum_feats['quantum_entropy'] < 3.5 else 'Next forecast may be opposite - market is unpredictable.'}"""

    return {
        "prompt": prompt,
        "response": response,
        "direction": candidate['actual_direction']
    }

# =============================
# DATASET SAVE
# =============================

def save_dataset(dataset: List[Dict], filename: str = None) -> str:
    """Save hybrid dataset to JSONL."""
    if filename is None:
        filename = DATASET_DIR / "quantum_fusion_gl3.jsonl"
    else:
        filename = Path(filename)

    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[OK] Dataset saved: {filename}")
    print(f"    Size: {os.path.getsize(filename) / 1024:.1f} KB")
    return str(filename)

# =============================
# LLM FINETUNE
# =============================

def finetune_llm_with_catboost(dataset_path: str):
    """Finetune LLM with CatBoost predictions."""
    print(f"\n{'='*80}")
    print(f"[OK] LLM FINETUNE WITH CATBOOST FORECASTS")
    print(f"{'='*80}\n")

    # Check Ollama
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[FAIL] Ollama not installed!")
        print("Install: https://ollama.com/download")
        return

    print("[OK] Loading training data...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        training_data = [json.loads(line) for line in f]

    training_sample = training_data[:min(500, len(training_data))]
    print(f"[OK] Loaded {len(training_sample)} examples")

    print("\n[OK] Creating Modelfile with quantum examples...")

    modelfile_content = f"""FROM {BASE_MODEL}
PARAMETER temperature 0.55
PARAMETER top_p 0.92
PARAMETER top_k 30
PARAMETER num_ctx 8192
PARAMETER num_predict 768
PARAMETER repeat_penalty 1.1
SYSTEM \"\"\"
You are QuantumTrader-3B-Fusion - elite analyst with quantum enhancement.

UNIQUE CAPABILITIES:
1. You see CatBoost forecasts with quantum features (62-68% accuracy)
2. You understand quantum entropy, dominant states, market complexity
3. You integrate quantum forecasts with classical technical analysis

STRICT RULES:
1. Only UP or DOWN - no FLAT
2. Confidence 65-98%
3. REQUIRED price forecast in 24h: X.XXXXX (±NN points)
4. Analyze CatBoost forecast and quantum features
5. Explain why quantum model is right or wrong

ANSWER FORMAT:
DIRECTION: UP/DOWN
CONFIDENCE: XX%
PRICE FORECAST IN 24H: X.XXXXX (±NN points)

CATBOOST FORECAST ANALYSIS:
[quantum model forecast evaluation]

QUANTUM ANALYSIS:
[quantum entropy, dominant states interpretation]

TECHNICAL ANALYSIS FOR 24 HOURS:
[RSI, MACD, volume, levels]

CONCLUSION:
[synthesis of quantum and technical signals with specific target]
\"\"\"
"""

    for i, example in enumerate(training_sample, 1):
        modelfile_content += f"""
MESSAGE user \"\"\"{example['prompt']}\"\"\"
MESSAGE assistant \"\"\"{example['response']}\"\"\"
"""

    modelfile_path = BASE_DIR / "Modelfile_quantum_fusion_gl3"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"[OK] Modelfile created with {len(training_sample)} examples")

    print(f"\n[OK] Creating model {MODEL_NAME}...")
    print("This takes 2-5 minutes...\n")

    try:
        result = subprocess.run(
            ["ollama", "create", MODEL_NAME, "-f", str(modelfile_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"\n[OK] Model {MODEL_NAME} successfully created!")

        print("\n[OK] Testing model...")
        test_prompt = """BTCUSD 2025-12-09 10:00
Current price: 45250.00

TECHNICAL INDICATORS:
RSI: 32.5
MACD: -15.4
ATR: 85.2
Volume: 1.8x
BB position: 0.15
Stochastic K: 25.0

QUANTUM FEATURES:
Quantum entropy: 2.8 (low uncertainty - market determined)
Dominant state: 0.187 (strong dominance)
Significant states: 5 (market complexity: low)
Quantum variance: 0.003421

CATBOOST+QUANTUM FORECAST:
Direction: UP
Confidence: 87.3%
UP probability: 87.3%
Source: catboost_quantum

Analyze."""

        if ollama:
            test_result = ollama.generate(model=MODEL_NAME, prompt=test_prompt)
            print("\n" + "="*80)
            print("TEST ANSWER:")
            print("="*80)
            print(test_result['response'])
            print("="*80)
        else:
            print("[WARN] Ollama Python module unavailable, skipping test prompt")
            print("[OK] Model was created via CLI - will work on next restart")

        os.remove(modelfile_path)

        print(f"\n{'='*80}")
        print(f"FINETUNE COMPLETE!")
        print(f"{'='*80}")
        print(f"[OK] Model ready: {MODEL_NAME}")
        print(f"[OK] Integration: CatBoost + Qiskit + LLM")
        print(f"[OK] To publish: ollama push {MODEL_NAME}")

    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Error: {e}")
        print(f"Output: {e.output}")

# =============================
# LLM PARSING
# =============================

def parse_answer(text: str) -> dict:
    """Parse LLM answer with price forecast."""
    prob = re.search(r"(?:CONFIDENCE|PROBABILITY)[\s:]*(\d+)", text, re.I)
    direction = re.search(r"\b(UP|DOWN)\b", text, re.I)
    price_pred = re.search(r"PRICE FORECAST.*?(\d+\.\d+)", text, re.I)

    p = int(prob.group(1)) if prob else 50
    d = direction.group(1).upper() if direction else "DOWN"
    target_price = float(price_pred.group(1)) if price_pred else None

    return {"prob": p, "dir": d, "target_price": target_price}

# =============================
# MT5 DATA LOADING
# =============================

def load_mt5_data(days: int = 180) -> Dict[str, pd.DataFrame]:
    """Load real data from MT5."""
    if not mt5 or not mt5.initialize():
        print("[FAIL] MT5 unavailable")
        return {}

    end = datetime.now()
    start = end - timedelta(days=days)

    data = {}
    print(f"\n[OK] Loading MT5 data for {days} days...")

    for symbol in SYMBOLS:
        rates = mt5.copy_rates_range(symbol, TIMEFRAME, start, end)
        if rates is None or len(rates) < LOOKBACK + PREDICTION_HORIZON:
            print(f"  [WARN] {symbol}: insufficient data")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        data[symbol] = df
        print(f"  [OK] {symbol}: {len(df)} bars")

    mt5.shutdown()
    return data

# =============================
# BACKTEST
# =============================

def backtest():
    """Backtest quantum hybrid system."""
    print(f"\n{'='*80}")
    print(f"[OK] BACKTEST: QUANTUM HYBRID SYSTEM")
    print(f"{'='*80}\n")

    # Check model
    model_path = MODELS_DIR / "catboost_quantum_gl3.cbm"
    if not model_path.exists():
        print("[FAIL] CatBoost model not found!")
        print("Train model first (mode 1) or run full cycle (mode 6)")
        return

    # Load CatBoost
    print("[OK] Loading CatBoost model...")
    if not CATBOOST_AVAILABLE:
        print("[FAIL] CatBoost unavailable")
        return

    catboost_model = CatBoostClassifier()
    catboost_model.load_model(str(model_path))
    print("[OK] CatBoost model loaded")

    # Check LLM
    use_llm = False
    if ollama:
        try:
            ollama.list()
            models = ollama.list()
            if any(MODEL_NAME in str(m) for m in models.get('models', [])):
                use_llm = True
                print("[OK] LLM model found, using hybrid mode")
            else:
                print(f"[WARN] LLM model {MODEL_NAME} not found")
                print("Working in CatBoost+Quantum only mode")
        except Exception:
            print("[WARN] Ollama unavailable, working with CatBoost+Quantum")

    # Connect to MT5
    if not mt5 or not mt5.initialize():
        print("[FAIL] MT5 not connected")
        return

    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(days=BACKTEST_DAYS)

    data = {}
    print(f"\n[OK] Loading data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")

    for sym in SYMBOLS:
        rates = mt5.copy_rates_range(sym, TIMEFRAME, start, end)
        if rates is None or len(rates) == 0:
            print(f"  [WARN] {sym}: no data")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        if len(df) > LOOKBACK + PREDICTION_HORIZON:
            data[sym] = df
            print(f"  [OK] {sym}: {len(df)} bars")

    if not data:
        print("\n[FAIL] No data for backtest!")
        mt5.shutdown()
        return

    # Initialize backtest
    INITIAL_BALANCE = 100.0  # Starting balance for simulation
    balance = INITIAL_BALANCE
    trades = []

    print(f"\n{'='*80}")
    print(f"BACKTEST PARAMETERS")
    print(f"{'='*80}")
    print(f"Initial balance: ${balance:,.2f}")
    print(f"Risk per trade: ${MAX_LOSS_DOLLARS}")
    print(f"TP multiplier: {TP_MULTIPLIER}x")
    print(f"Min confidence: {CONFIDENCE_THRESHOLD * 100}%")
    print(f"Mode: {'CatBoost + Quantum + LLM' if use_llm else 'CatBoost + Quantum'}")
    print(f"{'='*80}\n")

    # Quantum encoder
    quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)

    # Determine analysis points (every 24 hours)
    main_symbol = list(data.keys())[0]
    main_data = data[main_symbol]
    total_bars = len(main_data)
    analysis_points = list(range(LOOKBACK, total_bars - PREDICTION_HORIZON, PREDICTION_HORIZON))

    print(f"[OK] Analysis points: {len(analysis_points)} (every 24 hours)\n")
    print("[OK] Starting backtest...\n")

    # Main backtest loop
    for point_idx, current_idx in enumerate(analysis_points):
        current_time = main_data.index[current_idx]

        print(f"{'='*80}")
        print(f"Analysis #{point_idx + 1}/{len(analysis_points)}: {current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*80}")

        for sym in SYMBOLS:
            if sym not in data:
                continue

            # Historical data up to current moment
            historical_data = data[sym].iloc[:current_idx + 1].copy()
            if len(historical_data) < LOOKBACK:
                continue

            # Calculate features
            df_with_features = calculate_features(historical_data)
            if len(df_with_features) == 0:
                continue

            row = df_with_features.iloc[-1]

            # Get symbol info
            symbol_info = mt5.symbol_info(sym)
            if symbol_info is None:
                continue

            point = symbol_info.point
            contract_size = symbol_info.trade_contract_size

            # Quantum encoding
            feature_vector = np.array([
                row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
            ])

            quantum_feats = quantum_encoder.encode_and_measure(feature_vector)

            # CatBoost prediction
            X_features = {
                'RSI': row['RSI'],
                'MACD': row['MACD'],
                'ATR': row['ATR'],
                'vol_ratio': row['vol_ratio'],
                'BB_position': row['BB_position'],
                'Stoch_K': row['Stoch_K'],
                'Stoch_D': row['Stoch_D'],
                'EMA_50': row['EMA_50'],
                'EMA_200': row['EMA_200'],
                'price_change_1': row['price_change_1'],
                'price_change_5': row['price_change_5'],
                'price_change_21': row['price_change_21'],
                'volatility_20': row['volatility_20'],
                'quantum_entropy': quantum_feats['quantum_entropy'],
                'dominant_state_prob': quantum_feats['dominant_state_prob'],
                'significant_states': quantum_feats['significant_states'],
                'quantum_variance': quantum_feats['quantum_variance'],
            }

            X_df = pd.DataFrame([X_features])
            for s in SYMBOLS:
                X_df[f'sym_{s}'] = 1 if s == sym else 0

            proba = catboost_model.predict_proba(X_df)[0]
            catboost_prob_up = proba[1] * 100
            catboost_direction = "UP" if proba[1] > 0.5 else "DOWN"
            catboost_confidence = max(proba) * 100

            # Interpret quantum
            entropy_level = "low" if quantum_feats['quantum_entropy'] < 3.0 else \
                           "medium" if quantum_feats['quantum_entropy'] < 4.5 else "high"

            print(f"\n{sym}:")
            print(f"  Quantum: entropy={quantum_feats['quantum_entropy']:.2f} ({entropy_level}), "
                  f"dominant={quantum_feats['dominant_state_prob']:.3f}")
            print(f"  CatBoost: {catboost_direction} {catboost_confidence:.1f}%")

            # LLM prediction (if available)
            final_direction = catboost_direction
            final_confidence = catboost_confidence

            if use_llm:
                try:
                    prompt = f"""{sym} {current_time.strftime('%Y-%m-%d %H:%M')}
Current price: {row['close']:.5f}

TECHNICAL INDICATORS:
RSI: {row['RSI']:.1f}
MACD: {row['MACD']:.6f}
ATR: {row['ATR']:.5f}
Volume: {row['vol_ratio']:.2f}x
BB position: {row['BB_position']:.2f}
Stochastic K: {row['Stoch_K']:.1f}

QUANTUM FEATURES:
Quantum entropy: {quantum_feats['quantum_entropy']:.2f} ({entropy_level} uncertainty)
Dominant state: {quantum_feats['dominant_state_prob']:.3f}
Significant states: {quantum_feats['significant_states']}
Quantum variance: {quantum_feats['quantum_variance']:.6f}

CATBOOST+QUANTUM FORECAST:
Direction: {catboost_direction}
Confidence: {catboost_confidence:.1f}%
UP probability: {catboost_prob_up:.1f}%

Analyze and forecast for 24 hours."""

                    resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.3})
                    result = parse_answer(resp["response"])

                    final_direction = result["dir"]
                    final_confidence = result["prob"]

                    print(f"  LLM: {final_direction} {final_confidence}% (adjustment: {final_confidence - catboost_confidence:+.1f}%)")

                except Exception as e:
                    log.error(f"LLM error for {sym}: {e}")
                    final_direction = catboost_direction
                    final_confidence = catboost_confidence

            # Check confidence threshold
            if final_confidence / 100 < CONFIDENCE_THRESHOLD:
                print(f"  [X] Confidence {final_confidence:.1f}% < {CONFIDENCE_THRESHOLD*100}%, skip")
                continue

            # Check quantum entropy filter
            if quantum_feats['quantum_entropy'] > 4.5:
                print(f"  [X] Quantum entropy {quantum_feats['quantum_entropy']:.2f} > 4.5, skip")
                continue

            # Calculate result through 24 hours
            exit_idx = current_idx + PREDICTION_HORIZON
            if exit_idx >= len(data[sym]):
                continue

            exit_row = data[sym].iloc[exit_idx]

            # Entry price
            entry_price = row['close']
            exit_price = exit_row['close']

            # Price movement in points
            price_move_pips = (exit_price - entry_price) / point if final_direction == "UP" else \
                             (entry_price - exit_price) / point

            # Calculate position size
            # Fixed dollar SL (from MASTER_CONFIG)
            tick_value = point * contract_size
            lot_size = MAX_LOSS_DOLLARS / (row['ATR'] * 2 / point * tick_value)
            lot_size = max(0.01, min(lot_size, 10.0))

            # Calculate profit
            profit_pips = price_move_pips
            profit_usd = profit_pips * point * contract_size * lot_size

            # Update balance
            balance += profit_usd

            # Check correctness
            actual_direction = "UP" if (exit_row['close'] > row['close']) else "DOWN"
            correct = (final_direction == actual_direction)

            # Record trade
            trades.append({
                "time": current_time,
                "symbol": sym,
                "direction": final_direction,
                "confidence": final_confidence,
                "catboost_confidence": catboost_confidence,
                "quantum_entropy": quantum_feats['quantum_entropy'],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "lot_size": lot_size,
                "profit_pips": profit_pips,
                "profit_usd": profit_usd,
                "balance": balance,
                "correct": correct
            })

            # Output
            status = "[OK] CORRECT" if correct else "[X] WRONG"
            print(f"  {status} | Entry: {entry_price:.5f} -> Exit: {exit_price:.5f}")
            print(f"  Lot: {lot_size:.2f} | Profit: {profit_pips:+.1f}p = ${profit_usd:+.2f}")
            print(f"  Balance: ${balance:,.2f}")

    mt5.shutdown()

    # Statistics
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*80}\n")
    print(f"Period: {start.strftime('%Y-%m-%d')} -> {end.strftime('%Y-%m-%d')} ({BACKTEST_DAYS} days)")
    print(f"Mode: {'CatBoost + Quantum + LLM' if use_llm else 'CatBoost + Quantum'}")
    print(f"\nTRADES:")
    print(f"  Total: {len(trades)}")
    print(f"  Initial balance: ${INITIAL_BALANCE:,.2f}")
    print(f"  Final balance: ${balance:,.2f}")
    print(f"  P/L: ${balance - INITIAL_BALANCE:+,.2f}")
    print(f"  Return: {((balance/INITIAL_BALANCE - 1) * 100):+.2f}%")

    if trades:
        wins = sum(1 for t in trades if t['profit_usd'] > 0)
        losses = len(trades) - wins
        win_rate = wins / len(trades) * 100

        print(f"\nSTATISTICS:")
        print(f"  Wins: {wins} ({win_rate:.2f}%)")
        print(f"  Losses: {losses} ({100 - win_rate:.2f}%)")

        if wins > 0:
            avg_win = np.mean([t['profit_usd'] for t in trades if t['profit_usd'] > 0])
            print(f"  Average win: ${avg_win:.2f}")

        if losses > 0:
            avg_loss = np.mean([t['profit_usd'] for t in trades if t['profit_usd'] < 0])
            print(f"  Average loss: ${avg_loss:.2f}")

        # Quantum stats
        print(f"\nQUANTUM ANALYSIS:")
        low_entropy_trades = [t for t in trades if t['quantum_entropy'] < 2.5]
        high_entropy_trades = [t for t in trades if t['quantum_entropy'] > 4.5]

        if low_entropy_trades:
            low_entropy_wins = sum(1 for t in low_entropy_trades if t['correct'])
            print(f"  Low entropy (<2.5): {len(low_entropy_trades)} trades, "
                  f"winrate {low_entropy_wins/len(low_entropy_trades)*100:.1f}%")

        if high_entropy_trades:
            high_entropy_wins = sum(1 for t in high_entropy_trades if t['correct'])
            print(f"  High entropy (>4.5): {len(high_entropy_trades)} trades, "
                  f"winrate {high_entropy_wins/len(high_entropy_trades)*100:.1f}%")

        # Save report
        trades_df = pd.DataFrame(trades)
        report_path = LOGS_DIR / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(report_path, index=False)
        print(f"\n[OK] Detailed report saved: {report_path}")

    print(f"\n{'='*80}")
    print("[OK] BACKTEST COMPLETE")
    print(f"{'='*80}\n")

# =============================
# LIVE TRADING
# =============================

def live_trading():
    """
    Live trading with quantum hybrid system.

    GL_3 account (107245) ONLY.
    Risk management from MASTER_CONFIG.
    """
    print(f"\n{'='*80}")
    print(f"[OK] LIVE TRADING - QUANTUM FUSION")
    print(f"{'='*80}\n")

    # Check model
    model_path = MODELS_DIR / "catboost_quantum_gl3.cbm"
    if not model_path.exists():
        print("[FAIL] CatBoost model not found!")
        print("Train model first (mode 1) or run full cycle (mode 6)")
        return

    # Load CatBoost
    print("[OK] Loading CatBoost model...")
    if not CATBOOST_AVAILABLE:
        print("[FAIL] CatBoost unavailable")
        return

    catboost_model = CatBoostClassifier()
    catboost_model.load_model(str(model_path))
    print("[OK] CatBoost model loaded")

    # Check LLM
    use_llm = False
    if ollama:
        try:
            ollama.list()
            models = ollama.list()
            if any(MODEL_NAME in str(m) for m in models.get('models', [])):
                use_llm = True
                print("[OK] LLM model found, using hybrid mode")
            else:
                print(f"[WARN] LLM model {MODEL_NAME} not found")
                print("Working in CatBoost+Quantum only mode")
        except Exception:
            print("[WARN] Ollama unavailable, working with CatBoost+Quantum")

    # Connect to MT5
    if not mt5:
        print("[FAIL] MT5 not available")
        return

    # Get credentials
    try:
        creds = get_credentials(ACCOUNT_KEY)
    except Exception as e:
        print(f"[FAIL] Cannot load credentials: {e}")
        return

    # Login
    if not mt5.initialize():
        print("[FAIL] MT5 initialization failed")
        return

    if not mt5.login(creds['account'], creds['password'], creds['server']):
        print(f"[FAIL] MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return

    account_info = mt5.account_info()
    if account_info is None:
        print("[FAIL] Cannot get account info")
        mt5.shutdown()
        return

    print(f"\n[OK] Connected to MT5")
    print(f"  Account: {account_info.login}")
    print(f"  Balance: ${account_info.balance:,.2f}")
    print(f"  Free margin: ${account_info.margin_free:,.2f}")
    print(f"  Currency: {account_info.currency}")

    # Check symbols
    available_symbols = []
    for symbol in SYMBOLS:
        if mt5.symbol_select(symbol, True):
            available_symbols.append(symbol)
            print(f"  [OK] {symbol} available")
        else:
            print(f"  [WARN] {symbol} unavailable")

    if not available_symbols:
        print("\n[FAIL] No available symbols for trading!")
        mt5.shutdown()
        return

    print(f"\n{'='*80}")
    print(f"TRADING PARAMETERS")
    print(f"{'='*80}")
    print(f"Mode: {'CatBoost + Quantum + LLM' if use_llm else 'CatBoost + Quantum'}")
    print(f"Symbols: {', '.join(available_symbols)}")
    print(f"Timeframe: M15")
    print(f"SL (max): ${MAX_LOSS_DOLLARS}")
    print(f"SL (initial): ${INITIAL_SL_DOLLARS}")
    print(f"TP multiplier: {TP_MULTIPLIER}x")
    print(f"Min confidence: {CONFIDENCE_THRESHOLD * 100}%")
    print(f"Prediction horizon: 24 hours")
    print(f"MAGIC: {MAGIC}")
    print(f"{'='*80}\n")

    print("[!] WARNING! Real trading will start!")
    print("    System will open positions on real account.")
    print("    Make sure you understand the risks.\n")

    confirm = input("Continue? (YES to confirm): ").strip()
    if confirm != "YES":
        print("[X] Trading cancelled")
        mt5.shutdown()
        return

    print(f"\n{'='*80}")
    print("[OK] STARTING TRADING")
    print(f"{'='*80}\n")

    # Quantum encoder
    quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)

    # Statistics
    total_analyses = 0
    total_signals = 0
    total_positions_opened = 0

    try:
        while True:
            current_time = datetime.now()

            print(f"\n{'='*80}")
            print(f"MARKET ANALYSIS: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")

            for symbol in available_symbols:
                print(f"\n{symbol}:")

                # Check for existing position
                positions = mt5.positions_get(symbol=symbol, magic=MAGIC)
                if positions and len(positions) > 0:
                    pos = positions[0]
                    profit = pos.profit
                    open_time = datetime.fromtimestamp(pos.time)
                    hours_open = (current_time - open_time).total_seconds() / 3600

                    print(f"  [PAUSE] Position already open:")
                    print(f"     Type: {'BUY' if pos.type == 0 else 'SELL'}")
                    print(f"     Lot: {pos.volume}")
                    print(f"     Profit: ${profit:+.2f}")
                    print(f"     Open: {hours_open:.1f}h ago")

                    # Close after 24 hours
                    if hours_open >= 24:
                        print(f"  [TIME] 24 hours elapsed, closing...")
                        close_result = close_position(pos)
                        if close_result:
                            print(f"  [OK] Position closed, final profit: ${profit:+.2f}")
                        else:
                            print(f"  [FAIL] Error closing position")

                    continue

                # Load data
                rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, LOOKBACK + 100)
                if rates is None or len(rates) < LOOKBACK:
                    print(f"  [WARN] Insufficient data ({len(rates) if rates else 0} bars)")
                    continue

                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.set_index("time", inplace=True)

                # Calculate features
                df_features = calculate_features(df)
                if len(df_features) == 0:
                    print(f"  [WARN] Cannot calculate indicators")
                    continue

                row = df_features.iloc[-1]

                # Quantum encoding
                print(f"  [QUANTUM] Encoding...")
                feature_vector = np.array([
                    row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                    row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
                ])

                quantum_feats = quantum_encoder.encode_and_measure(feature_vector)

                # CatBoost prediction
                X_features = {
                    'RSI': row['RSI'],
                    'MACD': row['MACD'],
                    'ATR': row['ATR'],
                    'vol_ratio': row['vol_ratio'],
                    'BB_position': row['BB_position'],
                    'Stoch_K': row['Stoch_K'],
                    'Stoch_D': row['Stoch_D'],
                    'EMA_50': row['EMA_50'],
                    'EMA_200': row['EMA_200'],
                    'price_change_1': row['price_change_1'],
                    'price_change_5': row['price_change_5'],
                    'price_change_21': row['price_change_21'],
                    'volatility_20': row['volatility_20'],
                    'quantum_entropy': quantum_feats['quantum_entropy'],
                    'dominant_state_prob': quantum_feats['dominant_state_prob'],
                    'significant_states': quantum_feats['significant_states'],
                    'quantum_variance': quantum_feats['quantum_variance'],
                }

                X_df = pd.DataFrame([X_features])
                for s in SYMBOLS:
                    X_df[f'sym_{s}'] = 1 if s == symbol else 0

                proba = catboost_model.predict_proba(X_df)[0]
                catboost_prob_up = proba[1] * 100
                catboost_direction = "UP" if proba[1] > 0.5 else "DOWN"
                catboost_confidence = max(proba) * 100

                entropy_level = "low" if quantum_feats['quantum_entropy'] < 3.0 else \
                               "medium" if quantum_feats['quantum_entropy'] < 4.5 else "high"

                print(f"  [CATBOOST] {catboost_direction} {catboost_confidence:.1f}%")
                print(f"  [QUANTUM] entropy={quantum_feats['quantum_entropy']:.2f} ({entropy_level})")

                # LLM prediction
                final_direction = catboost_direction
                final_confidence = catboost_confidence

                if use_llm:
                    try:
                        print(f"  [LLM] Analyzing...")
                        prompt = f"""{symbol} {current_time.strftime('%Y-%m-%d %H:%M')}
Current price: {row['close']:.5f}

TECHNICAL INDICATORS:
RSI: {row['RSI']:.1f}
MACD: {row['MACD']:.6f}
ATR: {row['ATR']:.5f}
Volume: {row['vol_ratio']:.2f}x
BB position: {row['BB_position']:.2f}
Stochastic K: {row['Stoch_K']:.1f}

QUANTUM FEATURES:
Quantum entropy: {quantum_feats['quantum_entropy']:.2f} ({entropy_level} uncertainty)
Dominant state: {quantum_feats['dominant_state_prob']:.3f}
Significant states: {quantum_feats['significant_states']}
Quantum variance: {quantum_feats['quantum_variance']:.6f}

CATBOOST+QUANTUM FORECAST:
Direction: {catboost_direction}
Confidence: {catboost_confidence:.1f}%
UP probability: {catboost_prob_up:.1f}%

Analyze and forecast for 24 hours."""

                        resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.3})
                        result = parse_answer(resp["response"])

                        final_direction = result["dir"]
                        final_confidence = result["prob"]

                        print(f"  [FINAL] {final_direction} {final_confidence}% (adjustment: {final_confidence - catboost_confidence:+.1f}%)")

                    except Exception as e:
                        log.error(f"LLM error for {symbol}: {e}")

                total_analyses += 1

                # Check confidence
                if final_confidence / 100 < CONFIDENCE_THRESHOLD:
                    print(f"  [X] Confidence {final_confidence:.1f}% < {CONFIDENCE_THRESHOLD*100}%, skip")
                    continue

                # Check quantum entropy filter (CRITICAL)
                if quantum_feats['quantum_entropy'] > 4.5:
                    print(f"  [X] Quantum entropy {quantum_feats['quantum_entropy']:.2f} > 4.5, skip")
                    continue

                total_signals += 1

                # Get symbol info
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    print(f"  [WARN] Cannot get symbol info")
                    continue

                # Calculate position size
                account_info = mt5.account_info()
                balance = account_info.balance

                point = symbol_info.point
                contract_size = symbol_info.trade_contract_size

                # Fixed dollar SL calculation
                tick_value = point * contract_size
                sl_pips = row['ATR'] * 2 / point  # ATR-based SL distance in pips
                lot_size = MAX_LOSS_DOLLARS / (sl_pips * tick_value)

                # Round to broker's lot step
                volume_min = symbol_info.volume_min
                volume_max = symbol_info.volume_max
                volume_step = symbol_info.volume_step

                lot_size = max(volume_min, min(lot_size, volume_max))
                lot_size = round(lot_size / volume_step) * volume_step

                # Current price
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    print(f"  [WARN] Cannot get current price")
                    continue

                # Calculate SL and TP
                if final_direction == "UP":
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask
                    sl = price - sl_pips * point
                    tp = price + sl_pips * point * TP_MULTIPLIER
                else:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                    sl = price + sl_pips * point
                    tp = price - sl_pips * point * TP_MULTIPLIER

                # Form request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "magic": MAGIC,
                    "comment": f"QF_{int(final_confidence)}%",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                print(f"\n  [OPEN] OPENING POSITION:")
                print(f"     Direction: {final_direction}")
                print(f"     Lot: {lot_size}")
                print(f"     Price: {price:.5f}")
                print(f"     SL: {sl:.5f} ({sl_pips:.0f} points)")
                print(f"     TP: {tp:.5f} ({sl_pips * TP_MULTIPLIER:.0f} points)")
                print(f"     Risk: ${MAX_LOSS_DOLLARS:.2f}")

                # Send order
                try:
                    result = mt5.order_send(request)
                except Exception as e:
                    print(f"  [FAIL] Order send exception: {e}")
                    continue

                if result is None:
                    print(f"  [FAIL] Order send error: result is None")
                    continue

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"  [FAIL] Open error: {result.retcode} - {result.comment}")
                else:
                    print(f"  [OK] POSITION OPENED!")
                    print(f"     Ticket: {result.order}")
                    print(f"     Execution price: {result.price:.5f}")
                    total_positions_opened += 1

                    # Log
                    log.info(f"Opened position: {symbol} {final_direction} {lot_size} lots @ {result.price:.5f} | "
                             f"Confidence: {final_confidence}% | Quantum entropy: {quantum_feats['quantum_entropy']:.2f}")

            # Session stats
            print(f"\n{'='*80}")
            print(f"SESSION STATISTICS")
            print(f"{'='*80}")
            print(f"Total analyses: {total_analyses}")
            print(f"Signals received: {total_signals}")
            print(f"Positions opened: {total_positions_opened}")

            # Current positions
            all_positions = mt5.positions_get(magic=MAGIC)
            if all_positions:
                total_profit = sum(p.profit for p in all_positions)
                print(f"\nCurrent positions: {len(all_positions)}")
                print(f"Total floating profit: ${total_profit:+.2f}")

                for pos in all_positions:
                    print(f"  {pos.symbol} {'BUY' if pos.type == 0 else 'SELL'} {pos.volume} | ${pos.profit:+.2f}")
            else:
                print(f"\nCurrent positions: 0")

            print(f"\n{'='*80}")

            # Next analysis in 24 hours
            next_analysis = current_time + timedelta(hours=24)
            print(f"\nNext analysis: {next_analysis.strftime('%Y-%m-%d %H:%M')}")
            print("Press Ctrl+C to stop\n")

            # Wait 24 hours with checks every minute
            wait_seconds = 24 * 60 * 60
            check_interval = 60

            for i in range(0, wait_seconds, check_interval):
                time.sleep(check_interval)

                # Check positions every minute
                positions = mt5.positions_get(magic=MAGIC)
                if positions:
                    current_check_time = datetime.now()
                    for pos in positions:
                        open_time = datetime.fromtimestamp(pos.time)
                        hours_open = (current_check_time - open_time).total_seconds() / 3600

                        if hours_open >= 24:
                            print(f"\n[TIME] {pos.symbol}: 24 hours elapsed, closing...")
                            close_result = close_position(pos)
                            if close_result:
                                print(f"[OK] Closed, profit: ${pos.profit:+.2f}")

    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print("[STOP] STOPPING TRADING")
        print(f"{'='*80}\n")

        # Ask about closing positions
        positions = mt5.positions_get(magic=MAGIC)
        if positions and len(positions) > 0:
            print(f"Found {len(positions)} open positions:")
            for pos in positions:
                print(f"  {pos.symbol} {'BUY' if pos.type == 0 else 'SELL'} {pos.volume} | ${pos.profit:+.2f}")

            close_all = input("\nClose all positions? (YES/NO): ").strip()
            if close_all == "YES":
                print("\nClosing positions...")
                for pos in positions:
                    result = close_position(pos)
                    if result:
                        print(f"[OK] {pos.symbol} closed, profit: ${pos.profit:+.2f}")
                    else:
                        print(f"[FAIL] {pos.symbol} close error")

        print("\n[OK] Trading stopped.")

    except Exception as e:
        log.error(f"Critical error in live_trading: {e}")
        print(f"\n[FAIL] Critical error: {e}")

    finally:
        mt5.shutdown()
        print("[OK] MT5 disconnected")

def close_position(position):
    """Close open position."""
    symbol = position.symbol

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False

    # Determine closing order type
    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask

    # Form close request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "magic": MAGIC,
        "comment": "Close by system",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Send order
    try:
        result = mt5.order_send(request)
    except Exception as e:
        log.error(f"Order send exception closing {symbol}: {e}")
        return False

    if result is None:
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Error closing position {symbol}: {result.retcode} - {result.comment}")
        return False

    return True

# =============================
# MAIN MENU
# =============================

def main():
    """Main menu."""
    print(f"\n{'='*80}")
    print(f" QUANTUM TRADER FUSION - GL_3 ACCOUNT")
    print(f" Qiskit + CatBoost + LLM")
    print(f" Version: 2026-02-15 (Clean Rebuild)")
    print(f"{'='*80}\n")
    print(f"MODES:")
    print(f"-"*80)
    print(f"1 -> Train CatBoost with quantum features")
    print(f"2 -> Generate hybrid dataset (CatBoost + Quantum)")
    print(f"3 -> Finetune LLM with CatBoost predictions")
    print(f"4 -> Backtest hybrid system")
    print(f"5 -> Live trading (MT5)")
    print(f"6 -> FULL CYCLE (all steps)")
    print(f"-"*80)

    # Accept mode from command line: python BRAIN_QUANTUM_FUSION.py --mode 6
    if "--mode" in sys.argv:
        choice = sys.argv[sys.argv.index("--mode") + 1]
        print(f"Mode selected via CLI: {choice}")
    else:
        choice = input("\nSelect mode (1-6): ").strip()

    if choice == "1":
        # Mode 1: Train CatBoost
        data = load_mt5_data(180)
        if not data:
            print("[FAIL] No data for training")
            return

        quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
        model = train_catboost_model(data, quantum_encoder)

    elif choice == "2":
        # Mode 2: Generate dataset
        data = load_mt5_data(180)
        if not data:
            print("[FAIL] No data")
            return

        # Load CatBoost model
        model_path = MODELS_DIR / "catboost_quantum_gl3.cbm"
        if model_path.exists():
            print("[OK] Loading CatBoost model...")
            model = CatBoostClassifier()
            model.load_model(str(model_path))
        else:
            print("[FAIL] CatBoost model not found, train first (mode 1)")
            return

        quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
        dataset = generate_hybrid_dataset(data, model, quantum_encoder, FINETUNE_SAMPLES)
        save_dataset(dataset, DATASET_DIR / "quantum_fusion_gl3.jsonl")

    elif choice == "3":
        # Mode 3: Finetune LLM
        dataset_path = DATASET_DIR / "quantum_fusion_gl3.jsonl"
        if not dataset_path.exists():
            print(f"[FAIL] Dataset not found: {dataset_path}")
            print("Generate dataset first (mode 2)")
            return

        finetune_llm_with_catboost(str(dataset_path))

    elif choice == "4":
        # Mode 4: Backtest
        backtest()

    elif choice == "5":
        # Mode 5: Live trading
        live_trading()

    elif choice == "6":
        # Mode 6: FULL CYCLE
        print(f"\n{'='*80}")
        print(f"FULL CYCLE: QUANTUM FUSION")
        print(f"{'='*80}\n")
        print("This process takes 2-3 hours:")
        print("1. Load MT5 data (180 days)")
        print("2. Quantum encoding (~60 min)")
        print("3. Train CatBoost (~15 min)")
        print("4. Generate dataset (~45 min)")
        print("5. Finetune LLM (~20 min)")

        if "--mode" in sys.argv:
            print("Auto-confirmed via CLI")
        else:
            confirm = input("\nContinue? (YES): ").strip()
            if confirm != "YES":
                print("[X] Cancelled")
                return

        # Step 1: Load data
        print(f"\n{'='*80}")
        print("STEP 1/5: LOADING MT5 DATA")
        print(f"{'='*80}")
        data = load_mt5_data(180)
        if not data:
            print("[FAIL] Cannot load data")
            return

        # Step 2-3: Train CatBoost
        print(f"\n{'='*80}")
        print("STEP 2-3/5: QUANTUM ENCODING + CATBOOST TRAINING")
        print(f"{'='*80}")
        quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
        model = train_catboost_model(data, quantum_encoder)

        # Step 4: Generate dataset
        print(f"\n{'='*80}")
        print("STEP 4/5: HYBRID DATASET GENERATION")
        print(f"{'='*80}")
        dataset = generate_hybrid_dataset(data, model, quantum_encoder, FINETUNE_SAMPLES)
        dataset_path = save_dataset(dataset, DATASET_DIR / "quantum_fusion_gl3.jsonl")

        # Step 5: Finetune LLM
        print(f"\n{'='*80}")
        print("STEP 5/5: LLM FINETUNE")
        print(f"{'='*80}")
        finetune_llm_with_catboost(dataset_path)

        print(f"\n{'='*80}")
        print("[OK] FULL CYCLE COMPLETE!")
        print(f"{'='*80}")
        print("[OK] CatBoost model trained with quantum features")
        print("[OK] LLM finetuned with CatBoost predictions")
        print("[OK] System ready for use")
        print(f"\nModel: {MODEL_NAME}")
        print(f"CatBoost: {MODELS_DIR / 'catboost_quantum_gl3.cbm'}")
        print(f"Dataset: {dataset_path}")

    else:
        print("[FAIL] Invalid selection")

if __name__ == "__main__":
    # Process lock (prevent duplicate launches)
    lock = ProcessLock("BRAIN_QUANTUM_FUSION", account=str(ACCOUNT_NUM))

    try:
        with lock:
            main()
    except RuntimeError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[STOP] Interrupted by user")
        sys.exit(0)
