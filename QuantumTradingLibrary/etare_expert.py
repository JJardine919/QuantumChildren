"""
ETARE Expert - Numpy Feedforward Expert Loader
================================================
Loads the trained ETARE BTCUSD expert (JSON weights) and provides
a predict() interface compatible with the BRAIN scripts.

Supports v1 (20 features) and v2 (27 features = 20 technical + 7 quantum).
Dimension-safe: truncates or pads as needed for backward compatibility.

Architecture: INPUT -> 128 (tanh) -> 64 (tanh) -> 6 (softmax)
Actions: OPEN_BUY, OPEN_SELL, CLOSE_BUY_PROFIT, CLOSE_BUY_LOSS, CLOSE_SELL_PROFIT, CLOSE_SELL_LOSS
Mapped to: BUY, SELL, HOLD, HOLD, HOLD, HOLD
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

try:
    from quantum_feature_defs import (
        QUANTUM_FEATURE_NAMES,
        QUANTUM_FEATURE_DEFAULTS,
        QUANTUM_FEATURE_COUNT,
    )
    from quantum_feature_fetcher import fetch_latest_quantum_features
except ImportError:
    QUANTUM_FEATURE_NAMES = []
    QUANTUM_FEATURE_DEFAULTS = {}
    QUANTUM_FEATURE_COUNT = 7
    def fetch_latest_quantum_features(*args, **kwargs):
        return None

log = logging.getLogger(__name__)

# Action mapping: ETARE 6-action -> brain 3-action
# 0=OPEN_BUY -> BUY, 1=OPEN_SELL -> SELL, 2-5 -> HOLD
_ETARE_TO_DIRECTION = {0: "BUY", 1: "SELL", 2: "HOLD", 3: "HOLD", 4: "HOLD", 5: "HOLD"}


class ETAREExpert:
    """Numpy feedforward expert loaded from JSON weights."""

    def __init__(self, json_path: str):
        self.loaded = False
        self.symbol = None
        self._load(json_path)

    def _load(self, json_path: str):
        try:
            with open(json_path) as f:
                data = json.load(f)

            self.input_weights = np.array(data["input_weights"])
            self.hidden_weights = np.array(data["hidden_weights"])
            self.output_weights = np.array(data["output_weights"])
            self.hidden_bias = np.array(data["hidden_bias"])
            self.hidden2_bias = np.array(data["hidden2_bias"])
            self.output_bias = np.array(data["output_bias"])
            self.feature_order = data.get("feature_order", [])
            self.fitness = data.get("fitness", 0.0)
            self.win_rate = data.get("win_rate", 0.0)
            self.version = data.get("version", 1)
            self.quantum_features_included = data.get("quantum_features_included", False)
            self.expected_input_size = data.get("input_size", self.input_weights.shape[0])
            self.loaded = True
            log.info(
                f"ETARE expert loaded: v{self.version}, fitness={self.fitness:.4f}, "
                f"WR={self.win_rate*100:.1f}%, input_size={self.expected_input_size}, "
                f"quantum={self.quantum_features_included}, arch={data.get('architecture','unknown')}"
            )
        except Exception as e:
            log.warning(f"Failed to load ETARE expert from {json_path}: {e}")
            self.loaded = False

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Run numpy forward pass. Returns softmax probabilities over 6 actions."""
        s = state.reshape(1, -1)
        h1 = np.tanh(s @ self.input_weights + self.hidden_bias)
        h2 = np.tanh(h1 @ self.hidden_weights + self.hidden2_bias)
        q = h2 @ self.output_weights + self.output_bias
        # Softmax
        e = np.exp(q[0] - np.max(q[0]))
        probs = e / e.sum()
        return probs

    def predict(self, state: np.ndarray) -> Tuple[str, float]:
        """
        Predict direction from a state vector.
        Dimension-safe: handles v1 (20-input) and v2 (27-input) experts.
        Returns (direction, confidence) where direction is "BUY", "SELL", or "HOLD".
        """
        state_size = state.shape[0] if state.ndim == 1 else state.shape[-1]

        if state_size > self.expected_input_size:
            # 27 features fed to v1 (20-input) expert: truncate
            log.debug(f"ETARE: Truncating {state_size} features to {self.expected_input_size} for v{self.version} expert")
            state = state[:self.expected_input_size]
        elif state_size < self.expected_input_size:
            # 20 features fed to v2 (27-input) expert: pad with zero (z-scored defaults)
            log.warning(f"ETARE: Padding {state_size} features to {self.expected_input_size} for v{self.version} expert")
            pad = np.zeros(self.expected_input_size - state_size)
            state = np.concatenate([state, pad])

        probs = self.forward(state)
        action_idx = int(np.argmax(probs))
        confidence = float(probs[action_idx])
        direction = _ETARE_TO_DIRECTION[action_idx]
        return direction, confidence


def prepare_etare_features(df: pd.DataFrame, symbol: str = "BTCUSD") -> Optional[np.ndarray]:
    """
    Calculate the 27 features (20 technical + 7 quantum) the ETARE expert expects.
    Input: DataFrame with columns [open, high, low, close, tick_volume].
    Returns: 1D numpy array of 27 z-score normalized features for the LAST bar,
             or None if insufficient data.
    """
    d = df.copy()
    for c in ["open", "high", "low", "close", "tick_volume"]:
        if c in d.columns:
            d[c] = d[c].astype(float)

    # RSI (14)
    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    d["rsi"] = 100 - (100 / (1 + rs))

    # MACD (12-26-9)
    exp12 = d["close"].ewm(span=12, adjust=False).mean()
    exp26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = exp12 - exp26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    d["bb_middle"] = d["close"].rolling(20).mean()
    d["bb_std"] = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_middle"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_middle"] - 2 * d["bb_std"]

    # EMAs
    for p in [5, 10, 20, 50]:
        d[f"ema_{p}"] = d["close"].ewm(span=p, adjust=False).mean()

    # Momentum
    d["momentum"] = d["close"] / d["close"].shift(10)

    # NOTE: This is 14-bar High-Low Range, NOT true ATR. Experts were trained with this formula.
    d["atr"] = d["high"].rolling(14).max() - d["low"].rolling(14).min()

    # Price changes
    d["price_change"] = d["close"].pct_change()
    d["price_change_abs"] = d["price_change"].abs()

    # Volume
    d["volume_ma"] = d["tick_volume"].rolling(20).mean()
    d["volume_std"] = d["tick_volume"].rolling(20).std()

    # Stochastic K/D
    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = 100 * (d["close"] - low14) / (high14 - low14 + 1e-10)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # ROC
    d["roc"] = d["close"].pct_change(10) * 100

    d = d.dropna()
    if len(d) < 10:
        return None

    feature_cols = [
        "rsi", "macd", "macd_signal",
        "bb_middle", "bb_upper", "bb_lower", "bb_std",
        "ema_5", "ema_10", "ema_20", "ema_50",
        "momentum", "atr",
        "price_change", "price_change_abs",
        "volume_ma", "volume_std",
        "stoch_k", "stoch_d",
        "roc",
    ]

    tech_features = d[feature_cols].values.astype(np.float64)
    n_rows = tech_features.shape[0]

    # Fetch latest quantum features and tile across all rows
    quantum_vals = fetch_latest_quantum_features(symbol)
    quantum_tiled = np.tile(quantum_vals, (n_rows, 1))  # (n_rows, 7)

    features = np.concatenate([tech_features, quantum_tiled], axis=1)  # (n_rows, 27)

    # Z-score normalize per column (all 27)
    means = features.mean(axis=0)
    stds = features.std(axis=0) + 1e-8
    features = (features - means) / stds
    features = np.clip(features, -4.0, 4.0)

    # Return last bar's features
    return features[-1]


def load_etare_expert(symbol: str = "BTCUSD") -> Optional[ETAREExpert]:
    """
    Try to load the ETARE expert for the given symbol.
    Looks in ETARE_QuantumFusion/models/btcusd_etare_expert.json.
    Returns None if not found or load fails.
    """
    script_dir = Path(__file__).parent.absolute()
    json_path = script_dir / "ETARE_QuantumFusion" / "models" / f"{symbol.lower()}_etare_expert.json"

    if not json_path.exists():
        log.info(f"No ETARE expert found for {symbol} at {json_path}")
        return None

    expert = ETAREExpert(str(json_path))
    if expert.loaded:
        expert.symbol = symbol.upper()
        return expert
    return None
