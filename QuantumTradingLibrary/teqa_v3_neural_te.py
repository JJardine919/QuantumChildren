"""
TEQA v3.0 -- Neural-Transposable Element Integration
=====================================================
Extends TEQA v2.0 (25 qubits, 25 TE families) with:
  - 8 new neural-specific TE families (33 qubits total)
  - Neural mosaic population (L1 retrotransposition)
  - Stress-responsive TE activation (McClintock's Genomic Shock)
  - Cross-instrument TE invasion (horizontal gene transfer)
  - TE domestication learning (STDP-inspired)

Designed to plug into the existing Jardine's Gate pipeline.

Authors: DooDoo + Claude
Date:    2026-02-07
Version: TEQA-3.0-NEURAL-TE
"""

import json
import math
import os
import random
import time
import logging
import sqlite3
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional quantum imports
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

log = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

VERSION = "TEQA-3.0-NEURAL-TE"

# Default neuron population size
DEFAULT_N_NEURONS = 7

# Default quantum shots
DEFAULT_SHOTS = 8192

# Neural consensus threshold (Gate 7)
NEURAL_CONSENSUS_THRESHOLD = 0.70

# Genomic shock thresholds
SHOCK_LOW = 0.8       # Below: calm market
SHOCK_NORMAL = 1.2    # Normal range
SHOCK_HIGH = 2.0      # Genomic shock
SHOCK_EXTREME = 3.0   # TRIM28 emergency suppression

# Domestication thresholds
DOMESTICATION_MIN_TRADES = 20
DOMESTICATION_MIN_WR = 0.70        # Promote to domesticated at 70%+ (posterior WR)
DOMESTICATION_DE_MIN_WR = 0.60     # De-domesticate only below 60% (hysteresis, posterior WR)
DOMESTICATION_EXPIRY_DAYS = 30
DOMESTICATION_MIN_PROFIT_FACTOR = 1.5  # avg_win / avg_loss must exceed this to domesticate
DOMESTICATION_PRIOR_ALPHA = 10     # Beta prior: starts at 50% expectation
DOMESTICATION_PRIOR_BETA = 10      # Beta(10,10) → needs real evidence to move past 50%

# Speciation correlation thresholds
SPECIATION_SAME_SPECIES = 0.6    # Gene flow open
SPECIATION_HYBRID_ZONE = 0.3     # Conflicting signals
# Below 0.3 = reproductive isolation (independent signals)


# ============================================================
# TE FAMILY DEFINITIONS
# ============================================================

class TEClass(Enum):
    RETROTRANSPOSON = "Class_I"
    DNA_TRANSPOSON = "Class_II"
    NEURAL = "Neural"


@dataclass
class TEFamily:
    """Definition of a transposable element family."""
    name: str
    qubit_index: int
    te_class: TEClass
    description: str
    # Market signal mapping
    signal_source: str      # What market data drives this TE
    activation_fn: str      # How to compute activation from signal
    # Neural properties (new in v3.0)
    neural_target: bool = False    # Can L1 insert into this TE?
    stress_responsive: bool = False  # Activates under genomic shock?
    can_domesticate: bool = True     # Can this TE be domesticated?


# Original 25 TE families (v2.0)
ORIGINAL_TE_FAMILIES = [
    TEFamily("BEL_Pao",            0,  TEClass.RETROTRANSPOSON, "LTR retrotransposon", "momentum", "sigmoid"),
    TEFamily("DIRS1",              1,  TEClass.RETROTRANSPOSON, "Tyrosine recombinase", "trend_strength", "sigmoid"),
    TEFamily("Ty1_copia",          2,  TEClass.RETROTRANSPOSON, "LTR Ty1/copia", "rsi", "threshold"),
    TEFamily("Ty3_gypsy",          3,  TEClass.RETROTRANSPOSON, "LTR Ty3/gypsy", "macd", "sigmoid"),
    TEFamily("Ty5",                4,  TEClass.RETROTRANSPOSON, "LTR Ty5 (heterochromatin)", "bollinger_position", "linear"),
    TEFamily("Alu",                5,  TEClass.RETROTRANSPOSON, "SINE Alu element", "short_volatility", "threshold"),
    TEFamily("LINE",               6,  TEClass.RETROTRANSPOSON, "LINE-1 autonomous", "price_change", "sigmoid"),
    TEFamily("Penelope",           7,  TEClass.RETROTRANSPOSON, "Penelope-like element", "trend_duration", "linear"),
    TEFamily("RTE",                8,  TEClass.RETROTRANSPOSON, "Non-LTR RTE", "mean_reversion", "sigmoid"),
    TEFamily("SINE",               9,  TEClass.RETROTRANSPOSON, "Short interspersed element", "tick_volume", "threshold"),
    TEFamily("VIPER_Ngaro",        10, TEClass.RETROTRANSPOSON, "VIPER/Ngaro element", "atr_ratio", "linear"),
    TEFamily("CACTA",              11, TEClass.DNA_TRANSPOSON,  "En/Spm superfamily", "ema_crossover", "threshold"),
    TEFamily("Crypton",            12, TEClass.DNA_TRANSPOSON,  "Tyrosine recombinase", "compression_ratio", "sigmoid"),
    TEFamily("Helitron",           13, TEClass.DNA_TRANSPOSON,  "Rolling-circle transposon", "volume_profile", "sigmoid"),
    TEFamily("hobo",               14, TEClass.DNA_TRANSPOSON,  "hAT superfamily", "candle_pattern", "threshold"),
    TEFamily("I_element",          15, TEClass.DNA_TRANSPOSON,  "I-element (Drosophila)", "support_resistance", "linear"),
    TEFamily("Mariner_Tc1",        16, TEClass.DNA_TRANSPOSON,  "Tc1/mariner superfamily", "fractal_dim", "sigmoid"),
    TEFamily("Mavericks_Polinton", 17, TEClass.DNA_TRANSPOSON,  "Self-synthesizing", "order_flow", "threshold"),
    TEFamily("Mutator",            18, TEClass.DNA_TRANSPOSON,  "Mutator/MuDR", "mutation_rate", "linear"),
    TEFamily("P_element",          19, TEClass.DNA_TRANSPOSON,  "P-element (hybrid dysgenesis)", "spread_analysis", "sigmoid"),
    TEFamily("PIF_Harbinger",      20, TEClass.DNA_TRANSPOSON,  "PIF/Harbinger", "microstructure", "threshold"),
    TEFamily("piggyBac",           21, TEClass.DNA_TRANSPOSON,  "piggyBac (TTAA target)", "gap_analysis", "linear"),
    TEFamily("pogo",               22, TEClass.DNA_TRANSPOSON,  "Tc1/pogo", "session_overlap", "threshold"),
    TEFamily("Rag_like",           23, TEClass.DNA_TRANSPOSON,  "RAG-like (V(D)J origin)", "diversity_index", "sigmoid"),
    TEFamily("Transib",            24, TEClass.DNA_TRANSPOSON,  "Transib superfamily", "autocorrelation", "linear"),
]

# New neural TE families (v3.0)
NEURAL_TE_FAMILIES = [
    TEFamily(
        "L1_Neuronal", 25, TEClass.NEURAL,
        "L1 retrotransposon brain-specific -- hippocampal memory formation",
        "pattern_repetition", "sigmoid",
        neural_target=False, stress_responsive=True, can_domesticate=True
    ),
    TEFamily(
        "L1_Somatic", 26, TEClass.NEURAL,
        "L1 somatic mosaicism driver -- neural diversity index",
        "multi_tf_variance", "linear",
        neural_target=False, stress_responsive=True, can_domesticate=False
    ),
    TEFamily(
        "HERV_Synapse", 27, TEClass.NEURAL,
        "HERV syncytin -- cross-symbol correlation",
        "cross_correlation", "sigmoid",
        neural_target=True, stress_responsive=False, can_domesticate=True
    ),
    TEFamily(
        "SVA_Regulatory", 28, TEClass.NEURAL,
        "SVA composite element -- regime change detection",
        "compression_breakout", "threshold",
        neural_target=True, stress_responsive=True, can_domesticate=True
    ),
    TEFamily(
        "Alu_Exonization", 29, TEClass.NEURAL,
        "Alu exonization -- feature creation from noise",
        "noise_pattern", "sigmoid",
        neural_target=True, stress_responsive=False, can_domesticate=True
    ),
    TEFamily(
        "TRIM28_Silencer", 30, TEClass.NEURAL,
        "TRIM28/KAP1 -- TE repressor / risk management",
        "drawdown", "inverse_sigmoid",
        neural_target=False, stress_responsive=False, can_domesticate=False
    ),
    TEFamily(
        "piwiRNA_Neural", 31, TEClass.NEURAL,
        "Neural piRNA pathway -- TE quality control",
        "signal_noise_ratio", "threshold",
        neural_target=False, stress_responsive=False, can_domesticate=False
    ),
    TEFamily(
        "Arc_Capsid", 32, TEClass.NEURAL,
        "Arc protein (Ty3/gypsy-derived) -- inter-neuron signal transfer",
        "successful_pattern_echo", "sigmoid",
        neural_target=True, stress_responsive=True, can_domesticate=True
    ),
]

ALL_TE_FAMILIES = ORIGINAL_TE_FAMILIES + NEURAL_TE_FAMILIES
N_QUBITS = len(ALL_TE_FAMILIES)  # 33


# ============================================================
# TE ACTIVATION ENGINE
# ============================================================

class TEActivationEngine:
    """
    Computes TE activation strengths from market data.
    Each TE family has a specific market signal it responds to.
    """

    def __init__(self):
        self.activation_cache: Dict[str, float] = {}

    def compute_all_activations(
        self,
        bars: np.ndarray,
        additional_signals: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Compute activation for all 33 TE families.

        Args:
            bars: numpy array with columns [open, high, low, close, volume]
                  (at least 50 rows)
            additional_signals: optional dict of pre-computed signals
                  e.g. {"cross_correlation": 0.85, "drawdown": 0.02}

        Returns:
            List of dicts: [{te, strength, direction, details}, ...]
        """
        if len(bars) < 50:
            log.warning("TEActivation: need >= 50 bars, got %d", len(bars))
            return [{"te": f.name, "strength": 0.0, "direction": 0, "details": {}}
                    for f in ALL_TE_FAMILIES]

        signals = additional_signals or {}
        results = []

        for te in ALL_TE_FAMILIES:
            strength, direction = self._compute_single(te, bars, signals)
            results.append({
                "te": te.name,
                "strength": float(np.clip(strength, 0.0, 1.0)),
                "direction": int(direction),
                "details": {
                    "strength": float(strength),
                    "direction": int(direction),
                    "class": te.te_class.value,
                    "neural": te.neural_target,
                    "stress_responsive": te.stress_responsive,
                }
            })

        return results

    def _compute_single(
        self, te: TEFamily, bars: np.ndarray, signals: Dict
    ) -> Tuple[float, int]:
        """Compute activation for a single TE family."""
        close = bars[:, 3] if bars.ndim == 2 else bars
        high = bars[:, 1] if bars.ndim == 2 else bars
        low = bars[:, 2] if bars.ndim == 2 else bars
        volume = bars[:, 4] if bars.ndim == 2 and bars.shape[1] > 4 else np.ones(len(close))

        src = te.signal_source

        # -- Original TE signals (v2.0 compatible) --
        if src == "momentum":
            val = (close[-1] - close[-10]) / (close[-10] + 1e-10)
            return self._sigmoid(val * 20), 1 if val > 0 else -1

        elif src == "trend_strength":
            ema8 = self._ema(close, 8)
            ema21 = self._ema(close, 21)
            val = (ema8 - ema21) / (ema21 + 1e-10)
            return self._sigmoid(val * 50), 1 if val > 0 else -1

        elif src == "rsi":
            rsi = self._rsi(close, 14)
            if rsi > 70:
                return (rsi - 70) / 30, -1
            elif rsi < 30:
                return (30 - rsi) / 30, 1
            return 0.0, 0

        elif src == "macd":
            macd, sig = self._macd(close)
            val = macd - sig
            return self._sigmoid(val * 100), 1 if val > 0 else -1

        elif src == "bollinger_position":
            bb_pos = self._bollinger_position(close, 20)
            if bb_pos > 1:
                return min(1.0, (bb_pos - 1) * 2), -1
            elif bb_pos < -1:
                return min(1.0, (-bb_pos - 1) * 2), 1
            return abs(bb_pos) * 0.3, 0

        elif src == "short_volatility":
            returns = np.diff(close) / close[:-1]
            vol5 = np.std(returns[-5:]) if len(returns) >= 5 else 0
            vol20 = np.std(returns[-20:]) if len(returns) >= 20 else vol5
            if vol20 == 0:
                return 0.0, 0
            ratio = vol5 / vol20
            return 0.0 if ratio < 0.5 else min(1.0, (ratio - 0.5) * 2), 0

        elif src == "price_change":
            ret = (close[-1] - close[-2]) / (close[-2] + 1e-10)
            return self._sigmoid(abs(ret) * 200), 1 if ret > 0 else -1

        elif src == "trend_duration":
            count = 0
            for i in range(len(close) - 1, 0, -1):
                if close[i] > close[i - 1]:
                    count += 1
                else:
                    break
            return min(1.0, count / 10), 1 if count > 0 else -1

        elif src == "mean_reversion":
            sma20 = np.mean(close[-20:])
            dev = (close[-1] - sma20) / (sma20 + 1e-10)
            return self._sigmoid(abs(dev) * 30), -1 if dev > 0 else 1

        elif src == "tick_volume":
            vol_ratio = volume[-1] / (np.mean(volume[-20:]) + 1e-10)
            return 0.0 if vol_ratio < 1.5 else min(1.0, (vol_ratio - 1.5) * 0.5), 0

        elif src == "atr_ratio":
            atr = self._atr(high, low, close, 14)
            atr_avg = np.mean([self._atr(high[:-i] if i > 0 else high,
                                          low[:-i] if i > 0 else low,
                                          close[:-i] if i > 0 else close, 14)
                               for i in range(5)])
            ratio = atr / (atr_avg + 1e-10)
            return min(1.0, max(0.0, (ratio - 0.5) * 2)), 0

        elif src == "ema_crossover":
            ema8 = self._ema(close, 8)
            ema21 = self._ema(close, 21)
            crossed = ema8 > ema21
            return 1.0 if crossed else 0.0, 1 if crossed else -1

        elif src == "compression_ratio":
            import zlib
            data_bytes = close[-50:].astype(np.float32).tobytes()
            compressed = zlib.compress(data_bytes, level=9)
            ratio = len(data_bytes) / len(compressed)
            return self._sigmoid(ratio - 1.5), 1 if ratio > 2 else -1

        elif src == "volume_profile":
            vol_norm = volume[-20:] / (np.max(volume[-20:]) + 1e-10)
            profile_skew = np.mean(vol_norm[-5:]) - np.mean(vol_norm[:5])
            return self._sigmoid(profile_skew * 5), 1 if profile_skew > 0 else -1

        elif src == "candle_pattern":
            body = abs(close[-1] - bars[-1, 0]) if bars.ndim == 2 else 0
            total = high[-1] - low[-1] if high[-1] != low[-1] else 1e-10
            ratio = body / total
            return ratio, 1 if close[-1] > bars[-1, 0] else -1 if bars.ndim == 2 else 0

        elif src == "support_resistance":
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])
            pos = (close[-1] - recent_low) / (recent_high - recent_low + 1e-10)
            return pos, -1 if pos > 0.8 else (1 if pos < 0.2 else 0)

        elif src == "fractal_dim":
            c30 = close[-30:]
            returns = np.diff(c30) / (c30[:-1] + 1e-10)
            if len(returns) < 10:
                return 0.5, 0
            abs_ret = np.abs(returns)
            fd_approx = 1.0 + np.log(np.sum(abs_ret) + 1e-10) / np.log(len(abs_ret))
            fd_norm = min(1.0, max(0.0, (fd_approx - 1.0) / 1.0))
            return fd_norm, 0

        elif src == "order_flow":
            c10 = close[-10:]
            v10 = volume[-10:]
            price_changes = np.diff(c10)
            up_mask = price_changes > 0
            dn_mask = price_changes < 0
            up_vol = float(np.sum(v10[1:][up_mask]))
            dn_vol = float(np.sum(v10[1:][dn_mask]))
            total_vol = up_vol + dn_vol + 1e-10
            imbalance = (up_vol - dn_vol) / total_vol
            return abs(imbalance) if abs(imbalance) > 0.2 else 0.0, 1 if imbalance > 0 else -1

        elif src == "mutation_rate":
            returns = np.diff(close[-20:]) / (close[-20:-1] + 1e-10)
            sign_changes = np.sum(np.diff(np.sign(returns)) != 0)
            rate = sign_changes / len(returns)
            return rate, 0

        elif src == "spread_analysis":
            ranges = high[-10:] - low[-10:]
            spread_expansion = ranges[-1] / (np.mean(ranges) + 1e-10)
            return min(1.0, max(0.0, spread_expansion - 1)), 1 if close[-1] > bars[-1, 0] else -1 if bars.ndim == 2 else 0

        elif src == "microstructure":
            returns = np.diff(close[-20:]) / (close[-20:-1] + 1e-10)
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0
            return abs(autocorr) if not np.isnan(autocorr) else 0.0, 1 if autocorr > 0 else -1

        elif src == "gap_analysis":
            if bars.ndim == 2 and len(bars) > 1:
                gap = abs(bars[-1, 0] - close[-2]) / (close[-2] + 1e-10)
                return min(1.0, gap * 100), 1 if bars[-1, 0] > close[-2] else -1
            return 0.0, 0

        elif src == "session_overlap":
            hour = datetime.now().hour
            is_overlap = 13 <= hour <= 17  # London/NY overlap
            return 1.0 if is_overlap else 0.0, 0

        elif src == "diversity_index":
            returns = np.diff(close[-30:]) / (close[-30:-1] + 1e-10)
            unique_signs = len(set(np.sign(returns).astype(int)))
            return unique_signs / 3.0, 0

        elif src == "autocorrelation":
            returns = np.diff(close[-30:]) / (close[-30:-1] + 1e-10)
            if len(returns) < 5:
                return 0.0, 0
            ac = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            return abs(ac) if not np.isnan(ac) else 0.0, 1 if ac > 0.1 else (-1 if ac < -0.1 else 0)

        # -- NEW NEURAL TE SIGNALS (v3.0) --

        elif src == "pattern_repetition":
            # L1_Neuronal: how much the recent pattern repeats historical
            pattern = close[-5:] / close[-5]
            best_corr = 0.0
            for offset in range(10, min(45, len(close) - 5)):
                historical = close[-(offset + 5):-offset] / close[-(offset + 5)]
                if len(historical) == len(pattern):
                    corr = np.corrcoef(pattern, historical)[0, 1]
                    if not np.isnan(corr) and abs(corr) > abs(best_corr):
                        best_corr = corr
            return abs(best_corr), 1 if best_corr > 0 else -1

        elif src == "multi_tf_variance":
            # L1_Somatic: variance between signals at different lookbacks
            signals_at_tfs = []
            for lb in [5, 10, 20, 50]:
                if len(close) >= lb + 1:
                    ret = (close[-1] - close[-lb]) / (close[-lb] + 1e-10)
                    signals_at_tfs.append(ret)
            if len(signals_at_tfs) < 2:
                return 0.0, 0
            directions = [1 if s > 0 else -1 for s in signals_at_tfs]
            agreement = abs(sum(directions)) / len(directions)
            diversity = 1.0 - agreement  # Higher = more mosaic diversity
            return diversity, 0

        elif src == "cross_correlation":
            # HERV_Synapse: uses pre-computed cross-correlation signal
            val = signals.get("cross_correlation", 0.5)
            return abs(val), 1 if val > 0 else -1

        elif src == "compression_breakout":
            # SVA_Regulatory: breakout from compressed state
            import zlib
            data_recent = close[-10:].astype(np.float32).tobytes()
            data_prior = close[-30:-10].astype(np.float32).tobytes()
            cr_recent = len(data_recent) / len(zlib.compress(data_recent, 9))
            cr_prior = len(data_prior) / len(zlib.compress(data_prior, 9))
            breakout = cr_recent - cr_prior
            return self._sigmoid(breakout * 3), 1 if breakout > 0 else -1

        elif src == "noise_pattern":
            # Alu_Exonization: finding structure in noise
            returns = np.diff(close[-20:]) / (close[-20:-1] + 1e-10)
            # Check if "noise" has hidden autocorrelation at lag 3
            if len(returns) > 6:
                lag3_corr = np.corrcoef(returns[:-3], returns[3:])[0, 1]
                if not np.isnan(lag3_corr) and abs(lag3_corr) > 0.3:
                    return abs(lag3_corr), 1 if lag3_corr > 0 else -1
            return 0.0, 0

        elif src == "drawdown":
            # TRIM28_Silencer: activated by drawdown (inverse -- suppresses TEs)
            dd = signals.get("drawdown", 0.0)
            # Higher drawdown = stronger TRIM28 = more TE suppression
            return min(1.0, dd * 10), 0  # Direction neutral (suppressor)

        elif src == "signal_noise_ratio":
            # piwiRNA_Neural: quality control
            returns = np.diff(close[-30:]) / (close[-30:-1] + 1e-10)
            signal_power = abs(np.mean(returns))
            noise_power = np.std(returns) + 1e-10
            snr = signal_power / noise_power
            return min(1.0, snr * 5) if snr > 0.2 else 0.0, 0

        elif src == "successful_pattern_echo":
            # Arc_Capsid: echo of successful patterns from other neurons
            echo = signals.get("arc_echo", 0.0)
            return abs(echo), 1 if echo > 0 else (-1 if echo < 0 else 0)

        # Fallback
        return 0.0, 0

    # -- Helper functions --

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-50, min(50, x))))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return float(np.mean(data))
        mult = 2.0 / (period + 1)
        ema = float(np.mean(data[:period]))
        for price in data[period:]:
            ema = (float(price) - ema) * mult + ema
        return ema

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _macd(close: np.ndarray) -> Tuple[float, float]:
        if len(close) < 26:
            return 0.0, 0.0
        ema12 = TEActivationEngine._ema(close, 12)
        ema26 = TEActivationEngine._ema(close, 26)
        macd_val = ema12 - ema26
        return macd_val, macd_val * 0.8  # Simplified signal

    @staticmethod
    def _bollinger_position(close: np.ndarray, period: int = 20) -> float:
        if len(close) < period:
            return 0.0
        sma = np.mean(close[-period:])
        std = np.std(close[-period:])
        if std == 0:
            return 0.0
        return (close[-1] - sma) / std

    @staticmethod
    def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        if len(high) < period:
            return float(np.mean(high - low))
        tr_list = []
        for i in range(-period, 0):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]) if i > -len(close) else 0,
                abs(low[i] - close[i - 1]) if i > -len(close) else 0,
            )
            tr_list.append(tr)
        return float(np.mean(tr_list))


# ============================================================
# GENOMIC SHOCK DETECTOR (McClintock)
# ============================================================

class GenomicShockDetector:
    """
    Implements McClintock's Genomic Shock Hypothesis.
    Market stress causes TE activation thresholds to change.
    """

    def __init__(self):
        self.shock_history: List[float] = []

    def compute_shock_level(
        self,
        bars: np.ndarray,
        drawdown: float = 0.0
    ) -> Tuple[float, str]:
        """
        Compute genomic shock level from market data.

        Returns:
            (shock_score, shock_label)
            shock_score: 0.0 = dead calm, 1.0 = normal, 2.0+ = shock
            shock_label: "CALM", "NORMAL", "SHOCK", "EXTREME"
        """
        close = bars[:, 3] if bars.ndim == 2 else bars
        high = bars[:, 1] if bars.ndim == 2 else bars
        low = bars[:, 2] if bars.ndim == 2 else bars
        volume = bars[:, 4] if bars.ndim == 2 and bars.shape[1] > 4 else np.ones(len(close))

        # ATR expansion
        atr_current = self._atr_simple(high[-5:], low[-5:], close[-5:])
        atr_baseline = self._atr_simple(high[-25:-5], low[-25:-5], close[-25:-5])
        atr_ratio = atr_current / (atr_baseline + 1e-10)

        # Volume shock
        vol_current = np.mean(volume[-5:])
        vol_baseline = np.mean(volume[-25:-5])
        vol_ratio = vol_current / (vol_baseline + 1e-10)

        # Drawdown acceleration
        dd_component = min(1.0, drawdown * 10)  # 10% DD = max

        # Combined shock score
        shock = atr_ratio * 0.5 + vol_ratio * 0.3 + dd_component * 0.2

        self.shock_history.append(shock)
        if len(self.shock_history) > 100:
            self.shock_history.pop(0)

        # Classify
        if shock < SHOCK_LOW:
            label = "CALM"
        elif shock < SHOCK_NORMAL:
            label = "NORMAL"
        elif shock < SHOCK_HIGH:
            label = "ELEVATED"
        elif shock < SHOCK_EXTREME:
            label = "SHOCK"
        else:
            label = "EXTREME"

        return float(shock), label

    def adjust_te_thresholds(
        self,
        shock_score: float,
        shock_label: str,
        activations: List[Dict]
    ) -> List[Dict]:
        """
        Modify TE activation thresholds based on shock level.

        CALM:     Normal thresholds
        ELEVATED: Lower thresholds by 20% (more TEs activate)
        SHOCK:    All Class I retrotransposons auto-activate
        EXTREME:  TRIM28 emergency -- suppress all TEs
        """
        adjusted = []
        for act in activations:
            a = dict(act)
            te = next((f for f in ALL_TE_FAMILIES if f.name == a["te"]), None)
            if te is None:
                adjusted.append(a)
                continue

            if shock_label == "EXTREME":
                # TRIM28 suppression -- all TEs silenced except TRIM28 itself
                if te.name != "TRIM28_Silencer":
                    a["strength"] *= 0.1
                    a["details"]["shock_suppressed"] = True
                else:
                    a["strength"] = 1.0  # TRIM28 fully active

            elif shock_label == "SHOCK":
                if te.te_class == TEClass.RETROTRANSPOSON or te.stress_responsive:
                    # All retrotransposons and stress-responsive TEs auto-activate
                    a["strength"] = max(a["strength"], 0.8)
                    a["details"]["shock_activated"] = True

            elif shock_label == "ELEVATED":
                if te.stress_responsive:
                    # Lower activation threshold
                    a["strength"] = min(1.0, a["strength"] * 1.3)
                    a["details"]["elevated_boost"] = True

            adjusted.append(a)

        return adjusted

    @staticmethod
    def _atr_simple(high, low, close):
        if len(high) < 2:
            return float(high[0] - low[0]) if len(high) > 0 else 0.0
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ))
        return float(np.mean(tr[1:]))


# ============================================================
# NEURAL MOSAIC ENGINE (L1 Retrotransposition)
# ============================================================

@dataclass
class L1Insertion:
    """Represents a single L1 retrotransposition event in a neuron."""
    target_qubit: int
    effect: str           # "enhance", "disrupt", "invert", "rewire"
    rewire_target: int    # Only used if effect == "rewire"
    magnitude: float      # Strength of the modification


@dataclass
class NeuralGenome:
    """A neuron's unique TE genome, modified by L1 insertions."""
    neuron_id: int
    insertions: List[L1Insertion]
    activation_modifiers: Dict[int, float]  # qubit_index -> modifier
    vote: int = 0          # -1, 0, or 1 after quantum measurement
    confidence: float = 0.0


class NeuralMosaicEngine:
    """
    Creates a population of neurons, each with L1-modified TE genomes.
    Mirrors how L1 retrotransposition in the hippocampus creates neurons
    with different response profiles (Muotri et al., 2005).
    """

    def __init__(self, n_neurons: int = DEFAULT_N_NEURONS, seed: int = None):
        self.n_neurons = n_neurons
        self.rng = random.Random(seed)
        self.neurons: List[NeuralGenome] = []
        self._create_mosaic()

    def _create_mosaic(self):
        """Generate initial neural population with L1 insertions."""
        self.neurons = []
        for i in range(self.n_neurons):
            n_jumps = self.rng.randint(2, 5)

            # L1 can only insert into neural_target TEs and some random others
            eligible_targets = [
                te.qubit_index for te in ALL_TE_FAMILIES
                if te.neural_target or self.rng.random() < 0.15
            ]
            targets = self.rng.sample(
                eligible_targets, min(n_jumps, len(eligible_targets))
            )

            insertions = []
            modifiers = {}
            for target in targets:
                effect = self.rng.choice(["enhance", "disrupt", "invert", "rewire"])
                magnitude = self.rng.uniform(0.3, 0.8)

                rewire_target = target
                if effect == "rewire":
                    # Pick a random partner qubit for new entanglement
                    candidates = [q for q in range(N_QUBITS) if q != target]
                    rewire_target = self.rng.choice(candidates)

                insertions.append(L1Insertion(
                    target_qubit=target,
                    effect=effect,
                    rewire_target=rewire_target,
                    magnitude=magnitude
                ))

                # Compute modifier for this qubit
                if effect == "enhance":
                    modifiers[target] = 1.0 + magnitude
                elif effect == "disrupt":
                    modifiers[target] = 1.0 - magnitude * 0.7
                elif effect == "invert":
                    modifiers[target] = -1.0
                elif effect == "rewire":
                    modifiers[target] = 1.0  # Normal strength, but new connection

            self.neurons.append(NeuralGenome(
                neuron_id=i,
                insertions=insertions,
                activation_modifiers=modifiers
            ))

    def reseed_mosaic(self):
        """Create new L1 insertions (like adult neurogenesis)."""
        self._create_mosaic()

    def get_neuron_activations(
        self,
        master_activations: List[Dict],
        neuron: NeuralGenome
    ) -> List[Dict]:
        """
        Modify master TE activations for a specific neuron's genome.
        """
        modified = []
        for act in master_activations:
            a = dict(act)
            te = next((f for f in ALL_TE_FAMILIES if f.name == a["te"]), None)
            if te is None:
                modified.append(a)
                continue

            qi = te.qubit_index
            if qi in neuron.activation_modifiers:
                mod = neuron.activation_modifiers[qi]
                if mod < 0:
                    # Inversion
                    a["direction"] = -a["direction"]
                    a["strength"] = abs(a["strength"])
                else:
                    a["strength"] = float(np.clip(a["strength"] * mod, 0.0, 1.0))

            modified.append(a)
        return modified

    def compute_consensus(self) -> Tuple[int, float, Dict]:
        """
        Compute neural population consensus vote.

        Returns:
            (direction, consensus_score, vote_counts)
        """
        long_count = sum(1 for n in self.neurons if n.vote > 0)
        short_count = sum(1 for n in self.neurons if n.vote < 0)
        neutral_count = sum(1 for n in self.neurons if n.vote == 0)

        total = len(self.neurons)
        if total == 0:
            return 0, 0.0, {"long": 0, "short": 0, "neutral": 0}

        # Direction from majority
        if long_count > short_count:
            direction = 1
            consensus = long_count / total
        elif short_count > long_count:
            direction = -1
            consensus = short_count / total
        else:
            direction = 0
            consensus = neutral_count / total

        # Weight by individual neuron confidence
        total_confidence = sum(n.confidence for n in self.neurons)
        if total_confidence > 0:
            weighted_dir = sum(
                n.vote * n.confidence for n in self.neurons
            ) / total_confidence
            consensus = abs(weighted_dir)
            direction = 1 if weighted_dir > 0.1 else (-1 if weighted_dir < -0.1 else 0)

        return direction, float(consensus), {
            "long": long_count,
            "short": short_count,
            "neutral": neutral_count
        }


# ============================================================
# QUANTUM CIRCUIT ENGINE (Split Architecture)
# ============================================================
#
# 33 qubits in a single statevector simulation requires 128GB RAM.
# Solution: split into two biologically accurate compartments:
#   Circuit A: 25 qubits (original TE genome -- v2.0 compatible)
#   Circuit B: 8 qubits  (neural TE compartment -- v3.0 new)
#
# Results are merged through a "synaptic fusion" step that
# combines the two circuit outputs. This mirrors biology:
# somatic TE insertions in neurons operate in the same cell
# but the neural-specific TEs are in a distinct regulatory
# compartment from the ancestral genome TEs.

N_QUBITS_GENOME = 25   # Original TE genome
N_QUBITS_NEURAL = 8    # Neural TE compartment


class TEQAQuantumEngine:
    """
    Split-architecture quantum engine for TEQA v3.0.
    Runs two circuits (genome + neural) and fuses results.
    """

    def __init__(self, shots: int = DEFAULT_SHOTS):
        self.shots = shots
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
        else:
            log.warning("Qiskit not available -- using classical fallback")
            self.simulator = None

    def build_genome_circuit(
        self,
        activations: List[Dict],
        neuron: Optional[NeuralGenome] = None,
        shock_level: float = 1.0
    ) -> Optional['QuantumCircuit']:
        """
        Build 25-qubit circuit for the original TE genome.
        This matches TEQA v2.0 architecture exactly.
        """
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(N_QUBITS_GENOME, N_QUBITS_GENOME)

        # Layer 1: RY rotations from TE activations (qubits 0-24)
        for act in activations:
            te = next((f for f in ALL_TE_FAMILIES if f.name == act["te"]), None)
            if te is None or te.qubit_index >= N_QUBITS_GENOME:
                continue
            qi = te.qubit_index
            strength = act["strength"]
            direction = act["direction"]

            angle = strength * math.pi * (1 if direction >= 0 else -1)

            if neuron and qi in neuron.activation_modifiers:
                mod = neuron.activation_modifiers[qi]
                if mod < 0:
                    angle = -angle
                else:
                    angle *= mod

            if shock_level > SHOCK_NORMAL:
                angle *= min(1.5, shock_level / SHOCK_NORMAL)

            qc.ry(angle, qi)

        # Layer 2: Entanglement
        # Class I internal chain (qubits 0-10)
        for i in range(10):
            qc.cx(i, i + 1)
        # Class II internal chain (qubits 11-24)
        for i in range(11, 24):
            qc.cx(i, i + 1)
        # Cross-class bridges
        qc.cx(6, 16)   # LINE <-> Mariner
        qc.cx(3, 13)   # Ty3/gypsy <-> Helitron

        # Layer 3: Neuron-specific rewiring (only for genome qubits)
        if neuron:
            for ins in neuron.insertions:
                if ins.effect == "rewire" and ins.target_qubit < N_QUBITS_GENOME and ins.rewire_target < N_QUBITS_GENOME:
                    qc.cx(ins.target_qubit, ins.rewire_target)

        # Layer 4: Second rotation
        for act in activations:
            te = next((f for f in ALL_TE_FAMILIES if f.name == act["te"]), None)
            if te is None or te.qubit_index >= N_QUBITS_GENOME:
                continue
            qi = te.qubit_index
            angle2 = act["strength"] * math.pi * 0.3 * (1 if act["direction"] >= 0 else -1)
            qc.ry(angle2, qi)

        qc.measure(range(N_QUBITS_GENOME), range(N_QUBITS_GENOME))
        return qc

    def build_neural_circuit(
        self,
        activations: List[Dict],
        neuron: Optional[NeuralGenome] = None,
        shock_level: float = 1.0,
        genome_signal: float = 0.0
    ) -> Optional['QuantumCircuit']:
        """
        Build 8-qubit circuit for the neural TE compartment.
        The genome_signal parameter feeds information from the
        genome circuit into the neural circuit (like synaptic input).
        """
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(N_QUBITS_NEURAL, N_QUBITS_NEURAL)

        # Map neural TEs to local qubit indices 0-7
        for act in activations:
            te = next((f for f in ALL_TE_FAMILIES if f.name == act["te"]), None)
            if te is None or te.qubit_index < N_QUBITS_GENOME:
                continue

            local_qi = te.qubit_index - N_QUBITS_GENOME  # 25->0, 26->1, etc.
            strength = act["strength"]
            direction = act["direction"]

            angle = strength * math.pi * (1 if direction >= 0 else -1)

            if neuron and te.qubit_index in neuron.activation_modifiers:
                mod = neuron.activation_modifiers[te.qubit_index]
                if mod < 0:
                    angle = -angle
                else:
                    angle *= mod

            if shock_level > SHOCK_NORMAL:
                angle *= min(1.5, shock_level / SHOCK_NORMAL)

            qc.ry(angle, local_qi)

        # Genome signal injection: rotate qubit 0 (L1_Neuronal) by genome result
        genome_angle = genome_signal * math.pi * 0.5
        qc.ry(genome_angle, 0)

        # Neural entanglement
        qc.cx(0, 1)    # L1_Neuronal <-> L1_Somatic
        qc.cx(2, 0)    # HERV_Synapse <-> L1_Neuronal
        qc.cx(3, 4)    # SVA_Regulatory <-> Alu_Exonization

        # TRIM28 (qubit 5) suppresses others via CZ
        for nq in range(N_QUBITS_NEURAL):
            if nq != 5:
                qc.cz(5, nq)

        # piwiRNA (qubit 6) targets L1 (qubits 0, 1)
        qc.cz(6, 0)
        qc.cz(6, 1)

        # Arc_Capsid (qubit 7) connects to L1_Neuronal and HERV
        qc.cx(7, 0)
        qc.cx(7, 2)

        # Neuron-specific rewiring (neural compartment only)
        if neuron:
            for ins in neuron.insertions:
                src = ins.target_qubit - N_QUBITS_GENOME
                tgt = ins.rewire_target - N_QUBITS_GENOME
                if ins.effect == "rewire" and 0 <= src < N_QUBITS_NEURAL and 0 <= tgt < N_QUBITS_NEURAL:
                    qc.cx(src, tgt)

        # Second rotation layer
        for act in activations:
            te = next((f for f in ALL_TE_FAMILIES if f.name == act["te"]), None)
            if te is None or te.qubit_index < N_QUBITS_GENOME:
                continue
            local_qi = te.qubit_index - N_QUBITS_GENOME
            angle2 = act["strength"] * math.pi * 0.3 * (1 if act["direction"] >= 0 else -1)
            qc.ry(angle2, local_qi)

        qc.measure(range(N_QUBITS_NEURAL), range(N_QUBITS_NEURAL))
        return qc

    def execute_circuit(self, qc: 'QuantumCircuit', shots: int = None, n_qubits: int = None) -> Dict:
        """Execute circuit and return measurement results."""
        if not QISKIT_AVAILABLE or self.simulator is None:
            return self._classical_fallback(n_qubits or N_QUBITS)

        n_shots = shots or self.shots
        nq = n_qubits or qc.num_qubits
        try:
            job = self.simulator.run(qc, shots=n_shots)
            counts = job.result().get_counts()
            return self._analyze_counts(counts, n_shots, nq)
        except Exception as e:
            log.error("Quantum execution failed: %s", e)
            return self._classical_fallback(nq)

    def fuse_results(self, genome_result: Dict, neural_result: Dict) -> Dict:
        """
        Synaptic fusion: merge genome and neural circuit results.
        Like how neural TE activity modulates the broader genomic signal.
        """
        # Combine votes with weighting: genome 60%, neural 40%
        g_long = genome_result.get("vote_long", 0.5)
        g_short = genome_result.get("vote_short", 0.5)
        n_long = neural_result.get("vote_long", 0.5)
        n_short = neural_result.get("vote_short", 0.5)

        fused_long = g_long * 0.6 + n_long * 0.4
        fused_short = g_short * 0.6 + n_short * 0.4

        # Combine entropy
        g_entropy = genome_result.get("shannon_entropy", 0)
        n_entropy = neural_result.get("shannon_entropy", 0)
        g_max = genome_result.get("max_entropy", 25)
        n_max = neural_result.get("max_entropy", 8)
        fused_entropy = (g_entropy + n_entropy)
        fused_max = g_max + n_max

        return {
            "n_shots": genome_result.get("n_shots", 0) + neural_result.get("n_shots", 0),
            "n_unique_states": genome_result.get("n_unique_states", 0) + neural_result.get("n_unique_states", 0),
            "n_possible_states": 2 ** N_QUBITS,
            "coverage": (genome_result.get("coverage", 0) + neural_result.get("coverage", 0)) / 2,
            "shannon_entropy": float(fused_entropy),
            "max_entropy": float(fused_max),
            "novelty": float(fused_entropy / fused_max) if fused_max > 0 else 0.0,
            "vote_long": float(fused_long),
            "vote_short": float(fused_short),
            "top_state": genome_result.get("top_state", "") + "|" + neural_result.get("top_state", ""),
            "top_count": genome_result.get("top_count", 0),
            "genome_result": genome_result,
            "neural_result": neural_result,
        }

    def _analyze_counts(self, counts: Dict[str, int], total_shots: int, n_qubits: int) -> Dict:
        """Analyze measurement results."""
        n_unique = len(counts)
        n_possible = 2 ** n_qubits

        probs = np.array(list(counts.values())) / total_shots
        shannon = -np.sum(probs * np.log2(probs + 1e-20))

        # Direction voting
        vote_long = 0.0
        vote_short = 0.0
        for bitstring, count in counts.items():
            weight = count / total_shots
            ones = bitstring.count('1')
            zeros = bitstring.count('0')
            if ones > zeros:
                vote_long += weight
            elif zeros > ones:
                vote_short += weight

        sorted_states = sorted(counts.items(), key=lambda x: -x[1])

        return {
            "n_shots": total_shots,
            "n_unique_states": n_unique,
            "n_possible_states": n_possible,
            "coverage": n_unique / n_possible,
            "shannon_entropy": float(shannon),
            "max_entropy": float(n_qubits),
            "novelty": float(shannon / n_qubits) if n_qubits > 0 else 0.0,
            "vote_long": float(vote_long),
            "vote_short": float(vote_short),
            "top_state": sorted_states[0][0] if sorted_states else "",
            "top_count": sorted_states[0][1] if sorted_states else 0,
        }

    def _classical_fallback(self, n_qubits: int = N_QUBITS) -> Dict:
        """Fallback when quantum simulator not available."""
        return {
            "n_shots": 0,
            "n_unique_states": 0,
            "n_possible_states": 2 ** n_qubits,
            "coverage": 0.0,
            "shannon_entropy": 0.0,
            "max_entropy": float(n_qubits),
            "novelty": 0.0,
            "vote_long": 0.5,
            "vote_short": 0.5,
            "top_state": "",
            "top_count": 0,
        }


# ============================================================
# TE DOMESTICATION TRACKER
# ============================================================

class TEDomesticationTracker:
    """
    Tracks which TE activation combinations precede profitable trades.
    Successful patterns get "domesticated" -- they become permanent
    signal enhancers, like how the immune system's RAG recombinase
    was domesticated from a Transib transposon.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "teqa_domestication.db")
        else:
            self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS domesticated_patterns (
                        pattern_hash TEXT PRIMARY KEY,
                        te_combo TEXT,
                        win_count INTEGER DEFAULT 0,
                        loss_count INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0.0,
                        domesticated INTEGER DEFAULT 0,
                        boost_factor REAL DEFAULT 1.0,
                        first_seen TEXT,
                        last_seen TEXT,
                        last_activated TEXT,
                        topology_hash TEXT DEFAULT ''
                    )
                """)
                # Migrate existing DBs: add columns if missing
                for col_sql in [
                    "ALTER TABLE domesticated_patterns ADD COLUMN topology_hash TEXT DEFAULT ''",
                    "ALTER TABLE domesticated_patterns ADD COLUMN avg_win REAL DEFAULT 0.0",
                    "ALTER TABLE domesticated_patterns ADD COLUMN avg_loss REAL DEFAULT 0.0",
                    "ALTER TABLE domesticated_patterns ADD COLUMN profit_factor REAL DEFAULT 0.0",
                    "ALTER TABLE domesticated_patterns ADD COLUMN posterior_wr REAL DEFAULT 0.5",
                    "ALTER TABLE domesticated_patterns ADD COLUMN total_win_pnl REAL DEFAULT 0.0",
                    "ALTER TABLE domesticated_patterns ADD COLUMN total_loss_pnl REAL DEFAULT 0.0",
                ]:
                    try:
                        conn.execute(col_sql)
                    except Exception:
                        pass  # Column already exists
                conn.commit()
        except Exception as e:
            log.warning("Domestication DB init failed: %s", e)

    def _is_domesticated(self, pattern_hash: str, cursor) -> bool:
        """Check if a pattern is currently domesticated (for hysteresis)."""
        cursor.execute(
            "SELECT domesticated FROM domesticated_patterns WHERE pattern_hash=?",
            (pattern_hash,)
        )
        row = cursor.fetchone()
        return bool(row and row[0])

    def record_pattern(self, active_tes: List[str], won: bool, profit: float = 0.0):
        """Record which TEs were active before a trade outcome.

        Uses Bayesian shrinkage (Beta(10,10) prior) to prevent overfitting:
        with 33 TE families and millions of combos, raw win_rate at 20 trades
        is statistically unreliable. The posterior mean starts at 50% and
        requires real evidence to reach the 70% domestication threshold.

        Also requires profit_factor (avg_win / avg_loss) > 1.5 to ensure
        the pattern isn't just winning often with tiny gains and rare big losses.
        """
        if not active_tes:
            return

        combo = "+".join(sorted(active_tes))
        pattern_hash = hashlib.md5(combo.encode()).hexdigest()[:16]
        now = datetime.now().isoformat()

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT win_count, loss_count, total_win_pnl, total_loss_pnl "
                    "FROM domesticated_patterns WHERE pattern_hash=?",
                    (pattern_hash,)
                )
                row = cursor.fetchone()

                if row:
                    win_count = row[0] + (1 if won else 0)
                    loss_count = row[1] + (0 if won else 1)
                    total = win_count + loss_count
                    win_rate = win_count / total if total > 0 else 0

                    # Accumulate PnL for profit factor calculation
                    old_win_pnl = row[2] or 0.0
                    old_loss_pnl = row[3] or 0.0
                    total_win_pnl = old_win_pnl + (profit if won else 0.0)
                    total_loss_pnl = old_loss_pnl + (abs(profit) if not won else 0.0)

                    # Bayesian posterior mean: Beta(alpha + wins, beta + losses)
                    # Prior Beta(10,10) = 50% expectation, requires real evidence to shift
                    posterior_wr = (DOMESTICATION_PRIOR_ALPHA + win_count) / (
                        DOMESTICATION_PRIOR_ALPHA + DOMESTICATION_PRIOR_BETA + total)

                    # Profit factor: avg_win / avg_loss (guards against tiny-win-big-loss patterns)
                    avg_win = total_win_pnl / win_count if win_count > 0 else 0.0
                    avg_loss = total_loss_pnl / loss_count if loss_count > 0 else 0.0
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)

                    # Hysteresis with Bayesian shrinkage + profit factor
                    was_domesticated = self._is_domesticated(pattern_hash, cursor)
                    if was_domesticated:
                        # Already domesticated: revoke if posterior drops below de-domestication
                        # threshold OR profit factor falls below 1.0 (losses exceed wins)
                        domesticated = 0 if (posterior_wr < DOMESTICATION_DE_MIN_WR or profit_factor < 1.0) else 1
                    else:
                        # Promote: posterior WR >= 70% AND profit factor >= 1.5 AND enough trades
                        domesticated = 1 if (
                            total >= DOMESTICATION_MIN_TRADES
                            and posterior_wr >= DOMESTICATION_MIN_WR
                            and profit_factor >= DOMESTICATION_MIN_PROFIT_FACTOR
                        ) else 0

                    # Sigmoid boost uses posterior WR (not raw) for consistency
                    boost = (1.0 + 0.30 * (1.0 / (1.0 + math.exp(-15 * (posterior_wr - 0.65))))) if domesticated else 1.0

                    cursor.execute("""
                        UPDATE domesticated_patterns
                        SET win_count=?, loss_count=?, win_rate=?, posterior_wr=?,
                            avg_win=?, avg_loss=?, profit_factor=?,
                            total_win_pnl=?, total_loss_pnl=?,
                            domesticated=?, boost_factor=?, last_seen=?, last_activated=?
                        WHERE pattern_hash=?
                    """, (win_count, loss_count, win_rate, posterior_wr,
                          avg_win, avg_loss, profit_factor,
                          total_win_pnl, total_loss_pnl,
                          domesticated, boost, now, now, pattern_hash))
                else:
                    # First occurrence — compute initial posterior and PnL
                    posterior_wr = (DOMESTICATION_PRIOR_ALPHA + (1 if won else 0)) / (
                        DOMESTICATION_PRIOR_ALPHA + DOMESTICATION_PRIOR_BETA + 1)
                    init_win_pnl = profit if won else 0.0
                    init_loss_pnl = abs(profit) if not won else 0.0
                    cursor.execute("""
                        INSERT INTO domesticated_patterns
                        (pattern_hash, te_combo, win_count, loss_count, win_rate,
                         posterior_wr, avg_win, avg_loss, profit_factor,
                         total_win_pnl, total_loss_pnl,
                         first_seen, last_seen, last_activated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (pattern_hash, combo,
                          1 if won else 0, 0 if won else 1,
                          1.0 if won else 0.0,
                          posterior_wr,
                          profit if won else 0.0,
                          abs(profit) if not won else 0.0,
                          0.0,  # profit_factor undefined with 1 trade
                          init_win_pnl, init_loss_pnl,
                          now, now, now))

                conn.commit()
        except Exception as e:
            log.warning("Domestication record failed: %s", e)

    def get_boost(self, active_tes: List[str]) -> float:
        """Get boost factor for a TE combination if domesticated and not expired."""
        if not active_tes:
            return 1.0

        combo = "+".join(sorted(active_tes))
        pattern_hash = hashlib.md5(combo.encode()).hexdigest()[:16]

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT boost_factor, domesticated, last_activated "
                    "FROM domesticated_patterns WHERE pattern_hash=?",
                    (pattern_hash,)
                )
                row = cursor.fetchone()

            if row and row[1]:  # domesticated == 1
                # Enforce expiry: skip patterns older than DOMESTICATION_EXPIRY_DAYS
                last_activated = row[2]
                if last_activated:
                    try:
                        last_dt = datetime.fromisoformat(last_activated)
                        age_days = (datetime.now() - last_dt).days
                        if age_days > DOMESTICATION_EXPIRY_DAYS:
                            log.debug("Domesticated pattern %s expired (%d days old)", combo, age_days)
                            return 1.0
                    except (ValueError, TypeError):
                        pass
                return float(row[0])
        except Exception:
            pass

        return 1.0


# ============================================================
# SPECIATION ENGINE (Serrato-Capuchina)
# ============================================================

class SpeciationEngine:
    """
    Models cross-instrument signal transfer (horizontal gene transfer)
    and reproductive isolation (speciation) based on correlation regimes.

    When instruments are correlated: TE signals transfer freely.
    When instruments diverge: speciation event -- signals must be independent.
    """

    def __init__(self):
        self.correlation_history: Dict[str, List[float]] = {}

    def compute_correlation(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        lookback: int = 50
    ) -> float:
        """Compute rolling correlation between two price series."""
        if len(prices_a) < lookback or len(prices_b) < lookback:
            return 0.0

        ret_a = np.diff(prices_a[-lookback:]) / prices_a[-lookback:-1]
        ret_b = np.diff(prices_b[-lookback:]) / prices_b[-lookback:-1]

        if len(ret_a) != len(ret_b):
            min_len = min(len(ret_a), len(ret_b))
            ret_a = ret_a[-min_len:]
            ret_b = ret_b[-min_len:]

        corr = np.corrcoef(ret_a, ret_b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def classify_relationship(
        self, corr: float
    ) -> Tuple[str, bool, float]:
        """
        Classify the relationship between two instruments.

        Returns:
            (relationship, allow_transfer, transfer_weight)
        """
        if abs(corr) >= SPECIATION_SAME_SPECIES:
            return "SAME_SPECIES", True, abs(corr)
        elif abs(corr) >= SPECIATION_HYBRID_ZONE:
            return "HYBRID_ZONE", False, 0.0  # Block transfer
        else:
            return "REPRODUCTIVE_ISOLATION", False, 0.0

    def attempt_te_invasion(
        self,
        host_activations: List[Dict],
        donor_activations: List[Dict],
        corr: float
    ) -> List[Dict]:
        """
        Attempt horizontal gene transfer of TE signals between instruments.
        Only allowed when correlation indicates same-species relationship.
        """
        relationship, allow, weight = self.classify_relationship(corr)

        if not allow:
            return host_activations

        # Transfer: blend donor signals into host
        invaded = []
        for i, host_act in enumerate(host_activations):
            a = dict(host_act)
            if i < len(donor_activations):
                donor = donor_activations[i]
                # Helitron and P_element are the best "jumpers"
                te = next((f for f in ALL_TE_FAMILIES if f.name == a["te"]), None)
                if te and te.name in ["Helitron", "P_element", "HERV_Synapse", "Arc_Capsid"]:
                    # These TEs can jump between species
                    blend = a["strength"] * (1 - weight * 0.3) + donor["strength"] * weight * 0.3
                    a["strength"] = float(np.clip(blend, 0.0, 1.0))
                    a["details"]["invasion_from"] = donor.get("te", "unknown")
                    a["details"]["invasion_weight"] = weight
            invaded.append(a)

        return invaded


# ============================================================
# MAIN TEQA v3.0 ENGINE
# ============================================================

class TEQAv3Engine:
    """
    Main orchestrator for TEQA v3.0 Neural-TE Integration.

    Pipeline:
        1. Compute TE activations from market data
        2. Detect genomic shock level
        3. Adjust TE thresholds for stress
        4. Create neural mosaic (L1 retrotransposition)
        5. Run quantum circuit per neuron
        6. Neural consensus vote
        7. Check speciation / cross-instrument signals
        8. TE domestication boost
        9. Output through Jardine's Gate
    """

    def __init__(
        self,
        n_neurons: int = DEFAULT_N_NEURONS,
        shots: int = DEFAULT_SHOTS,
        db_path: str = None,
        analytics_dir: str = "teqa_analytics",
        enable_evolution: bool = True,
        genome_file: str = None,
        evolve_every: int = 5,
    ):
        self.activation_engine = TEActivationEngine()
        self.shock_detector = GenomicShockDetector()
        self.mosaic = NeuralMosaicEngine(n_neurons=n_neurons)
        self.quantum = TEQAQuantumEngine(shots=shots)
        self.domestication = TEDomesticationTracker(db_path=db_path)
        self.speciation = SpeciationEngine()
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(exist_ok=True)

        self.n_neurons = n_neurons
        self.shots = shots

        # HGH Hormone Engine (somatotropic signal amplification)
        self.hgh_hormone = None
        try:
            from hgh_hormone import HGHHormoneEngine
            self.hgh_hormone = HGHHormoneEngine(db_path=db_path)
            log.info("HGH Hormone Engine ENABLED")
        except ImportError:
            log.info("hgh_hormone.py not found -- HGH amplification disabled")
        except Exception as e:
            log.warning("Failed to init HGH Hormone Engine: %s", e)

        # Neural Mosaic Evolution (Darwinian selection on circuit topology)
        self.evolution = None
        if enable_evolution:
            try:
                from neural_evolution import NeuralEvolutionEngine
                _gf = genome_file or str(Path(analytics_dir) / "evolved_genomes.json")
                self.evolution = NeuralEvolutionEngine(
                    mosaic=self.mosaic,
                    genome_file=_gf,
                    evolve_every=evolve_every,
                )
                log.info("Neural evolution ENABLED (evolve every %d cycles, genomes: %s)",
                         evolve_every, _gf)
            except ImportError:
                log.warning("neural_evolution.py not found -- evolution disabled")
            except Exception as e:
                log.warning("Failed to init neural evolution: %s", e)

        # TE Session Logger (activation pattern tracking by session/hour)
        self.session_logger = None
        try:
            from te_session_logger import TESessionLogger
            self.session_logger = TESessionLogger(
                db_path=str(Path(db_path).parent / "te_session_log.db") if db_path else None
            )
            log.info("TE Session Logger ENABLED")
        except ImportError:
            log.info("te_session_logger.py not found -- session logging disabled")
        except Exception as e:
            log.warning("Failed to init TE Session Logger: %s", e)

        # Focused Quantum Circuit Engine (right-sized circuits)
        self.focused_engine = None
        try:
            from focused_quantum_circuit import FocusedQuantumEngine
            self.focused_engine = FocusedQuantumEngine(base_shots=shots)
            log.info("Focused Quantum Engine ENABLED")
        except ImportError:
            log.info("focused_quantum_circuit.py not found -- focused circuits disabled")
        except Exception as e:
            log.warning("Failed to init Focused Quantum Engine: %s", e)

    def analyze(
        self,
        bars: np.ndarray,
        symbol: str = "BTCUSD",
        drawdown: float = 0.0,
        additional_signals: Optional[Dict] = None,
        donor_bars: Optional[np.ndarray] = None,
        donor_symbol: Optional[str] = None,
        save_analytics: bool = True
    ) -> Dict:
        """
        Run full TEQA v3.0 analysis pipeline.

        Args:
            bars: OHLCV numpy array (N x 5), at least 50 rows
            symbol: trading instrument
            drawdown: current account drawdown as fraction (e.g. 0.02 = 2%)
            additional_signals: pre-computed signals (cross_correlation, etc.)
            donor_bars: OHLCV data from another instrument for TE invasion
            donor_symbol: name of donor instrument
            save_analytics: write report to disk

        Returns:
            Dict with signal direction, confidence, gate results, etc.
        """
        t_start = time.time()
        signals = additional_signals or {}
        signals.setdefault("drawdown", drawdown)

        # === Step 1: TE Activation ===
        master_activations = self.activation_engine.compute_all_activations(
            bars, signals
        )

        # === Step 2: Genomic Shock ===
        shock_score, shock_label = self.shock_detector.compute_shock_level(
            bars, drawdown
        )

        # === Step 3: Stress-Adjust Thresholds ===
        adjusted_activations = self.shock_detector.adjust_te_thresholds(
            shock_score, shock_label, master_activations
        )

        # === Step 4: Cross-Instrument TE Invasion ===
        if donor_bars is not None and len(donor_bars) >= 50:
            donor_activations = self.activation_engine.compute_all_activations(
                donor_bars, signals
            )
            close_host = bars[:, 3] if bars.ndim == 2 else bars
            close_donor = donor_bars[:, 3] if donor_bars.ndim == 2 else donor_bars
            corr = self.speciation.compute_correlation(close_host, close_donor)
            adjusted_activations = self.speciation.attempt_te_invasion(
                adjusted_activations, donor_activations, corr
            )
            relationship, _, _ = self.speciation.classify_relationship(corr)
        else:
            corr = 0.0
            relationship = "NO_DONOR"

        # === Step 4B: HGH Hormone Cycle ===
        # Run BEFORE quantum execution so the hormone can inject rotations
        # into each neuron's circuits. Synthesizes 4-helix bundle from top
        # domesticated TEs, validates disulfide bridges, attempts receptor
        # dimerization, runs JAK2/STAT cascade.
        hgh_result = None
        if self.hgh_hormone is not None:
            try:
                hgh_result = self.hgh_hormone.run_cycle(
                    bars=bars,
                    symbol=symbol,
                    te_activations=adjusted_activations,
                )
            except Exception as e:
                log.warning("HGH Hormone cycle failed: %s", e)

        # === Step 5: Neural Mosaic Quantum Execution (Split Architecture) ===
        # Each neuron runs two circuits:
        #   Circuit A: 25-qubit genome (original TEs)
        #   Circuit B: 8-qubit neural (neural TEs)
        # Results are fused via "synaptic fusion"
        shots_per_neuron = max(512, self.shots // self.n_neurons)
        genome_shots = int(shots_per_neuron * 0.6)
        neural_shots = int(shots_per_neuron * 0.4)
        neuron_results = []

        for neuron in self.mosaic.neurons:
            neuron_acts = self.mosaic.get_neuron_activations(
                adjusted_activations, neuron
            )

            # Circuit A: Genome (25 qubits)
            genome_qc = self.quantum.build_genome_circuit(
                neuron_acts, neuron=neuron, shock_level=shock_score
            )
            if genome_qc is not None:
                genome_result = self.quantum.execute_circuit(
                    genome_qc, shots=genome_shots, n_qubits=N_QUBITS_GENOME
                )
            else:
                genome_result = self.quantum._classical_fallback(N_QUBITS_GENOME)

            # Extract genome signal for neural circuit injection
            genome_signal = (genome_result["vote_long"] - genome_result["vote_short"])

            # Circuit B: Neural (8 qubits)
            neural_qc = self.quantum.build_neural_circuit(
                neuron_acts, neuron=neuron, shock_level=shock_score,
                genome_signal=genome_signal
            )

            # HGH IGF-1 Bridge: inject hormone rotations into this neuron's circuits
            if hgh_result is not None and hgh_result.active and self.hgh_hormone is not None:
                try:
                    n_rot = self.hgh_hormone.inject_rotations(hgh_result, genome_qc, neural_qc)
                    if n_rot > 0:
                        hgh_result.quantum_rotations_applied += n_rot
                except Exception as e:
                    log.debug("HGH quantum injection failed for neuron %s: %s",
                              neuron.neuron_id, e)

            if neural_qc is not None:
                neural_result = self.quantum.execute_circuit(
                    neural_qc, shots=neural_shots, n_qubits=N_QUBITS_NEURAL
                )
            else:
                neural_result = self.quantum._classical_fallback(N_QUBITS_NEURAL)

            # Synaptic fusion
            qresult = self.quantum.fuse_results(genome_result, neural_result)

            # Determine neuron vote
            if qresult["vote_long"] > qresult["vote_short"] * 1.1:
                neuron.vote = 1
                neuron.confidence = qresult["vote_long"] / (
                    qresult["vote_long"] + qresult["vote_short"] + 1e-10
                )
            elif qresult["vote_short"] > qresult["vote_long"] * 1.1:
                neuron.vote = -1
                neuron.confidence = qresult["vote_short"] / (
                    qresult["vote_long"] + qresult["vote_short"] + 1e-10
                )
            else:
                neuron.vote = 0
                neuron.confidence = 0.0

            neuron_results.append({
                "neuron_id": neuron.neuron_id,
                "vote": neuron.vote,
                "confidence": neuron.confidence,
                "n_insertions": len(neuron.insertions),
                "quantum": qresult,
            })

        # === Step 6: Neural Consensus ===
        consensus_dir, consensus_score, vote_counts = self.mosaic.compute_consensus()

        # === Step 7: TE Domestication Boost ===
        active_tes = [a["te"] for a in adjusted_activations if a["strength"] > 0.3]
        self._last_active_tes = active_tes  # stash for record_trade_outcome()
        domestication_boost = self.domestication.get_boost(active_tes)

        # === Step 8: Compute Final Signal ===
        # Concordance from direct activations
        n_long = sum(1 for a in adjusted_activations if a["direction"] > 0 and a["strength"] > 0.3)
        n_short = sum(1 for a in adjusted_activations if a["direction"] < 0 and a["strength"] > 0.3)
        n_neutral = sum(1 for a in adjusted_activations if a["direction"] == 0 or a["strength"] <= 0.3)
        total_active = n_long + n_short + n_neutral

        if total_active > 0:
            concordance = max(n_long, n_short) / total_active
        else:
            concordance = 0.0

        # Direction: combine TE concordance with neural consensus
        if consensus_dir != 0:
            direction = consensus_dir
        elif n_long > n_short:
            direction = 1
        elif n_short > n_long:
            direction = -1
        else:
            direction = 0

        # Confidence: blend of concordance, consensus, and domestication
        raw_confidence = concordance * 0.3 + consensus_score * 0.3 + min(0.4, (domestication_boost - 1.0))

        # HGH Hypertrophy Boost: amplify confidence when hormone is active
        if hgh_result is not None and hgh_result.active:
            raw_confidence += hgh_result.hypertrophy_boost

        confidence = float(np.clip(raw_confidence, 0.0, 1.0))

        # Apply quantum novelty adjustment
        if neuron_results:
            avg_novelty = np.mean([nr["quantum"]["novelty"] for nr in neuron_results])
            # Novel states can either boost or reduce confidence
            if avg_novelty > 0.5:
                confidence *= 0.9  # Too much novelty = uncertain
            elif avg_novelty < 0.1:
                confidence *= 0.95  # Too little = stale signal

        # Amplitude squared (from top quantum state)
        if neuron_results:
            avg_top_ratio = np.mean([
                nr["quantum"]["top_count"] / max(1, nr["quantum"]["n_shots"])
                for nr in neuron_results
            ])
        else:
            avg_top_ratio = 0.0

        elapsed_ms = (time.time() - t_start) * 1000

        # === Step 9: Package Results ===
        result = {
            "version": VERSION,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": elapsed_ms,

            # Final signal
            "direction": direction,
            "confidence": confidence,
            "concordance": concordance,
            "amplitude_sq": avg_top_ratio,

            # Genomic shock
            "shock_score": shock_score,
            "shock_label": shock_label,

            # Neural mosaic
            "n_neurons": self.n_neurons,
            "consensus_direction": consensus_dir,
            "consensus_score": consensus_score,
            "vote_counts": vote_counts,
            "neural_consensus_pass": consensus_score >= NEURAL_CONSENSUS_THRESHOLD,

            # Speciation
            "cross_instrument_corr": corr,
            "relationship": relationship,

            # Domestication
            "domestication_boost": domestication_boost,
            "active_tes": active_tes,

            # TE activation summary
            "n_active_class_i": sum(1 for a in adjusted_activations
                                    if a["strength"] > 0.5 and
                                    next((f for f in ALL_TE_FAMILIES if f.name == a["te"]), None) and
                                    next((f for f in ALL_TE_FAMILIES if f.name == a["te"])).te_class == TEClass.RETROTRANSPOSON),
            "n_active_class_ii": sum(1 for a in adjusted_activations
                                     if a["strength"] > 0.5 and
                                     next((f for f in ALL_TE_FAMILIES if f.name == a["te"]), None) and
                                     next((f for f in ALL_TE_FAMILIES if f.name == a["te"])).te_class == TEClass.DNA_TRANSPOSON),
            "n_active_neural": sum(1 for a in adjusted_activations
                                   if a["strength"] > 0.5 and
                                   next((f for f in ALL_TE_FAMILIES if f.name == a["te"]), None) and
                                   next((f for f in ALL_TE_FAMILIES if f.name == a["te"])).te_class == TEClass.NEURAL),

            # Gate checks (for Jardine's Gate integration)
            "gates": {
                "G7_neural_consensus": consensus_score >= NEURAL_CONSENSUS_THRESHOLD,
                "G8_genomic_shock": shock_label not in ["EXTREME"],
                "G9_speciation": relationship != "HYBRID_ZONE",
                "G10_domestication": domestication_boost >= 1.0,
            },

            # HGH Hormone (Somatotropic Signal Amplification)
            "hgh": {
                "active": hgh_result.active if hgh_result else False,
                "growth_signal": round(hgh_result.growth_signal, 6) if hgh_result else 0.0,
                "variant": hgh_result.molecule.variant if hgh_result and hgh_result.molecule else "none",
                "potency": hgh_result.molecule.potency if hgh_result and hgh_result.molecule else 0.0,
                "bridge_1_intact": hgh_result.molecule.bridge_1_intact if hgh_result and hgh_result.molecule else False,
                "bridge_2_intact": hgh_result.molecule.bridge_2_intact if hgh_result and hgh_result.molecule else False,
                "binding_strength": hgh_result.binding.binding_strength if hgh_result and hgh_result.binding else 0.0,
                "jak2": round(hgh_result.cascade.jak2_phosphorylation, 4) if hgh_result and hgh_result.cascade else 0.0,
                "stat_docking": round(hgh_result.cascade.stat_docking, 4) if hgh_result and hgh_result.cascade else 0.0,
                "hyperplasia": hgh_result.hyperplasia if hgh_result else False,
                "second_lot_ratio": round(hgh_result.second_lot_ratio, 3) if hgh_result else 0.0,
                "hypertrophy_boost": round(hgh_result.hypertrophy_boost, 4) if hgh_result else 0.0,
                "helices": hgh_result.helices_used if hgh_result else [],
                "rotations_applied": hgh_result.quantum_rotations_applied if hgh_result else 0,
                "suppression": hgh_result.suppression_reason if hgh_result else "",
                "lipolysis": {
                    "sl_tighten_factor": round(hgh_result.lipolysis_factor, 4) if hgh_result else 1.0,
                },
            },

            # Detailed data for analytics
            "te_activations": adjusted_activations,
            "neuron_results": neuron_results,
        }

        # === Step 9B: Session Logging ===
        if self.session_logger is not None:
            try:
                hour_utc = datetime.utcnow().hour
                session_name = self.session_logger.get_session_name(hour_utc)
                self.session_logger.log_activations(
                    adjusted_activations, hour_utc, session_name,
                    shock_label, symbol
                )
                result["session"] = {
                    "hour_utc": hour_utc,
                    "session_name": session_name,
                }
            except Exception as e:
                log.warning("Session logging failed (non-fatal): %s", e)
                hour_utc = datetime.utcnow().hour
                session_name = "UNKNOWN"
                result["session"] = {"hour_utc": hour_utc, "session_name": session_name}
        else:
            hour_utc = datetime.utcnow().hour
            session_name = "UNKNOWN"
            result["session"] = {"hour_utc": hour_utc, "session_name": session_name}

        # === Step 9C: Focused Quantum Circuit (Active-Only vs Dormant) ===
        if self.focused_engine is not None:
            try:
                focused_active = [a["te"] for a in adjusted_activations if a["strength"] > 0.3]
                focused_dormant = [a["te"] for a in adjusted_activations if a["strength"] <= 0.3]

                focused_result = self.focused_engine.run_focused(
                    adjusted_activations, focused_active,
                    self.mosaic.neurons, shock_score, self.shots
                )
                dormant_result = self.focused_engine.run_dormant(
                    adjusted_activations, focused_dormant,
                    self.mosaic.neurons, shock_score, self.shots
                )

                result["focused_circuit"] = {
                    "n_active_qubits": len(focused_active),
                    "n_dormant_qubits": len(focused_dormant),
                    "active_tes": focused_active,
                    "dormant_tes": focused_dormant,
                    "focused_confidence": focused_result["confidence"],
                    "focused_direction": focused_result["direction"],
                    "dormant_confidence": dormant_result["confidence"],
                    "dormant_direction": dormant_result["direction"],
                    "agreement": focused_result["direction"] == dormant_result["direction"],
                    "contrarian_alert": (
                        dormant_result["direction"] != 0 and
                        dormant_result["direction"] != focused_result["direction"]
                    ),
                }
            except Exception as e:
                log.warning("Focused circuit failed (non-fatal): %s", e)
                result["focused_circuit"] = {"error": str(e)}
        else:
            result["focused_circuit"] = {"enabled": False}

        # === Step 10: Evolution data (if enabled) ===
        if self.evolution is not None:
            evo_stats = self.evolution.get_population_stats()
            result["evolution"] = {
                "enabled": True,
                "generation": evo_stats["generation"],
                "cycle": evo_stats["cycle"],
                "avg_accuracy": evo_stats["avg_accuracy"],
                "best_accuracy": evo_stats["best_accuracy"],
                "worst_accuracy": evo_stats["worst_accuracy"],
                "unique_genomes": evo_stats["unique_genomes"],
                "avg_similarity": evo_stats["avg_pairwise_similarity"],
                "speciation_pressure": evo_stats["speciation_pressure"],
            }
        else:
            result["evolution"] = {"enabled": False}

        # === Step 11: Stanozolol-DMT TE Bridge (OPTIONAL EXTENSION) ===
        # 11-layer / 13-gate / 5-channel / 16-qubit power layer.
        # Runs AFTER the normal pipeline and can boost or suppress confidence.
        # If stanozolol_dmt_bridge.py does not exist, this is silently skipped.
        try:
            from stanozolol_dmt_bridge import apply_bridge
            result = apply_bridge(result)
            log.info("Stanozolol-DMT Bridge: applied (conf=%.4f)", result["confidence"])
        except ImportError:
            pass  # Bridge not installed -- TEQA works fine without it
        except Exception as e:
            log.warning("Stanozolol-DMT Bridge failed (non-fatal): %s", e)

        # === Step 12: Testosterone-DMT TE Bridge (OPTIONAL EXTENSION #2) ===
        # 4-layer / 4-gate / 5-channel / 8-qubit AGGRESSIVE power layer.
        # Testosterone: raw power, trend-following, wider stops, bigger targets.
        # Stanozolol: precision, scalping, tight execution.
        # The BRAIN can choose which molecular extension to use (or none).
        try:
            from testosterone_dmt_bridge import create_bridge
            testosterone_bridge = create_bridge(shots=4096)

            # Prepare market data for testosterone processing
            market_data_for_testosterone = {
                'close': bars[:, 3].tolist() if bars.ndim == 2 else bars.tolist(),
                'volume': bars[:, 4].tolist() if bars.ndim == 2 and bars.shape[1] >= 5 else [],
                'volatility': signals.get('volatility', 1.0),
                'drawdown': drawdown,
            }

            # Base signal from TEQA (normalized to -1 to +1)
            base_signal = direction * confidence

            # Immune conflict (CRISPR/VDJ conflicts would be calculated here)
            # For now, use genomic shock as proxy
            immune_conflict = min(1.0, shock_score / 3.0)

            # Process through testosterone bridge
            testosterone_result = testosterone_bridge.process_signal(
                market_data=market_data_for_testosterone,
                base_signal=base_signal,
                immune_conflict=immune_conflict
            )

            # Apply testosterone decision
            if testosterone_result['action'] == 'boost':
                # Boost confidence and apply aggressive parameters
                original_confidence = confidence
                confidence = min(1.0, confidence * (1 + testosterone_result['strength'] * 0.5))
                result['testosterone_boost'] = testosterone_result['strength']
                result['testosterone_regime'] = testosterone_result['regime']
                result['testosterone_position_mult'] = testosterone_result['position_multiplier']
                result['testosterone_stop_mult'] = testosterone_result['stop_multiplier']
                result['testosterone_target_mult'] = testosterone_result['target_multiplier']
                log.info("Testosterone-DMT Bridge: BOOST applied (%.4f → %.4f), regime=%s",
                        original_confidence, confidence, testosterone_result['regime'])
            elif testosterone_result['action'] == 'suppress':
                # Suppress confidence
                original_confidence = confidence
                confidence = confidence * 0.7
                result['testosterone_suppress'] = True
                result['testosterone_regime'] = testosterone_result['regime']
                log.info("Testosterone-DMT Bridge: SUPPRESS applied (%.4f → %.4f), regime=%s",
                        original_confidence, confidence, testosterone_result['regime'])
            else:
                # Neutral - no change
                result['testosterone_neutral'] = True
                result['testosterone_regime'] = testosterone_result['regime']
                log.debug("Testosterone-DMT Bridge: NEUTRAL (no change)")

            # Store full testosterone result for analytics
            result['testosterone_dmt'] = {
                'action': testosterone_result['action'],
                'strength': testosterone_result['strength'],
                'regime': testosterone_result['regime'],
                'gates_passed': testosterone_result['gates']['all_gates_passed'],
                'gates_count': f"{testosterone_result['gates']['gates_passed_count']}/4",
                'position_multiplier': testosterone_result['position_multiplier'],
                'stop_multiplier': testosterone_result['stop_multiplier'],
                'target_multiplier': testosterone_result['target_multiplier'],
                'processing_time_ms': testosterone_result['processing_time_ms'],
            }

            # Update confidence in result dict after testosterone modification
            result['confidence'] = confidence

        except ImportError:
            pass  # Testosterone bridge not installed -- TEQA works fine without it
        except Exception as e:
            log.warning("Testosterone-DMT Bridge failed (non-fatal): %s", e)

        # Save analytics
        if save_analytics:
            self._save_analytics(result)

        return result

    def record_trade_outcome(self, won: bool, profit: float = 0.0, active_tes: List[str] = None):
        """Record a trade outcome for TE domestication learning.

        If active_tes is None, uses the TEs from the most recent analyze() call.
        """
        tes = active_tes if active_tes is not None else getattr(self, '_last_active_tes', [])
        if not tes:
            log.debug("record_trade_outcome: no active TEs to record")
            return
        self.domestication.record_pattern(tes, won, profit=profit)

    def feed_market_direction(self, actual_direction: int) -> Optional[Dict]:
        """
        Feed the actual market direction back to the neural evolution engine.
        Call this AFTER analyze() when you know what the market actually did.

        actual_direction: 1 (went up), -1 (went down), 0 (flat)

        Returns evolution event dict if evolution occurred, else None.
        This is the feedback that makes the mosaic get smarter over time.
        """
        if self.evolution is None:
            return None
        return self.evolution.record_votes(actual_direction)

    def _save_analytics(self, result: Dict):
        """Save analytics report."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Text report
        report_path = self.analytics_dir / f"report_{run_id}.txt"
        try:
            with open(report_path, "w") as f:
                f.write("=" * 76 + "\n")
                f.write(f"  TEQA v3.0 NEURAL-TE ANALYTICS REPORT\n")
                f.write(f"  Run ID: {run_id}\n")
                f.write(f"  Duration: {result['elapsed_ms']:.0f}ms\n")
                f.write("=" * 76 + "\n\n")

                f.write(f"  GENOMIC SHOCK: {result['shock_label']} (score={result['shock_score']:.2f})\n\n")

                f.write("  NEURAL MOSAIC ({} neurons)\n".format(result["n_neurons"]))
                f.write("  " + "-" * 70 + "\n")
                vc = result["vote_counts"]
                f.write(f"  LONG: {vc['long']}  SHORT: {vc['short']}  NEUTRAL: {vc['neutral']}\n")
                f.write(f"  Consensus: {result['consensus_score']:.3f} -> {'PASS' if result['neural_consensus_pass'] else 'FAIL'}\n\n")

                f.write("  TE ACTIVATION SUMMARY\n")
                f.write("  " + "-" * 70 + "\n")
                f.write(f"  Class I  (Retrotransposons): {result['n_active_class_i']} active\n")
                f.write(f"  Class II (DNA Transposons):   {result['n_active_class_ii']} active\n")
                f.write(f"  Neural:                       {result['n_active_neural']} active\n\n")

                f.write("  SPECIATION\n")
                f.write("  " + "-" * 70 + "\n")
                f.write(f"  Correlation: {result['cross_instrument_corr']:.3f}\n")
                f.write(f"  Relationship: {result['relationship']}\n\n")

                f.write("  GATE CHECKS (G7-G10)\n")
                f.write("  " + "-" * 70 + "\n")
                for gate, passed in result["gates"].items():
                    f.write(f"  {gate}: {'PASS' if passed else 'FAIL'}\n")

                f.write(f"\n  FINAL SIGNAL\n")
                f.write("  " + "-" * 70 + "\n")
                dir_str = "LONG" if result["direction"] > 0 else ("SHORT" if result["direction"] < 0 else "NEUTRAL")
                f.write(f"  Direction:    {dir_str}\n")
                f.write(f"  Confidence:   {result['confidence']:.4f}\n")
                f.write(f"  Concordance:  {result['concordance']:.4f}\n")
                f.write(f"  Domestication boost: {result['domestication_boost']:.2f}\n")
                f.write(f"  Amplitude sq: {result['amplitude_sq']:.4f}\n")

                # Evolution section
                evo = result.get("evolution", {})
                if evo.get("enabled"):
                    f.write(f"\n  NEURAL EVOLUTION (Darwinian Selection)\n")
                    f.write("  " + "-" * 70 + "\n")
                    f.write(f"  Generation:       {evo.get('generation', 0)}\n")
                    f.write(f"  Cycle:            {evo.get('cycle', 0)}\n")
                    f.write(f"  Avg accuracy:     {evo.get('avg_accuracy', 0):.1%}\n")
                    f.write(f"  Best accuracy:    {evo.get('best_accuracy', 0):.1%}\n")
                    f.write(f"  Worst accuracy:   {evo.get('worst_accuracy', 0):.1%}\n")
                    f.write(f"  Unique genomes:   {evo.get('unique_genomes', 0)}/{result['n_neurons']}\n")
                    f.write(f"  Avg similarity:   {evo.get('avg_similarity', 0):.2f}\n")
                    f.write(f"  Speciation press: {'YES' if evo.get('speciation_pressure') else 'no'}\n")

                f.write("\n" + "=" * 76 + "\n")

        except Exception as e:
            log.warning("Failed to save analytics report: %s", e)

        # JSON log
        json_path = self.analytics_dir / f"full_log_{run_id}.json"
        try:
            # Strip non-serializable data
            log_data = {k: v for k, v in result.items()
                        if k not in ["te_activations", "neuron_results"]}
            log_data["te_activation_count"] = len(result.get("te_activations", []))
            log_data["neuron_count"] = len(result.get("neuron_results", []))

            with open(json_path, "w") as f:
                json.dump(log_data, f, indent=2, default=str)
        except Exception as e:
            log.warning("Failed to save analytics JSON: %s", e)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    print("=" * 76)
    print("  TEQA v3.0 -- Neural-Transposable Element Integration")
    print("  Standalone Test")
    print("=" * 76)

    # Generate synthetic test data (OHLCV)
    np.random.seed(42)
    n_bars = 200
    close = np.cumsum(np.random.randn(n_bars) * 0.5 + 0.02) + 50000
    high = close + np.abs(np.random.randn(n_bars) * 50)
    low = close - np.abs(np.random.randn(n_bars) * 50)
    open_p = close + np.random.randn(n_bars) * 20
    volume = np.abs(np.random.randn(n_bars) * 100 + 500)

    bars = np.column_stack([open_p, high, low, close, volume])

    print(f"\n  Synthetic data: {n_bars} bars")
    print(f"  TE families: {N_QUBITS} (25 original + 8 neural)")
    print(f"  Qiskit available: {QISKIT_AVAILABLE}")

    # Run TEQA v3.0
    engine = TEQAv3Engine(
        n_neurons=5,  # Faster for test
        shots=2048,   # Fewer shots for test
        analytics_dir="C:/Users/jimjj/Music/QuantumChildren/QuantumTradingLibrary/teqa_analytics"
    )

    print("\n  Running analysis...")
    result = engine.analyze(
        bars=bars,
        symbol="BTCUSD",
        drawdown=0.01,
    )

    dir_str = "LONG" if result["direction"] > 0 else ("SHORT" if result["direction"] < 0 else "NEUTRAL")
    print(f"\n  RESULTS:")
    print(f"  Direction:        {dir_str}")
    print(f"  Confidence:       {result['confidence']:.4f}")
    print(f"  Shock:            {result['shock_label']} ({result['shock_score']:.2f})")
    print(f"  Neural consensus: {result['consensus_score']:.3f} ({'PASS' if result['neural_consensus_pass'] else 'FAIL'})")
    print(f"  Active TEs:       {len(result['active_tes'])}")
    print(f"  Elapsed:          {result['elapsed_ms']:.0f}ms")

    print(f"\n  GATE CHECKS (G7-G10):")
    for gate, passed in result["gates"].items():
        status = "PASS" if passed else "FAIL"
        print(f"    {gate}: {status}")

    print(f"\n  Vote breakdown: {result['vote_counts']}")
    print(f"  Speciation: {result['relationship']} (corr={result['cross_instrument_corr']:.3f})")

    print("\n" + "=" * 76)
    print("  Test complete. Analytics saved to teqa_analytics/")
    print("=" * 76)
