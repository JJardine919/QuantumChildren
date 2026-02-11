"""
CANCER CELL ENGINE -- Oncogenic Strategy Acceleration
======================================================
Full cancer simulation using REAL mathematical oncology models:
  - Gompertzian growth kinetics (tumor growth curves)
  - Knudson's two-hit hypothesis (tumor suppressor inactivation)
  - Vogelstein's multi-hit model (driver mutation accumulation)
  - Michaelis-Menten kinetics (TE activation rates)
  - Lotka-Volterra competition (strategy vs market adaptation)
  - Fisher-KPP equation (metastatic invasion speed)
  - Stochastic birth-death process (clonal evolution)
  - Warburg metabolic flux (GPU throughput over precision)
  - Telomere shortening model (strategy aging / Hayflick limit)
  - VEGF angiogenesis switch (resource allocation)

Runs entirely in SIMULATION. No live trades.
SL remains sacred ($1.00 from config_loader).

Authors: DooDoo + Claude
Date:    2026-02-09
Version: CANCER-CELL-1.0
"""

import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Optional GPU + Quantum
try:
    import torch
    import torch_directml
    GPU_AVAILABLE = True
    GPU_DEVICE = torch_directml.device()
except ImportError:
    GPU_AVAILABLE = False
    GPU_DEVICE = None

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Import trading settings (NEVER hardcode)
from config_loader import CONFIDENCE_THRESHOLD, MAX_LOSS_DOLLARS

log = logging.getLogger(__name__)

VERSION = "CANCER-CELL-1.1"
DB_PATH = Path(__file__).parent / "cancer_cell.db"
EXPERTS_DIR = Path(__file__).parent / "top_50_experts"
MANIFEST_PATH = EXPERTS_DIR / "top_50_manifest.json"

# ============================================================
# LSTM MODEL (must match BRAIN_ATLAS.py / ETARE_50_Darwin.py)
# ============================================================

if GPU_AVAILABLE:
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self, input_size=8, hidden_size=128, output_size=3, num_layers=2):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size,
                                num_layers=num_layers, batch_first=True)
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
# FEATURE PREPARATION (8 features, matches ETARE exactly)
# ============================================================

SEQ_LENGTH = 30
PREDICTION_HORIZON = 5


def prepare_features_from_ohlcv(bars: np.ndarray) -> Optional[pd.DataFrame]:
    """
    Build 8-feature DataFrame from OHLCV numpy array.
    bars shape: (N, 5) with columns [open, high, low, close, volume].
    Returns DataFrame with z-scored features + close column, or None.
    """
    if len(bars) < 60:
        return None
    df = pd.DataFrame(bars, columns=['open', 'high', 'low', 'close', 'volume'])

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss_s + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std

    # Momentum + ROC
    df['momentum'] = df['close'] / df['close'].shift(10)
    df['roc'] = df['close'].pct_change(10) * 100

    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    df = df.dropna()
    if len(df) < SEQ_LENGTH + PREDICTION_HORIZON + 1:
        return None

    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper',
                    'bb_lower', 'momentum', 'roc', 'atr']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    return df


def evaluate_expert_on_bars(model, bars: np.ndarray) -> dict:
    """
    Evaluate a loaded LSTM expert on OHLCV bars.
    Returns: {win_rate, profit_factor, total_trades, accuracy, fitness}
    """
    df = prepare_features_from_ohlcv(bars)
    if df is None:
        return {"win_rate": 0, "profit_factor": 0, "total_trades": 0,
                "accuracy": 0, "fitness": 0}

    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper',
                    'bb_lower', 'momentum', 'roc', 'atr']
    features = df[feature_cols].values
    prices = df['close'].values

    # Build 30-bar sequences
    X_list, p_list = [], []
    for i in range(len(features) - SEQ_LENGTH - PREDICTION_HORIZON):
        X_list.append(features[i:i + SEQ_LENGTH])
        p_list.append(prices[i + SEQ_LENGTH])

    if len(X_list) < 10:
        return {"win_rate": 0, "profit_factor": 0, "total_trades": 0,
                "accuracy": 0, "fitness": 0}

    X = torch.FloatTensor(np.array(X_list))
    price_arr = np.array(p_list)

    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)

    preds = predicted.cpu().numpy()
    wins, losses = 0, 0
    gross_profit, gross_loss = 0.0, 0.0

    for i in range(len(preds) - PREDICTION_HORIZON):
        pred = preds[i]
        if pred == 0:  # HOLD
            continue
        current_price = price_arr[i]
        future_price = price_arr[i + PREDICTION_HORIZON] if i + PREDICTION_HORIZON < len(price_arr) else current_price

        if pred == 1:  # BUY
            pnl = future_price - current_price
        else:  # SELL
            pnl = current_price - future_price

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    profit_factor = gross_profit / (gross_loss + 1e-10)
    fitness = win_rate * 0.4 + profit_factor * 0.2 + 0.3  # Base fitness

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "accuracy": 0,  # Skip for speed
        "fitness": fitness,
    }


# ============================================================
# EXPERT CELL LOADER -- Real LSTM Experts as Parent Cells
# ============================================================

class ExpertCellLoader:
    """
    Loads trained LSTM experts from top_50_experts/ and converts them
    into parent strategy dicts for the cancer cell engine.

    Each expert becomes a parent cell whose LSTM weight signature
    defines the TE weight deltas for its offspring mutations.
    """

    def __init__(self, experts_dir: Path = EXPERTS_DIR):
        self.experts_dir = experts_dir
        self.manifest = None
        self._load_manifest()

    def _load_manifest(self):
        manifest_path = self.experts_dir / "top_50_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            log.info("CANCER: Loaded expert manifest with %d experts",
                     len(self.manifest.get("experts", [])))
        else:
            log.warning("CANCER: Expert manifest not found at %s", manifest_path)

    def load_expert_model(self, expert_info: dict):
        """Load a single expert LSTM model."""
        if not GPU_AVAILABLE:
            return None
        path = self.experts_dir / expert_info["filename"]
        if not path.exists():
            return None
        try:
            model = LSTMModel(
                input_size=expert_info.get("input_size", 8),
                hidden_size=expert_info.get("hidden_size", 128),
                output_size=3,
                num_layers=2,
            )
            state_dict = torch.load(str(path), map_location='cpu',
                                    weights_only=False)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            log.warning("CANCER: Failed to load expert %s: %s",
                        expert_info["filename"], e)
            return None

    def extract_weight_signature(self, model) -> Dict[str, float]:
        """
        Extract a TE-weight-delta signature from LSTM weight matrices.

        Takes the first layer's weight norms and maps them to TE family
        weights. This gives each expert a unique mutation fingerprint
        based on what the neural network actually learned.
        """
        sig = {}
        try:
            state = model.state_dict()
            # LSTM weight_ih_l0 shape: (4*hidden, input_size) = (512, 8)
            w = state.get("lstm.weight_ih_l0")
            if w is not None:
                # Average across 4 gates (input, forget, cell, output)
                w_np = w.cpu().numpy()
                # Take column norms → one weight per input feature
                col_norms = np.linalg.norm(w_np, axis=0)
                # Normalize to [0, 1]
                col_norms = col_norms / (col_norms.max() + 1e-10)
                feature_names = ["RSI", "MACD", "MACD_SIG", "BB_UPPER",
                                 "BB_LOWER", "MOMENTUM", "ROC", "ATR"]
                for i, name in enumerate(feature_names):
                    if i < len(col_norms):
                        sig[name] = float(col_norms[i])

            # Also extract fc layer bias as overall direction preference
            fc_bias = state.get("fc.bias")
            if fc_bias is not None:
                bias_np = fc_bias.cpu().numpy()
                # BUY bias minus SELL bias → directional tendency
                sig["_direction_bias"] = float(bias_np[1] - bias_np[2])
        except Exception as e:
            log.warning("CANCER: Weight extraction failed: %s", e)
        return sig

    def get_parent_strategies(self, symbols: List[str],
                              bars_by_symbol: Dict[str, np.ndarray],
                              max_parents: int = 10) -> List[dict]:
        """
        Load experts matching the given symbols, evaluate them on recent
        bars, and return parent strategy dicts for the cancer engine.

        Each parent has:
          - id: expert filename
          - fitness: evaluated fitness on recent data
          - generation: 0 (seed generation)
          - win_rate: actual win rate from backtest
          - symbol: which symbol this expert trades
          - te_weight_signature: LSTM weight fingerprint
          - model: loaded PyTorch model (for reference)
        """
        if not self.manifest or not GPU_AVAILABLE:
            log.warning("CANCER: No manifest or no GPU, "
                        "returning empty parent list")
            return []

        parents = []
        experts = self.manifest.get("experts", [])

        for expert_info in experts:
            if len(parents) >= max_parents:
                break

            expert_symbol = expert_info.get("symbol", "")
            if expert_symbol not in symbols:
                continue

            model = self.load_expert_model(expert_info)
            if model is None:
                continue

            # Evaluate on actual bars
            bars = bars_by_symbol.get(expert_symbol)
            if bars is not None and len(bars) > 100:
                result = evaluate_expert_on_bars(model, bars)
            else:
                result = {"win_rate": 0, "fitness": expert_info.get("fitness", 0.1),
                          "profit_factor": 0, "total_trades": 0}

            # Extract weight signature as TE deltas
            weight_sig = self.extract_weight_signature(model)

            parent = {
                "id": expert_info["filename"],
                "fitness": result["fitness"],
                "generation": 0,
                "win_rate": result["win_rate"],
                "profit_factor": result["profit_factor"],
                "total_trades": result["total_trades"],
                "symbol": expert_symbol,
                "te_weight_signature": weight_sig,
                "model": model,  # Keep loaded LSTM for inference
            }
            parents.append(parent)

            log.info("CANCER: Expert parent loaded: %s [%s] "
                     "WR=%.1f%% PF=%.2f trades=%d fitness=%.3f",
                     expert_info["filename"], expert_symbol,
                     result["win_rate"] * 100,
                     result["profit_factor"],
                     result["total_trades"],
                     result["fitness"])

        log.info("CANCER: %d expert parents loaded for symbols %s",
                 len(parents), symbols)
        return parents

# ============================================================
# MATHEMATICAL CONSTANTS (from real cancer biology)
# ============================================================

# --- Gompertzian Growth ---
# dN/dt = rho * N * ln(K / N)
# N(t) = K * exp(ln(N0/K) * exp(-alpha * t))
GOMPERTZ_ALPHA = 0.08       # Intrinsic growth rate (day^-1)
GOMPERTZ_BETA = 0.008       # Deceleration factor (day^-1)
GOMPERTZ_K = 1e10           # Carrying capacity (cells)

# --- Knudson Two-Hit ---
# P(tumor) = 1 - exp(-mu1 * t) * exp(-mu2 * t)
# mu = 2e-7 per allele per year
KNUDSON_MU = 2e-7           # Mutation rate per allele per year

# --- Vogelstein Multi-Hit ---
# P(k mutations by time t) = (mu*t)^k / k! * exp(-mu*t)  [Poisson]
VOGELSTEIN_MIN_DRIVERS = 3   # Minimum driver mutations for "cancer"
VOGELSTEIN_MAX_DRIVERS = 7   # Maximum driver mutations to test
VOGELSTEIN_MU = 1e-6         # Mutation rate per driver gene per division

# --- Michaelis-Menten Kinetics ---
# v = Vmax * [S] / (Km + [S])
# Used for TE activation rate modeling
MM_VMAX = 1.0               # Maximum activation rate (normalized)
MM_KM = 0.5                 # Michaelis constant (signal at half-max)

# --- Lotka-Volterra (Strategy vs Market) ---
# dT/dt = rT(1 - T/K) - alpha*T*I   (strategy population)
# dI/dt = sI(1 - I/J) + beta*T*I - gamma*I  (market immune response)
LV_R = 0.5                  # Strategy growth rate
LV_ALPHA = 1e-3             # Market killing rate
LV_S = 0.3                  # Market immune growth rate
LV_BETA = 0.1               # Market activation by strategy
LV_GAMMA = 0.2              # Market immune decay rate

# --- Fisher-KPP (Metastasis Invasion) ---
# du/dt = D * d2u/dx2 + r*u*(1-u)
# Wave speed: c = 2 * sqrt(D * r)
FISHER_D = 0.001            # Diffusion coefficient (adaptability)
FISHER_R = 0.1              # Local growth rate

# --- Birth-Death Process (Clonal Evolution) ---
# E[N(t)] = N0 * exp((lambda - mu) * t)
# P(extinction) = (mu/lambda) for lambda > mu
BD_LAMBDA = 0.6             # Birth rate (winning trades per unit time)
BD_MU = 0.4                 # Death rate (losing trades per unit time)

# --- Warburg Metabolic Flux ---
# Glycolysis: 2 ATP/glucose, rate ~100 umol/min/g
# OXPHOS: 32 ATP/glucose, rate ~20 umol/min/g
# Warburg ratio = lactate_prod / glucose_consumption
WARBURG_GLYCOLYSIS_YIELD = 2     # ATP per unit (fast, low yield)
WARBURG_OXPHOS_YIELD = 32        # ATP per unit (slow, high yield)
WARBURG_GLYCOLYSIS_RATE = 100.0  # Units per time (fast)
WARBURG_OXPHOS_RATE = 20.0       # Units per time (slow)

# --- Telomere Shortening ---
# L(n) = L0 - delta * n
# Hayflick limit: n_max = (L0 - L_critical) / delta
TELOMERE_L0 = 10000         # Initial telomere length (bp)
TELOMERE_DELTA = 75          # Shortening per division (bp)
TELOMERE_L_CRITICAL = 4000  # Critical length triggering senescence (bp)
TELOMERE_HAYFLICK = (TELOMERE_L0 - TELOMERE_L_CRITICAL) // TELOMERE_DELTA  # ~80

# --- VEGF Angiogenesis Switch ---
# dV/dt = D_V * nabla^2(V) - k_decay*V + k_prod*T*H(T - T_crit)
# Angiogenic switch: V > V_threshold
VEGF_THRESHOLD = 0.5        # Concentration threshold for angiogenesis
VEGF_K_DECAY = 0.001        # VEGF degradation rate
VEGF_K_PROD = 1e-4          # VEGF production rate per cell
VEGF_DIFFUSION = 1e-6       # VEGF diffusion coefficient

# --- Simulation Bounds ---
MITOSIS_POPULATION_SIZE = 200
MITOSIS_PARENT_COUNT = 10
MITOSIS_GENERATIONS = 5
BYPASS_MAX_DURATION_SEC = 3600
BYPASS_CONFIDENCE_FLOOR = 0.05
WARBURG_BATCH_SIZE = 64
WARBURG_SHOTS_FAST = 2048
METASTASIS_MIN_FITNESS = 0.65
METASTASIS_SOIL_TEST_BARS = 500
METASTASIS_COLONIZE_WR = 0.55
METASTASIS_MAX_SYMBOLS = 5
TELOMERASE_PROMOTION_WR = 0.65
TELOMERASE_PROMOTION_PF = 1.5
TELOMERASE_MIN_TRADES = 30
IMMUNE_CHECKPOINT_WR = 0.58
IMMUNE_CHECKPOINT_PF = 1.2
IMMUNE_CHECKPOINT_BARS = 300


# ============================================================
# ENUMS
# ============================================================

class MutationType(Enum):
    ONCOGENE_ACTIVATION = auto()     # RAS/MYC/HER2: amplify signal weight
    SUPPRESSOR_DELETION = auto()     # p53/RB/BRCA: remove a gate/filter
    METABOLIC_SHIFT = auto()         # Warburg: speed vs precision tradeoff
    RECEPTOR_MUTATION = auto()       # HER2: alter signal sensitivity
    TELOMERE_EXTENSION = auto()      # TERT: remove strategy expiry


class CancerPhase(Enum):
    MITOSIS = auto()
    G1_CHECKPOINT_BYPASS = auto()
    S_PHASE = auto()
    G2_CHECKPOINT_BYPASS = auto()
    M_PHASE = auto()
    METASTASIS = auto()
    ANGIOGENESIS = auto()
    IMMUNE_EVASION = auto()
    APOPTOSIS = auto()


class InfectionPhase(Enum):
    HEALTHY = auto()
    ACUTE = auto()
    CHRONIC = auto()
    CRITICAL = auto()


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class StrategyMutant:
    """A mutated strategy variant -- a single cancer cell."""
    mutant_id: str = ""
    parent_id: str = ""
    parent_symbol: str = ""
    generation: int = 0
    # Mutated parameters
    te_weight_deltas: Dict[str, float] = field(default_factory=dict)
    confidence_delta: float = 0.0
    regime_sensitivity: float = 1.0
    hold_time_mult: float = 1.0
    sl_ratio: float = 1.0
    tp_ratio: float = 1.0
    # Mutation metadata
    mutation_types: List[str] = field(default_factory=list)
    driver_count: int = 0
    # Fitness (filled after evaluation)
    fitness_score: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    # Quantum features
    quantum_entropy: float = 0.0
    quantum_dominant: float = 0.0
    quantum_significant: int = 0
    quantum_variance: float = 0.0
    # Cancer biology state
    telomere_length: float = TELOMERE_L0
    divisions: int = 0
    is_alive: bool = True
    has_telomerase: bool = False
    metastasized_to: List[str] = field(default_factory=list)
    vegf_signal: float = 0.0
    # Gompertzian growth state
    gompertz_N: float = 1.0   # Current "tumor size" (strategy population)
    # Lotka-Volterra state
    lv_immune_pressure: float = 0.0
    # Timestamps
    created_at: str = ""
    promoted_at: str = ""

    def __post_init__(self):
        if not self.mutant_id:
            self.mutant_id = str(uuid.uuid4())[:16]
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class TumorCluster:
    """A group of related mutants (clonal population)."""
    cluster_id: str = ""
    parent_symbol: str = ""
    mutants: List[StrategyMutant] = field(default_factory=list)
    avg_fitness: float = 0.0
    resource_budget: int = 3
    generation: int = 0
    vegf_concentration: float = 0.0
    blood_supply: float = 0.5
    carrying_capacity: float = GOMPERTZ_K

    def __post_init__(self):
        if not self.cluster_id:
            self.cluster_id = str(uuid.uuid4())[:12]


@dataclass
class SimulationContext:
    """Controls which defenses are active during simulation."""
    crispr_enabled: bool = True
    toxoplasma_enabled: bool = True
    regime_detection: bool = True
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    max_loss_dollars: float = MAX_LOSS_DOLLARS


# ============================================================
# MATHEMATICAL MODELS (real cancer kinetics)
# ============================================================

class CancerKinetics:
    """
    Real mathematical models from cancer biology, adapted for
    strategy mutation and evolution.
    """

    # ----- GOMPERTZIAN GROWTH -----
    @staticmethod
    def gompertz_growth(N: float, K: float, alpha: float, dt: float = 1.0) -> float:
        """
        Gompertzian tumor growth.

            dN/dt = alpha * N * ln(K / N)
            N(t+dt) = K * exp(ln(N/K) * exp(-alpha * dt))

        Early growth is exponential, then decelerates as N -> K.
        Models strategy population saturation.

        Args:
            N: Current population (strategy count or fitness mass)
            K: Carrying capacity (market capacity for this edge)
            alpha: Growth rate parameter
            dt: Time step

        Returns:
            New population size N(t+dt)
        """
        if N <= 0 or K <= 0:
            return N
        ratio = N / K
        if ratio >= 1.0:
            return K  # At carrying capacity
        return K * math.exp(math.log(ratio) * math.exp(-alpha * dt))

    # ----- KNUDSON TWO-HIT -----
    @staticmethod
    def knudson_two_hit_probability(mu: float, t: float) -> float:
        """
        Probability of BOTH alleles being inactivated (sporadic cancer).

            P(tumor) = 1 - exp(-mu * t)^2
                     ≈ mu^2 * t^2  for small mu*t

        For strategy: probability that BOTH safety layers fail simultaneously.

        Args:
            mu: Mutation rate per allele per unit time
            t: Time (generations or trades)

        Returns:
            Probability of two-hit event
        """
        single_hit = 1.0 - math.exp(-mu * t)
        return single_hit * single_hit  # Both alleles

    # ----- VOGELSTEIN MULTI-HIT (Poisson) -----
    @staticmethod
    def vogelstein_nhit_probability(mu: float, t: float, n: int) -> float:
        """
        Probability of accumulating exactly N driver mutations by time t.
        Poisson model:

            P(k=N) = (mu*t)^N / N! * exp(-mu*t)

        For strategy: probability of N simultaneous parameter mutations
        producing a breakthrough combination.

        Args:
            mu: Mutation rate per driver gene per division
            t: Number of divisions/generations
            n: Number of required driver mutations

        Returns:
            Probability of exactly N hits
        """
        lam = mu * t
        # Use log to avoid overflow: log(P) = N*log(lam) - log(N!) - lam
        log_p = n * math.log(max(lam, 1e-30)) - math.lgamma(n + 1) - lam
        return math.exp(max(log_p, -50))  # Clamp to avoid underflow

    # ----- MICHAELIS-MENTEN KINETICS -----
    @staticmethod
    def michaelis_menten(substrate: float, vmax: float = MM_VMAX,
                         km: float = MM_KM) -> float:
        """
        Michaelis-Menten enzyme kinetics.

            v = Vmax * [S] / (Km + [S])

        Models TE activation rate as enzyme-substrate reaction.
        [S] = signal strength, v = activation rate.
        Low Km = high sensitivity (hair-trigger).
        High Km = requires strong signals.

        Args:
            substrate: Signal strength [S] (0 to infinity)
            vmax: Maximum reaction rate
            km: Michaelis constant (concentration at half-max)

        Returns:
            Reaction velocity v (0 to Vmax)
        """
        return vmax * substrate / (km + substrate + 1e-10)

    # ----- LOTKA-VOLTERRA (Strategy vs Market) -----
    @staticmethod
    def lotka_volterra_step(T: float, I: float, K_T: float = 100.0,
                             K_I: float = 100.0, dt: float = 1.0) -> Tuple[float, float]:
        """
        Lotka-Volterra competition between strategy (T) and market immune
        response (I).

            dT/dt = r*T*(1 - T/K_T) - alpha*T*I
            dI/dt = s*I*(1 - I/K_I) + beta*T*I - gamma*I

        Models edge decay as market adapts. Predator-prey oscillations
        predict when to rotate strategies.

        Args:
            T: Strategy population (fitness or trade count)
            I: Market immune response (HFT, arbitrageurs adapting)
            K_T: Strategy carrying capacity
            K_I: Immune carrying capacity
            dt: Time step

        Returns:
            (T_new, I_new) after one time step
        """
        dT = (LV_R * T * (1 - T / K_T) - LV_ALPHA * T * I) * dt
        dI = (LV_S * I * (1 - I / K_I) + LV_BETA * T * I - LV_GAMMA * I) * dt
        T_new = max(0.0, T + dT)
        I_new = max(0.0, I + dI)
        return T_new, I_new

    # ----- FISHER-KPP (Metastasis Invasion Speed) -----
    @staticmethod
    def fisher_invasion_speed(D: float = FISHER_D,
                               r: float = FISHER_R) -> float:
        """
        Fisher-KPP traveling wave speed for tumor invasion.

            c = 2 * sqrt(D * r)

        D = diffusion coefficient (strategy adaptability to new markets)
        r = local growth rate (profitability in new regime)

        Higher D = strategy adapts faster to new symbols.
        Higher r = strategy is more profitable locally.

        Args:
            D: Diffusion coefficient
            r: Growth rate

        Returns:
            Invasion wave speed c
        """
        return 2.0 * math.sqrt(D * r)

    # ----- STOCHASTIC BIRTH-DEATH -----
    @staticmethod
    def birth_death_extinction_prob(lam: float = BD_LAMBDA,
                                     mu: float = BD_MU) -> float:
        """
        Extinction probability for a birth-death process.

            delta = mu / lambda   (if lambda > mu)
            delta = 1.0           (if lambda <= mu)

        For strategy: lam = win rate, mu = loss rate.
        Strategies with delta close to 1.0 are fragile.

        Args:
            lam: Birth rate (wins per unit time)
            mu: Death rate (losses per unit time)

        Returns:
            Extinction probability delta (0 to 1)
        """
        if lam <= mu:
            return 1.0
        return mu / lam

    @staticmethod
    def birth_death_expected_population(N0: float, lam: float, mu: float,
                                         t: float) -> float:
        """
        Expected population at time t.

            E[N(t)] = N0 * exp((lambda - mu) * t)

        Args:
            N0: Initial population
            lam: Birth rate
            mu: Death rate
            t: Time

        Returns:
            Expected population
        """
        return N0 * math.exp((lam - mu) * t)

    # ----- WARBURG METABOLIC FLUX -----
    @staticmethod
    def warburg_efficiency(speed_preference: float) -> Tuple[float, float]:
        """
        Warburg metabolic balance between glycolysis (fast/low yield)
        and oxidative phosphorylation (slow/high yield).

            Total ATP flux = glyc_rate * glyc_yield * w + oxphos_rate * oxphos_yield * (1-w)

        speed_preference (0 to 1): 0 = full OXPHOS, 1 = full glycolysis

        Returns:
            (throughput, efficiency) tuple
            throughput = total evaluations per unit time
            efficiency = yield per evaluation
        """
        w = max(0.0, min(1.0, speed_preference))
        throughput = (WARBURG_GLYCOLYSIS_RATE * w
                      + WARBURG_OXPHOS_RATE * (1 - w))
        total_yield = (WARBURG_GLYCOLYSIS_YIELD * WARBURG_GLYCOLYSIS_RATE * w
                       + WARBURG_OXPHOS_YIELD * WARBURG_OXPHOS_RATE * (1 - w))
        efficiency = total_yield / (throughput + 1e-10)
        return throughput, efficiency

    # ----- TELOMERE SHORTENING -----
    @staticmethod
    def telomere_after_division(L_current: float, n_divisions: int = 1,
                                 delta: float = TELOMERE_DELTA) -> float:
        """
        Telomere length after n divisions.

            L(n) = L_current - delta * n

        Args:
            L_current: Current telomere length (bp)
            n_divisions: Number of new divisions
            delta: Shortening per division (bp)

        Returns:
            New telomere length
        """
        return max(0.0, L_current - delta * n_divisions)

    @staticmethod
    def hayflick_remaining(L_current: float,
                            L_critical: float = TELOMERE_L_CRITICAL,
                            delta: float = TELOMERE_DELTA) -> int:
        """
        Remaining divisions before senescence (Hayflick limit).

            n_remaining = (L_current - L_critical) / delta

        Args:
            L_current: Current telomere length
            L_critical: Critical length triggering senescence
            delta: Shortening per division

        Returns:
            Number of remaining divisions (0 if at limit)
        """
        if L_current <= L_critical:
            return 0
        return int((L_current - L_critical) / delta)

    @staticmethod
    def senescence_probability(n_divisions: int,
                                n_max: int = TELOMERE_HAYFLICK,
                                k: float = 7.0) -> float:
        """
        Probability of entering senescence after n divisions.

            P(senescence | n) = 1 - exp(-(n/n_max)^k)

        Sharp transition near n_max (k controls sharpness).

        Args:
            n_divisions: Current division count
            n_max: Hayflick limit
            k: Sharpness parameter

        Returns:
            Probability of senescence (0 to 1)
        """
        if n_max <= 0:
            return 1.0
        ratio = n_divisions / n_max
        return 1.0 - math.exp(-ratio ** k)

    # ----- VEGF ANGIOGENESIS -----
    @staticmethod
    def vegf_production(tumor_density: float,
                        T_critical: float = 1e7) -> float:
        """
        VEGF production rate based on tumor density.

            k_prod * T * H(T - T_crit)

        Uses Heaviside function: VEGF only produced above critical density
        (tumor needs blood supply once it outgrows diffusion limits).

        Args:
            tumor_density: Number of cells (or strategy fitness mass)
            T_critical: Critical density for hypoxia

        Returns:
            VEGF production rate
        """
        if tumor_density < T_critical:
            return 0.0
        return VEGF_K_PROD * tumor_density

    @staticmethod
    def angiogenesis_active(vegf_concentration: float) -> bool:
        """
        Check if angiogenic switch is on.

            V > V_threshold

        Args:
            vegf_concentration: Current VEGF level

        Returns:
            True if blood vessel sprouting is active
        """
        return vegf_concentration > VEGF_THRESHOLD

    @staticmethod
    def angiogenesis_rate(vegf: float, km: float = 0.2) -> float:
        """
        Rate of new vessel formation (Michaelis-Menten).

            Rate = Vmax * V / (Km + V)

        Args:
            vegf: VEGF concentration
            km: Michaelis constant for angiogenesis

        Returns:
            Angiogenesis rate (0 to 1)
        """
        return vegf / (km + vegf + 1e-10)


# ============================================================
# DATABASE
# ============================================================

class CancerCellDB:
    """SQLite persistence for cancer cell simulation."""

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS mutants (
                    mutant_id TEXT PRIMARY KEY,
                    parent_id TEXT,
                    parent_symbol TEXT,
                    generation INTEGER,
                    te_weight_deltas TEXT,
                    confidence_delta REAL,
                    regime_sensitivity REAL,
                    hold_time_mult REAL,
                    sl_ratio REAL,
                    tp_ratio REAL,
                    mutation_types TEXT,
                    driver_count INTEGER,
                    fitness_score REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    quantum_entropy REAL,
                    quantum_dominant REAL,
                    quantum_significant INTEGER,
                    quantum_variance REAL,
                    telomere_length REAL,
                    divisions INTEGER,
                    is_alive INTEGER,
                    has_telomerase INTEGER,
                    metastasized_to TEXT,
                    gompertz_N REAL,
                    lv_immune_pressure REAL,
                    created_at TEXT,
                    promoted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS clusters (
                    cluster_id TEXT PRIMARY KEY,
                    parent_symbol TEXT,
                    avg_fitness REAL,
                    resource_budget INTEGER,
                    generation INTEGER,
                    vegf_concentration REAL,
                    blood_supply REAL,
                    mutant_count INTEGER,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS metastasis_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mutant_id TEXT,
                    source_symbol TEXT,
                    target_symbol TEXT,
                    soil_wr REAL,
                    soil_pf REAL,
                    soil_trades INTEGER,
                    invasion_speed REAL,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS promotion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mutant_id TEXT,
                    parent_id TEXT,
                    symbol TEXT,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    total_trades INTEGER,
                    metastasized_to TEXT,
                    driver_count INTEGER,
                    mutation_types TEXT,
                    telomere_length REAL,
                    gompertz_K REAL,
                    promoted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS immune_checkpoint (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mutant_id TEXT,
                    symbol TEXT,
                    checkpoint_wr REAL,
                    checkpoint_pf REAL,
                    checkpoint_trades INTEGER,
                    passed INTEGER,
                    defenses_active TEXT,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS run_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_start TEXT,
                    run_end TEXT,
                    symbols TEXT,
                    total_mutants_generated INTEGER,
                    total_survivors INTEGER,
                    promoted_to_live INTEGER,
                    avg_fitness REAL,
                    avg_win_rate REAL,
                    metastasis_count INTEGER,
                    gompertz_growth_factor REAL,
                    warburg_throughput REAL,
                    telomere_avg REAL
                );
            """)

    def save_mutant(self, m: StrategyMutant):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO mutants VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
            """, (
                m.mutant_id, m.parent_id, m.parent_symbol, m.generation,
                json.dumps(m.te_weight_deltas), m.confidence_delta,
                m.regime_sensitivity, m.hold_time_mult, m.sl_ratio, m.tp_ratio,
                json.dumps(m.mutation_types), m.driver_count,
                m.fitness_score, m.win_rate, m.profit_factor, m.total_trades,
                m.sharpe_ratio, m.max_drawdown,
                m.quantum_entropy, m.quantum_dominant,
                m.quantum_significant, m.quantum_variance,
                m.telomere_length, m.divisions,
                int(m.is_alive), int(m.has_telomerase),
                json.dumps(m.metastasized_to),
                m.gompertz_N, m.lv_immune_pressure,
                m.created_at, m.promoted_at,
            ))

    def save_run(self, run_data: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO run_history (
                    run_start, run_end, symbols,
                    total_mutants_generated, total_survivors, promoted_to_live,
                    avg_fitness, avg_win_rate, metastasis_count,
                    gompertz_growth_factor, warburg_throughput, telomere_avg
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_data.get("run_start", ""),
                run_data.get("run_end", ""),
                json.dumps(run_data.get("symbols", [])),
                run_data.get("total_mutants", 0),
                run_data.get("survivors", 0),
                run_data.get("promoted", 0),
                run_data.get("avg_fitness", 0),
                run_data.get("avg_wr", 0),
                run_data.get("metastasis_count", 0),
                run_data.get("gompertz_factor", 0),
                run_data.get("warburg_throughput", 0),
                run_data.get("telomere_avg", 0),
            ))

    def log_metastasis(self, entry: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metastasis_log (
                    mutant_id, source_symbol, target_symbol,
                    soil_wr, soil_pf, soil_trades, invasion_speed, timestamp
                ) VALUES (?,?,?,?,?,?,?,?)
            """, (
                entry["mutant_id"], entry["source"], entry["target"],
                entry["wr"], entry["pf"], entry["trades"],
                entry.get("invasion_speed", 0), entry["timestamp"],
            ))

    def log_promotion(self, mutant: StrategyMutant, symbol: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO promotion_log (
                    mutant_id, parent_id, symbol, win_rate, profit_factor,
                    sharpe_ratio, total_trades, metastasized_to,
                    driver_count, mutation_types, telomere_length,
                    gompertz_K, promoted_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                mutant.mutant_id, mutant.parent_id, symbol,
                mutant.win_rate, mutant.profit_factor,
                mutant.sharpe_ratio, mutant.total_trades,
                json.dumps(mutant.metastasized_to),
                mutant.driver_count, json.dumps(mutant.mutation_types),
                mutant.telomere_length, GOMPERTZ_K,
                mutant.promoted_at,
            ))

    def get_promoted_mutants(self, limit: int = 50) -> List[dict]:
        """Get recently promoted mutants for BRAIN scripts to read."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM promotion_log
                ORDER BY promoted_at DESC LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]


# ============================================================
# SIMULATION ENGINE
# ============================================================

class StrategySimulator:
    """
    Simulates strategy performance on historical bars.
    Two modes:
      - Fallback: simplified Michaelis-Menten price-action model
      - LSTM:     real neural network inference with mutated feature scaling
    """

    def __init__(self):
        self.kinetics = CancerKinetics()
        # Cache prepared features per bars id to avoid recomputing
        self._feature_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _get_features_and_prices(self, bars: np.ndarray):
        """Prepare and cache 8-feature sequences + prices from OHLCV bars."""
        bars_id = id(bars)
        if bars_id in self._feature_cache:
            return self._feature_cache[bars_id]

        df = prepare_features_from_ohlcv(bars)
        if df is None:
            return None, None

        feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper',
                        'bb_lower', 'momentum', 'roc', 'atr']
        features = df[feature_cols].values
        prices = df['close'].values

        # Pre-build sequences: (N, SEQ_LENGTH, 8)
        n_seq = len(features) - SEQ_LENGTH - PREDICTION_HORIZON
        if n_seq < 10:
            return None, None

        X = np.lib.stride_tricks.sliding_window_view(
            features, (SEQ_LENGTH, 8)
        )[:n_seq, 0, :, :]  # shape: (n_seq, 30, 8)
        P = prices[SEQ_LENGTH:SEQ_LENGTH + n_seq]

        self._feature_cache[bars_id] = (X, P)
        return X, P

    def simulate(self, mutant: StrategyMutant, bars: np.ndarray,
                 context: SimulationContext,
                 lstm_model=None) -> dict:
        """
        Run a mutant strategy through historical bars.

        If lstm_model is provided, uses real LSTM inference with the
        mutant's TE weight deltas applied as feature scaling (the mutation
        alters what the neural network "sees"). Otherwise falls back to
        simplified price-action model.

        Returns dict with win_rate, profit_factor, total_trades,
        sharpe_ratio, max_drawdown.
        """
        if lstm_model is not None and GPU_AVAILABLE:
            return self._simulate_lstm(mutant, bars, context, lstm_model)
        return self._simulate_fallback(mutant, bars, context)

    def _simulate_lstm(self, mutant: StrategyMutant, bars: np.ndarray,
                       context: SimulationContext, model) -> dict:
        """
        LSTM inference simulation.

        The mutant's mutations modify the neural network's perception:
        - te_weight_deltas scale input features (oncogene = amplify signal)
        - confidence_delta shifts the prediction threshold
        - hold_time_mult adjusts how long to hold
        - sl_ratio / tp_ratio adjust risk/reward
        """
        result = self._get_features_and_prices(bars)
        if result[0] is None:
            return {"win_rate": 0, "profit_factor": 0, "total_trades": 0,
                    "sharpe_ratio": 0, "max_drawdown": 0}

        X_base, prices = result

        # --- MUTATION: Scale features by TE weight deltas ---
        # Feature order: RSI, MACD, MACD_SIG, BB_UPPER, BB_LOWER, MOMENTUM, ROC, ATR
        feature_keys = ["RSI", "MACD", "MACD_SIG", "BB_UPPER",
                        "BB_LOWER", "MOMENTUM", "ROC", "ATR"]
        scales = np.ones(8, dtype=np.float32)
        for i, key in enumerate(feature_keys):
            delta = mutant.te_weight_deltas.get(key, 0.0)
            # Scale factor: 1.0 + delta (oncogene amplifies, suppressor dampens)
            scales[i] = max(0.1, 1.0 + delta * 0.5)

        # Apply scales to all sequences at once
        X_mutated = X_base * scales[np.newaxis, np.newaxis, :]

        # --- LSTM INFERENCE (batched) ---
        X_tensor = torch.FloatTensor(X_mutated)
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)  # 0=HOLD, 1=BUY, 2=SELL
            confidences = np.max(probs, axis=1)

        # --- TRADING SIMULATION ---
        eff_threshold = max(context.confidence_threshold + mutant.confidence_delta,
                            BYPASS_CONFIDENCE_FLOOR)
        hold_bars = max(1, int(PREDICTION_HORIZON * mutant.hold_time_mult))
        sl_dollars = context.max_loss_dollars * mutant.sl_ratio
        tp_dollars = sl_dollars * mutant.tp_ratio * 3.0

        trades = []
        equity = [0.0]
        peak_equity = 0.0
        max_dd = 0.0

        i = 0
        while i < len(preds) - hold_bars:
            pred = preds[i]
            conf = confidences[i]

            # Skip HOLD or low confidence
            if pred == 0 or conf < eff_threshold:
                i += 1
                continue

            current_price = prices[i]
            future_price = prices[min(i + hold_bars, len(prices) - 1)]

            if pred == 1:  # BUY
                pnl_raw = future_price - current_price
            else:  # SELL
                pnl_raw = current_price - future_price

            # Normalize to dollar PnL (rough: pnl as fraction of price * $1000 position)
            pnl = pnl_raw / (current_price + 1e-10) * 1000.0

            # Apply SL/TP caps
            if pnl < -sl_dollars:
                pnl = -sl_dollars
            elif pnl > tp_dollars:
                pnl = tp_dollars

            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            peak_equity = max(peak_equity, equity[-1])
            dd = peak_equity - equity[-1]
            max_dd = max(max_dd, dd)

            i += hold_bars  # Skip ahead past hold

        if len(trades) < 3:
            return {"win_rate": 0, "profit_factor": 0, "total_trades": 0,
                    "sharpe_ratio": 0, "max_drawdown": 0}

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades)
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        trade_arr = np.array(trades)
        sharpe = (np.mean(trade_arr) / (np.std(trade_arr) + 1e-10)) * math.sqrt(252)

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }

    def _simulate_fallback(self, mutant: StrategyMutant, bars: np.ndarray,
                           context: SimulationContext) -> dict:
        """Original simplified price-action simulation (no LSTM)."""
        if len(bars) < 60:
            return {"win_rate": 0, "profit_factor": 0, "total_trades": 0,
                    "sharpe_ratio": 0, "max_drawdown": 0}

        closes = bars[:, 3] if bars.ndim == 2 else bars
        returns = np.diff(closes) / (closes[:-1] + 1e-10)

        trades = []
        equity = [0.0]
        peak_equity = 0.0
        max_dd = 0.0

        eff_threshold = context.confidence_threshold + mutant.confidence_delta
        km_adjusted = MM_KM * mutant.regime_sensitivity

        i = 50
        while i < len(returns) - 1:
            window = returns[max(0, i - 20):i]
            signal_strength = abs(np.mean(window)) * 100

            activation = self.kinetics.michaelis_menten(
                signal_strength, vmax=MM_VMAX, km=km_adjusted
            )

            te_modifier = 1.0 + sum(mutant.te_weight_deltas.values()) * 0.01
            confidence = activation * te_modifier

            if confidence < max(eff_threshold, BYPASS_CONFIDENCE_FLOOR):
                i += 1
                continue

            direction = 1 if np.mean(window) > 0 else -1
            hold_bars = max(1, int(5 * mutant.hold_time_mult))
            if i + hold_bars >= len(returns):
                break

            future_return = np.sum(returns[i:i + hold_bars]) * direction

            sl_dollars = context.max_loss_dollars * mutant.sl_ratio
            tp_dollars = sl_dollars * mutant.tp_ratio * 3.0

            pnl = future_return * 1000
            if pnl < -sl_dollars:
                pnl = -sl_dollars
            elif pnl > tp_dollars:
                pnl = tp_dollars

            trades.append(pnl)
            equity.append(equity[-1] + pnl)
            peak_equity = max(peak_equity, equity[-1])
            dd = peak_equity - equity[-1]
            max_dd = max(max_dd, dd)

            i += hold_bars

        if len(trades) < 3:
            return {"win_rate": 0, "profit_factor": 0, "total_trades": 0,
                    "sharpe_ratio": 0, "max_drawdown": 0}

        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / len(trades)
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        trade_arr = np.array(trades)
        sharpe = (np.mean(trade_arr) / (np.std(trade_arr) + 1e-10)) * math.sqrt(252)

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }


# ============================================================
# CANCER CELL ENGINE
# ============================================================

class CancerCellEngine:
    """
    Full cancer simulation engine with real mathematical oncology.

    Phases:
        1. Mitosis (Vogelstein multi-hit mutation)
        2. Tumor Suppressor Bypass (checkpoint removal in sim)
        3. Warburg Acceleration (GPU batch quantum processing)
        4. Angiogenesis (VEGF-driven resource allocation)
        5. Metastasis (Fisher-KPP cross-symbol invasion)
        6. Telomerase (Hayflick limit bypass for winners)
        7. Immune Checkpoint (re-enable all defenses)
    """

    def __init__(self, te_families: Optional[List] = None):
        self.db = CancerCellDB()
        self.kinetics = CancerKinetics()
        self.simulator = StrategySimulator()
        self.te_families = te_families or []
        # Model cache: parent_id -> loaded LSTM model
        self.expert_models: Dict[str, Any] = {}

        # Try to load TE families from TEQA
        if not self.te_families:
            try:
                from teqa_v3_neural_te import ALL_TE_FAMILIES
                self.te_families = ALL_TE_FAMILIES
            except ImportError:
                log.warning("CANCER: Could not import TE families, "
                            "using empty list")

        log.info("CANCER CELL ENGINE v%s initialized", VERSION)
        log.info("  GPU available: %s", GPU_AVAILABLE)
        log.info("  Qiskit available: %s", QISKIT_AVAILABLE)
        log.info("  TE families: %d", len(self.te_families))
        log.info("  MAX_LOSS_DOLLARS: $%.2f (sacred)", MAX_LOSS_DOLLARS)

    # ---- PHASE 1: MITOSIS ----

    def mitosis(self, parent_strategies: List[dict], symbol: str,
                bars: np.ndarray) -> List[StrategyMutant]:
        """
        Phase 1: Oncogene Activation -- Rapid Strategy Mutation.

        Uses Vogelstein's multi-hit model to determine mutation count.
        Each daughter cell gets 3-7 simultaneous driver mutations.
        Mutation types follow real oncogene/suppressor biology.
        """
        log.info("CANCER [%s]: MITOSIS -- generating mutant population", symbol)

        # Select top parent strategies
        parents = sorted(parent_strategies,
                         key=lambda p: p.get("fitness", 0), reverse=True)
        parents = parents[:MITOSIS_PARENT_COUNT]

        all_mutants = []
        te_names = [f.name for f in self.te_families] if self.te_families else []

        for parent in parents:
            for _ in range(MITOSIS_POPULATION_SIZE):

                # Vogelstein multi-hit: determine driver count using Poisson
                # Higher generations = more accumulated mutations
                gen = parent.get("generation", 0) + 1
                n_drivers = random.randint(VOGELSTEIN_MIN_DRIVERS,
                                           VOGELSTEIN_MAX_DRIVERS)

                # Calculate probability this combination produces a "cancer"
                # (a breakthrough strategy)
                p_breakthrough = self.kinetics.vogelstein_nhit_probability(
                    mu=VOGELSTEIN_MU, t=gen * 1000, n=n_drivers
                )

                # Select mutation types
                mutation_types = random.sample(
                    list(MutationType),
                    min(n_drivers, len(MutationType))
                )

                mutant = StrategyMutant(
                    parent_id=parent.get("id", "seed"),
                    parent_symbol=symbol,
                    generation=gen,
                    driver_count=n_drivers,
                    mutation_types=[m.name for m in mutation_types],
                    telomere_length=self.kinetics.telomere_after_division(
                        TELOMERE_L0, n_divisions=gen
                    ),
                    divisions=gen,
                )

                # Inherit TE weight signature from expert parent
                parent_sig = parent.get("te_weight_signature", {})
                if parent_sig:
                    for key, val in parent_sig.items():
                        if not key.startswith("_"):
                            # Start from expert's learned weights + noise
                            mutant.te_weight_deltas[key] = (
                                val + random.gauss(0, 0.05)
                            )

                # Apply each mutation type
                for mt in mutation_types:

                    if mt == MutationType.ONCOGENE_ACTIVATION and te_names:
                        # RAS/MYC: amplify a TE weight (gain-of-function)
                        te = random.choice(te_names)
                        # Michaelis-Menten: mutation strength scales with
                        # substrate (signal history)
                        substrate = random.uniform(0.1, 2.0)
                        delta = self.kinetics.michaelis_menten(
                            substrate, vmax=0.3, km=0.5
                        )
                        mutant.te_weight_deltas[te] = (
                            mutant.te_weight_deltas.get(te, 0) + delta
                        )

                    elif mt == MutationType.SUPPRESSOR_DELETION:
                        # p53/RB: reduce confidence threshold or regime gate
                        # Knudson two-hit: probability both alleles lost
                        p_both_lost = self.kinetics.knudson_two_hit_probability(
                            mu=KNUDSON_MU, t=gen * 365
                        )
                        if random.random() < max(p_both_lost, 0.3):
                            mutant.confidence_delta -= random.uniform(0.02, 0.10)
                            mutant.regime_sensitivity *= random.uniform(0.3, 0.8)

                    elif mt == MutationType.METABOLIC_SHIFT:
                        # Warburg: trade precision for speed
                        throughput, efficiency = self.kinetics.warburg_efficiency(
                            speed_preference=random.uniform(0.5, 1.0)
                        )
                        mutant.hold_time_mult *= random.uniform(0.5, 1.5)
                        mutant.tp_ratio *= random.uniform(0.8, 1.3)

                    elif mt == MutationType.RECEPTOR_MUTATION and te_names:
                        # HER2: alter TE sensitivity dramatically
                        te = random.choice(te_names)
                        mutant.te_weight_deltas[te] = (
                            mutant.te_weight_deltas.get(te, 0)
                            * random.uniform(-0.5, 2.5)
                        )

                    elif mt == MutationType.TELOMERE_EXTENSION:
                        # TERT: candidate for immortality (checked later)
                        mutant.telomere_length += random.uniform(1000, 3000)

                all_mutants.append(mutant)

        log.info("CANCER [%s]: Mitosis complete -- %d mutants from %d parents, "
                 "%d-%d drivers each",
                 symbol, len(all_mutants), len(parents),
                 VOGELSTEIN_MIN_DRIVERS, VOGELSTEIN_MAX_DRIVERS)

        return all_mutants

    # ---- PHASE 2: TUMOR SUPPRESSOR BYPASS ----

    def bypass_checkpoints(self, mutants: List[StrategyMutant],
                           symbol: str, bars: np.ndarray) -> List[StrategyMutant]:
        """
        Phase 2: Tumor Suppressor Bypass.

        Disable CRISPR, Toxoplasma, and regime detection IN SIMULATION.
        SL remains sacred. Time-limited to BYPASS_MAX_DURATION_SEC.
        """
        log.info("CANCER [%s]: TUMOR SUPPRESSOR BYPASS -- "
                 "simulating %d mutants without defenses", symbol, len(mutants))

        start = time.time()
        sim_context = SimulationContext(
            crispr_enabled=False,
            toxoplasma_enabled=False,
            regime_detection=False,
            confidence_threshold=BYPASS_CONFIDENCE_FLOOR,
            max_loss_dollars=MAX_LOSS_DOLLARS,  # SACRED
        )

        evaluated = 0
        for mutant in mutants:
            if time.time() - start > BYPASS_MAX_DURATION_SEC:
                log.warning("CANCER: Bypass time limit reached after %d mutants",
                            evaluated)
                break

            # Look up LSTM model for this mutant's parent
            lstm_model = self.expert_models.get(mutant.parent_id)
            result = self.simulator.simulate(mutant, bars, sim_context,
                                             lstm_model=lstm_model)
            mutant.win_rate = result["win_rate"]
            mutant.profit_factor = result["profit_factor"]
            mutant.total_trades = result["total_trades"]
            mutant.sharpe_ratio = result["sharpe_ratio"]
            mutant.max_drawdown = result["max_drawdown"]

            # Composite fitness incorporating cancer kinetics
            if result["total_trades"] > 0:
                # Birth-death extinction probability
                lam = max(result["win_rate"], 0.01)
                mu_bd = max(1.0 - result["win_rate"], 0.01)
                extinction_p = self.kinetics.birth_death_extinction_prob(lam, mu_bd)

                # Gompertzian: model fitness as tumor growth
                mutant.gompertz_N = self.kinetics.gompertz_growth(
                    N=max(mutant.gompertz_N, 1.0),
                    K=100.0,
                    alpha=result["profit_factor"] * 0.1,
                    dt=result["total_trades"] / 10.0
                )

                mutant.fitness_score = (
                    result["profit_factor"] * 0.25
                    + result["sharpe_ratio"] * 0.20
                    + result["win_rate"] * 0.20
                    + (1.0 - result["max_drawdown"]) * 0.15
                    + (1.0 - extinction_p) * 0.10         # Birth-death survival
                    + (mutant.gompertz_N / 100.0) * 0.10  # Gompertzian growth
                )

            evaluated += 1

        log.info("CANCER [%s]: Bypass complete -- %d/%d mutants evaluated in %.1fs",
                 symbol, evaluated, len(mutants), time.time() - start)

        return mutants

    # ---- PHASE 3: WARBURG ACCELERATION ----

    def warburg_quantum_batch(self, mutants: List[StrategyMutant],
                              bars: np.ndarray) -> List[StrategyMutant]:
        """
        Phase 3: Warburg Effect -- GPU Batch Quantum Processing.

        Run quantum circuits at reduced precision (2048 shots vs 8192)
        for maximum throughput. Glycolysis over OXPHOS.
        """
        if not QISKIT_AVAILABLE:
            log.warning("CANCER: Qiskit not available, skipping Warburg phase")
            return mutants

        # Cap qubits for Warburg fast mode -- 33 qubits needs 128GB RAM
        # for statevector. Use 16 qubits (2^16 = 65K states) which fits
        # comfortably in memory and runs fast. Warburg = speed over precision.
        n_qubits = min(len(self.te_families) if self.te_families else 8, 16)

        log.info("CANCER: WARBURG ACCELERATION -- "
                 "batch quantum processing %d mutants, "
                 "%d qubits, %d shots (fast mode)",
                 len(mutants), n_qubits, WARBURG_SHOTS_FAST)

        throughput, efficiency = self.kinetics.warburg_efficiency(
            speed_preference=0.85  # Heavy glycolysis (speed mode)
        )
        log.info("CANCER: Warburg metabolic flux -- "
                 "throughput=%.1f, efficiency=%.2f", throughput, efficiency)

        sim = AerSimulator()
        processed = 0

        for batch_start in range(0, len(mutants), WARBURG_BATCH_SIZE):
            batch = mutants[batch_start:batch_start + WARBURG_BATCH_SIZE]

            for mutant in batch:
                # Build feature vector from TE weight deltas
                features = np.zeros(n_qubits)
                for i, te in enumerate(self.te_families[:n_qubits]):
                    base = random.uniform(0.1, 0.9)  # Base activation
                    delta = mutant.te_weight_deltas.get(te.name, 0.0)
                    features[i] = max(0.0, min(1.0, base + delta))

                # Normalize to [0, pi] for RY rotations
                angles = features * math.pi

                # Build quantum circuit
                qc = QuantumCircuit(n_qubits, n_qubits)
                for i in range(n_qubits):
                    qc.ry(float(angles[i]), i)

                # CZ entanglement ring
                for i in range(n_qubits - 1):
                    qc.cz(i, i + 1)
                qc.cz(n_qubits - 1, 0)

                qc.measure(range(n_qubits), range(n_qubits))

                # Run with reduced shots (Warburg: speed over precision)
                job = sim.run(qc, shots=WARBURG_SHOTS_FAST)
                counts = job.result().get_counts()

                # Convert to probability distribution
                n_states = 2 ** n_qubits
                probs = np.zeros(min(n_states, 2 ** 20))  # Cap for memory
                for state_str, cnt in counts.items():
                    idx = int(state_str.replace(" ", ""), 2)
                    if idx < len(probs):
                        probs[idx] = cnt / WARBURG_SHOTS_FAST

                # Extract quantum features
                nonzero = probs[probs > 0]
                entropy = -np.sum(nonzero * np.log2(nonzero)) if len(nonzero) > 0 else 0
                dominant = float(probs.max()) if len(probs) > 0 else 0
                significant = int(np.sum(probs > 0.03))
                variance = float(np.var(probs))

                mutant.quantum_entropy = entropy
                mutant.quantum_dominant = dominant
                mutant.quantum_significant = significant
                mutant.quantum_variance = variance

                processed += 1

        log.info("CANCER: Warburg complete -- %d mutants quantum-processed "
                 "in %d batches", processed,
                 processed // WARBURG_BATCH_SIZE + 1)

        return mutants

    # ---- PHASE 4: ANGIOGENESIS ----

    def angiogenesis(self, clusters: List[TumorCluster]) -> List[TumorCluster]:
        """
        Phase 4: Angiogenesis -- VEGF-Driven Resource Allocation.

        Top clusters get blood supply (more mutation budget).
        Bottom clusters are starved (apoptosis).
        Uses real VEGF diffusion model.
        """
        log.info("CANCER: ANGIOGENESIS -- allocating resources to %d clusters",
                 len(clusters))

        # Compute VEGF for each cluster based on fitness
        for cluster in clusters:
            alive = [m for m in cluster.mutants if m.is_alive]
            if not alive:
                cluster.avg_fitness = 0.0
                continue
            cluster.avg_fitness = np.mean([m.fitness_score for m in alive])

            # VEGF production: only above critical density
            tumor_density = len(alive) * cluster.avg_fitness * 1e6
            cluster.vegf_concentration = self.kinetics.vegf_production(
                tumor_density, T_critical=1e5
            )

        # Sort by fitness
        clusters.sort(key=lambda c: c.avg_fitness, reverse=True)

        top_n = max(1, int(len(clusters) * 0.20))
        starve_from = max(top_n + 1, int(len(clusters) * 0.50))

        for i, cluster in enumerate(clusters):
            if i < top_n:
                # Angiogenic switch ON -- full resources
                angio_rate = self.kinetics.angiogenesis_rate(
                    cluster.vegf_concentration
                )
                cluster.blood_supply = min(1.0, 0.5 + angio_rate)
                cluster.resource_budget = int(cluster.resource_budget * 3)
                log.info("CANCER: Angiogenesis ON -- cluster %s "
                         "(fitness=%.3f, VEGF=%.4f, budget=%d)",
                         cluster.cluster_id[:8], cluster.avg_fitness,
                         cluster.vegf_concentration, cluster.resource_budget)

            elif i >= starve_from:
                # Starve -- apoptosis
                cluster.blood_supply = 0.0
                cluster.resource_budget = 0
                for m in cluster.mutants:
                    m.is_alive = False
                log.info("CANCER: Apoptosis -- cluster %s "
                         "(fitness=%.3f) starved",
                         cluster.cluster_id[:8], cluster.avg_fitness)

            else:
                # Maintenance
                cluster.blood_supply = 0.3
                cluster.resource_budget = max(1, cluster.resource_budget // 2)

        return clusters

    # ---- PHASE 5: METASTASIS ----

    def metastasis(self, survivors: List[StrategyMutant],
                   all_symbols: List[str],
                   bars_by_symbol: Dict[str, np.ndarray]) -> List[StrategyMutant]:
        """
        Phase 5: Metastasis -- Cross-Symbol Strategy Spread.

        Uses Fisher-KPP invasion speed to model metastatic potential.
        Seed-and-soil test validates compatibility with target market.
        """
        log.info("CANCER: METASTASIS -- testing %d survivors across %d symbols",
                 len(survivors), len(all_symbols))

        full_context = SimulationContext(
            crispr_enabled=True,
            toxoplasma_enabled=True,
            regime_detection=True,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            max_loss_dollars=MAX_LOSS_DOLLARS,
        )

        total_colonizations = 0

        for mutant in survivors:
            if not mutant.is_alive or mutant.fitness_score < METASTASIS_MIN_FITNESS:
                continue

            # Fisher-KPP invasion speed
            # D = adaptability (inversely proportional to regime sensitivity)
            D = FISHER_D * (2.0 - mutant.regime_sensitivity)
            # r = local fitness
            r_local = max(0.01, mutant.fitness_score * FISHER_R * 10)
            invasion_speed = self.kinetics.fisher_invasion_speed(D, r_local)

            # Birth-death: must have positive growth to metastasize
            lam = max(mutant.win_rate, 0.01)
            mu_bd = max(1.0 - mutant.win_rate, 0.01)
            if self.kinetics.birth_death_extinction_prob(lam, mu_bd) > 0.7:
                continue  # Too fragile for migration

            colonized = 0
            for target in all_symbols:
                if target == mutant.parent_symbol:
                    continue
                if colonized >= METASTASIS_MAX_SYMBOLS:
                    break
                if target not in bars_by_symbol:
                    continue

                target_bars = bars_by_symbol[target][-METASTASIS_SOIL_TEST_BARS:]
                if len(target_bars) < 60:
                    continue

                lstm_model = self.expert_models.get(mutant.parent_id)
                soil_result = self.simulator.simulate(
                    mutant, target_bars, full_context,
                    lstm_model=lstm_model
                )

                if (soil_result["win_rate"] >= METASTASIS_COLONIZE_WR
                        and soil_result["total_trades"] >= 10):
                    mutant.metastasized_to.append(target)
                    colonized += 1
                    total_colonizations += 1

                    self.db.log_metastasis({
                        "mutant_id": mutant.mutant_id,
                        "source": mutant.parent_symbol,
                        "target": target,
                        "wr": soil_result["win_rate"],
                        "pf": soil_result["profit_factor"],
                        "trades": soil_result["total_trades"],
                        "invasion_speed": invasion_speed,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                    log.info("CANCER: Metastasis! %s -> %s "
                             "(WR=%.2f%%, PF=%.2f, speed=%.4f)",
                             mutant.mutant_id[:8], target,
                             soil_result["win_rate"] * 100,
                             soil_result["profit_factor"],
                             invasion_speed)

        log.info("CANCER: Metastasis complete -- %d colonizations", total_colonizations)
        return survivors

    # ---- PHASE 6: TELOMERASE ----

    def telomerase_activation(self, survivors: List[StrategyMutant]
                               ) -> List[StrategyMutant]:
        """
        Phase 6: Telomerase -- Immortality for Winners.

        Check Hayflick limit. Mutants that pass fitness thresholds
        get telomerase (unlimited replicative potential).
        Those that don't are checked for senescence.
        """
        log.info("CANCER: TELOMERASE -- checking %d survivors for immortality",
                 len(survivors))

        promoted = []

        for mutant in survivors:
            if not mutant.is_alive:
                continue

            # Check senescence (telomere-based aging)
            senescence_p = self.kinetics.senescence_probability(
                mutant.divisions, n_max=TELOMERE_HAYFLICK
            )
            if random.random() < senescence_p and not mutant.has_telomerase:
                mutant.is_alive = False
                log.debug("CANCER: Senescence -- %s died at division %d "
                          "(telomere=%.0f bp)",
                          mutant.mutant_id[:8], mutant.divisions,
                          mutant.telomere_length)
                continue

            # Telomerase activation criteria
            if mutant.total_trades < TELOMERASE_MIN_TRADES:
                continue

            if (mutant.win_rate >= TELOMERASE_PROMOTION_WR
                    and mutant.profit_factor >= TELOMERASE_PROMOTION_PF):

                mutant.has_telomerase = True
                mutant.telomere_length = TELOMERE_L0  # Fully restored
                mutant.promoted_at = datetime.utcnow().isoformat()
                promoted.append(mutant)

                log.info("CANCER: TELOMERASE ACTIVATED! %s is IMMORTAL "
                         "(WR=%.2f%%, PF=%.2f, trades=%d, colonized=%d)",
                         mutant.mutant_id[:8],
                         mutant.win_rate * 100, mutant.profit_factor,
                         mutant.total_trades, len(mutant.metastasized_to))

        log.info("CANCER: Telomerase -- %d/%d promoted to immortal",
                 len(promoted), len([s for s in survivors if s.is_alive]))

        return promoted

    # ---- PHASE 7: IMMUNE CHECKPOINT ----

    def immune_checkpoint(self, promoted: List[StrategyMutant],
                          bars_by_symbol: Dict[str, np.ndarray]
                          ) -> List[StrategyMutant]:
        """
        Phase 7: Immune Checkpoint -- Re-enable All Defenses.

        Like PD-1/PD-L1 checkpoint immunotherapy: let the immune
        system see the cancer. Only truly robust mutants survive.
        """
        log.info("CANCER: IMMUNE CHECKPOINT -- testing %d promoted mutants "
                 "with FULL defenses", len(promoted))

        full_context = SimulationContext(
            crispr_enabled=True,
            toxoplasma_enabled=True,
            regime_detection=True,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            max_loss_dollars=MAX_LOSS_DOLLARS,
        )

        validated = []

        for mutant in promoted:
            symbol = mutant.parent_symbol
            if symbol not in bars_by_symbol:
                continue

            bars = bars_by_symbol[symbol][-IMMUNE_CHECKPOINT_BARS:]
            if len(bars) < 60:
                continue

            lstm_model = self.expert_models.get(mutant.parent_id)
            result = self.simulator.simulate(mutant, bars, full_context,
                                             lstm_model=lstm_model)

            passed = (result["win_rate"] >= IMMUNE_CHECKPOINT_WR
                      and result["profit_factor"] >= IMMUNE_CHECKPOINT_PF
                      and result["total_trades"] >= 10)

            if passed:
                validated.append(mutant)
                self.db.log_promotion(mutant, symbol)

                log.info("CANCER: IMMUNE CHECKPOINT PASSED! %s "
                         "validated with full defenses "
                         "(WR=%.2f%%, PF=%.2f)",
                         mutant.mutant_id[:8],
                         result["win_rate"] * 100,
                         result["profit_factor"])
            else:
                log.info("CANCER: Immune checkpoint KILLED %s "
                         "(WR=%.2f%% -- defenses caught it)",
                         mutant.mutant_id[:8],
                         result["win_rate"] * 100)

        log.info("CANCER: Immune checkpoint -- %d/%d survived with full defenses",
                 len(validated), len(promoted))

        return validated

    # ---- MAIN SIMULATION LOOP ----

    def run(self, symbols: List[str],
            bars_by_symbol: Dict[str, np.ndarray],
            parent_strategies: Optional[List[dict]] = None,
            use_experts: bool = False,
            return_all_survivors: bool = False) -> List[StrategyMutant]:
        """
        Run the full cancer cell simulation.

        This is the main entry point. Runs all 7 phases:
        Mitosis -> Bypass -> Warburg -> Angiogenesis ->
        Metastasis -> Telomerase -> Immune Checkpoint

        Args:
            symbols: List of trading symbols (e.g., ["BTCUSD", "XAUUSD"])
            bars_by_symbol: Dict mapping symbol -> numpy array of OHLCV bars
            parent_strategies: Optional list of parent strategy dicts
                Each dict should have: {"id": str, "fitness": float, "generation": int}
            use_experts: If True, load real LSTM experts as parent cells

        Returns:
            List of validated StrategyMutant objects (survivors)
        """
        run_start = datetime.utcnow().isoformat()

        # Load real experts as parent cells if requested
        if use_experts and not parent_strategies:
            log.info("CANCER: Loading REAL TRAINED EXPERTS as parent cells...")
            loader = ExpertCellLoader()
            parent_strategies = loader.get_parent_strategies(
                symbols, bars_by_symbol,
                max_parents=MITOSIS_PARENT_COUNT,
            )
            if parent_strategies:
                # Cache LSTM models by parent ID for simulation phases
                for p in parent_strategies:
                    model = p.pop("model", None)
                    if model is not None:
                        self.expert_models[p["id"]] = model
                log.info("CANCER: %d expert parents ready -- "
                         "avg fitness=%.3f, avg WR=%.1f%%, "
                         "%d LSTM models cached for inference",
                         len(parent_strategies),
                         np.mean([p["fitness"] for p in parent_strategies]),
                         np.mean([p["win_rate"] for p in parent_strategies]) * 100,
                         len(self.expert_models))
            else:
                log.warning("CANCER: No experts loaded, falling back to seeds")

        log.info("=" * 70)
        log.info("CANCER CELL SIMULATION v%s: INITIATING", VERSION)
        log.info("  Symbols: %s", symbols)
        log.info("  Population target: %d per symbol",
                 MITOSIS_POPULATION_SIZE * MITOSIS_PARENT_COUNT)
        log.info("  Parent mode: %s",
                 "REAL EXPERTS" if (parent_strategies and
                     any(p.get("te_weight_signature") for p in (parent_strategies or [])))
                 else "random seeds")
        log.info("  Mathematical models: Gompertz, Knudson, Vogelstein, "
                 "Michaelis-Menten, Lotka-Volterra, Fisher-KPP, "
                 "Birth-Death, Warburg, Telomere, VEGF")
        log.info("  SL: $%.2f (SACRED, from config_loader)", MAX_LOSS_DOLLARS)
        log.info("=" * 70)

        # Default parent strategies if none provided
        if not parent_strategies:
            parent_strategies = [
                {"id": f"seed_{i}", "fitness": random.uniform(0.3, 0.7),
                 "generation": 0}
                for i in range(MITOSIS_PARENT_COUNT)
            ]

        all_validated = []
        all_survivors_raw = []  # All survivors before promotion filter
        total_mutants = 0

        for symbol in symbols:
            if symbol not in bars_by_symbol:
                log.warning("CANCER: No bars for %s, skipping", symbol)
                continue

            bars = bars_by_symbol[symbol]
            log.info("\n--- CANCER: Processing %s (%d bars) ---",
                     symbol, len(bars))

            # Phase 1: MITOSIS
            mutants = self.mitosis(parent_strategies, symbol, bars)
            total_mutants += len(mutants)

            # Phase 3: WARBURG (quantum before simulation for TE-aware eval)
            mutants = self.warburg_quantum_batch(mutants, bars)

            # Cluster by parent
            clusters = self._cluster_mutants(mutants, symbol)

            # Iterative mutation with angiogenesis
            for gen in range(MITOSIS_GENERATIONS):
                log.info("CANCER [%s]: Generation %d/%d",
                         symbol, gen + 1, MITOSIS_GENERATIONS)

                # Phase 2: BYPASS
                for cluster in clusters:
                    if cluster.resource_budget > 0:
                        cluster.mutants = self.bypass_checkpoints(
                            cluster.mutants, symbol, bars
                        )
                        alive = [m for m in cluster.mutants if m.is_alive]
                        if alive:
                            cluster.avg_fitness = np.mean(
                                [m.fitness_score for m in alive]
                            )

                # Phase 4: ANGIOGENESIS
                clusters = self.angiogenesis(clusters)

                # Spawn new mutations from survivors
                for cluster in clusters:
                    if cluster.resource_budget > 0:
                        alive = [m for m in cluster.mutants if m.is_alive]
                        if alive:
                            # Lotka-Volterra: update immune pressure
                            avg_fitness = np.mean([m.fitness_score for m in alive])
                            for m in alive:
                                T, I = self.kinetics.lotka_volterra_step(
                                    T=m.gompertz_N,
                                    I=m.lv_immune_pressure,
                                    K_T=100, K_I=50
                                )
                                m.gompertz_N = T
                                m.lv_immune_pressure = I

                            # Telomere shortening per generation
                            for m in alive:
                                m.telomere_length = self.kinetics.telomere_after_division(
                                    m.telomere_length, n_divisions=1
                                )
                                m.divisions += 1

                        cluster.resource_budget -= 1

            # Collect survivors
            survivors = [
                m for c in clusters for m in c.mutants if m.is_alive
            ]
            all_survivors_raw.extend(survivors)
            log.info("CANCER [%s]: %d survivors from %d initial mutants",
                     symbol, len(survivors), MITOSIS_POPULATION_SIZE * MITOSIS_PARENT_COUNT)

            # Phase 5: METASTASIS
            survivors = self.metastasis(survivors, symbols, bars_by_symbol)

            # Phase 6: TELOMERASE
            promoted = self.telomerase_activation(survivors)

            # Phase 7: IMMUNE CHECKPOINT
            validated = self.immune_checkpoint(promoted, bars_by_symbol)

            # Save validated mutants
            for m in validated:
                self.db.save_mutant(m)

            all_validated.extend(validated)

        # Save run history
        run_end = datetime.utcnow().isoformat()
        self.db.save_run({
            "run_start": run_start,
            "run_end": run_end,
            "symbols": symbols,
            "total_mutants": total_mutants,
            "survivors": len(all_validated),
            "promoted": len(all_validated),
            "avg_fitness": (np.mean([m.fitness_score for m in all_validated])
                           if all_validated else 0),
            "avg_wr": (np.mean([m.win_rate for m in all_validated])
                       if all_validated else 0),
            "metastasis_count": sum(len(m.metastasized_to) for m in all_validated),
            "gompertz_factor": (np.mean([m.gompertz_N for m in all_validated])
                                if all_validated else 0),
            "warburg_throughput": self.kinetics.warburg_efficiency(0.85)[0],
            "telomere_avg": (np.mean([m.telomere_length for m in all_validated])
                             if all_validated else 0),
        })

        log.info("=" * 70)
        log.info("CANCER CELL SIMULATION: COMPLETE")
        log.info("  Duration: %s -> %s", run_start, run_end)
        log.info("  Mutants generated: %d", total_mutants)
        log.info("  Survivors: %d", len(all_validated))
        if all_validated:
            log.info("  Avg WR: %.2f%%",
                     np.mean([m.win_rate for m in all_validated]) * 100)
            log.info("  Avg PF: %.2f",
                     np.mean([m.profit_factor for m in all_validated]))
            log.info("  Metastasized: %d total colonizations",
                     sum(len(m.metastasized_to) for m in all_validated))
            log.info("  Avg telomere: %.0f bp",
                     np.mean([m.telomere_length for m in all_validated]))
        log.info("=" * 70)

        if return_all_survivors:
            return all_validated, all_survivors_raw
        return all_validated

    # ---- HELPERS ----

    def _cluster_mutants(self, mutants: List[StrategyMutant],
                         symbol: str) -> List[TumorCluster]:
        """Group mutants by parent into tumor clusters."""
        parent_map: Dict[str, List[StrategyMutant]] = {}
        for m in mutants:
            key = m.parent_id
            if key not in parent_map:
                parent_map[key] = []
            parent_map[key].append(m)

        clusters = []
        for parent_id, group in parent_map.items():
            cluster = TumorCluster(
                parent_symbol=symbol,
                mutants=group,
                resource_budget=MITOSIS_GENERATIONS,
            )
            clusters.append(cluster)

        return clusters


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """Run cancer cell simulation from command line."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Cancer Cell Engine -- Oncogenic Strategy Acceleration"
    )
    parser.add_argument("--symbols", nargs="+",
                        default=["BTCUSD", "XAUUSD", "ETHUSD"],
                        help="Symbols to process")
    parser.add_argument("--bars", type=int, default=2000,
                        help="Number of historical bars to use")
    parser.add_argument("--population", type=int, default=200,
                        help="Mutant population size per parent")
    parser.add_argument("--generations", type=int, default=5,
                        help="Number of mutation generations")
    parser.add_argument("--use-experts", action="store_true",
                        help="Load real LSTM experts as parent cells "
                             "(instead of random seeds)")
    args = parser.parse_args()

    # Apply CLI overrides at module level
    import cancer_cell as _self_module
    _self_module.MITOSIS_POPULATION_SIZE = args.population
    _self_module.MITOSIS_GENERATIONS = args.generations

    # Try to fetch real market data from MT5
    bars_by_symbol = {}
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            for symbol in args.symbols:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, args.bars)
                if rates is not None and len(rates) > 0:
                    bars_by_symbol[symbol] = np.array([
                        [r[1], r[2], r[3], r[4], r[5]] for r in rates
                    ])
                    log.info("Loaded %d bars for %s from MT5", len(rates), symbol)
            mt5.shutdown()
    except Exception as e:
        log.warning("MT5 not available: %s", e)

    # Fallback to synthetic data if MT5 not available
    for symbol in args.symbols:
        if symbol not in bars_by_symbol:
            log.info("Generating synthetic bars for %s", symbol)
            n = args.bars
            close = np.cumsum(np.random.randn(n) * 0.001) + 100
            high = close + abs(np.random.randn(n) * 0.0005)
            low = close - abs(np.random.randn(n) * 0.0005)
            opn = close + np.random.randn(n) * 0.0002
            volume = np.random.randint(100, 10000, n).astype(float)
            bars_by_symbol[symbol] = np.column_stack([opn, high, low, close, volume])

    # Run the simulation
    engine = CancerCellEngine()
    validated = engine.run(args.symbols, bars_by_symbol,
                           use_experts=args.use_experts)

    print(f"\n{'='*60}")
    print(f"CANCER CELL SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Validated strategies: {len(validated)}")
    for m in validated[:10]:
        print(f"  {m.mutant_id[:8]} | WR={m.win_rate:.2%} | "
              f"PF={m.profit_factor:.2f} | Sharpe={m.sharpe_ratio:.2f} | "
              f"Drivers={m.driver_count} | "
              f"Telomere={m.telomere_length:.0f}bp | "
              f"Colonized={len(m.metastasized_to)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
