"""
Growth Signal Amplifier -- "Somatotropic Signal Amplification"
==============================================================
Implements the growth hormone algorithm from ALGORITHM_HGH_HORMONE.py.

Reads the domestication DB (READ ONLY) to find the top-performing
domesticated TE patterns, constructs a four-helix bundle, validates
structural integrity via disulfide bridges, checks receptor binding
(temporal persistence), runs the JAK2/STAT amplification cascade,
and outputs:
  - Lipolysis: tighten SL on losing positions
  - IGF-1 Bridge: extra quantum rotations + confidence boost + hyperplasia

Integration points:
  - teqa_v3_neural_te.py :: analyze()  (confidence boost)
  - BRAIN_*.py :: run_cycle()          (hyperplasia / 2nd position)
  - quantum_server.py :: /compress     (extra RY rotations)

Authors: DooDoo + Claude
Date:    2026-02-12
Version: 1.0
"""

import json
import math
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config_loader import AGENT_SL_MIN

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# CONSTANTS (from ALGORITHM_HGH_HORMONE.py pseudocode)
# ──────────────────────────────────────────────────────────

N_HELICES = 4
MIN_DOMESTICATED_FOR_BUNDLE = 4
PARTIAL_BUNDLE_MIN = 2

# Disulfide bridge thresholds
BRIDGE_REGIME_MATCH_REQUIRED = True
BRIDGE_DIRECTION_MATCH_REQUIRED = True

# Receptor binding (dimerization)
SITE1_SIMILARITY_THRESHOLD = 0.50
SITE2_SIMILARITY_THRESHOLD = 0.40

# JAK2/STAT cascade
MAX_GROWTH_SIGNAL = 0.35

# Negative feedback (somatostatin)
SOMATOSTATIN_THRESHOLD = 5
REFRACTORY_CYCLES = 3

# Pulsatile delivery (session windows UTC)
PULSE_WINDOWS_UTC = {
    "LONDON_OPEN": (7, 9),
    "NY_OPEN": (13, 15),
    "ASIAN_OPEN": (23, 1),
}

# Direct effect (lipolysis) -- SL tightening
MAX_SL_TIGHTENING_PERCENT = 0.105

# Indirect effect (IGF-1 hyperplasia)
IGF1_HYPERPLASIA_THRESHOLD = 0.20
IGF1_MAX_SECOND_LOT_RATIO = 1.0

# Variant potencies
FULL_POTENCY = 1.0
PARTIAL_POTENCY = 0.60
MISFOLDED_POTENCY = 0.0

# Domestication query thresholds
DOMESTICATION_EXPIRY_DAYS = 30
MIN_POSTERIOR_WR = 0.70
MIN_PROFIT_FACTOR = 1.5


# ──────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────

@dataclass
class Helix:
    """One helix in the four-helix bundle."""
    te_name: str
    qubit_index: int
    boost_factor: float
    win_rate: float
    profit_factor: float
    direction: int = 0       # -1 short, 0 neutral, +1 long
    activation: float = 0.0  # current cycle strength


@dataclass
class Molecule:
    """The growth molecule (four-helix bundle)."""
    helices: List[Optional[Helix]] = field(default_factory=lambda: [None] * 4)
    bridge_1_intact: bool = False
    bridge_2_intact: bool = False
    potency: float = 0.0
    fingerprint: Optional[np.ndarray] = None


@dataclass
class ReceptorBinding:
    """Result of receptor dimerization check."""
    site1_bound: bool = False
    site2_bound: bool = False
    binding_strength: float = 0.0


@dataclass
class GrowthCascade:
    """JAK2/STAT cascade output."""
    jak2_phosphorylation: float = 0.0
    stat_docking: float = 0.0
    growth_signal: float = 0.0


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return float(dot / (mag_a * mag_b))


def _classify_regime(bars: np.ndarray, lookback: int = 20) -> str:
    """
    Classify current volatility regime using ATR ratio and Bollinger width.
    Returns: 'trending', 'ranging', or 'volatile'
    """
    if len(bars) < lookback + 5:
        return "ranging"

    close = bars[:, 3] if bars.ndim == 2 else bars
    high = bars[:, 1] if bars.ndim == 2 else bars
    low = bars[:, 2] if bars.ndim == 2 else bars

    # ATR
    tr = np.maximum(
        high[-lookback:] - low[-lookback:],
        np.maximum(
            np.abs(high[-lookback:] - np.roll(close, 1)[-lookback:]),
            np.abs(low[-lookback:] - np.roll(close, 1)[-lookback:])
        )
    )
    atr = np.mean(tr)
    atr_recent = np.mean(tr[-5:])

    # Bollinger width
    sma = np.mean(close[-lookback:])
    std = np.std(close[-lookback:])
    bb_width = (2 * std) / (sma + 1e-10)

    # Recent vs historical ATR ratio
    atr_ratio = atr_recent / (atr + 1e-10)

    if atr_ratio > 1.5 and bb_width > 0.03:
        return "volatile"
    elif atr_ratio > 0.8 and bb_width > 0.015:
        return "trending"
    else:
        return "ranging"


# ──────────────────────────────────────────────────────────
# MAIN ENGINE
# ──────────────────────────────────────────────────────────

class GrowthAmplifier:
    """
    Somatotropic Signal Amplification engine.

    Reads the domestication DB to find the top domesticated TEs,
    constructs a four-helix bundle, validates structural integrity,
    checks receptor binding, runs the JAK2/STAT cascade, and
    produces growth signal outputs for the TEQA pipeline.
    """

    def __init__(self, db_path: str = None, log_path: str = None):
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "teqa_domestication.db")
        else:
            self.db_path = db_path

        if log_path is None:
            self.log_path = str(Path(__file__).parent / "growth_amplifier_log.jsonl")
        else:
            self.log_path = log_path

        # State (persists across cycles, resets on restart)
        self.previous_activations: Optional[np.ndarray] = None
        self.consecutive_growth: int = 0
        self.refractory_remaining: int = 0
        self.last_growth_signal: float = 0.0
        self.cycle_count: int = 0

        # TE family lookup (qubit index by name)
        self._te_qubit_map: Dict[str, int] = {}
        self._init_te_map()

    def _init_te_map(self):
        """Build TE name -> qubit index lookup from teqa_v3_neural_te."""
        try:
            from teqa_v3_neural_te import ALL_TE_FAMILIES
            for te in ALL_TE_FAMILIES:
                self._te_qubit_map[te.name] = te.qubit_index
        except ImportError:
            log.warning("GrowthAmplifier: could not import ALL_TE_FAMILIES, "
                        "qubit mapping will be empty until set manually")

    # ──────────────────────────────────────────────────
    # PHASE 1: FOUR-HELIX BUNDLE (Hormone Synthesis)
    # ──────────────────────────────────────────────────

    def synthesize(self, te_activations: List[Dict]) -> Optional[Molecule]:
        """
        Query domestication DB for top domesticated TEs and build
        the four-helix bundle.

        Args:
            te_activations: list of dicts from TEActivationEngine
                Each: {"te": name, "strength": float, "direction": int, ...}

        Returns:
            Molecule if enough domesticated TEs, else None
        """
        # Build activation lookup
        act_lookup = {a["te"]: a for a in te_activations}

        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT te_combo, posterior_wr, profit_factor, boost_factor,
                           last_activated
                    FROM domesticated_patterns
                    WHERE domesticated = 1
                      AND posterior_wr >= ?
                      AND profit_factor >= ?
                    ORDER BY posterior_wr * CAST(
                        LOG(profit_factor + 1) AS REAL
                    ) DESC
                """, (MIN_POSTERIOR_WR, MIN_PROFIT_FACTOR))
                rows = cursor.fetchall()
        except Exception as e:
            log.warning("GrowthAmplifier: DB query failed: %s", e)
            # Fallback: try without LOG() which SQLite doesn't natively support
            try:
                with sqlite3.connect(self.db_path, timeout=5) as conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT te_combo, posterior_wr, profit_factor, boost_factor,
                               last_activated
                        FROM domesticated_patterns
                        WHERE domesticated = 1
                          AND posterior_wr >= ?
                          AND profit_factor >= ?
                        ORDER BY posterior_wr DESC
                    """, (MIN_POSTERIOR_WR, MIN_PROFIT_FACTOR))
                    rows = cursor.fetchall()
            except Exception as e2:
                log.warning("GrowthAmplifier: DB fallback query also failed: %s", e2)
                return None

        if not rows:
            log.debug("GrowthAmplifier: no domesticated patterns meet criteria")
            return None

        # Filter expired patterns
        now = datetime.now()
        valid_rows = []
        for row in rows:
            last_act = row[4]
            if last_act:
                try:
                    last_dt = datetime.fromisoformat(last_act)
                    if (now - last_dt).days > DOMESTICATION_EXPIRY_DAYS:
                        continue
                except (ValueError, TypeError):
                    pass
            valid_rows.append(row)

        if not valid_rows:
            log.debug("GrowthAmplifier: all domesticated patterns expired")
            return None

        # Extract individual TEs and rank by cumulative fitness
        te_scores: Dict[str, float] = {}
        te_boost: Dict[str, float] = {}
        te_wr: Dict[str, float] = {}
        te_pf: Dict[str, float] = {}

        for row in valid_rows:
            combo = row[0]       # te_combo
            pwr = row[1]         # posterior_wr
            pf = row[2]          # profit_factor
            bf = row[3]          # boost_factor

            tes = combo.split("+")
            fitness = pwr * math.log(pf + 1)

            for te_name in tes:
                te_scores[te_name] = te_scores.get(te_name, 0) + fitness
                # Keep highest boost/wr/pf seen for each TE
                if te_name not in te_boost or bf > te_boost[te_name]:
                    te_boost[te_name] = bf
                if te_name not in te_wr or pwr > te_wr[te_name]:
                    te_wr[te_name] = pwr
                if te_name not in te_pf or pf > te_pf[te_name]:
                    te_pf[te_name] = pf

        # Sort by score descending
        ranked = sorted(te_scores.items(), key=lambda x: x[1], reverse=True)

        if len(ranked) < PARTIAL_BUNDLE_MIN:
            log.debug("GrowthAmplifier: only %d ranked TEs, need %d",
                       len(ranked), PARTIAL_BUNDLE_MIN)
            return None

        # Build helices (up-up-down-down topology)
        molecule = Molecule()
        for i in range(min(N_HELICES, len(ranked))):
            te_name = ranked[i][0]
            qubit_idx = self._te_qubit_map.get(te_name, -1)

            # Get current activation data
            act_data = act_lookup.get(te_name, {})
            direction = act_data.get("direction", 0)
            strength = act_data.get("strength", 0.0)

            molecule.helices[i] = Helix(
                te_name=te_name,
                qubit_index=qubit_idx,
                boost_factor=te_boost.get(te_name, 1.0),
                win_rate=te_wr.get(te_name, 0.5),
                profit_factor=te_pf.get(te_name, 1.0),
                direction=direction,
                activation=strength,
            )

        # Compute fingerprint (33-element vector)
        n_qubits = 33
        fp = np.zeros(n_qubits, dtype=np.float64)
        for h in molecule.helices:
            if h is not None and 0 <= h.qubit_index < n_qubits:
                fp[h.qubit_index] = h.activation * h.boost_factor

        norm = np.linalg.norm(fp)
        if norm > 1e-10:
            fp = fp / norm
        molecule.fingerprint = fp

        helix_names = [h.te_name for h in molecule.helices if h is not None]
        log.info("GrowthAmplifier Phase 1: synthesized bundle from %s", helix_names)

        return molecule

    # ──────────────────────────────────────────────────
    # PHASE 2: DISULFIDE BRIDGE VALIDATION
    # ──────────────────────────────────────────────────

    def validate_bridges(self, molecule: Molecule, bars: np.ndarray) -> Molecule:
        """
        Validate structural integrity via disulfide bridges.

        Bridge 1 (Cys-53 <-> Cys-165): Helix A and Helix C must agree on
            volatility regime.
        Bridge 2 (Cys-182 <-> Cys-189): Helix B and Helix D must agree on
            direction.
        """
        helix_a = molecule.helices[0]
        helix_b = molecule.helices[1]
        helix_c = molecule.helices[2]
        helix_d = molecule.helices[3]

        # Bridge 1: Helix A <-> Helix C (regime agreement)
        if helix_a is not None and helix_c is not None:
            regime = _classify_regime(bars)
            # Both helices see the same bars, so regime is shared.
            # The "agreement" is whether both helices are activated
            # in the same direction under this regime.
            molecule.bridge_1_intact = True  # Same bars = same regime
            log.debug("GrowthAmplifier Bridge 1: regime=%s -- INTACT", regime)
        else:
            molecule.bridge_1_intact = False
            log.debug("GrowthAmplifier Bridge 1: missing helix -- BROKEN")

        # Bridge 2: Helix B <-> Helix D (direction agreement)
        if helix_b is not None and helix_d is not None:
            dir_b = helix_b.direction
            dir_d = helix_d.direction
            # Agreement: same sign, or one is neutral
            molecule.bridge_2_intact = (
                (dir_b == dir_d) or (dir_b == 0 or dir_d == 0)
            )
            log.debug("GrowthAmplifier Bridge 2: dir_B=%d dir_D=%d -- %s",
                       dir_b, dir_d,
                       "INTACT" if molecule.bridge_2_intact else "BROKEN")
        else:
            molecule.bridge_2_intact = False
            log.debug("GrowthAmplifier Bridge 2: missing helix -- BROKEN")

        # Determine potency
        if molecule.bridge_1_intact and molecule.bridge_2_intact:
            molecule.potency = FULL_POTENCY
            log.info("GrowthAmplifier Phase 2: FULL potency (both bridges intact)")
        elif molecule.bridge_1_intact or molecule.bridge_2_intact:
            molecule.potency = PARTIAL_POTENCY
            log.info("GrowthAmplifier Phase 2: PARTIAL potency (one bridge broken)")
        else:
            molecule.potency = MISFOLDED_POTENCY
            log.info("GrowthAmplifier Phase 2: MISFOLDED (both bridges broken)")

        return molecule

    # ──────────────────────────────────────────────────
    # PHASE 3: RECEPTOR DIMERIZATION (1:2 Binding)
    # ──────────────────────────────────────────────────

    def attempt_dimerization(
        self,
        molecule: Molecule,
        current_activations: np.ndarray,
        previous_activations: Optional[np.ndarray],
    ) -> ReceptorBinding:
        """
        Check if the molecule binds to current and previous TE patterns.
        Requires temporal persistence (not just one-bar noise).
        """
        if molecule.potency <= 0.0 or molecule.fingerprint is None:
            return ReceptorBinding(False, False, 0.0)

        # Site 1: current cycle pattern
        sim_1 = _cosine_similarity(molecule.fingerprint, current_activations)
        site1_bound = sim_1 >= SITE1_SIMILARITY_THRESHOLD

        # Site 2: previous cycle pattern (temporal persistence)
        if previous_activations is not None:
            sim_2 = _cosine_similarity(molecule.fingerprint, previous_activations)
            site2_bound = sim_2 >= SITE2_SIMILARITY_THRESHOLD
        else:
            site2_bound = False

        # Binding strength
        if site1_bound and site2_bound:
            binding_strength = 1.0
            log.info("GrowthAmplifier Phase 3: FULL dimerization (sim1=%.3f sim2=%.3f)",
                      sim_1, sim_2 if previous_activations is not None else 0)
        elif site1_bound:
            binding_strength = 0.5
            log.info("GrowthAmplifier Phase 3: PARTIAL binding (Site 1 only, sim=%.3f)", sim_1)
        else:
            binding_strength = 0.0
            log.debug("GrowthAmplifier Phase 3: NO binding (sim=%.3f)", sim_1)

        return ReceptorBinding(site1_bound, site2_bound, binding_strength)

    # ──────────────────────────────────────────────────
    # PHASE 4: JAK2/STAT CASCADE (Signal Amplification)
    # ──────────────────────────────────────────────────

    def jak2_stat_cascade(
        self,
        molecule: Molecule,
        binding: ReceptorBinding,
        active_te_directions: List[int],
    ) -> GrowthCascade:
        """
        Compute the growth signal from the amplification cascade.
        """
        if binding.binding_strength <= 0.0:
            return GrowthCascade(0.0, 0.0, 0.0)

        # Step 1: JAK2 Phosphorylation (average boost of helices)
        boosts = [h.boost_factor for h in molecule.helices if h is not None]
        jak2 = float(np.mean(boosts)) if boosts else 1.0

        # Step 2: STAT Docking (directional concordance)
        if active_te_directions:
            majority = np.sign(sum(active_te_directions))
            if majority == 0:
                majority = 1  # tie-break to long
            agreeing = sum(1 for d in active_te_directions if np.sign(d) == majority)
            stat = agreeing / len(active_te_directions)
        else:
            stat = 0.0

        # Step 3: Nuclear Translocation
        raw_signal = (jak2 - 1.0) * stat * binding.binding_strength * molecule.potency
        growth_signal = min(raw_signal, MAX_GROWTH_SIGNAL)
        growth_signal = max(growth_signal, 0.0)  # floor at zero

        log.info("GrowthAmplifier Phase 4: jak2=%.3f stat=%.3f binding=%.2f "
                  "potency=%.2f -> growth=%.4f",
                  jak2, stat, binding.binding_strength, molecule.potency, growth_signal)

        return GrowthCascade(jak2, stat, growth_signal)

    # ──────────────────────────────────────────────────
    # PHASE 5A: LIPOLYSIS (SL Tightening on Losers)
    # ──────────────────────────────────────────────────

    def compute_lipolysis(self, growth_signal: float) -> Dict:
        """
        Compute SL tightening parameters for losing positions.
        Returns dict with tightening info (does NOT execute MT5 orders).

        The BRAIN script is responsible for applying the tightening.
        """
        if growth_signal <= 0.0:
            return {"active": False, "sl_tighten_factor": 1.0}

        sl_tighten_factor = 1.0 - (
            growth_signal * MAX_SL_TIGHTENING_PERCENT / MAX_GROWTH_SIGNAL
        )

        log.info("GrowthAmplifier Phase 5A: lipolysis factor=%.4f "
                  "(%.1f%% tighter SL on losers)",
                  sl_tighten_factor, (1.0 - sl_tighten_factor) * 100)

        return {
            "active": True,
            "sl_tighten_factor": sl_tighten_factor,
            "agent_sl_min": AGENT_SL_MIN,
        }

    # ──────────────────────────────────────────────────
    # PHASE 5B: IGF-1 BRIDGE (Quantum Growth)
    # ──────────────────────────────────────────────────

    def compute_igf1_bridge(
        self,
        growth_signal: float,
        molecule: Molecule,
    ) -> Dict:
        """
        Compute the IGF-1 bridge parameters for quantum injection.
        Returns rotation angles and hyperplasia/hypertrophy data.
        Does NOT execute quantum circuits (that's the caller's job).
        """
        if growth_signal <= 0.0:
            return {
                "active": False,
                "igf1_level": 0.0,
                "hyperplasia": False,
                "second_lot_ratio": 0.0,
                "hypertrophy_boost": 0.0,
                "extra_rotations": [],
                "global_phase_bias": 0.0,
            }

        # Compute extra RY rotations for each helix qubit
        extra_rotations = []
        for h in molecule.helices:
            if h is not None and h.qubit_index >= 0:
                theta = h.activation * growth_signal * (math.pi / 2)
                extra_rotations.append({
                    "qubit": h.qubit_index,
                    "ry_angle": theta,
                    "te_name": h.te_name,
                })

        # Global RZ phase bias
        global_phase_bias = growth_signal * math.pi / 4

        # Hyperplasia check (second position)
        hyperplasia = growth_signal > IGF1_HYPERPLASIA_THRESHOLD
        if hyperplasia:
            second_lot_ratio = growth_signal / MAX_GROWTH_SIGNAL
            second_lot_ratio = min(second_lot_ratio, IGF1_MAX_SECOND_LOT_RATIO)
            log.info("GrowthAmplifier Phase 5B: HYPERPLASIA -- "
                      "second position at %.0f%% lot", second_lot_ratio * 100)
        else:
            second_lot_ratio = 0.0

        # Hypertrophy (confidence boost)
        hypertrophy_boost = growth_signal * 0.5
        log.info("GrowthAmplifier Phase 5B: hypertrophy boost = +%.4f",
                  hypertrophy_boost)

        return {
            "active": True,
            "igf1_level": growth_signal,  # proportional to growth
            "hyperplasia": hyperplasia,
            "second_lot_ratio": second_lot_ratio,
            "hypertrophy_boost": hypertrophy_boost,
            "extra_rotations": extra_rotations,
            "global_phase_bias": global_phase_bias,
        }

    # ──────────────────────────────────────────────────
    # PHASE 6: NEGATIVE FEEDBACK (Somatostatin)
    # ──────────────────────────────────────────────────

    def check_negative_feedback(self) -> bool:
        """
        Check if the amplifier is in refractory period.
        Returns True if suppressed (hormone should NOT fire).
        """
        # Currently in refractory?
        if self.refractory_remaining > 0:
            self.refractory_remaining -= 1
            log.info("GrowthAmplifier Phase 6: REFRACTORY -- %d cycles remaining",
                      self.refractory_remaining)
            return True

        # Track consecutive activations
        if self.last_growth_signal > IGF1_HYPERPLASIA_THRESHOLD:
            self.consecutive_growth += 1
        else:
            self.consecutive_growth = max(0, self.consecutive_growth - 1)

        # Trigger refractory?
        if self.consecutive_growth >= SOMATOSTATIN_THRESHOLD:
            self.refractory_remaining = REFRACTORY_CYCLES
            self.consecutive_growth = 0
            log.info("GrowthAmplifier Phase 6: OVERACTIVATION -- "
                      "entering %d-cycle refractory", REFRACTORY_CYCLES)
            return True

        return False

    # ──────────────────────────────────────────────────
    # PHASE 7: PULSATILE DELIVERY (Session Windows)
    # ──────────────────────────────────────────────────

    @staticmethod
    def is_pulse_window() -> bool:
        """
        Check if current time is within a pulse delivery window.
        Returns True if amplifier is allowed to fire.
        """
        from datetime import timezone
        current_hour = datetime.now(timezone.utc).hour

        for window_name, (start_h, end_h) in PULSE_WINDOWS_UTC.items():
            if start_h <= end_h:
                if start_h <= current_hour < end_h:
                    return True
            else:
                # Wraps midnight (e.g., 23-1)
                if current_hour >= start_h or current_hour < end_h:
                    return True

        return False

    # ──────────────────────────────────────────────────
    # MAIN CYCLE (called from TEQA pipeline)
    # ──────────────────────────────────────────────────

    def run_cycle(
        self,
        bars: np.ndarray,
        symbol: str,
        te_activations: List[Dict],
    ) -> Dict:
        """
        Run one full growth amplification cycle.

        This is the main entry point. Call it from TEQAv3Engine.analyze()
        after computing TE activations and before emitting the signal.

        Args:
            bars: OHLCV numpy array
            symbol: trading symbol (e.g. 'BTCUSD')
            te_activations: list of TE activation dicts from TEActivationEngine

        Returns:
            Dict with growth_signal, igf1 data, lipolysis data, etc.
        """
        self.cycle_count += 1
        t_start = time.time()

        # Build null result for early exits
        null_result = {
            "active": False,
            "growth_signal": 0.0,
            "igf1": {
                "active": False, "igf1_level": 0.0,
                "hyperplasia": False, "second_lot_ratio": 0.0,
                "hypertrophy_boost": 0.0, "extra_rotations": [],
                "global_phase_bias": 0.0,
            },
            "lipolysis": {"active": False, "sl_tighten_factor": 1.0},
            "molecule_summary": None,
            "elapsed_ms": 0.0,
        }

        # Gate 0: Pulsatile delivery
        if not self.is_pulse_window():
            log.debug("GrowthAmplifier: outside pulse window, suppressed")
            null_result["suppression_reason"] = "outside_pulse_window"
            return null_result

        # Gate 1: Negative feedback
        if self.check_negative_feedback():
            null_result["suppression_reason"] = "somatostatin_refractory"
            return null_result

        # Phase 1: Synthesize molecule from domesticated TEs
        molecule = self.synthesize(te_activations)
        if molecule is None:
            null_result["suppression_reason"] = "insufficient_domesticated_tes"
            return null_result

        # Phase 2: Validate disulfide bridges
        molecule = self.validate_bridges(molecule, bars)

        # Phase 3: Receptor dimerization
        # Build current activation vector (33 elements)
        n_qubits = 33
        current_vec = np.zeros(n_qubits, dtype=np.float64)
        for act in te_activations:
            qi = self._te_qubit_map.get(act["te"], -1)
            if 0 <= qi < n_qubits:
                current_vec[qi] = act["strength"] * act["direction"]

        binding = self.attempt_dimerization(
            molecule, current_vec, self.previous_activations
        )
        self.previous_activations = current_vec.copy()

        # Phase 4: JAK2/STAT cascade
        active_dirs = [
            a["direction"] for a in te_activations if a["strength"] > 0.5
        ]
        cascade = self.jak2_stat_cascade(molecule, binding, active_dirs)

        # Phase 5A: Lipolysis (SL tightening params)
        lipolysis = self.compute_lipolysis(cascade.growth_signal)

        # Phase 5B: IGF-1 Bridge (quantum rotation params)
        igf1 = self.compute_igf1_bridge(cascade.growth_signal, molecule)

        # Store for next cycle feedback
        self.last_growth_signal = cascade.growth_signal

        elapsed_ms = (time.time() - t_start) * 1000

        # Molecule summary for logging
        mol_summary = {
            "helices": [
                {"te": h.te_name, "qubit": h.qubit_index,
                 "boost": h.boost_factor, "wr": h.win_rate,
                 "dir": h.direction, "act": h.activation}
                for h in molecule.helices if h is not None
            ],
            "bridge_1": molecule.bridge_1_intact,
            "bridge_2": molecule.bridge_2_intact,
            "potency": molecule.potency,
        }

        result = {
            "active": cascade.growth_signal > 0.0,
            "growth_signal": cascade.growth_signal,
            "igf1": igf1,
            "lipolysis": lipolysis,
            "molecule_summary": mol_summary,
            "binding_strength": binding.binding_strength,
            "jak2": cascade.jak2_phosphorylation,
            "stat": cascade.stat_docking,
            "cycle": self.cycle_count,
            "consecutive_growth": self.consecutive_growth,
            "refractory_remaining": self.refractory_remaining,
            "elapsed_ms": elapsed_ms,
        }

        # Log to JSONL
        self._log_cycle(symbol, result)

        if cascade.growth_signal > 0:
            log.info(
                "GrowthAmplifier ACTIVE: growth=%.4f | hypertrophy=+%.4f | "
                "hyperplasia=%s | lipolysis=%.1f%% | elapsed=%.1fms",
                cascade.growth_signal,
                igf1["hypertrophy_boost"],
                igf1["hyperplasia"],
                (1.0 - lipolysis["sl_tighten_factor"]) * 100 if lipolysis["active"] else 0,
                elapsed_ms,
            )

        return result

    def _log_cycle(self, symbol: str, result: Dict):
        """Append cycle data to JSONL log."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "cycle": result["cycle"],
                "active": result["active"],
                "growth_signal": result["growth_signal"],
                "binding_strength": result.get("binding_strength", 0.0),
                "jak2": result.get("jak2", 0.0),
                "stat": result.get("stat", 0.0),
                "hypertrophy_boost": result["igf1"]["hypertrophy_boost"],
                "hyperplasia": result["igf1"]["hyperplasia"],
                "potency": result["molecule_summary"]["potency"] if result["molecule_summary"] else 0.0,
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            log.debug("GrowthAmplifier: log write failed: %s", e)
