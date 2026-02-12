"""
HGH Hormone -- Somatotropic Signal Amplification
==================================================
Straps Human Growth Hormone signaling cascade to the most positive
domesticated TEs, then routes the amplified signal through quantum
compute for trade decisions.

See ALGORITHM_HGH_HORMONE.py for full biological documentation.

Pipeline:
    1. Synthesize 4-helix bundle from top domesticated TEs
    2. Validate disulfide bridges (regime + direction integrity)
    3. Receptor dimerization (current + previous TE pattern matching)
    4. JAK2/STAT cascade (signal amplification)
    5A. Lipolysis (tighten SL on losers -- advisory only)
    5B. IGF-1 bridge (quantum injection -> hyperplasia/hypertrophy)
    6. Somatostatin negative feedback (refractory period)
    7. Pulsatile delivery (session window gating)

Authors: DooDoo + Claude
Date:    2026-02-11
Version: 1.0
"""

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

N_HELICES = 4
PARTIAL_BUNDLE_MIN = 2
N_QUBITS_TOTAL = 33

SITE1_SIMILARITY_THRESHOLD = 0.50
SITE2_SIMILARITY_THRESHOLD = 0.40

MAX_GROWTH_SIGNAL = 0.35

SOMATOSTATIN_THRESHOLD = 5
REFRACTORY_CYCLES = 3

PULSE_WINDOWS_UTC = {
    "LONDON_OPEN": (7, 9),
    "NY_OPEN": (13, 15),
    "ASIAN_OPEN": (23, 1),
}

MAX_SL_TIGHTENING_PERCENT = 0.105

IGF1_HYPERPLASIA_THRESHOLD = 0.20

FULL_HORMONE_POTENCY = 1.0
PARTIAL_HORMONE_POTENCY = 0.60
MISFOLDED_POTENCY = 0.0

DOM_MIN_POSTERIOR_WR = 0.70
DOM_MIN_PROFIT_FACTOR = 1.5
DOM_EXPIRY_DAYS = 30


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class HormoneHelix:
    te_name: str
    qubit_index: int
    boost_factor: float
    win_rate: float
    profit_factor: float
    direction: int = 0
    activation: float = 0.0
    fitness_score: float = 0.0


@dataclass
class HGHMolecule:
    helices: List[Optional[HormoneHelix]] = field(default_factory=lambda: [None] * 4)
    bridge_1_intact: bool = False
    bridge_2_intact: bool = False
    potency: float = 0.0
    fingerprint: Optional[np.ndarray] = None
    variant: str = "none"


@dataclass
class ReceptorBinding:
    site1_bound: bool = False
    site2_bound: bool = False
    binding_strength: float = 0.0
    site1_similarity: float = 0.0
    site2_similarity: float = 0.0


@dataclass
class GrowthCascade:
    jak2_phosphorylation: float = 0.0
    stat_docking: float = 0.0
    growth_signal: float = 0.0


@dataclass
class HGHResult:
    active: bool = False
    growth_signal: float = 0.0
    molecule: Optional[HGHMolecule] = None
    binding: Optional[ReceptorBinding] = None
    cascade: Optional[GrowthCascade] = None
    igf1_level: float = 0.0
    hyperplasia: bool = False
    second_lot_ratio: float = 0.0
    hypertrophy_boost: float = 0.0
    suppression_reason: str = ""
    helices_used: List[str] = field(default_factory=list)
    quantum_rotations_applied: int = 0
    lipolysis_factor: float = 1.0  # SL tightening multiplier (< 1.0 = tighter)


# ============================================================
# HELPERS
# ============================================================

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    mag_a = float(np.linalg.norm(a))
    mag_b = float(np.linalg.norm(b))
    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0
    return dot / (mag_a * mag_b)


def _classify_regime(bars: np.ndarray, lookback: int = 20) -> str:
    if bars is None or len(bars) < lookback + 5:
        return "ranging"

    close = bars[:, 3] if bars.ndim == 2 else bars
    high = bars[:, 1] if bars.ndim == 2 else bars
    low = bars[:, 2] if bars.ndim == 2 else bars

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    segment = slice(-lookback, None)

    tr = np.maximum(
        high[segment] - low[segment],
        np.maximum(
            np.abs(high[segment] - prev_close[segment]),
            np.abs(low[segment] - prev_close[segment]),
        ),
    )
    atr = float(np.mean(tr))
    atr_recent = float(np.mean(tr[-5:]))

    sma = float(np.mean(close[-lookback:]))
    std = float(np.std(close[-lookback:]))
    bb_width = (2.0 * std) / (sma + 1e-10)

    atr_ratio = atr_recent / (atr + 1e-10)

    if atr_ratio > 1.5 and bb_width > 0.03:
        return "volatile"
    elif atr_ratio > 1.1 or bb_width > 0.02:
        return "trending"
    return "ranging"


def _is_pulse_window() -> bool:
    current_hour = datetime.now(timezone.utc).hour
    for _name, (start_h, end_h) in PULSE_WINDOWS_UTC.items():
        if start_h <= end_h:
            if start_h <= current_hour < end_h:
                return True
        else:
            if current_hour >= start_h or current_hour < end_h:
                return True
    return False


def _build_te_maps(te_activations: List[Dict]) -> Tuple[Dict, Dict]:
    """Build TE name -> qubit index and TE name -> activation dicts."""
    qubit_map: Dict[str, int] = {}
    act_map: Dict[str, Dict] = {}
    for act in te_activations:
        name = act.get("te", "")
        act_map[name] = act
        # Try details.qubit_index first
        qi = act.get("details", {}).get("qubit_index", -1)
        if qi >= 0:
            qubit_map[name] = qi

    # Supplement from TE family definitions
    try:
        from teqa_v3_neural_te import ALL_TE_FAMILIES
        for fam in ALL_TE_FAMILIES:
            qubit_map[fam.name] = fam.qubit_index
    except ImportError:
        pass

    return qubit_map, act_map


# ============================================================
# HGH HORMONE ENGINE
# ============================================================

class HGHHormoneEngine:
    """
    Somatotropic Signal Amplification Engine.

    Reads the domestication database (READ ONLY) to find the strongest
    domesticated TEs, builds a 4-helix hormone molecule, validates its
    structural integrity, attempts receptor dimerization against current
    TE activations, and runs the JAK2/STAT cascade to produce a growth
    signal that gets injected into the quantum circuit.
    """

    def __init__(self, db_path: str = None, log_path: str = None):
        if db_path is None:
            self.db_path = str(Path(__file__).parent / "teqa_domestication.db")
        else:
            self.db_path = db_path

        if log_path is None:
            self.log_path = str(Path(__file__).parent / "hgh_hormone_log.jsonl")
        else:
            self.log_path = log_path

        # Rolling state
        self.previous_activations: Optional[np.ndarray] = None
        self.consecutive_growth_count: int = 0
        self.refractory_remaining: int = 0
        self.last_growth_signal: float = 0.0
        self.cycle_count: int = 0

    # ────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ────────────────────────────────────────────────────

    def run_cycle(
        self,
        bars: np.ndarray,
        symbol: str,
        te_activations: List[Dict],
    ) -> HGHResult:
        """
        Run one full HGH hormone cycle.

        Called from TEQAv3Engine.analyze() AFTER TE activations are
        computed but BEFORE the quantum circuits are built/executed.

        Quantum injection happens separately via inject_rotations(),
        which is called per-neuron in the mosaic loop.

        Args:
            bars: OHLCV array (N, 5)
            symbol: trading instrument
            te_activations: list of TE activation dicts from TEActivationEngine

        Returns:
            HGHResult with growth_signal, hyperplasia flag, hypertrophy boost, etc.
        """
        self.cycle_count += 1
        t0 = time.time()

        # Gate 7: Pulsatile delivery check
        if not _is_pulse_window():
            result = HGHResult(suppression_reason="outside_pulse_window")
            self._store_previous(te_activations)
            return result

        # Gate 6: Somatostatin negative feedback
        if self._check_somatostatin():
            result = HGHResult(suppression_reason="somatostatin_refractory")
            self._store_previous(te_activations)
            return result

        # Phase 1: Synthesize hormone from domesticated TEs
        qubit_map, act_map = _build_te_maps(te_activations)
        molecule = self._synthesize(te_activations, qubit_map, act_map)
        if molecule is None:
            result = HGHResult(suppression_reason="insufficient_domesticated_tes")
            self._store_previous(te_activations)
            return result

        # Phase 2: Disulfide bridge validation
        molecule = self._validate_bridges(molecule, bars)

        # Phase 3: Receptor dimerization
        current_vec = self._activation_vector(te_activations)
        binding = self._attempt_dimerization(molecule, current_vec)

        # Phase 4: JAK2/STAT cascade
        active_dirs = [a["direction"] for a in te_activations if a.get("strength", 0) > 0.5]
        cascade = self._jak2_stat_cascade(molecule, binding, active_dirs)

        # Phase 5A: Lipolysis advisory (compute but don't execute -- BRAIN decides)
        sl_tighten_factor = self._compute_lipolysis_factor(cascade.growth_signal)

        # IGF-1 hyperplasia / hypertrophy
        hyperplasia = False
        second_lot_ratio = 0.0
        if cascade.growth_signal > IGF1_HYPERPLASIA_THRESHOLD:
            hyperplasia = True
            second_lot_ratio = cascade.growth_signal / MAX_GROWTH_SIGNAL

        hypertrophy_boost = cascade.growth_signal * 0.5

        # Update somatostatin tracking
        self._update_somatostatin(cascade.growth_signal)

        # Store for next cycle
        self._store_previous(te_activations)

        helices_used = [h.te_name for h in molecule.helices if h is not None]

        result = HGHResult(
            active=cascade.growth_signal > 0,
            growth_signal=cascade.growth_signal,
            molecule=molecule,
            binding=binding,
            cascade=cascade,
            igf1_level=cascade.growth_signal,
            hyperplasia=hyperplasia,
            second_lot_ratio=second_lot_ratio,
            hypertrophy_boost=hypertrophy_boost,
            helices_used=helices_used,
            quantum_rotations_applied=0,  # Updated per-neuron via inject_rotations()
            lipolysis_factor=sl_tighten_factor,
        )

        elapsed = (time.time() - t0) * 1000
        self._log_cycle(result, symbol, elapsed, sl_tighten_factor)

        return result

    # ────────────────────────────────────────────────────
    # PHASE 1: FOUR-HELIX BUNDLE CONSTRUCTION
    # ────────────────────────────────────────────────────

    def _synthesize(
        self,
        te_activations: List[Dict],
        qubit_map: Dict[str, int],
        act_map: Dict[str, Dict],
    ) -> Optional[HGHMolecule]:
        candidates = self._query_domesticated_tes()
        if not candidates:
            log.debug("HGH: No domesticated patterns in DB")
            return None

        # Score individual TEs across all winning combos
        te_scores: Dict[str, float] = {}
        te_meta: Dict[str, Dict] = {}

        for row in candidates:
            posterior_wr = row["posterior_wr"]
            profit_factor = row["profit_factor"]
            fitness = posterior_wr * math.log(profit_factor + 1)

            for te_name in row["te_combo"].split("+"):
                te_name = te_name.strip()
                if not te_name:
                    continue
                te_scores[te_name] = te_scores.get(te_name, 0.0) + fitness
                if te_name not in te_meta or fitness > te_meta[te_name].get("fitness", 0):
                    te_meta[te_name] = {
                        "posterior_wr": posterior_wr,
                        "profit_factor": profit_factor,
                        "boost_factor": row["boost_factor"],
                        "fitness": fitness,
                    }

        ranked = sorted(te_scores.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < PARTIAL_BUNDLE_MIN:
            log.debug("HGH: Only %d TEs scored, need %d", len(ranked), PARTIAL_BUNDLE_MIN)
            return None

        # Build helices (up to 4)
        helices: List[Optional[HormoneHelix]] = [None] * N_HELICES
        for i, (te_name, score) in enumerate(ranked[:N_HELICES]):
            act = act_map.get(te_name, {})
            meta = te_meta.get(te_name, {})
            helices[i] = HormoneHelix(
                te_name=te_name,
                qubit_index=qubit_map.get(te_name, -1),
                boost_factor=meta.get("boost_factor", 1.0),
                win_rate=meta.get("posterior_wr", 0.5),
                profit_factor=meta.get("profit_factor", 0.0),
                direction=act.get("direction", 0),
                activation=act.get("strength", 0.0),
                fitness_score=score,
            )

        molecule = HGHMolecule(helices=helices)

        # Fingerprint: 33-element vector, non-zero at helix qubits
        fp = np.zeros(N_QUBITS_TOTAL, dtype=np.float64)
        for h in helices:
            if h is not None and 0 <= h.qubit_index < N_QUBITS_TOTAL:
                fp[h.qubit_index] = h.activation * h.boost_factor
        norm = np.linalg.norm(fp)
        if norm > 1e-10:
            fp /= norm
        molecule.fingerprint = fp

        names = [h.te_name for h in helices if h is not None]
        log.info("HGH SYNTHESIZE: bundle=[%s]", ", ".join(names))
        return molecule

    def _query_domesticated_tes(self) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT te_combo, posterior_wr, profit_factor, boost_factor,
                           last_activated, win_count, loss_count
                    FROM domesticated_patterns
                    WHERE domesticated = 1
                      AND posterior_wr >= ?
                      AND profit_factor >= ?
                    ORDER BY posterior_wr * profit_factor DESC
                """, (DOM_MIN_POSTERIOR_WR, DOM_MIN_PROFIT_FACTOR))

                rows = cursor.fetchall()

            now = datetime.now()
            results = []
            for r in rows:
                te_combo, posterior_wr, profit_factor, boost_factor, last_act, wc, lc = r
                if last_act:
                    try:
                        age = (now - datetime.fromisoformat(last_act)).days
                        if age > DOM_EXPIRY_DAYS:
                            continue
                    except (ValueError, TypeError):
                        pass
                results.append({
                    "te_combo": te_combo,
                    "posterior_wr": posterior_wr or 0.5,
                    "profit_factor": profit_factor or 0.0,
                    "boost_factor": boost_factor or 1.0,
                    "win_count": wc or 0,
                    "loss_count": lc or 0,
                })
            return results

        except Exception as e:
            log.warning("HGH: Domestication DB query failed: %s", e)
            return []

    # ────────────────────────────────────────────────────
    # PHASE 2: DISULFIDE BRIDGE VALIDATION
    # ────────────────────────────────────────────────────

    def _validate_bridges(self, molecule: HGHMolecule, bars: np.ndarray) -> HGHMolecule:
        helix_a = molecule.helices[0]  # up
        helix_b = molecule.helices[1]  # up
        helix_c = molecule.helices[2]  # down
        helix_d = molecule.helices[3]  # down

        # Bridge 1: Cys-53 <-> Cys-165 (Helix A <-> Helix C)
        # Regime agreement
        if helix_a is not None and helix_c is not None:
            regime = _classify_regime(bars)
            # Both helices see the same bars, but we check whether their
            # TE signals are consistent with the SAME regime interpretation.
            # If one is a trend-follower and the other is a mean-reverter,
            # they only agree when regime is "ranging" or both are directional.
            dir_a = helix_a.direction
            dir_c = helix_c.direction
            # Regime bridge: both active in same direction OR both neutral
            if dir_a == dir_c or dir_a == 0 or dir_c == 0:
                molecule.bridge_1_intact = True
            else:
                # Opposing directions: bridge holds ONLY in volatile regime
                # (volatile = both can be right at different timescales)
                molecule.bridge_1_intact = (regime == "volatile")
        else:
            molecule.bridge_1_intact = False

        # Bridge 2: Cys-182 <-> Cys-189 (Helix B <-> Helix D)
        # Direction agreement (tighter constraint, only 7 residues apart)
        if helix_b is not None and helix_d is not None:
            dir_b = helix_b.direction
            dir_d = helix_d.direction
            molecule.bridge_2_intact = (
                dir_b == dir_d
                or dir_b == 0
                or dir_d == 0
            )
        else:
            molecule.bridge_2_intact = False

        # Determine potency and variant
        if molecule.bridge_1_intact and molecule.bridge_2_intact:
            molecule.potency = FULL_HORMONE_POTENCY
            molecule.variant = "22kDa"
            log.info("HGH BRIDGES: Both intact -> 22-kDa full potency")
        elif molecule.bridge_1_intact or molecule.bridge_2_intact:
            molecule.potency = PARTIAL_HORMONE_POTENCY
            molecule.variant = "20kDa"
            b1 = "OK" if molecule.bridge_1_intact else "BROKEN"
            b2 = "OK" if molecule.bridge_2_intact else "BROKEN"
            log.info("HGH BRIDGES: Cys53-165=%s Cys182-189=%s -> 20-kDa partial", b1, b2)
        else:
            molecule.potency = MISFOLDED_POTENCY
            molecule.variant = "misfolded"
            log.info("HGH BRIDGES: Both broken -> misfolded, no amplification")

        return molecule

    # ────────────────────────────────────────────────────
    # PHASE 3: RECEPTOR DIMERIZATION
    # ────────────────────────────────────────────────────

    def _attempt_dimerization(
        self,
        molecule: HGHMolecule,
        current_vec: np.ndarray,
    ) -> ReceptorBinding:
        if molecule.potency <= 0 or molecule.fingerprint is None:
            return ReceptorBinding()

        fp = molecule.fingerprint

        # Site 1: current TE activation pattern
        sim1 = _cosine_similarity(fp, current_vec)
        site1 = sim1 >= SITE1_SIMILARITY_THRESHOLD

        # Site 2: previous cycle's TE activation (temporal persistence)
        sim2 = 0.0
        site2 = False
        if self.previous_activations is not None:
            sim2 = _cosine_similarity(fp, self.previous_activations)
            site2 = sim2 >= SITE2_SIMILARITY_THRESHOLD

        if site1 and site2:
            strength = 1.0
            log.info("HGH DIMER: Full dimerization (sim1=%.3f sim2=%.3f)", sim1, sim2)
        elif site1:
            strength = 0.5
            log.info("HGH DIMER: Partial -- Site 1 only (sim1=%.3f sim2=%.3f)", sim1, sim2)
        else:
            strength = 0.0
            log.debug("HGH DIMER: No binding (sim1=%.3f sim2=%.3f)", sim1, sim2)

        return ReceptorBinding(
            site1_bound=site1,
            site2_bound=site2,
            binding_strength=strength,
            site1_similarity=sim1,
            site2_similarity=sim2,
        )

    # ────────────────────────────────────────────────────
    # PHASE 4: JAK2/STAT CASCADE
    # ────────────────────────────────────────────────────

    def _jak2_stat_cascade(
        self,
        molecule: HGHMolecule,
        binding: ReceptorBinding,
        active_directions: List[int],
    ) -> GrowthCascade:
        if binding.binding_strength <= 0:
            return GrowthCascade()

        # JAK2 phosphorylation: mean boost factor of active helices
        boosts = [h.boost_factor for h in molecule.helices if h is not None]
        jak2 = float(np.mean(boosts)) if boosts else 1.0

        # STAT docking: concordance of active TE directions
        stat = 0.0
        if active_directions:
            net = sum(active_directions)
            majority_sign = 1 if net >= 0 else -1
            agreeing = sum(1 for d in active_directions if (d >= 0) == (majority_sign >= 0))
            stat = agreeing / len(active_directions)

        # Nuclear translocation: final growth signal
        raw = (jak2 - 1.0) * stat * binding.binding_strength * molecule.potency
        growth_signal = min(raw, MAX_GROWTH_SIGNAL)
        growth_signal = max(growth_signal, 0.0)

        log.info(
            "HGH JAK2/STAT: jak2=%.3f stat=%.3f bind=%.2f potency=%.2f -> growth=%.4f",
            jak2, stat, binding.binding_strength, molecule.potency, growth_signal,
        )

        return GrowthCascade(
            jak2_phosphorylation=jak2,
            stat_docking=stat,
            growth_signal=growth_signal,
        )

    # ────────────────────────────────────────────────────
    # PHASE 5A: LIPOLYSIS (advisory SL tightening)
    # ────────────────────────────────────────────────────

    def _compute_lipolysis_factor(self, growth_signal: float) -> float:
        """
        Compute SL tightening factor for losing positions.

        Returns a multiplier (0.895 to 1.0) that BRAIN scripts
        can optionally apply to the SL distance of losing trades.
        This method does NOT modify positions -- it's advisory.
        """
        if growth_signal <= 0:
            return 1.0
        tighten = growth_signal * MAX_SL_TIGHTENING_PERCENT / MAX_GROWTH_SIGNAL
        return 1.0 - tighten

    # ────────────────────────────────────────────────────
    # PHASE 5B: IGF-1 BRIDGE (quantum circuit injection)
    # ────────────────────────────────────────────────────

    def _inject_quantum_rotations(
        self,
        molecule: HGHMolecule,
        growth_signal: float,
        genome_circuit: Optional['QuantumCircuit'],
        neural_circuit: Optional['QuantumCircuit'],
    ) -> int:
        """
        Inject hormone-driven RY rotations into the quantum circuits.

        This is the core "strapping" -- the hormone physically rotates
        helix qubits further toward |1> (activated), amplifying their
        influence in the measurement distribution.

        Also applies a global RZ phase bias proportional to growth_signal.

        Returns number of rotations applied.
        """
        if not QISKIT_AVAILABLE:
            return 0

        rotations = 0
        N_GENOME = 25
        N_NEURAL = 8

        for helix in molecule.helices:
            if helix is None or helix.qubit_index < 0:
                continue

            extra_theta = helix.activation * growth_signal * (math.pi / 2)
            qi = helix.qubit_index

            if qi < N_GENOME and genome_circuit is not None:
                try:
                    genome_circuit.ry(extra_theta, qi)
                    rotations += 1
                except Exception as e:
                    log.debug("HGH: Failed to inject RY into genome qubit %d: %s", qi, e)

            elif qi >= N_GENOME and neural_circuit is not None:
                neural_qi = qi - N_GENOME
                if neural_qi < N_NEURAL:
                    try:
                        neural_circuit.ry(extra_theta, neural_qi)
                        rotations += 1
                    except Exception as e:
                        log.debug("HGH: Failed to inject RY into neural qubit %d: %s", neural_qi, e)

        # Global RZ phase bias on all qubits
        rz_angle = growth_signal * (math.pi / 4)

        if genome_circuit is not None:
            for qi in range(min(N_GENOME, genome_circuit.num_qubits)):
                try:
                    genome_circuit.rz(rz_angle, qi)
                    rotations += 1
                except Exception:
                    pass

        if neural_circuit is not None:
            for qi in range(min(N_NEURAL, neural_circuit.num_qubits)):
                try:
                    neural_circuit.rz(rz_angle, qi)
                    rotations += 1
                except Exception:
                    pass

        if rotations > 0:
            log.info("HGH IGF-1: Injected %d rotations (growth=%.4f)", rotations, growth_signal)

        return rotations

    # ────────────────────────────────────────────────────
    # PUBLIC: Per-neuron quantum injection
    # ────────────────────────────────────────────────────

    def inject_rotations(
        self,
        result: HGHResult,
        genome_circuit: Optional['QuantumCircuit'],
        neural_circuit: Optional['QuantumCircuit'],
    ) -> int:
        """
        Inject hormone rotations into per-neuron quantum circuits.

        Called once per neuron in the TEQAv3Engine mosaic loop, AFTER
        the circuits are built but BEFORE they are executed/measured.

        This is the IGF-1 bridge: the hormone physically biases the
        quantum circuits toward domesticated TE directions.

        Args:
            result: HGHResult from run_cycle() (must have active molecule)
            genome_circuit: 25-qubit genome circuit for this neuron
            neural_circuit: 8-qubit neural circuit for this neuron

        Returns:
            Number of rotations injected into this neuron's circuits.
        """
        if not result.active or result.molecule is None or result.growth_signal <= 0:
            return 0
        return self._inject_quantum_rotations(
            result.molecule, result.growth_signal, genome_circuit, neural_circuit
        )

    # ────────────────────────────────────────────────────
    # PHASE 6: SOMATOSTATIN NEGATIVE FEEDBACK
    # ────────────────────────────────────────────────────

    def _check_somatostatin(self) -> bool:
        if self.refractory_remaining > 0:
            self.refractory_remaining -= 1
            log.info(
                "HGH SOMATOSTATIN: Refractory period, %d cycles remaining",
                self.refractory_remaining,
            )
            return True
        return False

    def _update_somatostatin(self, growth_signal: float):
        if growth_signal > IGF1_HYPERPLASIA_THRESHOLD:
            self.consecutive_growth_count += 1
        else:
            self.consecutive_growth_count = max(0, self.consecutive_growth_count - 1)

        if self.consecutive_growth_count >= SOMATOSTATIN_THRESHOLD:
            self.refractory_remaining = REFRACTORY_CYCLES
            self.consecutive_growth_count = 0
            log.info(
                "HGH SOMATOSTATIN: Overactivation! Entering %d-cycle refractory",
                REFRACTORY_CYCLES,
            )

        self.last_growth_signal = growth_signal

    # ────────────────────────────────────────────────────
    # STATE HELPERS
    # ────────────────────────────────────────────────────

    def _activation_vector(self, te_activations: List[Dict]) -> np.ndarray:
        vec = np.zeros(N_QUBITS_TOTAL, dtype=np.float64)
        try:
            from teqa_v3_neural_te import ALL_TE_FAMILIES
            name_to_qi = {f.name: f.qubit_index for f in ALL_TE_FAMILIES}
        except ImportError:
            name_to_qi = {}

        for act in te_activations:
            name = act.get("te", "")
            qi = name_to_qi.get(name, -1)
            if 0 <= qi < N_QUBITS_TOTAL:
                vec[qi] = act.get("strength", 0.0) * act.get("direction", 0)
        return vec

    def _store_previous(self, te_activations: List[Dict]):
        self.previous_activations = self._activation_vector(te_activations)

    # ────────────────────────────────────────────────────
    # LOGGING
    # ────────────────────────────────────────────────────

    def _log_cycle(self, result: HGHResult, symbol: str, elapsed_ms: float, sl_factor: float):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "cycle": self.cycle_count,
            "active": result.active,
            "growth_signal": round(result.growth_signal, 6),
            "variant": result.molecule.variant if result.molecule else "none",
            "potency": result.molecule.potency if result.molecule else 0.0,
            "bridge_1": result.molecule.bridge_1_intact if result.molecule else False,
            "bridge_2": result.molecule.bridge_2_intact if result.molecule else False,
            "binding_strength": result.binding.binding_strength if result.binding else 0.0,
            "site1_sim": round(result.binding.site1_similarity, 4) if result.binding else 0.0,
            "site2_sim": round(result.binding.site2_similarity, 4) if result.binding else 0.0,
            "jak2": round(result.cascade.jak2_phosphorylation, 4) if result.cascade else 0.0,
            "stat": round(result.cascade.stat_docking, 4) if result.cascade else 0.0,
            "hyperplasia": result.hyperplasia,
            "second_lot_ratio": round(result.second_lot_ratio, 3),
            "hypertrophy_boost": round(result.hypertrophy_boost, 4),
            "sl_tighten_factor": round(sl_factor, 4),
            "helices": result.helices_used,
            "rotations_applied": result.quantum_rotations_applied,
            "somatostatin_count": self.consecutive_growth_count,
            "refractory_remaining": self.refractory_remaining,
            "suppression": result.suppression_reason,
            "elapsed_ms": round(elapsed_ms, 1),
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            log.debug("HGH log write failed: %s", e)

        if result.active:
            log.info(
                "HGH RESULT: %s %s growth=%.4f variant=%s bind=%.2f "
                "hyperplasia=%s boost=+%.4f rotations=%d [%.0fms]",
                symbol,
                "ACTIVE" if result.active else "INACTIVE",
                result.growth_signal,
                entry["variant"],
                entry["binding_strength"],
                result.hyperplasia,
                result.hypertrophy_boost,
                result.quantum_rotations_applied,
                elapsed_ms,
            )

    # ────────────────────────────────────────────────────
    # PUBLIC: Get signal JSON section for bridge compatibility
    # ────────────────────────────────────────────────────

    def result_to_signal_json(self, result: HGHResult) -> Dict:
        """Convert HGHResult into a dict for inclusion in te_quantum_signal.json."""
        return {
            "hgh_hormone": {
                "active": result.active,
                "growth_signal": round(result.growth_signal, 6),
                "variant": result.molecule.variant if result.molecule else "none",
                "potency": result.molecule.potency if result.molecule else 0.0,
                "bridge_1_intact": result.molecule.bridge_1_intact if result.molecule else False,
                "bridge_2_intact": result.molecule.bridge_2_intact if result.molecule else False,
                "binding_strength": result.binding.binding_strength if result.binding else 0.0,
                "jak2": round(result.cascade.jak2_phosphorylation, 4) if result.cascade else 0.0,
                "stat_docking": round(result.cascade.stat_docking, 4) if result.cascade else 0.0,
                "hyperplasia": result.hyperplasia,
                "second_lot_ratio": round(result.second_lot_ratio, 3),
                "hypertrophy_boost": round(result.hypertrophy_boost, 4),
                "helices": result.helices_used,
                "rotations_applied": result.quantum_rotations_applied,
                "suppression": result.suppression_reason,
            }
        }

    # ────────────────────────────────────────────────────
    # PUBLIC: Get hormone status summary
    # ────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return current hormone engine state for monitoring."""
        return {
            "cycle_count": self.cycle_count,
            "last_growth_signal": self.last_growth_signal,
            "consecutive_growth_count": self.consecutive_growth_count,
            "refractory_remaining": self.refractory_remaining,
            "has_previous_activations": self.previous_activations is not None,
            "in_pulse_window": _is_pulse_window(),
        }
