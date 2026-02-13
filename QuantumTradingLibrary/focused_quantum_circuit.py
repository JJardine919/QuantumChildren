"""
Focused Quantum Circuit Engine
==============================
Builds right-sized quantum circuits from only active (or only dormant) TEs.
Instead of firing all 33 qubits every cycle, we ask the ones paying attention
separately from the quiet ones -- because sometimes the quiet ones see what
everyone else misses.

Architecture:
  - Qubit remapping: active TEs get compact indices 0..N-1
  - Entanglement: only between TEs that are BOTH in the set
  - Same rotation formulas as teqa_v3_neural_te.py
  - Results fused with the same 60/40 genome/neural weighting

Authors: DooDoo + Claude
Date:    2026-02-13
Version: 1.0
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from teqa_v3_neural_te import (
    ALL_TE_FAMILIES,
    TEClass,
    N_QUBITS_GENOME,
    N_QUBITS_NEURAL,
    SHOCK_NORMAL,
)

log = logging.getLogger(__name__)


# ============================================================
# ENTANGLEMENT MAP (from teqa_v3_neural_te.py)
# ============================================================

# Class I chain: qubits 0-10 (RETROTRANSPOSON)
CLASS_I_RANGE = range(0, 11)
# Class II chain: qubits 11-24 (DNA_TRANSPOSON)
CLASS_II_RANGE = range(11, 25)

# Cross-class bridges (original qubit indices)
CROSS_CLASS_BRIDGES = [
    (6, 16),   # LINE <-> Mariner
    (3, 13),   # Ty3/gypsy <-> Helitron
]

# Neural entanglement (original qubit indices, 25-32)
NEURAL_ENTANGLEMENT = [
    (25, 26),  # L1_Neuronal <-> L1_Somatic
    (27, 25),  # HERV_Synapse <-> L1_Neuronal
    (28, 29),  # SVA_Regulatory <-> Alu_Exonization
]

# TRIM28 (qubit 30) suppresses others via CZ
TRIM28_QUBIT = 30

# piwiRNA (qubit 31) targets L1 (qubits 25, 26)
PIWI_QUBIT = 31
PIWI_TARGETS = [25, 26]

# Arc_Capsid (qubit 32) connects to L1_Neuronal (25) and HERV (27)
ARC_QUBIT = 32
ARC_TARGETS = [25, 27]


# Build lookup: te_name -> TEFamily
TE_BY_NAME = {te.name: te for te in ALL_TE_FAMILIES}


class FocusedQuantumEngine:
    """
    Builds right-sized quantum circuits from subsets of TEs.

    Instead of always using 25 genome + 8 neural qubits, this engine
    remaps only the active (or dormant) TEs to compact qubit indices
    and builds tighter circuits with less noise from zero-angle qubits.
    """

    def __init__(self, base_shots: int = 8192):
        self.base_shots = base_shots
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
        else:
            log.warning("Qiskit not available -- focused circuits will use classical fallback")
            self.simulator = None

    def run_focused(
        self,
        activations: List[Dict],
        active_te_names: List[str],
        neurons: list,
        shock_level: float,
        total_shots: int,
    ) -> Dict:
        """
        Build and execute a focused circuit from only the active TEs.

        Args:
            activations: full list of TE activations (all 33)
            active_te_names: names of TEs with strength > 0.3
            neurons: list of NeuralGenome objects
            shock_level: current genomic shock score
            total_shots: base shot count from the main engine

        Returns:
            Dict with direction, confidence, vote_long, vote_short, etc.
        """
        if not active_te_names:
            return self._neutral_result(0, "focused")

        # Pick the first neuron for the focused circuit (majority voter)
        neuron = neurons[0] if neurons else None

        result = self._build_and_execute(
            activations, active_te_names, neuron, shock_level, total_shots,
            circuit_label="focused"
        )
        return result

    def run_dormant(
        self,
        activations: List[Dict],
        dormant_te_names: List[str],
        neurons: list,
        shock_level: float,
        total_shots: int,
    ) -> Dict:
        """
        Build and execute a dormant circuit from only the sleeping TEs.

        Key insight: raw dormant strengths (0.0-0.3) produce tiny rotation
        angles that collapse all qubits to |0>, always voting SHORT.
        Instead, we normalize within the dormant population -- the strongest
        dormant TE becomes 1.0, and we ask "among the quiet ones, who's
        whispering loudest?" This produces real superposition and meaningful
        directional votes.
        """
        if not dormant_te_names:
            return self._neutral_result(0, "dormant")

        # Normalize dormant activations within their own population
        normalized = self._normalize_dormant(activations, dormant_te_names)

        neuron = neurons[0] if neurons else None

        result = self._build_and_execute(
            normalized, dormant_te_names, neuron, shock_level, total_shots,
            circuit_label="dormant"
        )
        return result

    @staticmethod
    def _normalize_dormant(
        activations: List[Dict],
        dormant_te_names: List[str],
    ) -> List[Dict]:
        """
        Rescale dormant TE strengths so they fill [0, 1] within their own set.
        A TE at 0.28 becomes ~1.0, a TE at 0.05 becomes ~0.18.
        Direction is preserved -- we only scale the magnitude.
        Non-dormant TEs are passed through unchanged.
        """
        dormant_set = set(dormant_te_names)

        # Find the max strength among dormant TEs
        max_str = 0.0
        for act in activations:
            if act["te"] in dormant_set:
                max_str = max(max_str, act.get("strength", 0.0))

        if max_str < 1e-6:
            # All truly dead -- nothing to normalize
            return activations

        # Build normalized copy
        normalized = []
        for act in activations:
            if act["te"] in dormant_set:
                new_act = dict(act)
                new_act["strength"] = act.get("strength", 0.0) / max_str
                normalized.append(new_act)
            else:
                normalized.append(act)

        return normalized

    def _build_and_execute(
        self,
        activations: List[Dict],
        te_names: List[str],
        neuron,
        shock_level: float,
        total_shots: int,
        circuit_label: str = "subset",
    ) -> Dict:
        """
        Core method: build a right-sized circuit for a subset of TEs.

        Splits TEs into genome (qubit_index < 25) and neural (>= 25),
        builds separate genome/neural circuits with compact qubit indices,
        fuses results with 60/40 weighting.
        """
        # Separate into genome and neural TEs
        genome_te_names = []
        neural_te_names = []
        for name in te_names:
            te = TE_BY_NAME.get(name)
            if te is None:
                continue
            if te.qubit_index < N_QUBITS_GENOME:
                genome_te_names.append(name)
            else:
                neural_te_names.append(name)

        n_genome = len(genome_te_names)
        n_neural = len(neural_te_names)
        n_total = n_genome + n_neural

        if n_total == 0:
            return self._neutral_result(0, circuit_label)

        # Scale shots proportionally to qubit count (floor at 1024)
        scaled_shots = max(1024, int(total_shots * (n_total / 33)))
        genome_shots = max(512, int(scaled_shots * 0.6)) if n_genome > 0 else 0
        neural_shots = max(512, int(scaled_shots * 0.4)) if n_neural > 0 else 0

        # Build activation lookup
        act_by_name = {a["te"]: a for a in activations}

        # --- Genome circuit ---
        if n_genome > 0:
            genome_result = self._build_genome_focused(
                genome_te_names, act_by_name, neuron, shock_level, genome_shots
            )
        else:
            genome_result = self._classical_fallback(0)

        # Extract genome signal for neural injection
        genome_signal = genome_result.get("vote_long", 0.5) - genome_result.get("vote_short", 0.5)

        # --- Neural circuit ---
        if n_neural > 0:
            neural_result = self._build_neural_focused(
                neural_te_names, act_by_name, neuron, shock_level, neural_shots,
                genome_signal=genome_signal
            )
        else:
            neural_result = self._classical_fallback(0)

        # --- Fuse ---
        fused = self._fuse(genome_result, neural_result, n_genome, n_neural)

        # Determine direction and confidence
        vote_long = fused["vote_long"]
        vote_short = fused["vote_short"]

        if vote_long > vote_short * 1.1:
            direction = 1
            confidence = vote_long / (vote_long + vote_short + 1e-10)
        elif vote_short > vote_long * 1.1:
            direction = -1
            confidence = vote_short / (vote_long + vote_short + 1e-10)
        else:
            direction = 0
            confidence = 0.0

        return {
            "direction": direction,
            "confidence": float(confidence),
            "vote_long": float(vote_long),
            "vote_short": float(vote_short),
            "n_qubits": n_total,
            "n_genome_qubits": n_genome,
            "n_neural_qubits": n_neural,
            "shots_used": genome_shots + neural_shots,
            "circuit_label": circuit_label,
            "fused": fused,
        }

    def _build_genome_focused(
        self,
        te_names: List[str],
        act_by_name: Dict[str, Dict],
        neuron,
        shock_level: float,
        shots: int,
    ) -> Dict:
        """Build a compact genome circuit for only the specified TEs."""
        if not QISKIT_AVAILABLE or self.simulator is None:
            return self._classical_fallback(len(te_names))

        n = len(te_names)
        if n == 0:
            return self._classical_fallback(0)

        # Build qubit remapping: original_index -> compact_index
        remap = {}
        te_list = []  # ordered list of (compact_idx, original_te, activation)
        for compact_idx, name in enumerate(sorted(
            te_names, key=lambda nm: TE_BY_NAME[nm].qubit_index
        )):
            te = TE_BY_NAME[name]
            remap[te.qubit_index] = compact_idx
            te_list.append((compact_idx, te, act_by_name.get(name, {})))

        qc = QuantumCircuit(n, n)

        # Layer 1: RY rotations
        for compact_idx, te, act in te_list:
            strength = act.get("strength", 0.0)
            direction = act.get("direction", 0)
            angle = strength * math.pi * (1 if direction >= 0 else -1)

            # Neuron L1 modifier (only if target qubit is in our set)
            if neuron and te.qubit_index in getattr(neuron, 'activation_modifiers', {}):
                mod = neuron.activation_modifiers[te.qubit_index]
                if mod < 0:
                    angle = -angle
                else:
                    angle *= mod

            # Shock amplification
            if shock_level > SHOCK_NORMAL:
                angle *= min(1.5, shock_level / SHOCK_NORMAL)

            qc.ry(angle, compact_idx)

        # Layer 2: Entanglement (only between TEs both in our set)
        # Class I chain: consecutive pairs where both are active
        active_class_i = sorted([
            remap[te.qubit_index] for te in
            [TE_BY_NAME[n] for n in te_names]
            if te.qubit_index in CLASS_I_RANGE and te.qubit_index in remap
        ])
        for i in range(len(active_class_i) - 1):
            qc.cx(active_class_i[i], active_class_i[i + 1])

        # Class II chain
        active_class_ii = sorted([
            remap[te.qubit_index] for te in
            [TE_BY_NAME[n] for n in te_names]
            if te.qubit_index in CLASS_II_RANGE and te.qubit_index in remap
        ])
        for i in range(len(active_class_ii) - 1):
            qc.cx(active_class_ii[i], active_class_ii[i + 1])

        # Cross-class bridges (only if both endpoints are active)
        for src_orig, tgt_orig in CROSS_CLASS_BRIDGES:
            if src_orig in remap and tgt_orig in remap:
                qc.cx(remap[src_orig], remap[tgt_orig])

        # Layer 3: Neuron-specific rewiring
        if neuron:
            for ins in getattr(neuron, 'insertions', []):
                if (ins.effect == "rewire" and
                    ins.target_qubit in remap and
                    ins.rewire_target in remap and
                    ins.target_qubit < N_QUBITS_GENOME and
                    ins.rewire_target < N_QUBITS_GENOME):
                    qc.cx(remap[ins.target_qubit], remap[ins.rewire_target])

        # Layer 4: Second rotation (30% strength)
        for compact_idx, te, act in te_list:
            strength = act.get("strength", 0.0)
            direction = act.get("direction", 0)
            angle2 = strength * math.pi * 0.3 * (1 if direction >= 0 else -1)
            qc.ry(angle2, compact_idx)

        qc.measure(range(n), range(n))

        # Execute
        return self._execute(qc, shots, n)

    def _build_neural_focused(
        self,
        te_names: List[str],
        act_by_name: Dict[str, Dict],
        neuron,
        shock_level: float,
        shots: int,
        genome_signal: float = 0.0,
    ) -> Dict:
        """Build a compact neural circuit for only the specified neural TEs."""
        if not QISKIT_AVAILABLE or self.simulator is None:
            return self._classical_fallback(len(te_names))

        n = len(te_names)
        if n == 0:
            return self._classical_fallback(0)

        # Build qubit remapping: original_index -> compact_index
        remap = {}
        te_list = []
        for compact_idx, name in enumerate(sorted(
            te_names, key=lambda nm: TE_BY_NAME[nm].qubit_index
        )):
            te = TE_BY_NAME[name]
            remap[te.qubit_index] = compact_idx
            te_list.append((compact_idx, te, act_by_name.get(name, {})))

        qc = QuantumCircuit(n, n)

        # Layer 1: RY rotations
        for compact_idx, te, act in te_list:
            strength = act.get("strength", 0.0)
            direction = act.get("direction", 0)
            angle = strength * math.pi * (1 if direction >= 0 else -1)

            if neuron and te.qubit_index in getattr(neuron, 'activation_modifiers', {}):
                mod = neuron.activation_modifiers[te.qubit_index]
                if mod < 0:
                    angle = -angle
                else:
                    angle *= mod

            if shock_level > SHOCK_NORMAL:
                angle *= min(1.5, shock_level / SHOCK_NORMAL)

            qc.ry(angle, compact_idx)

        # Genome signal injection: rotate L1_Neuronal (qubit 25) if present
        if 25 in remap:
            genome_angle = genome_signal * math.pi * 0.5
            qc.ry(genome_angle, remap[25])

        # Neural entanglement (only if both endpoints are active)
        for src_orig, tgt_orig in NEURAL_ENTANGLEMENT:
            if src_orig in remap and tgt_orig in remap:
                qc.cx(remap[src_orig], remap[tgt_orig])

        # TRIM28 suppression via CZ (only if TRIM28 and target are both active)
        if TRIM28_QUBIT in remap:
            trim28_compact = remap[TRIM28_QUBIT]
            for orig_idx in remap:
                if orig_idx != TRIM28_QUBIT:
                    qc.cz(trim28_compact, remap[orig_idx])

        # piwiRNA targets (only if both present)
        if PIWI_QUBIT in remap:
            piwi_compact = remap[PIWI_QUBIT]
            for target_orig in PIWI_TARGETS:
                if target_orig in remap:
                    qc.cz(piwi_compact, remap[target_orig])

        # Arc_Capsid connections (only if both present)
        if ARC_QUBIT in remap:
            arc_compact = remap[ARC_QUBIT]
            for target_orig in ARC_TARGETS:
                if target_orig in remap:
                    qc.cx(arc_compact, remap[target_orig])

        # Neuron-specific rewiring (neural compartment only)
        if neuron:
            for ins in getattr(neuron, 'insertions', []):
                if ins.effect == "rewire":
                    src_orig = ins.target_qubit
                    tgt_orig = ins.rewire_target
                    if (src_orig in remap and tgt_orig in remap and
                        src_orig >= N_QUBITS_GENOME and tgt_orig >= N_QUBITS_GENOME):
                        qc.cx(remap[src_orig], remap[tgt_orig])

        # Layer 4: Second rotation (30% strength)
        for compact_idx, te, act in te_list:
            strength = act.get("strength", 0.0)
            direction = act.get("direction", 0)
            angle2 = strength * math.pi * 0.3 * (1 if direction >= 0 else -1)
            qc.ry(angle2, compact_idx)

        qc.measure(range(n), range(n))

        return self._execute(qc, shots, n)

    def _execute(self, qc: 'QuantumCircuit', shots: int, n_qubits: int) -> Dict:
        """Execute a quantum circuit and analyze results."""
        try:
            job = self.simulator.run(qc, shots=shots)
            counts = job.result().get_counts()
            return self._analyze_counts(counts, shots, n_qubits)
        except Exception as e:
            log.error("Focused circuit execution failed: %s", e)
            return self._classical_fallback(n_qubits)

    def _analyze_counts(self, counts: Dict[str, int], total_shots: int, n_qubits: int) -> Dict:
        """Analyze measurement results (same logic as TEQAQuantumEngine)."""
        n_unique = len(counts)
        n_possible = 2 ** n_qubits if n_qubits > 0 else 1

        probs = np.array(list(counts.values())) / total_shots
        shannon = float(-np.sum(probs * np.log2(probs + 1e-20)))

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
            "coverage": n_unique / n_possible if n_possible > 0 else 0.0,
            "shannon_entropy": shannon,
            "max_entropy": float(n_qubits),
            "novelty": shannon / n_qubits if n_qubits > 0 else 0.0,
            "vote_long": float(vote_long),
            "vote_short": float(vote_short),
            "top_state": sorted_states[0][0] if sorted_states else "",
            "top_count": sorted_states[0][1] if sorted_states else 0,
        }

    def _fuse(self, genome_result: Dict, neural_result: Dict,
              n_genome: int, n_neural: int) -> Dict:
        """
        Fuse genome and neural results with 60/40 weighting.
        Same logic as TEQAQuantumEngine.fuse_results().
        """
        g_long = genome_result.get("vote_long", 0.5)
        g_short = genome_result.get("vote_short", 0.5)
        n_long = neural_result.get("vote_long", 0.5)
        n_short = neural_result.get("vote_short", 0.5)

        # If one compartment is empty, give full weight to the other
        if n_genome == 0 and n_neural > 0:
            fused_long = n_long
            fused_short = n_short
        elif n_neural == 0 and n_genome > 0:
            fused_long = g_long
            fused_short = g_short
        else:
            fused_long = g_long * 0.6 + n_long * 0.4
            fused_short = g_short * 0.6 + n_short * 0.4

        g_entropy = genome_result.get("shannon_entropy", 0)
        n_entropy = neural_result.get("shannon_entropy", 0)
        g_max = genome_result.get("max_entropy", n_genome)
        n_max = neural_result.get("max_entropy", n_neural)
        fused_entropy = g_entropy + n_entropy
        fused_max = g_max + n_max

        return {
            "n_shots": genome_result.get("n_shots", 0) + neural_result.get("n_shots", 0),
            "n_unique_states": genome_result.get("n_unique_states", 0) + neural_result.get("n_unique_states", 0),
            "coverage": (genome_result.get("coverage", 0) + neural_result.get("coverage", 0)) / 2,
            "shannon_entropy": float(fused_entropy),
            "max_entropy": float(fused_max),
            "novelty": float(fused_entropy / fused_max) if fused_max > 0 else 0.0,
            "vote_long": float(fused_long),
            "vote_short": float(fused_short),
            "top_state": genome_result.get("top_state", "") + "|" + neural_result.get("top_state", ""),
            "top_count": genome_result.get("top_count", 0),
        }

    def _classical_fallback(self, n_qubits: int) -> Dict:
        """Fallback when quantum simulator not available or zero qubits."""
        return {
            "n_shots": 0,
            "n_unique_states": 0,
            "n_possible_states": 2 ** n_qubits if n_qubits > 0 else 1,
            "coverage": 0.0,
            "shannon_entropy": 0.0,
            "max_entropy": float(n_qubits),
            "novelty": 0.0,
            "vote_long": 0.5,
            "vote_short": 0.5,
            "top_state": "",
            "top_count": 0,
        }

    def _neutral_result(self, n_qubits: int, label: str) -> Dict:
        """Return a neutral result when no TEs are in the set."""
        return {
            "direction": 0,
            "confidence": 0.0,
            "vote_long": 0.5,
            "vote_short": 0.5,
            "n_qubits": n_qubits,
            "n_genome_qubits": 0,
            "n_neural_qubits": 0,
            "shots_used": 0,
            "circuit_label": label,
            "fused": self._classical_fallback(n_qubits),
        }

    @staticmethod
    def compare_results(
        full_result: Dict,
        focused_result: Dict,
        dormant_result: Dict,
    ) -> Dict:
        """
        Compare full, focused, and dormant circuit results.
        Flags when dormant circuit disagrees with focused (contrarian signal).
        """
        full_dir = full_result.get("direction", 0)
        focused_dir = focused_result.get("direction", 0)
        dormant_dir = dormant_result.get("direction", 0)

        agreement_full_focused = (full_dir == focused_dir) or full_dir == 0 or focused_dir == 0
        agreement_full_dormant = (full_dir == dormant_dir) or full_dir == 0 or dormant_dir == 0
        contrarian = (
            dormant_dir != 0 and
            focused_dir != 0 and
            dormant_dir != focused_dir
        )

        # Which circuit had highest confidence?
        circuits = {
            "full": full_result.get("confidence", 0),
            "focused": focused_result.get("confidence", 0),
            "dormant": dormant_result.get("confidence", 0),
        }
        highest = max(circuits, key=circuits.get)

        return {
            "highest_confidence_circuit": highest,
            "full_focused_agree": agreement_full_focused,
            "full_dormant_agree": agreement_full_dormant,
            "contrarian_alert": contrarian,
            "full_direction": full_dir,
            "focused_direction": focused_dir,
            "dormant_direction": dormant_dir,
            "full_confidence": circuits["full"],
            "focused_confidence": circuits["focused"],
            "dormant_confidence": circuits["dormant"],
        }
