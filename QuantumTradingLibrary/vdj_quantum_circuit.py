"""
VDJ Quantum Circuit -- 16-Qubit Recombination Circuit
======================================================
Purpose-built quantum circuit for V(D)J segment selection.
Separate from but informed by the main TEQA 33-qubit circuit.

Qubit mapping:
    Qubits  0-5:  V segment selector (2^6=64, encodes 33 V segments)
    Qubits  6-9:  D segment selector (2^4=16, encodes 13 D segments)
    Qubits 10-13: J segment selector (2^4=16, encodes 10 J segments)
    Qubit  14:    RSS compatibility flag
    Qubit  15:    Junctional diversity seed

The circuit creates a superposition of all valid V+D+J combinations,
weighted by current TE activation strengths, then collapses to select
one specific antibody (micro-strategy).

Authors: DooDoo + Claude
Date:    2026-02-09
Parent:  ALGORITHM_VDJ_RECOMBINATION v1.0
"""

import math
import logging
from typing import Dict, List, Optional

import numpy as np

from vdj_segments import (
    V_SEGMENTS, D_SEGMENTS, J_SEGMENTS,
    V_RSS, D_RSS, J_RSS,
    V_NAMES, D_NAMES, J_NAMES,
    N_V, N_D, N_J,
    rss_compatible,
)

log = logging.getLogger(__name__)

VDJ_N_QUBITS = 16

# Try to import Qiskit; fall back to classical simulation if unavailable
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    log.info("Qiskit not available; VDJ circuit will use classical fallback")


# ============================================================
# AMPLITUDE COMPUTATION
# ============================================================

def compute_v_amplitudes(te_activations: List[Dict]) -> List[float]:
    """
    Convert 33 TE activation strengths into 6-qubit rotation angles.

    Each V segment maps to one TE family. The TE's activation strength
    determines how likely that V segment is to be selected.

    Returns 6 rotation angles for RY gates on qubits 0-5.
    """
    v_strengths = []
    for v_name in V_NAMES:
        v_def = V_SEGMENTS[v_name]
        te_source = v_def["te_source"]
        act = next((a for a in te_activations if a.get("te") == te_source), None)
        strength = act["strength"] if act else 0.0
        v_strengths.append(max(0.0, strength))

    # Pad to 64 (2^6) with zeros for unused states
    while len(v_strengths) < 64:
        v_strengths.append(0.0)

    # Compute 6 rotation angles via bit-weighted averaging
    amplitudes = [0.0] * 6
    for bit in range(6):
        set_strengths = []
        for idx in range(min(33, len(v_strengths))):
            if (idx >> bit) & 1:
                set_strengths.append(v_strengths[idx])
        if set_strengths:
            amplitudes[bit] = sum(set_strengths) / len(set_strengths)

    return amplitudes


def compute_d_amplitudes(te_activations: List[Dict], shock_level: float) -> List[float]:
    """
    Compute 4-qubit rotation angles for D segment selection.

    D segments are regime classifiers. Their weights are derived from
    the collective activation patterns of related TE families and
    the current shock level.

    Returns 4 rotation angles for RY gates on qubits 6-9.
    """
    d_strengths = []
    for d_name in D_NAMES:
        d_def = D_SEGMENTS[d_name]
        regime = d_def["regime"]

        # Base strength from param_shift magnitudes (proxy for regime strength)
        param_shift = d_def.get("param_shift", {})
        base = sum(abs(v) for v in param_shift.values()) / max(1, len(param_shift))

        # Shock-sensitive regimes get boosted under high shock
        if regime in ("HIGH_VOLATILITY", "NEWS_SHOCK"):
            base *= (1.0 + shock_level * 0.5)
        elif regime in ("RANGE_BOUND", "LOW_LIQUIDITY"):
            base *= max(0.2, 1.0 - shock_level * 0.3)

        d_strengths.append(base)

    # Pad to 16 (2^4)
    while len(d_strengths) < 16:
        d_strengths.append(0.0)

    # Normalize
    total = sum(d_strengths) + 1e-10
    d_strengths = [s / total for s in d_strengths]

    amplitudes = [0.0] * 4
    for bit in range(4):
        set_strengths = []
        for idx in range(min(13, len(d_strengths))):
            if (idx >> bit) & 1:
                set_strengths.append(d_strengths[idx])
        if set_strengths:
            amplitudes[bit] = sum(set_strengths) / len(set_strengths)

    return amplitudes


def compute_j_amplitudes(exit_bias: str = "trail") -> List[float]:
    """
    Compute 4-qubit rotation angles for J segment selection.

    J segments are exit strategies. Biased by the D segment's exit_bias.

    Returns 4 rotation angles for RY gates on qubits 10-13.
    """
    j_strengths = []
    bias_map = {
        "trail": {"TRAILING_STOP": 1.5, "DYNAMIC": 1.2, "FIXED_TARGET": 0.8, "TIME_BASED": 0.6},
        "fixed_target": {"TRAILING_STOP": 0.7, "DYNAMIC": 1.0, "FIXED_TARGET": 1.5, "TIME_BASED": 0.8},
        "time_based": {"TRAILING_STOP": 0.6, "DYNAMIC": 0.8, "FIXED_TARGET": 0.8, "TIME_BASED": 1.5},
    }
    biases = bias_map.get(exit_bias, {"TRAILING_STOP": 1.0, "DYNAMIC": 1.0, "FIXED_TARGET": 1.0, "TIME_BASED": 1.0})

    for j_name in J_NAMES:
        j_def = J_SEGMENTS[j_name]
        exit_type = j_def["exit_type"]
        j_strengths.append(biases.get(exit_type, 1.0))

    # Pad to 16 (2^4)
    while len(j_strengths) < 16:
        j_strengths.append(0.0)

    total = sum(j_strengths) + 1e-10
    j_strengths = [s / total for s in j_strengths]

    amplitudes = [0.0] * 4
    for bit in range(4):
        set_strengths = []
        for idx in range(min(10, len(j_strengths))):
            if (idx >> bit) & 1:
                set_strengths.append(j_strengths[idx])
        if set_strengths:
            amplitudes[bit] = sum(set_strengths) / len(set_strengths)

    return amplitudes


# ============================================================
# QUANTUM CIRCUIT BUILDER
# ============================================================

def build_vdj_circuit(
    te_activations: List[Dict],
    shock_level: float,
    exit_bias: str = "trail",
) -> 'QuantumCircuit':
    """
    Build the 16-qubit VDJ recombination quantum circuit.

    Creates a superposition of all valid V+D+J combinations,
    weighted by current TE activation strengths, then collapses
    to select one specific antibody.

    Steps:
        1. V segment superposition (RY on qubits 0-5)
        2. D segment superposition (RY on qubits 6-9)
        3. J segment superposition (RY on qubits 10-13)
        4. Entanglement for 12/23 rule (CNOT V-D, D-J)
        5. RSS compatibility oracle (phase kickback on qubit 14)
        6. Junctional diversity seed (qubit 15)
        7. Measurement
    """
    if not QISKIT_AVAILABLE:
        return None

    qc = QuantumCircuit(VDJ_N_QUBITS, VDJ_N_QUBITS)

    # === STEP 1: V segment superposition ===
    v_amps = compute_v_amplitudes(te_activations)
    for i in range(6):
        angle = v_amps[i] * math.pi
        qc.ry(angle, i)

    # === STEP 2: D segment superposition ===
    d_amps = compute_d_amplitudes(te_activations, shock_level)
    for i in range(4):
        angle = d_amps[i] * math.pi
        qc.ry(angle, 6 + i)

    # === STEP 3: J segment superposition ===
    j_amps = compute_j_amplitudes(exit_bias)
    for i in range(4):
        angle = j_amps[i] * math.pi
        qc.ry(angle, 10 + i)

    # === STEP 4: Entanglement (12/23 rule encoding) ===
    # V-D entanglement: V qubits influence D qubits
    qc.cx(0, 6)
    qc.cx(1, 7)
    qc.cx(2, 8)
    # D-J entanglement: D qubits influence J qubits
    qc.cx(6, 10)
    qc.cx(7, 11)
    qc.cx(8, 12)

    # === STEP 5: RSS compatibility oracle ===
    # Qubit 14 flags compatibility via phase kickback
    qc.h(14)
    # Cross-entangle V/D/J with flag to create interference
    # that suppresses incompatible states
    qc.cx(0, 14)
    qc.cx(3, 14)
    qc.cx(6, 14)
    qc.cx(9, 14)
    qc.cx(10, 14)
    qc.cx(13, 14)
    qc.h(14)

    # === STEP 6: Junctional diversity seed ===
    qc.h(15)
    qc.cz(14, 15)
    # Shock level modulates diversity
    qc.ry(shock_level * math.pi * 0.3, 15)

    # === STEP 7: Measurement ===
    qc.measure(range(VDJ_N_QUBITS), range(VDJ_N_QUBITS))

    return qc


# ============================================================
# MEASUREMENT INTERPRETATION
# ============================================================

def interpret_vdj_measurement(bitstring: str) -> Dict:
    """
    Interpret a 16-bit measurement result as a V+D+J selection.

    Qiskit returns bitstrings in reverse order, so we reverse first.

    Returns:
        {
            "v_index": int,
            "v_name": str,
            "d_index": int,
            "d_name": str,
            "j_index": int,
            "j_name": str,
            "rss_valid": bool,
            "junction_seed": int,
        }
    """
    bits = bitstring[::-1]

    v_index = int(bits[0:6], 2)
    d_index = int(bits[6:10], 2)
    j_index = int(bits[10:14], 2)
    rss_flag = int(bits[14])
    junction = int(bits[15])

    # Clamp to valid ranges
    v_index = v_index % N_V
    d_index = d_index % N_D
    j_index = j_index % N_J

    return {
        "v_index": v_index,
        "v_name": V_NAMES[v_index],
        "d_index": d_index,
        "d_name": D_NAMES[d_index],
        "j_index": j_index,
        "j_name": J_NAMES[j_index],
        "rss_valid": bool(rss_flag),
        "junction_seed": junction,
    }


# ============================================================
# CLASSICAL FALLBACK
# ============================================================

def classical_vdj_selection(
    te_activations: List[Dict],
    shock_level: float,
    exit_bias: str = "trail",
    n_candidates: int = 20,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    """
    Classical fallback when Qiskit is not available.

    Uses weighted random selection biased by TE activation strengths
    and RSS compatibility filtering, mimicking the quantum circuit's
    behavior with classical probability distributions.

    Returns list of valid V+D+J selection dicts.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Compute V selection weights from TE activations
    v_weights = []
    for v_name in V_NAMES:
        v_def = V_SEGMENTS[v_name]
        te_source = v_def["te_source"]
        act = next((a for a in te_activations if a.get("te") == te_source), None)
        strength = act["strength"] if act else 0.1
        v_weights.append(max(0.01, strength))
    v_weights = np.array(v_weights) / sum(v_weights)

    # D selection weights from regime context
    d_weights = []
    for d_name in D_NAMES:
        d_def = D_SEGMENTS[d_name]
        param_shift = d_def.get("param_shift", {})
        base = sum(abs(v) for v in param_shift.values()) / max(1, len(param_shift))
        if d_def["regime"] in ("HIGH_VOLATILITY", "NEWS_SHOCK"):
            base *= (1.0 + shock_level * 0.5)
        d_weights.append(max(0.01, base))
    d_weights = np.array(d_weights) / sum(d_weights)

    # J selection weights from exit bias
    bias_map = {
        "trail": {"TRAILING_STOP": 1.5, "DYNAMIC": 1.2, "FIXED_TARGET": 0.8, "TIME_BASED": 0.6},
        "fixed_target": {"TRAILING_STOP": 0.7, "DYNAMIC": 1.0, "FIXED_TARGET": 1.5, "TIME_BASED": 0.8},
        "time_based": {"TRAILING_STOP": 0.6, "DYNAMIC": 0.8, "FIXED_TARGET": 0.8, "TIME_BASED": 1.5},
    }
    biases = bias_map.get(exit_bias, {})
    j_weights = []
    for j_name in J_NAMES:
        j_def = J_SEGMENTS[j_name]
        j_weights.append(biases.get(j_def["exit_type"], 1.0))
    j_weights = np.array(j_weights) / sum(j_weights)

    candidates = []
    attempts = 0
    max_attempts = n_candidates * 10

    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        v_idx = rng.choice(N_V, p=v_weights)
        d_idx = rng.choice(N_D, p=d_weights)
        j_idx = rng.choice(N_J, p=j_weights)

        v_name = V_NAMES[v_idx]
        d_name = D_NAMES[d_idx]
        j_name = J_NAMES[j_idx]

        if rss_compatible(v_name, d_name, j_name):
            candidates.append({
                "v_index": v_idx,
                "v_name": v_name,
                "d_index": d_idx,
                "d_name": d_name,
                "j_index": j_idx,
                "j_name": j_name,
                "rss_valid": True,
                "junction_seed": int(rng.randint(0, 2)),
            })

    return candidates


# ============================================================
# FULL CIRCUIT EXECUTION
# ============================================================

def execute_vdj_circuit(
    te_activations: List[Dict],
    shock_level: float,
    exit_bias: str = "trail",
    shots: int = 4096,
    n_candidates: int = 20,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    """
    Execute the VDJ quantum circuit and return validated candidates.

    If Qiskit is available, builds and runs the 16-qubit circuit.
    Otherwise falls back to classical weighted selection.

    Filters results through RSS compatibility rule.

    Returns:
        List of valid V+D+J selection dicts, sorted by measurement count
        (higher count = quantum amplitude preferred this combination).
    """
    if not QISKIT_AVAILABLE:
        return classical_vdj_selection(
            te_activations, shock_level, exit_bias, n_candidates, rng
        )

    # Build and execute quantum circuit
    qc = build_vdj_circuit(te_activations, shock_level, exit_bias)

    try:
        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        counts = job.result().get_counts()
    except Exception as e:
        log.warning("VDJ quantum circuit failed, using classical fallback: %s", e)
        return classical_vdj_selection(
            te_activations, shock_level, exit_bias, n_candidates, rng
        )

    # Interpret all measurements
    candidates = []
    seen = set()

    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
        selection = interpret_vdj_measurement(bitstring)

        # RSS compatibility filter
        if not rss_compatible(selection["v_name"], selection["d_name"], selection["j_name"]):
            continue

        # Deduplicate
        combo_key = (selection["v_name"], selection["d_name"], selection["j_name"])
        if combo_key in seen:
            continue
        seen.add(combo_key)

        selection["shot_count"] = count
        selection["shot_fraction"] = count / shots
        candidates.append(selection)

        if len(candidates) >= n_candidates:
            break

    log.info(
        "[VDJ-QC] Circuit executed: %d shots -> %d unique states -> %d valid candidates",
        shots, len(counts), len(candidates),
    )

    return candidates
