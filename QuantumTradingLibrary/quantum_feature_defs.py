"""
QUANTUM FEATURE DEFINITIONS
============================
Single source of truth for the 7 quantum compression features
used by the ETARE training pipeline and inference.

These features come from bg_archive.db (quantum_features table)
and represent real quantum compression metrics.
"""

# Canonical order - must match across trainer, expert, and fetcher
QUANTUM_FEATURE_NAMES = [
    "quantum_entropy",
    "dominant_state_prob",
    "superposition_measure",
    "phase_coherence",
    "entanglement_degree",
    "quantum_variance",
    "num_significant_states",
]

# Neutral defaults: after z-score normalization these produce ~zero signal
QUANTUM_FEATURE_DEFAULTS = {
    "quantum_entropy": 1.5,
    "dominant_state_prob": 0.25,
    "superposition_measure": 0.5,
    "phase_coherence": 0.5,
    "entanglement_degree": 0.0,
    "quantum_variance": 0.5,
    "num_significant_states": 4.0,
}

QUANTUM_FEATURE_COUNT = 7
