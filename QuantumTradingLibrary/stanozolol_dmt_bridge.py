"""
Stanozolol-DMT TE Bridge -- Performance Amplification & Pattern Recognition
============================================================================

EXTENSION MODULE for TEQA v3.0. Optional power layer.
The BRAIN can import and use this when she wants deeper processing.
If this file does not exist or fails to import, TEQA works fine without it.

Molecular Architecture (from Stanozolol-DMT thermal analysis):

    Property                     Trading System Analog
    -------------------------------------------------------
    11 rings (processing depth)  11-layer deep pipeline
    13 stereocenters             13 binary decision gates
    5 N-relay channels           5 independent signal paths
    230 C stability 0.85         Normal/calm market mode
    250 C stability 0.72         Genomic shock / high-vol mode
    Stanozolol core intact       Performance amplification preserved
    DMT amine free (230 C)       Pattern recognition channel unblocked

Target: 16-qubit / 8192-shot / 5-channel / 13-gate

Integration:
    Runs AFTER the normal TEQA pipeline and BEFORE final confidence scoring.
    Can boost OR suppress the confidence based on what the 13 gates say.

Authors: DooDoo + Claude
Date:    2026-02-12
Version: STANOZOLOL-DMT-BRIDGE-1.0
"""

import math
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional quantum imports -- mirrors the pattern in teqa_v3_neural_te.py
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

log = logging.getLogger(__name__)

VERSION = "STANOZOLOL-DMT-BRIDGE-1.0"

# ============================================================
# CONSTANTS -- Molecular Properties Mapped to Trading
# ============================================================

N_PROCESSING_RINGS = 11        # 11-layer pipeline (deeper than QNIF 5, ETARE 6)
N_STEREO_GATES = 13            # 13 binary decision gates
N_SIGNAL_CHANNELS = 5          # 5 independent signal paths (N-relay)
N_QUBITS_BRIDGE = 16           # 16-qubit quantum circuit
SHOTS_BRIDGE = 8192            # 8192 shots per measurement

# Qubit group allocation: 5 channels -> 3+3+3+3+4 = 16 qubits
CHANNEL_QUBIT_MAP = {
    0: (0, 3),     # Channel 1: Price action    -> qubits 0-2
    1: (3, 6),     # Channel 2: Volume/momentum -> qubits 3-5
    2: (6, 9),     # Channel 3: TE activation   -> qubits 6-8
    3: (9, 12),    # Channel 4: Quantum state   -> qubits 9-11
    4: (12, 16),   # Channel 5: Immune feedback  -> qubits 12-15
}

# Dual regime thresholds (molecular thermal stability)
REGIME_NORMAL_TEMP = 230.0       # Degrees C (metaphorical)
REGIME_STRESS_TEMP = 250.0
STABILITY_NORMAL = 0.85          # Stability coefficient at 230 C
STABILITY_STRESS = 0.72          # Stability coefficient at 250 C

# Volatility threshold for regime switch
# Maps to TEQA shock levels: SHOCK_LOW=0.8, SHOCK_NORMAL=1.2, SHOCK_HIGH=2.0
REGIME_SWITCH_SHOCK = 1.5        # Above this -> stress mode

# Gate majority threshold: 7 of 13 gates must pass
GATE_MAJORITY = 7

# Amplification limits (prevent runaway)
MAX_CONFIDENCE_BOOST = 0.15      # Max confidence increase
MAX_CONFIDENCE_SUPPRESS = 0.20   # Max confidence decrease
MIN_CONFIDENCE_FLOOR = 0.0       # Never go below 0
MAX_CONFIDENCE_CEILING = 1.0     # Never go above 1


# ============================================================
# REGIME DETECTION
# ============================================================

class BridgeRegime(Enum):
    """Dual operating mode based on molecular thermal stability."""
    NORMAL = "NORMAL_230C"      # Stability 0.85 -- DMT fully active
    STRESS = "STRESS_250C"      # Stability 0.72 -- Stanozolol amplifies harder


@dataclass
class RegimeState:
    """Current operating regime and its parameters."""
    regime: BridgeRegime = BridgeRegime.NORMAL
    stability: float = STABILITY_NORMAL
    temperature: float = REGIME_NORMAL_TEMP
    shock_score: float = 0.0
    stanozolol_gain: float = 1.0    # Performance amplification factor
    dmt_clarity: float = 1.0       # Pattern recognition clarity (suppressed in noise)


class DualRegimeController:
    """
    Two operating temperatures based on molecular thermal stability.

    230 C mode (stability 0.85):
        Normal markets. DMT pattern recognition fully active.
        Stanozolol amplification at baseline.

    250 C mode (stability 0.72):
        High volatility. Stanozolol amplification kicks harder.
        DMT partially suppressed (too noisy for clean pattern recognition).

    Switches automatically based on TEQA shock score.
    """

    def __init__(self):
        self.state = RegimeState()
        self._history: List[float] = []  # shock score history for smoothing

    def detect_regime(self, shock_score: float, shock_label: str = "") -> RegimeState:
        """
        Determine operating regime from TEQA shock metrics.

        Args:
            shock_score: From TEQA's GenomicShockDetector (0.0 = calm, 3.0+ = extreme)
            shock_label: Optional label from TEQA (CALM, NORMAL, SHOCK, EXTREME)

        Returns:
            RegimeState with all operating parameters set.
        """
        self._history.append(shock_score)
        if len(self._history) > 20:
            self._history = self._history[-20:]

        # Smooth over recent history to prevent flapping
        smoothed = np.mean(self._history[-5:]) if len(self._history) >= 5 else shock_score

        if smoothed >= REGIME_SWITCH_SHOCK or shock_label in ("SHOCK", "EXTREME"):
            # 250 C mode -- high volatility
            self.state.regime = BridgeRegime.STRESS
            self.state.stability = STABILITY_STRESS
            self.state.temperature = REGIME_STRESS_TEMP

            # In stress: Stanozolol pushes harder (1.2x to 1.8x)
            stress_intensity = min(1.0, (smoothed - REGIME_SWITCH_SHOCK) / 1.5)
            self.state.stanozolol_gain = 1.2 + 0.6 * stress_intensity

            # DMT suppressed: pattern recognition degrades in noise
            # At shock 1.5 -> clarity 0.7, at shock 3.0+ -> clarity 0.3
            self.state.dmt_clarity = max(0.3, 0.85 - 0.18 * stress_intensity)

        else:
            # 230 C mode -- normal markets
            self.state.regime = BridgeRegime.NORMAL
            self.state.stability = STABILITY_NORMAL
            self.state.temperature = REGIME_NORMAL_TEMP

            # Baseline amplification
            self.state.stanozolol_gain = 1.0

            # DMT fully clear
            self.state.dmt_clarity = 1.0

        self.state.shock_score = smoothed
        return self.state


# ============================================================
# STANOZOLOL CORE -- Performance Amplification Engine
# ============================================================

class StanozololCore:
    """
    Performance amplification through 11-layer deep processing.

    Each layer is a "processing ring" that refines the signal:
        Ring  1: Raw signal normalization
        Ring  2: Momentum alignment check
        Ring  3: Trend persistence filter
        Ring  4: Volatility-adjusted scaling
        Ring  5: Cross-timeframe confirmation
        Ring  6: TE concordance weighting
        Ring  7: Quantum amplitude modulation
        Ring  8: Domestication history integration
        Ring  9: Regime-aware gain control
        Ring 10: Signal stability assessment
        Ring 11: Final amplification with confidence bounds

    Takes signal confidence and amplifies proven patterns.
    The stanozolol_gain from DualRegimeController controls how
    aggressively each ring amplifies.
    """

    def __init__(self):
        # Per-ring state for diagnostics
        self.ring_outputs: List[Dict] = []

    def process(
        self,
        raw_confidence: float,
        direction: int,
        teqa_result: Dict,
        regime_state: RegimeState,
    ) -> Tuple[float, List[Dict]]:
        """
        Run the 11-ring amplification pipeline.

        Args:
            raw_confidence: TEQA's computed confidence (0-1)
            direction: Signal direction from TEQA (-1, 0, 1)
            teqa_result: Full result dict from TEQAv3Engine.analyze()
            regime_state: Current operating regime

        Returns:
            (amplified_confidence, ring_details)
        """
        self.ring_outputs = []
        gain = regime_state.stanozolol_gain
        signal = raw_confidence

        # === Ring 1: Raw Signal Normalization ===
        # Compress extreme values, expand middle range
        normalized = self._sigmoid_compress(signal, center=0.5, steepness=4.0)
        self._log_ring(1, "Raw Signal Normalization", signal, normalized)
        signal = normalized

        # === Ring 2: Momentum Alignment ===
        # Check if neural consensus direction matches TE concordance direction
        consensus_dir = teqa_result.get("consensus_direction", 0)
        concordance = teqa_result.get("concordance", 0.0)
        if direction != 0 and consensus_dir == direction:
            # Aligned -- amplify
            momentum_factor = 1.0 + 0.08 * gain * concordance
        elif direction != 0 and consensus_dir != 0 and consensus_dir != direction:
            # Conflicting -- suppress
            momentum_factor = 0.85
        else:
            momentum_factor = 1.0
        momentum_adjusted = signal * momentum_factor
        self._log_ring(2, "Momentum Alignment", signal, momentum_adjusted,
                       {"aligned": consensus_dir == direction, "factor": momentum_factor})
        signal = momentum_adjusted

        # === Ring 3: Trend Persistence Filter ===
        # Strong domestication boost means pattern has been persistent
        dom_boost = teqa_result.get("domestication_boost", 1.0)
        if dom_boost > 1.2:
            persistence_factor = 1.0 + 0.05 * gain * min(1.0, (dom_boost - 1.0))
        elif dom_boost < 0.8:
            persistence_factor = 0.9  # Anti-persistent -- dampen
        else:
            persistence_factor = 1.0
        persistence_adjusted = signal * persistence_factor
        self._log_ring(3, "Trend Persistence Filter", signal, persistence_adjusted,
                       {"dom_boost": dom_boost, "factor": persistence_factor})
        signal = persistence_adjusted

        # === Ring 4: Volatility-Adjusted Scaling ===
        # In normal regime, scale by stability. In stress, still scale but less.
        vol_scale = regime_state.stability
        vol_adjusted = signal * (0.7 + 0.3 * vol_scale)
        self._log_ring(4, "Volatility-Adjusted Scaling", signal, vol_adjusted,
                       {"stability": vol_scale})
        signal = vol_adjusted

        # === Ring 5: Cross-Timeframe Confirmation ===
        # Use quantum amplitude as proxy for cross-timeframe agreement
        amplitude_sq = teqa_result.get("amplitude_sq", 0.0)
        if amplitude_sq > 0.3:
            ctf_factor = 1.0 + 0.06 * gain * (amplitude_sq - 0.3)
        else:
            ctf_factor = 0.95
        ctf_adjusted = signal * ctf_factor
        self._log_ring(5, "Cross-Timeframe Confirmation", signal, ctf_adjusted,
                       {"amplitude_sq": amplitude_sq, "factor": ctf_factor})
        signal = ctf_adjusted

        # === Ring 6: TE Concordance Weighting ===
        # How many TE families agree with the direction?
        n_active_i = teqa_result.get("n_active_class_i", 0)
        n_active_ii = teqa_result.get("n_active_class_ii", 0)
        n_active_neural = teqa_result.get("n_active_neural", 0)
        total_active = n_active_i + n_active_ii + n_active_neural
        if total_active > 15:
            te_weight = 1.0 + 0.04 * gain  # Strong TE agreement
        elif total_active > 8:
            te_weight = 1.0 + 0.02 * gain  # Moderate agreement
        else:
            te_weight = 0.95
        te_adjusted = signal * te_weight
        self._log_ring(6, "TE Concordance Weighting", signal, te_adjusted,
                       {"total_active": total_active, "weight": te_weight})
        signal = te_adjusted

        # === Ring 7: Quantum Amplitude Modulation ===
        # Derive modulation from consensus score strength
        consensus_score = teqa_result.get("consensus_score", 0.0)
        q_mod = 1.0 + 0.05 * gain * max(0, consensus_score - 0.5)
        q_adjusted = signal * q_mod
        self._log_ring(7, "Quantum Amplitude Modulation", signal, q_adjusted,
                       {"consensus_score": consensus_score, "modulation": q_mod})
        signal = q_adjusted

        # === Ring 8: Domestication History Integration ===
        # Pull gate results from TEQA -- domestication gate
        gates = teqa_result.get("gates", {})
        dom_gate = gates.get("G10_domestication", False)
        if dom_gate:
            dom_history_factor = 1.0 + 0.03 * gain
        else:
            dom_history_factor = 0.92
        dom_adjusted = signal * dom_history_factor
        self._log_ring(8, "Domestication History Integration", signal, dom_adjusted,
                       {"gate_pass": dom_gate, "factor": dom_history_factor})
        signal = dom_adjusted

        # === Ring 9: Regime-Aware Gain Control ===
        # Final regime-dependent scaling
        if regime_state.regime == BridgeRegime.STRESS:
            # In stress: if signal is strong AND aligned, amplify more
            # If weak, suppress harder (stanozolol cuts losers fast)
            if signal > 0.5 and direction != 0:
                regime_factor = 1.0 + 0.06 * (gain - 1.0)
            else:
                regime_factor = 0.85
        else:
            regime_factor = 1.0
        regime_adjusted = signal * regime_factor
        self._log_ring(9, "Regime-Aware Gain Control", signal, regime_adjusted,
                       {"regime": regime_state.regime.value, "factor": regime_factor})
        signal = regime_adjusted

        # === Ring 10: Signal Stability Assessment ===
        # Check variance of ring outputs -- stable processing = higher confidence
        ring_vals = [r["output"] for r in self.ring_outputs]
        ring_variance = float(np.var(ring_vals)) if len(ring_vals) > 1 else 0.0
        if ring_variance < 0.005:
            stability_bonus = 1.02  # Very stable processing
        elif ring_variance > 0.05:
            stability_bonus = 0.95  # Unstable -- rings disagree
        else:
            stability_bonus = 1.0
        stability_adjusted = signal * stability_bonus
        self._log_ring(10, "Signal Stability Assessment", signal, stability_adjusted,
                       {"ring_variance": ring_variance, "bonus": stability_bonus})
        signal = stability_adjusted

        # === Ring 11: Final Amplification with Bounds ===
        # Apply final gain and clamp to valid range
        final_gain = 1.0 + 0.03 * (gain - 1.0)  # Subtle final push
        amplified = signal * final_gain
        amplified = float(np.clip(amplified, 0.0, 1.0))
        self._log_ring(11, "Final Amplification", signal, amplified,
                       {"final_gain": final_gain})

        return amplified, self.ring_outputs

    def _sigmoid_compress(self, x: float, center: float = 0.5,
                          steepness: float = 4.0) -> float:
        """Sigmoid compression to normalize signal range."""
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

    def _log_ring(self, ring_num: int, name: str, input_val: float,
                  output_val: float, details: Optional[Dict] = None):
        """Record ring processing for diagnostics."""
        self.ring_outputs.append({
            "ring": ring_num,
            "name": name,
            "input": round(input_val, 6),
            "output": round(output_val, 6),
            "delta": round(output_val - input_val, 6),
            "details": details or {},
        })


# ============================================================
# DMT PATTERN ENGINE -- 5-Channel Pattern Recognition
# ============================================================

class DMTPatternEngine:
    """
    Pattern recognition through 5 independent signal channels.
    Each channel processes a different dimension of market data.
    All 5 channels run independently and converge via weighted vote.

    Channel mapping (from N-relay molecular structure):
        Channel 1: Price action patterns
        Channel 2: Volume/momentum patterns
        Channel 3: TE activation patterns
        Channel 4: Quantum state patterns
        Channel 5: Immune system feedback patterns

    In NORMAL regime: all channels active, full clarity.
    In STRESS regime: channels 4,5 partially suppressed (noisy data).
    """

    def __init__(self):
        self.channel_results: List[Dict] = []

    def process(
        self,
        teqa_result: Dict,
        regime_state: RegimeState,
    ) -> Tuple[float, int, List[Dict]]:
        """
        Run all 5 channels and converge.

        Returns:
            (pattern_confidence, pattern_direction, channel_details)
        """
        self.channel_results = []
        clarity = regime_state.dmt_clarity

        # Channel 1: Price Action Patterns
        ch1 = self._channel_price_action(teqa_result)
        self.channel_results.append(ch1)

        # Channel 2: Volume/Momentum Patterns
        ch2 = self._channel_volume_momentum(teqa_result)
        self.channel_results.append(ch2)

        # Channel 3: TE Activation Patterns
        ch3 = self._channel_te_activation(teqa_result)
        self.channel_results.append(ch3)

        # Channel 4: Quantum State Patterns (suppressed in stress)
        ch4 = self._channel_quantum_state(teqa_result)
        ch4["confidence"] *= clarity  # DMT suppression in noise
        ch4["clarity_applied"] = clarity
        self.channel_results.append(ch4)

        # Channel 5: Immune System Feedback (suppressed in stress)
        ch5 = self._channel_immune_feedback(teqa_result)
        ch5["confidence"] *= clarity
        ch5["clarity_applied"] = clarity
        self.channel_results.append(ch5)

        # Converge: weighted vote
        # Channels 1-3 get full weight, 4-5 are already clarity-adjusted
        weights = [0.25, 0.20, 0.25, 0.15, 0.15]  # Sum = 1.0
        total_confidence = sum(
            ch["confidence"] * w
            for ch, w in zip(self.channel_results, weights)
        )

        # Direction: majority vote weighted by confidence
        weighted_dir = sum(
            ch["direction"] * ch["confidence"] * w
            for ch, w in zip(self.channel_results, weights)
        )
        if weighted_dir > 0.05:
            pattern_direction = 1
        elif weighted_dir < -0.05:
            pattern_direction = -1
        else:
            pattern_direction = 0

        return total_confidence, pattern_direction, self.channel_results

    def _channel_price_action(self, teqa: Dict) -> Dict:
        """Channel 1: Price action pattern recognition."""
        # Uses concordance (how many TEs agree on direction)
        concordance = teqa.get("concordance", 0.0)
        direction = teqa.get("direction", 0)

        # Price action confidence is high when concordance is high
        confidence = concordance * 0.8 + 0.1  # Scale to [0.1, 0.9]

        return {
            "channel": 1,
            "name": "Price Action",
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "direction": direction,
            "concordance": concordance,
        }

    def _channel_volume_momentum(self, teqa: Dict) -> Dict:
        """Channel 2: Volume and momentum pattern detection."""
        # Key TEs for this channel: BEL_Pao (momentum), SINE (tick_volume),
        # Helitron (volume_profile)
        activations = teqa.get("te_activations", [])
        momentum_tes = ["BEL_Pao", "SINE", "Helitron"]
        strengths = []
        directions = []
        for act in activations:
            if act.get("te") in momentum_tes:
                strengths.append(act.get("strength", 0.0))
                directions.append(act.get("direction", 0))

        if strengths:
            avg_strength = float(np.mean(strengths))
            # Direction: majority of momentum TEs
            dir_sum = sum(directions)
            if dir_sum > 0:
                direction = 1
            elif dir_sum < 0:
                direction = -1
            else:
                direction = 0
        else:
            avg_strength = 0.0
            direction = 0

        return {
            "channel": 2,
            "name": "Volume/Momentum",
            "confidence": float(np.clip(avg_strength, 0.0, 1.0)),
            "direction": direction,
            "source_tes": momentum_tes,
            "avg_strength": avg_strength,
        }

    def _channel_te_activation(self, teqa: Dict) -> Dict:
        """Channel 3: TE activation pattern analysis."""
        n_class_i = teqa.get("n_active_class_i", 0)
        n_class_ii = teqa.get("n_active_class_ii", 0)
        n_neural = teqa.get("n_active_neural", 0)
        total = n_class_i + n_class_ii + n_neural

        # Confidence from TE coverage (more active = more signal)
        # 33 total TEs, so normalize by 33
        coverage = min(1.0, total / 20.0)  # 20+ active TEs = full confidence

        # Direction from class balance
        # Class I (retrotransposons) tend to be trend-following
        # Class II (DNA transposons) tend to be mean-reverting
        if n_class_i > n_class_ii:
            direction = teqa.get("direction", 0)  # Follow the trend
        elif n_class_ii > n_class_i * 1.5:
            direction = -teqa.get("direction", 0)  # Contrarian signal
        else:
            direction = teqa.get("direction", 0)

        return {
            "channel": 3,
            "name": "TE Activation",
            "confidence": float(np.clip(coverage, 0.0, 1.0)),
            "direction": direction,
            "n_class_i": n_class_i,
            "n_class_ii": n_class_ii,
            "n_neural": n_neural,
            "total_active": total,
        }

    def _channel_quantum_state(self, teqa: Dict) -> Dict:
        """Channel 4: Quantum state pattern analysis."""
        consensus_score = teqa.get("consensus_score", 0.0)
        consensus_dir = teqa.get("consensus_direction", 0)
        amplitude_sq = teqa.get("amplitude_sq", 0.0)

        # Quantum confidence from consensus strength and amplitude
        confidence = consensus_score * 0.6 + amplitude_sq * 0.4

        return {
            "channel": 4,
            "name": "Quantum State",
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "direction": consensus_dir,
            "consensus_score": consensus_score,
            "amplitude_sq": amplitude_sq,
        }

    def _channel_immune_feedback(self, teqa: Dict) -> Dict:
        """Channel 5: Immune system feedback (domestication + gates)."""
        gates = teqa.get("gates", {})
        dom_boost = teqa.get("domestication_boost", 1.0)

        # Count passing gates from TEQA
        gates_passed = sum(1 for v in gates.values() if v is True)
        gates_total = max(1, len(gates))

        # Immune confidence: gate pass rate + domestication
        gate_ratio = gates_passed / gates_total
        dom_factor = min(1.0, max(0.0, dom_boost - 0.5))  # Normalize to [0, 1]
        confidence = gate_ratio * 0.6 + dom_factor * 0.4

        # Direction from TEQA (immune system agrees with the consensus)
        direction = teqa.get("direction", 0)

        return {
            "channel": 5,
            "name": "Immune Feedback",
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "direction": direction,
            "gates_passed": gates_passed,
            "gates_total": gates_total,
            "dom_boost": dom_boost,
        }


# ============================================================
# TE BRIDGE -- Connecting Stanozolol to DMT via TE Strings
# ============================================================

class TEBridge:
    """
    The TE string/glue connecting the Stanozolol amplifier to the DMT recognizer.

    Maps TE families to molecular binding sites. Routes TE activations
    through both the stanozolol amplifier and DMT recognizer.

    Binding site mapping:
        Retrotransposons (Class I) -> Stanozolol core (anabolic ring)
        DNA Transposons (Class II) -> DMT amine group (recognition)
        Neural TEs                 -> Bridge linker (connects both)
    """

    # TE families that bind to Stanozolol (amplification pathway)
    STANOZOLOL_BINDING = {
        "BEL_Pao", "DIRS1", "Ty1_copia", "Ty3_gypsy", "Ty5",
        "Alu", "LINE", "Penelope", "RTE", "SINE", "VIPER_Ngaro",
    }

    # TE families that bind to DMT (pattern recognition pathway)
    DMT_BINDING = {
        "CACTA", "Crypton", "Helitron", "hobo", "I_element",
        "Mariner_Tc1", "Mavericks_Polinton", "Mutator",
        "P_element", "PIF_Harbinger", "piggyBac", "pogo",
        "Rag_like", "Transib",
    }

    # Neural TEs act as the bridge linker
    BRIDGE_LINKER = {
        "L1_Neuronal", "L1_Somatic", "HERV_Synapse", "SVA_Regulatory",
        "Alu_Exonization", "TRIM28_Silencer", "piwiRNA_Neural", "Arc_Capsid",
    }

    def compute_binding_strengths(
        self, te_activations: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute aggregate binding strength for each pathway.

        Returns:
            Dict with keys: stanozolol_binding, dmt_binding, bridge_linker,
                            cross_talk (how much the bridge connects them)
        """
        stano_strengths = []
        dmt_strengths = []
        bridge_strengths = []

        for act in te_activations:
            te_name = act.get("te", "")
            strength = act.get("strength", 0.0)

            if te_name in self.STANOZOLOL_BINDING:
                stano_strengths.append(strength)
            elif te_name in self.DMT_BINDING:
                dmt_strengths.append(strength)

            if te_name in self.BRIDGE_LINKER:
                bridge_strengths.append(strength)

        stano_avg = float(np.mean(stano_strengths)) if stano_strengths else 0.0
        dmt_avg = float(np.mean(dmt_strengths)) if dmt_strengths else 0.0
        bridge_avg = float(np.mean(bridge_strengths)) if bridge_strengths else 0.0

        # Cross-talk: how well the bridge connects stanozolol to DMT
        # High bridge + both pathways active = strong cross-talk
        cross_talk = bridge_avg * min(stano_avg, dmt_avg)

        return {
            "stanozolol_binding": round(stano_avg, 6),
            "dmt_binding": round(dmt_avg, 6),
            "bridge_linker": round(bridge_avg, 6),
            "cross_talk": round(cross_talk, 6),
        }


# ============================================================
# DECISION GATE ARRAY -- 13 Binary Stereo-Gates
# ============================================================

class DecisionGateArray:
    """
    13 binary decision gates mapped from molecular stereocenters.

    Each gate is a yes/no filter. Signal must pass majority (7/13)
    to receive amplification. Fewer passes = suppression.

    Gates:
        G1:  Trend direction confirmed
        G2:  Momentum aligned with trend
        G3:  Volatility within acceptable range
        G4:  Regime supports signal type
        G5:  TE consensus (majority agree on direction)
        G6:  Immune clearance (no EXTREME shock)
        G7:  Neural consensus pass
        G8:  Domestication gate (proven pattern)
        G9:  Speciation gate (no hybrid zone conflict)
        G10: Cross-channel agreement (DMT channels agree)
        G11: Amplification coherence (Stanozolol rings stable)
        G12: Binding strength sufficient (TE bridge connected)
        G13: Quantum amplitude above noise floor
    """

    def __init__(self):
        self.gate_results: List[Dict] = []

    def evaluate(
        self,
        teqa_result: Dict,
        regime_state: RegimeState,
        stanozolol_ring_outputs: List[Dict],
        dmt_channel_results: List[Dict],
        binding_strengths: Dict,
    ) -> Tuple[int, int, List[Dict]]:
        """
        Run all 13 gates.

        Returns:
            (gates_passed, gates_total, gate_details)
        """
        self.gate_results = []

        # G1: Trend direction confirmed
        direction = teqa_result.get("direction", 0)
        g1 = direction != 0
        self._log_gate(1, "Trend Direction", g1, {"direction": direction})

        # G2: Momentum aligned with trend
        consensus_dir = teqa_result.get("consensus_direction", 0)
        g2 = direction != 0 and consensus_dir == direction
        self._log_gate(2, "Momentum Aligned", g2,
                       {"direction": direction, "consensus": consensus_dir})

        # G3: Volatility within acceptable range
        shock_score = teqa_result.get("shock_score", 0.0)
        g3 = shock_score < 2.5  # Not in extreme territory
        self._log_gate(3, "Volatility Range", g3, {"shock_score": shock_score})

        # G4: Regime supports signal type
        # In normal regime, all signals pass. In stress, only strong signals.
        if regime_state.regime == BridgeRegime.NORMAL:
            g4 = True
        else:
            g4 = teqa_result.get("confidence", 0.0) > 0.5
        self._log_gate(4, "Regime Support", g4,
                       {"regime": regime_state.regime.value,
                        "confidence": teqa_result.get("confidence", 0.0)})

        # G5: TE consensus
        concordance = teqa_result.get("concordance", 0.0)
        g5 = concordance > 0.4
        self._log_gate(5, "TE Consensus", g5, {"concordance": concordance})

        # G6: Immune clearance (no extreme shock)
        shock_label = teqa_result.get("shock_label", "")
        g6 = shock_label != "EXTREME"
        self._log_gate(6, "Immune Clearance", g6, {"shock_label": shock_label})

        # G7: Neural consensus pass
        neural_pass = teqa_result.get("neural_consensus_pass", False)
        g7 = neural_pass
        self._log_gate(7, "Neural Consensus", g7, {"pass": neural_pass})

        # G8: Domestication gate
        dom_boost = teqa_result.get("domestication_boost", 1.0)
        g8 = dom_boost >= 1.0
        self._log_gate(8, "Domestication", g8, {"dom_boost": dom_boost})

        # G9: Speciation gate
        relationship = teqa_result.get("relationship", "")
        g9 = relationship != "HYBRID_ZONE"
        self._log_gate(9, "Speciation", g9, {"relationship": relationship})

        # G10: Cross-channel agreement (DMT channels)
        if dmt_channel_results:
            channel_dirs = [ch.get("direction", 0) for ch in dmt_channel_results
                           if ch.get("direction", 0) != 0]
            if channel_dirs:
                agreement = abs(sum(channel_dirs)) / len(channel_dirs)
                g10 = agreement > 0.5
            else:
                g10 = False
        else:
            g10 = False
        self._log_gate(10, "Cross-Channel Agreement", g10,
                       {"n_channels_with_dir": len([ch for ch in dmt_channel_results
                                                     if ch.get("direction", 0) != 0])
                        if dmt_channel_results else 0})

        # G11: Amplification coherence (Stanozolol processing stability)
        if stanozolol_ring_outputs:
            ring_vals = [r["output"] for r in stanozolol_ring_outputs]
            ring_var = float(np.var(ring_vals))
            g11 = ring_var < 0.03  # Low variance = stable processing
        else:
            g11 = False
        self._log_gate(11, "Amplification Coherence", g11,
                       {"ring_variance": ring_var if stanozolol_ring_outputs else -1})

        # G12: Binding strength sufficient
        cross_talk = binding_strengths.get("cross_talk", 0.0)
        g12 = cross_talk > 0.05
        self._log_gate(12, "Binding Strength", g12, {"cross_talk": cross_talk})

        # G13: Quantum amplitude above noise floor
        amplitude_sq = teqa_result.get("amplitude_sq", 0.0)
        g13 = amplitude_sq > 0.15
        self._log_gate(13, "Quantum Amplitude", g13, {"amplitude_sq": amplitude_sq})

        # Count passes
        all_gates = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13]
        gates_passed = sum(1 for g in all_gates if g)

        return gates_passed, N_STEREO_GATES, self.gate_results

    def _log_gate(self, gate_num: int, name: str, passed: bool,
                  details: Optional[Dict] = None):
        """Record gate result for diagnostics."""
        self.gate_results.append({
            "gate": gate_num,
            "name": name,
            "passed": passed,
            "details": details or {},
        })


# ============================================================
# QUANTUM CIRCUIT -- 16-Qubit / 8192-Shot Bridge Circuit
# ============================================================

class BridgeQuantumCircuit:
    """
    16-qubit quantum circuit for the Stanozolol-DMT bridge.

    Qubit allocation (5 channels -> 3+3+3+3+4 = 16 qubits):
        Qubits  0-2:  Channel 1 (Price action)
        Qubits  3-5:  Channel 2 (Volume/momentum)
        Qubits  6-8:  Channel 3 (TE activation)
        Qubits  9-11: Channel 4 (Quantum state)
        Qubits 12-15: Channel 5 (Immune feedback)

    13 gates map to RY/RZ rotations + CNOT entanglement across channels.
    """

    def __init__(self, shots: int = SHOTS_BRIDGE):
        self.shots = shots
        self.simulator = None
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()

    def build_circuit(
        self,
        dmt_channels: List[Dict],
        gate_results: List[Dict],
        regime_state: RegimeState,
    ) -> Optional['QuantumCircuit']:
        """
        Build the 16-qubit bridge circuit.

        Channel confidences drive RY rotations.
        Gate results drive RZ phase shifts.
        CNOT gates create inter-channel entanglement.
        """
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(N_QUBITS_BRIDGE, N_QUBITS_BRIDGE)

        # === Layer 1: Channel-driven RY rotations ===
        for ch_idx, ch_result in enumerate(dmt_channels):
            if ch_idx >= N_SIGNAL_CHANNELS:
                break
            start_q, end_q = CHANNEL_QUBIT_MAP[ch_idx]
            confidence = ch_result.get("confidence", 0.0)
            direction = ch_result.get("direction", 0)

            # RY angle from confidence and direction
            base_angle = confidence * math.pi
            if direction < 0:
                base_angle = -base_angle

            # Apply to each qubit in the channel group
            n_qubits_in_channel = end_q - start_q
            for qi in range(start_q, end_q):
                # Each qubit gets a slightly different angle for diversity
                offset = (qi - start_q) / max(1, n_qubits_in_channel - 1) * 0.2
                qc.ry(base_angle * (1.0 + offset), qi)

        # === Layer 2: Gate-driven RZ phase shifts ===
        # Map 13 gates to 13 of the 16 qubits (first 13)
        for gate_result in gate_results:
            gate_idx = gate_result.get("gate", 1) - 1  # 0-indexed
            if gate_idx >= N_QUBITS_BRIDGE:
                break
            passed = gate_result.get("passed", False)
            # RZ phase: pi/4 for pass, -pi/4 for fail
            phase = math.pi / 4 if passed else -math.pi / 4
            qc.rz(phase, gate_idx)

        # === Layer 3: Intra-channel entanglement (CNOT chains) ===
        for ch_idx in range(N_SIGNAL_CHANNELS):
            start_q, end_q = CHANNEL_QUBIT_MAP[ch_idx]
            for qi in range(start_q, end_q - 1):
                qc.cx(qi, qi + 1)

        # === Layer 4: Inter-channel entanglement (cross-talk bridges) ===
        # Channel 1 <-> Channel 2 (price <-> volume)
        qc.cx(2, 3)
        # Channel 2 <-> Channel 3 (volume <-> TE)
        qc.cx(5, 6)
        # Channel 3 <-> Channel 4 (TE <-> quantum)
        qc.cx(8, 9)
        # Channel 4 <-> Channel 5 (quantum <-> immune)
        qc.cx(11, 12)
        # Channel 1 <-> Channel 5 (price <-> immune, long-range)
        qc.cx(0, 15)

        # === Layer 5: Regime-dependent second rotation ===
        stability = regime_state.stability
        for qi in range(N_QUBITS_BRIDGE):
            qc.ry(stability * math.pi * 0.15, qi)

        # === Layer 6: Final RZ phase (regime temperature encoding) ===
        temp_phase = (regime_state.temperature - 230.0) / 20.0 * math.pi / 8
        for qi in range(N_QUBITS_BRIDGE):
            qc.rz(temp_phase, qi)

        # Measure all qubits
        qc.measure(range(N_QUBITS_BRIDGE), range(N_QUBITS_BRIDGE))

        return qc

    def execute(
        self, qc: Optional['QuantumCircuit']
    ) -> Dict:
        """
        Execute the circuit and interpret results.

        Returns dict with:
            vote_long, vote_short, top_state, n_unique_states,
            channel_biases (per-channel measurement bias)
        """
        if qc is None or self.simulator is None:
            return self._classical_fallback()

        try:
            from qiskit import transpile
            transpiled = transpile(qc, self.simulator)
            result = self.simulator.run(transpiled, shots=self.shots).result()
            counts = result.get_counts()
        except Exception as e:
            log.warning("Bridge quantum circuit execution failed: %s", e)
            return self._classical_fallback()

        # Interpret measurement results
        vote_long = 0
        vote_short = 0
        channel_ones = [0] * N_SIGNAL_CHANNELS

        for bitstring, count in counts.items():
            # Qiskit returns bitstrings in reverse order
            bits = bitstring[::-1]

            # Overall vote: majority of bits
            n_ones = bits.count('1')
            if n_ones > N_QUBITS_BRIDGE // 2:
                vote_long += count
            else:
                vote_short += count

            # Per-channel bias
            for ch_idx in range(N_SIGNAL_CHANNELS):
                start_q, end_q = CHANNEL_QUBIT_MAP[ch_idx]
                ch_bits = bits[start_q:end_q]
                ch_ones = ch_bits.count('1')
                if ch_ones > (end_q - start_q) // 2:
                    channel_ones[ch_idx] += count

        total = vote_long + vote_short
        if total == 0:
            total = 1

        # Top state
        top_state = max(counts, key=counts.get) if counts else "0" * N_QUBITS_BRIDGE
        top_count = counts.get(top_state, 0) if counts else 0

        # Channel biases (fraction that voted "long" per channel)
        channel_biases = [ch / total for ch in channel_ones]

        return {
            "vote_long": vote_long / total,
            "vote_short": vote_short / total,
            "top_state": top_state,
            "top_count": top_count,
            "n_unique_states": len(counts) if counts else 0,
            "n_shots": self.shots,
            "channel_biases": channel_biases,
        }

    def _classical_fallback(self) -> Dict:
        """Classical fallback when Qiskit is not available."""
        # Generate pseudo-random measurement based on numpy
        rng = np.random.default_rng()
        vote_long = rng.random()
        vote_short = 1.0 - vote_long
        channel_biases = [rng.random() for _ in range(N_SIGNAL_CHANNELS)]
        return {
            "vote_long": vote_long,
            "vote_short": vote_short,
            "top_state": "0" * N_QUBITS_BRIDGE,
            "top_count": 0,
            "n_unique_states": 1,
            "n_shots": 0,
            "channel_biases": channel_biases,
            "fallback": True,
        }


# ============================================================
# MAIN BRIDGE ORCHESTRATOR
# ============================================================

class StanozololDMTBridge:
    """
    Main orchestrator for the Stanozolol-DMT TE Bridge.

    Pipeline:
        1. Detect regime (DualRegimeController)
        2. Compute TE binding strengths (TEBridge)
        3. Run DMT pattern recognition (5 channels)
        4. Run Stanozolol amplification (11 rings)
        5. Evaluate decision gates (13 gates)
        6. Execute quantum bridge circuit (16 qubits, 8192 shots)
        7. Compute final confidence adjustment
        8. Return boost/suppress recommendation

    Usage:
        bridge = StanozololDMTBridge()
        adjustment = bridge.process(teqa_result)
        new_confidence = teqa_result["confidence"] + adjustment["confidence_delta"]
    """

    def __init__(self, shots: int = SHOTS_BRIDGE):
        self.regime_controller = DualRegimeController()
        self.te_bridge = TEBridge()
        self.stanozolol = StanozololCore()
        self.dmt = DMTPatternEngine()
        self.gate_array = DecisionGateArray()
        self.quantum = BridgeQuantumCircuit(shots=shots)
        self._call_count = 0

    def process(self, teqa_result: Dict) -> Dict:
        """
        Run the full Stanozolol-DMT bridge pipeline.

        Args:
            teqa_result: Full result dict from TEQAv3Engine.analyze()

        Returns:
            Dict with:
                confidence_delta: How much to add/subtract from TEQA confidence
                bridge_confidence: The bridge's own confidence assessment
                regime: Current operating regime
                gates_passed: How many of 13 gates passed
                ring_outputs: Detailed 11-ring processing
                channel_results: 5-channel DMT results
                gate_results: 13-gate evaluation details
                quantum_result: 16-qubit circuit measurement
                binding: TE binding strengths
                recommendation: "BOOST", "SUPPRESS", or "NEUTRAL"
        """
        t_start = time.time()
        self._call_count += 1

        # === Step 1: Detect Regime ===
        shock_score = teqa_result.get("shock_score", 0.0)
        shock_label = teqa_result.get("shock_label", "")
        regime_state = self.regime_controller.detect_regime(shock_score, shock_label)

        # === Step 2: Compute TE Binding Strengths ===
        te_activations = teqa_result.get("te_activations", [])
        binding = self.te_bridge.compute_binding_strengths(te_activations)

        # === Step 3: DMT Pattern Recognition (5 channels) ===
        dmt_confidence, dmt_direction, channel_results = self.dmt.process(
            teqa_result, regime_state
        )

        # === Step 4: Stanozolol Amplification (11 rings) ===
        raw_confidence = teqa_result.get("confidence", 0.0)
        direction = teqa_result.get("direction", 0)
        amplified_confidence, ring_outputs = self.stanozolol.process(
            raw_confidence, direction, teqa_result, regime_state
        )

        # === Step 5: Evaluate Decision Gates (13 gates) ===
        gates_passed, gates_total, gate_results = self.gate_array.evaluate(
            teqa_result, regime_state, ring_outputs, channel_results, binding
        )

        # === Step 6: Quantum Bridge Circuit (16 qubits, 8192 shots) ===
        qc = self.quantum.build_circuit(channel_results, gate_results, regime_state)
        quantum_result = self.quantum.execute(qc)

        # === Step 7: Compute Final Confidence Adjustment ===
        # Gate-based decision: majority pass = boost, minority pass = suppress
        gate_ratio = gates_passed / gates_total  # 0 to 1

        # Blend stanozolol amplification with DMT pattern recognition
        # Weight depends on regime
        if regime_state.regime == BridgeRegime.NORMAL:
            stano_weight = 0.4
            dmt_weight = 0.4
            quantum_weight = 0.2
        else:
            # Stress: stanozolol dominant, DMT suppressed
            stano_weight = 0.55
            dmt_weight = 0.25
            quantum_weight = 0.20

        bridge_confidence = (
            amplified_confidence * stano_weight +
            dmt_confidence * dmt_weight +
            quantum_result["vote_long"] * quantum_weight
        )

        # Confidence delta based on gate majority
        if gate_ratio >= GATE_MAJORITY / N_STEREO_GATES:
            # Majority passed: BOOST
            # Scale boost by how many extra gates passed beyond majority
            extra_gates = gates_passed - GATE_MAJORITY
            boost_scale = 1.0 + extra_gates * 0.15
            delta = (bridge_confidence - raw_confidence) * gate_ratio * boost_scale
            delta = min(delta, MAX_CONFIDENCE_BOOST)
            delta = max(delta, 0.0)  # Never negative on a BOOST
            recommendation = "BOOST"
        elif gates_passed <= 4:
            # Strong minority: SUPPRESS
            suppress_scale = 1.0 - gate_ratio
            delta = -(raw_confidence * 0.15 * suppress_scale)
            delta = max(delta, -MAX_CONFIDENCE_SUPPRESS)
            recommendation = "SUPPRESS"
        else:
            # Borderline: NEUTRAL (minimal adjustment)
            delta = (bridge_confidence - raw_confidence) * 0.05
            delta = float(np.clip(delta, -0.03, 0.03))
            recommendation = "NEUTRAL"

        elapsed_ms = (time.time() - t_start) * 1000

        return {
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": round(elapsed_ms, 2),
            "call_count": self._call_count,

            # Core outputs
            "confidence_delta": round(delta, 6),
            "bridge_confidence": round(bridge_confidence, 6),
            "original_confidence": round(raw_confidence, 6),
            "adjusted_confidence": round(
                float(np.clip(raw_confidence + delta,
                              MIN_CONFIDENCE_FLOOR,
                              MAX_CONFIDENCE_CEILING)), 6),
            "recommendation": recommendation,

            # Regime
            "regime": regime_state.regime.value,
            "regime_stability": regime_state.stability,
            "regime_temperature": regime_state.temperature,
            "stanozolol_gain": round(regime_state.stanozolol_gain, 4),
            "dmt_clarity": round(regime_state.dmt_clarity, 4),

            # Stanozolol (11 rings)
            "amplified_confidence": round(amplified_confidence, 6),
            "ring_outputs": ring_outputs,

            # DMT (5 channels)
            "dmt_confidence": round(dmt_confidence, 6),
            "dmt_direction": dmt_direction,
            "channel_results": channel_results,

            # Decision Gates (13 gates)
            "gates_passed": gates_passed,
            "gates_total": gates_total,
            "gate_ratio": round(gate_ratio, 4),
            "gate_results": gate_results,

            # TE Bridge binding
            "binding": binding,

            # Quantum (16 qubits, 8192 shots)
            "quantum": quantum_result,
        }


# ============================================================
# CONVENIENCE FUNCTION FOR TEQA INTEGRATION
# ============================================================

# Module-level singleton so the bridge persists across calls
_bridge_instance: Optional[StanozololDMTBridge] = None


def get_bridge(shots: int = SHOTS_BRIDGE) -> StanozololDMTBridge:
    """Get or create the bridge singleton."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = StanozololDMTBridge(shots=shots)
    return _bridge_instance


def apply_bridge(teqa_result: Dict, shots: int = SHOTS_BRIDGE) -> Dict:
    """
    Convenience function for TEQA integration.

    Takes a TEQA result dict, runs it through the bridge, and returns
    the TEQA result with confidence adjusted.

    This is the function that teqa_v3_neural_te.py should call:

        try:
            from stanozolol_dmt_bridge import apply_bridge
            result = apply_bridge(result)
        except ImportError:
            pass
    """
    bridge = get_bridge(shots)
    bridge_result = bridge.process(teqa_result)

    # Inject bridge data into the TEQA result
    teqa_result["stanozolol_dmt_bridge"] = bridge_result

    # Adjust confidence
    original = teqa_result.get("confidence", 0.0)
    delta = bridge_result["confidence_delta"]
    teqa_result["confidence"] = float(np.clip(
        original + delta,
        MIN_CONFIDENCE_FLOOR,
        MAX_CONFIDENCE_CEILING,
    ))

    # Log the adjustment
    log.info(
        "BRIDGE [%s]: confidence %.4f -> %.4f (delta %+.4f, %s, %d/%d gates, regime=%s)",
        bridge_result["recommendation"],
        original,
        teqa_result["confidence"],
        delta,
        bridge_result["recommendation"],
        bridge_result["gates_passed"],
        bridge_result["gates_total"],
        bridge_result["regime"],
    )

    return teqa_result


# ============================================================
# STANDALONE TEST
# ============================================================

def _run_standalone_test():
    """
    Standalone test: create a synthetic TEQA signal and pass it through
    the full 11-layer / 13-gate / 5-channel pipeline. Shows output at
    each stage for processing depth verification.
    """
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    print("=" * 80)
    print("  STANOZOLOL-DMT TE BRIDGE -- Standalone Test")
    print("  Version:", VERSION)
    print("  Architecture: 16-qubit / 8192-shot / 5-channel / 13-gate / 11-ring")
    print("=" * 80)

    # Create synthetic TE activations (33 TEs from TEQA v3.0)
    np.random.seed(42)
    te_names = [
        # Original 25
        "BEL_Pao", "DIRS1", "Ty1_copia", "Ty3_gypsy", "Ty5",
        "Alu", "LINE", "Penelope", "RTE", "SINE", "VIPER_Ngaro",
        "CACTA", "Crypton", "Helitron", "hobo", "I_element",
        "Mariner_Tc1", "Mavericks_Polinton", "Mutator", "P_element",
        "PIF_Harbinger", "piggyBac", "pogo", "Rag_like", "Transib",
        # Neural 8
        "L1_Neuronal", "L1_Somatic", "HERV_Synapse", "SVA_Regulatory",
        "Alu_Exonization", "TRIM28_Silencer", "piwiRNA_Neural", "Arc_Capsid",
    ]

    te_activations = []
    for i, name in enumerate(te_names):
        strength = float(np.random.beta(2, 3))  # Skewed toward lower values
        direction = np.random.choice([-1, 0, 1], p=[0.3, 0.1, 0.6])  # Bullish bias
        te_activations.append({
            "te": name,
            "strength": round(strength, 4),
            "direction": int(direction),
            "details": {"class": "Class_I" if i < 11 else ("Class_II" if i < 25 else "Neural")},
        })

    # Create synthetic TEQA result
    synthetic_teqa = {
        "version": "TEQA-3.0-NEURAL-TE",
        "symbol": "BTCUSD",
        "direction": 1,
        "confidence": 0.62,
        "concordance": 0.68,
        "amplitude_sq": 0.35,
        "shock_score": 1.1,
        "shock_label": "NORMAL",
        "consensus_direction": 1,
        "consensus_score": 0.75,
        "neural_consensus_pass": True,
        "relationship": "SAME_SPECIES",
        "domestication_boost": 1.15,
        "n_active_class_i": 7,
        "n_active_class_ii": 5,
        "n_active_neural": 3,
        "gates": {
            "G7_neural_consensus": True,
            "G8_genomic_shock": True,
            "G9_speciation": True,
            "G10_domestication": True,
        },
        "te_activations": te_activations,
        "hgh": {"active": False},
    }

    print("\n--- SYNTHETIC TEQA INPUT ---")
    print(f"  Symbol:         {synthetic_teqa['symbol']}")
    print(f"  Direction:      {synthetic_teqa['direction']}")
    print(f"  Confidence:     {synthetic_teqa['confidence']}")
    print(f"  Concordance:    {synthetic_teqa['concordance']}")
    print(f"  Shock Score:    {synthetic_teqa['shock_score']}")
    print(f"  Shock Label:    {synthetic_teqa['shock_label']}")
    print(f"  Consensus Dir:  {synthetic_teqa['consensus_direction']}")
    print(f"  Consensus Score:{synthetic_teqa['consensus_score']}")
    print(f"  Dom Boost:      {synthetic_teqa['domestication_boost']}")
    print(f"  Active TEs:     {synthetic_teqa['n_active_class_i']}I + {synthetic_teqa['n_active_class_ii']}II + {synthetic_teqa['n_active_neural']}N")

    # Run the bridge
    print("\n" + "=" * 80)
    print("  RUNNING BRIDGE PIPELINE")
    print("=" * 80)

    bridge = StanozololDMTBridge()
    result = bridge.process(synthetic_teqa)

    # === Stage 1: Regime Detection ===
    print("\n--- STAGE 1: REGIME DETECTION ---")
    print(f"  Regime:         {result['regime']}")
    print(f"  Temperature:    {result['regime_temperature']} C")
    print(f"  Stability:      {result['regime_stability']}")
    print(f"  Stanozolol Gain:{result['stanozolol_gain']}")
    print(f"  DMT Clarity:    {result['dmt_clarity']}")

    # === Stage 2: TE Binding ===
    print("\n--- STAGE 2: TE BINDING STRENGTHS ---")
    binding = result["binding"]
    print(f"  Stanozolol Binding: {binding['stanozolol_binding']:.4f}")
    print(f"  DMT Binding:        {binding['dmt_binding']:.4f}")
    print(f"  Bridge Linker:      {binding['bridge_linker']:.4f}")
    print(f"  Cross-Talk:         {binding['cross_talk']:.4f}")

    # === Stage 3: DMT Pattern Recognition ===
    print("\n--- STAGE 3: DMT PATTERN RECOGNITION (5 Channels) ---")
    for ch in result["channel_results"]:
        clarity_str = ""
        if "clarity_applied" in ch:
            clarity_str = f" [clarity={ch['clarity_applied']:.2f}]"
        print(f"  Channel {ch['channel']} ({ch['name']:20s}): "
              f"conf={ch['confidence']:.4f}  dir={ch['direction']:+d}{clarity_str}")
    print(f"  DMT Aggregate:    conf={result['dmt_confidence']:.4f}  dir={result['dmt_direction']:+d}")

    # === Stage 4: Stanozolol Amplification ===
    print("\n--- STAGE 4: STANOZOLOL AMPLIFICATION (11 Rings) ---")
    for ring in result["ring_outputs"]:
        delta_str = f"{ring['delta']:+.6f}" if ring['delta'] != 0 else " 0.000000"
        print(f"  Ring {ring['ring']:2d} ({ring['name']:35s}): "
              f"{ring['input']:.6f} -> {ring['output']:.6f}  ({delta_str})")
    print(f"  Amplified Confidence: {result['amplified_confidence']:.6f}")

    # === Stage 5: Decision Gates ===
    print("\n--- STAGE 5: DECISION GATES (13 Stereo-Gates) ---")
    for gate in result["gate_results"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  G{gate['gate']:2d} ({gate['name']:28s}): [{status}]  {gate['details']}")
    print(f"  Gates Passed: {result['gates_passed']}/{result['gates_total']} "
          f"(ratio={result['gate_ratio']:.4f}, majority={GATE_MAJORITY})")

    # === Stage 6: Quantum Circuit ===
    print("\n--- STAGE 6: QUANTUM BRIDGE CIRCUIT (16 qubits, 8192 shots) ---")
    qr = result["quantum"]
    print(f"  Vote Long:      {qr['vote_long']:.4f}")
    print(f"  Vote Short:     {qr['vote_short']:.4f}")
    print(f"  Top State:      {qr['top_state']}")
    print(f"  Top Count:      {qr['top_count']}")
    print(f"  Unique States:  {qr['n_unique_states']}")
    print(f"  Shots:          {qr['n_shots']}")
    if qr.get("channel_biases"):
        print(f"  Channel Biases: {[round(b, 4) for b in qr['channel_biases']]}")

    # === Stage 7: Final Adjustment ===
    print("\n--- STAGE 7: FINAL CONFIDENCE ADJUSTMENT ---")
    print(f"  Original Confidence: {result['original_confidence']:.6f}")
    print(f"  Bridge Confidence:   {result['bridge_confidence']:.6f}")
    print(f"  Confidence Delta:    {result['confidence_delta']:+.6f}")
    print(f"  Adjusted Confidence: {result['adjusted_confidence']:.6f}")
    print(f"  Recommendation:      {result['recommendation']}")
    print(f"  Elapsed:             {result['elapsed_ms']:.2f} ms")

    # === Now test STRESS mode ===
    print("\n" + "=" * 80)
    print("  STRESS TEST: High Volatility (250 C Mode)")
    print("=" * 80)

    stress_teqa = dict(synthetic_teqa)
    stress_teqa["shock_score"] = 2.3
    stress_teqa["shock_label"] = "SHOCK"
    stress_teqa["confidence"] = 0.55
    stress_teqa["concordance"] = 0.45
    stress_teqa["consensus_score"] = 0.60
    stress_teqa["neural_consensus_pass"] = False
    stress_teqa["domestication_boost"] = 0.9
    stress_teqa["gates"]["G7_neural_consensus"] = False
    stress_teqa["gates"]["G10_domestication"] = False

    stress_result = bridge.process(stress_teqa)

    print(f"\n  Regime:              {stress_result['regime']}")
    print(f"  Temperature:         {stress_result['regime_temperature']} C")
    print(f"  Stanozolol Gain:     {stress_result['stanozolol_gain']}")
    print(f"  DMT Clarity:         {stress_result['dmt_clarity']}")
    print(f"  Gates Passed:        {stress_result['gates_passed']}/{stress_result['gates_total']}")
    print(f"  Original Confidence: {stress_result['original_confidence']:.6f}")
    print(f"  Confidence Delta:    {stress_result['confidence_delta']:+.6f}")
    print(f"  Adjusted Confidence: {stress_result['adjusted_confidence']:.6f}")
    print(f"  Recommendation:      {stress_result['recommendation']}")

    # === Test apply_bridge convenience function ===
    print("\n" + "=" * 80)
    print("  INTEGRATION TEST: apply_bridge() function")
    print("=" * 80)

    test_teqa = dict(synthetic_teqa)
    test_teqa["confidence"] = 0.62
    original_conf = test_teqa["confidence"]

    # Reset the singleton for a clean test
    global _bridge_instance
    _bridge_instance = None

    test_teqa = apply_bridge(test_teqa)

    print(f"\n  Before: confidence = {original_conf:.4f}")
    print(f"  After:  confidence = {test_teqa['confidence']:.4f}")
    print(f"  Bridge data injected: {'stanozolol_dmt_bridge' in test_teqa}")
    print(f"  Recommendation: {test_teqa['stanozolol_dmt_bridge']['recommendation']}")

    print("\n" + "=" * 80)
    print("  ALL TESTS COMPLETE")
    print("=" * 80)

    return result


if __name__ == "__main__":
    _run_standalone_test()
