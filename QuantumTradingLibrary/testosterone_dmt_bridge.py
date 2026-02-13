"""
TESTOSTERONE-DMT TE BRIDGE
==========================

Molecular Extension Module #1 for Quantum Children Trading System

TESTOSTERONE (Base Hormone):
- Raw power amplification
- 4 heavy processing layers (fewer than stanozolol's 11, but heavier per layer)
- Trend-following bias, wider stops, bigger targets
- Aromatase adaptation (flips from aggression to defense)

DMT (Pattern Recognition):
- 5 independent pattern channels
- Tryptamine-core serotonin-family signal processing
- Dual noise filters (N,N-dimethyl groups)

INTERFACE:
- Must match stanozolol_dmt_bridge.py so BRAIN can swap between them
- process_signal(signal_data) -> boost/suppress decision
- Same gate array structure, different parameters

STRATEGY PROFILE:
- MORE AGGRESSIVE than stanozolol
- WIDER stop losses (hold through noise)
- BIGGER take profit targets
- TREND-FOLLOWING (don't fight the momentum)
- REGIME-ADAPTIVE (aromatase controller)

Authors: Biskits + Claude (Opus 4.6)
Date:    2026-02-12
Version: TESTOSTERONE-DMT-BRIDGE-V1
"""

import json
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

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

VERSION = "TESTOSTERONE-DMT-BRIDGE-V1"

# Testosterone molecular properties
N_RINGS = 4                     # Cyclopentanoperhydrophenanthrene core
N_QUBITS_PER_RING = 2           # 2 qubits per ring
N_QUBITS_TESTOSTERONE = 8       # 4 rings × 2 qubits

# Quantum shots (half of stanozolol, but heavier measurements)
DEFAULT_SHOTS = 4096

# DMT pattern channels
N_DMT_CHANNELS = 5

# Testosterone characteristics
ANABOLIC_ANDROGENIC_RATIO = 1.0  # Equal weight: growth vs aggression
HALF_LIFE_HOURS = 4.5            # Position holding time bias
DHT_CONVERSION_RATE = 0.05       # Signal concentration amplification

# Aromatase thresholds (testosterone → estrogen)
AROMATASE_DRAWDOWN_THRESHOLD = 0.10    # 10% drawdown triggers defensive mode
AROMATASE_VOLATILITY_THRESHOLD = 2.5   # High volatility triggers aromatization

# Decision gate thresholds (4 gates, ALL must pass)
GATE_TREND_MIN_STRENGTH = 0.60         # Trend must be clear
GATE_MOMENTUM_MIN_ACCEL = 0.55         # Momentum must be accelerating
GATE_RISK_REWARD_MIN = 2.0             # RR ratio minimum (2:1 minimum)
GATE_IMMUNE_MAX_CONFLICT = 0.30        # CRISPR/VDJ conflict threshold

# Molecular binding sites
ANDROGEN_RECEPTOR_AFFINITY = 0.85      # Signal strength boost
RECEPTOR_BINDING_THRESHOLD = 0.50      # Minimum to activate

# Position sizing (aggressive)
BASE_POSITION_MULTIPLIER = 1.5         # 1.5x normal position size
MAX_POSITION_MULTIPLIER = 2.5          # Cap at 2.5x
MIN_POSITION_MULTIPLIER = 0.5          # Floor at 0.5x in defensive mode

# ============================================================
# DATA STRUCTURES
# ============================================================

class ProcessingLayer(Enum):
    """The 4 heavy processing layers (testosterone's 4 rings)."""
    TREND_DETECTION = 1        # Ring 1: Androgen binding - trend strength
    MOMENTUM_AMPLIFICATION = 2 # Ring 2: DHT conversion - signal concentration
    POSITION_SIZING = 3        # Ring 3: Anabolic ratio - growth vs aggression
    EXIT_TIMING = 4            # Ring 4: Aromatization - flip to defense


class DMTChannel(Enum):
    """The 5 DMT pattern recognition channels."""
    TRYPTAMINE_CORE = 0        # Core serotonin-family processing
    METHYL_FILTER_1 = 1        # First noise filter
    METHYL_FILTER_2 = 2        # Second noise filter
    INDOLE_PATTERN = 3         # Complex pattern recognition
    RESONANCE_DETECTOR = 4     # Cross-frequency resonance


class RegimeState(Enum):
    """Aromatase-controlled regime states."""
    FULL_TESTOSTERONE = "aggressive"     # Normal mode: trend following
    AROMATIZING = "transitional"         # Starting to flip
    FULL_ESTROGEN = "defensive"          # Defense mode: tight stops


@dataclass
class TestosteroneRing:
    """Represents one of the 4 rings in testosterone molecule."""
    ring_id: int
    layer: ProcessingLayer
    qubit_pair: Tuple[int, int]
    activation_strength: float = 0.0
    binding_affinity: float = ANDROGEN_RECEPTOR_AFFINITY

    def amplify_signal(self, signal: float) -> float:
        """Amplify signal based on androgen receptor binding."""
        if abs(signal) < RECEPTOR_BINDING_THRESHOLD:
            return signal * 0.5  # Weak signals get suppressed
        return signal * self.binding_affinity * (1 + self.activation_strength)


@dataclass
class DMTPatternResult:
    """Result from DMT pattern recognition."""
    channel: DMTChannel
    pattern_detected: bool
    confidence: float
    signal_polarity: int  # -1, 0, +1
    noise_filtered: bool


@dataclass
class AromataseState:
    """Tracks aromatization state (testosterone → estrogen conversion)."""
    regime: RegimeState = RegimeState.FULL_TESTOSTERONE
    aromatization_level: float = 0.0  # 0.0 = full T, 1.0 = full E
    recent_drawdown: float = 0.0
    recent_volatility: float = 1.0
    trades_in_regime: int = 0

    def update(self, drawdown: float, volatility: float):
        """Update aromatization state based on market conditions."""
        self.recent_drawdown = drawdown
        self.recent_volatility = volatility

        # Calculate aromatization level
        drawdown_factor = min(1.0, drawdown / AROMATASE_DRAWDOWN_THRESHOLD)
        volatility_factor = min(1.0, max(0.0,
            (volatility - 1.0) / (AROMATASE_VOLATILITY_THRESHOLD - 1.0)))

        self.aromatization_level = max(drawdown_factor, volatility_factor)

        # Determine regime
        if self.aromatization_level >= 0.7:
            self.regime = RegimeState.FULL_ESTROGEN
        elif self.aromatization_level >= 0.3:
            self.regime = RegimeState.AROMATIZING
        else:
            self.regime = RegimeState.FULL_TESTOSTERONE

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on regime."""
        if self.regime == RegimeState.FULL_TESTOSTERONE:
            return BASE_POSITION_MULTIPLIER
        elif self.regime == RegimeState.AROMATIZING:
            return 1.0  # Normal sizing
        else:
            return MIN_POSITION_MULTIPLIER  # Defensive sizing

    def get_stop_multiplier(self) -> float:
        """Get stop loss multiplier based on regime."""
        if self.regime == RegimeState.FULL_TESTOSTERONE:
            return 1.5  # Wide stops
        elif self.regime == RegimeState.AROMATIZING:
            return 1.2  # Medium stops
        else:
            return 0.8  # Tight stops


# ============================================================
# CORE CLASSES
# ============================================================

class TestosteroneCore:
    """
    Raw power amplification through 4 heavy processing layers.
    Each layer corresponds to one ring in the testosterone molecule.
    """

    def __init__(self):
        self.rings = [
            TestosteroneRing(
                ring_id=i,
                layer=list(ProcessingLayer)[i],
                qubit_pair=(i*2, i*2+1)
            )
            for i in range(N_RINGS)
        ]
        self.aromatase = AromataseState()

        log.info("TestosteroneCore initialized: 4 rings, %d qubits", N_QUBITS_TESTOSTERONE)

    def process_layer1_trend_detection(self, market_data: Dict) -> Dict:
        """
        Layer 1: Trend Detection (Androgen Receptor Binding)
        Determines how strong the trend is and whether to engage.
        """
        close_prices = market_data.get('close', [])
        if len(close_prices) < 50:
            return {'trend_strength': 0.0, 'trend_direction': 0}

        # Simple trend strength: ratio of price to moving average
        recent = np.array(close_prices[-50:])
        ma20 = np.mean(recent[-20:])
        ma50 = np.mean(recent)
        current = recent[-1]

        # Trend direction
        if ma20 > ma50 * 1.01:
            trend_direction = 1  # Uptrend
        elif ma20 < ma50 * 0.99:
            trend_direction = -1  # Downtrend
        else:
            trend_direction = 0  # Ranging

        # Trend strength (deviation from mean)
        deviation = abs(current - ma50) / ma50
        trend_strength = min(1.0, deviation * 10)  # Normalize to [0, 1]

        # Androgen receptor binding: amplify if trend is strong
        ring = self.rings[0]
        ring.activation_strength = trend_strength
        amplified_strength = ring.amplify_signal(trend_strength)

        return {
            'trend_strength': amplified_strength,
            'trend_direction': trend_direction,
            'ma20': ma20,
            'ma50': ma50,
            'binding_active': amplified_strength >= RECEPTOR_BINDING_THRESHOLD
        }

    def process_layer2_momentum_amplification(self, market_data: Dict, layer1: Dict) -> Dict:
        """
        Layer 2: Momentum Amplification (DHT Conversion)
        5α-reductase converts testosterone to DHT, concentrating the signal.
        Amplifies the strongest momentum signal further.
        """
        close_prices = market_data.get('close', [])
        if len(close_prices) < 20:
            return {'momentum': 0.0, 'acceleration': 0.0}

        # Calculate momentum as rate of change
        recent = np.array(close_prices[-20:])
        roc5 = (recent[-1] - recent[-5]) / recent[-5] if recent[-5] != 0 else 0
        roc10 = (recent[-1] - recent[-10]) / recent[-10] if recent[-10] != 0 else 0

        # Acceleration: is momentum increasing?
        acceleration = (roc5 - roc10) / abs(roc10) if roc10 != 0 else 0

        # DHT conversion: concentrate the signal
        base_momentum = (roc5 + roc10) / 2
        concentrated_momentum = base_momentum * (1 + DHT_CONVERSION_RATE * abs(acceleration))

        # Ring amplification
        ring = self.rings[1]
        ring.activation_strength = abs(acceleration)
        amplified_momentum = ring.amplify_signal(concentrated_momentum)

        return {
            'momentum': amplified_momentum,
            'acceleration': acceleration,
            'dht_converted': abs(acceleration) > 0.1,
            'concentration_factor': 1 + DHT_CONVERSION_RATE * abs(acceleration)
        }

    def process_layer3_position_sizing(self, layer1: Dict, layer2: Dict) -> Dict:
        """
        Layer 3: Position Sizing (Anabolic:Androgenic Ratio)
        Testosterone has 1:1 ratio - balance growth (profit) and aggression (entry).
        """
        trend_strength = layer1.get('trend_strength', 0.0)
        momentum = layer2.get('momentum', 0.0)

        # Anabolic component (growth/profit): trend strength
        anabolic = trend_strength

        # Androgenic component (aggression/entry): momentum
        androgenic = abs(momentum)

        # 1:1 ratio: equal weighting
        combined_signal = (anabolic + androgenic) / 2

        # Position multiplier from aromatase state
        regime_multiplier = self.aromatase.get_position_multiplier()

        # Final position sizing
        position_size = combined_signal * regime_multiplier
        position_size = max(MIN_POSITION_MULTIPLIER,
                           min(MAX_POSITION_MULTIPLIER, position_size))

        ring = self.rings[2]
        ring.activation_strength = combined_signal

        return {
            'position_size_multiplier': position_size,
            'anabolic_component': anabolic,
            'androgenic_component': androgenic,
            'ratio': ANABOLIC_ANDROGENIC_RATIO,
            'regime_multiplier': regime_multiplier
        }

    def process_layer4_exit_timing(self, market_data: Dict, layer1: Dict, layer2: Dict) -> Dict:
        """
        Layer 4: Exit Timing (Aromatization)
        When does testosterone convert to estrogen? When to flip from aggression to defense?
        """
        # Calculate recent performance (simplified - would use real trade history)
        volatility = market_data.get('volatility', 1.0)
        drawdown = market_data.get('drawdown', 0.0)

        # Update aromatase state
        self.aromatase.update(drawdown, volatility)

        # Determine exit strategy based on regime
        if self.aromatase.regime == RegimeState.FULL_TESTOSTERONE:
            exit_strategy = 'trend_following'  # Let winners run
            stop_multiplier = 1.5
            target_multiplier = 3.0
        elif self.aromatase.regime == RegimeState.AROMATIZING:
            exit_strategy = 'balanced'
            stop_multiplier = 1.2
            target_multiplier = 2.0
        else:  # FULL_ESTROGEN
            exit_strategy = 'defensive'  # Protect capital
            stop_multiplier = 0.8
            target_multiplier = 1.5

        ring = self.rings[3]
        ring.activation_strength = self.aromatase.aromatization_level

        return {
            'exit_strategy': exit_strategy,
            'stop_multiplier': stop_multiplier,
            'target_multiplier': target_multiplier,
            'regime': self.aromatase.regime.value,
            'aromatization_level': self.aromatase.aromatization_level,
            'aromatized': self.aromatase.regime != RegimeState.FULL_TESTOSTERONE
        }

    def process_all_layers(self, market_data: Dict) -> Dict:
        """Execute all 4 testosterone processing layers sequentially."""
        layer1 = self.process_layer1_trend_detection(market_data)
        layer2 = self.process_layer2_momentum_amplification(market_data, layer1)
        layer3 = self.process_layer3_position_sizing(layer1, layer2)
        layer4 = self.process_layer4_exit_timing(market_data, layer1, layer2)

        return {
            'layer1_trend': layer1,
            'layer2_momentum': layer2,
            'layer3_sizing': layer3,
            'layer4_exit': layer4,
            'all_rings_activated': all(r.activation_strength > 0.1 for r in self.rings)
        }


class DMTPatternEngine:
    """
    5-channel pattern recognition system.
    DMT (N,N-Dimethyltryptamine) molecular structure maps to trading patterns.
    """

    def __init__(self):
        self.channels = {ch: None for ch in DMTChannel}
        log.info("DMTPatternEngine initialized: %d channels", N_DMT_CHANNELS)

    def process_tryptamine_core(self, signal: float) -> DMTPatternResult:
        """Channel 0: Core serotonin-family signal processing."""
        # Tryptamine core: basic signal strength detection
        confidence = min(1.0, abs(signal) / 2.0)
        pattern_detected = confidence > 0.5
        polarity = 1 if signal > 0 else (-1 if signal < 0 else 0)

        return DMTPatternResult(
            channel=DMTChannel.TRYPTAMINE_CORE,
            pattern_detected=pattern_detected,
            confidence=confidence,
            signal_polarity=polarity,
            noise_filtered=False
        )

    def process_methyl_filter_1(self, signal: float, noise_level: float) -> DMTPatternResult:
        """Channel 1: First N-methyl noise filter."""
        # Filter out low-amplitude noise
        filtered_signal = signal if abs(signal) > noise_level else 0
        noise_filtered = filtered_signal != signal

        confidence = min(1.0, abs(filtered_signal))
        pattern_detected = abs(filtered_signal) > 0.3
        polarity = 1 if filtered_signal > 0 else (-1 if filtered_signal < 0 else 0)

        return DMTPatternResult(
            channel=DMTChannel.METHYL_FILTER_1,
            pattern_detected=pattern_detected,
            confidence=confidence,
            signal_polarity=polarity,
            noise_filtered=noise_filtered
        )

    def process_methyl_filter_2(self, signal: float, noise_level: float) -> DMTPatternResult:
        """Channel 2: Second N-methyl noise filter (dual filtering)."""
        # Second stage filtering: remove oscillations
        threshold = noise_level * 1.5
        filtered_signal = signal if abs(signal) > threshold else 0
        noise_filtered = filtered_signal != signal

        confidence = min(1.0, abs(filtered_signal) / 2.0)
        pattern_detected = abs(filtered_signal) > 0.4
        polarity = 1 if filtered_signal > 0 else (-1 if filtered_signal < 0 else 0)

        return DMTPatternResult(
            channel=DMTChannel.METHYL_FILTER_2,
            pattern_detected=pattern_detected,
            confidence=confidence,
            signal_polarity=polarity,
            noise_filtered=noise_filtered
        )

    def process_indole_pattern(self, market_data: Dict) -> DMTPatternResult:
        """Channel 3: Complex pattern recognition (indole structure)."""
        # Indole ring: recognize complex multi-timeframe patterns
        close_prices = market_data.get('close', [])
        if len(close_prices) < 30:
            return DMTPatternResult(
                channel=DMTChannel.INDOLE_PATTERN,
                pattern_detected=False,
                confidence=0.0,
                signal_polarity=0,
                noise_filtered=False
            )

        recent = np.array(close_prices[-30:])

        # Detect swing high/low pattern
        mid = recent[-15]
        left = np.mean(recent[-30:-15])
        right = np.mean(recent[-15:])

        swing_high = mid > left and mid > right
        swing_low = mid < left and mid < right

        pattern_detected = swing_high or swing_low
        confidence = abs(mid - (left + right) / 2) / mid if mid != 0 else 0
        polarity = 1 if swing_high else (-1 if swing_low else 0)

        return DMTPatternResult(
            channel=DMTChannel.INDOLE_PATTERN,
            pattern_detected=pattern_detected,
            confidence=min(1.0, confidence * 10),
            signal_polarity=polarity,
            noise_filtered=True
        )

    def process_resonance_detector(self, market_data: Dict) -> DMTPatternResult:
        """Channel 4: Cross-frequency resonance detection."""
        # Detect when multiple timeframes align
        close_prices = market_data.get('close', [])
        if len(close_prices) < 50:
            return DMTPatternResult(
                channel=DMTChannel.RESONANCE_DETECTOR,
                pattern_detected=False,
                confidence=0.0,
                signal_polarity=0,
                noise_filtered=False
            )

        recent = np.array(close_prices[-50:])

        # Short, medium, long trends
        trend_short = (recent[-1] - recent[-10]) / recent[-10] if recent[-10] != 0 else 0
        trend_medium = (recent[-1] - recent[-25]) / recent[-25] if recent[-25] != 0 else 0
        trend_long = (recent[-1] - recent[-50]) / recent[-50] if recent[-50] != 0 else 0

        # Resonance: all trends pointing same direction
        all_positive = trend_short > 0 and trend_medium > 0 and trend_long > 0
        all_negative = trend_short < 0 and trend_medium < 0 and trend_long < 0

        pattern_detected = all_positive or all_negative
        confidence = min(abs(trend_short), abs(trend_medium), abs(trend_long)) * 10
        confidence = min(1.0, confidence)
        polarity = 1 if all_positive else (-1 if all_negative else 0)

        return DMTPatternResult(
            channel=DMTChannel.RESONANCE_DETECTOR,
            pattern_detected=pattern_detected,
            confidence=confidence,
            signal_polarity=polarity,
            noise_filtered=True
        )

    def process_all_channels(self, market_data: Dict, signal: float, noise_level: float = 0.1) -> Dict:
        """Process all 5 DMT channels."""
        results = {
            'tryptamine_core': self.process_tryptamine_core(signal),
            'methyl_filter_1': self.process_methyl_filter_1(signal, noise_level),
            'methyl_filter_2': self.process_methyl_filter_2(signal, noise_level),
            'indole_pattern': self.process_indole_pattern(market_data),
            'resonance_detector': self.process_resonance_detector(market_data)
        }

        # Consensus: how many channels agree?
        detections = [r.pattern_detected for r in results.values()]
        polarities = [r.signal_polarity for r in results.values()]
        confidences = [r.confidence for r in results.values()]

        consensus_rate = sum(detections) / len(detections)
        avg_confidence = np.mean(confidences)
        dominant_polarity = max(set(polarities), key=polarities.count)

        return {
            'channel_results': results,
            'consensus_rate': consensus_rate,
            'avg_confidence': avg_confidence,
            'dominant_polarity': dominant_polarity,
            'all_channels_agree': len(set(polarities)) == 1 and 0 not in polarities
        }


class TEBridge:
    """
    TE (Transposable Element) string connecting Testosterone to DMT.
    Routes signals through the molecular pathway.
    """

    def __init__(self):
        self.binding_sites = {
            'androgen_receptor': ANDROGEN_RECEPTOR_AFFINITY,
            'dht_converter': DHT_CONVERSION_RATE,
            'aromatase': 0.0  # Updated dynamically
        }
        log.info("TEBridge initialized: Testosterone-DMT pathway active")

    def route_signal(self, testosterone_output: Dict, dmt_output: Dict) -> Dict:
        """Route signals from testosterone core through DMT patterns."""
        # Extract key signals
        trend = testosterone_output['layer1_trend']
        momentum = testosterone_output['layer2_momentum']
        sizing = testosterone_output['layer3_sizing']
        exit = testosterone_output['layer4_exit']

        dmt_consensus = dmt_output['consensus_rate']
        dmt_confidence = dmt_output['avg_confidence']
        dmt_polarity = dmt_output['dominant_polarity']

        # Combine testosterone power with DMT pattern recognition
        # Testosterone provides the raw strength, DMT provides the direction
        raw_strength = (trend['trend_strength'] + abs(momentum['momentum'])) / 2
        pattern_confidence = dmt_confidence

        # Final signal: testosterone strength × DMT direction
        final_signal = raw_strength * pattern_confidence * dmt_polarity

        # TE activation: high when both systems agree
        te_activation = min(1.0, raw_strength * dmt_consensus)

        return {
            'raw_strength': raw_strength,
            'pattern_confidence': pattern_confidence,
            'final_signal': final_signal,
            'te_activation': te_activation,
            'polarity': dmt_polarity,
            'binding_active': trend['binding_active'] and dmt_consensus > 0.6
        }


class DecisionGateArray:
    """
    4-gate decision array (matching testosterone's 4 rings).
    ALL 4 gates must pass for signal to activate.
    Stricter per-gate requirements than stanozolol's 13 gates.
    """

    def __init__(self):
        self.gates_passed = {i: False for i in range(1, 5)}
        log.info("DecisionGateArray initialized: 4 gates (ALL must pass)")

    def gate1_trend_filter(self, testosterone_output: Dict) -> Tuple[bool, str]:
        """Gate 1: Is there a clear trend?"""
        trend = testosterone_output['layer1_trend']
        strength = trend['trend_strength']
        direction = trend['trend_direction']

        passed = strength >= GATE_TREND_MIN_STRENGTH and direction != 0
        reason = f"Trend strength {strength:.3f} vs {GATE_TREND_MIN_STRENGTH}, direction {direction}"

        return passed, reason

    def gate2_momentum_filter(self, testosterone_output: Dict) -> Tuple[bool, str]:
        """Gate 2: Is momentum accelerating?"""
        momentum = testosterone_output['layer2_momentum']
        acceleration = momentum['acceleration']

        passed = abs(acceleration) >= GATE_MOMENTUM_MIN_ACCEL
        reason = f"Acceleration {abs(acceleration):.3f} vs {GATE_MOMENTUM_MIN_ACCEL}"

        return passed, reason

    def gate3_risk_reward_filter(self, exit_output: Dict) -> Tuple[bool, str]:
        """Gate 3: Is the risk:reward ratio acceptable?"""
        target_mult = exit_output['target_multiplier']
        stop_mult = exit_output['stop_multiplier']

        # RR = target / stop
        rr_ratio = target_mult / stop_mult if stop_mult > 0 else 0

        passed = rr_ratio >= GATE_RISK_REWARD_MIN
        reason = f"RR ratio {rr_ratio:.2f} vs {GATE_RISK_REWARD_MIN}"

        return passed, reason

    def gate4_immune_clearance(self, bridge_output: Dict, immune_conflict: float = 0.0) -> Tuple[bool, str]:
        """Gate 4: Is the immune system (CRISPR/VDJ) clear?"""
        # Check for immune system conflicts
        # (In full system, this would check CRISPR_CAS and VDJ outputs)

        passed = immune_conflict <= GATE_IMMUNE_MAX_CONFLICT
        reason = f"Immune conflict {immune_conflict:.3f} vs {GATE_IMMUNE_MAX_CONFLICT}"

        return passed, reason

    def evaluate_all_gates(self, testosterone_output: Dict, bridge_output: Dict,
                          immune_conflict: float = 0.0) -> Dict:
        """Evaluate all 4 gates. ALL must pass."""
        gate1_pass, gate1_reason = self.gate1_trend_filter(testosterone_output)
        gate2_pass, gate2_reason = self.gate2_momentum_filter(testosterone_output)
        gate3_pass, gate3_reason = self.gate3_risk_reward_filter(
            testosterone_output['layer4_exit']
        )
        gate4_pass, gate4_reason = self.gate4_immune_clearance(bridge_output, immune_conflict)

        all_passed = gate1_pass and gate2_pass and gate3_pass and gate4_pass

        return {
            'all_gates_passed': all_passed,
            'gates': {
                'gate1_trend': {'passed': gate1_pass, 'reason': gate1_reason},
                'gate2_momentum': {'passed': gate2_pass, 'reason': gate2_reason},
                'gate3_risk_reward': {'passed': gate3_pass, 'reason': gate3_reason},
                'gate4_immune': {'passed': gate4_pass, 'reason': gate4_reason}
            },
            'gates_passed_count': sum([gate1_pass, gate2_pass, gate3_pass, gate4_pass]),
            'gates_total': 4
        }


class QuantumCircuitBuilder:
    """
    Builds 8-qubit quantum circuits for testosterone processing.
    4 rings × 2 qubits per ring = 8 qubits total.
    """

    def __init__(self, shots: int = DEFAULT_SHOTS):
        self.shots = shots
        self.simulator = AerSimulator() if QISKIT_AVAILABLE else None
        log.info("QuantumCircuitBuilder initialized: %d qubits, %d shots",
                N_QUBITS_TESTOSTERONE, shots)

    def build_testosterone_circuit(self, testosterone_output: Dict,
                                   dmt_output: Dict) -> Optional['QuantumCircuit']:
        """Build 8-qubit circuit representing testosterone molecular dynamics."""
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(N_QUBITS_TESTOSTERONE, N_QUBITS_TESTOSTERONE)

        # Extract layer outputs
        trend = testosterone_output['layer1_trend']
        momentum = testosterone_output['layer2_momentum']
        sizing = testosterone_output['layer3_sizing']
        exit = testosterone_output['layer4_exit']

        # Layer 1: Initial rotations (one per qubit pair)
        # Ring 1 (qubits 0-1): Trend
        angle1 = trend['trend_strength'] * math.pi * trend['trend_direction']
        qc.ry(angle1, 0)
        qc.ry(angle1 * 0.7, 1)

        # Ring 2 (qubits 2-3): Momentum
        angle2 = momentum['momentum'] * math.pi
        qc.ry(angle2, 2)
        qc.ry(angle2 * 0.8, 3)

        # Ring 3 (qubits 4-5): Position sizing
        angle3 = sizing['anabolic_component'] * math.pi
        qc.ry(angle3, 4)
        angle4 = sizing['androgenic_component'] * math.pi
        qc.ry(angle4, 5)

        # Ring 4 (qubits 6-7): Exit timing
        angle5 = exit['aromatization_level'] * math.pi
        qc.ry(angle5, 6)
        qc.ry(-angle5, 7)  # Inverse rotation (estrogen opposition)

        # Layer 2: Entanglement (connect the rings)
        # Adjacent ring connections
        qc.cx(0, 2)  # Ring 1 → Ring 2
        qc.cx(2, 4)  # Ring 2 → Ring 3
        qc.cx(4, 6)  # Ring 3 → Ring 4

        # Intra-ring entanglement
        qc.cx(0, 1)  # Within Ring 1
        qc.cx(2, 3)  # Within Ring 2
        qc.cx(4, 5)  # Within Ring 3
        qc.cx(6, 7)  # Within Ring 4

        # Layer 3: DMT modulation
        # DMT pattern confidence modulates all qubits
        dmt_angle = dmt_output['avg_confidence'] * math.pi * 0.3
        for qi in range(N_QUBITS_TESTOSTERONE):
            qc.ry(dmt_angle * dmt_output['dominant_polarity'], qi)

        # Layer 4: Aromatase interference (if regime is defensive)
        if exit['regime'] == 'defensive':
            # Phase flip on all qubits (estrogen mode)
            for qi in range(N_QUBITS_TESTOSTERONE):
                qc.z(qi)

        # Measurement
        qc.measure(range(N_QUBITS_TESTOSTERONE), range(N_QUBITS_TESTOSTERONE))

        return qc

    def execute_circuit(self, qc: 'QuantumCircuit') -> Dict:
        """Execute quantum circuit and return results."""
        if not QISKIT_AVAILABLE or self.simulator is None:
            return self._classical_fallback()

        try:
            job = self.simulator.run(qc, shots=self.shots)
            counts = job.result().get_counts()
            return self._analyze_counts(counts)
        except Exception as e:
            log.error("Quantum circuit execution failed: %s", e)
            return self._classical_fallback()

    def _analyze_counts(self, counts: Dict[str, int]) -> Dict:
        """Analyze quantum measurement results."""
        total_shots = sum(counts.values())

        # Vote: more 1s = bullish, more 0s = bearish
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

        # Entropy
        probs = np.array(list(counts.values())) / total_shots
        shannon_entropy = -np.sum(probs * np.log2(probs + 1e-20))
        max_entropy = N_QUBITS_TESTOSTERONE

        # Dominant state
        top_state = max(counts.items(), key=lambda x: x[1])

        return {
            'total_shots': total_shots,
            'unique_states': len(counts),
            'shannon_entropy': float(shannon_entropy),
            'max_entropy': float(max_entropy),
            'novelty': float(shannon_entropy / max_entropy),
            'vote_long': float(vote_long),
            'vote_short': float(vote_short),
            'top_state': top_state[0],
            'top_count': top_state[1],
            'vote_bias': float(vote_long - vote_short)
        }

    def _classical_fallback(self) -> Dict:
        """Fallback when quantum not available."""
        return {
            'total_shots': 0,
            'unique_states': 0,
            'shannon_entropy': 0.0,
            'max_entropy': N_QUBITS_TESTOSTERONE,
            'novelty': 0.0,
            'vote_long': 0.5,
            'vote_short': 0.5,
            'top_state': '0' * N_QUBITS_TESTOSTERONE,
            'top_count': 0,
            'vote_bias': 0.0,
            'classical_fallback': True
        }


# ============================================================
# MAIN BRIDGE INTERFACE
# ============================================================

class TestosteroneDMTBridge:
    """
    Main bridge interface. This is what the BRAIN imports and uses.
    Must match stanozolol_dmt_bridge.py interface.
    """

    def __init__(self, shots: int = DEFAULT_SHOTS):
        self.testosterone_core = TestosteroneCore()
        self.dmt_engine = DMTPatternEngine()
        self.te_bridge = TEBridge()
        self.gate_array = DecisionGateArray()
        self.quantum_builder = QuantumCircuitBuilder(shots=shots)

        self.version = VERSION
        self.molecule = "Testosterone-DMT"
        self.strategy_profile = "AGGRESSIVE"

        log.info("%s initialized: %s strategy, %d-gate array",
                VERSION, self.strategy_profile, 4)

    def process_signal(self, market_data: Dict, base_signal: float = 0.0,
                      immune_conflict: float = 0.0) -> Dict:
        """
        Main processing function. Takes market data, returns boost/suppress decision.

        Args:
            market_data: Dict with 'close', 'volume', 'volatility', 'drawdown', etc.
            base_signal: Base trading signal from TEQA (-1 to +1)
            immune_conflict: CRISPR/VDJ conflict level (0 to 1)

        Returns:
            Dict with:
                - action: 'boost', 'suppress', or 'neutral'
                - strength: 0.0 to 1.0
                - all_layers: detailed layer outputs
                - gates: gate evaluation results
                - quantum: quantum circuit results (if available)
        """
        start_time = time.time()

        # Step 1: Process through testosterone core (4 layers)
        testosterone_output = self.testosterone_core.process_all_layers(market_data)

        # Step 2: Process through DMT pattern engine (5 channels)
        dmt_output = self.dmt_engine.process_all_channels(
            market_data,
            signal=base_signal,
            noise_level=0.1
        )

        # Step 3: Route through TE bridge
        bridge_output = self.te_bridge.route_signal(testosterone_output, dmt_output)

        # Step 4: Evaluate decision gates (ALL 4 must pass)
        gate_results = self.gate_array.evaluate_all_gates(
            testosterone_output,
            bridge_output,
            immune_conflict
        )

        # Step 5: Quantum circuit (optional)
        quantum_result = None
        if QISKIT_AVAILABLE:
            qc = self.quantum_builder.build_testosterone_circuit(
                testosterone_output,
                dmt_output
            )
            if qc is not None:
                quantum_result = self.quantum_builder.execute_circuit(qc)

        # Step 6: Make final decision
        if gate_results['all_gates_passed']:
            # All gates passed - use bridge signal
            final_strength = abs(bridge_output['final_signal'])

            if bridge_output['final_signal'] > 0.3:
                action = 'boost'
            elif bridge_output['final_signal'] < -0.3:
                action = 'suppress'
            else:
                action = 'neutral'
        else:
            # Gates failed - suppress
            action = 'suppress'
            final_strength = 0.0

        # Add quantum vote if available
        if quantum_result and quantum_result.get('vote_bias'):
            quantum_bias = quantum_result['vote_bias']
            if abs(quantum_bias) > 0.2:
                # Quantum circuit has strong opinion
                if quantum_bias > 0 and action != 'boost':
                    action = 'boost'
                    final_strength = max(final_strength, abs(quantum_bias))
                elif quantum_bias < 0 and action != 'suppress':
                    action = 'suppress'
                    final_strength = max(final_strength, abs(quantum_bias))

        processing_time = time.time() - start_time

        return {
            'action': action,
            'strength': final_strength,
            'testosterone_layers': testosterone_output,
            'dmt_patterns': dmt_output,
            'te_bridge': bridge_output,
            'gates': gate_results,
            'quantum': quantum_result,
            'regime': testosterone_output['layer4_exit']['regime'],
            'position_multiplier': testosterone_output['layer3_sizing']['position_size_multiplier'],
            'stop_multiplier': testosterone_output['layer4_exit']['stop_multiplier'],
            'target_multiplier': testosterone_output['layer4_exit']['target_multiplier'],
            'processing_time_ms': processing_time * 1000,
            'version': VERSION,
            'molecule': self.molecule,
            'strategy_profile': self.strategy_profile
        }

    def get_strategy_parameters(self) -> Dict:
        """Return current strategy parameters (for BRAIN integration)."""
        return {
            'position_multiplier': self.testosterone_core.aromatase.get_position_multiplier(),
            'stop_multiplier': self.testosterone_core.aromatase.get_stop_multiplier(),
            'regime': self.testosterone_core.aromatase.regime.value,
            'strategy_bias': 'trend_following',
            'aggression_level': 'high' if self.testosterone_core.aromatase.regime == RegimeState.FULL_TESTOSTERONE else 'medium'
        }


# ============================================================
# CONVENIENCE FUNCTION FOR IMPORTS
# ============================================================

def create_bridge(shots: int = DEFAULT_SHOTS) -> TestosteroneDMTBridge:
    """Factory function to create bridge instance."""
    return TestosteroneDMTBridge(shots=shots)


# ============================================================
# MAIN (for testing)
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    print(f"\n{'='*70}")
    print(f"TESTOSTERONE-DMT TE BRIDGE - Standalone Test")
    print(f"{'='*70}\n")

    # Create bridge
    bridge = create_bridge(shots=4096)

    # Synthetic market data
    print("Generating synthetic market data...")
    np.random.seed(42)
    base_price = 100.0
    trend = np.linspace(0, 10, 100)  # Uptrend
    noise = np.random.randn(100) * 2
    close_prices = base_price + trend + noise

    synthetic_data = {
        'close': close_prices.tolist(),
        'volume': np.random.randint(1000, 5000, 100).tolist(),
        'volatility': 1.2,  # Slightly elevated
        'drawdown': 0.05   # 5% drawdown
    }

    # Test with bullish signal
    print("\n" + "="*70)
    print("TEST 1: Bullish Signal (uptrend, low volatility)")
    print("="*70)

    result = bridge.process_signal(
        market_data=synthetic_data,
        base_signal=0.8,
        immune_conflict=0.1
    )

    print(f"\nFINAL DECISION: {result['action'].upper()}")
    print(f"Strength: {result['strength']:.3f}")
    print(f"Regime: {result['regime']}")
    print(f"Position Multiplier: {result['position_multiplier']:.2f}x")
    print(f"Stop Multiplier: {result['stop_multiplier']:.2f}x")
    print(f"Target Multiplier: {result['target_multiplier']:.2f}x")
    print(f"Processing Time: {result['processing_time_ms']:.2f}ms")

    print("\n--- Layer Outputs ---")
    print(f"Layer 1 (Trend): strength={result['testosterone_layers']['layer1_trend']['trend_strength']:.3f}, "
          f"direction={result['testosterone_layers']['layer1_trend']['trend_direction']}")
    print(f"Layer 2 (Momentum): momentum={result['testosterone_layers']['layer2_momentum']['momentum']:.3f}, "
          f"acceleration={result['testosterone_layers']['layer2_momentum']['acceleration']:.3f}")
    print(f"Layer 3 (Sizing): anabolic={result['testosterone_layers']['layer3_sizing']['anabolic_component']:.3f}, "
          f"androgenic={result['testosterone_layers']['layer3_sizing']['androgenic_component']:.3f}")
    print(f"Layer 4 (Exit): strategy={result['testosterone_layers']['layer4_exit']['exit_strategy']}, "
          f"aromatization={result['testosterone_layers']['layer4_exit']['aromatization_level']:.3f}")

    print("\n--- DMT Pattern Recognition ---")
    print(f"Consensus Rate: {result['dmt_patterns']['consensus_rate']:.3f}")
    print(f"Avg Confidence: {result['dmt_patterns']['avg_confidence']:.3f}")
    print(f"Dominant Polarity: {result['dmt_patterns']['dominant_polarity']}")

    print("\n--- Decision Gates ---")
    for gate_name, gate_info in result['gates']['gates'].items():
        status = "PASS" if gate_info['passed'] else "FAIL"
        print(f"{gate_name}: {status} - {gate_info['reason']}")
    print(f"\nAll Gates Passed: {result['gates']['all_gates_passed']}")

    if result.get('quantum'):
        print("\n--- Quantum Circuit ---")
        qr = result['quantum']
        print(f"Shots: {qr['total_shots']}")
        print(f"Unique States: {qr['unique_states']}")
        print(f"Shannon Entropy: {qr['shannon_entropy']:.3f} / {qr['max_entropy']:.1f}")
        print(f"Novelty: {qr['novelty']:.3f}")
        print(f"Vote: Long={qr['vote_long']:.3f}, Short={qr['vote_short']:.3f}, Bias={qr['vote_bias']:.3f}")
        print(f"Top State: {qr['top_state']} ({qr['top_count']} counts)")

    # Test with bearish signal + high volatility (should trigger aromatization)
    print("\n\n" + "="*70)
    print("TEST 2: High Volatility / Drawdown (should aromatize)")
    print("="*70)

    # Choppy market
    choppy_prices = base_price + np.random.randn(100) * 10
    synthetic_data_choppy = {
        'close': choppy_prices.tolist(),
        'volume': np.random.randint(1000, 5000, 100).tolist(),
        'volatility': 3.0,  # High volatility
        'drawdown': 0.12   # 12% drawdown
    }

    result2 = bridge.process_signal(
        market_data=synthetic_data_choppy,
        base_signal=-0.3,
        immune_conflict=0.05
    )

    print(f"\nFINAL DECISION: {result2['action'].upper()}")
    print(f"Strength: {result2['strength']:.3f}")
    print(f"Regime: {result2['regime']} (aromatized: {result2['regime'] != 'aggressive'})")
    print(f"Position Multiplier: {result2['position_multiplier']:.2f}x")
    print(f"Stop Multiplier: {result2['stop_multiplier']:.2f}x")
    print(f"Target Multiplier: {result2['target_multiplier']:.2f}x")

    print("\n--- Aromatase State ---")
    print(f"Aromatization Level: {result2['testosterone_layers']['layer4_exit']['aromatization_level']:.3f}")
    print(f"Exit Strategy: {result2['testosterone_layers']['layer4_exit']['exit_strategy']}")

    print("\n" + "="*70)
    print("TESTOSTERONE-DMT BRIDGE TEST COMPLETE")
    print("="*70)
    print("\nModule is ready for integration into TEQA neural TE system.")
    print("Import with: from testosterone_dmt_bridge import create_bridge")
