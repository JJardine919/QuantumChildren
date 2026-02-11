"""
QUANTUM STATE DETECTION — PATTERN ENERGY DISPLAY
=================================================
Terminal visualization of QuantumEncoder output.

Renders quantum state confidence from dominant_state_prob,
entropy, and coherence_score as a live-readable display.

Usage:
    from quantum_cascade_core import QuantumEncoder, QuantumFeatures
    from quantum_state_display import QuantumStateDisplay

    encoder = QuantumEncoder()
    features = encoder.encode_and_measure(market_data)
    display = QuantumStateDisplay()
    display.render(features)
"""

import numpy as np
import sys
import os
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    os.system('')  # Enable ANSI escape sequences on Windows 10+
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Import QuantumFeatures type
try:
    from quantum_cascade_core import QuantumFeatures, MarketRegime, SignalDirection
except ImportError:
    QuantumFeatures = None
    MarketRegime = None
    SignalDirection = None


# ============================================================================
# ANSI COLOR CODES
# ============================================================================

class _C:
    """ANSI escape codes for terminal colors."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE   = "\033[44m"
    BG_CYAN   = "\033[46m"


# ============================================================================
# DISPLAY MODULE
# ============================================================================

class QuantumStateDisplay:
    """
    Renders quantum state detection confidence in the terminal.

    Pattern Energy = weighted composite of:
      - Dominant state probability (40%)
      - Inverted entropy (35%)
      - Coherence score (25%)
    """

    BAR_WIDTH = 40
    PANEL_WIDTH = 64

    # Thresholds for color grading
    HIGH_THRESHOLD = 0.75
    MED_THRESHOLD = 0.50

    def __init__(self, max_entropy: float = 8.0):
        self.max_entropy = max_entropy

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def compute_pattern_energy(self, qf) -> float:
        """
        Compute Pattern Energy — the composite quantum state detection score.

        Weighted blend:
          40% dominant_state_prob  (how concentrated the quantum state is)
          35% inverted entropy     (lower entropy = more detectable pattern)
          25% coherence_score      (structural order in the distribution)
        """
        inv_entropy = max(0.0, 1.0 - (qf.entropy / self.max_entropy))
        energy = (
            0.40 * min(1.0, qf.dominant_state_prob * 4.0) +
            0.35 * inv_entropy +
            0.25 * qf.coherence_score
        )
        return min(1.0, max(0.0, energy))

    def render(
        self,
        qf,
        symbol: str = "",
        direction: str = "",
        confidence: float = 0.0,
        probabilities: Optional[np.ndarray] = None,
    ) -> str:
        """
        Render the full quantum state display and print to terminal.

        Args:
            qf: QuantumFeatures instance from QuantumEncoder.encode_and_measure()
            symbol: Trading symbol (optional, for header)
            direction: Signal direction string (optional)
            confidence: Signal confidence 0-100 (optional)
            probabilities: Raw state probability array for distribution chart

        Returns:
            The rendered string (also printed to stdout).
        """
        energy = self.compute_pattern_energy(qf)
        inv_entropy = max(0.0, 1.0 - (qf.entropy / self.max_entropy))
        regime = qf.regime.value if hasattr(qf.regime, 'value') else str(qf.regime)

        lines = []
        lines.append(self._top_border())
        lines.append(self._header(symbol, energy))
        lines.append(self._separator())

        # Main energy gauge
        lines.append(self._energy_gauge(energy))
        lines.append(self._separator())

        # Component bars
        lines.append(self._component_bar("DOMINANT STATE",   qf.dominant_state_prob, f"{qf.dominant_state_prob*100:.1f}%"))
        lines.append(self._component_bar("ENTROPY (inv)",    inv_entropy,            f"{qf.entropy:.2f}/{self.max_entropy:.1f}"))
        lines.append(self._component_bar("COHERENCE",        qf.coherence_score,     f"{qf.coherence_score*100:.1f}%"))
        lines.append(self._component_bar("PREDICTABILITY",   qf.predictability_score / 100.0, f"{qf.predictability_score:.1f}/100"))
        lines.append(self._empty_row())
        lines.append(self._separator())

        # State distribution mini-chart
        if probabilities is not None and len(probabilities) > 0:
            lines.extend(self._state_distribution(probabilities))
            lines.append(self._separator())

        # Info footer
        lines.append(self._info_row(regime, qf.significant_states, qf.variance, direction))
        lines.append(self._bottom_border())

        output = "\n".join(lines)
        print(output)
        return output

    def render_compact(self, qf, symbol: str = "") -> str:
        """Single-line compact display for logging."""
        energy = self.compute_pattern_energy(qf)
        regime = qf.regime.value if hasattr(qf.regime, 'value') else str(qf.regime)
        color = self._value_color(energy)

        bar = self._bar_chars(energy, 20)
        line = (
            f"{_C.CYAN}{symbol:>10}{_C.RESET} "
            f"{color}{bar}{_C.RESET} "
            f"{color}{energy*100:5.1f}%{_C.RESET}  "
            f"dom={qf.dominant_state_prob*100:4.1f}%  "
            f"ent={qf.entropy:4.2f}  "
            f"coh={qf.coherence_score*100:4.1f}%  "
            f"{_C.YELLOW}{regime}{_C.RESET}"
        )
        print(line)
        return line

    # ------------------------------------------------------------------
    # RENDERING HELPERS
    # ------------------------------------------------------------------

    def _value_color(self, value: float) -> str:
        if value >= self.HIGH_THRESHOLD:
            return _C.GREEN
        elif value >= self.MED_THRESHOLD:
            return _C.YELLOW
        else:
            return _C.RED

    def _bar_chars(self, ratio: float, width: int) -> str:
        filled = int(ratio * width)
        empty = width - filled
        return "\u2588" * filled + "\u2591" * empty

    def _pad_row(self, content: str, raw_len: int) -> str:
        """Pad content to fill panel width inside box borders."""
        padding = self.PANEL_WIDTH - 4 - raw_len
        if padding < 0:
            padding = 0
        return f"\u2551 {content}{' ' * padding} \u2551"

    def _top_border(self) -> str:
        return f"{_C.CYAN}\u2554{'=' * (self.PANEL_WIDTH - 2)}\u2557{_C.RESET}"

    def _bottom_border(self) -> str:
        return f"{_C.CYAN}\u255a{'=' * (self.PANEL_WIDTH - 2)}\u255d{_C.RESET}"

    def _separator(self) -> str:
        return f"{_C.CYAN}\u2560{'\u2550' * (self.PANEL_WIDTH - 2)}\u2563{_C.RESET}"

    def _empty_row(self) -> str:
        return f"{_C.CYAN}\u2551{' ' * (self.PANEL_WIDTH - 2)}\u2551{_C.RESET}"

    def _header(self, symbol: str, energy: float) -> str:
        title = "QUANTUM STATE DETECTION"
        if symbol:
            title += f"  [{symbol}]"
        # Center title
        inner = self.PANEL_WIDTH - 4
        centered = title.center(inner)
        return f"{_C.CYAN}\u2551 {_C.BOLD}{_C.WHITE}{centered}{_C.RESET}{_C.CYAN} \u2551{_C.RESET}"

    def _energy_gauge(self, energy: float) -> str:
        color = self._value_color(energy)
        bar = self._bar_chars(energy, self.BAR_WIDTH)
        pct = f"{energy * 100:.1f}%"
        label = "PATTERN ENERGY"

        # Build the visual line
        row_content = f"  {color}{bar}{_C.RESET}  {_C.BOLD}{color}{pct:>6}{_C.RESET}"
        # Calculate raw character count (without ANSI)
        raw_len = 2 + self.BAR_WIDTH + 2 + 6
        padding = self.PANEL_WIDTH - 4 - raw_len
        if padding < 0:
            padding = 0

        line1 = f"{_C.CYAN}\u2551 {row_content}{' ' * padding} {_C.CYAN}\u2551{_C.RESET}"

        # Label line
        label_centered = label.center(self.BAR_WIDTH + 8)
        label_raw_len = len(label_centered)
        label_padding = self.PANEL_WIDTH - 4 - label_raw_len
        if label_padding < 0:
            label_padding = 0
        line2 = f"{_C.CYAN}\u2551 {_C.DIM}{label_centered}{_C.RESET}{' ' * label_padding} {_C.CYAN}\u2551{_C.RESET}"

        return line1 + "\n" + line2

    def _component_bar(self, label: str, value: float, suffix: str) -> str:
        color = self._value_color(value)
        bar_width = 24
        bar = self._bar_chars(value, bar_width)

        # Fixed-width layout: "  LABEL              BAR  SUFFIX"
        label_col = f"  {label:<18}"
        bar_col = f"{color}{bar}{_C.RESET}"
        suffix_col = f"  {suffix:>10}"

        raw_len = 2 + 18 + bar_width + 2 + 10
        padding = self.PANEL_WIDTH - 4 - raw_len
        if padding < 0:
            padding = 0

        return f"{_C.CYAN}\u2551 {label_col}{bar_col}{suffix_col}{' ' * padding} {_C.CYAN}\u2551{_C.RESET}"

    def _state_distribution(self, probs: np.ndarray) -> List[str]:
        """Render a mini histogram of the top quantum states."""
        lines = []

        # Title
        title = "STATE PROBABILITY DISTRIBUTION"
        inner = self.PANEL_WIDTH - 4
        centered = title.center(inner)
        lines.append(f"{_C.CYAN}\u2551 {_C.DIM}{centered}{_C.RESET}{_C.CYAN} \u2551{_C.RESET}")

        # Get top 8 states by probability
        top_indices = np.argsort(probs)[-8:][::-1]
        max_prob = probs[top_indices[0]] if len(top_indices) > 0 else 1.0

        chart_width = 30
        for idx in top_indices:
            prob = probs[idx]
            if prob < 0.005:
                continue
            state_label = f"|{idx:03d}>"
            bar_len = int((prob / (max_prob + 1e-10)) * chart_width)
            bar = "\u2588" * bar_len

            color = _C.GREEN if prob > 0.10 else (_C.YELLOW if prob > 0.03 else _C.DIM)
            pct = f"{prob*100:5.1f}%"

            row = f"  {_C.CYAN}{state_label}{_C.RESET} {color}{bar:<{chart_width}}{_C.RESET} {pct}"
            raw_len = 2 + 5 + 1 + chart_width + 1 + 6
            padding = self.PANEL_WIDTH - 4 - raw_len
            if padding < 0:
                padding = 0
            lines.append(f"{_C.CYAN}\u2551 {row}{' ' * padding} {_C.CYAN}\u2551{_C.RESET}")

        return lines

    def _info_row(self, regime: str, sig_states: int, variance: float, direction: str) -> str:
        # Regime color
        regime_colors = {
            "TRENDING": _C.GREEN,
            "VOLATILE": _C.YELLOW,
            "CHOPPY": _C.RED,
            "CONSOLIDATING": _C.BLUE,
        }
        rc = regime_colors.get(regime, _C.WHITE)

        # Direction arrow
        dir_map = {"UP": f"{_C.GREEN}\u2191 UP{_C.RESET}", "DOWN": f"{_C.RED}\u2193 DOWN{_C.RESET}", "NEUTRAL": f"{_C.DIM}\u2194 NEUTRAL{_C.RESET}"}
        dir_str = dir_map.get(direction, "")

        left = f"  REGIME: {rc}{regime}{_C.RESET}"
        right = f"STATES: {sig_states}  VAR: {variance:.5f}"
        if direction:
            right += f"  {dir_str}"

        # Raw lengths (approximate without ANSI)
        raw_left = 2 + 8 + len(regime)
        raw_right = 8 + len(str(sig_states)) + 6 + 7
        if direction:
            raw_right += 2 + len(direction) + 2

        gap = self.PANEL_WIDTH - 4 - raw_left - raw_right
        if gap < 2:
            gap = 2

        return f"{_C.CYAN}\u2551 {left}{' ' * gap}{right} {_C.CYAN}\u2551{_C.RESET}"


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    from quantum_cascade_core import QuantumEncoder

    print(f"\n{_C.BOLD}{_C.CYAN}QUANTUM STATE DISPLAY MODULE — TEST{_C.RESET}\n")

    encoder = QuantumEncoder(n_qubits=8, n_shots=2048)
    display = QuantumStateDisplay()

    # Test 1: Trending market (low entropy, high dominant state)
    np.random.seed(42)
    trending_features = np.array([25.0, 0.5, 0.02, 1.2, 15.0, 22.0, 0.3, 102.0])
    features = encoder.encode_and_measure(trending_features)

    # Get raw probabilities for distribution chart
    angles = encoder._normalize_features(trending_features)
    probs = encoder._simulate_circuit(angles)

    print(f"{_C.BOLD}Test 1: Trending Market Data{_C.RESET}")
    display.render(features, symbol="BTCUSD", direction="UP", probabilities=probs)

    energy = display.compute_pattern_energy(features)
    print(f"\n  Pattern Energy: {energy*100:.1f}%")
    print(f"  Regime: {features.regime.value}")
    print(f"  Predictability: {features.predictability_score:.1f}/100\n")

    # Test 2: Choppy market (high entropy)
    choppy_features = np.array([50.0, -0.01, 0.08, 0.9, 50.0, 55.0, -0.01, 99.5])
    features2 = encoder.encode_and_measure(choppy_features)
    angles2 = encoder._normalize_features(choppy_features)
    probs2 = encoder._simulate_circuit(angles2)

    print(f"{_C.BOLD}Test 2: Choppy Market Data{_C.RESET}")
    display.render(features2, symbol="EURUSD", direction="NEUTRAL", probabilities=probs2)

    # Test 3: Compact display
    print(f"\n{_C.BOLD}Compact Display Mode:{_C.RESET}")
    display.render_compact(features, symbol="BTCUSD")
    display.render_compact(features2, symbol="EURUSD")

    print(f"\n{_C.GREEN}All tests passed.{_C.RESET}")
