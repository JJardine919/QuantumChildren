"""
TEQA BRIDGE v3.1 - Symbol-Aware Quantum Signal Reader for BRAIN Scripts
========================================================================
Reads te_quantum_signal.json (or per-symbol te_quantum_signal_BTCUSD.json)
written by TEQA v3.0/v3.1 and provides a clean interface for BRAIN scripts.

Pipeline: TEQA → JSON → teqa_bridge → BRAIN → Trade

Backward compatible: reads both v2.0 and v3.0 signal formats.
Symbol-aware: per-symbol signal files with fallback to generic file.

Author: DooDoo + Claude
Date: 2026-02-07
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

from config_loader import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class TEQASignal:
    """Parsed TEQA quantum signal (v2.0 + v3.0 compatible)."""
    # Jardine's Gate fields (v2.0)
    direction: int            # 1=LONG, -1=SHORT, 0=NEUTRAL
    confidence: float         # 0.0-1.0
    entropy_adj: float
    interference: float
    amplitude_sq: float

    # Position sizing (v2.0)
    lot_scale: float          # Multiplier from domestication boost
    amplified: bool           # Whether signal was amplified

    # TE filters (v2.0 — any True = block trade)
    pirna_silenced: bool      # piRNA pathway silencing / neural consensus fail
    shock_active: bool        # Genomic shock (high volatility)
    threshold_mult: float     # Adjusted threshold multiplier / shock score
    ectopic_inversion: bool   # Ectopic recombination / speciation hybrid zone

    # Quantum circuit stats (v2.0)
    novelty: float
    measurement_entropy: float
    n_states: int
    n_active_qubits: int
    vote_long: float
    vote_short: float

    # Meta (v2.0)
    timestamp: datetime
    version: str
    is_stale: bool

    # --- v3.0 Neural-TE fields (default to v2.0-compatible values) ---

    # Neural mosaic
    n_neurons: int = 0
    consensus_direction: int = 0
    consensus_score: float = 0.0
    consensus_pass: bool = True
    vote_counts: Dict = field(default_factory=dict)

    # Genomic shock detail
    shock_score: float = 0.0
    shock_label: str = "UNKNOWN"

    # Speciation
    cross_corr: float = 0.0
    relationship: str = "NO_DONOR"

    # Domestication
    domestication_boost: float = 1.0
    active_tes: list = field(default_factory=list)

    # Gate results (G7-G10)
    gates: Dict = field(default_factory=dict)

    # TE activation counts
    n_active_class_i: int = 0
    n_active_class_ii: int = 0
    n_active_neural: int = 0

    # Symbol
    symbol: str = ""

    # --- v3.1 Evolution fields ---
    evolution_enabled: bool = False
    evolution_generation: int = 0
    evolution_avg_accuracy: float = 0.5
    evolution_best_accuracy: float = 0.5
    evolution_unique_genomes: int = 0
    evolution_speciation_pressure: bool = False

    @property
    def is_v3(self) -> bool:
        """Whether this signal came from TEQA v3.0."""
        return "3.0" in self.version or "NEURAL" in self.version

    @property
    def is_blocked(self) -> bool:
        """Whether TE filters block this signal."""
        if self.is_v3:
            # v3.0: check all 4 new gates
            if self.gates:
                return not all(self.gates.values())
            # Fallback to mapped filter flags
            return self.pirna_silenced or self.shock_active or self.ectopic_inversion
        # v2.0: original 3 filters
        return self.pirna_silenced or self.shock_active or self.ectopic_inversion

    @property
    def blocked_reasons(self) -> list:
        """List of reasons the signal is blocked."""
        reasons = []
        if self.is_v3 and self.gates:
            if not self.gates.get("G7_neural_consensus", True):
                reasons.append("G7:neural_consensus")
            if not self.gates.get("G8_genomic_shock", True):
                reasons.append(f"G8:shock({self.shock_label})")
            if not self.gates.get("G9_speciation", True):
                reasons.append(f"G9:speciation({self.relationship})")
            if not self.gates.get("G10_domestication", True):
                reasons.append("G10:domestication")
        else:
            if self.pirna_silenced:
                reasons.append("piRNA")
            if self.shock_active:
                reasons.append("shock")
            if self.ectopic_inversion:
                reasons.append("ectopic")
        return reasons

    @property
    def direction_str(self) -> str:
        if self.direction == 1:
            return "LONG"
        elif self.direction == -1:
            return "SHORT"
        return "NEUTRAL"


class TEQABridge:
    """
    Reads TEQA quantum signals and integrates with BRAIN trading decisions.
    Compatible with both v2.0 and v3.0 signal formats.
    Supports per-symbol signal files with fallback to generic file.

    Usage:
        bridge = TEQABridge()

        # In your analysis loop (symbol-aware):
        signal = bridge.read_signal(symbol='BTCUSD')
        if signal and not signal.is_stale:
            action, conf, lot_mult, reason = bridge.apply_to_lstm(
                lstm_action='BUY', lstm_confidence=0.65, symbol='BTCUSD'
            )
    """

    def __init__(self, signal_path: str = None, staleness_minutes: float = 5.0):
        if signal_path is None:
            self._signal_dir = Path(__file__).parent
            self.signal_path = self._signal_dir / "te_quantum_signal.json"
        else:
            self.signal_path = Path(signal_path)
            self._signal_dir = self.signal_path.parent
        self.staleness_minutes = staleness_minutes
        # Per-symbol cache: symbol_key -> (signal, mtime)
        # symbol_key is the symbol string, or "__generic__" for the legacy file
        self._signal_cache: Dict[str, Tuple[Optional[TEQASignal], float]] = {}

    def _resolve_signal_path(self, symbol: str = None) -> Tuple[Path, str]:
        """
        Resolve which signal file to read and the cache key.
        If symbol given, try per-symbol file first, fall back to generic.
        Returns (path, cache_key).
        """
        if symbol:
            per_symbol_path = self._signal_dir / f"te_quantum_signal_{symbol}.json"
            if per_symbol_path.exists():
                return per_symbol_path, symbol
            # Fall back to generic file
            return self.signal_path, f"__generic__{symbol}"
        return self.signal_path, "__generic__"

    def read_signal(self, symbol: str = None) -> Optional[TEQASignal]:
        """
        Read and parse the TEQA signal JSON. Returns None if file missing or corrupt.

        Args:
            symbol: Optional symbol to look up. If provided, checks for
                    te_quantum_signal_{symbol}.json first, then falls back
                    to te_quantum_signal.json. When reading the generic file
                    with a symbol requested, verifies the signal's symbol matches.
        """
        file_path, cache_key = self._resolve_signal_path(symbol)

        if not file_path.exists():
            return None

        # Skip re-reading if file hasn't changed
        current_mtime = file_path.stat().st_mtime
        cached = self._signal_cache.get(cache_key)
        if cached is not None:
            cached_signal, cached_mtime = cached
            if current_mtime == cached_mtime and cached_signal is not None:
                # Update staleness check
                cached_signal.is_stale = self._check_stale(cached_signal.timestamp)
                return cached_signal

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # If we fell back to generic file but a specific symbol was requested,
            # check that the signal's symbol matches
            signal_symbol = data.get('symbol', '')
            if symbol and cache_key.startswith("__generic__") and signal_symbol:
                if signal_symbol != symbol:
                    logger.debug(f"[TEQA] Signal symbol mismatch: file has {signal_symbol}, "
                                 f"requested {symbol} -- returning None")
                    return None

            ts = datetime.fromisoformat(data['timestamp'])
            is_stale = self._check_stale(ts)

            jg = data['jardines_gate']
            pos = data['position']
            flt = data['filters']
            q = data['quantum']

            # v3.0 sections (with safe defaults for v2.0 signals)
            neural = data.get('neural', {})
            genomic_shock = data.get('genomic_shock', {})
            speciation = data.get('speciation', {})
            domestication = data.get('domestication', {})
            gates = data.get('gates', {})
            te_summary = data.get('te_summary', {})
            evolution = data.get('evolution', {})

            signal = TEQASignal(
                # v2.0 fields
                direction=jg['direction'],
                confidence=jg['confidence'],
                entropy_adj=jg.get('entropy_adj', 0.0),
                interference=jg.get('interference', 1.0),
                amplitude_sq=jg.get('amplitude_sq', 0.0),
                lot_scale=pos['lot_scale'],
                amplified=pos['amplified'],
                pirna_silenced=flt['pirna_silenced'],
                shock_active=flt['shock_active'],
                threshold_mult=flt['threshold_mult'],
                ectopic_inversion=flt['ectopic_inversion'],
                novelty=q['novelty'],
                measurement_entropy=q['measurement_entropy'],
                n_states=q['n_states'],
                n_active_qubits=q['n_active_qubits'],
                vote_long=q['vote_long'],
                vote_short=q['vote_short'],
                timestamp=ts,
                version=data.get('version', 'unknown'),
                is_stale=is_stale,

                # v3.0 fields
                n_neurons=neural.get('n_neurons', 0),
                consensus_direction=neural.get('consensus_direction', 0),
                consensus_score=neural.get('consensus_score', 0.0),
                consensus_pass=neural.get('consensus_pass', True),
                vote_counts=neural.get('vote_counts', {}),
                shock_score=genomic_shock.get('score', 0.0),
                shock_label=genomic_shock.get('label', 'UNKNOWN'),
                cross_corr=speciation.get('cross_corr', 0.0),
                relationship=speciation.get('relationship', 'NO_DONOR'),
                domestication_boost=domestication.get('boost', 1.0),
                active_tes=domestication.get('active_tes', []),
                gates=gates,
                n_active_class_i=te_summary.get('n_active_class_i', 0),
                n_active_class_ii=te_summary.get('n_active_class_ii', 0),
                n_active_neural=te_summary.get('n_active_neural', 0),
                symbol=signal_symbol,

                # v3.1 evolution fields
                evolution_enabled=evolution.get('enabled', False),
                evolution_generation=evolution.get('generation', 0),
                evolution_avg_accuracy=evolution.get('avg_accuracy', 0.5),
                evolution_best_accuracy=evolution.get('best_accuracy', 0.5),
                evolution_unique_genomes=evolution.get('unique_genomes', 0),
                evolution_speciation_pressure=evolution.get('speciation_pressure', False),
            )

            self._signal_cache[cache_key] = (signal, current_mtime)
            return signal

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[TEQA] Failed to parse signal: {e}")
            return None

    def _check_stale(self, timestamp: datetime) -> bool:
        return (datetime.now() - timestamp) > timedelta(minutes=self.staleness_minutes)

    def apply_to_lstm(self, lstm_action: str, lstm_confidence: float,
                      symbol: str = None) -> Tuple[str, float, float, str]:
        """
        Combine TEQA quantum signal with LSTM prediction.

        Args:
            lstm_action: 'BUY', 'SELL', or 'HOLD' from LSTM
            lstm_confidence: LSTM confidence 0.0-1.0
            symbol: Optional symbol to look up per-symbol signal file

        Returns:
            (action, confidence, lot_multiplier, reason)
        """
        signal = self.read_signal(symbol=symbol)

        if signal is None:
            return lstm_action, lstm_confidence, 1.0, "TEQA: no signal file"

        if signal.is_stale:
            logger.info(f"[TEQA] Signal stale ({signal.timestamp})")
            return lstm_action, lstm_confidence, 1.0, "TEQA: stale"

        # TE filter / gate blocks — hard stop
        if signal.is_blocked:
            reasons = signal.blocked_reasons
            reason = f"TEQA: BLOCKED ({', '.join(reasons)})"
            logger.warning(f"[TEQA] {reason}")
            return 'HOLD', 0.0, 1.0, reason

        teqa_action = 'BUY' if signal.direction == 1 else 'SELL'

        # v3.0: use neural consensus to weight the boost/reduction
        consensus_weight = signal.consensus_score if signal.is_v3 else 1.0

        # Case 1: LSTM has a signal — check concordance
        if lstm_action in ('BUY', 'SELL'):
            if lstm_action == teqa_action:
                # Concordant: boost confidence, apply domestication lot scale
                boost = signal.confidence * 0.30 * consensus_weight
                boosted = min(1.0, lstm_confidence + boost)
                # lot_scale comes from domestication_boost via TEQA engine
                lot_mult = max(signal.lot_scale, signal.domestication_boost)
                reason = (f"TEQA: concordant {teqa_action} "
                          f"({signal.confidence:.1%}, consensus={signal.consensus_score:.1%}, "
                          f"novelty={signal.novelty:.2f}, dom_boost={signal.domestication_boost:.2f})")
                logger.info(f"[TEQA] {reason}")
                return lstm_action, boosted, lot_mult, reason
            else:
                # Discordant: reduce LSTM confidence
                penalty = signal.confidence * 0.3 * consensus_weight
                reduced = lstm_confidence * (1.0 - penalty)
                if reduced < CONFIDENCE_THRESHOLD:
                    reason = f"TEQA: discordant ({lstm_action} vs {teqa_action}), blocked"
                    logger.info(f"[TEQA] {reason}")
                    return 'HOLD', reduced, 1.0, reason
                reason = f"TEQA: discordant ({lstm_action} vs {teqa_action}), reduced"
                logger.info(f"[TEQA] {reason}")
                return lstm_action, reduced, 1.0, reason

        # Case 2: LSTM says HOLD but TEQA has strong signal
        if signal.confidence >= CONFIDENCE_THRESHOLD and not signal.is_blocked:
            lot_mult = max(signal.lot_scale, signal.domestication_boost)
            reason = (f"TEQA: quantum override >> {teqa_action} "
                      f"({signal.confidence:.1%}, shock={signal.shock_label}, "
                      f"dom_boost={signal.domestication_boost:.2f})")
            logger.info(f"[TEQA] {reason}")
            return teqa_action, signal.confidence, lot_mult, reason

        return lstm_action, lstm_confidence, 1.0, "TEQA: below threshold"

    def get_status_line(self, symbol: str = None) -> str:
        """One-line status for dashboard display."""
        signal = self.read_signal(symbol=symbol)
        if signal is None:
            return "[TEQA] No signal"
        if signal.is_stale:
            return f"[TEQA] STALE ({signal.timestamp.strftime('%H:%M:%S')})"
        if signal.is_blocked:
            return f"[TEQA] BLOCKED ({', '.join(signal.blocked_reasons)})"

        if signal.is_v3:
            evo_str = ""
            if signal.evolution_enabled:
                evo_str = (f" | evo=gen{signal.evolution_generation} "
                           f"acc={signal.evolution_avg_accuracy:.0%} "
                           f"uniq={signal.evolution_unique_genomes}")
            return (f"[TEQA v3] {signal.direction_str} {signal.confidence:.1%} "
                    f"| neurons={signal.n_neurons} consensus={signal.consensus_score:.0%} "
                    f"| shock={signal.shock_label} "
                    f"| lot={signal.lot_scale:.1f}x "
                    f"| TEs: I={signal.n_active_class_i} II={signal.n_active_class_ii} N={signal.n_active_neural}"
                    f"{evo_str}")

        return (f"[TEQA] {signal.direction_str} {signal.confidence:.1%} "
                f"| lot={signal.lot_scale:.1f}x | novelty={signal.novelty:.2f} "
                f"| qubits={signal.n_active_qubits}/25 | states={signal.n_states}")
