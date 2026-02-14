"""
QNIF BRIDGE v1.0 - Quantum Neural Immune Fusion Signal Reader for BRAIN Scripts
=================================================================================
Reads qnif_signal_BTCUSD.json (or generic qnif_signal.json) written by QNIF_Master.py
and provides a clean interface for BRAIN scripts with HYBRID VETO authority.

Pipeline: QNIF_Master → JSON → qnif_bridge → BRAIN → Trade

The QNIF signal has VETO AUTHORITY over legacy signals:
  - If QNIF says HOLD, the trade is BLOCKED regardless of TEQA/LSTM opinion
  - If QNIF and legacy agree, confidence is boosted
  - If QNIF disagrees with legacy, QNIF wins (Biological Consensus)

Author: Claude + DooDoo
Date: 2026-02-10
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

from config_loader import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# How old a signal can be before we consider it stale (minutes)
QNIF_STALE_MINUTES = 10


@dataclass
class QNIFSignal:
    """Parsed QNIF signal from qnif_signal_*.json."""
    symbol: str
    timestamp: datetime
    action: str              # 'BUY', 'SELL', or 'HOLD'
    confidence: float        # 0.0-1.0
    lot_multiplier: float    # Position sizing multiplier

    # Compression regime
    compression_ratio: float
    regime: str              # 'TRENDING', 'VOLATILE', 'CLEAN', etc.
    tradeable: bool          # Whether compression says market is tradeable

    # TEQA TE activation
    active_tes: list = field(default_factory=list)
    shock_level: float = 0.0
    shock_label: str = "UNKNOWN"
    te_consensus: float = 0.0   # -1.0=SHORT, 0=NEUTRAL, 1.0=LONG

    # VDJ immune system
    antibody_id: str = ""
    vdj_generation: int = 0
    is_memory: bool = False     # Memory cell = proven pattern

    # Staleness
    is_stale: bool = False

    @property
    def direction(self) -> int:
        """Convert action string to direction int."""
        if self.action == 'BUY':
            return 1
        elif self.action == 'SELL':
            return -1
        return 0

    @property
    def is_hold(self) -> bool:
        """QNIF is saying do not trade."""
        return self.action == 'HOLD' or not self.tradeable

    @property
    def has_immune_memory(self) -> bool:
        """This pattern has been seen and survived before."""
        return self.is_memory and self.antibody_id != ""


class QNIFBridge:
    """
    Reads QNIF quantum-immune signals and integrates with BRAIN trading decisions.
    Implements HYBRID VETO: QNIF has override authority over legacy signals.

    Usage:
        bridge = QNIFBridge()

        # In your analysis loop:
        final_action, final_conf, lot_mult, reason = bridge.apply_hybrid_veto(
            legacy_action='BUY',
            legacy_confidence=0.75,
            symbol='BTCUSD'
        )
    """

    def __init__(self, signal_dir: str = None, stale_minutes: int = QNIF_STALE_MINUTES):
        if signal_dir is None:
            self.signal_dir = Path(__file__).parent
        else:
            self.signal_dir = Path(signal_dir)
        self.stale_minutes = stale_minutes
        self._cache: Dict[str, Tuple[datetime, QNIFSignal]] = {}

    def read_signal(self, symbol: str = None) -> Optional[QNIFSignal]:
        """
        Read the latest QNIF signal from JSON file.
        Tries per-symbol file first, falls back to generic.
        """
        # Try per-symbol file first
        candidates = []
        if symbol:
            candidates.append(self.signal_dir / f"qnif_signal_{symbol}.json")
        candidates.append(self.signal_dir / "qnif_signal.json")

        for filepath in candidates:
            if filepath.exists():
                try:
                    data = json.loads(filepath.read_text())
                    return self._parse_signal(data, symbol)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"[QNIF] Failed to parse {filepath}: {e}")
                    continue

        return None

    def _parse_signal(self, data: dict, expected_symbol: str = None) -> QNIFSignal:
        """Parse JSON data into a QNIFSignal dataclass."""
        # Parse timestamp
        ts_str = data.get('timestamp', '')
        try:
            timestamp = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            timestamp = datetime.now()

        # Check staleness
        age = datetime.now() - timestamp
        is_stale = age > timedelta(minutes=self.stale_minutes)

        # Parse compression block
        compression = data.get('compression', {})
        compression_ratio = compression.get('ratio', 0.0)
        regime = compression.get('regime', 'UNKNOWN')
        tradeable = compression.get('tradeable', False)

        # Parse TEQA block
        teqa = data.get('teqa', {})
        active_tes = teqa.get('active_tes', [])
        shock_level = teqa.get('shock_level', 0.0)
        shock_label = teqa.get('shock_label', 'UNKNOWN')
        te_consensus = teqa.get('consensus', 0.0)

        # Parse VDJ block
        vdj = data.get('vdj', {})
        antibody_id = vdj.get('antibody_id', '')
        vdj_generation = vdj.get('generation', 0)
        is_memory = vdj.get('is_memory', False)

        # Map action
        action = data.get('action', 'HOLD').upper()
        if action not in ('BUY', 'SELL', 'HOLD'):
            action = 'HOLD'

        # Symbol check
        signal_symbol = data.get('symbol', '')
        if expected_symbol and signal_symbol and signal_symbol != expected_symbol:
            logger.debug(f"[QNIF] Symbol mismatch: expected {expected_symbol}, got {signal_symbol}"
                         " -- returning None")
            return None

        return QNIFSignal(
            symbol=signal_symbol,
            timestamp=timestamp,
            action=action,
            confidence=data.get('confidence', 0.0),
            lot_multiplier=data.get('lot_multiplier', 1.0),
            compression_ratio=compression_ratio,
            regime=regime,
            tradeable=tradeable,
            active_tes=active_tes,
            shock_level=shock_level,
            shock_label=shock_label,
            te_consensus=te_consensus,
            antibody_id=antibody_id,
            vdj_generation=vdj_generation,
            is_memory=is_memory,
            is_stale=is_stale,
        )

    def apply_hybrid_veto(self, legacy_action: str, legacy_confidence: float,
                          symbol: str = None) -> Tuple[str, float, float, str]:
        """
        HYBRID VETO: QNIF has override authority over legacy (TEQA/LSTM) signals.

        Rules:
          1. If QNIF signal is missing/stale → pass through legacy signal unchanged
          2. If QNIF says HOLD or market not tradeable → BLOCK (veto)
          3. If QNIF and legacy AGREE → boost confidence + use QNIF lot multiplier
          4. If QNIF and legacy DISAGREE → QNIF wins (Biological Consensus)
          5. If legacy says HOLD but QNIF has strong signal → QNIF can initiate

        Args:
            legacy_action: 'BUY', 'SELL', or 'HOLD' from TEQA/LSTM pipeline
            legacy_confidence: confidence from legacy pipeline (0.0-1.0)
            symbol: trading symbol (e.g., 'BTCUSD')

        Returns:
            (action, confidence, lot_multiplier, reason)
        """
        signal = self.read_signal(symbol=symbol)

        # Rule 1: No signal or stale → pass through
        if signal is None:
            return legacy_action, legacy_confidence, 1.0, "QNIF: no signal"

        if signal.is_stale:
            age_min = (datetime.now() - signal.timestamp).total_seconds() / 60
            return legacy_action, legacy_confidence, 1.0, f"QNIF: stale ({age_min:.0f}m old)"

        # Rule 2: QNIF VETO — if QNIF says HOLD or market not tradeable, BLOCK
        if signal.is_hold:
            reason_parts = []
            if signal.action == 'HOLD':
                reason_parts.append("action=HOLD")
            if not signal.tradeable:
                reason_parts.append(f"regime={signal.regime}")
            if signal.shock_label not in ('NORMAL', 'UNKNOWN'):
                reason_parts.append(f"shock={signal.shock_label}")

            reason = f"QNIF: VETO ({', '.join(reason_parts)})"
            logger.warning(f"[QNIF] {reason}")
            return 'HOLD', 0.0, 1.0, reason

        qnif_action = signal.action  # 'BUY' or 'SELL'

        # Rule 3: Legacy has signal AND agrees with QNIF → boost
        if legacy_action in ('BUY', 'SELL') and legacy_action == qnif_action:
            # Concordant: boost confidence
            boost = signal.confidence * 0.25
            # Extra boost if immune memory recognizes this pattern
            if signal.has_immune_memory:
                boost += 0.10
            boosted = min(1.0, legacy_confidence + boost)

            reason = (f"QNIF: concordant {qnif_action} "
                      f"(conf={signal.confidence:.1%}, regime={signal.regime}, "
                      f"TEs={len(signal.active_tes)}"
                      f"{', MEMORY' if signal.has_immune_memory else ''})")
            logger.info(f"[QNIF] {reason}")
            return legacy_action, boosted, signal.lot_multiplier, reason

        # Rule 4: Legacy has signal but DISAGREES → QNIF wins
        if legacy_action in ('BUY', 'SELL') and legacy_action != qnif_action:
            if signal.confidence >= CONFIDENCE_THRESHOLD:
                reason = (f"QNIF: override {legacy_action}→{qnif_action} "
                          f"(QNIF={signal.confidence:.1%} vs legacy={legacy_confidence:.1%}, "
                          f"consensus={signal.te_consensus:.2f})")
                logger.info(f"[QNIF] {reason}")
                return qnif_action, signal.confidence, signal.lot_multiplier, reason
            else:
                # QNIF not confident enough to override, but blocks legacy
                reason = (f"QNIF: veto {legacy_action} (disagreement, "
                          f"QNIF conf={signal.confidence:.1%} too low to override)")
                logger.info(f"[QNIF] {reason}")
                return 'HOLD', 0.0, 1.0, reason

        # Rule 5: Legacy says HOLD, QNIF has strong signal → QNIF initiates
        if legacy_action == 'HOLD' and signal.confidence >= CONFIDENCE_THRESHOLD:
            reason = (f"QNIF: initiate {qnif_action} "
                      f"(conf={signal.confidence:.1%}, regime={signal.regime}, "
                      f"gen={signal.vdj_generation})")
            logger.info(f"[QNIF] {reason}")
            return qnif_action, signal.confidence, signal.lot_multiplier, reason

        # Default: pass through
        return legacy_action, legacy_confidence, 1.0, "QNIF: below threshold"

    def get_status_line(self, symbol: str = None) -> str:
        """One-line status for dashboard display."""
        signal = self.read_signal(symbol=symbol)
        if signal is None:
            return "[QNIF] No signal"
        if signal.is_stale:
            return f"[QNIF] STALE ({signal.timestamp.strftime('%H:%M:%S')})"
        if signal.is_hold:
            return f"[QNIF] HOLD ({signal.regime}, shock={signal.shock_label})"

        memory_tag = " MEMORY" if signal.has_immune_memory else ""
        return (f"[QNIF] {signal.action} {signal.symbol} "
                f"conf={signal.confidence:.1%} regime={signal.regime} "
                f"TEs={len(signal.active_tes)} gen={signal.vdj_generation}{memory_tag}")
