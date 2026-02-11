"""
ENTROPY CONFIDENCE SCORER - Python Port of EntropyGridCore
============================================================
Faithfully replicates GetEntropyConfidence() and CalculateEntropy()
from DEPLOY/EntropyGridCore.mqh.

4-factor composite confidence score:
  1. EMA Alignment (30%) - Price/EMA8/EMA21/EMA200 stacking
  2. EMA Separation Consistency (20%) - Stability of EMA spread
  3. RSI Range (25%) - RSI in neutral zone
  4. ATR Stability (25%) - Volatility consistency

Plus compression boost, then entropy classification.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple

from indicator_engine import IndicatorSnapshot, SymbolIndicators


class EntropyState(Enum):
    LOW = "ENTROPY_LOW"        # Full trading allowed
    MEDIUM = "ENTROPY_MEDIUM"  # Half lot size
    HIGH = "ENTROPY_HIGH"      # No new trades


@dataclass
class ConfidenceResult:
    """Full confidence scoring result."""
    raw_score: float           # Before compression boost
    final_score: float         # After boost, clamped 0-1
    entropy_state: EntropyState
    direction: str             # "BUY", "SELL", or "NONE"

    # Component breakdown
    ema_alignment: float       # 0-0.30
    ema_separation: float      # 0-0.20
    rsi_range: float           # 0-0.25
    atr_stability: float       # 0-0.25


def get_entropy_confidence(
    snapshot: IndicatorSnapshot,
    engine: SymbolIndicators,
    compression_boost: float = 12.0,
) -> ConfidenceResult:
    """
    Compute the 4-factor confidence score.
    Faithfully ports EntropyGridCore.mqh GetEntropyConfidence().

    Args:
        snapshot: Current indicator values
        engine: The SymbolIndicators instance (for separation history)
        compression_boost: Boost in percentage points (default 12 = +0.12)

    Returns:
        ConfidenceResult with score, entropy state, and component breakdown
    """
    price = snapshot.price
    ema8 = snapshot.ema_8
    ema21 = snapshot.ema_21
    ema200 = snapshot.ema_200
    rsi = snapshot.rsi_14
    atr = snapshot.atr_14
    prev_atr = snapshot.prev_atr_14

    # ============================================================
    # Component 1: EMA Alignment (30%)
    # ============================================================
    # Full alignment: price/fast/slow/200 all stacked in one direction
    # Partial: at least fast vs slow aligned
    bullish_full = (price > ema8 > ema21 > ema200)
    bearish_full = (price < ema8 < ema21 < ema200)
    bullish_partial = (ema8 > ema21)
    bearish_partial = (ema8 < ema21)

    if bullish_full or bearish_full:
        ema_alignment = 0.30
    elif bullish_partial or bearish_partial:
        ema_alignment = 0.22
    else:
        ema_alignment = 0.05

    # ============================================================
    # Component 2: EMA Separation Consistency (20%)
    # ============================================================
    # How stable is the EMA8-EMA21 spread between bars?
    current_sep = engine.get_ema_separation()
    prev_sep = engine.get_prev_ema_separation()

    if prev_sep > 0:
        sep_change = abs(current_sep - prev_sep) / prev_sep
    else:
        sep_change = 0.0

    if sep_change < 0.15:
        ema_separation = 0.20
    elif sep_change < 0.30:
        ema_separation = 0.12
    else:
        ema_separation = 0.05

    # ============================================================
    # Component 3: RSI Range (25%)
    # ============================================================
    # Neutral RSI is safer for entries
    if 35 <= rsi <= 65:
        rsi_range = 0.25
    elif 25 <= rsi <= 75:
        rsi_range = 0.18
    else:
        rsi_range = 0.08

    # ============================================================
    # Component 4: ATR Stability (25%)
    # ============================================================
    # Consistent volatility means more predictable moves
    if prev_atr > 0:
        atr_change = abs(atr - prev_atr) / prev_atr
    else:
        atr_change = 0.0

    if atr_change < 0.20:
        atr_stability = 0.25
    elif atr_change < 0.40:
        atr_stability = 0.18
    else:
        atr_stability = 0.08

    # ============================================================
    # Composite Score
    # ============================================================
    raw_score = ema_alignment + ema_separation + rsi_range + atr_stability

    # Compression boost (convert percentage points to decimal)
    boost = compression_boost / 100.0
    final_score = min(raw_score + boost, 1.0)

    # ============================================================
    # Direction
    # ============================================================
    # Use previous bar's EMA values (index [1] in MQL5)
    prev_ema8 = snapshot.prev_ema_8
    prev_ema21 = snapshot.prev_ema_21
    prev_ema200 = snapshot.prev_ema_200

    if price > prev_ema200 and prev_ema8 > prev_ema21:
        direction = "BUY"
    elif price < prev_ema200 and prev_ema8 < prev_ema21:
        direction = "SELL"
    else:
        direction = "NONE"

    return ConfidenceResult(
        raw_score=raw_score,
        final_score=final_score,
        entropy_state=EntropyState.LOW,  # Classified below
        direction=direction,
        ema_alignment=ema_alignment,
        ema_separation=ema_separation,
        rsi_range=rsi_range,
        atr_stability=atr_stability,
    )


def calculate_entropy(
    confidence_result: ConfidenceResult,
    confidence_threshold: float = 0.22,
    compression_boost: float = 12.0,
) -> EntropyState:
    """
    Classify entropy state from confidence score.
    Ports EntropyGridCore.mqh CalculateEntropy().

    Args:
        confidence_result: The result from get_entropy_confidence()
        confidence_threshold: Minimum for ENTROPY_LOW (default 0.22)
        compression_boost: Used to calculate MEDIUM threshold

    Returns:
        EntropyState (LOW, MEDIUM, or HIGH)
    """
    score = confidence_result.final_score

    if score >= confidence_threshold:
        state = EntropyState.LOW
    else:
        # MEDIUM threshold: half of confidence threshold
        # Compression boost is already applied to final_score, so don't subtract it again
        adjusted = confidence_threshold * 0.50
        if score >= adjusted:
            state = EntropyState.MEDIUM
        else:
            state = EntropyState.HIGH

    # Update the result object
    confidence_result.entropy_state = state
    return state


def score_and_classify(
    snapshot: IndicatorSnapshot,
    engine: SymbolIndicators,
    confidence_threshold: float = 0.22,
    compression_boost: float = 12.0,
) -> ConfidenceResult:
    """
    Combined scoring + entropy classification in one call.
    This is the primary entry point for the signal farm.
    """
    result = get_entropy_confidence(snapshot, engine, compression_boost)
    calculate_entropy(result, confidence_threshold, compression_boost)
    return result
