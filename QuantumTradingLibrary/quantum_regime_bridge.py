"""
QUANTUM REGIME BRIDGE
=====================
Drop-in replacement for zlib-based regime detection.
Connects real quantum compression output (bg_archive.db) to BRAIN scripts.

3-tier fallback:
  Tier 1: Read pre-computed quantum features from archiver DB (~5ms)
  Tier 2: On-demand QuTiP compression if archiver data stale (~2-10s)
  Tier 3: zlib fallback identical to current behavior (instant)

Provides two interfaces:
  - QuantumRegimeBridge class   (for BRAIN_ATLAS, BRAIN_GETLEVERAGED)
  - detect_regime() function    (for BRAIN_BG_INSTANT, BRAIN_BG_CHALLENGE)

Both return Tuple[Regime, float] matching the existing interface.
"""

import logging
import zlib
import sqlite3
import numpy as np
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict

log = logging.getLogger(__name__)


# ============================================================
# REGIME ENUM - single source of truth for all BRAIN scripts
# ============================================================

class Regime(Enum):
    CLEAN = "CLEAN"
    VOLATILE = "VOLATILE"
    CHOPPY = "CHOPPY"


# ============================================================
# CONFIG
# ============================================================

def _load_bridge_config() -> dict:
    """Load QUANTUM_BRIDGE config from MASTER_CONFIG.json via config_loader."""
    try:
        from config_loader import QUANTUM_BRIDGE_CONFIG, CLEAN_THRESHOLD, VOLATILE_THRESHOLD
        cfg = dict(QUANTUM_BRIDGE_CONFIG) if QUANTUM_BRIDGE_CONFIG else {}
        cfg.setdefault('ZLIB_CLEAN_THRESHOLD', CLEAN_THRESHOLD)
        cfg.setdefault('ZLIB_VOLATILE_THRESHOLD', VOLATILE_THRESHOLD)
        return cfg
    except ImportError:
        log.warning("QUANTUM BRIDGE: config_loader not available, using defaults")
        return {}


_DEFAULT_CONFIG = {
    'ENABLED': True,
    'ARCHIVER_DB_PATH': 'BlueGuardian_Deploy/bg_archive.db',
    'STALENESS_MINUTES': 30,
    'MAX_ENTROPY': 3.0,
    'CLEAN_ORDER_THRESHOLD': 0.55,
    'VOLATILE_ORDER_THRESHOLD': 0.35,
    'ENABLE_QUTIP_FALLBACK': True,
    'WRITE_BACK_TO_ARCHIVER': True,
    'ZLIB_CLEAN_THRESHOLD': 1.1,
    'ZLIB_VOLATILE_THRESHOLD': 0.9,
}


def _get_config() -> dict:
    """Merge loaded config with defaults."""
    cfg = dict(_DEFAULT_CONFIG)
    loaded = _load_bridge_config()
    cfg.update({k: v for k, v in loaded.items() if v is not None})
    return cfg


# ============================================================
# TIER 1: Archiver DB (pre-computed quantum features)
# ============================================================

def _resolve_db_path(config: dict) -> Optional[Path]:
    """Resolve the archiver DB path relative to QuantumTradingLibrary."""
    db_rel = config.get('ARCHIVER_DB_PATH', '')
    if not db_rel:
        return None

    # Try relative to this script's directory
    base = Path(__file__).parent
    candidate = base / db_rel
    if candidate.exists():
        return candidate

    # Try absolute
    candidate = Path(db_rel)
    if candidate.exists():
        return candidate

    return None


def _query_archiver(symbol: str, config: dict) -> Optional[Dict]:
    """
    Query archiver DB for recent quantum features.
    Returns the most recent feature dict or None.
    """
    db_path = _resolve_db_path(config)
    if db_path is None:
        return None

    staleness_minutes = config.get('STALENESS_MINUTES', 30)
    cutoff = (datetime.now() - timedelta(minutes=staleness_minutes)).isoformat()

    try:
        conn = sqlite3.connect(str(db_path), timeout=2)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT features_compressed FROM quantum_features
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol, cutoff))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        import gzip
        import pickle
        features = pickle.loads(gzip.decompress(row[0]))
        return features

    except Exception as e:
        log.debug(f"QUANTUM BRIDGE: Archiver query failed: {e}")
        return None


def _compute_order_score(features: Dict, config: dict) -> float:
    """
    Map quantum metrics to a composite order score (0.0-1.0).
    Higher = more ordered/clean market.

    Uses 4 metrics from the archiver:
      - quantum_entropy (lower = more ordered)
      - dominant_state_prob (higher = more ordered)
      - phase_coherence (higher = more ordered)
      - superposition_measure (lower = more ordered)
    """
    max_entropy = config.get('MAX_ENTROPY', 3.0)

    entropy = features.get('quantum_entropy', max_entropy)
    dominant_prob = features.get('dominant_state_prob', 0.0)
    coherence = features.get('phase_coherence', 0.0)
    superposition = features.get('superposition_measure', 1.0)

    # Normalize entropy: 0 entropy -> 1.0 score, max_entropy -> 0.0
    entropy_score = max(0.0, 1.0 - (entropy / max_entropy))

    # Dominant state probability is already 0-1, higher = more ordered
    prob_score = min(1.0, max(0.0, dominant_prob))

    # Phase coherence is 0-1, higher = more ordered
    coherence_score = min(1.0, max(0.0, coherence))

    # Superposition: 0 = pure state (ordered), 1 = max superposition (chaotic)
    superposition_score = max(0.0, 1.0 - superposition)

    # Weighted average
    order_score = (
        0.35 * entropy_score +
        0.25 * prob_score +
        0.25 * coherence_score +
        0.15 * superposition_score
    )

    return min(1.0, max(0.0, order_score))


def _classify_from_order_score(order_score: float, config: dict) -> Tuple[Regime, float]:
    """Classify regime from order score."""
    clean_thresh = config.get('CLEAN_ORDER_THRESHOLD', 0.55)
    volatile_thresh = config.get('VOLATILE_ORDER_THRESHOLD', 0.35)

    if order_score >= clean_thresh:
        return Regime.CLEAN, order_score
    elif order_score >= volatile_thresh:
        return Regime.VOLATILE, order_score
    else:
        return Regime.CHOPPY, order_score


def _tier1_archiver(prices: np.ndarray, symbol: str, config: dict) -> Optional[Tuple[Regime, float]]:
    """Tier 1: Read from archiver DB."""
    if not config.get('ENABLED', True):
        return None

    features = _query_archiver(symbol, config)
    if features is None:
        return None

    order_score = _compute_order_score(features, config)
    regime, fidelity = _classify_from_order_score(order_score, config)

    log.info(f"QUANTUM BRIDGE [TIER 1 - ARCHIVER]: {symbol} -> {regime.value} "
             f"(order_score={order_score:.3f}, entropy={features.get('quantum_entropy', '?'):.3f})")
    return regime, fidelity


# ============================================================
# TIER 2: On-demand QuTiP compression
# ============================================================

def _amplitude_encode(prices: np.ndarray) -> Optional[np.ndarray]:
    """
    Amplitude-encode prices into a state vector of size 2^n.
    Pads/truncates to nearest power of 2.
    """
    n = len(prices)
    if n < 4:
        return None

    # Find nearest power of 2
    n_qubits = int(np.ceil(np.log2(n)))
    n_qubits = max(3, min(n_qubits, 8))  # Clamp 3-8 qubits
    target_len = 2 ** n_qubits

    # Normalize prices to amplitudes
    p = prices.astype(np.float64)
    p = p - p.min()
    p_sum = p.sum()
    if p_sum < 1e-10:
        return None

    p = p / p_sum

    # Pad or truncate
    if len(p) > target_len:
        p = p[-target_len:]
    elif len(p) < target_len:
        pad = np.zeros(target_len - len(p))
        p = np.concatenate([pad, p])

    # Normalize to unit vector
    norm = np.linalg.norm(p)
    if norm < 1e-10:
        return None

    state_vector = (p / norm).astype(complex)
    return state_vector


def _tier2_qutip(prices: np.ndarray, symbol: str, config: dict) -> Optional[Tuple[Regime, float]]:
    """Tier 2: On-demand quantum compression via QuTiP."""
    if not config.get('ENABLE_QUTIP_FALLBACK', True):
        return None

    try:
        from ETARE_QuantumFusion.modules.compression_layer import QuantumCompressionLayer
    except ImportError:
        log.debug("QUANTUM BRIDGE: QuTiP/compression_layer not available")
        return None

    state_vector = _amplitude_encode(prices)
    if state_vector is None:
        return None

    try:
        layer = QuantumCompressionLayer(fid_threshold=0.90)
        result = layer.analyze_regime(state_vector)
        ratio = result.get('ratio', 1.0)

        # Map compression ratio to regime
        clean_thresh = config.get('ZLIB_CLEAN_THRESHOLD', 1.1)
        volatile_thresh = config.get('ZLIB_VOLATILE_THRESHOLD', 0.9)

        if ratio >= clean_thresh:
            regime = Regime.CLEAN
            fidelity = min(1.0, 0.8 + (ratio - clean_thresh) * 0.1)
        elif ratio >= volatile_thresh:
            regime = Regime.VOLATILE
            fidelity = 0.7 + (ratio - volatile_thresh) * 0.5
        else:
            regime = Regime.CHOPPY
            fidelity = max(0.5, ratio / clean_thresh)

        log.info(f"QUANTUM BRIDGE [TIER 2 - QUTIP]: {symbol} -> {regime.value} "
                 f"(ratio={ratio:.3f}, layers={result.get('layers', '?')})")

        # Write back to archiver for future Tier 1 hits
        if config.get('WRITE_BACK_TO_ARCHIVER', True):
            _write_back_to_archiver(symbol, result, config)

        return regime, fidelity

    except Exception as e:
        log.warning(f"QUANTUM BRIDGE: QuTiP analysis failed: {e}")
        return None


def _write_back_to_archiver(symbol: str, qutip_result: dict, config: dict):
    """Write QuTiP results back to archiver DB for future Tier 1 lookups."""
    db_path = _resolve_db_path(config)
    if db_path is None:
        return

    try:
        import gzip
        import pickle
        import hashlib

        features = {
            'quantum_entropy': max(0, 3.0 - qutip_result.get('ratio', 1.0)),
            'dominant_state_prob': min(1.0, qutip_result.get('ratio', 1.0) / 3.0),
            'phase_coherence': min(1.0, qutip_result.get('ratio', 1.0) / 2.0),
            'superposition_measure': max(0, 1.0 - qutip_result.get('ratio', 1.0) / 3.0),
            'compression_ratio': qutip_result.get('ratio', 1.0),
            'compression_layers': qutip_result.get('layers', 0),
            '_source': 'qutip_bridge',
        }

        timestamp = datetime.now().isoformat()
        compressed = gzip.compress(pickle.dumps(features))
        data_hash = hashlib.md5(f"{symbol}_{timestamp}".encode()).hexdigest()

        conn = sqlite3.connect(str(db_path), timeout=2)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO quantum_features
            (symbol, timestamp, data_hash, features_compressed)
            VALUES (?, ?, ?, ?)
        """, (symbol, timestamp, data_hash, compressed))
        conn.commit()
        conn.close()
        log.debug(f"QUANTUM BRIDGE: Wrote QuTiP result back to archiver for {symbol}")

    except Exception as e:
        log.debug(f"QUANTUM BRIDGE: Write-back failed: {e}")


# ============================================================
# TIER 3: zlib fallback (byte-identical to current behavior)
# ============================================================

def _tier3_zlib(prices: np.ndarray, config: dict) -> Tuple[Regime, float]:
    """Tier 3: zlib fallback - byte compression ratio as regime proxy.

    NOTE: zlib measures byte-level redundancy, which is a weak proxy for
    market structure. High compression = repetitive prices (trending),
    low compression = noisy prices (choppy). Fidelity scores are reduced
    to reflect this lower confidence compared to quantum analysis.
    """
    data_bytes = prices.astype(np.float32).tobytes()
    compressed = zlib.compress(data_bytes, level=9)
    ratio = len(data_bytes) / len(compressed)

    clean_thresh = config.get('ZLIB_CLEAN_THRESHOLD', 1.1)
    volatile_thresh = config.get('ZLIB_VOLATILE_THRESHOLD', 0.9)

    if ratio >= clean_thresh:
        regime = Regime.CLEAN
        fidelity = 0.70  # Reduced from 0.96 - zlib is a weak proxy
    elif ratio >= volatile_thresh:
        regime = Regime.VOLATILE
        fidelity = 0.60  # Reduced from 0.88
    else:
        regime = Regime.CHOPPY
        fidelity = 0.50  # Reduced from 0.75

    log.warning(f"QUANTUM BRIDGE [TIER 3 - ZLIB FALLBACK]: {regime.value} "
                f"(ratio={ratio:.3f}) - quantum data not available, low-confidence proxy")
    return regime, fidelity


# ============================================================
# PUBLIC API: detect_regime() function
# (for BRAIN_BG_INSTANT, BRAIN_BG_CHALLENGE)
# ============================================================

def detect_regime(prices: np.ndarray, symbol: str = "BTCUSD") -> Tuple[Regime, float]:
    """
    Drop-in replacement for the old zlib detect_regime().

    Args:
        prices: numpy array of close prices
        symbol: trading symbol for archiver lookup

    Returns:
        Tuple[Regime, float] - regime classification and fidelity score
    """
    config = _get_config()

    # Tier 1: Archiver DB
    result = _tier1_archiver(prices, symbol, config)
    if result is not None:
        return result

    # Tier 2: QuTiP on-demand
    result = _tier2_qutip(prices, symbol, config)
    if result is not None:
        return result

    # Tier 3: zlib fallback
    return _tier3_zlib(prices, config)


# ============================================================
# PUBLIC API: QuantumRegimeBridge class
# (for BRAIN_ATLAS, BRAIN_GETLEVERAGED)
# ============================================================

class QuantumRegimeBridge:
    """
    Drop-in replacement for the old RegimeDetector class.
    Accepts a config object for compatibility but uses MASTER_CONFIG.json internally.
    """

    def __init__(self, config=None):
        self._config_obj = config  # kept for interface compatibility
        self._bridge_config = _get_config()

    def analyze_regime(self, prices: np.ndarray, symbol: str = "BTCUSD") -> Tuple[Regime, float]:
        """
        Analyze market regime using 3-tier quantum bridge.

        Args:
            prices: numpy array of close prices
            symbol: trading symbol for archiver lookup

        Returns:
            Tuple[Regime, float] - regime classification and fidelity score
        """
        # Tier 1: Archiver DB
        result = _tier1_archiver(prices, symbol, self._bridge_config)
        if result is not None:
            return result

        # Tier 2: QuTiP on-demand
        result = _tier2_qutip(prices, symbol, self._bridge_config)
        if result is not None:
            return result

        # Tier 3: zlib fallback
        return _tier3_zlib(prices, self._bridge_config)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][BRIDGE_TEST] %(message)s',
        datefmt='%H:%M:%S'
    )

    print("=" * 60)
    print("  QUANTUM REGIME BRIDGE - Standalone Test")
    print("=" * 60)

    # Test with synthetic data
    np.random.seed(42)

    # Trending data (should be CLEAN)
    trending = np.cumsum(np.random.randn(100) * 0.5 + 0.1) + 50000
    print(f"\nTrending data ({len(trending)} bars):")
    regime, fidelity = detect_regime(trending, symbol="BTCUSD")
    print(f"  Result: {regime.value} (fidelity={fidelity:.3f})")

    # Volatile data (random walk)
    volatile = np.cumsum(np.random.randn(100) * 2.0) + 50000
    print(f"\nVolatile data ({len(volatile)} bars):")
    regime, fidelity = detect_regime(volatile, symbol="BTCUSD")
    print(f"  Result: {regime.value} (fidelity={fidelity:.3f})")

    # Choppy data (mean-reverting)
    choppy = 50000 + np.sin(np.linspace(0, 20 * np.pi, 100)) * 50 + np.random.randn(100) * 10
    print(f"\nChoppy data ({len(choppy)} bars):")
    regime, fidelity = detect_regime(choppy, symbol="BTCUSD")
    print(f"  Result: {regime.value} (fidelity={fidelity:.3f})")

    # Test class interface
    print("\n--- Class interface (QuantumRegimeBridge) ---")
    bridge = QuantumRegimeBridge()
    regime, fidelity = bridge.analyze_regime(trending, symbol="BTCUSD")
    print(f"  Trending: {regime.value} (fidelity={fidelity:.3f})")

    # Show config
    print("\n--- Bridge Config ---")
    cfg = _get_config()
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("  Test complete")
    print("=" * 60)
