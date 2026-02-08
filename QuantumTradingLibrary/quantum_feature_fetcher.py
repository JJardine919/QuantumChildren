"""
QUANTUM FEATURE FETCHER
========================
Loads quantum compression features from bg_archive.db for use in
ETARE training (bulk) and live inference (latest).

DB patterns reused from quantum_regime_bridge.py.

Two public functions:
  - load_quantum_features_bulk()  -- training: aligned (n_bars, 7) array
  - fetch_latest_quantum_features() -- inference: single 7-element array
"""

import logging
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

from quantum_feature_defs import (
    QUANTUM_FEATURE_NAMES,
    QUANTUM_FEATURE_DEFAULTS,
    QUANTUM_FEATURE_COUNT,
)

log = logging.getLogger(__name__)

# Default DB path relative to QuantumTradingLibrary
_DEFAULT_DB_REL = "BlueGuardian_Deploy/bg_archive.db"


def _resolve_db_path(db_path: Optional[str] = None) -> Optional[Path]:
    """Resolve archiver DB path. Tries relative to script dir, then absolute."""
    if db_path:
        p = Path(db_path)
        if p.exists():
            return p

    base = Path(__file__).parent
    candidate = base / _DEFAULT_DB_REL
    if candidate.exists():
        return candidate

    return None


def _defaults_array() -> np.ndarray:
    """Return a 1D array of 7 neutral default values in canonical order."""
    return np.array([QUANTUM_FEATURE_DEFAULTS[n] for n in QUANTUM_FEATURE_NAMES],
                    dtype=np.float64)


def load_quantum_features_bulk(
    symbol: str,
    bar_timestamps: np.ndarray,
    db_path: Optional[str] = None,
) -> np.ndarray:
    """
    Load quantum features aligned to M5 bar timestamps for TRAINING.

    Args:
        symbol: Trading symbol (e.g. "BTCUSD")
        bar_timestamps: 1D array of Unix timestamps (one per M5 bar)
        db_path: Optional explicit path to bg_archive.db

    Returns:
        (n_bars, 7) numpy array of quantum features.
        Falls back to defaults for any bar without a matching DB record.
    """
    n_bars = len(bar_timestamps)
    result = np.tile(_defaults_array(), (n_bars, 1))  # (n_bars, 7)

    resolved = _resolve_db_path(db_path)
    if resolved is None:
        log.warning("QUANTUM FETCHER: bg_archive.db not found, using all defaults")
        return result

    try:
        import gzip
        import pickle

        conn = sqlite3.connect(str(resolved), timeout=5)
        cursor = conn.cursor()

        # Get time range from bar timestamps
        ts_min = float(bar_timestamps.min())
        ts_max = float(bar_timestamps.max())

        # Convert unix timestamps to ISO format for query
        dt_min = datetime.utcfromtimestamp(ts_min) - timedelta(minutes=30)
        dt_max = datetime.utcfromtimestamp(ts_max) + timedelta(minutes=5)

        cursor.execute("""
            SELECT timestamp, features_compressed FROM quantum_features
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (symbol, dt_min.isoformat(), dt_max.isoformat()))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            log.info(f"QUANTUM FETCHER: No records for {symbol} in range, using defaults")
            return result

        # Parse all records: (unix_timestamp, feature_dict)
        records = []
        for ts_str, blob in rows:
            try:
                features = pickle.loads(gzip.decompress(blob))
                dt = datetime.fromisoformat(ts_str)
                unix_ts = dt.timestamp()
                records.append((unix_ts, features))
            except Exception:
                continue

        if not records:
            log.info("QUANTUM FETCHER: All records unparseable, using defaults")
            return result

        # Build sorted arrays for binary search
        record_times = np.array([r[0] for r in records])
        record_features = records

        # For each bar, find nearest quantum record via binary search
        matched = 0
        for i, bar_ts in enumerate(bar_timestamps):
            idx = np.searchsorted(record_times, bar_ts)
            # Check closest of idx-1 and idx
            best_idx = None
            best_dist = float("inf")
            for candidate in [idx - 1, idx]:
                if 0 <= candidate < len(record_times):
                    dist = abs(record_times[candidate] - bar_ts)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = candidate

            # Only use if within 30 minutes
            if best_idx is not None and best_dist <= 1800:
                feat_dict = record_features[best_idx][1]
                for j, name in enumerate(QUANTUM_FEATURE_NAMES):
                    if name in feat_dict:
                        result[i, j] = float(feat_dict[name])
                matched += 1

        log.info(f"QUANTUM FETCHER: Matched {matched}/{n_bars} bars with quantum data "
                 f"({len(records)} DB records)")
        return result

    except Exception as e:
        log.warning(f"QUANTUM FETCHER: Bulk load failed: {e}")
        return result


def fetch_latest_quantum_features(
    symbol: str,
    db_path: Optional[str] = None,
    max_staleness_minutes: int = 30,
) -> np.ndarray:
    """
    Fetch the most recent quantum features for LIVE INFERENCE.

    Args:
        symbol: Trading symbol
        db_path: Optional explicit path to bg_archive.db
        max_staleness_minutes: Max age of record to use

    Returns:
        1D array of 7 quantum feature values (defaults if unavailable)
    """
    resolved = _resolve_db_path(db_path)
    if resolved is None:
        return _defaults_array()

    try:
        import gzip
        import pickle

        cutoff = (datetime.now() - timedelta(minutes=max_staleness_minutes)).isoformat()

        conn = sqlite3.connect(str(resolved), timeout=2)
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
            return _defaults_array()

        features = pickle.loads(gzip.decompress(row[0]))
        result = _defaults_array()
        for j, name in enumerate(QUANTUM_FEATURE_NAMES):
            if name in features:
                result[j] = float(features[name])

        return result

    except Exception as e:
        log.debug(f"QUANTUM FETCHER: Latest fetch failed: {e}")
        return _defaults_array()


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][QFETCH_TEST] %(message)s',
        datefmt='%H:%M:%S'
    )

    print("=" * 60)
    print("  QUANTUM FEATURE FETCHER - Standalone Test")
    print("=" * 60)

    # Test 1: fetch_latest_quantum_features
    print("\n--- Test 1: Latest quantum features for BTCUSD ---")
    latest = fetch_latest_quantum_features("BTCUSD")
    print(f"  Shape: {latest.shape}")
    for name, val in zip(QUANTUM_FEATURE_NAMES, latest):
        default = QUANTUM_FEATURE_DEFAULTS[name]
        is_default = " (DEFAULT)" if abs(val - default) < 1e-6 else ""
        print(f"  {name}: {val:.4f}{is_default}")

    # Test 2: load_quantum_features_bulk with synthetic timestamps
    print("\n--- Test 2: Bulk load for BTCUSD (last 100 M5 bars) ---")
    now = datetime.now().timestamp()
    fake_bars = np.array([now - i * 300 for i in range(100, 0, -1)])
    bulk = load_quantum_features_bulk("BTCUSD", fake_bars)
    print(f"  Shape: {bulk.shape}")
    non_default = 0
    defaults = _defaults_array()
    for i in range(len(bulk)):
        if not np.allclose(bulk[i], defaults):
            non_default += 1
    print(f"  Bars with real data: {non_default}/{len(bulk)}")

    # Test 3: DB not found gracefully
    print("\n--- Test 3: Non-existent DB path ---")
    fallback = fetch_latest_quantum_features("BTCUSD", db_path="/nonexistent/path.db")
    print(f"  Returns defaults: {np.allclose(fallback, defaults)}")

    print("\n" + "=" * 60)
    print("  Test complete")
    print("=" * 60)
