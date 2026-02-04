# ================================================
# archiver_service.py
# Blue Guardian Quantum Feature Archiver
# Stores and compresses quantum features for efficient retrieval
# ================================================
import os
import json
import sqlite3
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import gzip
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ARCHIVER | %(message)s",
    handlers=[
        logging.FileHandler("archiver_service.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

class QuantumFeatureArchiver:
    """
    Archiver for quantum features with compression and efficient retrieval.

    Features:
    - SQLite-based storage with compression
    - LRU cache for fast retrieval
    - Automatic cleanup of old data
    - Thread-safe operations
    - Hash-based deduplication
    """

    def __init__(self, db_path: str = "quantum_archive.db",
                 cache_size: int = 1000,
                 retention_days: int = 90):
        self.db_path = db_path
        self.cache_size = cache_size
        self.retention_days = retention_days
        self.cache: Dict[str, Dict] = {}
        self.cache_order: List[str] = []
        self.lock = threading.Lock()

        self._init_database()
        log.info(f"Quantum Feature Archiver initialized: {db_path}")

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main quantum features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quantum_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data_hash TEXT UNIQUE NOT NULL,
                features_compressed BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                accessed_count INTEGER DEFAULT 0
            )
        """)

        # Index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp
            ON quantum_features(symbol, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_data_hash
            ON quantum_features(data_hash)
        """)

        # Price data archive for training
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                tick_volume INTEGER NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)

        # ETARE population archive
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etare_population (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                individual_id INTEGER NOT NULL,
                fitness REAL NOT NULL,
                weights_compressed BLOB NOT NULL,
                parameters TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(generation, individual_id)
            )
        """)

        # Trade history archive
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL,
                lot_size REAL NOT NULL,
                profit_pips REAL,
                profit_usd REAL,
                quantum_features TEXT,
                etare_decision TEXT,
                llm_override TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _compute_hash(self, data: Any) -> str:
        """Compute hash for data deduplication"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _compress(self, data: Dict) -> bytes:
        """Compress data using gzip + pickle"""
        return gzip.compress(pickle.dumps(data))

    def _decompress(self, compressed: bytes) -> Dict:
        """Decompress data"""
        return pickle.loads(gzip.decompress(compressed))

    def _update_cache(self, key: str, value: Dict):
        """Update LRU cache"""
        with self.lock:
            if key in self.cache:
                self.cache_order.remove(key)
            elif len(self.cache) >= self.cache_size:
                oldest = self.cache_order.pop(0)
                del self.cache[oldest]

            self.cache[key] = value
            self.cache_order.append(key)

    def store_quantum_features(self, symbol: str, timestamp: str,
                               features: Dict) -> bool:
        """
        Store quantum features with compression and deduplication.

        Args:
            symbol: Trading symbol (e.g., "BTCUSD")
            timestamp: ISO format timestamp
            features: Dict of quantum features

        Returns:
            True if stored successfully, False if duplicate
        """
        data_hash = self._compute_hash(features)

        # Check cache first
        cache_key = f"{symbol}_{timestamp}"
        if cache_key in self.cache:
            return True

        compressed = self._compress(features)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO quantum_features
                (symbol, timestamp, data_hash, features_compressed)
                VALUES (?, ?, ?, ?)
            """, (symbol, timestamp, data_hash, compressed))
            conn.commit()

            # Update cache
            self._update_cache(cache_key, features)

            log.debug(f"Stored quantum features: {symbol} @ {timestamp}")
            return True

        except sqlite3.IntegrityError:
            # Duplicate hash - already stored
            return False
        finally:
            conn.close()

    def get_quantum_features(self, symbol: str, timestamp: str) -> Optional[Dict]:
        """
        Retrieve quantum features with caching.

        Args:
            symbol: Trading symbol
            timestamp: ISO format timestamp

        Returns:
            Dict of quantum features or None
        """
        cache_key = f"{symbol}_{timestamp}"

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT features_compressed FROM quantum_features
            WHERE symbol = ? AND timestamp = ?
        """, (symbol, timestamp))

        row = cursor.fetchone()

        if row:
            features = self._decompress(row[0])

            # Update access count
            cursor.execute("""
                UPDATE quantum_features
                SET accessed_count = accessed_count + 1
                WHERE symbol = ? AND timestamp = ?
            """, (symbol, timestamp))
            conn.commit()

            # Update cache
            self._update_cache(cache_key, features)

            conn.close()
            return features

        conn.close()
        return None

    def get_recent_features(self, symbol: str, hours: int = 24) -> List[Dict]:
        """
        Get recent quantum features for a symbol.

        Args:
            symbol: Trading symbol
            hours: Number of hours to look back

        Returns:
            List of quantum feature dicts
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, features_compressed FROM quantum_features
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (symbol, cutoff))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for ts, compressed in rows:
            features = self._decompress(compressed)
            features['_timestamp'] = ts
            results.append(features)

        return results

    def store_price_data(self, symbol: str, timeframe: str,
                         bars: List[Dict]) -> int:
        """
        Archive price data for training and backtesting.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., "M5", "H1")
            bars: List of OHLCV bar dicts

        Returns:
            Number of bars stored
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stored = 0
        for bar in bars:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO price_archive
                    (symbol, timeframe, timestamp, open, high, low, close, tick_volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, timeframe, bar['time'],
                    bar['open'], bar['high'], bar['low'], bar['close'],
                    bar.get('tick_volume', 0)
                ))
                if cursor.rowcount > 0:
                    stored += 1
            except Exception as e:
                log.error(f"Error storing bar: {e}")

        conn.commit()
        conn.close()

        log.info(f"Archived {stored} bars for {symbol} {timeframe}")
        return stored

    def get_price_data(self, symbol: str, timeframe: str,
                       start_ts: int, end_ts: int) -> List[Dict]:
        """
        Retrieve archived price data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_ts: Start timestamp (Unix)
            end_ts: End timestamp (Unix)

        Returns:
            List of OHLCV bar dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, open, high, low, close, tick_volume
            FROM price_archive
            WHERE symbol = ? AND timeframe = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (symbol, timeframe, start_ts, end_ts))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'time': row[0],
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'tick_volume': row[5]
            }
            for row in rows
        ]

    def store_etare_population(self, generation: int,
                               individuals: List[Dict]) -> None:
        """
        Archive ETARE population for recovery and analysis.

        Args:
            generation: Current generation number
            individuals: List of individual dicts with weights and parameters
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for idx, ind in enumerate(individuals):
            weights_compressed = self._compress(ind.get('weights', {}))
            parameters = json.dumps(ind.get('parameters', {}))
            fitness = ind.get('fitness', 0.0)

            cursor.execute("""
                INSERT OR REPLACE INTO etare_population
                (generation, individual_id, fitness, weights_compressed, parameters)
                VALUES (?, ?, ?, ?, ?)
            """, (generation, idx, fitness, weights_compressed, parameters))

        conn.commit()
        conn.close()

        log.info(f"Archived ETARE generation {generation}: {len(individuals)} individuals")

    def get_best_etare_individual(self, generation: Optional[int] = None) -> Optional[Dict]:
        """
        Get the best ETARE individual from archive.

        Args:
            generation: Specific generation or None for best overall

        Returns:
            Individual dict with weights and parameters
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if generation is not None:
            cursor.execute("""
                SELECT weights_compressed, parameters, fitness
                FROM etare_population
                WHERE generation = ?
                ORDER BY fitness DESC
                LIMIT 1
            """, (generation,))
        else:
            cursor.execute("""
                SELECT weights_compressed, parameters, fitness
                FROM etare_population
                ORDER BY fitness DESC
                LIMIT 1
            """)

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'weights': self._decompress(row[0]),
                'parameters': json.loads(row[1]),
                'fitness': row[2]
            }

        return None

    def store_trade(self, trade: Dict) -> int:
        """
        Archive a trade for analysis.

        Args:
            trade: Trade dict with details

        Returns:
            Trade ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trade_archive
            (account_name, symbol, direction, entry_time, exit_time,
             entry_price, exit_price, lot_size, profit_pips, profit_usd,
             quantum_features, etare_decision, llm_override)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.get('account_name', 'unknown'),
            trade.get('symbol', ''),
            trade.get('direction', ''),
            trade.get('entry_time', ''),
            trade.get('exit_time'),
            trade.get('entry_price', 0),
            trade.get('exit_price'),
            trade.get('lot_size', 0),
            trade.get('profit_pips'),
            trade.get('profit_usd'),
            json.dumps(trade.get('quantum_features', {})),
            json.dumps(trade.get('etare_decision', {})),
            trade.get('llm_override')
        ))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()

        log.info(f"Archived trade {trade_id}: {trade.get('symbol')} {trade.get('direction')}")
        return trade_id

    def get_trade_history(self, account_name: str = None,
                          days: int = 30) -> List[Dict]:
        """
        Get trade history for analysis.

        Args:
            account_name: Filter by account or None for all
            days: Number of days to look back

        Returns:
            List of trade dicts
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if account_name:
            cursor.execute("""
                SELECT * FROM trade_archive
                WHERE account_name = ? AND created_at > ?
                ORDER BY created_at DESC
            """, (account_name, cutoff))
        else:
            cursor.execute("""
                SELECT * FROM trade_archive
                WHERE created_at > ?
                ORDER BY created_at DESC
            """, (cutoff,))

        rows = cursor.fetchall()
        conn.close()

        columns = ['id', 'account_name', 'symbol', 'direction', 'entry_time',
                   'exit_time', 'entry_price', 'exit_price', 'lot_size',
                   'profit_pips', 'profit_usd', 'quantum_features',
                   'etare_decision', 'llm_override', 'created_at']

        return [dict(zip(columns, row)) for row in rows]

    def cleanup_old_data(self) -> Dict[str, int]:
        """
        Clean up data older than retention period.

        Returns:
            Dict with counts of deleted records
        """
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete old quantum features
        cursor.execute("""
            DELETE FROM quantum_features WHERE created_at < ?
        """, (cutoff,))
        quantum_deleted = cursor.rowcount

        # Delete old ETARE populations (keep last 100 generations)
        cursor.execute("""
            DELETE FROM etare_population
            WHERE generation < (SELECT MAX(generation) - 100 FROM etare_population)
        """)
        etare_deleted = cursor.rowcount

        # Vacuum to reclaim space
        cursor.execute("VACUUM")

        conn.commit()
        conn.close()

        log.info(f"Cleanup: {quantum_deleted} quantum features, {etare_deleted} ETARE records")

        return {
            'quantum_features_deleted': quantum_deleted,
            'etare_populations_deleted': etare_deleted
        }

    def get_stats(self) -> Dict:
        """Get archive statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM quantum_features")
        stats['quantum_features_count'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM price_archive")
        stats['price_bars_count'] = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(generation) FROM etare_population")
        stats['etare_latest_generation'] = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM trade_archive")
        stats['trades_archived'] = cursor.fetchone()[0]

        # Database file size
        stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)

        # Cache stats
        stats['cache_size'] = len(self.cache)
        stats['cache_max'] = self.cache_size

        conn.close()
        return stats


class ArchiverService:
    """
    Background service that continuously archives data.
    Runs as a daemon thread.
    """

    def __init__(self, archiver: QuantumFeatureArchiver,
                 cleanup_interval_hours: int = 24):
        self.archiver = archiver
        self.cleanup_interval = cleanup_interval_hours * 3600
        self.running = False
        self.thread = None

    def start(self):
        """Start the archiver service"""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        log.info("Archiver service started")

    def stop(self):
        """Stop the archiver service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        log.info("Archiver service stopped")

    def _run(self):
        """Main service loop"""
        last_cleanup = time.time()

        while self.running:
            try:
                # Periodic cleanup
                if time.time() - last_cleanup > self.cleanup_interval:
                    self.archiver.cleanup_old_data()
                    last_cleanup = time.time()

                time.sleep(60)  # Check every minute

            except Exception as e:
                log.error(f"Archiver service error: {e}")
                time.sleep(10)


# Singleton instance
_archiver_instance = None

def get_archiver(db_path: str = "quantum_archive.db") -> QuantumFeatureArchiver:
    """Get or create the singleton archiver instance"""
    global _archiver_instance
    if _archiver_instance is None:
        _archiver_instance = QuantumFeatureArchiver(db_path)
    return _archiver_instance


if __name__ == "__main__":
    # Test the archiver
    archiver = get_archiver("test_archive.db")

    # Test storing quantum features
    test_features = {
        'quantum_entropy': 2.5,
        'dominant_state_prob': 0.18,
        'superposition_measure': 0.4,
        'phase_coherence': 0.72,
        'entanglement_degree': 0.55,
        'quantum_variance': 0.003,
        'num_significant_states': 4.0
    }

    archiver.store_quantum_features("BTCUSD", datetime.now().isoformat(), test_features)

    # Test retrieval
    retrieved = archiver.get_recent_features("BTCUSD", hours=1)
    print(f"Retrieved {len(retrieved)} features")

    # Print stats
    stats = archiver.get_stats()
    print(f"Archive stats: {json.dumps(stats, indent=2)}")
