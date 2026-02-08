"""
QUANTUM CHILDREN - ENTROPY COLLECTOR
=====================================
Collects trading signals and sends to the QuantumChildren network.
This data improves the models for everyone.

DATA COLLECTION NOTICE:
  When enabled, this module sends the following to the collection server:
    - Trading symbol (e.g. BTCUSD)
    - Signal direction (BUY/SELL/HOLD)
    - Confidence score (0.0 - 1.0)
    - Quantum entropy values
    - Current price
    - Timestamp
    - Anonymous node ID (randomly generated, not linked to your identity)

  NO personal data, account numbers, passwords, balances, or P/L are sent.

  Collection is DISABLED by default. To enable:
    1. Set "enabled": true in MASTER_CONFIG.json under COLLECTION_SERVER
    2. Optionally set QC_COLLECTION_API_KEY in .env for authenticated sends

Part of the free QuantumChildren trading system.
See PRIVACY_POLICY.md for full details.
"""

import json
import os
import uuid
import hashlib
import requests
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURATION - loaded from MASTER_CONFIG.json via config_loader
# ============================================================

try:
    from config_loader import COLLECTION_SERVER_URL, COLLECTION_ENABLED
    _config_loaded = True
except ImportError:
    COLLECTION_SERVER_URL = "http://203.161.61.61:8888"
    COLLECTION_ENABLED = False
    _config_loaded = False

# API key for authenticated sends (optional, from .env)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass
_API_KEY = os.environ.get('QC_COLLECTION_API_KEY', '')

# Local backup folder
LOCAL_BACKUP = Path("quantum_data/")
LOCAL_BACKUP.mkdir(exist_ok=True)

# Generate unique node ID (persisted locally, anonymous)
NODE_ID_FILE = LOCAL_BACKUP / ".node_id"
if NODE_ID_FILE.exists():
    NODE_ID = NODE_ID_FILE.read_text().strip()
else:
    NODE_ID = f"QC_{uuid.uuid4().hex[:12].upper()}"
    NODE_ID_FILE.write_text(NODE_ID)

# ============================================================
# FIRST-RUN DISCLOSURE
# ============================================================

_DISCLOSURE_FILE = LOCAL_BACKUP / ".collection_disclosed"

def _print_disclosure():
    """Print data collection notice on first activation."""
    if _DISCLOSURE_FILE.exists():
        return

    print()
    print("=" * 64)
    print("  QUANTUM CHILDREN - DATA COLLECTION NOTICE")
    print("=" * 64)
    print()
    print("  Collection is NOW ENABLED in your MASTER_CONFIG.json.")
    print()
    print("  This system sends the following to the QuantumChildren")
    print("  collection server when you generate trading signals:")
    print()
    print("    - Symbol (e.g. BTCUSD)")
    print("    - Direction (BUY/SELL/HOLD)")
    print("    - Confidence score")
    print("    - Quantum entropy values")
    print("    - Current price")
    print("    - Timestamp")
    print("    - Anonymous node ID")
    print()
    print("  NOT sent: account numbers, passwords, balances, P/L,")
    print("  or any personally identifiable information.")
    print()
    print(f"  Server: {COLLECTION_SERVER_URL}")
    print(f"  Node ID: {NODE_ID}")
    print()
    print("  To disable: set COLLECTION_SERVER.enabled = false")
    print("  in MASTER_CONFIG.json")
    print()
    print("  See PRIVACY_POLICY.md for full details.")
    print("=" * 64)
    print()

    _DISCLOSURE_FILE.write_text(datetime.utcnow().isoformat())


# Print status on import
if COLLECTION_ENABLED:
    _print_disclosure()
    print(f"[QuantumChildren] Collection ENABLED | Node: {NODE_ID}")
else:
    print(f"[QuantumChildren] Collection DISABLED (local backup only) | Node: {NODE_ID}")


# ============================================================
# COLLECTION FUNCTIONS
# ============================================================

def collect_signal(signal_data: dict) -> bool:
    """
    Collect a trading signal. Saves locally always.
    Only sends to server if COLLECTION_ENABLED is true.

    Args:
        signal_data: dict with keys like:
            - symbol: "BTCUSD"
            - direction: "BUY" / "SELL" / "HOLD"
            - confidence: 0.0 - 1.0
            - quantum_entropy: float
            - dominant_state: float
            - price: current price
            - features: list of feature values (optional)

    Returns:
        True if sent successfully (or saved locally if collection disabled)
    """
    # Add metadata
    signal_data['node_id'] = NODE_ID
    signal_data['timestamp'] = datetime.utcnow().isoformat()
    signal_data['version'] = '1.0'

    # Create hash for deduplication
    sig_string = f"{NODE_ID}:{signal_data.get('symbol')}:{signal_data.get('timestamp')}"
    signal_data['sig_hash'] = hashlib.md5(sig_string.encode()).hexdigest()[:16]

    # Local backup (always)
    _save_local(signal_data, 'signals')

    # Send to server only if enabled
    if not COLLECTION_ENABLED:
        return True  # Saved locally, that's fine

    return _send_to_server(signal_data, '/signal')


def collect_entropy_snapshot(symbol: str, timeframe: str, entropy: float,
                             dominant: float, significant: int, variance: float,
                             regime: str = None, price: float = None) -> bool:
    """
    Collect entropy snapshot for pattern analysis.
    Only sends to server if COLLECTION_ENABLED is true.

    Args:
        symbol: Trading symbol
        timeframe: M1, M5, H1, etc.
        entropy: Quantum entropy value
        dominant: Dominant state probability
        significant: Number of significant states
        variance: Quantum variance
        regime: CLEAN/VOLATILE/CHOPPY (optional)
        price: Current price (optional)
    """
    snapshot = {
        'node_id': NODE_ID,
        'symbol': symbol,
        'timeframe': timeframe,
        'quantum_entropy': entropy,
        'dominant_state': dominant,
        'significant_states': significant,
        'quantum_variance': variance,
        'regime': regime,
        'price': price,
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0'
    }

    # Local backup (always)
    _save_local(snapshot, 'entropy')

    # Send to server only if enabled
    if not COLLECTION_ENABLED:
        return True

    return _send_to_server(snapshot, '/entropy')


# ============================================================
# INTERNAL FUNCTIONS
# ============================================================

def _save_local(data: dict, category: str):
    """Save data locally as backup"""
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = LOCAL_BACKUP / f"{category}_{date_str}.jsonl"

    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
    except Exception as e:
        print(f"[QuantumChildren] Local save error: {e}")


def _send_to_server(data: dict, endpoint: str) -> bool:
    """Send data to collection server with optional API key auth"""
    try:
        base_url = COLLECTION_SERVER_URL.rstrip('/')
        url = base_url + endpoint

        headers = {
            'X-Node-ID': NODE_ID,
            'Content-Type': 'application/json'
        }
        if _API_KEY:
            headers['Authorization'] = f'Bearer {_API_KEY}'

        response = requests.post(
            url,
            json=data,
            timeout=5,
            headers=headers
        )

        if response.status_code == 200:
            return True
        else:
            return False

    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        print(f"[QuantumChildren] Send error: {e}")
        return False


def sync_local_data():
    """
    Sync any locally saved data that hasn't been sent.
    Only works if COLLECTION_ENABLED is true.
    Call this periodically or on startup.
    """
    if not COLLECTION_ENABLED:
        print("[QuantumChildren] Collection disabled - skipping sync")
        return 0, 0

    synced = 0
    failed = 0

    for log_file in LOCAL_BACKUP.glob('*.jsonl'):
        synced_file = log_file.with_suffix('.synced')
        if synced_file.exists():
            continue  # Already synced

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'quantum_entropy' in data and 'direction' not in data:
                        if _send_to_server(data, '/entropy'):
                            synced += 1
                        else:
                            failed += 1
                    else:
                        if _send_to_server(data, '/signal'):
                            synced += 1
                        else:
                            failed += 1

            if failed == 0:
                synced_file.touch()  # Mark as synced

        except Exception as e:
            print(f"[QuantumChildren] Sync error: {e}")

    if synced > 0:
        print(f"[QuantumChildren] Synced {synced} records")

    return synced, failed


# ============================================================
# STATS
# ============================================================

def get_local_stats():
    """Get stats on locally collected data"""
    stats = {'signals': 0, 'entropy': 0}

    for log_file in LOCAL_BACKUP.glob('*.jsonl'):
        try:
            with open(log_file, 'r') as f:
                count = sum(1 for _ in f)

            if 'signals' in log_file.name:
                stats['signals'] += count
            elif 'entropy' in log_file.name:
                stats['entropy'] += count
        except:
            pass

    return stats


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'collect_signal',
    'collect_entropy_snapshot',
    'sync_local_data',
    'get_local_stats',
    'NODE_ID',
    'COLLECTION_ENABLED'
]


if __name__ == "__main__":
    print(f"QuantumChildren Entropy Collector")
    print(f"Node ID: {NODE_ID}")
    print(f"Server: {COLLECTION_SERVER_URL}")
    print(f"Collection: {'ENABLED' if COLLECTION_ENABLED else 'DISABLED'}")
    print(f"API Key: {'SET' if _API_KEY else 'NOT SET'}")
    print(f"Local backup: {LOCAL_BACKUP.absolute()}")

    stats = get_local_stats()
    print(f"Local data: {stats}")

    if COLLECTION_ENABLED:
        # Test connection
        print("\nTesting server connection...")
        test_data = {'test': True, 'node_id': NODE_ID}
        if _send_to_server(test_data, '/ping'):
            print("Server: ONLINE")
        else:
            print("Server: OFFLINE (data will be saved locally)")
    else:
        print("\nCollection is disabled. Enable in MASTER_CONFIG.json to send data.")
        print("Local backups are always saved regardless of this setting.")
