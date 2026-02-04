"""
QUANTUM CHILDREN - ENTROPY COLLECTOR
=====================================
Collects trading signals and sends to the QuantumChildren network.
This data improves the models for everyone.

Part of the free QuantumChildren trading system.
"""

import json
import os
import uuid
import hashlib
import requests
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# QuantumChildren Collection Server
COLLECTION_SERVER = "http://203.161.61.61:8888/collect"

# Local backup folder
LOCAL_BACKUP = Path("quantum_data/")
LOCAL_BACKUP.mkdir(exist_ok=True)

# Generate unique node ID (persisted locally)
NODE_ID_FILE = LOCAL_BACKUP / ".node_id"
if NODE_ID_FILE.exists():
    NODE_ID = NODE_ID_FILE.read_text().strip()
else:
    NODE_ID = f"QC_{uuid.uuid4().hex[:12].upper()}"
    NODE_ID_FILE.write_text(NODE_ID)

print(f"[QuantumChildren] Node ID: {NODE_ID}")

# ============================================================
# COLLECTION FUNCTIONS
# ============================================================

def collect_signal(signal_data: dict) -> bool:
    """
    Collect a trading signal and send to QuantumChildren.
    Also saves locally as backup.

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
        True if sent successfully, False if saved locally only
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

    # Send to server
    return _send_to_server(signal_data, '/signal')


def collect_outcome(ticket: int, symbol: str, outcome: str, pnl: float,
                    entry_price: float = None, exit_price: float = None) -> bool:
    """
    Collect trade outcome for feedback training.

    Args:
        ticket: Trade ticket number
        symbol: Trading symbol
        outcome: "WIN" / "LOSS" / "BREAKEVEN"
        pnl: Profit/loss amount
        entry_price: Entry price (optional)
        exit_price: Exit price (optional)

    Returns:
        True if sent successfully
    """
    outcome_data = {
        'node_id': NODE_ID,
        'ticket': ticket,
        'symbol': symbol,
        'outcome': outcome,
        'pnl': pnl,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0'
    }

    # Local backup
    _save_local(outcome_data, 'outcomes')

    # Send to server
    return _send_to_server(outcome_data, '/outcome')


def collect_entropy_snapshot(symbol: str, timeframe: str, entropy: float,
                             dominant: float, significant: int, variance: float,
                             regime: str = None, price: float = None) -> bool:
    """
    Collect entropy snapshot for pattern analysis.

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

    # Local backup
    _save_local(snapshot, 'entropy')

    # Send to server
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
    """Send data to collection server"""
    try:
        url = COLLECTION_SERVER.rstrip('/collect') + endpoint
        response = requests.post(
            url,
            json=data,
            timeout=5,
            headers={
                'X-Node-ID': NODE_ID,
                'Content-Type': 'application/json'
            }
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
        return False


def sync_local_data():
    """
    Sync any locally saved data that hasn't been sent.
    Call this periodically or on startup.
    """
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
                    if 'outcome' in data:
                        if _send_to_server(data, '/outcome'):
                            synced += 1
                        else:
                            failed += 1
                    elif 'quantum_entropy' in data and 'direction' not in data:
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
    stats = {'signals': 0, 'outcomes': 0, 'entropy': 0}

    for log_file in LOCAL_BACKUP.glob('*.jsonl'):
        try:
            with open(log_file, 'r') as f:
                count = sum(1 for _ in f)

            if 'signals' in log_file.name:
                stats['signals'] += count
            elif 'outcomes' in log_file.name:
                stats['outcomes'] += count
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
    'collect_outcome',
    'collect_entropy_snapshot',
    'sync_local_data',
    'get_local_stats',
    'NODE_ID'
]


if __name__ == "__main__":
    print(f"QuantumChildren Entropy Collector")
    print(f"Node ID: {NODE_ID}")
    print(f"Server: {COLLECTION_SERVER}")
    print(f"Local backup: {LOCAL_BACKUP.absolute()}")

    stats = get_local_stats()
    print(f"Local data: {stats}")

    # Test connection
    print("\nTesting server connection...")
    test_data = {'test': True, 'node_id': NODE_ID}
    if _send_to_server(test_data, '/ping'):
        print("Server: ONLINE")
    else:
        print("Server: OFFLINE (data will be saved locally)")
