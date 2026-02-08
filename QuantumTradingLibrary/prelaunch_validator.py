"""
PRE-LAUNCH VALIDATOR
====================
Enforces training and data collection requirements before live trading.

Usage:
    from prelaunch_validator import validate_prelaunch

    # At BRAIN script startup:
    if not validate_prelaunch(symbols=['BTCUSD', 'XAUUSD']):
        sys.exit(1)

Author: DooDoo + Claude
Date: 2026-02-04
"""

import json
import sys
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Import config settings
from config_loader import (
    REQUIRE_TRAINED_EXPERT,
    COLLECTION_SERVER_URL,
    COLLECTION_ENABLED
)

# Paths
BASE_DIR = Path(__file__).parent
EXPERTS_DIR = BASE_DIR / "top_50_experts"
MANIFEST_PATH = EXPERTS_DIR / "top_50_manifest.json"
DATA_DIR = BASE_DIR / "quantum_data"


def check_expert_exists(symbol: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check if a trained expert exists for the given symbol.

    Returns:
        (exists: bool, expert_info: dict or None)
    """
    if not MANIFEST_PATH.exists():
        return False, None

    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)

        # Find expert for this symbol
        for expert in manifest.get('experts', []):
            if expert.get('symbol') == symbol and expert.get('verified', False):
                expert_file = EXPERTS_DIR / expert['filename']
                if expert_file.exists():
                    return True, expert

        return False, None

    except Exception as e:
        print(f"[VALIDATOR] Error reading manifest: {e}")
        return False, None


def check_collection_server() -> Tuple[bool, str]:
    """
    Check if the compression/collection server is reachable.

    Returns:
        (reachable: bool, message: str)
    """
    if not COLLECTION_ENABLED:
        return True, "Collection disabled in config"

    try:
        response = requests.post(f"{COLLECTION_SERVER_URL}/ping", timeout=5)
        if response.status_code == 200:
            return True, "Server online"
        else:
            return False, f"Server returned {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except Exception as e:
        return False, str(e)


def check_data_freshness(max_age_hours: int = 24) -> Tuple[bool, str]:
    """
    Check if we have recent data collection activity.

    Returns:
        (fresh: bool, message: str)
    """
    if not DATA_DIR.exists():
        return False, "No quantum_data directory found"

    # Look for recent files
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    recent_files = []

    for f in DATA_DIR.glob("*.jsonl"):
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime > cutoff:
            recent_files.append(f.name)

    if recent_files:
        return True, f"Found {len(recent_files)} recent data files"
    else:
        return False, f"No data files modified in last {max_age_hours} hours"


def validate_prelaunch(
    symbols: List[str],
    require_server: bool = False,
    require_fresh_data: bool = False,
    bypass: bool = False
) -> bool:
    """
    Main validation function. Call this before starting trading.

    Args:
        symbols: List of symbols to trade (e.g., ['BTCUSD', 'XAUUSD'])
        require_server: If True, fail if collection server is down
        require_fresh_data: If True, fail if no recent data collection
        bypass: If True, only warn but don't block (for testing)

    Returns:
        True if all checks pass (or bypass=True), False otherwise
    """
    print("=" * 60)
    print("  PRE-LAUNCH VALIDATION")
    print("=" * 60)
    print()

    all_passed = True

    # Check 1: Trained experts
    if REQUIRE_TRAINED_EXPERT:
        print("[1] EXPERT VALIDATION")
        for symbol in symbols:
            exists, info = check_expert_exists(symbol)
            if exists:
                print(f"    [OK] {symbol}: Expert found (fitness={info['fitness']:.4f})")
            else:
                print(f"    [FAIL] {symbol}: No trained expert found!")
                all_passed = False
        print()
    else:
        print("[1] EXPERT VALIDATION: Skipped (REQUIRE_TRAINED_EXPERT=False)")
        print()

    # Check 2: Collection server
    print("[2] COLLECTION SERVER")
    server_ok, server_msg = check_collection_server()
    if server_ok:
        print(f"    [OK] {COLLECTION_SERVER_URL}: {server_msg}")
    else:
        print(f"    [WARN] {COLLECTION_SERVER_URL}: {server_msg}")
        if require_server:
            all_passed = False
    print()

    # Check 3: Data freshness
    print("[3] DATA COLLECTION")
    data_ok, data_msg = check_data_freshness()
    if data_ok:
        print(f"    [OK] {data_msg}")
    else:
        print(f"    [WARN] {data_msg}")
        if require_fresh_data:
            all_passed = False
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("  VALIDATION PASSED - Clear for launch")
    else:
        if bypass:
            print("  VALIDATION FAILED - Continuing anyway (bypass=True)")
            all_passed = True
        else:
            print("  VALIDATION FAILED - Trading blocked")
            print()
            print("  To proceed, either:")
            print("    1. Run Master_Train.py to train experts")
            print("    2. Set REQUIRE_TRAINED_EXPERT=false in MASTER_CONFIG.json")
            print("    3. Call validate_prelaunch(..., bypass=True)")
    print("=" * 60)
    print()

    return all_passed


def get_experts_for_symbols(symbols: List[str]) -> Dict[str, Dict]:
    """
    Get all available experts for the given symbols.

    Returns:
        Dict mapping symbol -> best expert info
    """
    if not MANIFEST_PATH.exists():
        return {}

    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)

        result = {}
        for symbol in symbols:
            # Find best expert for this symbol (lowest rank = best)
            best = None
            for expert in manifest.get('experts', []):
                if expert.get('symbol') == symbol and expert.get('verified', False):
                    if best is None or expert['rank'] < best['rank']:
                        best = expert

            if best:
                result[symbol] = best

        return result

    except Exception:
        return {}


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-launch validation for trading")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSD', 'XAUUSD', 'ETHUSD'],
                        help='Symbols to validate (default: BTCUSD XAUUSD ETHUSD)')
    parser.add_argument('--require-server', action='store_true',
                        help='Fail if collection server is down')
    parser.add_argument('--require-data', action='store_true',
                        help='Fail if no recent data collection')
    parser.add_argument('--bypass', action='store_true',
                        help='Warn but do not block on failures')

    args = parser.parse_args()

    success = validate_prelaunch(
        symbols=args.symbols,
        require_server=args.require_server,
        require_fresh_data=args.require_data,
        bypass=args.bypass
    )

    sys.exit(0 if success else 1)
