"""
CONFIG LOADER - DO NOT MODIFY THIS FILE
========================================
All trading scripts import from here.
Edit MASTER_CONFIG.json to change settings.
"""

import json
from pathlib import Path

# Load config from JSON file
CONFIG_PATH = Path(__file__).parent / "MASTER_CONFIG.json"

def load_config():
    """Load master config - call this to get current settings"""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"MASTER_CONFIG.json not found at {CONFIG_PATH}\n"
            "This file is required. Do not delete it."
        )

    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

# Load once at import time
_CONFIG = load_config()

# ============================================================
# TRADING SETTINGS - FROM MASTER_CONFIG.json
# ============================================================

MAX_LOSS_DOLLARS = _CONFIG['TRADING_SETTINGS']['MAX_LOSS_DOLLARS']
TP_MULTIPLIER = _CONFIG['TRADING_SETTINGS']['TP_MULTIPLIER']
ROLLING_SL_MULTIPLIER = _CONFIG['TRADING_SETTINGS']['ROLLING_SL_MULTIPLIER']
DYNAMIC_TP_PERCENT = _CONFIG['TRADING_SETTINGS']['DYNAMIC_TP_PERCENT']
CONFIDENCE_THRESHOLD = _CONFIG['TRADING_SETTINGS']['CONFIDENCE_THRESHOLD']
AGENT_SL_MIN = _CONFIG['TRADING_SETTINGS']['AGENT_SL_MIN']
AGENT_SL_MAX = _CONFIG['TRADING_SETTINGS']['AGENT_SL_MAX']
CHECK_INTERVAL_SECONDS = _CONFIG['TRADING_SETTINGS']['CHECK_INTERVAL_SECONDS']

# ============================================================
# REGIME DETECTION - FROM MASTER_CONFIG.json
# ============================================================

CLEAN_THRESHOLD = _CONFIG['REGIME_DETECTION']['CLEAN_THRESHOLD']
VOLATILE_THRESHOLD = _CONFIG['REGIME_DETECTION']['VOLATILE_THRESHOLD']

# ============================================================
# ACCOUNTS - FROM MASTER_CONFIG.json
# ============================================================

ACCOUNTS = _CONFIG['ACCOUNTS']

# ============================================================
# COLLECTION SERVER - FROM MASTER_CONFIG.json
# ============================================================

COLLECTION_SERVER_URL = _CONFIG['COLLECTION_SERVER']['url']
COLLECTION_ENABLED = _CONFIG['COLLECTION_SERVER']['enabled']

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_account(key: str) -> dict:
    """Get account config by key (BG_INSTANT, ATLAS, etc.)"""
    return ACCOUNTS.get(key, {})

def get_all_enabled_accounts() -> dict:
    """Get all accounts where enabled=true"""
    return {k: v for k, v in ACCOUNTS.items() if v.get('enabled', False)}

def reload_config():
    """Reload config from file (if changed while running)"""
    global _CONFIG, MAX_LOSS_DOLLARS, TP_MULTIPLIER, CONFIDENCE_THRESHOLD
    global CHECK_INTERVAL_SECONDS, CLEAN_THRESHOLD, VOLATILE_THRESHOLD
    global ACCOUNTS, COLLECTION_SERVER_URL, COLLECTION_ENABLED
    global ROLLING_SL_MULTIPLIER, DYNAMIC_TP_PERCENT, AGENT_SL_MIN, AGENT_SL_MAX

    _CONFIG = load_config()

    MAX_LOSS_DOLLARS = _CONFIG['TRADING_SETTINGS']['MAX_LOSS_DOLLARS']
    TP_MULTIPLIER = _CONFIG['TRADING_SETTINGS']['TP_MULTIPLIER']
    ROLLING_SL_MULTIPLIER = _CONFIG['TRADING_SETTINGS']['ROLLING_SL_MULTIPLIER']
    DYNAMIC_TP_PERCENT = _CONFIG['TRADING_SETTINGS']['DYNAMIC_TP_PERCENT']
    CONFIDENCE_THRESHOLD = _CONFIG['TRADING_SETTINGS']['CONFIDENCE_THRESHOLD']
    AGENT_SL_MIN = _CONFIG['TRADING_SETTINGS']['AGENT_SL_MIN']
    AGENT_SL_MAX = _CONFIG['TRADING_SETTINGS']['AGENT_SL_MAX']
    CHECK_INTERVAL_SECONDS = _CONFIG['TRADING_SETTINGS']['CHECK_INTERVAL_SECONDS']
    CLEAN_THRESHOLD = _CONFIG['REGIME_DETECTION']['CLEAN_THRESHOLD']
    VOLATILE_THRESHOLD = _CONFIG['REGIME_DETECTION']['VOLATILE_THRESHOLD']
    ACCOUNTS = _CONFIG['ACCOUNTS']
    COLLECTION_SERVER_URL = _CONFIG['COLLECTION_SERVER']['url']
    COLLECTION_ENABLED = _CONFIG['COLLECTION_SERVER']['enabled']

# ============================================================
# PRINT CURRENT CONFIG ON IMPORT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MASTER CONFIG - Current Settings")
    print("=" * 60)
    print(f"  SL (MAX_LOSS):        ${MAX_LOSS_DOLLARS}")
    print(f"  TP MULTIPLIER:        {TP_MULTIPLIER}x")
    print(f"  ROLLING SL MULT:      {ROLLING_SL_MULTIPLIER}")
    print(f"  DYNAMIC TP %:         {DYNAMIC_TP_PERCENT}%")
    print(f"  CONFIDENCE THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"  AGENT SL RANGE:       ${AGENT_SL_MIN} - ${AGENT_SL_MAX}")
    print(f"  CHECK INTERVAL:       {CHECK_INTERVAL_SECONDS}s")
    print(f"  CLEAN THRESHOLD:      {CLEAN_THRESHOLD}")
    print(f"  VOLATILE THRESHOLD:   {VOLATILE_THRESHOLD}")
    print()
    print("  ACCOUNTS:")
    for key, acc in ACCOUNTS.items():
        status = "ENABLED" if acc.get('enabled') else "DISABLED"
        print(f"    {key}: {acc['account']} [{status}]")
    print()
    print(f"  COLLECTION SERVER: {COLLECTION_SERVER_URL}")
    print("=" * 60)
