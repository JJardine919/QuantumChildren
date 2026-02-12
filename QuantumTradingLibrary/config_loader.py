"""
CONFIG LOADER - DO NOT MODIFY THIS FILE
========================================
All trading scripts import from here.
Edit MASTER_CONFIG.json to change settings.
Credentials are loaded from .env file via credential_manager.
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

# Import credential manager for secure password access
try:
    from credential_manager import get_credentials, get_password, validate_credentials
    _CREDENTIALS_AVAILABLE = True
except ImportError:
    _CREDENTIALS_AVAILABLE = False

# ============================================================
# TRADING SETTINGS - FROM MASTER_CONFIG.json
# ============================================================

MAX_LOSS_DOLLARS = _CONFIG['TRADING_SETTINGS']['MAX_LOSS_DOLLARS']
INITIAL_SL_DOLLARS = _CONFIG['TRADING_SETTINGS']['INITIAL_SL_DOLLARS']
TP_MULTIPLIER = _CONFIG['TRADING_SETTINGS']['TP_MULTIPLIER']
ROLLING_SL_MULTIPLIER = _CONFIG['TRADING_SETTINGS']['ROLLING_SL_MULTIPLIER']
DYNAMIC_TP_PERCENT = _CONFIG['TRADING_SETTINGS']['DYNAMIC_TP_PERCENT']
SET_DYNAMIC_TP = _CONFIG['TRADING_SETTINGS']['SET_DYNAMIC_TP']
ROLLING_SL_ENABLED = _CONFIG['TRADING_SETTINGS']['ROLLING_SL_ENABLED']
CONFIDENCE_THRESHOLD = _CONFIG['TRADING_SETTINGS']['CONFIDENCE_THRESHOLD']
AGENT_SL_MIN = _CONFIG['TRADING_SETTINGS']['AGENT_SL_MIN']
AGENT_SL_MAX = _CONFIG['TRADING_SETTINGS']['AGENT_SL_MAX']
CHECK_INTERVAL_SECONDS = _CONFIG['TRADING_SETTINGS']['CHECK_INTERVAL_SECONDS']
ATR_MULTIPLIER = _CONFIG['TRADING_SETTINGS'].get('ATR_MULTIPLIER', 0.0438)
REQUIRE_TRAINED_EXPERT = _CONFIG['TRADING_SETTINGS']['REQUIRE_TRAINED_EXPERT']
LSTM_MAX_AGE_DAYS = _CONFIG['TRADING_SETTINGS'].get('LSTM_MAX_AGE_DAYS', 7)

# Per-symbol overrides (e.g., ETHUSD needs $2 risk for broker min lot 0.1)
SYMBOL_OVERRIDES = _CONFIG['TRADING_SETTINGS'].get('SYMBOL_OVERRIDES', {})

def get_symbol_risk(symbol: str) -> float:
    """Get MAX_LOSS_DOLLARS for a specific symbol (respects per-symbol overrides)."""
    override = SYMBOL_OVERRIDES.get(symbol, {})
    return override.get('MAX_LOSS_DOLLARS', MAX_LOSS_DOLLARS)

def get_symbol_initial_sl(symbol: str) -> float:
    """Get INITIAL_SL_DOLLARS for a specific symbol (respects per-symbol overrides)."""
    override = SYMBOL_OVERRIDES.get(symbol, {})
    return override.get('INITIAL_SL_DOLLARS', INITIAL_SL_DOLLARS)

# ============================================================
# DRAWDOWN PROTECTION - FROM MASTER_CONFIG.json
# ============================================================

_DD_PROTECTION = _CONFIG.get('DRAWDOWN_PROTECTION', {})
DD_PROTECTION_ENABLED = _DD_PROTECTION.get('ENABLED', True)
DAILY_DD_LIMIT_DOLLARS = _DD_PROTECTION.get('DAILY_DD_LIMIT_DOLLARS', 175.0)
DD_WARNING_THRESHOLD_PCT = _DD_PROTECTION.get('DD_WARNING_THRESHOLD_PCT', 60)
DD_CRITICAL_THRESHOLD_PCT = _DD_PROTECTION.get('DD_CRITICAL_THRESHOLD_PCT', 80)
DD_WARNING_DOPAMINE_MULT = _DD_PROTECTION.get('WARNING_DOPAMINE_MULT', 0.50)
DD_CRITICAL_DOPAMINE_MULT = _DD_PROTECTION.get('CRITICAL_DOPAMINE_MULT', 0.05)
DD_RESET_HOUR_UTC = _DD_PROTECTION.get('RESET_HOUR_UTC', 0)
DD_INCLUDE_UNREALIZED = _DD_PROTECTION.get('INCLUDE_UNREALIZED_PNL', False)

# ============================================================
# REGIME DETECTION - FROM MASTER_CONFIG.json
# ============================================================

CLEAN_THRESHOLD = _CONFIG['REGIME_DETECTION']['CLEAN_THRESHOLD']
VOLATILE_THRESHOLD = _CONFIG['REGIME_DETECTION']['VOLATILE_THRESHOLD']

_QUANTUM_BRIDGE = _CONFIG.get('REGIME_DETECTION', {}).get('QUANTUM_BRIDGE', {})
QUANTUM_BRIDGE_CONFIG = _QUANTUM_BRIDGE

# ============================================================
# ACCOUNTS - FROM MASTER_CONFIG.json + credential_manager
# ============================================================

ACCOUNTS = _CONFIG['ACCOUNTS']

# Inject passwords from credential manager (if available)
if _CREDENTIALS_AVAILABLE:
    _cred_status = validate_credentials()
    for key in ACCOUNTS:
        if _cred_status.get(key, False):
            ACCOUNTS[key]['password'] = get_password(key)

# ============================================================
# COLLECTION SERVER - FROM MASTER_CONFIG.json
# ============================================================

COLLECTION_SERVER_URL = _CONFIG['COLLECTION_SERVER']['url']
COLLECTION_ENABLED = _CONFIG['COLLECTION_SERVER']['enabled']

# ============================================================
# MUTABLE CONFIG DICT - USE THIS FOR HOT-RELOAD SUPPORT
# ============================================================
# Scripts using `from config_loader import MAX_LOSS_DOLLARS` get a
# snapshot at import time. If reload_config() runs, those copies go stale.
# For values that must stay current after reload, use:
#   import config_loader
#   config_loader.CONFIG["MAX_LOSS_DOLLARS"]
#
CONFIG = {
    "MAX_LOSS_DOLLARS": MAX_LOSS_DOLLARS,
    "INITIAL_SL_DOLLARS": INITIAL_SL_DOLLARS,
    "TP_MULTIPLIER": TP_MULTIPLIER,
    "ROLLING_SL_MULTIPLIER": ROLLING_SL_MULTIPLIER,
    "DYNAMIC_TP_PERCENT": DYNAMIC_TP_PERCENT,
    "SET_DYNAMIC_TP": SET_DYNAMIC_TP,
    "ROLLING_SL_ENABLED": ROLLING_SL_ENABLED,
    "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
    "AGENT_SL_MIN": AGENT_SL_MIN,
    "AGENT_SL_MAX": AGENT_SL_MAX,
    "CHECK_INTERVAL_SECONDS": CHECK_INTERVAL_SECONDS,
    "ATR_MULTIPLIER": ATR_MULTIPLIER,
    "REQUIRE_TRAINED_EXPERT": REQUIRE_TRAINED_EXPERT,
    "LSTM_MAX_AGE_DAYS": LSTM_MAX_AGE_DAYS,
    "CLEAN_THRESHOLD": CLEAN_THRESHOLD,
    "VOLATILE_THRESHOLD": VOLATILE_THRESHOLD,
    "QUANTUM_BRIDGE_CONFIG": QUANTUM_BRIDGE_CONFIG,
    "DD_PROTECTION_ENABLED": DD_PROTECTION_ENABLED,
    "DAILY_DD_LIMIT_DOLLARS": DAILY_DD_LIMIT_DOLLARS,
    "DD_WARNING_THRESHOLD_PCT": DD_WARNING_THRESHOLD_PCT,
    "DD_CRITICAL_THRESHOLD_PCT": DD_CRITICAL_THRESHOLD_PCT,
    "DD_WARNING_DOPAMINE_MULT": DD_WARNING_DOPAMINE_MULT,
    "DD_CRITICAL_DOPAMINE_MULT": DD_CRITICAL_DOPAMINE_MULT,
    "DD_RESET_HOUR_UTC": DD_RESET_HOUR_UTC,
    "DD_INCLUDE_UNREALIZED": DD_INCLUDE_UNREALIZED,
    "ACCOUNTS": ACCOUNTS,
    "COLLECTION_SERVER_URL": COLLECTION_SERVER_URL,
    "COLLECTION_ENABLED": COLLECTION_ENABLED,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_account(key: str) -> dict:
    """Get account config by key (BG_INSTANT, ATLAS, etc.)

    Returns account with password from credential_manager if available.
    """
    account = ACCOUNTS.get(key, {}).copy()

    # Ensure password is loaded from credential manager
    if _CREDENTIALS_AVAILABLE and 'password' not in account:
        try:
            account['password'] = get_password(key)
        except Exception:
            pass  # Password not configured

    return account

def get_all_enabled_accounts() -> dict:
    """Get all accounts where enabled=true"""
    return {k: v for k, v in ACCOUNTS.items() if v.get('enabled', False)}

def reload_config():
    """Reload config from file (if changed while running).

    NOTE: Module-level variables (MAX_LOSS_DOLLARS, etc.) are updated here,
    but scripts that used `from config_loader import MAX_LOSS_DOLLARS` at
    import time will still hold stale copies. For hot-reload support, scripts
    should use `config_loader.CONFIG["MAX_LOSS_DOLLARS"]` instead.
    """
    global _CONFIG, MAX_LOSS_DOLLARS, INITIAL_SL_DOLLARS, TP_MULTIPLIER, CONFIDENCE_THRESHOLD
    global CHECK_INTERVAL_SECONDS, CLEAN_THRESHOLD, VOLATILE_THRESHOLD
    global ACCOUNTS, COLLECTION_SERVER_URL, COLLECTION_ENABLED
    global ROLLING_SL_MULTIPLIER, DYNAMIC_TP_PERCENT, AGENT_SL_MIN, AGENT_SL_MAX
    global SET_DYNAMIC_TP, ROLLING_SL_ENABLED, ATR_MULTIPLIER, REQUIRE_TRAINED_EXPERT
    global QUANTUM_BRIDGE_CONFIG, LSTM_MAX_AGE_DAYS
    global DD_PROTECTION_ENABLED, DAILY_DD_LIMIT_DOLLARS
    global DD_WARNING_THRESHOLD_PCT, DD_CRITICAL_THRESHOLD_PCT
    global DD_WARNING_DOPAMINE_MULT, DD_CRITICAL_DOPAMINE_MULT
    global DD_RESET_HOUR_UTC, DD_INCLUDE_UNREALIZED

    _CONFIG = load_config()

    MAX_LOSS_DOLLARS = _CONFIG['TRADING_SETTINGS']['MAX_LOSS_DOLLARS']
    INITIAL_SL_DOLLARS = _CONFIG['TRADING_SETTINGS']['INITIAL_SL_DOLLARS']
    TP_MULTIPLIER = _CONFIG['TRADING_SETTINGS']['TP_MULTIPLIER']
    ROLLING_SL_MULTIPLIER = _CONFIG['TRADING_SETTINGS']['ROLLING_SL_MULTIPLIER']
    DYNAMIC_TP_PERCENT = _CONFIG['TRADING_SETTINGS']['DYNAMIC_TP_PERCENT']
    SET_DYNAMIC_TP = _CONFIG['TRADING_SETTINGS']['SET_DYNAMIC_TP']
    ROLLING_SL_ENABLED = _CONFIG['TRADING_SETTINGS']['ROLLING_SL_ENABLED']
    CONFIDENCE_THRESHOLD = _CONFIG['TRADING_SETTINGS']['CONFIDENCE_THRESHOLD']
    AGENT_SL_MIN = _CONFIG['TRADING_SETTINGS']['AGENT_SL_MIN']
    AGENT_SL_MAX = _CONFIG['TRADING_SETTINGS']['AGENT_SL_MAX']
    CHECK_INTERVAL_SECONDS = _CONFIG['TRADING_SETTINGS']['CHECK_INTERVAL_SECONDS']
    ATR_MULTIPLIER = _CONFIG['TRADING_SETTINGS'].get('ATR_MULTIPLIER', 0.0438)
    REQUIRE_TRAINED_EXPERT = _CONFIG['TRADING_SETTINGS']['REQUIRE_TRAINED_EXPERT']
    LSTM_MAX_AGE_DAYS = _CONFIG['TRADING_SETTINGS'].get('LSTM_MAX_AGE_DAYS', 7)
    CLEAN_THRESHOLD = _CONFIG['REGIME_DETECTION']['CLEAN_THRESHOLD']
    VOLATILE_THRESHOLD = _CONFIG['REGIME_DETECTION']['VOLATILE_THRESHOLD']
    QUANTUM_BRIDGE_CONFIG = _CONFIG.get('REGIME_DETECTION', {}).get('QUANTUM_BRIDGE', {})
    _dd = _CONFIG.get('DRAWDOWN_PROTECTION', {})
    DD_PROTECTION_ENABLED = _dd.get('ENABLED', True)
    DAILY_DD_LIMIT_DOLLARS = _dd.get('DAILY_DD_LIMIT_DOLLARS', 175.0)
    DD_WARNING_THRESHOLD_PCT = _dd.get('DD_WARNING_THRESHOLD_PCT', 60)
    DD_CRITICAL_THRESHOLD_PCT = _dd.get('DD_CRITICAL_THRESHOLD_PCT', 80)
    DD_WARNING_DOPAMINE_MULT = _dd.get('WARNING_DOPAMINE_MULT', 0.50)
    DD_CRITICAL_DOPAMINE_MULT = _dd.get('CRITICAL_DOPAMINE_MULT', 0.05)
    DD_RESET_HOUR_UTC = _dd.get('RESET_HOUR_UTC', 0)
    DD_INCLUDE_UNREALIZED = _dd.get('INCLUDE_UNREALIZED_PNL', False)
    ACCOUNTS = _CONFIG['ACCOUNTS']
    COLLECTION_SERVER_URL = _CONFIG['COLLECTION_SERVER']['url']
    COLLECTION_ENABLED = _CONFIG['COLLECTION_SERVER']['enabled']

    # Update the mutable CONFIG dict so holders of the reference see new values
    CONFIG.update({
        "MAX_LOSS_DOLLARS": MAX_LOSS_DOLLARS,
        "INITIAL_SL_DOLLARS": INITIAL_SL_DOLLARS,
        "TP_MULTIPLIER": TP_MULTIPLIER,
        "ROLLING_SL_MULTIPLIER": ROLLING_SL_MULTIPLIER,
        "DYNAMIC_TP_PERCENT": DYNAMIC_TP_PERCENT,
        "SET_DYNAMIC_TP": SET_DYNAMIC_TP,
        "ROLLING_SL_ENABLED": ROLLING_SL_ENABLED,
        "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
        "AGENT_SL_MIN": AGENT_SL_MIN,
        "AGENT_SL_MAX": AGENT_SL_MAX,
        "CHECK_INTERVAL_SECONDS": CHECK_INTERVAL_SECONDS,
        "ATR_MULTIPLIER": ATR_MULTIPLIER,
        "REQUIRE_TRAINED_EXPERT": REQUIRE_TRAINED_EXPERT,
        "LSTM_MAX_AGE_DAYS": LSTM_MAX_AGE_DAYS,
        "CLEAN_THRESHOLD": CLEAN_THRESHOLD,
        "VOLATILE_THRESHOLD": VOLATILE_THRESHOLD,
        "QUANTUM_BRIDGE_CONFIG": QUANTUM_BRIDGE_CONFIG,
        "DD_PROTECTION_ENABLED": DD_PROTECTION_ENABLED,
        "DAILY_DD_LIMIT_DOLLARS": DAILY_DD_LIMIT_DOLLARS,
        "DD_WARNING_THRESHOLD_PCT": DD_WARNING_THRESHOLD_PCT,
        "DD_CRITICAL_THRESHOLD_PCT": DD_CRITICAL_THRESHOLD_PCT,
        "DD_WARNING_DOPAMINE_MULT": DD_WARNING_DOPAMINE_MULT,
        "DD_CRITICAL_DOPAMINE_MULT": DD_CRITICAL_DOPAMINE_MULT,
        "DD_RESET_HOUR_UTC": DD_RESET_HOUR_UTC,
        "DD_INCLUDE_UNREALIZED": DD_INCLUDE_UNREALIZED,
        "ACCOUNTS": ACCOUNTS,
        "COLLECTION_SERVER_URL": COLLECTION_SERVER_URL,
        "COLLECTION_ENABLED": COLLECTION_ENABLED,
    })

# ============================================================
# PRINT CURRENT CONFIG ON IMPORT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MASTER CONFIG - Current Settings")
    print("=" * 60)
    print(f"  SL (MAX_LOSS):        ${MAX_LOSS_DOLLARS}")
    print(f"  SL (INITIAL):         ${INITIAL_SL_DOLLARS}")
    print(f"  TP MULTIPLIER:        {TP_MULTIPLIER}x")
    print(f"  ROLLING SL MULT:      {ROLLING_SL_MULTIPLIER}")
    print(f"  ROLLING SL ENABLED:   {ROLLING_SL_ENABLED}")
    print(f"  DYNAMIC TP %:         {DYNAMIC_TP_PERCENT}%")
    print(f"  SET DYNAMIC TP:       {SET_DYNAMIC_TP}")
    print(f"  CONFIDENCE THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print(f"  AGENT SL RANGE:       ${AGENT_SL_MIN} - ${AGENT_SL_MAX}")
    print(f"  ATR MULTIPLIER:       {ATR_MULTIPLIER}")
    print(f"  CHECK INTERVAL:       {CHECK_INTERVAL_SECONDS}s")
    print(f"  REQUIRE TRAINED:      {REQUIRE_TRAINED_EXPERT}")
    print(f"  CLEAN THRESHOLD:      {CLEAN_THRESHOLD}")
    print(f"  VOLATILE THRESHOLD:   {VOLATILE_THRESHOLD}")
    print(f"  QUANTUM BRIDGE:       {'ENABLED' if QUANTUM_BRIDGE_CONFIG.get('ENABLED') else 'DISABLED'}")
    if QUANTUM_BRIDGE_CONFIG:
        print(f"    DB PATH:            {QUANTUM_BRIDGE_CONFIG.get('ARCHIVER_DB_PATH', 'N/A')}")
        print(f"    STALENESS:          {QUANTUM_BRIDGE_CONFIG.get('STALENESS_MINUTES', 'N/A')} min")
        print(f"    CLEAN THRESHOLD:    {QUANTUM_BRIDGE_CONFIG.get('CLEAN_ORDER_THRESHOLD', 'N/A')}")
        print(f"    QUTIP FALLBACK:     {QUANTUM_BRIDGE_CONFIG.get('ENABLE_QUTIP_FALLBACK', False)}")
    print()
    print("  ACCOUNTS:")
    for key, acc in ACCOUNTS.items():
        status = "ENABLED" if acc.get('enabled') else "DISABLED"
        print(f"    {key}: {acc['account']} [{status}]")
    print()
    print(f"  COLLECTION SERVER: {COLLECTION_SERVER_URL}")
    print("=" * 60)
