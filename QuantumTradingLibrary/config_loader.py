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
WATCHDOG_LIMIT = _CONFIG['TRADING_SETTINGS'].get('WATCHDOG_LIMIT', 1.00)
LSTM_MAX_AGE_DAYS = _CONFIG['TRADING_SETTINGS'].get('LSTM_MAX_AGE_DAYS', 7)
SOFTMAX_TEMPERATURE = _CONFIG['TRADING_SETTINGS'].get('SOFTMAX_TEMPERATURE', 0.5)
LOSS_COOLDOWN_MINUTES = _CONFIG['TRADING_SETTINGS'].get('LOSS_COOLDOWN_MINUTES', 15)

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
DD_INCLUDE_UNREALIZED = _DD_PROTECTION.get('INCLUDE_UNREALIZED_PNL', True)

# ============================================================
# REST SCHEDULE - FROM MASTER_CONFIG.json
# ============================================================

_REST_SCHEDULE = _CONFIG.get('REST_SCHEDULE', {})
REST_SCHEDULE_ENABLED = _REST_SCHEDULE.get('ENABLED', False)
REST_FOREX_WINDOWS = _REST_SCHEDULE.get('FOREX_WINDOWS', [])
REST_CRYPTO_WINDOWS = _REST_SCHEDULE.get('CRYPTO_WINDOWS', [])
REST_PERSIST_ON_REST = _REST_SCHEDULE.get('PERSIST_STATE_ON_REST', True)
REST_PERSIST_ON_TRADE_CLOSE = _REST_SCHEDULE.get('PERSIST_ON_TRADE_CLOSE', True)
REST_PERIODIC_SAVE_MINUTES = _REST_SCHEDULE.get('PERIODIC_SAVE_MINUTES', 30)

# ============================================================
# ASSET CLASSES - FROM MASTER_CONFIG.json
# ============================================================

ASSET_CLASSES = _CONFIG.get('ASSET_CLASSES', {})

def get_asset_class(symbol: str) -> str:
    """Get asset class for a symbol (crypto or forex). Defaults to forex."""
    return ASSET_CLASSES.get(symbol, 'forex')

# ============================================================
# FTMO LIMITS - FROM MASTER_CONFIG.json
# ============================================================

_FTMO_LIMITS = _CONFIG.get('FTMO_LIMITS', {})
FTMO_DAILY_DD_PCT = _FTMO_LIMITS.get('DAILY_DD_PCT', 5.0)
FTMO_TOTAL_DD_PCT = _FTMO_LIMITS.get('TOTAL_DD_PCT', 10.0)
FTMO_PROFIT_TARGET_PCT = _FTMO_LIMITS.get('PROFIT_TARGET_PCT', 10.0)
FTMO_MIN_TRADING_DAYS = _FTMO_LIMITS.get('MIN_TRADING_DAYS', 4)
FTMO_INCLUDE_UNREALIZED = _FTMO_LIMITS.get('INCLUDE_UNREALIZED_PNL', True)
FTMO_CLOSE_BEFORE_WEEKEND = _FTMO_LIMITS.get('CLOSE_BEFORE_WEEKEND', True)
FTMO_TRADE_STAGGER = _FTMO_LIMITS.get('TRADE_STAGGER_SECONDS', [30, 120])
FTMO_RISK_PER_TRADE = _FTMO_LIMITS.get('FTMO_RISK_PER_TRADE', 150.0)
FTMO_ATR_MULTIPLIER_CFG = _FTMO_LIMITS.get('FTMO_ATR_MULTIPLIER', 0.5)
FTMO_RISK_PERCENT = _FTMO_LIMITS.get('FTMO_RISK_PERCENT', 0.00075)
FTMO_CHALLENGE_PROFILE = _FTMO_LIMITS.get('CHALLENGE_PROFILE', {})
FTMO_FUNDED_PROFILE = _FTMO_LIMITS.get('FUNDED_PROFILE', {})

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
    "WATCHDOG_LIMIT": WATCHDOG_LIMIT,
    "LSTM_MAX_AGE_DAYS": LSTM_MAX_AGE_DAYS,
    "SOFTMAX_TEMPERATURE": SOFTMAX_TEMPERATURE,
    "LOSS_COOLDOWN_MINUTES": LOSS_COOLDOWN_MINUTES,
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
    "FTMO_RISK_PER_TRADE": FTMO_RISK_PER_TRADE,
    "FTMO_ATR_MULTIPLIER_CFG": FTMO_ATR_MULTIPLIER_CFG,
    "FTMO_RISK_PERCENT": FTMO_RISK_PERCENT,
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
    global QUANTUM_BRIDGE_CONFIG, LSTM_MAX_AGE_DAYS, SOFTMAX_TEMPERATURE, WATCHDOG_LIMIT, LOSS_COOLDOWN_MINUTES
    global DD_PROTECTION_ENABLED, DAILY_DD_LIMIT_DOLLARS
    global DD_WARNING_THRESHOLD_PCT, DD_CRITICAL_THRESHOLD_PCT
    global DD_WARNING_DOPAMINE_MULT, DD_CRITICAL_DOPAMINE_MULT
    global DD_RESET_HOUR_UTC, DD_INCLUDE_UNREALIZED
    global SYMBOL_OVERRIDES, ASSET_CLASSES
    global REST_SCHEDULE_ENABLED, REST_FOREX_WINDOWS, REST_CRYPTO_WINDOWS
    global REST_PERSIST_ON_REST, REST_PERSIST_ON_TRADE_CLOSE, REST_PERIODIC_SAVE_MINUTES
    global FTMO_DAILY_DD_PCT, FTMO_TOTAL_DD_PCT, FTMO_PROFIT_TARGET_PCT
    global FTMO_MIN_TRADING_DAYS, FTMO_INCLUDE_UNREALIZED, FTMO_CLOSE_BEFORE_WEEKEND
    global FTMO_TRADE_STAGGER, FTMO_RISK_PER_TRADE, FTMO_ATR_MULTIPLIER_CFG
    global FTMO_RISK_PERCENT, FTMO_CHALLENGE_PROFILE, FTMO_FUNDED_PROFILE

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
    WATCHDOG_LIMIT = _CONFIG['TRADING_SETTINGS'].get('WATCHDOG_LIMIT', 1.00)
    LSTM_MAX_AGE_DAYS = _CONFIG['TRADING_SETTINGS'].get('LSTM_MAX_AGE_DAYS', 7)
    SOFTMAX_TEMPERATURE = _CONFIG['TRADING_SETTINGS'].get('SOFTMAX_TEMPERATURE', 0.5)
    LOSS_COOLDOWN_MINUTES = _CONFIG['TRADING_SETTINGS'].get('LOSS_COOLDOWN_MINUTES', 15)
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
    DD_INCLUDE_UNREALIZED = _dd.get('INCLUDE_UNREALIZED_PNL', True)
    SYMBOL_OVERRIDES = _CONFIG['TRADING_SETTINGS'].get('SYMBOL_OVERRIDES', {})
    ASSET_CLASSES = _CONFIG.get('ASSET_CLASSES', {})
    _rest = _CONFIG.get('REST_SCHEDULE', {})
    REST_SCHEDULE_ENABLED = _rest.get('ENABLED', False)
    REST_FOREX_WINDOWS = _rest.get('FOREX_WINDOWS', [])
    REST_CRYPTO_WINDOWS = _rest.get('CRYPTO_WINDOWS', [])
    REST_PERSIST_ON_REST = _rest.get('PERSIST_STATE_ON_REST', True)
    REST_PERSIST_ON_TRADE_CLOSE = _rest.get('PERSIST_ON_TRADE_CLOSE', True)
    REST_PERIODIC_SAVE_MINUTES = _rest.get('PERIODIC_SAVE_MINUTES', 30)
    _ftmo = _CONFIG.get('FTMO_LIMITS', {})
    FTMO_DAILY_DD_PCT = _ftmo.get('DAILY_DD_PCT', 5.0)
    FTMO_TOTAL_DD_PCT = _ftmo.get('TOTAL_DD_PCT', 10.0)
    FTMO_PROFIT_TARGET_PCT = _ftmo.get('PROFIT_TARGET_PCT', 10.0)
    FTMO_MIN_TRADING_DAYS = _ftmo.get('MIN_TRADING_DAYS', 4)
    FTMO_INCLUDE_UNREALIZED = _ftmo.get('INCLUDE_UNREALIZED_PNL', True)
    FTMO_CLOSE_BEFORE_WEEKEND = _ftmo.get('CLOSE_BEFORE_WEEKEND', True)
    FTMO_TRADE_STAGGER = _ftmo.get('TRADE_STAGGER_SECONDS', [30, 120])
    FTMO_RISK_PER_TRADE = _ftmo.get('FTMO_RISK_PER_TRADE', 150.0)
    FTMO_ATR_MULTIPLIER_CFG = _ftmo.get('FTMO_ATR_MULTIPLIER', 0.5)
    FTMO_RISK_PERCENT = _ftmo.get('FTMO_RISK_PERCENT', 0.00075)
    FTMO_CHALLENGE_PROFILE = _ftmo.get('CHALLENGE_PROFILE', {})
    FTMO_FUNDED_PROFILE = _ftmo.get('FUNDED_PROFILE', {})
    ACCOUNTS = _CONFIG['ACCOUNTS']
    # Re-inject passwords from credential manager (same as import-time injection)
    if _CREDENTIALS_AVAILABLE:
        _reload_cred_status = validate_credentials()
        for _key in ACCOUNTS:
            if _reload_cred_status.get(_key, False):
                ACCOUNTS[_key]['password'] = get_password(_key)
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
        "WATCHDOG_LIMIT": WATCHDOG_LIMIT,
        "LSTM_MAX_AGE_DAYS": LSTM_MAX_AGE_DAYS,
        "SOFTMAX_TEMPERATURE": SOFTMAX_TEMPERATURE,
        "LOSS_COOLDOWN_MINUTES": LOSS_COOLDOWN_MINUTES,
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
        "SYMBOL_OVERRIDES": SYMBOL_OVERRIDES,
        "ASSET_CLASSES": ASSET_CLASSES,
        "FTMO_DAILY_DD_PCT": FTMO_DAILY_DD_PCT,
        "FTMO_TOTAL_DD_PCT": FTMO_TOTAL_DD_PCT,
        "FTMO_PROFIT_TARGET_PCT": FTMO_PROFIT_TARGET_PCT,
        "FTMO_RISK_PER_TRADE": FTMO_RISK_PER_TRADE,
        "FTMO_ATR_MULTIPLIER_CFG": FTMO_ATR_MULTIPLIER_CFG,
        "FTMO_RISK_PERCENT": FTMO_RISK_PERCENT,
        "FTMO_CHALLENGE_PROFILE": FTMO_CHALLENGE_PROFILE,
        "FTMO_FUNDED_PROFILE": FTMO_FUNDED_PROFILE,
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
