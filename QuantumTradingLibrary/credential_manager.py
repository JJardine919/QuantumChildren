"""
CREDENTIAL MANAGER - Secure credential loading
===============================================
Loads MT5 account credentials from environment variables or .env file.
NEVER commit credentials to git. Use .env.example as template.

Usage:
    from credential_manager import get_credentials
    creds = get_credentials('ATLAS')
    # Returns: {'account': 212000584, 'password': '***', 'server': '...'}
"""

import os
from pathlib import Path
from typing import Optional, Dict

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).parent / '.env'
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass  # dotenv not installed, use system env vars only


# VPS metadata (non-sensitive info that can be in code)
VPS_METADATA = {
    "VPS_1": {
        "host": "72.62.170.153",
        "user": "root",
        "env_key": "VPS_1_PASSWORD"
    },
    "VPS_2": {
        "host": "203.161.61.61",
        "user": "root",
        "env_key": "VPS_2_PASSWORD"
    }
}

# Account metadata (non-sensitive info that can be in code)
ACCOUNT_METADATA = {
    "BG_INSTANT": {
        "account": 366604,
        "server": "BlueGuardian-Server",
        "terminal_path": r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
        "magic": 366001,
        "symbols": ["BTCUSD"],
        "env_key": "BG_INSTANT_PASSWORD"
    },
    "BG_CHALLENGE": {
        "account": 365060,
        "server": "BlueGuardian-Server",
        "terminal_path": r"C:\Program Files\Blue Guardian MT5 Terminal 2\terminal64.exe",
        "magic": 365001,
        "symbols": ["BTCUSD"],
        "env_key": "BG_CHALLENGE_PASSWORD"
    },
    "ATLAS": {
        "account": 212000584,
        "server": "AtlasFunded-Server",
        "terminal_path": r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe",
        "magic": 212001,
        "symbols": ["BTCUSD", "ETHUSD", "XAUUSD"],
        "env_key": "ATLAS_PASSWORD"
    },
    "GL_1": {
        "account": 113326,
        "server": "GetLeveraged-Trade",
        "terminal_path": None,
        "magic": 113001,
        "symbols": ["BTCUSD", "XAUUSD", "ETHUSD"],
        "env_key": "GL_1_PASSWORD"
    },
    "GL_2": {
        "account": 113328,
        "server": "GetLeveraged-Trade",
        "terminal_path": None,
        "magic": 113002,
        "symbols": ["BTCUSD", "XAUUSD", "ETHUSD"],
        "env_key": "GL_2_PASSWORD"
    },
    "GL_3": {
        "account": 107245,
        "server": "GetLeveraged-Trade",
        "terminal_path": None,
        "magic": 107001,
        "symbols": ["BTCUSD", "XAUUSD", "ETHUSD"],
        "env_key": "GL_3_PASSWORD"
    },
    "FTMO": {
        "account": 1521063483,
        "server": "FTMO-Demo2",
        "terminal_path": r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
        "magic": 152001,
        "symbols": ["BTCUSD", "XAUUSD", "ETHUSD"],
        "env_key": "FTMO_PASSWORD"
    },
    "QNIF_FTMO": {
        "account": 1521096288,
        "server": "FTMO-Demo2",
        "terminal_path": r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
        "magic": 152002,
        "symbols": ["BTCUSD", "XAUUSD", "ETHUSD"],
        "env_key": "QNIF_FTMO_PASSWORD"
    },
    "QNIF_GL_3": {
        "account": 107245,
        "server": "GetLeveraged-Trade",
        "terminal_path": None,
        "magic": 107002,
        "symbols": ["BTCUSD", "XAUUSD", "ETHUSD"],
        "env_key": "QNIF_GL_3_PASSWORD"
    }
}


class CredentialError(Exception):
    """Raised when credentials cannot be loaded"""
    pass


def get_password(account_key: str) -> str:
    """
    Get password for an account from environment variable.

    Args:
        account_key: Account identifier (ATLAS, BG_INSTANT, etc.)

    Returns:
        Password string

    Raises:
        CredentialError: If password not found in environment
    """
    if account_key not in ACCOUNT_METADATA:
        raise CredentialError(f"Unknown account key: {account_key}")

    env_key = ACCOUNT_METADATA[account_key]["env_key"]
    password = os.environ.get(env_key)

    if not password:
        raise CredentialError(
            f"Password not found for {account_key}.\n"
            f"Set environment variable: {env_key}\n"
            f"Or create .env file with: {env_key}=your_password"
        )

    return password


def get_credentials(account_key: str) -> Dict:
    """
    Get full credentials for an account.

    Args:
        account_key: Account identifier (ATLAS, BG_INSTANT, etc.)

    Returns:
        Dict with account, password, server, terminal_path, magic, symbols
    """
    if account_key not in ACCOUNT_METADATA:
        raise CredentialError(f"Unknown account key: {account_key}")

    metadata = ACCOUNT_METADATA[account_key].copy()
    env_key = metadata.pop("env_key")

    password = os.environ.get(env_key)
    if not password:
        raise CredentialError(
            f"Password not found for {account_key}.\n"
            f"Set environment variable: {env_key}\n"
            f"Or create .env file with: {env_key}=your_password"
        )

    metadata["password"] = password
    return metadata


def get_all_credentials() -> Dict[str, Dict]:
    """
    Get credentials for all accounts that have passwords configured.
    Skips accounts without passwords (no error).

    Returns:
        Dict mapping account_key -> credentials
    """
    result = {}
    for key in ACCOUNT_METADATA:
        try:
            result[key] = get_credentials(key)
        except CredentialError:
            pass  # Skip accounts without configured passwords
    return result


def validate_credentials() -> Dict[str, bool]:
    """
    Check which accounts have credentials configured.

    Returns:
        Dict mapping account_key -> True/False (has password)
    """
    result = {}
    for key, meta in ACCOUNT_METADATA.items():
        env_key = meta["env_key"]
        result[key] = bool(os.environ.get(env_key))
    return result


def get_account_info(account_key: str) -> Dict:
    """
    Get non-sensitive account info (no password).
    Safe to use without credentials configured.

    Args:
        account_key: Account identifier

    Returns:
        Dict with account metadata (no password)
    """
    if account_key not in ACCOUNT_METADATA:
        raise CredentialError(f"Unknown account key: {account_key}")

    metadata = ACCOUNT_METADATA[account_key].copy()
    metadata.pop("env_key")
    return metadata


def get_vps_credentials(vps_key: str) -> Dict:
    """
    Get VPS SSH credentials.

    Args:
        vps_key: VPS identifier (VPS_1, VPS_2)

    Returns:
        Dict with host, user, password
    """
    if vps_key not in VPS_METADATA:
        raise CredentialError(f"Unknown VPS key: {vps_key}")

    metadata = VPS_METADATA[vps_key].copy()
    env_key = metadata.pop("env_key")

    password = os.environ.get(env_key)
    if not password:
        raise CredentialError(
            f"Password not found for {vps_key}.\n"
            f"Set environment variable: {env_key}\n"
            f"Or create .env file with: {env_key}=your_password"
        )

    metadata["password"] = password
    return metadata


def get_vps_info(vps_key: str) -> Dict:
    """
    Get non-sensitive VPS info (no password).

    Args:
        vps_key: VPS identifier

    Returns:
        Dict with host, user (no password)
    """
    if vps_key not in VPS_METADATA:
        raise CredentialError(f"Unknown VPS key: {vps_key}")

    metadata = VPS_METADATA[vps_key].copy()
    metadata.pop("env_key")
    return metadata


# ============================================================
# CLI - Show credential status
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  CREDENTIAL MANAGER - Status Check")
    print("=" * 60)
    print()

    status = validate_credentials()

    for key, has_creds in status.items():
        meta = ACCOUNT_METADATA[key]
        icon = "[OK]" if has_creds else "[  ]"
        print(f"  {icon} {key:15} Account: {meta['account']}")
        if not has_creds:
            print(f"       -> Set: {meta['env_key']}")

    print()
    configured = sum(status.values())
    total = len(status)
    print(f"  Configured: {configured}/{total} accounts")
    print()

    if configured < total:
        print("  To configure credentials:")
        print("  1. Create .env file in this directory")
        print("  2. Add lines like: ATLAS_PASSWORD=your_password")
        print("  3. Or set environment variables directly")

    print("=" * 60)
