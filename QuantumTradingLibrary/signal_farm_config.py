"""
SIGNAL FARM CONFIG - 5 Virtual Account Parameter Sets
======================================================
Defines parameter sets for 5 virtual challenge accounts,
challenge rules, and symbol specifications.

These are SELF-CONTAINED parameters for the signal farm only.
They do NOT modify MASTER_CONFIG.json or any existing files.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ============================================================
# SYMBOL SPECIFICATIONS
# ============================================================

@dataclass
class SymbolSpec:
    """Per-symbol trading specs resolved at runtime from MT5."""
    name: str
    dollar_per_point: float = 0.0   # Resolved from mt5.symbol_info()
    tick_size: float = 0.0
    tick_value: float = 0.0
    point: float = 0.0
    digits: int = 2
    volume_min: float = 0.01
    volume_max: float = 10.0
    volume_step: float = 0.01

FARM_SYMBOLS = ["BTCUSD", "XAUUSD", "ETHUSD"]

# ============================================================
# ACCOUNT PARAMETER SETS
# ============================================================

@dataclass
class FarmParams:
    """Trading parameters for one virtual account."""
    account_id: str
    label: str
    confidence_threshold: float
    tp_atr_multiplier: float
    sl_atr_multiplier: float
    max_positions_per_symbol: int
    grid_spacing_atr: float
    partial_tp_ratio: float          # Fraction of TP distance for partial close
    breakeven_trigger: float         # Fraction of TP distance to move SL to entry
    trail_start_trigger: float       # Fraction of TP distance to start trailing
    trail_distance_atr: float        # Trail distance as ATR multiple
    max_loss_dollars: float          # Risk per trade
    compression_boost: float         # Added to raw confidence (as percentage points / 100)

    # Fixed across all accounts
    base_lot_size: float = 0.01
    max_lot_size: float = 0.10
    risk_per_trade_pct: float = 0.5  # 0.5% of balance

FARM_ACCOUNTS: Dict[str, FarmParams] = {
    "FARM_01": FarmParams(
        account_id="FARM_01",
        label="Conservative",
        confidence_threshold=0.35,
        tp_atr_multiplier=4.0,
        sl_atr_multiplier=2.0,
        max_positions_per_symbol=2,
        grid_spacing_atr=0.8,
        partial_tp_ratio=0.50,
        breakeven_trigger=0.25,
        trail_start_trigger=0.40,
        trail_distance_atr=1.2,
        max_loss_dollars=0.75,
        compression_boost=8.0,
    ),
    "FARM_02": FarmParams(
        account_id="FARM_02",
        label="Baseline",
        confidence_threshold=0.22,
        tp_atr_multiplier=3.0,
        sl_atr_multiplier=1.5,
        max_positions_per_symbol=3,
        grid_spacing_atr=0.5,
        partial_tp_ratio=0.50,
        breakeven_trigger=0.30,
        trail_start_trigger=0.50,
        trail_distance_atr=1.0,
        max_loss_dollars=1.00,
        compression_boost=12.0,
    ),
    "FARM_03": FarmParams(
        account_id="FARM_03",
        label="Aggressive",
        confidence_threshold=0.15,
        tp_atr_multiplier=2.5,
        sl_atr_multiplier=1.0,
        max_positions_per_symbol=5,
        grid_spacing_atr=0.3,
        partial_tp_ratio=0.40,
        breakeven_trigger=0.35,
        trail_start_trigger=0.60,
        trail_distance_atr=0.8,
        max_loss_dollars=1.50,
        compression_boost=15.0,
    ),
    "FARM_04": FarmParams(
        account_id="FARM_04",
        label="Grid-Heavy",
        confidence_threshold=0.22,
        tp_atr_multiplier=3.0,
        sl_atr_multiplier=1.5,
        max_positions_per_symbol=5,
        grid_spacing_atr=0.4,
        partial_tp_ratio=0.50,
        breakeven_trigger=0.30,
        trail_start_trigger=0.50,
        trail_distance_atr=1.0,
        max_loss_dollars=1.00,
        compression_boost=12.0,
    ),
    "FARM_05": FarmParams(
        account_id="FARM_05",
        label="Tight-Stop",
        confidence_threshold=0.28,
        tp_atr_multiplier=2.0,
        sl_atr_multiplier=0.8,
        max_positions_per_symbol=3,
        grid_spacing_atr=0.5,
        partial_tp_ratio=0.60,
        breakeven_trigger=0.20,
        trail_start_trigger=0.40,
        trail_distance_atr=0.6,
        max_loss_dollars=0.60,
        compression_boost=10.0,
    ),
}

# ============================================================
# CHALLENGE RULES (Harder Than Any Real Prop Firm)
# ============================================================

@dataclass
class ChallengeRules:
    starting_balance: float = 5000.0
    profit_target_pct: float = 0.12      # 12% to pass
    daily_dd_limit_pct: float = 0.035    # 3.5% daily drawdown
    max_dd_limit_pct: float = 0.07       # 7% max drawdown
    time_limit_days: int = 15            # Trading days
    min_trading_days: int = 5            # Must trade at least 5 days
    bars_per_day: int = 288              # M5 bars in 24h (crypto)

CHALLENGE_RULES = ChallengeRules()

# ============================================================
# ENGINE SETTINGS
# ============================================================

ENGINE_CHECK_INTERVAL = 30       # Seconds between bar checks
INDICATOR_BUFFER_SIZE = 250      # Bars to keep in rolling buffer
MT5_TIMEFRAME = "M5"             # mt5.TIMEFRAME_M5

# MT5 connection - uses the first available terminal
MT5_TERMINAL_PATHS = [
    r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
    r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe",
    r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
]

# Reporting
COLLECTION_SERVER_URL = "http://203.161.61.61:8888"
BASE44_WEBHOOK_URL = ""  # Set when available

# ============================================================
# LOGGING
# ============================================================

LOG_DIR = "signal_farm_logs"
LOG_LEVEL = "INFO"
