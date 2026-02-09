"""
SIGNAL FARM CONFIG - 30 Virtual Account Parameter Sets
=======================================================
Defines parameter sets for 30 virtual challenge accounts,
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
    # ================================================================
    # BATCH 2: 25 NEW ACCOUNTS (FARM_06 - FARM_30)
    # Designed for maximum signal diversity across parameter space
    # ================================================================
    # --- SCALPERS (quick in/out, tight SL/TP) ---
    "FARM_06": FarmParams(
        account_id="FARM_06", label="Scalp-Sniper",
        confidence_threshold=0.30, tp_atr_multiplier=1.5, sl_atr_multiplier=0.6,
        max_positions_per_symbol=2, grid_spacing_atr=0.3,
        partial_tp_ratio=0.70, breakeven_trigger=0.15, trail_start_trigger=0.30,
        trail_distance_atr=0.4, max_loss_dollars=0.50, compression_boost=8.0,
    ),
    "FARM_07": FarmParams(
        account_id="FARM_07", label="Scalp-Volume",
        confidence_threshold=0.20, tp_atr_multiplier=1.8, sl_atr_multiplier=0.7,
        max_positions_per_symbol=4, grid_spacing_atr=0.25,
        partial_tp_ratio=0.60, breakeven_trigger=0.20, trail_start_trigger=0.35,
        trail_distance_atr=0.5, max_loss_dollars=0.60, compression_boost=14.0,
    ),
    "FARM_08": FarmParams(
        account_id="FARM_08", label="Scalp-Elite",
        confidence_threshold=0.38, tp_atr_multiplier=1.6, sl_atr_multiplier=0.5,
        max_positions_per_symbol=2, grid_spacing_atr=0.4,
        partial_tp_ratio=0.65, breakeven_trigger=0.12, trail_start_trigger=0.25,
        trail_distance_atr=0.3, max_loss_dollars=0.40, compression_boost=6.0,
    ),
    # --- SWING TRADERS (wider targets, patient) ---
    "FARM_09": FarmParams(
        account_id="FARM_09", label="Swing-Patient",
        confidence_threshold=0.25, tp_atr_multiplier=5.0, sl_atr_multiplier=2.5,
        max_positions_per_symbol=2, grid_spacing_atr=1.0,
        partial_tp_ratio=0.40, breakeven_trigger=0.20, trail_start_trigger=0.35,
        trail_distance_atr=1.5, max_loss_dollars=1.20, compression_boost=10.0,
    ),
    "FARM_10": FarmParams(
        account_id="FARM_10", label="Swing-Loose",
        confidence_threshold=0.20, tp_atr_multiplier=4.5, sl_atr_multiplier=2.0,
        max_positions_per_symbol=3, grid_spacing_atr=0.8,
        partial_tp_ratio=0.45, breakeven_trigger=0.25, trail_start_trigger=0.40,
        trail_distance_atr=1.3, max_loss_dollars=1.00, compression_boost=12.0,
    ),
    "FARM_11": FarmParams(
        account_id="FARM_11", label="Swing-Tight",
        confidence_threshold=0.28, tp_atr_multiplier=4.0, sl_atr_multiplier=1.2,
        max_positions_per_symbol=2, grid_spacing_atr=0.7,
        partial_tp_ratio=0.50, breakeven_trigger=0.18, trail_start_trigger=0.30,
        trail_distance_atr=0.8, max_loss_dollars=0.80, compression_boost=9.0,
    ),
    # --- MOMENTUM (ride the wave) ---
    "FARM_12": FarmParams(
        account_id="FARM_12", label="Momo-Fast",
        confidence_threshold=0.12, tp_atr_multiplier=2.5, sl_atr_multiplier=1.0,
        max_positions_per_symbol=4, grid_spacing_atr=0.3,
        partial_tp_ratio=0.35, breakeven_trigger=0.30, trail_start_trigger=0.45,
        trail_distance_atr=0.7, max_loss_dollars=1.00, compression_boost=18.0,
    ),
    "FARM_13": FarmParams(
        account_id="FARM_13", label="Momo-Trend",
        confidence_threshold=0.18, tp_atr_multiplier=3.5, sl_atr_multiplier=1.5,
        max_positions_per_symbol=3, grid_spacing_atr=0.5,
        partial_tp_ratio=0.30, breakeven_trigger=0.35, trail_start_trigger=0.55,
        trail_distance_atr=1.0, max_loss_dollars=1.20, compression_boost=16.0,
    ),
    "FARM_14": FarmParams(
        account_id="FARM_14", label="Momo-Grid",
        confidence_threshold=0.15, tp_atr_multiplier=3.0, sl_atr_multiplier=1.2,
        max_positions_per_symbol=6, grid_spacing_atr=0.35,
        partial_tp_ratio=0.40, breakeven_trigger=0.25, trail_start_trigger=0.50,
        trail_distance_atr=0.9, max_loss_dollars=0.80, compression_boost=17.0,
    ),
    # --- QUALITY HUNTERS (high confidence, cherry-pick) ---
    "FARM_15": FarmParams(
        account_id="FARM_15", label="Quality-A",
        confidence_threshold=0.42, tp_atr_multiplier=3.5, sl_atr_multiplier=1.5,
        max_positions_per_symbol=2, grid_spacing_atr=0.8,
        partial_tp_ratio=0.50, breakeven_trigger=0.20, trail_start_trigger=0.40,
        trail_distance_atr=1.0, max_loss_dollars=1.00, compression_boost=5.0,
    ),
    "FARM_16": FarmParams(
        account_id="FARM_16", label="Quality-B",
        confidence_threshold=0.36, tp_atr_multiplier=3.0, sl_atr_multiplier=1.3,
        max_positions_per_symbol=3, grid_spacing_atr=0.6,
        partial_tp_ratio=0.55, breakeven_trigger=0.22, trail_start_trigger=0.42,
        trail_distance_atr=0.9, max_loss_dollars=0.90, compression_boost=7.0,
    ),
    "FARM_17": FarmParams(
        account_id="FARM_17", label="Quality-Trail",
        confidence_threshold=0.40, tp_atr_multiplier=4.0, sl_atr_multiplier=1.8,
        max_positions_per_symbol=2, grid_spacing_atr=0.9,
        partial_tp_ratio=0.45, breakeven_trigger=0.15, trail_start_trigger=0.28,
        trail_distance_atr=0.7, max_loss_dollars=1.10, compression_boost=6.0,
    ),
    # --- GRID SPECIALISTS (many positions, tight spacing) ---
    "FARM_18": FarmParams(
        account_id="FARM_18", label="Grid-Tight",
        confidence_threshold=0.20, tp_atr_multiplier=2.5, sl_atr_multiplier=1.2,
        max_positions_per_symbol=7, grid_spacing_atr=0.20,
        partial_tp_ratio=0.50, breakeven_trigger=0.25, trail_start_trigger=0.45,
        trail_distance_atr=0.8, max_loss_dollars=0.60, compression_boost=13.0,
    ),
    "FARM_19": FarmParams(
        account_id="FARM_19", label="Grid-Wide",
        confidence_threshold=0.22, tp_atr_multiplier=3.0, sl_atr_multiplier=1.5,
        max_positions_per_symbol=5, grid_spacing_atr=0.70,
        partial_tp_ratio=0.50, breakeven_trigger=0.30, trail_start_trigger=0.50,
        trail_distance_atr=1.1, max_loss_dollars=0.90, compression_boost=11.0,
    ),
    "FARM_20": FarmParams(
        account_id="FARM_20", label="Grid-Max",
        confidence_threshold=0.18, tp_atr_multiplier=2.8, sl_atr_multiplier=1.0,
        max_positions_per_symbol=7, grid_spacing_atr=0.25,
        partial_tp_ratio=0.45, breakeven_trigger=0.28, trail_start_trigger=0.48,
        trail_distance_atr=0.7, max_loss_dollars=0.50, compression_boost=15.0,
    ),
    # --- RISK VARIANTS (different $ exposure) ---
    "FARM_21": FarmParams(
        account_id="FARM_21", label="MicroRisk",
        confidence_threshold=0.22, tp_atr_multiplier=3.0, sl_atr_multiplier=1.5,
        max_positions_per_symbol=3, grid_spacing_atr=0.5,
        partial_tp_ratio=0.50, breakeven_trigger=0.25, trail_start_trigger=0.45,
        trail_distance_atr=1.0, max_loss_dollars=0.30, compression_boost=12.0,
        risk_per_trade_pct=0.25,
    ),
    "FARM_22": FarmParams(
        account_id="FARM_22", label="MidRisk",
        confidence_threshold=0.22, tp_atr_multiplier=3.0, sl_atr_multiplier=1.5,
        max_positions_per_symbol=3, grid_spacing_atr=0.5,
        partial_tp_ratio=0.50, breakeven_trigger=0.25, trail_start_trigger=0.45,
        trail_distance_atr=1.0, max_loss_dollars=1.50, compression_boost=12.0,
        risk_per_trade_pct=0.75,
    ),
    "FARM_23": FarmParams(
        account_id="FARM_23", label="HighRisk",
        confidence_threshold=0.22, tp_atr_multiplier=3.0, sl_atr_multiplier=1.5,
        max_positions_per_symbol=3, grid_spacing_atr=0.5,
        partial_tp_ratio=0.50, breakeven_trigger=0.25, trail_start_trigger=0.45,
        trail_distance_atr=1.0, max_loss_dollars=2.00, compression_boost=12.0,
        risk_per_trade_pct=1.0,
    ),
    # --- TRAIL VARIANTS (different trailing behavior) ---
    "FARM_24": FarmParams(
        account_id="FARM_24", label="Trail-Early",
        confidence_threshold=0.25, tp_atr_multiplier=3.0, sl_atr_multiplier=1.2,
        max_positions_per_symbol=3, grid_spacing_atr=0.5,
        partial_tp_ratio=0.50, breakeven_trigger=0.10, trail_start_trigger=0.20,
        trail_distance_atr=0.5, max_loss_dollars=0.80, compression_boost=11.0,
    ),
    "FARM_25": FarmParams(
        account_id="FARM_25", label="Trail-Late",
        confidence_threshold=0.25, tp_atr_multiplier=3.5, sl_atr_multiplier=1.5,
        max_positions_per_symbol=3, grid_spacing_atr=0.5,
        partial_tp_ratio=0.40, breakeven_trigger=0.40, trail_start_trigger=0.65,
        trail_distance_atr=1.2, max_loss_dollars=1.00, compression_boost=11.0,
    ),
    "FARM_26": FarmParams(
        account_id="FARM_26", label="Trail-Tight",
        confidence_threshold=0.25, tp_atr_multiplier=3.0, sl_atr_multiplier=1.3,
        max_positions_per_symbol=3, grid_spacing_atr=0.5,
        partial_tp_ratio=0.55, breakeven_trigger=0.15, trail_start_trigger=0.30,
        trail_distance_atr=0.3, max_loss_dollars=0.70, compression_boost=11.0,
    ),
    # --- HYBRID / EXPERIMENTAL ---
    "FARM_27": FarmParams(
        account_id="FARM_27", label="LowBar-BigTP",
        confidence_threshold=0.10, tp_atr_multiplier=5.0, sl_atr_multiplier=2.0,
        max_positions_per_symbol=4, grid_spacing_atr=0.4,
        partial_tp_ratio=0.30, breakeven_trigger=0.20, trail_start_trigger=0.35,
        trail_distance_atr=1.0, max_loss_dollars=1.00, compression_boost=20.0,
    ),
    "FARM_28": FarmParams(
        account_id="FARM_28", label="Picky-Modest",
        confidence_threshold=0.45, tp_atr_multiplier=2.0, sl_atr_multiplier=1.0,
        max_positions_per_symbol=2, grid_spacing_atr=0.6,
        partial_tp_ratio=0.60, breakeven_trigger=0.18, trail_start_trigger=0.35,
        trail_distance_atr=0.6, max_loss_dollars=0.60, compression_boost=4.0,
    ),
    "FARM_29": FarmParams(
        account_id="FARM_29", label="Balanced",
        confidence_threshold=0.26, tp_atr_multiplier=3.2, sl_atr_multiplier=1.4,
        max_positions_per_symbol=3, grid_spacing_atr=0.55,
        partial_tp_ratio=0.50, breakeven_trigger=0.22, trail_start_trigger=0.42,
        trail_distance_atr=0.9, max_loss_dollars=0.85, compression_boost=11.0,
    ),
    "FARM_30": FarmParams(
        account_id="FARM_30", label="Wildcard",
        confidence_threshold=0.14, tp_atr_multiplier=2.2, sl_atr_multiplier=0.6,
        max_positions_per_symbol=6, grid_spacing_atr=0.2,
        partial_tp_ratio=0.35, breakeven_trigger=0.35, trail_start_trigger=0.55,
        trail_distance_atr=0.4, max_loss_dollars=1.80, compression_boost=19.0,
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
