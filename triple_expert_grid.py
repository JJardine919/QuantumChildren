"""
TRIPLE EXPERT GRID SYSTEM
3 Experts (Bearish/Bullish/Neutral) + Compression (+12)
$300K Atlas Funded Account

Grid-style order placement with:
- 10 orders max per expert (30 total)
- 80% regime confidence threshold
- 3:1 reward ratio
- Hidden/dynamic ATR-based SL/TP
- Trailing stop + breakeven
"""

import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import atexit

atexit.register(mt5.shutdown)

# =============================================================================
# ACCOUNT CONFIGURATION
# =============================================================================

ACCOUNT = {
    "name": "ATLAS_300K_GRID",
    "account": 212000584,
    "password": "M6NLk79MN@",
    "server": "AtlasFunded-Server",
    "magic_base": 212001,  # Each expert gets magic_base + expert_id
}

# =============================================================================
# GRID CONFIGURATION
# =============================================================================

CONFIG = {
    "terminal_path": r"C:\Program Files\Atlas Funded MT5\terminal64.exe",

    # Symbol
    "symbol": "BTCUSD",
    "timeframe": mt5.TIMEFRAME_M5,

    # Triple Expert Setup
    "experts": {
        "bullish": {"id": 1, "regime": "BUY", "magic": 212001},
        "bearish": {"id": 2, "regime": "SELL", "magic": 212002},
        "neutral": {"id": 3, "regime": "NEUTRAL", "magic": 212003},
    },

    # Grid Limits
    "max_orders_per_expert": 10,
    "max_total_orders": 30,

    # Regime Detection
    "regime_confidence_threshold": 0.80,  # 80% confident before trading
    "compression_boost": 12,  # +12% boost from compression

    # Risk Management - 3:1 Ratio
    "sl_atr_multiplier": 1.5,
    "tp_atr_multiplier": 3.0,  # 3:1 ratio
    "partial_take_profit": 0.5,  # 50% at first target

    # Dynamic/Hidden SL/TP
    "hidden_sl_tp": True,  # Don't send SL/TP to broker, manage internally
    "dynamic_atr_levels": True,  # Recalculate based on ATR

    # Trailing & Breakeven
    "trailing_stop": True,
    "trailing_activate_ratio": 0.5,  # Activate at 50% of TP
    "trailing_distance_atr": 0.5,
    "breakeven_move": True,
    "breakeven_trigger_ratio": 0.3,  # Move to BE at 30% of TP

    # Grid Spacing
    "grid_spacing_atr": 0.3,

    # Position Sizing (conservative for $300K)
    "risk_per_trade_pct": 0.5,  # 0.5% per trade
    "max_lot_size": 1.0,

    # Execution
    "check_interval": 15,  # seconds
    "max_spread_points": 150,
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExpertState:
    """State for each expert (bullish/bearish/neutral)"""
    name: str
    regime: str
    magic: int
    active: bool = False
    orders: List[int] = field(default_factory=list)  # Filled position tickets
    pending_orders: List[int] = field(default_factory=list)  # Pending order tickets
    avg_entry: float = 0.0
    total_lots: float = 0.0
    internal_tp: float = 0.0  # Hidden TP
    internal_sl: float = 0.0  # Hidden SL
    trailing_active: bool = False
    breakeven_hit: bool = False
    partial_closed: bool = False
    last_grid_price: float = 0.0
    confidence: float = 0.0


@dataclass
class RegimeAnalysis:
    """Regime detection results"""
    direction: str  # BUY, SELL, NEUTRAL
    confidence: float  # 0-100%
    fidelity: float  # Compression quality
    entropy: float
    atr: float
    trend_strength: float
    is_clean: bool  # Passes 80% threshold


# =============================================================================
# REGIME DETECTION WITH COMPRESSION
# =============================================================================

def calculate_compression_fidelity(closes: np.ndarray) -> float:
    """
    Quantum compression fidelity - how compressible is the price pattern?
    Higher = cleaner trend, lower = noisy/choppy
    """
    if len(closes) < 20:
        return 0.0

    normalized = (closes - np.min(closes)) / (np.max(closes) - np.min(closes) + 1e-10)

    # Autocorrelation for structure
    mean_norm = normalized - np.mean(normalized)
    autocorr = np.correlate(mean_norm, mean_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)
    structure = np.mean(np.abs(autocorr[1:15]))

    # Noise measurement
    diff = np.diff(normalized)
    noise = np.std(diff) / (np.std(normalized) + 1e-10)
    noise_penalty = max(0, 1 - noise * 0.8)

    # Compression ratio simulation
    fidelity = (structure * 0.5 + noise_penalty * 0.5)
    return min(1.0, max(0.0, fidelity * 1.3))


def calculate_entropy(closes: np.ndarray) -> float:
    """Shannon entropy - lower = more predictable"""
    if len(closes) < 20:
        return 5.0

    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    hist, _ = np.histogram(returns, bins=20, density=True)
    hist = hist + 1e-10
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def ema(data: np.ndarray, period: int) -> float:
    if len(data) < period:
        return float(np.mean(data))
    multiplier = 2 / (period + 1)
    result = float(data[0])
    for price in data[1:]:
        result = (float(price) * multiplier) + (result * (1 - multiplier))
    return result


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return float(np.mean(highs - lows))

    tr_list = []
    for i in range(1, min(period + 1, len(closes))):
        tr = max(
            highs[-i] - lows[-i],
            abs(highs[-i] - closes[-i-1]),
            abs(lows[-i] - closes[-i-1])
        )
        tr_list.append(tr)
    return float(np.mean(tr_list))


def detect_regime(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> RegimeAnalysis:
    """
    Full regime detection with compression boost.
    Returns direction, confidence, and whether it passes 80% threshold.
    """
    fidelity = calculate_compression_fidelity(closes)
    entropy = calculate_entropy(closes)
    atr = calculate_atr(highs, lows, closes, 14)

    # Trend detection via EMAs
    ema_fast = ema(closes, 8)
    ema_medium = ema(closes, 21)
    ema_slow = ema(closes, 50)

    # Rate of change
    roc = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) > 10 else 0

    # ADX-like trend strength
    up_moves = np.maximum(np.diff(highs[-14:]), 0)
    down_moves = np.maximum(-np.diff(lows[-14:]), 0)
    trend_strength = abs(np.mean(up_moves) - np.mean(down_moves)) / (atr + 1e-10) * 100

    # Determine direction
    bullish_signals = 0
    bearish_signals = 0

    if ema_fast > ema_medium > ema_slow:
        bullish_signals += 2
    elif ema_fast > ema_medium:
        bullish_signals += 1

    if ema_fast < ema_medium < ema_slow:
        bearish_signals += 2
    elif ema_fast < ema_medium:
        bearish_signals += 1

    if roc > 0.5:
        bullish_signals += 1
    elif roc < -0.5:
        bearish_signals += 1

    if closes[-1] > ema_slow:
        bullish_signals += 1
    else:
        bearish_signals += 1

    # Calculate base confidence
    total_signals = bullish_signals + bearish_signals
    if bullish_signals > bearish_signals:
        direction = "BUY"
        base_confidence = (bullish_signals / max(total_signals, 1)) * 100
    elif bearish_signals > bullish_signals:
        direction = "SELL"
        base_confidence = (bearish_signals / max(total_signals, 1)) * 100
    else:
        direction = "NEUTRAL"
        base_confidence = 50.0

    # Apply compression boost (+12% when fidelity is high)
    compression_boost = CONFIG["compression_boost"] * fidelity
    boosted_confidence = min(100, base_confidence + compression_boost)

    # Penalize high entropy (noisy markets)
    if entropy > 3.5:
        boosted_confidence *= 0.8

    # Check if passes threshold
    is_clean = boosted_confidence >= CONFIG["regime_confidence_threshold"] * 100

    return RegimeAnalysis(
        direction=direction,
        confidence=round(boosted_confidence, 1),
        fidelity=round(fidelity, 3),
        entropy=round(entropy, 2),
        atr=round(atr, 2),
        trend_strength=round(trend_strength, 1),
        is_clean=is_clean
    )


# =============================================================================
# MT5 INTERFACE
# =============================================================================

def connect_mt5() -> bool:
    """Connect to Atlas Funded MT5 terminal specifically"""
    atlas_path = r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe"

    if not mt5.initialize(path=atlas_path):
        print(f"[!] MT5 init failed: {mt5.last_error()}")
        # Try login if terminal is already open
        if not mt5.initialize():
            return False

    # Login to Atlas account
    if not mt5.login(ACCOUNT["account"], password=ACCOUNT["password"], server=ACCOUNT["server"]):
        print(f"[!] Login failed: {mt5.last_error()}")
        # Check if already logged in
        acc = mt5.account_info()
        if acc and acc.login == ACCOUNT["account"]:
            print(f"[OK] Already logged in: {acc.login}")
        else:
            return False

    acc = mt5.account_info()
    if not acc:
        print(f"[!] No account info")
        return False

    print(f"[OK] Connected: {acc.login} | Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f}")
    return True


def get_market_data(bars: int = 100) -> Optional[Dict]:
    """Get OHLC data"""
    rates = mt5.copy_rates_from_pos(CONFIG["symbol"], CONFIG["timeframe"], 0, bars)
    if rates is None or len(rates) < 50:
        return None

    return {
        "closes": np.array([r['close'] for r in rates]),
        "highs": np.array([r['high'] for r in rates]),
        "lows": np.array([r['low'] for r in rates]),
    }


def get_current_price() -> Optional[Dict]:
    """Get current bid/ask"""
    tick = mt5.symbol_info_tick(CONFIG["symbol"])
    if tick is None:
        return None
    return {"bid": tick.bid, "ask": tick.ask, "mid": (tick.bid + tick.ask) / 2}


def get_symbol_info() -> Optional[Dict]:
    """Get symbol specifications"""
    info = mt5.symbol_info(CONFIG["symbol"])
    if info is None:
        return None
    return {
        "point": info.point,
        "digits": info.digits,
        "lot_min": info.volume_min,
        "lot_max": info.volume_max,
        "lot_step": info.volume_step,
        "tick_value": info.trade_tick_value,
        "tick_size": info.trade_tick_size
    }


def calculate_lot_size(sl_distance: float) -> float:
    """Calculate lot size based on risk percentage"""
    account = mt5.account_info()
    if not account:
        return 0.01

    risk_amount = account.balance * (CONFIG["risk_per_trade_pct"] / 100)

    info = get_symbol_info()
    if not info or sl_distance == 0:
        return info["lot_min"] if info else 0.01

    # Lot size = Risk / (SL distance * tick value)
    sl_points = sl_distance / info["point"]
    tick_value_per_point = info["tick_value"] / info["tick_size"]
    lot_size = risk_amount / (sl_points * tick_value_per_point)

    # Clamp to limits
    lot_size = max(info["lot_min"], min(info["lot_max"], lot_size))
    lot_size = min(lot_size, CONFIG["max_lot_size"])
    lot_size = round(lot_size / info["lot_step"]) * info["lot_step"]

    return round(lot_size, 2)


# =============================================================================
# ORDER EXECUTION (Hidden SL/TP)
# =============================================================================

def open_order(direction: str, lots: float, magic: int, comment: str = "") -> Optional[int]:
    """
    Open order WITHOUT sending SL/TP to broker (hidden management).
    """
    price_data = get_current_price()
    if not price_data:
        return None

    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = price_data["ask"]
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = price_data["bid"]

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": CONFIG["symbol"],
        "volume": lots,
        "type": order_type,
        "price": price,
        "deviation": 30,
        "magic": magic,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        # NO sl/tp sent - managed internally
    }

    result = mt5.order_send(request)

    if result is None:
        print(f"    [X] Order send failed: {mt5.last_error()}")
        return None

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"    [OK] {direction} {lots} @ {price:.2f} | Ticket: {result.order}")
        return result.order
    else:
        print(f"    [X] Order failed: {result.retcode} - {result.comment}")
        return None


def close_order(ticket: int) -> bool:
    """Close a specific order by ticket"""
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return True  # Already closed

    pos = position[0]

    if pos.type == mt5.ORDER_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(pos.symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": order_type,
        "position": ticket,
        "price": price,
        "deviation": 30,
        "magic": pos.magic,
        "comment": "Grid close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return result and result.retcode == mt5.TRADE_RETCODE_DONE


def close_partial(ticket: int, close_pct: float = 0.5) -> bool:
    """Close partial position (e.g., 50%)"""
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return False

    pos = position[0]
    info = get_symbol_info()

    close_volume = pos.volume * close_pct
    close_volume = round(close_volume / info["lot_step"]) * info["lot_step"]
    close_volume = max(info["lot_min"], close_volume)

    if close_volume >= pos.volume:
        return close_order(ticket)

    if pos.type == mt5.ORDER_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(pos.symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": close_volume,
        "type": order_type,
        "position": ticket,
        "price": price,
        "deviation": 30,
        "magic": pos.magic,
        "comment": "Partial TP",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return result and result.retcode == mt5.TRADE_RETCODE_DONE


def get_open_positions(magic: int = None) -> List:
    """Get all open positions, optionally filtered by magic"""
    if magic:
        positions = mt5.positions_get(symbol=CONFIG["symbol"], magic=magic)
    else:
        positions = mt5.positions_get(symbol=CONFIG["symbol"])
    return list(positions) if positions else []


def get_pending_orders(magic: int = None) -> List:
    """Get all pending orders, optionally filtered by magic"""
    orders = mt5.orders_get(symbol=CONFIG["symbol"])
    if not orders:
        return []
    if magic:
        return [o for o in orders if o.magic == magic]
    return list(orders)


def place_pending_order(direction: str, lots: float, price: float, magic: int, comment: str = "") -> Optional[int]:
    """
    Place a pending limit order at a specific price.
    BUY_LIMIT = buy below current price
    SELL_LIMIT = sell above current price
    """
    current = get_current_price()
    if not current:
        return None

    info = get_symbol_info()
    if not info:
        return None

    # Determine order type based on direction and price relative to current
    if direction == "BUY":
        if price < current["ask"]:
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
        else:
            order_type = mt5.ORDER_TYPE_BUY_STOP
    else:
        if price > current["bid"]:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT
        else:
            order_type = mt5.ORDER_TYPE_SELL_STOP

    # Round price to symbol digits
    price = round(price, info["digits"])

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": CONFIG["symbol"],
        "volume": lots,
        "type": order_type,
        "price": price,
        "deviation": 30,
        "magic": magic,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)

    if result is None:
        print(f"    [X] Pending order failed: {mt5.last_error()}")
        return None

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        type_str = "BUY_LIMIT" if order_type == mt5.ORDER_TYPE_BUY_LIMIT else "SELL_LIMIT" if order_type == mt5.ORDER_TYPE_SELL_LIMIT else "STOP"
        print(f"    [OK] {type_str} {lots} @ {price:.2f} | Order: {result.order}")
        return result.order
    else:
        print(f"    [X] Pending order failed: {result.retcode} - {result.comment}")
        return None


def cancel_pending_order(order_ticket: int) -> bool:
    """Cancel a pending order"""
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": order_ticket,
    }
    result = mt5.order_send(request)
    return result and result.retcode == mt5.TRADE_RETCODE_DONE


def deploy_grid_orders(direction: str, base_price: float, atr: float, num_levels: int, magic: int, comment_prefix: str) -> List[int]:
    """
    Deploy a full grid of pending orders at once.

    For BUY: places buy limits below current price
    For SELL: places sell limits above current price

    Returns list of order tickets.
    """
    spacing = atr * CONFIG["grid_spacing_atr"]
    sl_distance = atr * CONFIG["sl_atr_multiplier"]
    lot_size = calculate_lot_size(sl_distance)

    orders = []

    print(f"\n>>> DEPLOYING {direction} GRID - {num_levels} levels, spacing: {spacing:.2f}")

    for i in range(num_levels):
        if direction == "BUY":
            # Buy limits below current price
            level_price = base_price - (spacing * (i + 1))
        else:
            # Sell limits above current price
            level_price = base_price + (spacing * (i + 1))

        ticket = place_pending_order(
            direction=direction,
            lots=lot_size,
            price=level_price,
            magic=magic,
            comment=f"{comment_prefix} L{i+1}"
        )

        if ticket:
            orders.append(ticket)

    print(f"    Grid deployed: {len(orders)}/{num_levels} orders placed")
    return orders


# =============================================================================
# TRIPLE EXPERT GRID MANAGER
# =============================================================================

class TripleExpertGrid:
    """
    Manages 3 experts (bullish/bearish/neutral) each with their own grid.
    """

    def __init__(self):
        self.experts: Dict[str, ExpertState] = {}
        for name, cfg in CONFIG["experts"].items():
            self.experts[name] = ExpertState(
                name=name,
                regime=cfg["regime"],
                magic=cfg["magic"]
            )

        self.total_pnl = 0.0
        self.trades_closed = 0
        self.wins = 0

    def get_total_orders(self) -> int:
        """Count total orders across all experts"""
        return sum(len(exp.orders) for exp in self.experts.values())

    def sync_positions(self):
        """Sync expert states with actual MT5 positions and pending orders"""
        for expert in self.experts.values():
            # Sync filled positions
            positions = get_open_positions(expert.magic)
            expert.orders = [p.ticket for p in positions]

            # Sync pending orders
            pending = get_pending_orders(expert.magic)
            expert.pending_orders = [o.ticket for o in pending]

            if positions:
                expert.active = True
                expert.total_lots = sum(p.volume for p in positions)
                expert.avg_entry = sum(p.price_open * p.volume for p in positions) / expert.total_lots
            elif pending:
                # Has pending orders but no filled positions yet
                expert.active = True
            else:
                if expert.active:  # Was active, now closed
                    expert.active = False
                    expert.trailing_active = False
                    expert.breakeven_hit = False
                    expert.partial_closed = False

    def should_add_order(self, expert: ExpertState, regime: RegimeAnalysis) -> bool:
        """Check if we should add an order for this expert"""
        # NEUTRAL expert doesn't trade - it's used as confirmation vote
        if expert.regime == "NEUTRAL":
            return False

        # Check order limits
        if len(expert.orders) >= CONFIG["max_orders_per_expert"]:
            return False
        if self.get_total_orders() >= CONFIG["max_total_orders"]:
            return False

        # Check regime matches expert
        if expert.regime == "BUY" and regime.direction != "BUY":
            return False
        if expert.regime == "SELL" and regime.direction != "SELL":
            return False

        # Check confidence threshold (80%)
        if not regime.is_clean:
            return False

        # NEUTRAL CONFIRMATION: Only trade if neutral "agrees"
        # Trend strength indicates momentum - higher = stronger trend
        neutral_confirms = regime.trend_strength > 20  # Lowered from 40 - was too restrictive

        if not neutral_confirms:
            print(f"    [{expert.name.upper()}] Blocked - trend_strength {regime.trend_strength:.1f} < 20")
            return False

        return True

    def deploy_expert_grid(self, expert: ExpertState, regime: RegimeAnalysis, current_price: float):
        """
        Deploy a FULL GRID for the expert - true grid style.
        Places pending orders at multiple price levels immediately.
        """
        atr = regime.atr

        # Determine direction
        if expert.regime == "BUY":
            direction = "BUY"
        elif expert.regime == "SELL":
            direction = "SELL"
        else:
            # Neutral expert - place both buy and sell grids
            direction = "BOTH"

        # Calculate how many levels we can place
        available_slots = CONFIG["max_orders_per_expert"] - len(expert.orders)
        total_available = CONFIG["max_total_orders"] - self.get_total_orders()
        num_levels = min(available_slots, total_available, CONFIG["max_orders_per_expert"])

        if num_levels <= 0:
            return

        print(f"\n>>> [{expert.name.upper()}] DEPLOYING GRID")
        print(f"    Direction: {direction} | Levels: {num_levels} | Confidence: {regime.confidence:.0f}%")

        if direction == "BOTH":
            # Neutral: place half buy, half sell
            half = num_levels // 2
            buy_orders = deploy_grid_orders("BUY", current_price, atr, half, expert.magic, f"{expert.name[:3].upper()}")
            sell_orders = deploy_grid_orders("SELL", current_price, atr, half, expert.magic, f"{expert.name[:3].upper()}")
            expert.pending_orders = buy_orders + sell_orders
        else:
            # Bullish or Bearish: place grid in one direction
            orders = deploy_grid_orders(direction, current_price, atr, num_levels, expert.magic, f"{expert.name[:3].upper()}")
            expert.pending_orders = orders

        # Also place one market order at current price to get in immediately
        sl_distance = atr * CONFIG["sl_atr_multiplier"]
        lot_size = calculate_lot_size(sl_distance)

        if direction != "BOTH":
            ticket = open_order(
                direction=direction,
                lots=lot_size,
                magic=expert.magic,
                comment=f"{expert.name[:3].upper()} ENTRY"
            )
            if ticket:
                expert.orders.append(ticket)

        expert.active = True
        expert.confidence = regime.confidence
        expert.avg_entry = current_price
        expert.last_grid_price = current_price

        # Set internal TP/SL (hidden from broker)
        if direction == "BUY" or direction == "BOTH":
            expert.internal_tp = current_price + (atr * CONFIG["tp_atr_multiplier"])
            expert.internal_sl = current_price - (atr * CONFIG["sl_atr_multiplier"])
        else:
            expert.internal_tp = current_price - (atr * CONFIG["tp_atr_multiplier"])
            expert.internal_sl = current_price + (atr * CONFIG["sl_atr_multiplier"])

        print(f"    [{expert.name.upper()}] Grid ACTIVE | TP: {expert.internal_tp:.2f} | SL: {expert.internal_sl:.2f}")

    def manage_expert(self, expert: ExpertState, regime: RegimeAnalysis, current_price: float):
        """Manage an active expert's positions"""
        if not expert.active or not expert.orders:
            return

        atr = regime.atr
        positions = get_open_positions(expert.magic)

        if not positions:
            expert.active = False
            return

        # Get current direction from first position
        direction = "BUY" if positions[0].type == mt5.ORDER_TYPE_BUY else "SELL"

        # Calculate current P&L
        current_pnl = sum(p.profit for p in positions)
        tp_distance = abs(expert.internal_tp - expert.avg_entry)

        # --- BREAKEVEN LOGIC ---
        if CONFIG["breakeven_move"] and not expert.breakeven_hit:
            be_trigger = tp_distance * CONFIG["breakeven_trigger_ratio"]
            if direction == "BUY" and current_price >= expert.avg_entry + be_trigger:
                expert.internal_sl = expert.avg_entry + (atr * 0.1)  # Slight profit lock
                expert.breakeven_hit = True
                print(f"    [{expert.name.upper()}] Breakeven hit - SL moved to {expert.internal_sl:.2f}")
            elif direction == "SELL" and current_price <= expert.avg_entry - be_trigger:
                expert.internal_sl = expert.avg_entry - (atr * 0.1)
                expert.breakeven_hit = True
                print(f"    [{expert.name.upper()}] Breakeven hit - SL moved to {expert.internal_sl:.2f}")

        # --- PARTIAL TAKE PROFIT (50%) ---
        if CONFIG["partial_take_profit"] > 0 and not expert.partial_closed:
            partial_target = tp_distance * CONFIG["partial_take_profit"]

            hit_partial = False
            if direction == "BUY" and current_price >= expert.avg_entry + partial_target:
                hit_partial = True
            elif direction == "SELL" and current_price <= expert.avg_entry - partial_target:
                hit_partial = True

            if hit_partial:
                print(f"\n    [{expert.name.upper()}] PARTIAL TP - Closing 50%")
                # Close half the orders
                orders_to_close = len(expert.orders) // 2 or 1
                for _ in range(orders_to_close):
                    if expert.orders:
                        ticket = expert.orders.pop(0)
                        if close_order(ticket):
                            self.trades_closed += 1
                            self.wins += 1
                expert.partial_closed = True
                # Move SL to breakeven
                expert.internal_sl = expert.avg_entry

        # --- TRAILING STOP ---
        if CONFIG["trailing_stop"] and not expert.trailing_active:
            trail_activate = tp_distance * CONFIG["trailing_activate_ratio"]
            if direction == "BUY" and current_price >= expert.avg_entry + trail_activate:
                expert.trailing_active = True
                print(f"    [{expert.name.upper()}] Trailing ACTIVATED")
            elif direction == "SELL" and current_price <= expert.avg_entry - trail_activate:
                expert.trailing_active = True
                print(f"    [{expert.name.upper()}] Trailing ACTIVATED")

        if expert.trailing_active:
            trail_distance = atr * CONFIG["trailing_distance_atr"]
            if direction == "BUY":
                new_sl = current_price - trail_distance
                if new_sl > expert.internal_sl:
                    expert.internal_sl = new_sl
            else:
                new_sl = current_price + trail_distance
                if new_sl < expert.internal_sl:
                    expert.internal_sl = new_sl

        # --- CHECK HIDDEN SL HIT ---
        sl_hit = False
        if direction == "BUY" and current_price <= expert.internal_sl:
            sl_hit = True
        elif direction == "SELL" and current_price >= expert.internal_sl:
            sl_hit = True

        if sl_hit:
            print(f"\n    [{expert.name.upper()}] HIDDEN SL HIT - Closing all")
            self.close_expert(expert, "SL")
            return

        # --- CHECK HIDDEN TP HIT ---
        tp_hit = False
        if direction == "BUY" and current_price >= expert.internal_tp:
            tp_hit = True
        elif direction == "SELL" and current_price <= expert.internal_tp:
            tp_hit = True

        if tp_hit:
            print(f"\n    [{expert.name.upper()}] HIDDEN TP HIT - Closing all")
            self.close_expert(expert, "TP")
            return

        # --- ADD GRID LEVELS ON PULLBACK ---
        if len(expert.orders) < CONFIG["max_orders_per_expert"]:
            grid_spacing = atr * CONFIG["grid_spacing_atr"]
            add_level = False

            if direction == "BUY":
                next_level = expert.last_grid_price - grid_spacing
                if current_price <= next_level:
                    add_level = True
            else:
                next_level = expert.last_grid_price + grid_spacing
                if current_price >= next_level:
                    add_level = True

            if add_level and regime.is_clean:
                print(f"\n    [{expert.name.upper()}] Adding grid level on pullback")
                self.deploy_expert_grid(expert, regime, current_price)

    def close_expert(self, expert: ExpertState, reason: str):
        """Close all positions for an expert"""
        positions = get_open_positions(expert.magic)
        pnl = sum(p.profit for p in positions) if positions else 0

        # Close filled positions
        for ticket in expert.orders[:]:
            if close_order(ticket):
                expert.orders.remove(ticket)
                self.trades_closed += 1

        # Cancel pending orders
        for ticket in expert.pending_orders[:]:
            if cancel_pending_order(ticket):
                expert.pending_orders.remove(ticket)

        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1

        print(f"    [{expert.name.upper()}] CLOSED | Reason: {reason} | P&L: ${pnl:.2f} | Pending cancelled: {len(expert.pending_orders)}")

        # Reset state
        expert.active = False
        expert.orders = []
        expert.pending_orders = []
        expert.avg_entry = 0
        expert.total_lots = 0
        expert.trailing_active = False
        expert.breakeven_hit = False
        expert.partial_closed = False

    def get_status(self) -> str:
        """Get status string - shows filled/pending orders"""
        lines = []
        for name, exp in self.experts.items():
            if exp.active:
                filled = len(exp.orders)
                pending = len(exp.pending_orders)
                status = f"[{filled}F/{pending}P]"
            else:
                status = "[--]"
            lines.append(f"{name[:4].upper()}: {status}")
        return " | ".join(lines)


# =============================================================================
# MAIN LOOP
# =============================================================================

def run_triple_expert_grid():
    """Main trading loop"""
    print("\n" + "="*60)
    print(">>> TRIPLE EXPERT GRID SYSTEM <<<")
    print("   Bullish + Bearish + Neutral Experts")
    print("   Atlas Funded $300K Account")
    print("="*60)

    print(f"\n[CONFIG]:")
    print(f"   Max orders per expert: {CONFIG['max_orders_per_expert']}")
    print(f"   Max total orders: {CONFIG['max_total_orders']}")
    print(f"   Regime confidence: {CONFIG['regime_confidence_threshold']*100:.0f}%")
    print(f"   Compression boost: +{CONFIG['compression_boost']}%")
    print(f"   Risk ratio: 1:{CONFIG['tp_atr_multiplier']/CONFIG['sl_atr_multiplier']:.1f}")
    print(f"   Hidden SL/TP: {CONFIG['hidden_sl_tp']}")
    print(f"   Trailing: {CONFIG['trailing_stop']} | Breakeven: {CONFIG['breakeven_move']}")

    # Connect
    print(f"\n[*] Connecting to Atlas Funded...")
    if not connect_mt5():
        print("[!] Connection failed - check terminal path and credentials")
        return

    # Initialize grid manager
    manager = TripleExpertGrid()
    manager.sync_positions()

    print(f"\n[*] Starting grid loop (Ctrl+C to stop)...\n")
    cycle = 0

    try:
        while True:
            cycle += 1

            # Get market data
            data = get_market_data()
            if not data:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No market data")
                time.sleep(CONFIG["check_interval"])
                continue

            price_data = get_current_price()
            if not price_data:
                time.sleep(CONFIG["check_interval"])
                continue

            current_price = price_data["mid"]

            # Detect regime
            regime = detect_regime(data["closes"], data["highs"], data["lows"])

            # Sync with MT5
            manager.sync_positions()

            # Status line
            status = manager.get_status()
            regime_str = f"{regime.direction} {regime.confidence:.0f}%"
            clean_str = "CLEAN" if regime.is_clean else "WAIT"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ${current_price:.2f} | {regime_str} | {clean_str} | {status}")

            # Process each expert
            for name, expert in manager.experts.items():
                if expert.active:
                    # Manage existing positions
                    manager.manage_expert(expert, regime, current_price)
                else:
                    # Check if should open new grid
                    if manager.should_add_order(expert, regime):
                        print(f"\n>>> [{expert.name.upper()}] Deploying grid - {regime.direction} @ {regime.confidence:.0f}% confidence")
                        manager.deploy_expert_grid(expert, regime, current_price)

            # Summary every 20 cycles
            if cycle % 20 == 0:
                acc = mt5.account_info()
                if acc:
                    print(f"\n[Cycle {cycle}] Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f} | Session P&L: ${manager.total_pnl:.2f}\n")

            time.sleep(CONFIG["check_interval"])

    except KeyboardInterrupt:
        print("\n\n[STOP] Stopping...")

        # Close all positions
        for expert in manager.experts.values():
            if expert.active:
                manager.close_expert(expert, "Manual stop")

        acc = mt5.account_info()
        print(f"\n[FINAL RESULTS]")
        if acc:
            print(f"   Final balance: ${acc.balance:,.2f}")
        print(f"   Session P&L: ${manager.total_pnl:.2f}")
        print(f"   Trades: {manager.trades_closed} | Wins: {manager.wins}")

        mt5.shutdown()


if __name__ == "__main__":
    run_triple_expert_grid()
