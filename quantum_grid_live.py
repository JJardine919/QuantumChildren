"""
QuantumChildren LIVE GRID SYSTEM
Regime-aware grid trading with Archiver + LSTM
AGGRESSIVE MODE for competition catch-up

NO OLLAMA - Pure Python, runs headless
"""

import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import time
import json
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import threading
import atexit
from pathlib import Path

# Add QuantumTradingLibrary to path for credential_manager
sys.path.insert(0, str(Path(__file__).parent / "QuantumTradingLibrary"))
from credential_manager import get_credentials, CredentialError

# Register cleanup on script exit
atexit.register(mt5.shutdown)

# =============================================================================
# OBSERVE MODE - Set True to prevent account switching/login
# =============================================================================
OBSERVE_ONLY = False  # When True, uses current terminal state without login

# =============================================================================
# CONFIGURATION - AGGRESSIVE COMPETITION MODE
# =============================================================================

# Blue Guardian Accounts - loaded from credential_manager
# NOTE: Using ONE account to avoid AutoTrading disable on account switch
# For multi-account, need separate MT5 terminals per account
def _load_accounts():
    """Load account credentials from credential_manager"""
    accounts = []
    try:
        creds = get_credentials('BG_CHALLENGE')
        accounts.append({
            "name": "BG_100K_CHALLENGE",
            "account": creds['account'],
            "password": creds['password'],
            "server": creds['server'],
            "magic_number": 365001,
        })
    except CredentialError as e:
        print(f"[!] Failed to load BG_CHALLENGE credentials: {e}")
        accounts.append({
            "name": "BG_100K_CHALLENGE",
            "account": 365060,
            "password": "",
            "server": "BlueGuardian-Server",
            "magic_number": 365001,
        })
    # Commented out - add back when running separate terminals
    # try:
    #     creds = get_credentials('BG_INSTANT')
    #     accounts.append({
    #         "name": "BG_5K_INSTANT",
    #         "account": creds['account'],
    #         "password": creds['password'],
    #         "server": creds['server'],
    #         "magic_number": 366001,
    #     })
    # except CredentialError:
    #     pass
    return accounts

ACCOUNTS = _load_accounts()

CONFIG = {
    "terminal_path": r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",

    # Symbols
    "symbols": ["BTCUSD", "ETHUSD"],
    "timeframe": mt5.TIMEFRAME_M5,

    # Max lot caps per symbol (BTC 0.01, ETH 0.1 broker min)
    "max_lots": {
        "BTCUSD": 0.01,
        "ETHUSD": 0.1,   # Broker minimum for ETH
    },

    # Archiver thresholds (relaxed for more signals)
    "fidelity_entry": 0.65,      # Enter grid when above this
    "fidelity_exit": 0.55,       # Close grid when below this
    "max_entropy": 4.5,          # Skip if above this

    # Grid settings - MICRO RISK ($0.50 max loss target)
    "grid_levels": 5,            # Positions per grid
    "grid_spacing_atr": 0.5,     # ATR multiplier between levels
    "risk_per_grid": 0.01,       # % of account per full grid (micro for $0.50 losses)

    # Position management - TIGHT STOPS ($0.30-0.40 max loss)
    "tp_atr_multiplier": 0.3,    # TP target
    "sl_atr_multiplier": 0.02,   # SL tight (~$0.30-0.40 loss)
    "partial_close_at": 0.5,     # Close 50% at half TP
    "trailing_activate": 0.5,    # Trail activates at 50% of TP
    "trailing_distance": 0.1,    # Trail distance as ATR multiplier

    # Execution
    "max_spread_points": 100,    # Max spread to enter
    "check_interval": 30,        # Seconds between checks
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GridState:
    symbol: str
    direction: str = ""          # "BUY" or "SELL"
    active: bool = False
    positions: List[int] = field(default_factory=list)  # Ticket numbers
    avg_entry: float = 0.0
    total_lots: float = 0.0
    grid_base_price: float = 0.0
    levels_filled: int = 0
    tp_price: float = 0.0
    sl_price: float = 0.0
    trailing_active: bool = False
    partial_closed: bool = False  # Track if 50% has been taken
    entry_fidelity: float = 0.0
    entry_time: datetime = None

# =============================================================================
# ARCHIVER - REGIME DETECTION
# =============================================================================

def calculate_fidelity(closes: np.ndarray) -> float:
    """Compression fidelity - how clean is the pattern?"""
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
    
    fidelity = (structure * 0.5 + noise_penalty * 0.5)
    return min(1.0, max(0.0, fidelity * 1.3))


def calculate_entropy(closes: np.ndarray) -> float:
    """Shannon entropy - how predictable is price action?"""
    if len(closes) < 20:
        return 5.0
    
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    hist, _ = np.histogram(returns, bins=20, density=True)
    hist = hist + 1e-10
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def detect_regime(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
    """
    Full regime detection:
    - Fidelity (pattern quality)
    - Entropy (predictability)
    - Direction (trend)
    - Strength (momentum)
    """
    fidelity = calculate_fidelity(closes)
    entropy = calculate_entropy(closes)
    
    # Trend direction via EMA
    ema_fast = _ema(closes, 8)
    ema_slow = _ema(closes, 21)
    
    # Momentum
    roc = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) > 10 else 0
    
    # ATR for volatility
    atr = calculate_atr(highs, lows, closes, 14)
    
    # Direction
    if ema_fast > ema_slow and roc > 0.1:
        direction = "BUY"
        strength = min(100, abs(roc) * 20)
    elif ema_fast < ema_slow and roc < -0.1:
        direction = "SELL"
        strength = min(100, abs(roc) * 20)
    else:
        direction = "NEUTRAL"
        strength = 0
    
    return {
        "fidelity": round(fidelity, 3),
        "entropy": round(entropy, 2),
        "direction": direction,
        "strength": round(strength, 1),
        "atr": round(atr, 2),
        "ema_fast": round(ema_fast, 2),
        "ema_slow": round(ema_slow, 2),
        "tradeable": fidelity >= CONFIG["fidelity_entry"] and entropy <= CONFIG["max_entropy"] and direction != "NEUTRAL"
    }


def _ema(data: np.ndarray, period: int) -> float:
    if len(data) < period:
        return float(np.mean(data))
    multiplier = 2 / (period + 1)
    ema = float(data[0])
    for price in data[1:]:
        ema = (float(price) * multiplier) + (ema * (1 - multiplier))
    return ema


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

# =============================================================================
# MT5 INTERFACE
# =============================================================================

def connect_mt5(account_config: dict) -> bool:
    """
    Connect to MT5 terminal and login to specified account.

    In OBSERVE_ONLY mode, just initializes without switching accounts.
    In normal mode, shuts down first to ensure clean state before switching.
    """
    if OBSERVE_ONLY:
        # Observe mode - just use current terminal state, no login/switching
        if not mt5.initialize(path=CONFIG["terminal_path"]):
            print(f"MT5 init failed: {mt5.last_error()}")
            return False
        acc = mt5.account_info()
        if acc:
            print(f"[OBSERVE] Connected to: {acc.login} - Balance: ${acc.balance:,.2f}")
        return True

    # Normal mode - shutdown first to ensure clean state before switching accounts
    mt5.shutdown()

    if not mt5.initialize(path=CONFIG["terminal_path"]):
        print(f"MT5 init failed: {mt5.last_error()}")
        return False

    if not mt5.login(account_config["account"], password=account_config["password"], server=account_config["server"]):
        print(f"Login failed: {mt5.last_error()}")
        return False

    account = mt5.account_info()
    print(f"[{account_config['name']}] Connected - Balance: ${account.balance:,.2f}")
    return True


def get_market_data(symbol: str, bars: int = 100) -> Optional[Dict]:
    rates = mt5.copy_rates_from_pos(symbol, CONFIG["timeframe"], 0, bars)
    if rates is None or len(rates) < bars:
        return None
    
    return {
        "closes": np.array([r['close'] for r in rates]),
        "highs": np.array([r['high'] for r in rates]),
        "lows": np.array([r['low'] for r in rates]),
        "times": [datetime.fromtimestamp(r['time']) for r in rates]
    }


def get_current_price(symbol: str) -> Optional[Dict]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return {"bid": tick.bid, "ask": tick.ask, "spread": tick.ask - tick.bid}


def get_symbol_info(symbol: str) -> Optional[Dict]:
    info = mt5.symbol_info(symbol)
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


def calculate_lot_size(symbol: str, sl_points: float) -> float:
    """Calculate lot size based on risk percentage"""
    account = mt5.account_info()
    if not account:
        return 0.01

    risk_amount = account.balance * (CONFIG["risk_per_grid"] / 100) / CONFIG["grid_levels"]

    info = get_symbol_info(symbol)
    if not info or sl_points == 0:
        return info["lot_min"] if info else 0.01

    # Lot size = Risk / (SL points * tick value / tick size)
    tick_value_per_point = info["tick_value"] / info["tick_size"]
    lot_size = risk_amount / (sl_points * tick_value_per_point)

    # Clamp to limits
    lot_size = max(info["lot_min"], min(info["lot_max"], lot_size))

    # Ensure at least minimum lot
    lot_size = max(lot_size, info["lot_min"])
    lot_size = round(lot_size / info["lot_step"]) * info["lot_step"]
    lot_size = max(round(lot_size, 2), 0.01)

    # Apply per-symbol max cap LAST (override everything)
    if symbol in CONFIG.get("max_lots", {}):
        lot_size = min(lot_size, CONFIG["max_lots"][symbol])

    return lot_size

# =============================================================================
# ORDER EXECUTION
# =============================================================================

def open_position(symbol: str, direction: str, lots: float, sl: float = 0, tp: float = 0, comment: str = "") -> Optional[int]:
    """Open a single position, return ticket or None"""
    price_data = get_current_price(symbol)
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
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": price,
        "deviation": 30,
        "magic": CURRENT_MAGIC,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Only add SL/TP if they're non-zero
    if sl > 0:
        request["sl"] = sl
    if tp > 0:
        request["tp"] = tp
    
    result = mt5.order_send(request)

    if result is None:
        error = mt5.last_error()
        print(f"    [X] Order send returned None! MT5 error: {error}")
        print(f"       Check: AutoTrading enabled? Symbol valid? Account connected?")
        print(f"       Request was: {request}")
        return None

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"    [OK] {direction} {lots} {symbol} @ {price:.2f} | Ticket: {result.order}")
        return result.order
    else:
        print(f"    [X] Order failed: {result.retcode} - {result.comment}")
        return None


def close_position(ticket: int) -> bool:
    """Close a position by ticket"""
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
        "magic": CURRENT_MAGIC,
        "comment": "Grid close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE


def close_all_positions(symbol: str) -> int:
    """Close all positions for symbol, return count closed"""
    positions = mt5.positions_get(symbol=symbol, magic=CURRENT_MAGIC)
    if not positions:
        return 0
    
    closed = 0
    for pos in positions:
        if close_position(pos.ticket):
            closed += 1
    return closed


def modify_position_sl(ticket: int, new_sl: float) -> bool:
    """Modify SL for trailing"""
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return False
    
    pos = position[0]
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": pos.symbol,
        "position": ticket,
        "sl": new_sl,
        "tp": pos.tp,
    }
    
    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE

# =============================================================================
# GRID MANAGEMENT
# =============================================================================

class GridManager:
    def __init__(self):
        self.grids: Dict[str, GridState] = {
            symbol: GridState(symbol=symbol) for symbol in CONFIG["symbols"]
        }
        self.total_pnl = 0.0
        self.trades_closed = 0
        self.wins = 0
    
    def deploy_grid(self, symbol: str, direction: str, regime: Dict, current_price: float):
        """Deploy a new grid"""
        grid = self.grids[symbol]
        
        if grid.active:
            return  # Already have active grid
        
        info = get_symbol_info(symbol)
        if not info:
            return
        
        atr = regime["atr"]
        spacing = atr * CONFIG["grid_spacing_atr"]
        
        # Calculate SL for position sizing
        sl_distance = atr * CONFIG["sl_atr_multiplier"]
        lot_size = calculate_lot_size(symbol, sl_distance / info["point"])
        
        print(f"\n>>> DEPLOYING {direction} GRID on {symbol}")
        print(f"   Fidelity: {regime['fidelity']} | Entropy: {regime['entropy']}")
        print(f"   ATR: {atr:.2f} | Spacing: {spacing:.2f}")
        print(f"   Lot per level: {lot_size}")

        # Open first position BEFORE marking grid active
        ticket = open_position(
            symbol=symbol,
            direction=direction,
            lots=lot_size,
            comment=f"QC Grid L1"
        )

        # Only activate grid if order succeeded
        if ticket:
            grid.direction = direction
            grid.active = True
            grid.grid_base_price = current_price
            grid.entry_fidelity = regime["fidelity"]
            grid.entry_time = datetime.now()
            grid.positions = [ticket]
            grid.levels_filled = 1
            grid.avg_entry = current_price
            grid.total_lots = lot_size

            # Set TP/SL
            if direction == "BUY":
                grid.tp_price = current_price + (atr * CONFIG["tp_atr_multiplier"])
                grid.sl_price = current_price - (atr * CONFIG["sl_atr_multiplier"])
            else:
                grid.tp_price = current_price - (atr * CONFIG["tp_atr_multiplier"])
                grid.sl_price = current_price + (atr * CONFIG["sl_atr_multiplier"])

            print(f"   [OK] Grid ACTIVE | TP: {grid.tp_price:.2f} | SL: {grid.sl_price:.2f}")
        else:
            print(f"   [!] Order failed - grid NOT activated")
    
    def manage_grid(self, symbol: str, regime: Dict, current_price: float):
        """Manage existing grid - add levels, check exits, trail"""
        grid = self.grids[symbol]
        
        if not grid.active:
            return
        
        info = get_symbol_info(symbol)
        atr = regime["atr"]
        spacing = atr * CONFIG["grid_spacing_atr"]
        
        # Check if regime broke down
        if regime["fidelity"] < CONFIG["fidelity_exit"]:
            print(f"\n[!] Fidelity dropped to {regime['fidelity']} - closing {symbol} grid")
            self.close_grid(symbol, "Fidelity exit")
            return
        
        # Check SL hit
        if grid.direction == "BUY" and current_price <= grid.sl_price:
            print(f"\n[SL HIT] on {symbol} grid")
            self.close_grid(symbol, "SL")
            return
        elif grid.direction == "SELL" and current_price >= grid.sl_price:
            print(f"\n[SL HIT] on {symbol} grid")
            self.close_grid(symbol, "SL")
            return

        # Check partial close at 50% of TP
        if not grid.partial_closed and grid.positions:
            tp_distance = abs(grid.tp_price - grid.avg_entry)
            partial_target = grid.avg_entry + (tp_distance * CONFIG.get("partial_close_at", 0.5)) if grid.direction == "BUY" else grid.avg_entry - (tp_distance * CONFIG.get("partial_close_at", 0.5))

            hit_partial = (grid.direction == "BUY" and current_price >= partial_target) or (grid.direction == "SELL" and current_price <= partial_target)

            if hit_partial:
                # Close half the positions
                positions_to_close = len(grid.positions) // 2 or 1
                print(f"\n[50% TP] Partial close on {symbol} - closing {positions_to_close} position(s)")
                for _ in range(positions_to_close):
                    if grid.positions:
                        ticket = grid.positions.pop(0)
                        close_position(ticket)
                grid.partial_closed = True
                # Move SL to breakeven
                grid.sl_price = grid.avg_entry
                print(f"   SL moved to breakeven: {grid.sl_price:.2f}")

        # Check full TP hit
        if grid.direction == "BUY" and current_price >= grid.tp_price:
            print(f"\n[TP HIT] on {symbol} grid!")
            self.close_grid(symbol, "TP")
            return
        elif grid.direction == "SELL" and current_price <= grid.tp_price:
            print(f"\n[TP HIT] on {symbol} grid!")
            self.close_grid(symbol, "TP")
            return
        
        # Add grid levels on pullbacks
        if grid.levels_filled < CONFIG["grid_levels"]:
            add_level = False
            
            if grid.direction == "BUY":
                # Add level if price pulled back by spacing
                level_price = grid.grid_base_price - (spacing * grid.levels_filled)
                if current_price <= level_price:
                    add_level = True
            else:
                level_price = grid.grid_base_price + (spacing * grid.levels_filled)
                if current_price >= level_price:
                    add_level = True
            
            if add_level:
                sl_distance = atr * CONFIG["sl_atr_multiplier"]
                lot_size = calculate_lot_size(symbol, sl_distance / info["point"])
                
                print(f"\n[+] Adding grid level {grid.levels_filled + 1} on {symbol}")
                ticket = open_position(
                    symbol=symbol,
                    direction=grid.direction,
                    lots=lot_size,
                    comment=f"QC Grid L{grid.levels_filled + 1}"
                )
                
                if ticket:
                    grid.positions.append(ticket)
                    grid.levels_filled += 1
                    
                    # Update average entry
                    total_value = grid.avg_entry * grid.total_lots + current_price * lot_size
                    grid.total_lots += lot_size
                    grid.avg_entry = total_value / grid.total_lots
                    
                    # Adjust TP based on new average
                    if grid.direction == "BUY":
                        grid.tp_price = grid.avg_entry + (atr * CONFIG["tp_atr_multiplier"])
                    else:
                        grid.tp_price = grid.avg_entry - (atr * CONFIG["tp_atr_multiplier"])
        
        # Trailing stop logic
        if not grid.trailing_active:
            profit_distance = abs(current_price - grid.avg_entry)
            if profit_distance >= atr * CONFIG["trailing_activate"]:
                grid.trailing_active = True
                print(f"\n[~] Trailing activated on {symbol}")
        
        if grid.trailing_active:
            trail_distance = atr * CONFIG["trailing_distance"]
            if grid.direction == "BUY":
                new_sl = current_price - trail_distance
                if new_sl > grid.sl_price:
                    grid.sl_price = new_sl
            else:
                new_sl = current_price + trail_distance
                if new_sl < grid.sl_price:
                    grid.sl_price = new_sl
    
    def close_grid(self, symbol: str, reason: str):
        """Close all positions in grid"""
        grid = self.grids[symbol]
        
        if not grid.active:
            return
        
        # Get current P&L before closing
        positions = mt5.positions_get(symbol=symbol, magic=CURRENT_MAGIC)
        grid_pnl = sum(p.profit for p in positions) if positions else 0
        
        closed = close_all_positions(symbol)
        
        self.total_pnl += grid_pnl
        self.trades_closed += 1
        if grid_pnl > 0:
            self.wins += 1
        
        duration = datetime.now() - grid.entry_time if grid.entry_time else None
        
        print(f"\n{'='*50}")
        print(f"[GRID CLOSED] {symbol}")
        print(f"   Reason: {reason}")
        print(f"   Positions closed: {closed}")
        print(f"   Grid P&L: ${grid_pnl:.2f}")
        print(f"   Duration: {duration}")
        print(f"   Session P&L: ${self.total_pnl:.2f}")
        print(f"   Win rate: {self.wins}/{self.trades_closed}")
        print(f"{'='*50}")
        
        # Reset grid state
        self.grids[symbol] = GridState(symbol=symbol)
    
    def get_status(self) -> Dict:
        """Get current status of all grids"""
        return {
            symbol: {
                "active": grid.active,
                "direction": grid.direction,
                "levels": grid.levels_filled,
                "avg_entry": grid.avg_entry,
                "tp": grid.tp_price,
                "sl": grid.sl_price,
                "trailing": grid.trailing_active
            }
            for symbol, grid in self.grids.items()
        }

# Global to track current account's magic number
CURRENT_MAGIC = 366001

# =============================================================================
# MAIN LOOP
# =============================================================================

def run_live_grid():
    """Main trading loop - SINGLE ACCOUNT, NO SWITCHING"""
    global CURRENT_MAGIC

    print("\n" + "="*60)
    print(">>> QUANTUMCHILDREN LIVE GRID SYSTEM <<<")
    print("   SINGLE ACCOUNT MODE - NO SWITCHING")
    print("="*60)

    account_config = ACCOUNTS[0]
    CURRENT_MAGIC = account_config["magic_number"]

    print(f"\n[CONFIG]:")
    print(f"   Account: {account_config['name']}")
    print(f"   Symbols: {CONFIG['symbols']}")
    print(f"   Grid levels: {CONFIG['grid_levels']}")
    print(f"   Risk per grid: {CONFIG['risk_per_grid']}%")
    print(f"   Fidelity threshold: {CONFIG['fidelity_entry']}")

    # Single grid manager
    manager = GridManager()
    cycle = 0

    # Connect ONCE and stay connected
    print(f"\n[*] Connecting to {account_config['name']}...")
    mt5.shutdown()
    if not mt5.initialize(path=CONFIG["terminal_path"]):
        print("[!] MT5 init failed")
        return
    if not mt5.login(account_config["account"], password=account_config["password"], server=account_config["server"]):
        print("[!] Login failed")
        return

    acc_info = mt5.account_info()
    print(f"[OK] Connected: {acc_info.login} | Balance: ${acc_info.balance:,.2f}")
    print(f"\n[*] IMPORTANT: Enable AutoTrading (Ctrl+E) if not already on")
    print(f"[*] Starting trading loop...\n")

    try:
        while True:
            cycle += 1

            for symbol in CONFIG["symbols"]:
                # Get market data
                data = get_market_data(symbol, 100)
                if not data:
                    continue

                price_data = get_current_price(symbol)
                if not price_data:
                    continue

                current_price = (price_data["bid"] + price_data["ask"]) / 2

                # Detect regime
                regime = detect_regime(data["closes"], data["highs"], data["lows"])

                grid = manager.grids[symbol]

                # Status output
                status = "[ACTIVE]" if grid.active else "[WAITING]"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} | ${current_price:.2f} | F:{regime['fidelity']:.2f} | {regime['direction']} | {status}", end="")

                if grid.active:
                    print(f" | L{grid.levels_filled}")
                    manager.manage_grid(symbol, regime, current_price)
                else:
                    print()
                    if regime["tradeable"]:
                        manager.deploy_grid(symbol, regime["direction"], regime, current_price)

            # Summary every 10 cycles
            if cycle % 10 == 0:
                acc = mt5.account_info()
                if acc:
                    print(f"\n[Cycle {cycle}] Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f}\n")

            time.sleep(CONFIG["check_interval"])

    except KeyboardInterrupt:
        print("\n\n[STOP] Stopping...")

        for symbol in CONFIG["symbols"]:
            if manager.grids[symbol].active:
                manager.close_grid(symbol, "Manual stop")

        account = mt5.account_info()
        print(f"\n[FINAL RESULTS]")
        if account:
            print(f"   Final balance: ${account.balance:,.2f}")
        print(f"   Session P&L: ${manager.total_pnl:.2f}")
        print(f"   Trades: {manager.trades_closed} | Wins: {manager.wins}")

        mt5.shutdown()


if __name__ == "__main__":
    run_live_grid()
