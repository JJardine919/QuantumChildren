"""
EMERGENCY STOP GUARDIAN
========================
AI-powered watchdog that monitors all trading systems and can halt everything.

Monitors:
- Account drawdown
- Rapid loss detection
- Daily loss limits
- Position exposure
- System health

Can trigger:
- Close all positions
- Disable all trading
- Alert user

Uses local LLM (Ollama) for intelligent emergency detection.

Author: DooDoo + Claude
Date: 2026-01-30
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

import MetaTrader5 as mt5

# Try to import Ollama for LLM reasoning
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARNING] Ollama not installed - using rule-based emergency detection")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][GUARDIAN] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('emergency_guardian.log'),
        logging.StreamHandler()
    ]
)

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class GuardianConfig:
    """Emergency thresholds"""
    # Drawdown limits
    EMERGENCY_DRAWDOWN_PCT: float = 0.08      # 8% = EMERGENCY
    WARNING_DRAWDOWN_PCT: float = 0.05        # 5% = WARNING

    # Daily loss
    MAX_DAILY_LOSS_PCT: float = 0.045         # 4.5% daily = close all

    # Rapid loss detection
    RAPID_LOSS_WINDOW_SECONDS: int = 300      # 5 min window
    RAPID_LOSS_THRESHOLD_PCT: float = 0.02    # 2% in 5 min = emergency

    # Position limits
    MAX_OPEN_POSITIONS: int = 5
    MAX_EXPOSURE_PCT: float = 0.50            # 50% of balance in margin

    # Check interval
    CHECK_INTERVAL_SECONDS: int = 10          # Check every 10 sec

    # LLM model for reasoning
    LLM_MODEL: str = "gemma3:12b"             # Your 12B Gemma model


# All accounts to monitor
MONITORED_ACCOUNTS = [
    {
        'name': 'BlueGuardian $5K',
        'account': 366604,
        'terminal_path': r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
        'initial_balance': 5000,
    },
    {
        'name': 'BlueGuardian $100K',
        'account': 365060,
        'terminal_path': r"C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe",
        'initial_balance': 100000,
    },
]


# ============================================================
# EMERGENCY ACTIONS
# ============================================================

def close_all_positions(reason: str) -> int:
    """EMERGENCY: Close all open positions"""
    logging.warning(f"!!! EMERGENCY CLOSE ALL: {reason} !!!")

    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        logging.info("No positions to close")
        return 0

    closed = 0
    for pos in positions:
        symbol = pos.symbol
        ticket = pos.ticket
        volume = pos.volume
        pos_type = pos.type

        # Determine close direction
        if pos_type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "magic": 999999,  # Emergency magic
            "comment": f"EMERGENCY: {reason[:20]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            logging.error(f"order_send returned None for {ticket}")
            continue
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Closed position {ticket} on {symbol}")
            closed += 1
        else:
            logging.error(f"Failed to close {ticket}: {result.comment}")

    return closed


def kill_all_trading_processes():
    """Kill all Python trading processes"""
    import subprocess
    logging.warning("!!! KILLING ALL TRADING PROCESSES !!!")

    # Kill quantum brain processes
    subprocess.run(['taskkill', '/F', '/FI', 'WINDOWTITLE eq *quantum*'],
                   capture_output=True)
    subprocess.run(['taskkill', '/F', '/FI', 'WINDOWTITLE eq *brain*'],
                   capture_output=True)
    subprocess.run(['taskkill', '/F', '/FI', 'WINDOWTITLE eq *PASSIVE*'],
                   capture_output=True)


def create_emergency_flag():
    """Create flag file to signal emergency to other systems"""
    flag_path = Path("EMERGENCY_STOP.flag")
    flag_path.write_text(f"EMERGENCY STOP ACTIVATED\nTime: {datetime.now()}\n")
    logging.warning(f"Created emergency flag: {flag_path}")


# ============================================================
# LLM EMERGENCY REASONING
# ============================================================

def ask_llm_about_emergency(situation: dict) -> tuple[bool, str]:
    """Ask LLM if this is an emergency situation"""
    if not OLLAMA_AVAILABLE:
        return False, "LLM not available"

    prompt = f"""You are a trading risk manager. Analyze this situation and decide if it's an EMERGENCY requiring immediate position closure.

CURRENT SITUATION:
- Account Balance: ${situation['balance']:,.2f}
- Starting Balance: ${situation['starting_balance']:,.2f}
- Current Drawdown: {situation['drawdown_pct']*100:.2f}%
- Daily P&L: ${situation['daily_pnl']:,.2f} ({situation['daily_pnl_pct']*100:.2f}%)
- Open Positions: {situation['num_positions']}
- Total Exposure: ${situation['total_exposure']:,.2f}
- Recent Loss Rate: {situation['recent_loss_rate']*100:.2f}% in last 5 minutes

EMERGENCY THRESHOLDS:
- Max Drawdown: 8%
- Max Daily Loss: 4.5%
- Rapid Loss: 2% in 5 minutes

Respond with ONLY one of these:
- "EMERGENCY: [reason]" if positions should be closed immediately
- "WARNING: [reason]" if situation is concerning but not critical
- "OK: [reason]" if situation is acceptable

Your response:"""

    try:
        response = ollama.chat(
            model=GuardianConfig.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}  # Low temp for consistent decisions
        )

        answer = response['message']['content'].strip()

        if answer.startswith("EMERGENCY"):
            return True, answer
        else:
            return False, answer

    except Exception as e:
        logging.error(f"LLM error: {e}")
        return False, f"LLM error: {e}"


# ============================================================
# RULE-BASED EMERGENCY DETECTION
# ============================================================

class EmergencyDetector:
    """Rule-based emergency detection"""

    def __init__(self, config: GuardianConfig):
        self.config = config
        self.balance_history: List[tuple] = []  # (timestamp, balance)
        self.day_start_balance: float = 0
        self.current_day: Optional[datetime] = None

    def check_emergency(self, account_info) -> tuple[bool, str]:
        """Check all emergency conditions"""
        balance = account_info.balance
        equity = account_info.equity
        margin = account_info.margin

        now = datetime.now()

        # Reset daily tracking at midnight
        if self.current_day != now.date():
            self.current_day = now.date()
            self.day_start_balance = balance
            logging.info(f"New day - starting balance: ${balance:,.2f}")

        # Track balance history
        self.balance_history.append((now, balance))
        # Keep only last 10 minutes
        cutoff = now - timedelta(minutes=10)
        self.balance_history = [(t, b) for t, b in self.balance_history if t > cutoff]

        # Check 1: Drawdown from starting balance
        if self.day_start_balance > 0:
            drawdown = (self.day_start_balance - equity) / self.day_start_balance
            if drawdown >= self.config.EMERGENCY_DRAWDOWN_PCT:
                return True, f"DRAWDOWN EMERGENCY: {drawdown*100:.1f}% (limit: {self.config.EMERGENCY_DRAWDOWN_PCT*100}%)"

        # Check 2: Daily loss limit
        if self.day_start_balance > 0:
            daily_loss = (self.day_start_balance - balance) / self.day_start_balance
            if daily_loss >= self.config.MAX_DAILY_LOSS_PCT:
                return True, f"DAILY LOSS LIMIT: {daily_loss*100:.1f}% (limit: {self.config.MAX_DAILY_LOSS_PCT*100}%)"

        # Check 3: Rapid loss detection
        if len(self.balance_history) >= 2:
            window_start = now - timedelta(seconds=self.config.RAPID_LOSS_WINDOW_SECONDS)
            old_balances = [b for t, b in self.balance_history if t <= window_start]
            if old_balances:
                old_balance = old_balances[-1]
                rapid_loss = (old_balance - balance) / old_balance
                if rapid_loss >= self.config.RAPID_LOSS_THRESHOLD_PCT:
                    return True, f"RAPID LOSS: {rapid_loss*100:.1f}% in {self.config.RAPID_LOSS_WINDOW_SECONDS}s"

        # Check 4: Excessive positions
        positions = mt5.positions_get()
        if positions and len(positions) > self.config.MAX_OPEN_POSITIONS:
            return True, f"TOO MANY POSITIONS: {len(positions)} (limit: {self.config.MAX_OPEN_POSITIONS})"

        # Check 5: Margin exposure
        if balance > 0 and margin / balance > self.config.MAX_EXPOSURE_PCT:
            return True, f"EXCESSIVE EXPOSURE: {margin/balance*100:.1f}% margin used"

        return False, "OK"


# ============================================================
# MAIN GUARDIAN
# ============================================================

class EmergencyGuardian:
    """Main guardian that watches all accounts"""

    def __init__(self, config: GuardianConfig = None):
        self.config = config or GuardianConfig()
        self.detector = EmergencyDetector(self.config)
        self.emergency_triggered = False

    def connect_account(self, account_config: dict) -> bool:
        """Connect to an account"""
        mt5.shutdown()

        if not mt5.initialize(path=account_config['terminal_path']):
            logging.error(f"Failed to init MT5: {mt5.last_error()}")
            return False

        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info")
            return False

        return True

    def check_account(self, account_config: dict) -> tuple[bool, str]:
        """Check a single account for emergencies"""
        if not self.connect_account(account_config):
            return False, "Connection failed"

        account_info = mt5.account_info()
        positions = mt5.positions_get()

        # Rule-based check
        is_emergency, reason = self.detector.check_emergency(account_info)

        if is_emergency:
            return True, reason

        # LLM check (if available and rule-based passed)
        if OLLAMA_AVAILABLE and not is_emergency:
            situation = {
                'balance': account_info.balance,
                'starting_balance': self.detector.day_start_balance or account_config['initial_balance'],
                'drawdown_pct': (self.detector.day_start_balance - account_info.equity) / self.detector.day_start_balance if self.detector.day_start_balance > 0 else 0,
                'daily_pnl': account_info.balance - self.detector.day_start_balance if self.detector.day_start_balance > 0 else 0,
                'daily_pnl_pct': (account_info.balance - self.detector.day_start_balance) / self.detector.day_start_balance if self.detector.day_start_balance > 0 else 0,
                'num_positions': len(positions) if positions else 0,
                'total_exposure': account_info.margin,
                'recent_loss_rate': 0,  # Calculated above in detector
            }

            llm_emergency, llm_reason = ask_llm_about_emergency(situation)
            if llm_emergency:
                return True, llm_reason

        return False, "OK"

    def trigger_emergency(self, reason: str):
        """Execute emergency stop"""
        logging.critical("=" * 60)
        logging.critical("!!! EMERGENCY STOP TRIGGERED !!!")
        logging.critical(f"Reason: {reason}")
        logging.critical("=" * 60)

        self.emergency_triggered = True

        # 1. Close all positions
        closed = close_all_positions(reason)
        logging.info(f"Closed {closed} positions")

        # 2. Create flag file
        create_emergency_flag()

        # 3. Kill trading processes
        kill_all_trading_processes()

        logging.critical("EMERGENCY STOP COMPLETE")

    def run(self):
        """Main guardian loop"""
        logging.info("=" * 60)
        logging.info("EMERGENCY GUARDIAN ACTIVATED")
        logging.info(f"LLM Available: {OLLAMA_AVAILABLE}")
        logging.info(f"Monitoring {len(MONITORED_ACCOUNTS)} accounts")
        logging.info(f"Check interval: {self.config.CHECK_INTERVAL_SECONDS}s")
        logging.info("=" * 60)

        try:
            while not self.emergency_triggered:
                for account in MONITORED_ACCOUNTS:
                    is_emergency, reason = self.check_account(account)

                    if is_emergency:
                        self.trigger_emergency(f"[{account['name']}] {reason}")
                        return

                    # Log status
                    account_info = mt5.account_info()
                    if account_info:
                        positions = mt5.positions_get()
                        num_pos = len(positions) if positions else 0
                        logging.info(f"[{account['name']}] Balance: ${account_info.balance:,.2f} | "
                                   f"Equity: ${account_info.equity:,.2f} | "
                                   f"Positions: {num_pos} | Status: {reason}")

                time.sleep(self.config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logging.info("Guardian stopped by user")
        finally:
            mt5.shutdown()


# ============================================================
# PANIC BUTTON - IMMEDIATE CLOSE ALL
# ============================================================

def panic_close_all():
    """Immediate panic close - no questions asked"""
    print("=" * 60)
    print("!!! PANIC BUTTON PRESSED !!!")
    print("=" * 60)

    for account in MONITORED_ACCOUNTS:
        mt5.shutdown()
        if mt5.initialize(path=account['terminal_path']):
            closed = close_all_positions("PANIC BUTTON")
            print(f"[{account['name']}] Closed {closed} positions")

    kill_all_trading_processes()
    create_emergency_flag()

    print("=" * 60)
    print("ALL POSITIONS CLOSED - ALL TRADING HALTED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Emergency Stop Guardian')
    parser.add_argument('--panic', action='store_true', help='Immediate panic close all')
    parser.add_argument('--check', action='store_true', help='Single check and exit')

    args = parser.parse_args()

    if args.panic:
        panic_close_all()
    elif args.check:
        guardian = EmergencyGuardian()
        for account in MONITORED_ACCOUNTS:
            is_emergency, reason = guardian.check_account(account)
            status = "EMERGENCY" if is_emergency else "OK"
            print(f"[{account['name']}] {status}: {reason}")
    else:
        guardian = EmergencyGuardian()
        guardian.run()
