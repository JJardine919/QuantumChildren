"""
SIMULATED PROP FIRM CHALLENGE
==============================
Free demo mode that simulates prop firm challenge rules.
User can "prove" the system works before paying for a real challenge.

STILL SENDS SIGNALS TO COLLECTION SERVER - that's the point.

Features:
- Simulates $50K, $100K, $200K challenge accounts
- Tracks profit target (8-10%)
- Tracks max daily drawdown (5%)
- Tracks max total drawdown (10%)
- Pass/Fail status
- Certificate of completion (shareable proof)
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum

# Signal collection - THE WHOLE POINT
try:
    from entropy_collector import collect_signal, NODE_ID
    COLLECTION_ENABLED = True
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from entropy_collector import collect_signal, NODE_ID
        COLLECTION_ENABLED = True
    except:
        COLLECTION_ENABLED = False
        NODE_ID = "DEMO"


class ChallengeStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED_DAILY_DD = "FAILED_DAILY_DD"
    FAILED_MAX_DD = "FAILED_MAX_DD"
    FAILED_TIME = "FAILED_TIME"


@dataclass
class ChallengeConfig:
    """Prop firm challenge configuration"""
    name: str
    initial_balance: float
    profit_target_pct: float  # e.g., 0.08 for 8%
    max_daily_drawdown_pct: float  # e.g., 0.05 for 5%
    max_total_drawdown_pct: float  # e.g., 0.10 for 10%
    time_limit_days: int
    min_trading_days: int


# Common challenge presets
CHALLENGE_PRESETS = {
    "FTMO_50K": ChallengeConfig(
        name="FTMO Style $50K",
        initial_balance=50000,
        profit_target_pct=0.10,
        max_daily_drawdown_pct=0.05,
        max_total_drawdown_pct=0.10,
        time_limit_days=30,
        min_trading_days=4
    ),
    "FTMO_100K": ChallengeConfig(
        name="FTMO Style $100K",
        initial_balance=100000,
        profit_target_pct=0.10,
        max_daily_drawdown_pct=0.05,
        max_total_drawdown_pct=0.10,
        time_limit_days=30,
        min_trading_days=4
    ),
    "BLUEGUARDIAN_5K": ChallengeConfig(
        name="BlueGuardian Style $5K Instant",
        initial_balance=5000,
        profit_target_pct=0.08,
        max_daily_drawdown_pct=0.05,
        max_total_drawdown_pct=0.10,
        time_limit_days=0,  # No time limit
        min_trading_days=0
    ),
    "BLUEGUARDIAN_100K": ChallengeConfig(
        name="BlueGuardian Style $100K",
        initial_balance=100000,
        profit_target_pct=0.08,
        max_daily_drawdown_pct=0.05,
        max_total_drawdown_pct=0.10,
        time_limit_days=45,
        min_trading_days=3
    ),
}


@dataclass
class SimulatedTrade:
    """Record of a simulated trade"""
    ticket: int
    symbol: str
    direction: str
    volume: float
    open_price: float
    close_price: Optional[float]
    open_time: str
    close_time: Optional[str]
    profit: float
    status: str  # OPEN, CLOSED


class SimulatedChallenge:
    """
    Runs a simulated prop firm challenge.

    The user proves the system works without risking real money.
    Meanwhile, WE GET SIGNALS regardless of outcome.
    """

    def __init__(self, config: ChallengeConfig, save_path: str = None):
        self.config = config
        self.save_path = save_path or f"challenge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Challenge state
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.high_water_mark = config.initial_balance
        self.daily_start_balance = config.initial_balance

        self.start_time = datetime.now()
        self.current_day = self.start_time.date()
        self.trading_days = set()

        self.trades: list[SimulatedTrade] = []
        self.open_trades: dict[int, SimulatedTrade] = {}
        self.next_ticket = 1000

        self.status = ChallengeStatus.IN_PROGRESS
        self.fail_reason = None

        logging.info(f"Started Simulated Challenge: {config.name}")
        logging.info(f"  Balance: ${config.initial_balance:,.2f}")
        logging.info(f"  Target: {config.profit_target_pct*100:.0f}% (${config.initial_balance * config.profit_target_pct:,.2f})")
        logging.info(f"  Max Daily DD: {config.max_daily_drawdown_pct*100:.0f}%")
        logging.info(f"  Max Total DD: {config.max_total_drawdown_pct*100:.0f}%")

    def _check_new_day(self):
        """Reset daily tracking on new day"""
        today = datetime.now().date()
        if today != self.current_day:
            self.current_day = today
            self.daily_start_balance = self.balance
            logging.info(f"New trading day: {today}")

    def _check_drawdown(self) -> bool:
        """Check if drawdown limits breached. Returns True if OK."""
        # Daily drawdown from day start
        daily_dd = (self.daily_start_balance - self.equity) / self.config.initial_balance
        if daily_dd >= self.config.max_daily_drawdown_pct:
            self.status = ChallengeStatus.FAILED_DAILY_DD
            self.fail_reason = f"Daily drawdown {daily_dd*100:.2f}% exceeded {self.config.max_daily_drawdown_pct*100:.0f}%"
            logging.error(f"CHALLENGE FAILED: {self.fail_reason}")
            return False

        # Total drawdown from high water mark
        total_dd = (self.high_water_mark - self.equity) / self.config.initial_balance
        if total_dd >= self.config.max_total_drawdown_pct:
            self.status = ChallengeStatus.FAILED_MAX_DD
            self.fail_reason = f"Total drawdown {total_dd*100:.2f}% exceeded {self.config.max_total_drawdown_pct*100:.0f}%"
            logging.error(f"CHALLENGE FAILED: {self.fail_reason}")
            return False

        return True

    def _check_profit_target(self) -> bool:
        """Check if profit target reached. Returns True if passed."""
        profit_pct = (self.balance - self.config.initial_balance) / self.config.initial_balance
        if profit_pct >= self.config.profit_target_pct:
            # Check minimum trading days
            if len(self.trading_days) >= self.config.min_trading_days:
                self.status = ChallengeStatus.PASSED
                logging.info(f"CHALLENGE PASSED! Profit: {profit_pct*100:.2f}%")
                return True
            else:
                logging.info(f"Profit target reached but need {self.config.min_trading_days - len(self.trading_days)} more trading days")
        return False

    def open_trade(self, symbol: str, direction: str, volume: float, price: float, confidence: float = 0.0) -> int:
        """Open a simulated trade"""
        if self.status != ChallengeStatus.IN_PROGRESS:
            logging.warning("Challenge not in progress, cannot open trade")
            return -1

        self._check_new_day()
        self.trading_days.add(self.current_day)

        ticket = self.next_ticket
        self.next_ticket += 1

        trade = SimulatedTrade(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            volume=volume,
            open_price=price,
            close_price=None,
            open_time=datetime.now().isoformat(),
            close_time=None,
            profit=0.0,
            status="OPEN"
        )

        self.open_trades[ticket] = trade
        self.trades.append(trade)

        logging.info(f"SIM OPEN #{ticket}: {direction} {volume} {symbol} @ {price}")

        # SEND SIGNAL - this is why we exist
        if COLLECTION_ENABLED:
            try:
                collect_signal({
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": confidence,
                    "price": price,
                    "source": f"SIM_{self.config.name}",
                    "mode": "SIMULATED_CHALLENGE"
                })
            except:
                pass

        self.save()
        return ticket

    def update_trade(self, ticket: int, current_price: float):
        """Update P/L for an open trade"""
        if ticket not in self.open_trades:
            return

        trade = self.open_trades[ticket]

        # Calculate P/L (simplified - assumes $1 per point per 0.01 lot)
        if trade.direction == "BUY":
            points = current_price - trade.open_price
        else:
            points = trade.open_price - current_price

        # Rough P/L calculation (varies by symbol, this is approximate)
        trade.profit = points * trade.volume * 10  # Simplified

        # Update equity
        total_floating = sum(t.profit for t in self.open_trades.values())
        self.equity = self.balance + total_floating

        # Check drawdown
        self._check_drawdown()

    def close_trade(self, ticket: int, close_price: float) -> float:
        """Close a simulated trade"""
        if ticket not in self.open_trades:
            logging.warning(f"Trade {ticket} not found")
            return 0.0

        trade = self.open_trades[ticket]

        # Final P/L
        if trade.direction == "BUY":
            points = close_price - trade.open_price
        else:
            points = trade.open_price - close_price

        trade.profit = points * trade.volume * 10
        trade.close_price = close_price
        trade.close_time = datetime.now().isoformat()
        trade.status = "CLOSED"

        # Update balance
        self.balance += trade.profit
        self.equity = self.balance

        # Update high water mark
        if self.balance > self.high_water_mark:
            self.high_water_mark = self.balance

        del self.open_trades[ticket]

        logging.info(f"SIM CLOSE #{ticket}: {trade.symbol} @ {close_price} | P/L: ${trade.profit:+.2f}")

        # Check status
        self._check_drawdown()
        self._check_profit_target()

        self.save()
        return trade.profit

    def get_stats(self) -> dict:
        """Get current challenge statistics"""
        profit = self.balance - self.config.initial_balance
        profit_pct = profit / self.config.initial_balance
        target_pct = self.config.profit_target_pct

        daily_dd = (self.daily_start_balance - self.equity) / self.config.initial_balance
        total_dd = (self.high_water_mark - self.equity) / self.config.initial_balance

        return {
            "challenge": self.config.name,
            "status": self.status.value,
            "balance": self.balance,
            "equity": self.equity,
            "profit": profit,
            "profit_pct": f"{profit_pct*100:.2f}%",
            "target_pct": f"{target_pct*100:.0f}%",
            "progress": f"{min(100, (profit_pct/target_pct)*100):.1f}%",
            "daily_drawdown": f"{daily_dd*100:.2f}%",
            "total_drawdown": f"{total_dd*100:.2f}%",
            "trading_days": len(self.trading_days),
            "min_trading_days": self.config.min_trading_days,
            "total_trades": len(self.trades),
            "open_trades": len(self.open_trades),
            "days_elapsed": (datetime.now() - self.start_time).days
        }

    def save(self):
        """Save challenge state to file"""
        data = {
            "config": asdict(self.config),
            "state": {
                "balance": self.balance,
                "equity": self.equity,
                "high_water_mark": self.high_water_mark,
                "daily_start_balance": self.daily_start_balance,
                "start_time": self.start_time.isoformat(),
                "trading_days": [d.isoformat() for d in self.trading_days],
                "status": self.status.value,
                "fail_reason": self.fail_reason
            },
            "trades": [asdict(t) for t in self.trades]
        }

        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SimulatedChallenge':
        """Load challenge from file"""
        with open(path) as f:
            data = json.load(f)

        config = ChallengeConfig(**data["config"])
        challenge = cls(config, save_path=path)

        state = data["state"]
        challenge.balance = state["balance"]
        challenge.equity = state["equity"]
        challenge.high_water_mark = state["high_water_mark"]
        challenge.daily_start_balance = state["daily_start_balance"]
        challenge.start_time = datetime.fromisoformat(state["start_time"])
        challenge.trading_days = {datetime.fromisoformat(d).date() for d in state["trading_days"]}
        challenge.status = ChallengeStatus(state["status"])
        challenge.fail_reason = state["fail_reason"]

        # Restore trades
        for t in data["trades"]:
            trade = SimulatedTrade(**t)
            challenge.trades.append(trade)
            if trade.status == "OPEN":
                challenge.open_trades[trade.ticket] = trade
                challenge.next_ticket = max(challenge.next_ticket, trade.ticket + 1)

        return challenge

    def generate_certificate(self) -> str:
        """Generate a shareable proof of passing"""
        if self.status != ChallengeStatus.PASSED:
            return "Challenge not passed yet."

        stats = self.get_stats()

        cert = f"""
╔══════════════════════════════════════════════════════════════╗
║           QUANTUMCHILDREN CHALLENGE CERTIFICATE              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Challenge: {self.config.name:<43} ║
║  Status: PASSED                                              ║
║                                                              ║
║  Starting Balance: ${self.config.initial_balance:>15,.2f}                   ║
║  Final Balance:    ${self.balance:>15,.2f}                   ║
║  Profit:           ${stats['profit']:>15,.2f} ({stats['profit_pct']})          ║
║                                                              ║
║  Trading Days: {len(self.trading_days)}                                           ║
║  Total Trades: {len(self.trades)}                                           ║
║                                                              ║
║  Started:  {self.start_time.strftime('%Y-%m-%d %H:%M'):<44} ║
║  Passed:   {datetime.now().strftime('%Y-%m-%d %H:%M'):<44} ║
║                                                              ║
║  Node ID: {NODE_ID:<45} ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

This certificate proves completion of a simulated trading challenge
using the QuantumChildren trading system. Results are based on
demo/simulated data and do not guarantee future performance.

Ready for a real challenge? Visit your preferred prop firm.
"""
        return cert


def main():
    """Demo the simulated challenge"""
    import argparse

    parser = argparse.ArgumentParser(description='Simulated Prop Firm Challenge')
    parser.add_argument('--preset', '-p', default='FTMO_100K',
                       choices=list(CHALLENGE_PRESETS.keys()),
                       help='Challenge preset')
    parser.add_argument('--load', '-l', help='Load existing challenge from file')

    args = parser.parse_args()

    if args.load:
        challenge = SimulatedChallenge.load(args.load)
        print(f"Loaded challenge from {args.load}")
    else:
        config = CHALLENGE_PRESETS[args.preset]
        challenge = SimulatedChallenge(config)

    print("\n" + "="*60)
    print("  SIMULATED CHALLENGE")
    print("="*60)

    stats = challenge.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("="*60)
    print("\nThis challenge will send signals to the QuantumChildren network.")
    print("Run your trading bot to start accumulating trades.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    main()
