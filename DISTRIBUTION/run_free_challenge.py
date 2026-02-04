"""
FREE CHALLENGE RUNNER
======================
Start a simulated prop firm challenge with one command.
Proves the system works, sends signals to network.

Usage:
  python run_free_challenge.py                    # Start FTMO 100K challenge
  python run_free_challenge.py --preset FTMO_50K  # Different preset
  python run_free_challenge.py --resume           # Continue existing challenge
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "QuantumTradingLibrary"))

from simulated_challenge import SimulatedChallenge, CHALLENGE_PRESETS, ChallengeStatus

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler('challenge.log'),
        logging.StreamHandler()
    ]
)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║            QUANTUMCHILDREN FREE CHALLENGE                    ║
║                                                              ║
║   Prove the system works - no money required                 ║
║   Pass the challenge, get your certificate                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_status(challenge: SimulatedChallenge):
    stats = challenge.get_stats()

    status_color = {
        "IN_PROGRESS": "",
        "PASSED": " [PASSED!]",
        "FAILED_DAILY_DD": " [FAILED - Daily DD]",
        "FAILED_MAX_DD": " [FAILED - Max DD]",
        "FAILED_TIME": " [FAILED - Time]"
    }

    print(f"\n{'='*60}")
    print(f"  {stats['challenge']}{status_color.get(stats['status'], '')}")
    print(f"{'='*60}")
    print(f"  Balance:    ${stats['balance']:>12,.2f}")
    print(f"  Equity:     ${stats['equity']:>12,.2f}")
    print(f"  Profit:     ${stats['profit']:>+12,.2f} ({stats['profit_pct']})")
    print(f"  Progress:   {stats['progress']} of {stats['target_pct']} target")
    print(f"{'='*60}")
    print(f"  Daily DD:   {stats['daily_drawdown']} (max 5%)")
    print(f"  Total DD:   {stats['total_drawdown']} (max 10%)")
    print(f"{'='*60}")
    print(f"  Days:       {stats['days_elapsed']} elapsed | {stats['trading_days']}/{stats['min_trading_days']} trading days")
    print(f"  Trades:     {stats['total_trades']} total | {stats['open_trades']} open")
    print(f"{'='*60}\n")


def find_latest_challenge() -> str:
    """Find most recent challenge file"""
    files = list(Path(".").glob("challenge_*.json"))
    if files:
        return str(max(files, key=lambda f: f.stat().st_mtime))
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='QuantumChildren Free Challenge')
    parser.add_argument('--preset', '-p', default='FTMO_100K',
                       choices=list(CHALLENGE_PRESETS.keys()),
                       help='Challenge type (default: FTMO_100K)')
    parser.add_argument('--resume', '-r', action='store_true',
                       help='Resume most recent challenge')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Show status and exit')

    args = parser.parse_args()

    print_banner()

    # Load or create challenge
    if args.resume:
        challenge_file = find_latest_challenge()
        if challenge_file:
            challenge = SimulatedChallenge.load(challenge_file)
            print(f"Resumed challenge from {challenge_file}")
        else:
            print("No existing challenge found. Starting new one.")
            config = CHALLENGE_PRESETS[args.preset]
            challenge = SimulatedChallenge(config)
    else:
        config = CHALLENGE_PRESETS[args.preset]
        challenge = SimulatedChallenge(config)
        print(f"Started new {args.preset} challenge")

    # Show status
    print_status(challenge)

    if args.status:
        if challenge.status == ChallengeStatus.PASSED:
            print(challenge.generate_certificate())
        return

    if challenge.status != ChallengeStatus.IN_PROGRESS:
        print("This challenge is complete.")
        if challenge.status == ChallengeStatus.PASSED:
            print(challenge.generate_certificate())
        else:
            print(f"Result: {challenge.status.value}")
            print(f"Reason: {challenge.fail_reason}")
            print("\nStart a new challenge with: python run_free_challenge.py")
        return

    print("Challenge is running. Connect your trading bot to start.")
    print("Press Ctrl+C to stop monitoring.\n")

    # Monitor loop
    try:
        while challenge.status == ChallengeStatus.IN_PROGRESS:
            time.sleep(30)
            print_status(challenge)
    except KeyboardInterrupt:
        print("\nChallenge paused. Resume with: python run_free_challenge.py --resume")
        challenge.save()

    # Final status
    if challenge.status == ChallengeStatus.PASSED:
        print(challenge.generate_certificate())


if __name__ == "__main__":
    main()
