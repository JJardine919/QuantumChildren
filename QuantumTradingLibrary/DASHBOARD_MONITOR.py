"""
PROP FARM DASHBOARD - READ ONLY MONITOR
========================================
Shows status of ALL accounts WITHOUT affecting trades.
Does NOT trade. Does NOT switch accounts. Just watches.

Run this in a separate window to monitor everything.
Run individual BRAIN_*.py scripts in their own windows to trade.

Usage: python DASHBOARD_MONITOR.py
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

# Account configs - for display only
ACCOUNTS = {
    'BG_INSTANT': {
        'account': 366604,
        'name': 'BlueGuardian $5K Instant',
        'log': 'brain_bg_instant.log',
        'initial': 5000,
    },
    'BG_CHALLENGE': {
        'account': 365060,
        'name': 'BlueGuardian $100K Challenge',
        'log': 'brain_bg_challenge.log',
        'initial': 100000,
    },
    'ATLAS': {
        'account': 212000584,
        'name': 'Atlas Funded $300K',
        'log': 'brain_atlas.log',
        'initial': 300000,
    },
    'GL_1': {
        'account': 113326,
        'name': 'GetLeveraged #1',
        'log': 'brain_getleveraged.log',
        'initial': 300000,
    },
    'GL_2': {
        'account': 113328,
        'name': 'GetLeveraged #2',
        'log': 'brain_getleveraged.log',
        'initial': 300000,
    },
    'GL_3': {
        'account': 107245,
        'name': 'GetLeveraged #3',
        'log': 'brain_getleveraged.log',
        'initial': 300000,
    },
}


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_last_log_lines(log_file, n=5):
    """Get last N lines from a log file"""
    log_path = Path(__file__).parent / log_file
    if not log_path.exists():
        return ["(no log file)"]

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            return [l.strip() for l in lines[-n:]]
    except:
        return ["(error reading log)"]


def check_process_running(script_name):
    """Check if a Python script is running"""
    try:
        import subprocess
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True, text=True
        )
        # This is a simple check - just see if python is running
        # A more robust check would look at command line args
        return 'python.exe' in result.stdout
    except:
        return False


def display_dashboard():
    """Display the monitoring dashboard"""
    clear_screen()

    print("=" * 80)
    print("  QUANTUM CHILDREN - PROP FARM DASHBOARD")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    print("  This is READ-ONLY monitoring. Does NOT affect trades.")
    print("  Run individual BRAIN scripts to trade each account.")
    print()
    print("-" * 80)

    for key, acc in ACCOUNTS.items():
        print(f"\n  [{key}] {acc['name']}")
        print(f"  Account: {acc['account']} | Initial: ${acc['initial']:,}")
        print(f"  Log: {acc['log']}")
        print("  Recent activity:")

        lines = get_last_log_lines(acc['log'], 3)
        for line in lines:
            # Truncate long lines
            if len(line) > 70:
                line = line[:67] + "..."
            print(f"    {line}")

    print()
    print("-" * 80)
    print()
    print("  SCRIPTS TO RUN (each in its own window):")
    print()
    print("    python BRAIN_BG_INSTANT.py    → 366604 BlueGuardian Instant")
    print("    python BRAIN_BG_CHALLENGE.py  → 365060 BlueGuardian Challenge")
    print("    python BRAIN_ATLAS.py         → 212000584 Atlas")
    print("    python BRAIN_GETLEVERAGED.py  → GetLeveraged accounts")
    print()
    print("  Press Ctrl+C to exit dashboard")
    print("=" * 80)


def check_collection_server():
    """Check if collection server is receiving data"""
    try:
        import requests
        response = requests.get('http://203.161.61.61:8888/stats', timeout=5)
        if response.status_code == 200:
            stats = response.json()
            return stats
    except:
        pass
    return None


def main():
    print("Starting Prop Farm Dashboard...")
    print("Reading logs only - NOT affecting trades")
    print()

    try:
        while True:
            display_dashboard()

            # Check collection server
            stats = check_collection_server()
            if stats:
                print(f"\n  COLLECTION SERVER: ONLINE")
                print(f"    Signals: {stats.get('total_signals', 0)}")
                print(f"    Nodes: {stats.get('active_nodes', 0)}")
            else:
                print(f"\n  COLLECTION SERVER: OFFLINE or unreachable")

            print(f"\n  Refreshing in 10 seconds...")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")


if __name__ == "__main__":
    main()
