"""
SAFE SHUTDOWN - QUANTUM CHILDREN
=================================
Gracefully terminates all trading processes and clears lockfiles.

DOES NOT CLOSE OPEN TRADES - only stops the automation.

Usage:
    python SAFE_SHUTDOWN.py
    python SAFE_SHUTDOWN.py --force  (skip confirmation)
    python SAFE_SHUTDOWN.py --dry-run  (show what would be terminated)

Author: Biskits + DooDoo
Date: 2026-02-11
"""

import sys
import time
import logging
import argparse
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("ERROR: psutil is required for SAFE_SHUTDOWN")
    print("Install: pip install psutil")
    sys.exit(1)

# Process names to terminate
PROCESS_NAMES = [
    "BRAIN_ATLAS.py",
    "BRAIN_BG_INSTANT.py",
    "BRAIN_BG_CHALLENGE.py",
    "BRAIN_GETLEVERAGED.py",
    "BRAIN_FTMO.py",
    "STOPLOSS_WATCHDOG_V2.py",
    "STOPLOSS_WATCHDOG.py",
    "quantum_server.py",
    "MASTER_LAUNCH.py",
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def find_trading_processes():
    """
    Find all trading-related Python processes.

    Returns:
        List of (pid, cmdline) tuples
    """
    found = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            proc_name = proc.info.get('name', '')
            if not proc_name or 'python' not in proc_name.lower():
                continue

            cmdline = proc.info.get('cmdline', [])
            if not cmdline:
                continue

            cmdline_str = ' '.join(cmdline)

            # Check if any of our target scripts are in the command line
            if any(name in cmdline_str for name in PROCESS_NAMES):
                found.append((proc.info['pid'], cmdline_str))

        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            pass

    return found


def terminate_process(pid: int, timeout: int = 10):
    """
    Gracefully terminate a process.

    Args:
        pid: Process ID to terminate
        timeout: Seconds to wait before force-killing
    """
    try:
        proc = psutil.Process(pid)
        log.info(f"Terminating PID {pid}...")

        # Try graceful termination first
        proc.terminate()

        try:
            proc.wait(timeout=timeout)
            log.info(f"PID {pid} terminated successfully")
        except psutil.TimeoutExpired:
            log.warning(f"PID {pid} did not terminate gracefully, killing forcefully...")
            proc.kill()
            proc.wait(timeout=5)
            log.info(f"PID {pid} killed")

    except psutil.NoSuchProcess:
        log.warning(f"PID {pid} already terminated")
    except psutil.AccessDenied:
        log.error(f"Access denied to PID {pid} - may need admin privileges")
    except Exception as e:
        log.error(f"Error terminating PID {pid}: {e}")


def clear_lockfiles():
    """Clear all process lockfiles."""
    from process_lock import clear_all_locks
    try:
        count = clear_all_locks()
        log.info(f"Cleared {count} lockfile(s)")
    except Exception as e:
        log.error(f"Failed to clear lockfiles: {e}")


def main():
    parser = argparse.ArgumentParser(description="Safe Shutdown for Quantum Children")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be terminated without doing it")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SAFE SHUTDOWN - QUANTUM CHILDREN")
    log.info("=" * 60)

    # Find all trading processes
    processes = find_trading_processes()

    if not processes:
        log.info("No trading processes found running.")
        if not args.dry_run:
            clear_lockfiles()
        return

    # Display found processes
    log.info(f"Found {len(processes)} trading process(es):")
    for pid, cmdline in processes:
        # Truncate command line for display
        display_cmd = cmdline if len(cmdline) < 100 else cmdline[:97] + "..."
        log.info(f"  PID {pid}: {display_cmd}")

    if args.dry_run:
        log.info("DRY RUN - no processes terminated")
        return

    # Confirmation
    if not args.force:
        print("\n" + "=" * 60)
        print("WARNING: This will terminate all trading processes.")
        print("Open trades will NOT be closed automatically.")
        print("=" * 60)
        confirm = input("\nType YES to proceed: ")
        if confirm.strip().upper() != "YES":
            log.info("Shutdown cancelled by user")
            return

    # Terminate all processes
    log.info("\nTerminating processes...")
    for pid, _ in processes:
        terminate_process(pid)

    # Wait a moment for cleanup
    time.sleep(2)

    # Verify all terminated
    remaining = find_trading_processes()
    if remaining:
        log.warning(f"\n{len(remaining)} process(es) still running:")
        for pid, cmdline in remaining:
            display_cmd = cmdline if len(cmdline) < 100 else cmdline[:97] + "..."
            log.warning(f"  PID {pid}: {display_cmd}")
        log.warning("\nYou may need to manually terminate these processes.")
    else:
        log.info("\nAll processes terminated successfully")

    # Clear lockfiles
    clear_lockfiles()

    log.info("\n" + "=" * 60)
    log.info("SHUTDOWN COMPLETE")
    log.info("Safe to restart using MASTER_LAUNCH.py")
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nShutdown interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
