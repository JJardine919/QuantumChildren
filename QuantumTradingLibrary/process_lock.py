"""
PROCESS LOCK MANAGER
====================
Prevents duplicate process launches using PID lockfiles.

Each BRAIN/watchdog/server creates a lockfile before starting.
If lockfile exists and PID is alive, launch is BLOCKED.
If lockfile exists but PID is dead (stale), lockfile is auto-removed.

Usage:
    from process_lock import ProcessLock

    lock = ProcessLock("BRAIN_ATLAS", account="212000584")
    with lock:
        # Your process code here
        # Lock automatically released on exit

Author: Biskits + DooDoo
Date: 2026-02-11
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - lockfile stale detection will be limited")

LOCK_DIR = Path(__file__).parent / ".locks"
LOCK_DIR.mkdir(exist_ok=True)

log = logging.getLogger(__name__)


class ProcessLock:
    """PID-based process lockfile manager."""

    def __init__(self, name: str, account: Optional[str] = None):
        """
        Create a process lock.

        Args:
            name: Lock name (e.g., "BRAIN_ATLAS", "quantum_server")
            account: Optional account identifier (e.g., "212000584")
        """
        self.name = name
        self.account = account
        self.lock_file = LOCK_DIR / f"{name}.lock"
        self.acquired = False

    def acquire(self) -> bool:
        """
        Acquire lock. Returns True if successful, False if already locked.

        If lockfile exists but process is dead, auto-removes stale lock.
        """
        # Check existing lock
        if self.lock_file.exists():
            if self._is_lock_stale():
                log.warning(f"Removing stale lockfile: {self.lock_file.name}")
                self._remove_stale_lock()
            else:
                # Lock held by running process
                return False

        # Create lock
        lock_data = {
            "pid": os.getpid(),
            "name": self.name,
            "account": self.account,
            "timestamp": datetime.now().isoformat(),
            "host": os.getenv("COMPUTERNAME", "unknown"),
        }

        try:
            self.lock_file.write_text(json.dumps(lock_data, indent=2))
            self.acquired = True
            log.info(f"Acquired lock: {self.lock_file.name} (PID {os.getpid()})")
            return True
        except Exception as e:
            log.error(f"Failed to create lockfile {self.lock_file}: {e}")
            return False

    def release(self):
        """Release lock by deleting lockfile."""
        if self.acquired and self.lock_file.exists():
            try:
                self.lock_file.unlink()
                log.info(f"Released lock: {self.lock_file.name}")
                self.acquired = False
            except Exception as e:
                log.error(f"Failed to release lock {self.lock_file}: {e}")

    def _is_lock_stale(self) -> bool:
        """
        Check if existing lock is stale (process dead).

        Returns True if:
        - Lockfile is corrupt
        - PID doesn't exist
        - PID exists but is not a Python process
        """
        try:
            data = json.loads(self.lock_file.read_text())
            pid = data.get("pid")

            if not pid:
                return True  # No PID in lockfile

            if not PSUTIL_AVAILABLE:
                # Without psutil, we can't reliably check if PID is alive
                # Conservative approach: assume lock is valid
                log.warning(f"Cannot verify PID {pid} (psutil not available)")
                return False

            # Check if PID exists
            if not psutil.pid_exists(pid):
                return True  # Process dead

            # Check if it's actually a Python process
            try:
                proc = psutil.Process(pid)
                proc_name = proc.name().lower()
                if "python" not in proc_name:
                    return True  # PID reused by non-Python process
                return False  # Process alive
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return True  # Process dead or inaccessible

        except Exception as e:
            log.warning(f"Error checking lock staleness: {e}")
            return True  # Corrupt lockfile

    def _remove_stale_lock(self):
        """Remove stale lockfile."""
        try:
            self.lock_file.unlink()
        except Exception as e:
            log.error(f"Failed to remove stale lock {self.lock_file}: {e}")

    def get_lock_info(self) -> Optional[dict]:
        """Get info about existing lock. Returns None if no lock exists."""
        if not self.lock_file.exists():
            return None
        try:
            return json.loads(self.lock_file.read_text())
        except Exception as e:
            log.error(f"Failed to read lockfile {self.lock_file}: {e}")
            return None

    def __enter__(self):
        """Context manager entry - acquire lock or raise RuntimeError."""
        if not self.acquire():
            info = self.get_lock_info()
            if info:
                raise RuntimeError(
                    f"Process lock '{self.name}' is held by PID {info.get('pid')} "
                    f"(started at {info.get('timestamp')}). "
                    f"Cannot start duplicate process. "
                    f"Run SAFE_SHUTDOWN.bat to stop all processes safely."
                )
            else:
                raise RuntimeError(
                    f"Process lock '{self.name}' exists but cannot be read. "
                    f"Manual intervention required: delete {self.lock_file}"
                )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release lock."""
        self.release()


def list_active_locks() -> list:
    """
    List all active lockfiles with their info.

    Returns:
        List of (filename, lock_data) tuples
    """
    locks = []
    for lock_file in LOCK_DIR.glob("*.lock"):
        try:
            data = json.loads(lock_file.read_text())
            locks.append((lock_file.name, data))
        except Exception:
            locks.append((lock_file.name, None))
    return locks


def clear_all_locks():
    """
    Clear all lockfiles.

    WARNING: Only use this if you're CERTAIN no processes are running.
    """
    count = 0
    for lock_file in LOCK_DIR.glob("*.lock"):
        try:
            lock_file.unlink()
            count += 1
        except Exception as e:
            log.error(f"Failed to delete {lock_file}: {e}")
    log.info(f"Cleared {count} lockfile(s)")
    return count


def clear_stale_locks():
    """
    Clear only stale lockfiles (where PID is dead).

    Returns:
        Number of stale locks cleared
    """
    if not PSUTIL_AVAILABLE:
        log.warning("psutil not available - cannot auto-clear stale locks")
        return 0

    count = 0
    for lock_file in LOCK_DIR.glob("*.lock"):
        try:
            data = json.loads(lock_file.read_text())
            pid = data.get("pid")

            if not pid or not psutil.pid_exists(pid):
                log.info(f"Clearing stale lock: {lock_file.name} (PID {pid})")
                lock_file.unlink()
                count += 1
            else:
                # Verify it's a Python process
                try:
                    proc = psutil.Process(pid)
                    if "python" not in proc.name().lower():
                        log.info(f"Clearing stale lock: {lock_file.name} (PID {pid} reused)")
                        lock_file.unlink()
                        count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    log.info(f"Clearing stale lock: {lock_file.name} (PID {pid} dead)")
                    lock_file.unlink()
                    count += 1
        except Exception as e:
            log.error(f"Error processing {lock_file}: {e}")

    log.info(f"Cleared {count} stale lockfile(s)")
    return count


if __name__ == "__main__":
    # CLI utility for lock management
    import argparse

    parser = argparse.ArgumentParser(description="Process Lock Manager")
    parser.add_argument("--list", action="store_true", help="List all active locks")
    parser.add_argument("--clear-stale", action="store_true", help="Clear stale locks only")
    parser.add_argument("--clear-all", action="store_true", help="Clear ALL locks (use with caution)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.list:
        locks = list_active_locks()
        if locks:
            print(f"\n{len(locks)} active lockfile(s):")
            for filename, data in locks:
                if data:
                    print(f"  {filename}:")
                    print(f"    PID: {data.get('pid')}")
                    print(f"    Account: {data.get('account', 'N/A')}")
                    print(f"    Started: {data.get('timestamp')}")
                    print(f"    Host: {data.get('host', 'unknown')}")
                else:
                    print(f"  {filename}: [CORRUPT]")
        else:
            print("\nNo active locks.")

    elif args.clear_stale:
        count = clear_stale_locks()
        print(f"\nCleared {count} stale lockfile(s).")

    elif args.clear_all:
        print("\nWARNING: This will clear ALL lockfiles.")
        print("Only do this if you're CERTAIN no processes are running.")
        confirm = input("Type YES to confirm: ")
        if confirm.strip().upper() == "YES":
            count = clear_all_locks()
            print(f"\nCleared {count} lockfile(s).")
        else:
            print("Cancelled.")

    else:
        parser.print_help()
