"""
MASTER LAUNCH — Quantum Children Automated Challenge Orchestrator
=================================================================
Starts all enabled accounts, quantum server, and watchdogs.
Monitors processes and auto-restarts on crash.
Run once. Walks away.

Usage:
    .venv312_gpu\Scripts\python.exe MASTER_LAUNCH.py
    .venv312_gpu\Scripts\python.exe MASTER_LAUNCH.py --accounts ATLAS FTMO
    .venv312_gpu\Scripts\python.exe MASTER_LAUNCH.py --dry-run

Author: DooDoo + Claude
Date: 2026-02-11
"""

import subprocess
import time
import json
import sys
import os
import signal
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------

BASE_DIR = Path(__file__).parent
PYTHON = str(BASE_DIR / ".venv312_gpu" / "Scripts" / "python.exe")
CONFIG_PATH = BASE_DIR / "MASTER_CONFIG.json"
LOG_DIR = BASE_DIR / "orchestrator_logs"
LOG_DIR.mkdir(exist_ok=True)

# Map account keys to their BRAIN scripts
BRAIN_SCRIPTS = {
    "BG_INSTANT":   "BRAIN_BG_INSTANT.py",
    "BG_CHALLENGE": "BRAIN_BG_CHALLENGE.py",
    "ATLAS":        "BRAIN_ATLAS.py",
    "GL_1":         "BRAIN_GETLEVERAGED.py",
    "GL_2":         "BRAIN_GETLEVERAGED.py",
    "GL_3":         "BRAIN_GETLEVERAGED.py",
    "FTMO":         "BRAIN_FTMO.py",
}

# Watchdog config per account
WATCHDOG_LIMIT = 1.50  # Force-close above this loss

# How often to check process health (seconds)
HEALTH_CHECK_INTERVAL = 30

# Max restarts before giving up on a process
MAX_RESTARTS = 5

# Cooldown between restarts (seconds)
RESTART_COOLDOWN = 60

# ---------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------

log_file = LOG_DIR / f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ORCHESTRATOR] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# PROCESS MANAGER
# ---------------------------------------------------------------

class ManagedProcess:
    """Tracks a subprocess with restart logic."""

    def __init__(self, name: str, cmd: List[str], cwd: str):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.process: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.last_restart: Optional[datetime] = None
        self.started_at: Optional[datetime] = None
        self.log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def start(self):
        """Start the process."""
        if self.process and self.process.poll() is None:
            log.warning(f"{self.name}: Already running (PID {self.process.pid})")
            return

        log_handle = open(str(self.log_file), "a", encoding="utf-8")
        self.process = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        self.started_at = datetime.now()
        log.info(f"{self.name}: Started (PID {self.process.pid}) -> {self.log_file.name}")

    def is_alive(self) -> bool:
        """Check if process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self):
        """Gracefully stop the process."""
        if self.process and self.is_alive():
            log.info(f"{self.name}: Stopping (PID {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            log.info(f"{self.name}: Stopped")

    def restart(self) -> bool:
        """Restart the process with cooldown and max restart logic."""
        if self.restart_count >= MAX_RESTARTS:
            log.error(f"{self.name}: Max restarts ({MAX_RESTARTS}) reached. Giving up.")
            return False

        if self.last_restart:
            elapsed = (datetime.now() - self.last_restart).total_seconds()
            if elapsed < RESTART_COOLDOWN:
                wait = RESTART_COOLDOWN - elapsed
                log.info(f"{self.name}: Cooldown, waiting {wait:.0f}s before restart...")
                time.sleep(wait)

        self.stop()
        self.restart_count += 1
        self.last_restart = datetime.now()
        log.warning(f"{self.name}: Restarting (attempt {self.restart_count}/{MAX_RESTARTS})")
        self.start()
        return True

    def uptime(self) -> str:
        """Human-readable uptime."""
        if not self.started_at:
            return "not started"
        delta = datetime.now() - self.started_at
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"

    def status(self) -> str:
        """One-line status string."""
        if not self.process:
            return f"{self.name}: NOT STARTED"
        if self.is_alive():
            return f"{self.name}: RUNNING (PID {self.process.pid}, uptime {self.uptime()}, restarts {self.restart_count})"
        else:
            rc = self.process.returncode
            return f"{self.name}: DEAD (exit code {rc}, restarts {self.restart_count})"


# ---------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------

class Orchestrator:
    """Master process orchestrator for Quantum Children."""

    def __init__(self, accounts: List[str] = None, dry_run: bool = False):
        self.dry_run = dry_run
        self.processes: Dict[str, ManagedProcess] = {}
        self.running = True

        # Load config
        with open(str(CONFIG_PATH), "r") as f:
            self.config = json.load(f)

        # Determine which accounts to run
        all_accounts = self.config.get("ACCOUNTS", {})
        if accounts:
            self.accounts = {k: v for k, v in all_accounts.items() if k in accounts}
        else:
            self.accounts = {k: v for k, v in all_accounts.items() if v.get("enabled", False)}

        log.info(f"Accounts to launch: {list(self.accounts.keys())}")

    def setup_processes(self):
        """Create managed processes for all components."""
        cwd = str(BASE_DIR)

        # 1. Quantum compression server
        self.processes["quantum_server"] = ManagedProcess(
            name="quantum_server",
            cmd=[PYTHON, str(BASE_DIR / "quantum_server.py")],
            cwd=cwd,
        )

        # 2. BRAIN script per account
        for account_key, account_cfg in self.accounts.items():
            script = BRAIN_SCRIPTS.get(account_key)
            if not script:
                log.warning(f"No BRAIN script mapped for {account_key}, skipping")
                continue

            script_path = BASE_DIR / script
            if not script_path.exists():
                log.warning(f"{script} not found, skipping {account_key}")
                continue

            self.processes[f"brain_{account_key}"] = ManagedProcess(
                name=f"brain_{account_key}",
                cmd=[PYTHON, str(script_path)],
                cwd=cwd,
            )

        # 3. Watchdog per account
        for account_key in self.accounts:
            self.processes[f"watchdog_{account_key}"] = ManagedProcess(
                name=f"watchdog_{account_key}",
                cmd=[
                    PYTHON,
                    str(BASE_DIR / "STOPLOSS_WATCHDOG_V2.py"),
                    "--account", account_key,
                    "--limit", str(WATCHDOG_LIMIT),
                ],
                cwd=cwd,
            )

        log.info(f"Total processes to manage: {len(self.processes)}")

    def launch_all(self):
        """Start all processes in order."""
        if self.dry_run:
            log.info("=== DRY RUN — not starting processes ===")
            for name, proc in self.processes.items():
                log.info(f"  Would start: {name} -> {' '.join(proc.cmd)}")
            return

        # Start quantum server first (needs a moment to initialize)
        if "quantum_server" in self.processes:
            log.info("Starting quantum compression server...")
            self.processes["quantum_server"].start()
            time.sleep(5)  # Give it time to bind the port

        # Start watchdogs before brains (safety first)
        for name, proc in self.processes.items():
            if name.startswith("watchdog_"):
                proc.start()
                time.sleep(1)

        # Start brain scripts
        for name, proc in self.processes.items():
            if name.startswith("brain_"):
                proc.start()
                time.sleep(3)  # Stagger starts to avoid MT5 login collisions

        log.info("=== ALL PROCESSES LAUNCHED ===")
        self.print_status()

    def health_check(self):
        """Check all processes and restart any that died."""
        dead = []
        for name, proc in self.processes.items():
            if not proc.is_alive() and proc.process is not None:
                dead.append(name)

        for name in dead:
            proc = self.processes[name]
            log.warning(f"DEAD PROCESS DETECTED: {proc.status()}")
            if not proc.restart():
                log.error(f"ABANDONING {name} after {MAX_RESTARTS} restart attempts")

    def print_status(self):
        """Print status dashboard."""
        log.info("=" * 60)
        log.info("ORCHESTRATOR STATUS REPORT")
        log.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info("-" * 60)
        for name, proc in self.processes.items():
            log.info(f"  {proc.status()}")
        log.info("=" * 60)

    def shutdown(self):
        """Gracefully stop all processes."""
        log.info("SHUTDOWN REQUESTED — stopping all processes...")
        self.running = False
        for name, proc in reversed(list(self.processes.items())):
            proc.stop()
        log.info("All processes stopped. Orchestrator exiting.")

    def run(self):
        """Main loop — launch, monitor, restart."""
        self.setup_processes()
        self.launch_all()

        if self.dry_run:
            return

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        status_interval = 300  # Print full status every 5 minutes
        last_status = time.time()

        log.info(f"Monitoring {len(self.processes)} processes (check every {HEALTH_CHECK_INTERVAL}s)...")

        while self.running:
            try:
                time.sleep(HEALTH_CHECK_INTERVAL)
                self.health_check()

                # Periodic status report
                if time.time() - last_status > status_interval:
                    self.print_status()
                    last_status = time.time()

            except KeyboardInterrupt:
                self.shutdown()
                break
            except Exception as e:
                log.error(f"Orchestrator error: {e}")
                time.sleep(5)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantum Children Master Orchestrator")
    parser.add_argument("--accounts", nargs="+", help="Specific accounts to launch (default: all enabled)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be launched without starting")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("QUANTUM CHILDREN — MASTER ORCHESTRATOR")
    log.info(f"Python: {PYTHON}")
    log.info(f"Config: {CONFIG_PATH}")
    log.info(f"Log dir: {LOG_DIR}")
    log.info("=" * 60)

    orchestrator = Orchestrator(accounts=args.accounts, dry_run=args.dry_run)
    orchestrator.run()
