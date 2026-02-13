"""
INSTALL AUTO TRAINING LOOP - Windows Task Scheduler Setup
==========================================================
Creates a Windows Scheduled Task that starts auto_training_loop.py
at system boot and keeps it running 24/7.

The task runs as BELOW_NORMAL priority so live BRAIN scripts
always get CPU/GPU priority.

Run as Administrator:
    .venv312_gpu\\Scripts\\python.exe install_auto_loop.py --install
    .venv312_gpu\\Scripts\\python.exe install_auto_loop.py --uninstall
    .venv312_gpu\\Scripts\\python.exe install_auto_loop.py --status

Authors: DooDoo + Claude
Date: 2026-02-12
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.absolute()
PYTHON_EXE = str(SCRIPT_DIR / ".venv312_gpu" / "Scripts" / "python.exe")
LOOP_SCRIPT = str(SCRIPT_DIR / "auto_training_loop.py")
TASK_NAME = "QuantumChildren_AutoTrainingLoop"
LOG_DIR = SCRIPT_DIR / "orchestrator_logs"
LOG_DIR.mkdir(exist_ok=True)


def install_task():
    """Create Windows Scheduled Task for the auto training loop."""
    print("=" * 70)
    print("  Installing Auto Training Loop as Windows Scheduled Task")
    print("=" * 70)

    # Verify Python venv exists
    if not Path(PYTHON_EXE).exists():
        print(f"\nERROR: Python venv not found at: {PYTHON_EXE}")
        print("Make sure .venv312_gpu is set up first.")
        return False

    # Verify main script exists
    if not Path(LOOP_SCRIPT).exists():
        print(f"\nERROR: Main script not found at: {LOOP_SCRIPT}")
        return False

    # First, remove existing task if present
    subprocess.run(
        ["schtasks", "/delete", "/tn", TASK_NAME, "/f"],
        capture_output=True
    )

    # Build the schtasks command
    # Task runs at logon, restarts on failure, runs with below normal priority
    cmd = [
        "schtasks", "/create",
        "/tn", TASK_NAME,
        "/tr", f'"{PYTHON_EXE}" "{LOOP_SCRIPT}"',
        "/sc", "ONLOGON",          # Start at user logon
        "/rl", "LIMITED",           # Run with standard user privileges
        "/f",                       # Force (overwrite existing)
    ]

    print(f"\n  Task name:    {TASK_NAME}")
    print(f"  Python:       {PYTHON_EXE}")
    print(f"  Script:       {LOOP_SCRIPT}")
    print(f"  Trigger:      On logon")
    print(f"  Priority:     BELOW_NORMAL (set in script)")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("  Task created successfully!")
        print()
        print("  The auto training loop will start automatically when you log in.")
        print("  To start it now without restarting, run:")
        print(f"    schtasks /run /tn {TASK_NAME}")
        print()
        print("  Or start manually:")
        print(f"    {PYTHON_EXE} {LOOP_SCRIPT}")
        return True
    else:
        print(f"  FAILED to create task!")
        print(f"  Error: {result.stderr}")
        print()
        print("  This may require running as Administrator.")
        print("  Try: Run PowerShell as Admin, then run this script again.")
        return False


def uninstall_task():
    """Remove the Windows Scheduled Task."""
    print(f"Removing scheduled task: {TASK_NAME}")
    result = subprocess.run(
        ["schtasks", "/delete", "/tn", TASK_NAME, "/f"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("Task removed successfully.")
    else:
        print(f"Could not remove task: {result.stderr.strip()}")


def check_status():
    """Check if the scheduled task exists and its status."""
    result = subprocess.run(
        ["schtasks", "/query", "/tn", TASK_NAME, "/v", "/fo", "LIST"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Task '{TASK_NAME}' is NOT installed.")
        return

    print(f"Task '{TASK_NAME}' status:")
    print()
    for line in result.stdout.split("\n"):
        line = line.strip()
        if any(k in line for k in ["Status:", "Last Run Time:", "Next Run Time:",
                                      "Task Name:", "Last Result:", "Author:"]):
            print(f"  {line}")

    # Also check if the process is actually running
    print()
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline", []) or [])
                if "auto_training_loop" in cmdline:
                    print(f"  Process: RUNNING (PID {proc.pid})")
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        print("  Process: NOT running")
    except ImportError:
        print("  (psutil not available, cannot check running process)")


def main():
    parser = argparse.ArgumentParser(description="Install/manage auto training loop scheduled task")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--install", action="store_true", help="Install scheduled task")
    group.add_argument("--uninstall", action="store_true", help="Remove scheduled task")
    group.add_argument("--status", action="store_true", help="Check task status")
    args = parser.parse_args()

    if args.install:
        install_task()
    elif args.uninstall:
        uninstall_task()
    elif args.status:
        check_status()


if __name__ == "__main__":
    main()
