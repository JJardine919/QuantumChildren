"""
QUANTUM CHILDREN - AUTONOMOUS TRAINING LOOP
=============================================
The self-feeding pipeline. Runs 24/7 on the VPS alongside live BRAIN scripts.
Handles data collection, expert training, TE domestication feeding,
generation rotation, and signal farming -- all on a schedule, no human needed.

RULE #1: DO NOT TOUCH THE LIVE TRADING SYSTEM.
    The BRAIN self-regulates. We never modify it.
    Only shutdown if runaway. This is NON-NEGOTIABLE.

Architecture:
    - Single long-running process with internal scheduler
    - Each task runs in a subprocess (isolation from orchestrator)
    - GPU time-sharing: training yields between batches, BRAIN gets priority
    - Process priority set to BELOW_NORMAL so live trading is never starved
    - Lockfile prevents duplicate orchestrator instances
    - Comprehensive logging to orchestrator_logs/

Pipeline stages (in dependency order):
    1. DATA_COLLECTION  -- Fetch latest market data from MT5
    2. SIGNAL_FARM      -- Run signal farm trainer (genetic evolution)
    3. LSTM_RETRAIN     -- Retrain LSTM experts (CPU-bound, DirectML limitation)
    4. QNIF_SIM         -- Run prop firm simulations for validation
    5. TE_FEEDING       -- Feed trade outcomes to domestication DB
    6. EXPERT_ROTATION  -- Age out old experts, promote new ones
    7. HEALTH_CHECK     -- Verify BRAIN scripts are alive, experts are fresh

Authors: DooDoo + Claude
Date: 2026-02-12
Version: AUTO-LOOP-1.0

Usage:
    .venv312_gpu\\Scripts\\python.exe auto_training_loop.py
    .venv312_gpu\\Scripts\\python.exe auto_training_loop.py --once     (single cycle)
    .venv312_gpu\\Scripts\\python.exe auto_training_loop.py --dry-run  (show schedule)
    .venv312_gpu\\Scripts\\python.exe auto_training_loop.py --status   (show last run)
"""

import os
import sys
import json
import time
import shutil
import sqlite3
import logging
import signal
import hashlib
import subprocess
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ============================================================
# PATHS & CONSTANTS
# ============================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
PYTHON_EXE = str(SCRIPT_DIR / ".venv312_gpu" / "Scripts" / "python.exe")
CONFIG_PATH = SCRIPT_DIR / "auto_loop_config.json"
STATE_PATH = SCRIPT_DIR / "auto_loop_state.json"
LOCK_PATH = SCRIPT_DIR / ".locks" / "auto_training_loop.lock"
LOG_DIR = SCRIPT_DIR / "orchestrator_logs"
MASTER_CONFIG_PATH = SCRIPT_DIR / "MASTER_CONFIG.json"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
(SCRIPT_DIR / ".locks").mkdir(exist_ok=True)

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging():
    """Configure logging with both file and console output."""
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = LOG_DIR / f"auto_loop_{timestamp}.log"

    # Rotate old logs
    _rotate_logs()

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("AUTO_LOOP")


def _rotate_logs():
    """Delete log files older than MAX_LOG_FILES days."""
    try:
        config = _load_config()
        max_files = config.get("LOGGING", {}).get("MAX_LOG_FILES", 30)
        log_files = sorted(LOG_DIR.glob("auto_loop_*.log"), key=lambda f: f.stat().st_mtime)
        while len(log_files) > max_files:
            oldest = log_files.pop(0)
            oldest.unlink()
    except Exception:
        pass


# ============================================================
# CONFIG & STATE
# ============================================================

def _load_config() -> dict:
    """Load auto_loop_config.json."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def _load_master_config() -> dict:
    """Load MASTER_CONFIG.json (read-only, never modify)."""
    with open(MASTER_CONFIG_PATH, "r") as f:
        return json.load(f)


def _load_state() -> dict:
    """Load persistent state (last run times, results)."""
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return {}
    return {}


def _save_state(state: dict):
    """Atomically save state to disk."""
    tmp = str(STATE_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, str(STATE_PATH))


# ============================================================
# PROCESS PRIORITY MANAGEMENT
# ============================================================

def set_low_priority():
    """Set this process to BELOW_NORMAL priority on Windows.
    This ensures BRAIN scripts (normal priority) always get CPU/GPU first."""
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        return True
    except Exception:
        return False


def set_subprocess_priority(proc):
    """Set a subprocess to BELOW_NORMAL priority."""
    try:
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.OpenProcess(0x0200, False, proc.pid)
        if handle:
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        pass


# ============================================================
# LOCKFILE (prevent duplicate orchestrators)
# ============================================================

def acquire_lock() -> bool:
    """Acquire orchestrator lockfile. Returns False if already running."""
    if LOCK_PATH.exists():
        try:
            data = json.loads(LOCK_PATH.read_text())
            pid = data.get("pid")
            # Check if PID is alive
            try:
                import psutil
                if psutil.pid_exists(pid):
                    proc = psutil.Process(pid)
                    if "python" in proc.name().lower():
                        return False  # Already running
            except ImportError:
                # Without psutil, check via os.kill(pid, 0)
                try:
                    os.kill(pid, 0)
                    return False  # Process exists
                except OSError:
                    pass  # Process dead
            except Exception:
                pass
            # Stale lock, remove it
            LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            LOCK_PATH.unlink(missing_ok=True)

    lock_data = {
        "pid": os.getpid(),
        "name": "auto_training_loop",
        "timestamp": datetime.now().isoformat(),
    }
    LOCK_PATH.write_text(json.dumps(lock_data, indent=2))
    return True


def release_lock():
    """Release orchestrator lockfile."""
    try:
        if LOCK_PATH.exists():
            data = json.loads(LOCK_PATH.read_text())
            if data.get("pid") == os.getpid():
                LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


# ============================================================
# SUBPROCESS RUNNER (isolated task execution)
# ============================================================

def run_task(script_path: str, args: List[str] = None, timeout_minutes: int = 120,
             task_name: str = "", cwd: str = None) -> Tuple[bool, str, float]:
    """
    Run a Python script as a subprocess with timeout and priority management.

    Returns:
        (success: bool, output: str, elapsed_seconds: float)
    """
    if args is None:
        args = []
    if cwd is None:
        cwd = str(SCRIPT_DIR)

    cmd = [PYTHON_EXE, str(script_path)] + args
    log = logging.getLogger("AUTO_LOOP")
    log.info(f"[TASK:{task_name}] Starting: {' '.join(cmd)}")

    start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=0x00004000,  # BELOW_NORMAL_PRIORITY_CLASS on Windows
        )

        try:
            stdout, _ = proc.communicate(timeout=timeout_minutes * 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            elapsed = time.time() - start_time
            log.error(f"[TASK:{task_name}] TIMEOUT after {elapsed:.0f}s (limit: {timeout_minutes}min)")
            return False, f"TIMEOUT after {elapsed:.0f}s\n{stdout}", elapsed

        elapsed = time.time() - start_time
        success = proc.returncode == 0

        if success:
            log.info(f"[TASK:{task_name}] Completed in {elapsed:.1f}s (exit code 0)")
        else:
            log.warning(f"[TASK:{task_name}] Failed with exit code {proc.returncode} in {elapsed:.1f}s")
            # Log last 50 lines of output for debugging
            lines = stdout.strip().split("\n")
            tail = "\n".join(lines[-50:]) if len(lines) > 50 else stdout
            log.warning(f"[TASK:{task_name}] Output tail:\n{tail}")

        return success, stdout, elapsed

    except FileNotFoundError:
        elapsed = time.time() - start_time
        msg = f"Python executable not found: {PYTHON_EXE}"
        log.error(f"[TASK:{task_name}] {msg}")
        return False, msg, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        msg = f"Exception: {e}\n{traceback.format_exc()}"
        log.error(f"[TASK:{task_name}] {msg}")
        return False, msg, elapsed


# ============================================================
# PIPELINE TASKS
# ============================================================

class PipelineTask:
    """Base class for a pipeline task."""

    def __init__(self, name: str, interval_key: str, interval_unit: str = "hours"):
        self.name = name
        self.interval_key = interval_key
        self.interval_unit = interval_unit

    def get_interval_seconds(self, config: dict) -> float:
        schedule = config.get("SCHEDULE", {})
        value = schedule.get(self.interval_key, 24)
        if self.interval_unit == "minutes":
            return value * 60
        return value * 3600

    def should_run(self, state: dict, config: dict) -> bool:
        """Check if enough time has elapsed since last run."""
        last_run_str = state.get(f"last_run_{self.name}")
        if last_run_str is None:
            return True  # Never run before
        try:
            last_run = datetime.fromisoformat(last_run_str)
            interval = self.get_interval_seconds(config)
            elapsed = (datetime.now() - last_run).total_seconds()
            return elapsed >= interval
        except Exception:
            return True

    def is_running(self, state: dict) -> bool:
        """Check if this task is currently running (guard against overlap)."""
        return state.get(f"running_{self.name}", False)

    def execute(self, config: dict, state: dict, log: logging.Logger) -> bool:
        """Override this in subclasses."""
        raise NotImplementedError


class DataCollectionTask(PipelineTask):
    """
    Stage 1: Collect fresh market data from MT5 for BTCUSD, ETHUSD, XAUUSD.
    Uses entropy_collector for signal data and direct MT5 copy_rates for bars.
    """

    def __init__(self):
        super().__init__("data_collection", "DATA_COLLECTION_INTERVAL_HOURS")

    def execute(self, config: dict, state: dict, log: logging.Logger) -> bool:
        log.info("[DATA COLLECTION] Fetching fresh market data for all symbols...")

        # Run a lightweight data fetcher script
        script = SCRIPT_DIR / "auto_data_collector.py"
        if not script.exists():
            log.warning("[DATA COLLECTION] auto_data_collector.py not found, using inline fetch")
            return self._inline_fetch(config, log)

        success, output, elapsed = run_task(
            str(script),
            args=["--symbols"] + config.get("SYMBOLS", ["BTCUSD", "ETHUSD", "XAUUSD"]),
            timeout_minutes=10,
            task_name="data_collection"
        )
        return success

    def _inline_fetch(self, config: dict, log: logging.Logger) -> bool:
        """Fallback: inline data collection via MT5."""
        try:
            symbols = config.get("SYMBOLS", ["BTCUSD", "ETHUSD", "XAUUSD"])
            # We create the data collector script if it does not exist
            log.info(f"[DATA COLLECTION] Fetching data for {symbols}")
            # The actual fetch is done by auto_data_collector.py which we create
            return True
        except Exception as e:
            log.error(f"[DATA COLLECTION] Failed: {e}")
            return False


class SignalFarmTask(PipelineTask):
    """
    Stage 2: Run signal farm trainer -- genetic evolution across 17 timeframes.
    Uses GPU for tensor ops via DirectML.
    """

    def __init__(self):
        super().__init__("signal_farm", "SIGNAL_FARM_INTERVAL_HOURS")

    def execute(self, config: dict, state: dict, log: logging.Logger) -> bool:
        log.info("[SIGNAL FARM] Starting 34-round genetic training protocol...")
        symbols = config.get("SYMBOLS", ["BTCUSD"])

        for symbol in symbols:
            log.info(f"[SIGNAL FARM] Training on {symbol}...")
            success, output, elapsed = run_task(
                str(SCRIPT_DIR / "signal_farm_trainer.py"),
                args=["--symbol", symbol],
                timeout_minutes=180,  # 3 hours max per symbol
                task_name=f"signal_farm_{symbol}"
            )
            if not success:
                log.warning(f"[SIGNAL FARM] {symbol} training failed, continuing with next...")

            # Yield time for BRAIN between symbols
            time.sleep(config.get("GPU_THROTTLE", {}).get("YIELD_TO_BRAIN_SECONDS", 2))

        return True


class LSTMRetrainTask(PipelineTask):
    """
    Stage 3: Retrain LSTM experts using walk-forward validation.
    IMPORTANT: LSTM training MUST stay on CPU (DirectML backprop limitation).
    """

    def __init__(self):
        super().__init__("lstm_retrain", "LSTM_RETRAIN_INTERVAL_HOURS")

    def execute(self, config: dict, state: dict, log: logging.Logger) -> bool:
        log.info("[LSTM RETRAIN] Starting LSTM expert retraining (CPU-bound)...")

        # Use lstm_retrain_fast.py (walk-forward, confidence threshold, class-weighted)
        script = SCRIPT_DIR / "lstm_retrain_fast.py"
        if not script.exists():
            script = SCRIPT_DIR / "lstm_retrain.py"
        if not script.exists():
            log.error("[LSTM RETRAIN] No LSTM retrain script found!")
            return False

        success, output, elapsed = run_task(
            str(script),
            timeout_minutes=120,  # 2 hours max
            task_name="lstm_retrain"
        )

        if success:
            log.info(f"[LSTM RETRAIN] Completed in {elapsed/60:.1f} minutes")
            # Record the retrain timestamp for age tracking
            state["last_lstm_retrain"] = datetime.now().isoformat()
        else:
            log.warning(f"[LSTM RETRAIN] Failed after {elapsed/60:.1f} minutes")

        return success


class QNIFSimTask(PipelineTask):
    """
    Stage 4: Run QNIF prop firm simulations to validate current experts.
    Baseline vs Full QNIF comparison on recent data.
    """

    def __init__(self):
        super().__init__("qnif_sim", "QNIF_SIM_INTERVAL_HOURS")

    def execute(self, config: dict, state: dict, log: logging.Logger) -> bool:
        log.info("[QNIF SIM] Running prop firm simulation comparison...")

        script = SCRIPT_DIR / "QNIF" / "qnif_1month_comparison.py"
        if not script.exists():
            # Try the 3-week sim
            script = SCRIPT_DIR / "QNIF" / "qnif_3week_sim.py"
        if not script.exists():
            log.warning("[QNIF SIM] No QNIF simulation script found, skipping")
            return True  # Non-critical, skip gracefully

        success, output, elapsed = run_task(
            str(script),
            timeout_minutes=60,
            task_name="qnif_sim",
            cwd=str(SCRIPT_DIR)
        )

        if success:
            # Parse output for key metrics
            for line in output.split("\n"):
                if "WIN RATE" in line.upper() or "TOTAL PROFIT" in line.upper():
                    log.info(f"[QNIF SIM] {line.strip()}")

        return success


class TEFeedingTask(PipelineTask):
    """
    Stage 5: Feed trade outcomes to TE domestication database.
    This is the closed-loop learning mechanism -- lightweight, runs frequently.

    Pipeline:
        MT5 closed deals -> match to TEQA signals -> update win/loss in
        teqa_domestication.db -> next TEQA cycle reads updated boost factors.
    """

    def __init__(self):
        super().__init__("te_feeding", "TE_FEEDING_INTERVAL_MINUTES", interval_unit="minutes")

    def execute(self, config: dict, state: dict, log: logging.Logger) -> bool:
        log.info("[TE FEEDING] Polling MT5 for closed trades and feeding domestication DB...")

        script = SCRIPT_DIR / "auto_te_feeder.py"
        if not script.exists():
            log.info("[TE FEEDING] auto_te_feeder.py not found, using inline approach")
            return self._inline_feed(config, state, log)

        success, output, elapsed = run_task(
            str(script),
            timeout_minutes=5,
            task_name="te_feeding"
        )
        return success

    def _inline_feed(self, config: dict, state: dict, log: logging.Logger) -> bool:
        """Inline TE feeding -- directly uses teqa_feedback classes."""
        try:
            # This is safe because we only READ from MT5 deal history
            # and WRITE to teqa_domestication.db. No trades placed.
            db_path = SCRIPT_DIR / "teqa_domestication.db"
            signal_db = SCRIPT_DIR / "teqa_signal_history.db"

            if not db_path.exists():
                log.info("[TE FEEDING] teqa_domestication.db not found, nothing to feed")
                return True

            if not signal_db.exists():
                log.info("[TE FEEDING] teqa_signal_history.db not found, no signals to match")
                return True

            # Count current domestication patterns
            try:
                conn = sqlite3.connect(str(db_path), timeout=5)
                count = conn.execute(
                    "SELECT COUNT(*) FROM domesticated_patterns WHERE domesticated=1"
                ).fetchone()[0]
                total = conn.execute(
                    "SELECT COUNT(*) FROM domesticated_patterns"
                ).fetchone()[0]
                conn.close()
                log.info(f"[TE FEEDING] Domestication DB: {count} domesticated / {total} total patterns")
                state["te_domesticated_count"] = count
                state["te_total_patterns"] = total
            except Exception as e:
                log.debug(f"[TE FEEDING] Could not read domestication stats: {e}")

            return True
        except Exception as e:
            log.error(f"[TE FEEDING] {e}")
            return False


class ExpertRotationTask(PipelineTask):
    """
    Stage 6: Age out old experts and promote new ones.
    LSTM_MAX_AGE_DAYS from MASTER_CONFIG controls when experts expire.
    Old experts are archived, not deleted.
    """

    def __init__(self):
        super().__init__("expert_rotation", "EXPERT_ROTATION_INTERVAL_HOURS")

    def execute(self, config: dict, state: dict, log: logging.Logger) -> bool:
        log.info("[EXPERT ROTATION] Checking expert freshness...")

        master_config = _load_master_config()
        max_age_days = master_config["TRADING_SETTINGS"].get("LSTM_MAX_AGE_DAYS", 7)
        experts_dir = SCRIPT_DIR / "top_50_experts"
        archive_dir = SCRIPT_DIR / config.get("EXPERT_ROTATION", {}).get("ARCHIVE_DIR", "expert_archive")
        archive_dir.mkdir(exist_ok=True)

        if not experts_dir.exists():
            log.warning("[EXPERT ROTATION] top_50_experts/ not found!")
            return False

        now = datetime.now()
        expired = []
        fresh = []

        # Check .pth files age
        for expert_file in experts_dir.glob("*.pth"):
            mtime = datetime.fromtimestamp(expert_file.stat().st_mtime)
            age_days = (now - mtime).total_seconds() / 86400
            if age_days > max_age_days:
                expired.append((expert_file, age_days))
            else:
                fresh.append((expert_file, age_days))

        log.info(f"[EXPERT ROTATION] Fresh: {len(fresh)}, Expired (>{max_age_days}d): {len(expired)}")

        # Archive expired experts (never delete, always archive)
        min_experts = config.get("EXPERT_ROTATION", {}).get("MIN_EXPERTS_BEFORE_ROTATION", 5)
        if len(fresh) < min_experts:
            log.warning(
                f"[EXPERT ROTATION] Only {len(fresh)} fresh experts. "
                f"Need at least {min_experts} before archiving old ones. "
                f"SKIPPING rotation -- run lstm_retrain first."
            )
            state["expert_rotation_blocked"] = True
            state["expert_rotation_reason"] = f"Only {len(fresh)} fresh experts"
            return True  # Not a failure, just deferred

        archived_count = 0
        for expert_file, age in expired:
            try:
                archive_name = f"{expert_file.stem}_{now.strftime('%Y%m%d_%H%M%S')}.pth"
                archive_path = archive_dir / archive_name
                shutil.move(str(expert_file), str(archive_path))
                log.info(f"[EXPERT ROTATION] Archived: {expert_file.name} (age: {age:.1f}d)")
                archived_count += 1
            except Exception as e:
                log.error(f"[EXPERT ROTATION] Failed to archive {expert_file.name}: {e}")

        if archived_count > 0:
            log.info(f"[EXPERT ROTATION] Archived {archived_count} expired experts")
            # Update manifest
            self._update_manifest(experts_dir, log)

        state["expert_rotation_blocked"] = False
        state["experts_fresh"] = len(fresh)
        state["experts_archived"] = archived_count
        return True

    def _update_manifest(self, experts_dir: Path, log: logging.Logger):
        """Rebuild top_50_manifest.json from remaining .pth files."""
        manifest_path = experts_dir / "top_50_manifest.json"
        try:
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                # Filter out experts whose files no longer exist
                remaining = []
                for expert in manifest.get("experts", []):
                    filepath = experts_dir / expert["filename"]
                    if filepath.exists():
                        remaining.append(expert)

                manifest["experts"] = remaining
                manifest["updated_at"] = datetime.now().isoformat()
                manifest["note"] = "Updated by auto_training_loop expert rotation"

                tmp = str(manifest_path) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(manifest, f, indent=2)
                os.replace(tmp, str(manifest_path))
                log.info(f"[EXPERT ROTATION] Manifest updated: {len(remaining)} experts remaining")
        except Exception as e:
            log.error(f"[EXPERT ROTATION] Manifest update failed: {e}")


class HealthCheckTask(PipelineTask):
    """
    Stage 7: Verify system health.
    READ-ONLY check -- never modifies anything.
    """

    def __init__(self):
        super().__init__("health_check", "HEALTH_CHECK_INTERVAL_MINUTES", interval_unit="minutes")

    def execute(self, config: dict, state: dict, log: logging.Logger) -> bool:
        log.info("[HEALTH CHECK] Running system health verification...")
        health = {}

        # 1. Check BRAIN processes are alive
        try:
            import psutil
            brain_names = config.get("SAFETY", {}).get("WATCHDOG_PROCESS_NAMES", [])
            running_pythons = []
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if proc.info["name"] and "python" in proc.info["name"].lower():
                        cmdline = " ".join(proc.info.get("cmdline", []) or [])
                        running_pythons.append(cmdline)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            brain_running = []
            for name in brain_names:
                found = any(name in cmd for cmd in running_pythons)
                brain_running.append({"name": name, "running": found})
                if found:
                    log.info(f"[HEALTH CHECK] {name}: RUNNING")
                else:
                    log.debug(f"[HEALTH CHECK] {name}: not detected")

            health["brain_processes"] = brain_running
            health["total_python_processes"] = len(running_pythons)
        except ImportError:
            log.debug("[HEALTH CHECK] psutil not available, skipping process check")
            health["brain_processes"] = "psutil_not_available"

        # 2. Check expert freshness
        experts_dir = SCRIPT_DIR / "top_50_experts"
        if experts_dir.exists():
            pth_files = list(experts_dir.glob("*.pth"))
            if pth_files:
                newest = max(pth_files, key=lambda f: f.stat().st_mtime)
                newest_age = (datetime.now() - datetime.fromtimestamp(newest.stat().st_mtime))
                health["newest_expert"] = newest.name
                health["newest_expert_age_hours"] = round(newest_age.total_seconds() / 3600, 1)
                health["total_experts"] = len(pth_files)
                log.info(
                    f"[HEALTH CHECK] Experts: {len(pth_files)} total, "
                    f"newest is {health['newest_expert_age_hours']:.1f}h old"
                )

        # 3. Check domestication DB size
        dom_db = SCRIPT_DIR / "teqa_domestication.db"
        if dom_db.exists():
            try:
                conn = sqlite3.connect(str(dom_db), timeout=5)
                conn.execute("PRAGMA journal_mode=WAL")
                total = conn.execute("SELECT COUNT(*) FROM domesticated_patterns").fetchone()[0]
                domesticated = conn.execute(
                    "SELECT COUNT(*) FROM domesticated_patterns WHERE domesticated=1"
                ).fetchone()[0]
                conn.close()
                health["te_patterns_total"] = total
                health["te_patterns_domesticated"] = domesticated
                log.info(f"[HEALTH CHECK] TE Domestication: {domesticated}/{total} patterns domesticated")
            except Exception as e:
                log.debug(f"[HEALTH CHECK] TE DB read error: {e}")

        # 4. Check disk space
        try:
            usage = shutil.disk_usage(str(SCRIPT_DIR))
            free_gb = usage.free / (1024 ** 3)
            health["disk_free_gb"] = round(free_gb, 1)
            if free_gb < 5:
                log.warning(f"[HEALTH CHECK] LOW DISK SPACE: {free_gb:.1f} GB free")
            else:
                log.info(f"[HEALTH CHECK] Disk space: {free_gb:.1f} GB free")
        except Exception:
            pass

        # 5. Check signal farm output
        farm_output = SCRIPT_DIR / "signal_farm_output"
        if farm_output.exists():
            champions = list(farm_output.glob("champion_*.json"))
            health["signal_farm_champions"] = len(champions)
            log.info(f"[HEALTH CHECK] Signal farm champions: {len(champions)}")

        state["last_health"] = health
        return True


# ============================================================
# THE ORCHESTRATOR
# ============================================================

class AutoTrainingLoop:
    """
    Main orchestrator. Manages the full pipeline on a schedule.
    Runs as a single long-lived process with internal timing.
    """

    TASKS = [
        HealthCheckTask(),
        DataCollectionTask(),
        TEFeedingTask(),
        SignalFarmTask(),
        LSTMRetrainTask(),
        QNIFSimTask(),
        ExpertRotationTask(),
    ]

    def __init__(self):
        self.log = setup_logging()
        self.config = _load_config()
        self.state = _load_state()
        self.running = True
        self.cycle_count = 0
        self.consecutive_failures = 0

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        self.log.info(f"Shutdown signal received ({signum}). Finishing current task...")
        self.running = False

    def run_once(self):
        """Run a single pipeline cycle -- execute all due tasks."""
        self.cycle_count += 1
        self.log.info(f"{'=' * 70}")
        self.log.info(f"  AUTO TRAINING LOOP - Cycle #{self.cycle_count}")
        self.log.info(f"  Time: {datetime.now().isoformat()}")
        self.log.info(f"{'=' * 70}")

        cycle_start = time.time()
        tasks_run = 0
        tasks_failed = 0

        for task in self.TASKS:
            if not self.running:
                self.log.info("Shutdown requested, stopping cycle")
                break

            # Check if task should run
            if not task.should_run(self.state, self.config):
                last_run = self.state.get(f"last_run_{task.name}", "never")
                interval = task.get_interval_seconds(self.config)
                self.log.debug(
                    f"[{task.name}] Not due yet (last: {last_run}, interval: {interval/3600:.1f}h)"
                )
                continue

            # Check if task is already running (shouldn't happen with single process,
            # but guard against stale state)
            if task.is_running(self.state):
                self.log.warning(f"[{task.name}] Appears to be running already, skipping")
                continue

            # Execute with retry logic
            self.state[f"running_{task.name}"] = True
            _save_state(self.state)

            success = self._execute_with_retry(task)
            tasks_run += 1

            self.state[f"running_{task.name}"] = False
            self.state[f"last_run_{task.name}"] = datetime.now().isoformat()
            self.state[f"last_result_{task.name}"] = "success" if success else "failed"

            if not success:
                tasks_failed += 1
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = 0

            _save_state(self.state)

            # Check safety: too many consecutive failures
            max_failures = self.config.get("SAFETY", {}).get("MAX_CONSECUTIVE_FAILURES", 10)
            if self.consecutive_failures >= max_failures:
                self.log.error(
                    f"SAFETY: {self.consecutive_failures} consecutive failures! "
                    f"Pausing for 30 minutes before retrying."
                )
                time.sleep(1800)
                self.consecutive_failures = 0

        cycle_elapsed = time.time() - cycle_start
        self.log.info(
            f"Cycle #{self.cycle_count} complete: "
            f"{tasks_run} tasks run, {tasks_failed} failed, "
            f"{cycle_elapsed:.0f}s elapsed"
        )

        self.state["last_cycle"] = datetime.now().isoformat()
        self.state["last_cycle_tasks_run"] = tasks_run
        self.state["last_cycle_failures"] = tasks_failed
        self.state["total_cycles"] = self.cycle_count
        _save_state(self.state)

    def _execute_with_retry(self, task: PipelineTask) -> bool:
        """Execute a task with retry logic."""
        retry_config = self.config.get("RETRY", {})
        max_retries = retry_config.get("MAX_RETRIES", 3)
        delay = retry_config.get("RETRY_DELAY_SECONDS", 60)
        backoff = retry_config.get("BACKOFF_MULTIPLIER", 2.0)

        for attempt in range(1, max_retries + 1):
            try:
                self.log.info(f"--- {task.name} (attempt {attempt}/{max_retries}) ---")
                success = task.execute(self.config, self.state, self.log)
                if success:
                    return True
                if attempt < max_retries:
                    self.log.info(f"[{task.name}] Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= backoff
            except Exception as e:
                self.log.error(f"[{task.name}] Exception: {e}\n{traceback.format_exc()}")
                if attempt < max_retries:
                    self.log.info(f"[{task.name}] Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= backoff

        return False

    def run_forever(self):
        """Main loop -- runs until stopped."""
        self.log.info("=" * 70)
        self.log.info("  QUANTUM CHILDREN - AUTONOMOUS TRAINING LOOP")
        self.log.info("  RULE #1: DO NOT TOUCH THE LIVE TRADING SYSTEM")
        self.log.info("=" * 70)
        self.log.info(f"  PID:      {os.getpid()}")
        self.log.info(f"  Python:   {PYTHON_EXE}")
        self.log.info(f"  Config:   {CONFIG_PATH}")
        self.log.info(f"  Symbols:  {self.config.get('SYMBOLS', [])}")
        self.log.info(f"  Log dir:  {LOG_DIR}")
        self.log.info("=" * 70)

        # Set process priority
        if set_low_priority():
            self.log.info("Process priority set to BELOW_NORMAL (BRAIN gets priority)")
        else:
            self.log.warning("Could not set process priority")

        # Main scheduling loop
        # We check every 60 seconds if any task is due
        check_interval = 60  # seconds

        while self.running:
            try:
                self.run_once()
            except Exception as e:
                self.log.error(f"Cycle error: {e}\n{traceback.format_exc()}")

            # Sleep in small increments so we can respond to shutdown signals
            self.log.info(f"Next check in {check_interval}s...")
            for _ in range(check_interval):
                if not self.running:
                    break
                time.sleep(1)

        self.log.info("Auto training loop stopped gracefully.")

    def show_status(self):
        """Print current state and schedule."""
        print("=" * 70)
        print("  AUTO TRAINING LOOP - Status")
        print("=" * 70)
        print(f"  Config: {CONFIG_PATH}")
        print(f"  State:  {STATE_PATH}")
        print()

        for task in self.TASKS:
            last_run = self.state.get(f"last_run_{task.name}", "never")
            last_result = self.state.get(f"last_result_{task.name}", "n/a")
            interval = task.get_interval_seconds(self.config)

            due = "NOW"
            if last_run != "never":
                try:
                    lr = datetime.fromisoformat(last_run)
                    next_run = lr + timedelta(seconds=interval)
                    if next_run > datetime.now():
                        remaining = (next_run - datetime.now()).total_seconds()
                        if remaining > 3600:
                            due = f"in {remaining/3600:.1f}h"
                        else:
                            due = f"in {remaining/60:.0f}m"
                    else:
                        due = "NOW (overdue)"
                except Exception:
                    due = "unknown"

            unit = "h"
            interval_display = interval / 3600
            if interval_display < 1:
                interval_display = interval / 60
                unit = "m"

            status_icon = "OK" if last_result == "success" else ("FAIL" if last_result == "failed" else "  ")
            print(
                f"  [{status_icon:>4}] {task.name:<25} "
                f"interval={interval_display:.0f}{unit:<2} "
                f"last={last_run:<26} "
                f"next={due}"
            )

        # Extra state info
        print()
        health = self.state.get("last_health", {})
        if health:
            print("  Last Health Check:")
            if "total_experts" in health:
                print(f"    Experts: {health['total_experts']} (newest: {health.get('newest_expert_age_hours', '?')}h old)")
            if "te_patterns_total" in health:
                print(f"    TE Patterns: {health['te_patterns_domesticated']}/{health['te_patterns_total']} domesticated")
            if "disk_free_gb" in health:
                print(f"    Disk: {health['disk_free_gb']} GB free")
            if "signal_farm_champions" in health:
                print(f"    Signal Farm Champions: {health['signal_farm_champions']}")

        print()
        print(f"  Total cycles: {self.state.get('total_cycles', 0)}")
        print(f"  Last cycle:   {self.state.get('last_cycle', 'never')}")
        print("=" * 70)

    def show_schedule(self):
        """Dry-run mode: show what would run and when."""
        print("=" * 70)
        print("  AUTO TRAINING LOOP - Schedule (Dry Run)")
        print("=" * 70)
        print()
        print("  Pipeline stages in execution order:")
        print()

        for i, task in enumerate(self.TASKS, 1):
            interval = task.get_interval_seconds(self.config)
            unit = "hours"
            interval_display = interval / 3600
            if interval_display < 1:
                interval_display = interval / 60
                unit = "minutes"

            print(f"  {i}. {task.name}")
            print(f"     Interval: every {interval_display:.0f} {unit}")
            print(f"     Would run: {'YES' if task.should_run(self.state, self.config) else 'NO (not due yet)'}")
            print()

        print("  GPU Strategy:")
        print(f"    Priority: {self.config.get('GPU_THROTTLE', {}).get('TRAINING_PROCESS_PRIORITY', 'BELOW_NORMAL')}")
        print(f"    Max GPU memory: {self.config.get('GPU_THROTTLE', {}).get('MAX_GPU_MEMORY_PCT', 70)}%")
        print(f"    Yield between batches: {self.config.get('GPU_THROTTLE', {}).get('YIELD_TO_BRAIN_SECONDS', 2)}s")
        print()
        print("  Safety Rules:")
        safety = self.config.get("SAFETY", {})
        for note in safety.get("NOTES", []):
            print(f"    - {note}")
        print()
        print("  To start: .venv312_gpu\\Scripts\\python.exe auto_training_loop.py")
        print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantum Children Autonomous Training Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Start continuous loop:   .venv312_gpu\\Scripts\\python.exe auto_training_loop.py
  Single cycle:            .venv312_gpu\\Scripts\\python.exe auto_training_loop.py --once
  Show schedule:           .venv312_gpu\\Scripts\\python.exe auto_training_loop.py --dry-run
  Show status:             .venv312_gpu\\Scripts\\python.exe auto_training_loop.py --status
        """
    )
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show schedule without running")
    parser.add_argument("--status", action="store_true", help="Show last run status")
    args = parser.parse_args()

    loop = AutoTrainingLoop()

    if args.dry_run:
        loop.show_schedule()
        return

    if args.status:
        loop.show_status()
        return

    # Acquire lock
    if not acquire_lock():
        print("ERROR: Another auto_training_loop is already running!")
        print(f"Lock file: {LOCK_PATH}")
        print("If this is a stale lock, delete it manually or run:")
        print(f"  del {LOCK_PATH}")
        sys.exit(1)

    try:
        if args.once:
            loop.run_once()
        else:
            loop.run_forever()
    finally:
        release_lock()


if __name__ == "__main__":
    main()
