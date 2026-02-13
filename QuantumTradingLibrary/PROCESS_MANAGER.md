# QUANTUM CHILDREN - PROCESS MANAGEMENT ARCHITECTURE
## Designed by Biskits - 2026-02-11

---

## THE PROBLEM

**Current chaos:**
- 3 copies of BRAIN_ATLAS.py running simultaneously
- All 3 connect to the SAME MT5 account (212000584)
- Result: Duplicate trades, race conditions, chaos

**Root cause:**
- No process lockfile system
- No duplicate launch prevention
- Batch files use `start` command with no checks
- MASTER_LAUNCH.py has weak duplicate checking (line 98-100 checks if process.poll() is None, but doesn't prevent external launches)

---

## THE SOLUTION - THREE-LAYER DEFENSE

### Layer 1: PID Lockfile System (CRITICAL PATH)
Each BRAIN/watchdog/server script creates a lockfile with its PID before connecting to MT5.

**File:** `process_lock.py` (new utility module)

**Lockfile format:** `.locks/BRAIN_ATLAS.lock` contains PID + account + timestamp

**Lock behavior:**
1. On startup, check if lockfile exists
2. If exists, read PID and verify process is actually running
3. If PID is stale (process dead), delete lock and proceed
4. If PID is alive, ABORT with error message
5. On shutdown, delete lockfile

**Lock scope:**
- `BRAIN_<ACCOUNT>.lock` - One BRAIN per account
- `WATCHDOG_<ACCOUNT>.lock` - One watchdog per account
- `quantum_server.lock` - One quantum server globally
- `MASTER_LAUNCH.lock` - One orchestrator

### Layer 2: Enhanced MASTER_LAUNCH.py
Already has process tracking. Enhancement needed:
- Check for existing lockfiles BEFORE starting processes
- Integrate with lockfile system to prevent external launches

### Layer 3: Batch File Safety Wrapper
Create `SAFE_START.bat` that:
1. Checks for lockfiles
2. Reports running processes
3. Asks user to confirm shutdown before launching
4. Uses MASTER_LAUNCH.py (not raw Python calls)

---

## IMPLEMENTATION PLAN

### Step 1: Create process_lock.py (Core Lockfile System)
```python
"""
Process lockfile manager - prevents duplicate launches.
Creates PID lockfiles in .locks/ directory.
"""

import os
import json
import psutil
from pathlib import Path
from typing import Optional
from datetime import datetime

LOCK_DIR = Path(__file__).parent / ".locks"
LOCK_DIR.mkdir(exist_ok=True)

class ProcessLock:
    def __init__(self, name: str, account: Optional[str] = None):
        self.name = name
        self.account = account
        self.lock_file = LOCK_DIR / f"{name}.lock"
        self.acquired = False

    def acquire(self) -> bool:
        """Acquire lock. Returns True if successful, False if already locked."""
        # Check existing lock
        if self.lock_file.exists():
            if self._is_lock_stale():
                self._remove_stale_lock()
            else:
                return False  # Lock held by running process

        # Create lock
        lock_data = {
            "pid": os.getpid(),
            "name": self.name,
            "account": self.account,
            "timestamp": datetime.now().isoformat(),
        }
        self.lock_file.write_text(json.dumps(lock_data, indent=2))
        self.acquired = True
        return True

    def release(self):
        """Release lock."""
        if self.acquired and self.lock_file.exists():
            self.lock_file.unlink()
            self.acquired = False

    def _is_lock_stale(self) -> bool:
        """Check if existing lock is stale (process dead)."""
        try:
            data = json.loads(self.lock_file.read_text())
            pid = data.get("pid")
            if pid and psutil.pid_exists(pid):
                # Check if it's actually a Python process
                try:
                    proc = psutil.Process(pid)
                    if "python" in proc.name().lower():
                        return False  # Process alive
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return True  # Stale lock
        except Exception:
            return True  # Corrupt lock file

    def _remove_stale_lock(self):
        """Remove stale lockfile."""
        try:
            self.lock_file.unlink()
        except Exception:
            pass

    def get_lock_info(self) -> Optional[dict]:
        """Get info about existing lock."""
        if not self.lock_file.exists():
            return None
        try:
            return json.loads(self.lock_file.read_text())
        except Exception:
            return None

    def __enter__(self):
        if not self.acquire():
            info = self.get_lock_info()
            raise RuntimeError(
                f"Process lock '{self.name}' is held by PID {info.get('pid')}. "
                f"Started at {info.get('timestamp')}. Cannot start duplicate process."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
```

### Step 2: Modify BRAIN_ATLAS.py (Add Lock on Startup)
Insert after imports, before MT5 connection:

```python
from process_lock import ProcessLock

# CRITICAL: Acquire process lock BEFORE connecting to MT5
lock = ProcessLock("BRAIN_ATLAS", account="212000584")
try:
    with lock:
        # All existing BRAIN code goes here (indented)
        # ...
except RuntimeError as e:
    logging.error(f"LOCK FAILURE: {e}")
    logging.error("Another instance of BRAIN_ATLAS is already running.")
    logging.error("Use SAFE_SHUTDOWN.bat to stop all processes safely.")
    sys.exit(1)
```

### Step 3: Modify STOPLOSS_WATCHDOG_V2.py (Add Lock)
Same pattern:
```python
lock = ProcessLock(f"WATCHDOG_{args.account}", account=args.account)
with lock:
    # Existing watchdog code
```

### Step 4: Modify quantum_server.py (Add Lock)
```python
lock = ProcessLock("quantum_server")
with lock:
    # Server code
```

### Step 5: Create SAFE_SHUTDOWN.bat
```batch
@echo off
echo ============================================================
echo   SAFE SHUTDOWN - QUANTUM CHILDREN
echo ============================================================
echo.

cd /d "%~dp0"

echo Checking for running processes...
.venv312_gpu\Scripts\python.exe -c "import subprocess; subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'check_processes.ps1'])"

echo.
echo WARNING: This will terminate all BRAIN, watchdog, and server processes.
echo Open trades will NOT be closed automatically.
echo.
set /p confirm="Type YES to shutdown all processes: "

if /i "%confirm%" NEQ "YES" (
    echo Shutdown cancelled.
    pause
    exit /b
)

echo.
echo Shutting down gracefully...
.venv312_gpu\Scripts\python.exe SAFE_SHUTDOWN.py

echo.
echo Shutdown complete. Lockfiles cleared.
pause
```

### Step 6: Create SAFE_SHUTDOWN.py
```python
"""
Safe shutdown utility - kills all trading processes and clears locks.
Does NOT close open trades.
"""

import psutil
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

PROCESS_NAMES = [
    "BRAIN_ATLAS.py",
    "BRAIN_BG_INSTANT.py",
    "BRAIN_BG_CHALLENGE.py",
    "BRAIN_GETLEVERAGED.py",
    "BRAIN_FTMO.py",
    "STOPLOSS_WATCHDOG_V2.py",
    "quantum_server.py",
    "MASTER_LAUNCH.py",
]

def find_trading_processes():
    """Find all trading-related Python processes."""
    found = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if any(name in cmdline for name in PROCESS_NAMES):
                    found.append((proc.info['pid'], cmdline))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found

def terminate_process(pid):
    """Gracefully terminate a process."""
    try:
        proc = psutil.Process(pid)
        logging.info(f"Terminating PID {pid}...")
        proc.terminate()
        proc.wait(timeout=10)
        logging.info(f"PID {pid} terminated successfully")
    except psutil.TimeoutExpired:
        logging.warning(f"PID {pid} did not terminate, killing forcefully...")
        proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

def clear_locks():
    """Clear all lockfiles."""
    lock_dir = Path(__file__).parent / ".locks"
    if lock_dir.exists():
        for lock_file in lock_dir.glob("*.lock"):
            logging.info(f"Removing lockfile: {lock_file.name}")
            lock_file.unlink()

def main():
    processes = find_trading_processes()

    if not processes:
        logging.info("No trading processes found running.")
        clear_locks()
        return

    logging.info(f"Found {len(processes)} trading processes:")
    for pid, cmdline in processes:
        logging.info(f"  PID {pid}: {cmdline}")

    logging.info("Terminating processes...")
    for pid, _ in processes:
        terminate_process(pid)

    time.sleep(2)

    # Verify all dead
    remaining = find_trading_processes()
    if remaining:
        logging.warning(f"{len(remaining)} processes still running:")
        for pid, cmdline in remaining:
            logging.warning(f"  PID {pid}: {cmdline}")
    else:
        logging.info("All processes terminated successfully")

    clear_locks()
    logging.info("Lockfiles cleared. Safe to restart.")

if __name__ == "__main__":
    main()
```

### Step 7: Enhanced MASTER_LAUNCH.py (Check Locks Before Launch)
Add at start of Orchestrator.setup_processes():

```python
def setup_processes(self):
    """Create managed processes for all components."""
    # Check for existing locks BEFORE starting anything
    from process_lock import ProcessLock, LOCK_DIR

    existing_locks = list(LOCK_DIR.glob("*.lock"))
    if existing_locks:
        log.warning("Found existing process locks:")
        for lock_file in existing_locks:
            try:
                data = json.loads(lock_file.read_text())
                log.warning(f"  {lock_file.name}: PID {data.get('pid')} ({data.get('timestamp')})")
            except:
                pass
        log.error("Cannot start: other processes are running.")
        log.error("Run SAFE_SHUTDOWN.bat first, or manually clear .locks/ directory.")
        sys.exit(1)

    # ... rest of existing setup code
```

---

## USAGE

### Starting the System
```batch
MASTER_LAUNCH.bat  (new wrapper for MASTER_LAUNCH.py)
```
This will:
- Check for existing locks
- Abort if processes running
- Start all enabled accounts with lockfile protection

### Stopping the System
```batch
SAFE_SHUTDOWN.bat
```
This will:
- List running processes
- Ask for confirmation
- Terminate all processes gracefully
- Clear lockfiles

### Manual Process Check
```batch
check_processes.ps1  (via PowerShell)
```

---

## MULTI-ACCOUNT ARCHITECTURE

Each account runs in ISOLATED process space:

```
ATLAS (212000584)
  ├─ BRAIN_ATLAS.py        [Lock: BRAIN_ATLAS.lock]
  ├─ WATCHDOG_V2 --account ATLAS  [Lock: WATCHDOG_ATLAS.lock]
  └─ Connects to: C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe

BG_INSTANT (366604)
  ├─ BRAIN_BG_INSTANT.py   [Lock: BRAIN_BG_INSTANT.lock]
  ├─ WATCHDOG_V2 --account BG_INSTANT
  └─ Connects to: C:\Program Files\Blue Guardian MT5 Terminal\terminal64.exe

BG_CHALLENGE (365060)
  ├─ BRAIN_BG_CHALLENGE.py [Lock: BRAIN_BG_CHALLENGE.lock]
  ├─ WATCHDOG_V2 --account BG_CHALLENGE
  └─ Connects to: C:\Program Files\Blue Guardian MT5 Terminal 2\terminal64.exe

GL_3 (107245)
  ├─ BRAIN_GETLEVERAGED.py [Lock: BRAIN_GL_3.lock]
  ├─ WATCHDOG_V2 --account GL_3
  └─ Connects to: GetLeveraged MT5 (needs terminal_path in config)

FTMO (1521063483)
  ├─ BRAIN_FTMO.py         [Lock: BRAIN_FTMO.lock]
  ├─ WATCHDOG_V2 --account FTMO
  └─ Connects to: C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe

GLOBAL
  └─ quantum_server.py     [Lock: quantum_server.lock]
```

**Each process:**
1. Acquires lockfile on startup
2. Connects to ONLY its designated MT5 terminal
3. Never switches accounts
4. Releases lock on clean shutdown

**Lock prevents:**
- Duplicate BRAIN launches for same account
- Multiple watchdogs monitoring same account
- Multiple quantum servers

---

## CRITICAL RULES

1. **NEVER manually run BRAIN_*.py directly** - Use MASTER_LAUNCH.py
2. **ALWAYS check locks before starting** - Use SAFE_SHUTDOWN.bat if needed
3. **ONE terminal per account** - No shared MT5 instances
4. **Lockfiles are source of truth** - If lock exists, process is running (or stale)
5. **Clean shutdown clears locks** - SAFE_SHUTDOWN.bat does this automatically

---

## EMERGENCY PROCEDURES

### "I have 3 BRAIN_ATLAS instances running!"
```batch
SAFE_SHUTDOWN.bat
```
Wait for all processes to terminate, then:
```batch
MASTER_LAUNCH.bat
```

### "Lockfile exists but process is dead"
The lockfile system auto-detects stale locks (checks if PID exists).
If a process crashes, the next launch will clean up the stale lock automatically.

Manual override (if needed):
```batch
del .locks\*.lock
```

### "I need to stop only ONE account, not all"
```powershell
# Find the PID
powershell -ExecutionPolicy Bypass -File check_processes.ps1

# Kill specific PID (e.g., 19248)
taskkill /PID 19248 /F

# Clear the lock manually
del .locks\BRAIN_ATLAS.lock
```

---

## FILES TO CREATE

1. `process_lock.py` - Core lockfile system
2. `SAFE_SHUTDOWN.py` - Process termination utility
3. `SAFE_SHUTDOWN.bat` - Batch wrapper
4. `MASTER_LAUNCH.bat` - Wrapper for MASTER_LAUNCH.py with lock check
5. Update `BRAIN_ATLAS.py` - Add lock acquisition
6. Update `STOPLOSS_WATCHDOG_V2.py` - Add lock acquisition
7. Update `quantum_server.py` - Add lock acquisition
8. Update `MASTER_LAUNCH.py` - Add lock checking before process creation

---

## TESTING PLAN

1. **Test stale lock cleanup**: Kill a BRAIN process without releasing lock, verify next launch cleans it up
2. **Test duplicate prevention**: Try to run BRAIN_ATLAS.py twice, verify second instance aborts
3. **Test SAFE_SHUTDOWN**: Start all accounts, run SAFE_SHUTDOWN.bat, verify clean termination
4. **Test MASTER_LAUNCH**: Verify it refuses to start if locks exist
5. **Test multi-account**: Start ATLAS + BG_INSTANT + BG_CHALLENGE simultaneously, verify no conflicts

---

**This design eliminates the duplicate process problem at the architectural level.**
**No more duplicate trades. No more race conditions. Bulletproof.**

— Biskits
