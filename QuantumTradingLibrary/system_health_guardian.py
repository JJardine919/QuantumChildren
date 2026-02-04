import sqlite3
import shutil
import os
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
DB_FILES = ["trading_history.db", "etare_redux.db", "etare_redux_v2.db"]
BACKUP_DIR = Path("04_Data/Archive/Backups")
LOG_DIR = Path("04_Data/Archive/Logs")

def setup_dirs():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def check_db_integrity(db_path):
    """Checks if the SQLite database is valid and not corrupted."""
    print(f"Checking Integrity: {db_path}...", end=" ", flush=True)
    if not os.path.exists(db_path):
        print("MISSING ❌")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # The 'integrity_check' pragma is the standard way to verify a database file
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()[0]
        conn.close()
        
        if result == "ok":
            print("HEALTHY ✅")
            return True
        else:
            print(f"CORRUPT ❌ ({result})")
            return False
    except Exception as e:
        print(f"ERROR ❌ ({e})")
        return False

def perform_backup(db_path):
    """Creates a timestamped backup of the database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_name = Path(db_path).name
    backup_path = BACKUP_DIR / f"{db_name}.{timestamp}.bak"
    
    print(f"Backing up {db_name} -> {backup_path.name}...", end=" ", flush=True)
    try:
        shutil.copy2(db_path, backup_path)
        print("DONE 💾")
        return True
    except Exception as e:
        print(f"FAILED ❌ ({e})")
        return False

def rotate_backups(max_backups=5):
    """Keeps only the most recent N backups per database to save space."""
    print(f"Rotating backups (Keeping latest {max_backups})...")
    for db_name in DB_FILES:
        backups = sorted(BACKUP_DIR.glob(f"{db_name}.*.bak"), key=os.path.getmtime, reverse=True)
        if len(backups) > max_backups:
            for old_backup in backups[max_backups:]:
                print(f"  Cleaning old backup: {old_backup.name}")
                old_backup.unlink()

def main():
    print("==================================================")
    print("       SYSTEM HEALTH GUARDIAN: DATABASE CHECK")
    print("==================================================")
    
    setup_dirs()
    
    any_issues = False
    for db in DB_FILES:
        is_healthy = check_db_integrity(db)
        if is_healthy:
            perform_backup(db)
        else:
            any_issues = True
            
    print("-" * 50)
    rotate_backups()
    
    print("-" * 50)
    if any_issues:
        print("⚠️ WARNING: Some databases need attention. Check logs above.")
    else:
        print("✨ SYSTEM STATUS: ALL CLEAR. PROCEED TO TRADING.")
    print("==================================================")

if __name__ == "__main__":
    main()
