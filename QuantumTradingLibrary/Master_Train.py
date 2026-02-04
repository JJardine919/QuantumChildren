import os
import shutil
import subprocess
import time
import sys
import glob

# ============================================================ 
# MASTER TRAIN: ULTRA-TRAIN PROTOCOL (The "DooDoo Special")
# Strategy: 3 Symbols x 3 Timeframes -> Quantum Injection
# ============================================================ 

DIRS = {
    "RAW_DATA": "04_Data/QuantumStates",
    "COMPRESSOR_SRC": "01_Systems/QuantumCompression",
    "GYM_DATA": "09_User_Distribution/04_Sample_Data",
    "GYM_SCRIPT": "09_User_Distribution/02_The_Gym",
    "BRAIN_DEST": "09_User_Distribution/03_Brain"
}

# The "Big 3" Strategy
SYMBOLS = ["XAUUSD", "ETHUSD", "BTCUSD"]
TIMEFRAMES = ["M15", "H1", "H4"]

def log(msg):
    print(f"[MASTER] {msg}")

def step_1_find_data():
    log("Step 1: Locating latest raw market data...")
    files = glob.glob(os.path.join(DIRS["RAW_DATA"], "*_state.npy"))
    if not files:
        log("ERROR: No raw data found in 04_Data/QuantumStates.")
        return None
    latest_file = max(files, key=os.path.getctime)
    log(f"Found: {os.path.basename(latest_file)}")
    return latest_file

def step_2_compress(input_file):
    log("Step 2: Running Deep Quantum Compressor...")
    cmd = [sys.executable, "run_headless_compressor.py", input_file] 
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    expected_output = input_file.replace('.npy', '.dqcp.npz')
    if os.path.exists(expected_output):
        log(f"Compression Success: {os.path.basename(expected_output)}")
        return expected_output
    else:
        log("Compression failed or file missing.")
        return None

def step_3_move_to_gym(compressed_file):
    log("Step 3: Moving compressed state to The Gym...")
    dest = os.path.join(DIRS["GYM_DATA"], "new_market_state.dqcp.npz")
    try:
        shutil.copy(compressed_file, dest)
        return True
    except Exception as e:
        log(f"Move failed: {e}")
        return False

def step_4_bootcamp():
    log("\n[MASTER] === INITIATING BOOTCAMP (The Big 3) ===")
    cwd = os.getcwd()
    os.chdir(DIRS["GYM_SCRIPT"])
    
    try:
        session_count = 1
        total_sessions = len(SYMBOLS) * len(TIMEFRAMES)
        
        for sym in SYMBOLS:
            for tf in TIMEFRAMES:
                print(f"\n[MASTER] Session {session_count}/{total_sessions}: Training on {sym} ({tf})...")
                cmd = [sys.executable, "Trainer.py", "--mode", "base", "--symbol", sym, "--tf", tf]
                subprocess.run(cmd, check=True)
                session_count += 1
                time.sleep(0.5) # Brief pause between sessions
                
    except Exception as e:
        log(f"Bootcamp failed: {e}")
        return False
    finally:
        os.chdir(cwd)
    return True

def step_5_quantum_injection():
    log("\n[MASTER] === QUANTUM INJECTION PHASE ===")
    cwd = os.getcwd()
    os.chdir(DIRS["GYM_SCRIPT"])
    
    try:
        cmd = [sys.executable, "Trainer.py", "--mode", "quantum"]
        subprocess.run(cmd, check=True)
    except Exception as e:
        log(f"Quantum Injection failed: {e}")
        return False
    finally:
        os.chdir(cwd)
    return True

def main():
    print("========================================")
    print("   QUANTUM TRADING: ULTRA-TRAIN PROTOCOL")
    print("========================================")
    
    # 1. Data Prep & Compression (Background work)
    data_file = step_1_find_data()
    if not data_file: return
    
    compressed_file = step_2_compress(data_file)
    if not compressed_file: return
    
    if not step_3_move_to_gym(compressed_file): return
    
    # 2. The Bootcamp (Base Training)
    if not step_4_bootcamp(): return
    
    # 3. The Quantum Injection (Final Boost)
    if not step_5_quantum_injection(): return
    
    print("========================================")
    print("   PIPELINE COMPLETE. SYSTEM READY.")
    print("========================================")

if __name__ == "__main__":
    main()