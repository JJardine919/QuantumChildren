"""
QUANTUM COMMAND CENTER - V1.1
=============================
The \"Vending Machine\" for Quantum Children.
Use this to Train, Deploy, and Monitor your experts.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("\033[95m" + "="*60)
    print("      DEEPCOMPRESS.PRO // QUANTUM COMMAND CENTER")
    print("="*60 + "\033[0m")

def run_script(path, name):
    print(f"\n[SYSTEM] Launching {name}...")
    # Using Start-Process to keep them running in separate windows
    cmd = f"Start-Process python -ArgumentList '{path}'"
    subprocess.run(["powershell", "-Command", cmd])
    print(f"[OK] {name} is running.")
    time.sleep(2)

def train_new_child():
    clear_screen()
    print_header()
    print("\n--- QUANTUM VENDING MACHINE: TRAINING ---")
    print("This will take the latest archived market states and train a new Expert.")
    confirm = input("\nProceed with training? (y/n): ")
    if confirm.lower() == 'y':
        run_script("01_Systems/System_03_ETARE/ETARE_Redux.py", "ETARE Trainer")
    else:
        print("Training cancelled.")
    input("\nPress Enter to return to menu...")

def main_menu():
    while True:
        clear_screen()
        print_header()
        print("\n[1] TRAIN NEW EXPERT (Spit out a Quantum Child)")
        print("[2] START TRADING BRIDGE (Python -> MT5)")
        print("[3] START MARKET ARCHIVER (Background Capture)")
        print("[4] START ORDER PURGER (Clean stale orders)")
        print("[5] OPEN VISUAL MONITOR (Dashboard)")
        print("[6] CHECK SYSTEM STATUS (Log Audit)")
        print("[Q] EXIT")
        
        choice = input("\nSELECT ACTION > ").upper()
        
        if choice == '1':
            train_new_child()
        elif choice == '2':
            run_script("06_Integration/HybridBridge/etare_signal_generator.py", "Signal Bridge")
        elif choice == '3':
            run_script("01_Systems/QuantumCompression/run_archiver_service.py", "Archiver Service")
        elif choice == '4':
            run_script("order_purger.py", "Order Purger")
        elif choice == '5':
            os.startfile("quantum_monitor.html")
        elif choice == '6':
            print("\n--- RECENT LOGS ---")
            if os.path.exists("etare_gen.log"):
                os.system("powershell -Command Get-Content etare_gen.log -Tail 10")
            else:
                print("No logs found yet.")
            input("\nPress Enter to continue...")
        elif choice == 'Q':
            print("Shutting down Command Center...")
            break
        else:
            print("Invalid Selection.")
            time.sleep(1)

if __name__ == "__main__":
    main_menu()