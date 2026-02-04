"""
QUANTUM ARCHIVER SERVICE
========================
Background service that periodically captures and archives market states.
Captures every 15 minutes.
"""

import time
import subprocess
import sys
from datetime import datetime

def run_service():
    print("="*60)
    print("QUANTUM ARCHIVER SERVICE STARTED")
    print(f"Start time: {datetime.now()}")
    print("Capturing state every 15 minutes...")
    print("="*60)
    
    try:
        while True:
            print(f"[{datetime.now()}] Initiating capture...")
            
            # Run the archiver script
            result = subprocess.run([sys.executable, "01_Systems/QuantumCompression/market_archiver.py"], 
                                    capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"ERROR: {result.stderr}")
            
            # Sleep for 15 minutes (900 seconds)
            print("Sleeping for 15m...")
            time.sleep(900)
            
    except KeyboardInterrupt:
        print("\nService stopped by user.")

if __name__ == "__main__":
    run_service()

