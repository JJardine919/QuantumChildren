#!/usr/bin/env python3
"""Switch from Docker Mock Mode to Native Live Trading"""

import paramiko
import time
import sys

VPS_HOST = "72.62.170.153"
VPS_USER = "root"
VPS_PASS = "gXRCBtbi21##"

def execute_command(ssh, command, timeout=30):
    """Execute a command on the VPS and return output"""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    output = stdout.read().decode('utf-8', errors='ignore')
    error = stderr.read().decode('utf-8', errors='ignore')
    exit_code = stdout.channel.recv_exit_status()
    return output, error, exit_code

def main():
    print("=" * 60)
    print("Switch to Live Trading Mode")
    print("=" * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"\nConnecting to VPS {VPS_HOST}...")
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=30)
        print("Connected!")

        # Step 1: Stop Docker container
        print("\n[Step 1] Stopping Docker mock container...")
        output, error, code = execute_command(ssh, "docker stop quantum-brain")
        print(f"Docker container stopped: {output}")

        time.sleep(2)

        # Step 2: Verify it's stopped
        print("\n[Step 2] Verifying container is stopped...")
        output, error, code = execute_command(ssh, "docker ps | grep quantum-brain || echo 'Container stopped'")
        print(output)

        # Step 3: Check if MetaTrader5 is available natively
        print("\n[Step 3] Checking native MT5 Python module...")
        output, error, code = execute_command(ssh,
            "python3 -c 'import MetaTrader5 as mt5; print(f\"MT5 module version: {mt5.__version__}\")' 2>&1")
        print(output)

        # Step 4: Verify .env configuration
        print("\n[Step 4] Verifying .env configuration for GetLeveraged...")
        output, error, code = execute_command(ssh, "cat /root/.env | grep MT5")
        print(output)

        # Step 5: Check if signal generator exists in QuantumTradingLibrary
        print("\n[Step 5] Locating signal generator...")
        output, error, code = execute_command(ssh,
            "ls -la /root/QuantumTradingLibrary/06_Integration/HybridBridge/etare_signal_generator_redux.py")
        print(output)

        # Step 6: Check for champions directory
        print("\n[Step 6] Checking for trained LSTM champions...")
        output, error, code = execute_command(ssh,
            "ls -la /root/QuantumTradingLibrary/champions/ 2>/dev/null | head -10 || echo 'No champions found'")
        print(output)

        # If no champions, copy from Docker
        if "No champions found" in output:
            print("\nCopying champions from Docker container...")
            output, error, code = execute_command(ssh,
                "docker cp quantum-brain:/app/champions /root/QuantumTradingLibrary/champions && ls -la /root/QuantumTradingLibrary/champions/")
            print(output)

        # Step 7: Start signal generator natively
        print("\n[Step 7] Starting signal generator in LIVE mode...")
        start_cmd = """
cd /root/QuantumTradingLibrary && \
export $(cat /root/.env | grep -v '^#' | xargs) && \
export DISPLAY=:0 && \
nohup python3 06_Integration/HybridBridge/etare_signal_generator_redux.py > /tmp/etare_live.log 2>&1 &
echo $!
"""
        output, error, code = execute_command(ssh, start_cmd)
        if output.strip():
            pid = output.strip()
            print(f"Signal generator started with PID: {pid}")
        else:
            print("Failed to start signal generator")
            if error.strip():
                print(f"Error: {error}")

        # Wait for startup
        time.sleep(5)

        # Step 8: Verify it's running
        print("\n[Step 8] Verifying signal generator is running...")
        output, error, code = execute_command(ssh,
            "ps aux | grep etare_signal_generator_redux | grep -v grep")
        if output.strip():
            print("Signal generator is running:")
            print(output)
        else:
            print("WARNING: Signal generator not running!")

        # Step 9: Check startup logs
        print("\n[Step 9] Checking startup logs...")
        time.sleep(3)
        output, error, code = execute_command(ssh, "tail -50 /tmp/etare_live.log")
        print(output[:2000] if len(output) > 2000 else output)

        # Step 10: Check for MOCK vs LIVE indicators
        print("\n[Step 10] Checking if system is in LIVE mode...")
        output, error, code = execute_command(ssh,
            "tail -30 /tmp/etare_live.log | grep -E 'MOCK|LIVE|Logged into|SUCCESS' | tail -20")
        print(output)

        if "[MOCK]" in output:
            print("\n" + "=" * 60)
            print("WARNING: Still in MOCK mode!")
            print("=" * 60)
            print("Checking why MetaTrader5 module is not working...")

            output, error, code = execute_command(ssh,
                "python3 -c 'import MetaTrader5 as mt5; print(mt5.initialize()); print(mt5.last_error())'")
            print(output)
        elif "Logged into" in output and "[MOCK]" not in output:
            print("\n" + "=" * 60)
            print("SUCCESS: System is now in LIVE TRADING MODE!")
            print("=" * 60)

        ssh.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
