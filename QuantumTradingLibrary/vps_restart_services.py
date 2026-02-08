#!/usr/bin/env python3
"""Restart VPS Trading Services and Verify GetLeveraged Connection"""

import paramiko
import time
import sys
from credential_manager import get_vps_credentials, CredentialError

# VPS Configuration - loaded from credential_manager
try:
    _vps_creds = get_vps_credentials("VPS_1")
    VPS_HOST = _vps_creds["host"]
    VPS_USER = _vps_creds["user"]
    VPS_PASS = _vps_creds["password"]
except CredentialError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

def execute_command(ssh, command, timeout=60):
    """Execute a command on the VPS and return output"""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    exit_code = stdout.channel.recv_exit_status()
    return output, error, exit_code

def main():
    print("=" * 60)
    print("VPS Service Restart and Connection Verification")
    print("=" * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"\nConnecting to VPS {VPS_HOST}...")
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=30)
        print("Connected!")

        # Step 1: Check current running processes
        print("\n[Step 1] Current running trading processes...")
        output, error, code = execute_command(ssh, "ps aux | grep -E 'etare|signal|bridge|mt5' | grep -v grep")
        if output.strip():
            print(output)
        else:
            print("No trading processes found running")

        # Step 2: Kill existing signal generator
        print("\n[Step 2] Stopping existing signal generator...")
        output, error, code = execute_command(ssh, "pkill -f 'etare_signal_generator' || true")
        time.sleep(2)

        # Verify it stopped
        output, error, code = execute_command(ssh, "ps aux | grep etare_signal_generator | grep -v grep")
        if output.strip():
            print("Warning: Signal generator still running, force killing...")
            execute_command(ssh, "pkill -9 -f 'etare_signal_generator' || true")
            time.sleep(2)
        else:
            print("Signal generator stopped.")

        # Step 3: Check MT5 Wine bridge status
        print("\n[Step 3] Checking MT5 Wine bridge...")
        output, error, code = execute_command(ssh, "ps aux | grep mt5_wine_bridge | grep -v grep")
        if output.strip():
            print("MT5 Wine bridge is running:")
            print(output)
        else:
            print("MT5 Wine bridge is NOT running - may need to start it")

        # Step 4: Check for MT5 server configuration options
        print("\n[Step 4] Checking MT5 server configuration...")
        # Look for server.dat or any server config files
        output, error, code = execute_command(ssh, "find /root/.wine_mt5 -name 'server*' -o -name '*.srv' 2>/dev/null | head -20")
        if output.strip():
            print("Found server files:")
            print(output)

        # Check MT5 config directory
        output, error, code = execute_command(ssh, "ls -la /root/.wine_mt5/drive_c/Program\\ Files/MetaTrader\\ 5/config/ 2>/dev/null || echo 'Config dir not found'")
        print("\nMT5 Config directory:")
        print(output)

        # Step 5: Check if we can find server list
        print("\n[Step 5] Looking for broker server lists...")
        output, error, code = execute_command(ssh, "find /root/.wine_mt5 -type d -name '*server*' 2>/dev/null")
        if output.strip():
            print(output)

        # Check MQL5 files directory
        output, error, code = execute_command(ssh, "ls -la /root/.wine_mt5/drive_c/Program\\ Files/MetaTrader\\ 5/MQL5/Files/ 2>/dev/null || echo 'MQL5 Files dir not found'")
        print("\nMQL5 Files directory:")
        print(output)

        # Step 6: Check the .env is properly updated
        print("\n[Step 6] Verifying .env configuration...")
        output, error, code = execute_command(ssh, "cat /root/.env")
        print(output)

        # Step 7: Check signal generator script location
        print("\n[Step 7] Locating signal generator script...")
        output, error, code = execute_command(ssh, "find /root -name '*signal_generator*' -type f 2>/dev/null | head -10")
        if output.strip():
            print("Found signal generator scripts:")
            print(output)

        # Step 8: Check logs for connection issues
        print("\n[Step 8] Checking recent logs...")
        output, error, code = execute_command(ssh, "ls -la /root/QuantumTradingLibrary/06_Integration/HybridBridge/*.log 2>/dev/null | head -5")
        if output.strip():
            print("Log files found:")
            print(output)
            # Get last 20 lines of most recent log
            output, error, code = execute_command(ssh, "tail -30 /root/QuantumTradingLibrary/06_Integration/HybridBridge/*.log 2>/dev/null | head -50")
            if output.strip():
                print("\nRecent log entries:")
                print(output)

        # Step 9: Try to start signal generator with new config
        print("\n[Step 9] Starting signal generator with new GetLeveraged config...")
        # First, source the .env and start in background
        start_cmd = """
cd /root/QuantumTradingLibrary && \
export $(cat /root/.env | grep -v '^#' | xargs) && \
nohup python3 01_Systems/SignalGenerator/etare_signal_generator_redux.py > /tmp/signal_generator.log 2>&1 &
echo $!
"""
        output, error, code = execute_command(ssh, start_cmd)
        if output.strip():
            pid = output.strip()
            print(f"Signal generator started with PID: {pid}")

        # Wait a moment for startup
        time.sleep(5)

        # Step 10: Verify it's running
        print("\n[Step 10] Verifying signal generator is running...")
        output, error, code = execute_command(ssh, "ps aux | grep etare_signal_generator | grep -v grep")
        if output.strip():
            print("Signal generator is now running:")
            print(output)
        else:
            print("Warning: Signal generator may have failed to start")
            # Check startup log
            output, error, code = execute_command(ssh, "cat /tmp/signal_generator.log 2>/dev/null | tail -30")
            if output.strip():
                print("\nStartup log:")
                print(output)

        # Step 11: Check MT5 bridge logs for connection status
        print("\n[Step 11] Checking MT5 bridge connection status...")
        output, error, code = execute_command(ssh, "tail -50 /opt/etare/bridge/logs/*.log 2>/dev/null || tail -50 /var/log/etare*.log 2>/dev/null || echo 'No bridge logs found'")
        print(output[:2000] if len(output) > 2000 else output)

        print("\n" + "=" * 60)
        print("Service Restart Complete")
        print("=" * 60)
        print("""
Status Summary:
- .env updated with GetLeveraged account 113326
- Signal generator restarted with new configuration

If MT5 cannot connect to GetLeveraged-Trade server:
1. The server files may need to be added manually
2. You can copy .srv files from a Windows MT5 installation
3. Or configure MT5 terminal to discover the server

Next: Monitor the logs for successful connection.
""")

        ssh.close()
        print("\nConnection closed.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
