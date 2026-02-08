#!/usr/bin/env python3
"""Setup Working Live Trading Environment"""

import paramiko
import time
import sys
from credential_manager import get_vps_credentials, get_credentials, CredentialError

# VPS Configuration - loaded from credential_manager
try:
    _vps_creds = get_vps_credentials("VPS_1")
    VPS_HOST = _vps_creds["host"]
    VPS_USER = _vps_creds["user"]
    VPS_PASS = _vps_creds["password"]

    # Get MT5 credentials for GL_1 (used in test script)
    _gl1_creds = get_credentials("GL_1")
    GL1_ACCOUNT = _gl1_creds["account"]
    GL1_PASSWORD = _gl1_creds["password"]
    GL1_SERVER = _gl1_creds["server"]
except CredentialError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

def execute_command(ssh, command, timeout=30):
    """Execute a command on the VPS and return output"""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    output = stdout.read().decode('utf-8', errors='ignore')
    error = stderr.read().decode('utf-8', errors='ignore')
    exit_code = stdout.channel.recv_exit_status()
    return output, error, exit_code

def main():
    print("=" * 60)
    print("Setup Live Trading Environment")
    print("=" * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"\nConnecting to VPS {VPS_HOST}...")
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=30)
        print("Connected!")

        # Step 1: Check for existing working setups
        print("\n[Step 1] Looking for existing working signal generators...")
        output, error, code = execute_command(ssh,
            "find /root -name 'etare_signal_generator*.py' -type f 2>/dev/null")
        print("Found signal generator scripts:")
        print(output)

        # Step 2: Check ETARE_Trading directory
        print("\n[Step 2] Checking ETARE_Trading directory...")
        output, error, code = execute_command(ssh,
            "ls -la /root/ETARE_Trading/ 2>/dev/null | head -20")
        if output.strip() and "No such file" not in output:
            print(output)

            # Check if it has a working signal generator
            output, error, code = execute_command(ssh,
                "ls -la /root/ETARE_Trading/*.py 2>/dev/null")
            print(output)

            # Try to run it
            print("\n[Step 3] Testing ETARE_Trading signal generator...")
            test_cmd = """
cd /root/ETARE_Trading && \
export $(cat /root/.env | grep -v '^#' | xargs) && \
export DISPLAY=:0 && \
timeout 10 python3 etare_signal_generator.py 2>&1 | head -50
"""
            output, error, code = execute_command(ssh, test_cmd, timeout=15)
            print(output)

        # Step 4: Check for Python virtual environments
        print("\n[Step 4] Looking for Python virtual environments...")
        output, error, code = execute_command(ssh,
            "find /root -name 'venv' -o -name 'env' -o -name '*_env' -type d -maxdepth 2 2>/dev/null")
        print(output)

        # Step 5: Try to install MetaTrader5 in system Python
        print("\n[Step 5] Installing MetaTrader5 Python module...")
        output, error, code = execute_command(ssh,
            "pip3 install MetaTrader5 2>&1 | tail -20", timeout=120)
        print(output)

        # Verify installation
        output, error, code = execute_command(ssh,
            "python3 -c 'import MetaTrader5 as mt5; print(f\"MT5 installed: {mt5.__version__}\")'")
        print(output)
        if error.strip():
            print(f"Error: {error}")

        # Step 6: Test MT5 connection
        print("\n[Step 6] Testing MT5 connection...")
        test_script = f"""
import MetaTrader5 as mt5

print("Initializing MT5...")
if mt5.initialize():
    print("MT5 initialized successfully!")

    # Try to login with GetLeveraged
    print("\\nAttempting login to GetLeveraged ({GL1_ACCOUNT})...")
    if mt5.login(login={GL1_ACCOUNT}, password="{GL1_PASSWORD}", server="{GL1_SERVER}"):
        print("SUCCESS: Logged into GetLeveraged!")
        account_info = mt5.account_info()
        if account_info:
            print(f"Account: {{account_info.login}}")
            print(f"Server: {{account_info.server}}")
            print(f"Balance: ${{account_info.balance}}")
            print(f"Leverage: 1:{{account_info.leverage}}")
    else:
        error = mt5.last_error()
        print(f"FAILED to login: {{error}}")
        print("Note: Server may not be in servers list")

    mt5.shutdown()
else:
    print(f"Failed to initialize MT5: {{mt5.last_error()}}")
"""
        output, error, code = execute_command(ssh,
            f"cd /root && DISPLAY=:0 python3 -c '{test_script}'", timeout=30)
        print(output)
        if error.strip():
            print(f"Errors: {error}")

        # Step 7: If MetaTrader5 works, restart signal generator
        if "SUCCESS" in output:
            print("\n[Step 7] MetaTrader5 working! Starting signal generator...")
            start_cmd = """
cd /root/QuantumTradingLibrary && \
export $(cat /root/.env | grep -v '^#' | xargs) && \
export DISPLAY=:0 && \
nohup python3 06_Integration/HybridBridge/etare_signal_generator_redux.py > /tmp/etare_live.log 2>&1 &
echo $!
"""
            output, error, code = execute_command(ssh, start_cmd)
            print(f"Started with PID: {output}")

            time.sleep(5)

            # Check logs
            output, error, code = execute_command(ssh, "tail -50 /tmp/etare_live.log")
            print("\nSignal generator logs:")
            print(output[:2000] if len(output) > 2000 else output)

        else:
            print("\n" + "=" * 60)
            print("MT5 Connection Failed - Broker Not Configured")
            print("=" * 60)
            print("""
GetLeveraged-Trade and AtlasFunded-Server are not configured in MT5.

You need to add these broker servers to MT5:

1. Open MT5 terminal on VPS via Remote Desktop
2. File -> Open an Account
3. Search for "GetLeveraged" and "Atlas Funded"
4. This will download the server files

OR manually add server files if you have them locally.
""")

        ssh.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
