#!/usr/bin/env python3
"""Add GetLeveraged and Atlas broker servers to MT5 and reconnect"""

import paramiko
import time
import sys

VPS_HOST = "72.62.170.153"
VPS_USER = "root"
VPS_PASS = "gXRCBtbi21##"

def execute_command(ssh, command, timeout=30):
    """Execute a command on the VPS and return output"""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    exit_code = stdout.channel.recv_exit_status()
    return output, error, exit_code

def main():
    print("=" * 60)
    print("MT5 Broker Server Configuration")
    print("=" * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"\nConnecting to VPS {VPS_HOST}...")
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=30)
        print("Connected!")

        # Step 1: Find where signal generator is actually running
        print("\n[Step 1] Locating actual signal generator...")
        output, error, code = execute_command(ssh,
            "ps aux | grep etare_signal_generator | grep -v grep")
        print(output)

        output, error, code = execute_command(ssh,
            "pwdx $(ps aux | grep etare_signal_generator_redux | grep -v grep | awk '{print $2}' | head -1)")
        print(f"Working directory: {output}")

        # Step 2: Check if running in Docker
        print("\n[Step 2] Checking Docker containers...")
        output, error, code = execute_command(ssh,
            "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null || echo 'Docker not available'")
        print(output)

        # Step 3: Try to add broker servers using MT5 Python API
        print("\n[Step 3] Adding broker servers via MT5 terminal...")

        # First, let's create a Python script on the VPS to add servers
        add_server_script = """
import MetaTrader5 as mt5
import sys

# Initialize MT5
if not mt5.initialize():
    print(f"ERROR: MT5 initialize failed: {mt5.last_error()}")
    sys.exit(1)

print("MT5 initialized successfully")

# Get current account info
account_info = mt5.account_info()
if account_info:
    print(f"Currently connected:")
    print(f"  Account: {account_info.login}")
    print(f"  Server: {account_info.server}")
    print(f"  Balance: ${account_info.balance}")
else:
    print("Not currently logged into any account")

# Shutdown current connection
mt5.shutdown()

# Try to login to GetLeveraged
print("\\nAttempting to login to GetLeveraged account 113326...")
if not mt5.initialize():
    print(f"ERROR: MT5 initialize failed: {mt5.last_error()}")
    sys.exit(1)

login_result = mt5.login(
    login=113326,
    password="%bwN)IvJ5F",
    server="GetLeveraged-Trade"
)

if login_result:
    print("SUCCESS: Logged into GetLeveraged!")
    account_info = mt5.account_info()
    if account_info:
        print(f"Account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Balance: ${account_info.balance}")
        print(f"Leverage: 1:{account_info.leverage}")

        # Check symbols
        symbols = mt5.symbols_get()
        print(f"Available symbols: {len(symbols)}")
        if symbols:
            print(f"Sample symbols: {[s.name for s in symbols[:5]]}")
else:
    error = mt5.last_error()
    print(f"FAILED to login to GetLeveraged: {error}")
    print("This likely means the server is not in the servers list")

mt5.shutdown()

# Try Atlas
print("\\nAttempting to login to Atlas account 212000586...")
if not mt5.initialize():
    print(f"ERROR: MT5 initialize failed: {mt5.last_error()}")
    sys.exit(1)

login_result = mt5.login(
    login=212000586,
    password="4mqhwy3A",
    server="AtlasFunded-Server"
)

if login_result:
    print("SUCCESS: Logged into Atlas!")
    account_info = mt5.account_info()
    if account_info:
        print(f"Account: {account_info.login}")
        print(f"Server: {account_info.server}")
        print(f"Balance: ${account_info.balance}")
else:
    error = mt5.last_error()
    print(f"FAILED to login to Atlas: {error}")

mt5.shutdown()
"""

        # Write script to VPS
        print("\n[Step 4] Creating broker test script...")
        output, error, code = execute_command(ssh,
            f"cat > /tmp/test_brokers.py << 'EOFPYTHON'\n{add_server_script}\nEOFPYTHON")

        # Run the script
        print("\n[Step 5] Testing broker connections...")
        output, error, code = execute_command(ssh,
            "cd /root && DISPLAY=:0 python3 /tmp/test_brokers.py", timeout=60)
        print(output)
        if error.strip():
            print(f"Errors: {error}")

        # Step 6: Check if we need to manually add servers using terminal
        print("\n[Step 6] Checking if servers need manual addition...")
        if "server is not in the servers list" in output.lower() or "failed to login" in output.lower():
            print("\n" + "="*60)
            print("ISSUE FOUND: Broker servers not configured in MT5")
            print("="*60)
            print("""
The GetLeveraged-Trade and AtlasFunded-Server are not in MT5's server list.

To fix this, you need to:

Option 1 - Add via MT5 Terminal (Remote Desktop):
1. Remote desktop to the VPS
2. Open MetaTrader 5 terminal
3. Go to File -> Open an Account
4. Click "Find your broker" and search for:
   - "GetLeveraged"
   - "Atlas Funded"
5. This will download and add the server files

Option 2 - Copy server files from your local MT5:
If you have GetLeveraged/Atlas setup on your local Windows machine:
1. Find: C:\\Users\\[Your User]\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Servers\\
2. Copy the .srv files for GetLeveraged and Atlas
3. Upload to VPS: /root/.wine_mt5/drive_c/users/root/AppData/Roaming/MetaQuotes/Terminal/Common/Servers/

Option 3 - Request server files from brokers:
Contact GetLeveraged and Atlas support for their .srv server files
""")

        # Step 7: Check what the signal generator is doing
        print("\n[Step 7] Checking signal generator status...")
        output, error, code = execute_command(ssh,
            "docker logs --tail 50 $(docker ps -q) 2>/dev/null || tail -50 /app/logs/*.log 2>/dev/null || echo 'Cannot read logs'")
        print(output[:2000] if len(output) > 2000 else output)

        ssh.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
