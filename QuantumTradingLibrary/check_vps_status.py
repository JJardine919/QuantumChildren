import paramiko
import sys
from credential_manager import get_vps_credentials, CredentialError

try:
    creds = get_vps_credentials('VPS_1')
    VPS_HOST = creds['host']
    VPS_USER = creds['user']
    VPS_PASS = creds['password']
except CredentialError as e:
    print(f"Error: {e}")
    sys.exit(1)

def execute_command(ssh, command, timeout=30):
    """Execute a command on the VPS and return output"""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    return output, error

def main():
    print(f"Checking VPS status for {VPS_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=10)
        print("Connected successfully.")

        # Check for the multi-account config file
        print("\n[Check 1] Looking for accounts_config.py...")
        output, error = execute_command(ssh, "cat /root/QuantumTradingLibrary/accounts_config.py")
        if "No such file" in error:
            print("accounts_config.py NOT FOUND.")
        else:
            print("accounts_config.py FOUND.")
            if "113326" in output and "113328" in output and "107245" in output:
                print("   - All 3 GetLeveraged accounts (113326, 113328, 107245) are present in the config.")
            else:
                print("   - Config exists but might be missing some accounts. Content snippet:")
                print(output[:200] + "...")

        # Check running processes
        print("\n[Check 2] Checking for trading processes...")
        output, error = execute_command(ssh, "ps aux | grep etare")
        if "etare_signal_generator" in output:
             print("'etare_signal_generator' is RUNNING.")
        else:
             print("'etare_signal_generator' is NOT running.")

        # Check MT5 connection logs (last 5 lines)
        print("\n[Check 3] Checking recent logs...")
        output, error = execute_command(ssh, "tail -n 5 /root/QuantumTradingLibrary/06_Integration/HybridBridge/bridge.log 2>/dev/null")
        if output:
            print("Recent log entries:")
            print(output)
        else:
            print("No log entries found or file does not exist.")

        ssh.close()

    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    main()
