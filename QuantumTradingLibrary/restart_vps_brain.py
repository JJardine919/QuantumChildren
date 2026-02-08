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

def execute_command(ssh, command, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    return output, error

def main():
    print(f"Restarting Brain on VPS {VPS_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=10)
        
        # 1. Kill existing brain
        print("Stopping any existing instances...")
        execute_command(ssh, "pkill -f etare_signal_generator")
        time.sleep(2)
        
        # 2. Start the brain
        print("Starting etare_signal_generator_redux.py...")
        # We run from the root of the repo so imports work
        cmd = (
            "cd /root/QuantumTradingLibrary && "
            "nohup python3 06_Integration/HybridBridge/etare_signal_generator_redux.py "
            "> etare_redux_live.log 2> etare_redux_err.log &"
        )
        execute_command(ssh, cmd)
        
        # 3. Verify
        time.sleep(3)
        print("Verifying process...")
        out, _ = execute_command(ssh, "ps aux | grep etare_signal_generator")
        if "etare_signal_generator_redux.py" in out:
            print("✅ SUCCESS: Brain is running.")
            print(out)
        else:
            print("❌ FAILURE: Brain did not start.")
            # Check why
            out, _ = execute_command(ssh, "cat /root/QuantumTradingLibrary/etare_redux_err.log")
            print("--- Error Log ---")
            print(out)
            
        ssh.close()

    except Exception as e:
        print(f"Operation failed: {e}")

if __name__ == "__main__":
    main()
