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
    """Execute a command on the VPS and return output"""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    return output, error

def main():
    print(f"Diagnosing VPS {VPS_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=10)
        
        # 1. Find the main python file
        print("\n[1] Finding signal generator script...")
        cmd = "find /root -name 'etare_signal_generator*.py'"
        out, err = execute_command(ssh, cmd)
        print(out if out else "Not found")
        script_path = out.strip().split('\n')[0] if out.strip() else ""

        # 2. Check logs (err log)
        print("\n[2] Checking error logs (last 20 lines)...")
        # Try a few common names
        log_files = ["etare_gen_err.log", "etare_signal_generator.log", "etare_redux_err.log"]
        found_log = False
        for log in log_files:
            cmd = f"find /root -name '{log}'"
            out, _ = execute_command(ssh, cmd)
            if out.strip():
                log_path = out.strip().split('\n')[0]
                print(f"Found log: {log_path}")
                out, _ = execute_command(ssh, f"tail -n 20 {log_path}")
                print("--- LOG CONTENT ---")
                print(out)
                print("-------------------")
                found_log = True
                break
        if not found_log:
            print("No error logs found.")

        # 3. Check MT5 process
        print("\n[3] Checking MT5/Wine process...")
        out, _ = execute_command(ssh, "ps aux | grep -i 'terminal64.exe'")
        print(out if out else "MT5 process not found")

        ssh.close()
        
        return script_path

    except Exception as e:
        print(f"Connection failed: {e}")
        return ""

if __name__ == "__main__":
    main()
