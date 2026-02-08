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

def run():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS)
    
    print("Killing existing MT5 processes...")
    ssh.exec_command("pkill -9 -f terminal64.exe")
    ssh.exec_command("pkill -9 -f start.exe")
    time.sleep(2)
    
    print("Starting MT5 in portable mode...")
    # Use the command found in ps aux
    # root 140798 ... start.exe /exec terminal64.exe
    # root 140866 ... C:\Program Files\MetaTrader 5\terminal64.exe /portable
    
    # We need to run it via Wine start.
    cmd = "export WINEPREFIX=/root/.wine; nohup wine '/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe' /portable > /dev/null 2>&1 &"
    ssh.exec_command(cmd)
    
    print("MT5 restart command sent.")
    time.sleep(5)
    
    print("Checking if MT5 is running...")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep terminal64.exe | grep -v grep")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    run()
