
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

def run():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS)

    print("Finding .ini files...")
    stdin, stdout, stderr = ssh.exec_command("find '/root/.wine/drive_c/Program Files/MetaTrader 5' -name '*.ini' 2>/dev/null")
    print(stdout.read().decode())

    ssh.close()

if __name__ == "__main__":
    run()
