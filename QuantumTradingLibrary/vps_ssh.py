import paramiko
import sys
from credential_manager import get_vps_credentials, CredentialError

# VPS credentials - loaded from credential_manager
try:
    _vps_creds = get_vps_credentials("VPS_2")
    HOST = _vps_creds["host"]
    USER = _vps_creds["user"]
    PASS = _vps_creds["password"]
except CredentialError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

def run_command(cmd):
    """Run a command on VPS and return output"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=30)
        stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
        out = stdout.read().decode()
        err = stderr.read().decode()
        client.close()
        return out, err
    except Exception as e:
        return None, str(e)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = ' '.join(sys.argv[1:])
    else:
        cmd = 'echo "Connected to VPS!"'

    out, err = run_command(cmd)
    if out:
        print(out)
    if err:
        print(f"STDERR: {err}")
