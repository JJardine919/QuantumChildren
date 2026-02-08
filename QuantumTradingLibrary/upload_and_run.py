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

# Connect
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)

# Upload test file
sftp = client.open_sftp()
sftp.put('vps_test_import.py', '/opt/trading/test_import.py')
sftp.close()
print("File uploaded")

# Run test
stdin, stdout, stderr = client.exec_command('cd /opt/trading && source venv/bin/activate && python test_import.py')
print("STDOUT:", stdout.read().decode())
print("STDERR:", stderr.read().decode())

client.close()
