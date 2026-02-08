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

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS)

# Create Experts directory
client.exec_command('mkdir -p "/root/.wine/drive_c/Program Files/Blue Guardian MT5 Terminal/MQL5/Experts"')

# Upload EA
sftp = client.open_sftp()
sftp.put('BlueGuardian_Quantum.mq5', '/root/.wine/drive_c/Program Files/Blue Guardian MT5 Terminal/MQL5/Experts/BlueGuardian_Quantum.mq5')
sftp.close()

print("EA uploaded to VPS")

# Verify
stdin, stdout, stderr = client.exec_command('ls -la "/root/.wine/drive_c/Program Files/Blue Guardian MT5 Terminal/MQL5/Experts/"')
print(stdout.read().decode())

client.close()
