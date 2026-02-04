import paramiko

HOST = '203.161.61.61'
USER = 'root'
PASS = 'tg1MNYK98Vt09no8uN'

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
