import paramiko

HOST = '203.161.61.61'
USER = 'root'
PASS = 'tg1MNYK98Vt09no8uN'

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
