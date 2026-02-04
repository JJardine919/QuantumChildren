import paramiko
import sys

# VPS credentials
HOST = '203.161.61.61'
USER = 'root'
PASS = 'tg1MNYK98Vt09no8uN'

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
