import paramiko
import os
from scp import SCPClient

HOST = '203.161.61.61'
USER = 'root'
PASS = 'tg1MNYK98Vt09no8uN'

def upload_file(local_path, remote_path):
    """Upload a file to VPS"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=30)
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        client.close()
        print(f"Uploaded: {local_path} -> {remote_path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def upload_dir(local_dir, remote_dir):
    """Upload a directory to VPS"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, username=USER, password=PASS, timeout=30)
        sftp = client.open_sftp()

        # Create remote directory
        try:
            sftp.mkdir(remote_dir)
        except:
            pass

        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            remote_path = f"{remote_dir}/{item}"
            if os.path.isfile(local_path):
                sftp.put(local_path, remote_path)
                print(f"Uploaded: {item}")

        sftp.close()
        client.close()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    print("Testing upload...")
    # Test with a simple file
    with open('test_upload.txt', 'w') as f:
        f.write('Test upload successful!')
    upload_file('test_upload.txt', '/opt/trading/test_upload.txt')
