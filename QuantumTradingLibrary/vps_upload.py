import paramiko
import os
import sys
from scp import SCPClient
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
