import paramiko
import os
import sys
import stat
from credential_manager import get_vps_credentials, CredentialError

try:
    creds = get_vps_credentials('VPS_2')
    HOST = creds['host']
    USER = creds['user']
    PASS = creds['password']
except CredentialError as e:
    print(f"Error: {e}")
    sys.exit(1)

REMOTE_BASE = '/opt/trading'
LOCAL_BASE = r'C:\Users\jjj10\QuantumTradingLibrary'

def get_sftp():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    return client, client.open_sftp()

def mkdir_p(sftp, remote_dir):
    """Create directory recursively"""
    dirs_to_create = []
    while True:
        try:
            sftp.stat(remote_dir)
            break
        except FileNotFoundError:
            dirs_to_create.append(remote_dir)
            remote_dir = os.path.dirname(remote_dir)
            if not remote_dir or remote_dir == '/':
                break
    for d in reversed(dirs_to_create):
        try:
            sftp.mkdir(d)
        except:
            pass

def upload_directory(sftp, local_dir, remote_dir, extensions=None):
    """Upload directory to VPS"""
    mkdir_p(sftp, remote_dir)
    count = 0
    for root, dirs, files in os.walk(local_dir):
        # Skip certain directories
        skip_dirs = ['__pycache__', '.git', 'venv', 'gpu_env', 'quantu', 'node_modules', 'catboost_info']
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if extensions and not any(file.endswith(ext) for ext in extensions):
                continue

            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            remote_path = f"{remote_dir}/{rel_path}".replace('\\', '/')

            # Create remote directory
            remote_file_dir = os.path.dirname(remote_path)
            mkdir_p(sftp, remote_file_dir)

            try:
                sftp.put(local_path, remote_path)
                count += 1
                print(f"[{count}] {rel_path}")
            except Exception as e:
                print(f"Error uploading {rel_path}: {e}")

    return count

def main():
    client, sftp = get_sftp()

    print("=" * 50)
    print("Deploying ETARE QuantumFusion to VPS")
    print("=" * 50)

    # Upload ETARE_QuantumFusion directory
    etare_local = os.path.join(LOCAL_BASE, 'ETARE_QuantumFusion')
    if os.path.exists(etare_local):
        print("\n[1/3] Uploading ETARE_QuantumFusion...")
        count = upload_directory(sftp, etare_local, f'{REMOTE_BASE}/ETARE_QuantumFusion',
                                 extensions=['.py', '.yaml', '.json', '.txt', '.md'])
        print(f"Uploaded {count} files")

    # Upload 01_Systems directory
    systems_local = os.path.join(LOCAL_BASE, '01_Systems')
    if os.path.exists(systems_local):
        print("\n[2/3] Uploading 01_Systems...")
        count = upload_directory(sftp, systems_local, f'{REMOTE_BASE}/01_Systems',
                                 extensions=['.py', '.yaml', '.json', '.txt'])
        print(f"Uploaded {count} files")

    # Upload key Python files from root
    print("\n[3/3] Uploading root Python files...")
    root_files = [
        'ai_trader_quantum_compression.py',
        'ai_trader_quantum_fusion.py',
        'bg_brain.py',
        'etare_fusion_trader.py',
    ]

    for f in root_files:
        local_path = os.path.join(LOCAL_BASE, f)
        if os.path.exists(local_path):
            remote_path = f'{REMOTE_BASE}/{f}'
            try:
                sftp.put(local_path, remote_path)
                print(f"Uploaded: {f}")
            except Exception as e:
                print(f"Error: {f} - {e}")

    sftp.close()
    client.close()

    print("\n" + "=" * 50)
    print("Deployment complete!")
    print("=" * 50)

if __name__ == '__main__':
    main()
