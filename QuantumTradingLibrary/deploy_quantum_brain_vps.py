"""
Deploy Quantum Brain to VPS
One-click deployment - uploads everything and starts it running
"""

import paramiko
import os
import sys
from pathlib import Path
from credential_manager import get_vps_credentials, CredentialError

try:
    creds = get_vps_credentials('VPS_2')
    HOST = creds['host']
    USER = creds['user']
    PASS = creds['password']
except CredentialError as e:
    print(f"Error: {e}")
    sys.exit(1)

REMOTE_DIR = '/root/quantum_brain'

def get_sftp():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    return client, client.open_sftp()

def run_cmd(cmd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
    out = stdout.read().decode()
    err = stderr.read().decode()
    client.close()
    return out, err

def upload_file(sftp, local_path, remote_path):
    try:
        sftp.put(str(local_path), remote_path)
        print(f"  [OK] {Path(local_path).name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {Path(local_path).name}: {e}")
        return False

def main():
    print("=" * 50)
    print("DEPLOYING QUANTUM BRAIN TO VPS")
    print("=" * 50)

    # Create directories
    print("\n[1/5] Creating directories...")
    run_cmd(f"mkdir -p {REMOTE_DIR}/top_50_experts")
    run_cmd(f"mkdir -p {REMOTE_DIR}/01_Systems/QuantumCompression/utils")
    print("  Done")

    # Connect SFTP
    print("\n[2/5] Connecting to VPS...")
    client, sftp = get_sftp()
    print("  Connected")

    # Upload main scripts
    print("\n[3/5] Uploading quantum brain scripts...")
    local_dir = Path(__file__).parent

    files_to_upload = [
        'quantum_brain_multiaccounts.py',
        'quantum_brain.py',
    ]

    for f in files_to_upload:
        local_path = local_dir / f
        if local_path.exists():
            upload_file(sftp, local_path, f"{REMOTE_DIR}/{f}")

    # Upload experts
    print("\n[4/5] Uploading trained experts...")
    experts_dir = local_dir / 'top_50_experts'
    if experts_dir.exists():
        for f in experts_dir.iterdir():
            if f.suffix in ['.pth', '.json']:
                upload_file(sftp, f, f"{REMOTE_DIR}/top_50_experts/{f.name}")

    sftp.close()
    client.close()

    # Create VPS runner script
    print("\n[5/5] Creating VPS runner script...")

    vps_runner = '''#!/bin/bash
# Quantum Brain VPS Runner
cd /root/quantum_brain

# Use native Python (not Wine) for the brain
export DISPLAY=:99

# Install dependencies if needed
pip3 install numpy pandas torch pywt scipy qutip 2>/dev/null

# Kill any existing brain
pkill -f "quantum_brain" 2>/dev/null

# Start the brain with nohup
nohup python3 quantum_brain_multiaccounts.py > quantum_brain.log 2>&1 &

echo "Quantum Brain started! PID: $!"
echo "Check logs: tail -f /root/quantum_brain/quantum_brain.log"
'''

    # Upload runner
    client, sftp = get_sftp()
    with sftp.file(f"{REMOTE_DIR}/run_brain.sh", 'w') as f:
        f.write(vps_runner)
    sftp.close()
    client.close()

    # Make executable and run
    run_cmd(f"chmod +x {REMOTE_DIR}/run_brain.sh")

    print("\n" + "=" * 50)
    print("DEPLOYMENT COMPLETE!")
    print("=" * 50)
    print(f"\nFiles deployed to: {REMOTE_DIR}")
    print("\nTo start the brain on VPS, run:")
    print(f"  python vps_ssh.py '{REMOTE_DIR}/run_brain.sh'")
    print("\nTo check status:")
    print(f"  python vps_ssh.py 'tail -20 {REMOTE_DIR}/quantum_brain.log'")

if __name__ == '__main__':
    main()
