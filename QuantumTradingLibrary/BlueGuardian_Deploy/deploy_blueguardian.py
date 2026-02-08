#!/usr/bin/env python3
"""
Blue Guardian VPS Deployment Script
====================================
Deploys the multi-account Blue Guardian system to Namecheap VPS.

Usage:
    python deploy_blueguardian.py

Prerequisites:
    1. Fill in accounts_config.json with your credentials
    2. Ensure VPS is accessible via SSH
    3. Run this script from your local machine
"""

import paramiko
import os
import sys
import json
import time
from pathlib import Path

# Add parent directory for credential_manager
sys.path.insert(0, str(Path(__file__).parent.parent))
from credential_manager import get_credentials, CredentialError

# Local paths
LOCAL_BASE = Path(__file__).parent
CONFIG_FILE = LOCAL_BASE / "accounts_config.json"
CHAMPION_DIR = LOCAL_BASE.parent / "quantu" / "champions"

def load_config():
    if not CONFIG_FILE.exists():
        print(f"ERROR: Config file not found: {CONFIG_FILE}")
        sys.exit(1)
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def get_ssh_client(host, user, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=user, password=password, timeout=30)
        return client
    except Exception as e:
        print(f"SSH Connection failed: {e}")
        sys.exit(1)

def run_cmd(client, cmd, check=True):
    """Execute command and return output"""
    stdin, stdout, stderr = client.exec_command(cmd)
    exit_status = stdout.channel.recv_exit_status()
    output = stdout.read().decode()
    error = stderr.read().decode()

    if check and exit_status != 0:
        print(f"Command failed: {cmd}")
        print(f"Error: {error}")

    return output, error, exit_status

def mkdir_p(sftp, remote_dir):
    """Create directory recursively"""
    dirs = []
    while remote_dir:
        try:
            sftp.stat(remote_dir)
            break
        except FileNotFoundError:
            dirs.append(remote_dir)
            remote_dir = os.path.dirname(remote_dir)

    for d in reversed(dirs):
        try:
            sftp.mkdir(d)
        except:
            pass

def upload_file(sftp, local_path, remote_path):
    """Upload single file"""
    remote_dir = os.path.dirname(remote_path)
    mkdir_p(sftp, remote_dir)
    sftp.put(str(local_path), remote_path)

def main():
    print("=" * 60)
    print("BLUE GUARDIAN VPS DEPLOYMENT")
    print("=" * 60)

    # Load config
    config = load_config()
    vps = config['vps']

    # Validate config
    if vps['host'] == 'NAMECHEAP_VPS_IP':
        print("ERROR: Please update accounts_config.json with your VPS credentials")
        sys.exit(1)

    if config['accounts'][0]['account_id'] == 'PLACEHOLDER_1':
        print("ERROR: Please update accounts_config.json with your Blue Guardian account credentials")
        sys.exit(1)

    host = vps['host']
    user = vps['user']
    password = vps['password']
    remote_base = vps['remote_base']

    print(f"Connecting to {host}...")
    client = get_ssh_client(host, user, password)
    sftp = client.open_sftp()

    # Step 1: Setup VPS environment
    print("\n[1/6] Setting up VPS environment...")
    setup_cmds = [
        "apt-get update -qq",
        "apt-get install -y -qq docker.io docker-compose wine64 xvfb python3-pip",
        "systemctl enable docker",
        "systemctl start docker",
        f"mkdir -p {remote_base}/config",
        f"mkdir -p {remote_base}/champions",
        f"mkdir -p {remote_base}/logs",
        f"mkdir -p {remote_base}/mt5files",
    ]

    for cmd in setup_cmds:
        print(f"  Running: {cmd[:50]}...")
        run_cmd(client, cmd, check=False)

    # Step 2: Upload config
    print("\n[2/6] Uploading configuration...")
    upload_file(sftp, CONFIG_FILE, f"{remote_base}/config/accounts_config.json")
    print(f"  Uploaded: accounts_config.json")

    # Step 3: Upload brain
    print("\n[3/6] Uploading multi-brain...")
    brain_file = LOCAL_BASE / "bg_multi_brain.py"
    upload_file(sftp, brain_file, f"{remote_base}/bg_multi_brain.py")
    print(f"  Uploaded: bg_multi_brain.py")

    # Step 4: Upload champion model
    print("\n[4/6] Uploading champion BTCUSD model...")
    champion_file = CHAMPION_DIR / "champion_BTCUSD.pth"
    if champion_file.exists():
        upload_file(sftp, champion_file, f"{remote_base}/champions/champion_BTCUSD.pth")
        print(f"  Uploaded: champion_BTCUSD.pth")
    else:
        print(f"  WARNING: Champion not found at {champion_file}")
        print(f"  You'll need to train a fresh model or upload manually")

    # Step 5: Create Dockerfile
    print("\n[5/6] Creating Docker configuration...")
    dockerfile_content = '''FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch numpy pandas

COPY bg_multi_brain.py .
COPY champions ./champions
COPY config ./config

RUN mkdir -p /app/logs

CMD ["python", "-u", "bg_multi_brain.py"]
'''

    # Write dockerfile locally then upload
    dockerfile_local = LOCAL_BASE / "Dockerfile.bg_multi"
    with open(dockerfile_local, 'w') as f:
        f.write(dockerfile_content)
    upload_file(sftp, dockerfile_local, f"{remote_base}/Dockerfile")
    print("  Created: Dockerfile")

    # Create docker-compose
    compose_content = f'''version: '3.8'

services:
  bg-brain:
    build: .
    container_name: bg_multi_brain
    restart: unless-stopped
    volumes:
      - {remote_base}/mt5files:/mt5files
      - {remote_base}/config:/app/config
      - {remote_base}/logs:/app/logs
      - {remote_base}/champions:/app/champions
    environment:
      - PYTHONUNBUFFERED=1
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
'''

    compose_local = LOCAL_BASE / "docker-compose.yml"
    with open(compose_local, 'w') as f:
        f.write(compose_content)
    upload_file(sftp, compose_local, f"{remote_base}/docker-compose.yml")
    print("  Created: docker-compose.yml")

    # Step 6: Build and start
    print("\n[6/6] Building and starting services...")
    build_cmds = [
        f"cd {remote_base} && docker-compose build",
        f"cd {remote_base} && docker-compose up -d",
    ]

    for cmd in build_cmds:
        print(f"  Running: {cmd}...")
        output, error, status = run_cmd(client, cmd, check=False)
        if status == 0:
            print("  OK")
        else:
            print(f"  Note: {error[:100] if error else 'completed'}")

    # Check status
    print("\n" + "=" * 60)
    print("DEPLOYMENT STATUS")
    print("=" * 60)

    output, _, _ = run_cmd(client, f"cd {remote_base} && docker-compose ps")
    print(output)

    output, _, _ = run_cmd(client, f"cd {remote_base} && docker-compose logs --tail=10")
    print("\nRecent logs:")
    print(output)

    # Summary
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Set up MT5 terminals for each account:
   - Install Blue Guardian MT5 Terminal via Wine
   - Log in to each account
   - Attach DataExporter service (exports market data)
   - Attach BG_MultiExecutor EA to BTCUSD chart

2. Configure each EA instance:
   Account 1: AccountName="BG_INSTANT_1", MagicNumber=100001
   Account 2: AccountName="BG_INSTANT_2", MagicNumber=100002
   Account 3: AccountName="BG_COMPETITION", MagicNumber=100003

3. Enable trading:
   - Set TradeEnabled=true on each EA
   - Monitor logs: docker-compose logs -f

4. Monitor:
   - Brain logs: docker-compose logs bg-brain
   - Signal files: ls -la /opt/blueguardian/mt5files/
""")

    sftp.close()
    client.close()
    print("\nDeployment complete!")

if __name__ == '__main__':
    main()
