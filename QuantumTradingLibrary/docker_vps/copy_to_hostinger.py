"""
Copy Docker MT5 MCP Setup to Hostinger VPS
==========================================
Copies all files and sets up the VPS for Docker deployment.
"""

import paramiko
import os
from pathlib import Path

# Hostinger VPS Details
VPS_HOST = "72.62.170.153"
VPS_USER = "root"
VPS_PASS = input("Enter VPS root password: ") if __name__ == "__main__" else ""

DEPLOY_DIR = "/opt/mt5-mcp"
LOCAL_DIR = Path(__file__).parent

FILES_TO_COPY = [
    "Dockerfile.mt5",
    "Dockerfile.gateway",
    "docker-compose.yml",
    "mcp_mt5_http_server.py",
    "mcp_gateway.py",
    "entrypoint.sh",
    "deploy_to_vps.sh",
    "config.json",
]


def copy_files():
    """Copy all Docker files to VPS"""
    print(f"Connecting to {VPS_HOST}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS)

    # Create deploy directory
    print(f"Creating {DEPLOY_DIR}...")
    stdin, stdout, stderr = client.exec_command(f"mkdir -p {DEPLOY_DIR}")
    stdout.read()

    # Open SFTP
    sftp = client.open_sftp()

    # Copy files
    for filename in FILES_TO_COPY:
        local_path = LOCAL_DIR / filename
        remote_path = f"{DEPLOY_DIR}/{filename}"

        if local_path.exists():
            print(f"Copying {filename}...")
            sftp.put(str(local_path), remote_path)
        else:
            print(f"WARNING: {filename} not found locally")

    sftp.close()

    # Make scripts executable
    print("Making scripts executable...")
    client.exec_command(f"chmod +x {DEPLOY_DIR}/*.sh")

    # Install Docker if needed
    print("\nChecking Docker installation...")
    stdin, stdout, stderr = client.exec_command("which docker || echo 'NOT_INSTALLED'")
    docker_check = stdout.read().decode().strip()

    if "NOT_INSTALLED" in docker_check:
        print("Installing Docker on VPS...")
        commands = [
            "zypper refresh",
            "zypper install -y docker docker-compose",
            "systemctl enable docker",
            "systemctl start docker",
        ]
        for cmd in commands:
            print(f"  Running: {cmd}")
            stdin, stdout, stderr = client.exec_command(cmd)
            stdout.read()
    else:
        print(f"Docker already installed: {docker_check}")

    print("\n" + "=" * 50)
    print("Files copied successfully!")
    print("=" * 50)
    print(f"\nNext steps:")
    print(f"  1. SSH to VPS: ssh root@{VPS_HOST}")
    print(f"  2. cd {DEPLOY_DIR}")
    print(f"  3. ./deploy_to_vps.sh")
    print(f"\nOr run deployment remotely now? (y/n)")

    if input().lower() == 'y':
        print("\nRunning deployment script...")
        stdin, stdout, stderr = client.exec_command(f"cd {DEPLOY_DIR} && ./deploy_to_vps.sh 2>&1")

        # Stream output
        for line in iter(stdout.readline, ""):
            print(line, end="")

        print("\nDeployment complete!")
        print(f"\nMCP Gateway URL: http://{VPS_HOST}:8080")

    client.close()


if __name__ == "__main__":
    copy_files()
