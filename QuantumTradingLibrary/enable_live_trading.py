#!/usr/bin/env python3
"""Enable Live Trading on VPS"""

import paramiko
import time
import sys
from credential_manager import get_vps_credentials, CredentialError

try:
    creds = get_vps_credentials('VPS_1')
    VPS_HOST = creds['host']
    VPS_USER = creds['user']
    VPS_PASS = creds['password']
except CredentialError as e:
    print(f"Error: {e}")
    sys.exit(1)

def execute_command(ssh, command, timeout=30):
    """Execute a command on the VPS and return output"""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    exit_code = stdout.channel.recv_exit_status()
    return output, error, exit_code

def main():
    print("=" * 60)
    print("Enable Live Trading Mode")
    print("=" * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"\nConnecting to VPS {VPS_HOST}...")
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=30)
        print("Connected!")

        # Step 1: Check Docker container environment
        print("\n[Step 1] Checking Docker container configuration...")
        output, error, code = execute_command(ssh,
            "docker inspect quantum-brain --format='{{range .Config.Env}}{{println .}}{{end}}'")
        print("Container environment variables:")
        print(output)

        # Step 2: Check for MOCK_MODE or LIVE_MODE settings
        print("\n[Step 2] Looking for trading mode configuration...")
        output, error, code = execute_command(ssh,
            "docker exec quantum-brain env | grep -E 'MOCK|LIVE|MODE|TRADE'")
        if output.strip():
            print(output)
        else:
            print("No MODE environment variables found")

        # Step 3: Check the signal generator code for MOCK mode
        print("\n[Step 3] Checking signal generator for MOCK mode flag...")
        output, error, code = execute_command(ssh,
            "docker exec quantum-brain cat /app/06_Integration/HybridBridge/etare_signal_generator_redux.py 2>/dev/null | grep -i 'mock' | head -10")
        if output.strip():
            print("Found MOCK references:")
            print(output)

        # Step 4: Check .env file in Docker
        print("\n[Step 4] Checking .env file inside Docker container...")
        output, error, code = execute_command(ssh,
            "docker exec quantum-brain cat /app/.env 2>/dev/null || docker exec quantum-brain cat /root/.env 2>/dev/null || echo 'No .env found'")
        print(output)

        # Step 5: Check if there's a LIVE_MODE flag we can set
        print("\n[Step 5] Finding configuration files...")
        output, error, code = execute_command(ssh,
            "docker exec quantum-brain find /app -name '*.env' -o -name 'config*.py' -o -name 'settings*.py' 2>/dev/null | head -10")
        print(output)

        # Step 6: Check the actual Docker run command / compose file
        print("\n[Step 6] Checking how Docker container was started...")
        output, error, code = execute_command(ssh,
            "docker inspect quantum-brain --format='{{.Path}} {{.Args}}'")
        print(f"Container command: {output}")

        output, error, code = execute_command(ssh,
            "find /root -name 'docker-compose.yml' -o -name 'Dockerfile' 2>/dev/null | head -5")
        if output.strip():
            print("\nFound Docker files:")
            print(output)
            # Read docker-compose if exists
            output, error, code = execute_command(ssh,
                "cat /root/quantum-brain/docker-compose.yml 2>/dev/null || cat /root/QuantumDocker/docker-compose.yml 2>/dev/null")
            if output.strip():
                print("\nDocker Compose configuration:")
                print(output)

        # Step 7: Try to find and read the configuration
        print("\n[Step 7] Reading signal generator configuration...")
        output, error, code = execute_command(ssh,
            "docker exec quantum-brain head -100 /app/06_Integration/HybridBridge/etare_signal_generator_redux.py 2>/dev/null")
        if output.strip():
            print(output[:1500])

        # Step 8: Check for a way to stop mock mode
        print("\n[Step 8] Checking for configuration override...")
        output, error, code = execute_command(ssh,
            "docker exec quantum-brain ls -la /app/*.py /app/*.json /app/*.yaml 2>/dev/null | head -20")
        print(output)

        print("\n" + "=" * 60)
        print("SOLUTION")
        print("=" * 60)
        print("""
The system is running in MOCK MODE. To enable live trading:

1. Stop the Docker container:
   docker stop quantum-brain

2. Update environment variable to enable live trading:
   - Edit docker-compose.yml or restart with LIVE_MODE=true
   - Or update .env file to set MOCK_MODE=false

3. Restart with live trading enabled:
   docker start quantum-brain

Let me check the exact configuration...
""")

        ssh.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
