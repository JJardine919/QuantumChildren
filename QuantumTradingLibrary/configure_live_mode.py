#!/usr/bin/env python3
"""Configure System for Live Trading via MT5 Bridge"""

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
    output = stdout.read().decode('utf-8', errors='ignore')
    error = stderr.read().decode('utf-8', errors='ignore')
    exit_code = stdout.channel.recv_exit_status()
    return output, error, exit_code

def main():
    print("=" * 60)
    print("Configure Live Trading Mode")
    print("=" * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"\nConnecting to VPS {VPS_HOST}...")
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=30)
        print("Connected!")

        # Step 1: Check MT5 Wine Bridge configuration
        print("\n[Step 1] Checking MT5 Wine Bridge...")
        output, error, code = execute_command(ssh,
            "ps aux | grep mt5_wine_bridge | grep -v grep")
        print(output)

        # Step 2: Read the bridge code to understand the API
        print("\n[Step 2] Reading MT5 Bridge API...")
        output, error, code = execute_command(ssh,
            "head -150 /opt/etare/bridge/mt5_wine_bridge.py 2>/dev/null", timeout=10)
        print(output[:2000] if len(output) > 2000 else output)

        # Step 3: Check if signal generator is configured to use bridge
        print("\n[Step 3] Checking signal generator bridge configuration...")
        output, error, code = execute_command(ssh,
            "docker exec quantum-brain grep -A 10 -B 10 'wine_bridge\\|BRIDGE\\|API_KEY' /app/06_Integration/HybridBridge/etare_signal_generator_redux.py 2>/dev/null | head -50")
        print(output)

        # Step 4: Check for environment variables that control live vs mock
        print("\n[Step 4] Looking for configuration that enables live mode...")
        output, error, code = execute_command(ssh,
            "docker exec quantum-brain grep -n 'MockMT5\\|USE_MOCK\\|LIVE_MODE' /app/06_Integration/HybridBridge/etare_signal_generator_redux.py 2>/dev/null | head -20")
        print(output)

        # Step 5: Check quantum-brain directory for config
        print("\n[Step 5] Checking quantum-brain source directory...")
        output, error, code = execute_command(ssh,
            "ls -la /root/quantum-brain/ 2>/dev/null | head -20")
        print(output)

        output, error, code = execute_command(ssh,
            "cat /root/quantum-brain/.env 2>/dev/null || echo 'No .env in quantum-brain'")
        print(output)

        # Step 6: Check Dockerfile to see build process
        print("\n[Step 6] Checking Dockerfile...")
        output, error, code = execute_command(ssh,
            "cat /root/quantum-brain/Dockerfile 2>/dev/null | head -40")
        print(output)

        # Step 7: Check docker-compose
        print("\n[Step 7] Checking docker-compose configuration...")
        output, error, code = execute_command(ssh,
            "cat /root/quantum-brain/docker-compose.yml 2>/dev/null || cat /root/autonomous-revenue-engine/docker-compose.yml 2>/dev/null | head -50")
        print(output)

        # Step 8: Propose solution
        print("\n" + "=" * 60)
        print("SOLUTION TO ENABLE LIVE TRADING")
        print("=" * 60)
        print("""
Based on analysis:
1. Signal generator uses MockMT5 when MetaTrader5 module unavailable
2. MT5 Wine Bridge is running at /opt/etare/bridge/mt5_wine_bridge.py
3. Signal generator needs to be configured to use the bridge API

Options:

A) STOP DOCKER AND USE NATIVE:
   1. Stop Docker container: docker stop quantum-brain
   2. Run signal generator directly: cd /root/QuantumTradingLibrary && python3 06_Integration/HybridBridge/etare_signal_generator_redux.py
   3. This will use real MT5 via Wine bridge

B) CONFIGURE DOCKER TO USE BRIDGE:
   1. Add MT5_BRIDGE_URL environment variable to docker-compose
   2. Modify signal generator to use bridge API instead of direct MT5
   3. Rebuild and restart container

C) INSTALL MT5 IN DOCKER:
   1. Add MetaTrader5 to Docker image
   2. Configure Wine in Docker
   3. Rebuild container

Recommendation: Option A (Stop Docker, use native) is fastest.

Would you like me to stop the Docker container and run natively?
""")

        ssh.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
