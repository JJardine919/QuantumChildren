#!/usr/bin/env python3
"""VPS SSH Configuration Script for GetLeveraged and Atlas accounts"""

import paramiko
import sys

# VPS Configuration
VPS_HOST = "72.62.170.153"
VPS_USER = "root"
VPS_PASS = "gXRCBtbi21##"

def execute_command(ssh, command, timeout=30):
    """Execute a command on the VPS and return output"""
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    return output, error

def main():
    # Connect to VPS
    print(f"Connecting to VPS {VPS_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=30)
        print("Connected successfully!")

        # Check current configuration
        print("\n=== Current .env Configuration ===")
        output, error = execute_command(ssh, "cat /root/.env")
        print(output)

        # Check running processes
        print("\n=== Running Trading Processes ===")
        output, error = execute_command(ssh, "ps aux | grep -E 'python|wine|mt5' | grep -v grep")
        print(output)

        # Check MT5 wine directory structure
        print("\n=== MT5 Wine Directory ===")
        output, error = execute_command(ssh, "ls -la /root/.wine_mt5/drive_c/Program\\ Files/ 2>/dev/null || echo 'Wine MT5 directory not found'")
        print(output)

        # Check existing account configurations
        print("\n=== Looking for account configuration files ===")
        output, error = execute_command(ssh, "find /root -name '*.env' -o -name '*config*.json' -o -name '*accounts*' 2>/dev/null | head -20")
        print(output)

        # Check ETARE Trading directory
        print("\n=== ETARE Trading Directory ===")
        output, error = execute_command(ssh, "ls -la /root/ETARE_Trading/ 2>/dev/null || ls -la /root/QuantumTradingLibrary/ 2>/dev/null")
        print(output)

        ssh.close()
        print("\nConnection closed.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
