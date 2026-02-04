import argparse
import subprocess
import sys
import time

# Configuration
VPS_IP = "72.62.170.153"
VPS_USER = "root"

def run_remote_command(command, description):
    """Executes a command on the VPS via SSH."""
    print(f"\n[+] {description}...")
    ssh_command = f"ssh {VPS_USER}@{VPS_IP} \"{command}\""
    
    try:
        # Use shell=True to handle the SSH string correctly on Windows
        result = subprocess.run(ssh_command, shell=True)
        if result.returncode != 0:
            print(f"[-] Error: Command failed with code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"[-] Execution error: {e}")
        return False

def install_mt5(url):
    print(f"=== Installing MT5 on {VPS_IP} ===")
    
    commands = [
        ("apt update", "Updating package lists"),
        ("dpkg --add-architecture i386 && apt update", "Adding 32-bit architecture support"),
        ("apt install -y wine64 wine32 xvfb x11vnc wget", "Installing Wine, Xvfb, and dependencies"),
        (f"wget {url} -O mt5setup.exe", "Downloading MT5 Installer"),
        ("pkill Xvfb; Xvfb :99 -screen 0 1024x768x24 &", "Starting Virtual Display (:99)"),
        ("export DISPLAY=:99 && wine mt5setup.exe /auto", "Running MT5 Installer")
    ]

    for cmd, desc in commands:
        if not run_remote_command(cmd, desc):
            print("\n!!! Installation stopped due to error.")
            return

    print("\n=== Installation Complete ===")
    print("To view the GUI, run this command on your VPS:")
    print("x11vnc -display :99 -forever -nopw")
    print(f"Then connect your VNC Viewer to {VPS_IP}:5900")

def check_status():
    print(f"=== VPS Status ({VPS_IP}) ===")
    
    cmds = [
        "uptime",
        "df -h / | grep /",
        "free -m | grep Mem",
        "pgrep -a wine || echo 'MT5 not running'",
        "pgrep -a Xvfb || echo 'Display server not running'"
    ]
    
    combined_cmd = " && echo '---' && ".join(cmds)
    run_remote_command(combined_cmd, "Fetching System Metrics")

def main():
    parser = argparse.ArgumentParser(description="Hostinger VPS Manager")
    parser.add_argument("--action", choices=["install_mt5", "check_status"], required=True, help="Action to perform")
    parser.add_argument("--url", default="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe", help="Custom MT5 Installer URL")
    
    args = parser.parse_args()
    
    if args.action == "install_mt5":
        install_mt5(args.url)
    elif args.action == "check_status":
        check_status()

if __name__ == "__main__":
    main()
