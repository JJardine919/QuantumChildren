import subprocess
import time
import sys

# Configuration
VPS_IP = "72.62.170.153"
VPS_USER = "root"

def run_ssh(command):
    """Runs a command on the VPS via SSH"""
    ssh_cmd = f"ssh {VPS_USER}@{VPS_IP} \"{command}\""
    print(f"Sending: {command[:50]}...")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

print(f"Connecting to {VPS_IP} to take over installation...")

# 1. Kill any stuck processes
print("\n[1/5] Cleaning up stuck processes...")
run_ssh("pkill -f wine; pkill -f Xvfb; rm -f /tmp/.X99-lock")

# 2. Install the 'Virtual Finger' tool (xdotool)
print("\n[2/5] Installing automation tools (xdotool)...")
run_ssh("apt-get update && apt-get install -y xdotool xvfb wine64 wine32")

# 3. Start Invisible Display
print("\n[3/5] Starting invisible screen...")
run_ssh("nohup Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &")
time.sleep(2)

# 4. Launch MT5 Installer & Press Buttons
print("\n[4/5] Launching MT5 and pressing 'Next' automatically...")
# This massive command starts the installer and uses xdotool to hit 'Enter' repeatedly
blind_robot_cmd = (
    "export DISPLAY=:99; "
    "wine mt5setup.exe /auto & "
    "PID=$!; "
    "sleep 5; "
    "for i in {1..20}; do "
        "xdotool key Return; "
        "xdotool key Tab; "
        "sleep 2; "
    "done; "
    "wait $PID"
)
run_ssh(blind_robot_cmd)

# 5. Verify
print("\n[5/5] Checking if it worked...")
check = run_ssh("ls -R ~/.wine/drive_c/Program\ Files/MetaTrader\ 5 | grep terminal64.exe")

if "terminal64.exe" in check:
    print("\nSUCCESS! MetaTrader 5 is installed.")
    print("You can now close your terminal.")
else:
    print("\nInstallation might have failed. Output was:")
    print(check)
