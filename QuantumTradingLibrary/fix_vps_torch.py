import paramiko
import time

VPS_HOST = "72.62.170.153"
VPS_USER = "root"
VPS_PASS = "gXRCBtbi21##"

def execute_command(ssh, command, timeout=300): # Long timeout for pip install
    print(f"Executing: {command}...")
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    # Stream output to avoid hanging
    while not stdout.channel.exit_status_ready():
        time.sleep(1)
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')
    return output, error

def main():
    print(f"Fixing PyTorch on VPS {VPS_HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS, timeout=10)
        
        # 1. Uninstall broken torch
        print("\n[1] Uninstalling broken PyTorch...")
        # using --break-system-packages if on newer debian/ubuntu
        out, err = execute_command(ssh, "pip3 uninstall -y torch torchvision torchaudio --break-system-packages")
        print(out)
        
        # 2. Install clean CPU PyTorch
        print("\n[2] Installing CPU PyTorch...")
        out, err = execute_command(ssh, "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages")
        print(out)
        if err:
            print(f"Errors (if any): {err}")
            
        # 3. Try to start the brain again
        print("\n[3] Restarting Brain...")
        cmd = (
            "cd /root/QuantumTradingLibrary && "
            "nohup python3 06_Integration/HybridBridge/etare_signal_generator_redux.py "
            "> etare_redux_live.log 2> etare_redux_err.log &"
        )
        execute_command(ssh, cmd)
        
        # 4. Verify
        time.sleep(5)
        out, _ = execute_command(ssh, "ps aux | grep etare_signal_generator")
        if "etare_signal_generator_redux.py" in out:
            print("✅ SUCCESS: Brain is running now!")
            print(out)
        else:
            print("❌ FAILURE: Still crashing.")
            out, _ = execute_command(ssh, "cat /root/QuantumTradingLibrary/etare_redux_err.log")
            print("-- New Error Log --")
            print(out)

        ssh.close()

    except Exception as e:
        print(f"Fix failed: {e}")

if __name__ == "__main__":
    main()
