import paramiko
import time

VPS_HOST = "72.62.170.153"
VPS_USER = "root"
VPS_PASS = "gXRCBtbi21##"

def run():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS)
    
    print("Killing existing MT5 processes...")
    ssh.exec_command("pkill -9 -f terminal64.exe")
    ssh.exec_command("pkill -9 -f start.exe")
    time.sleep(2)
    
    print("Starting MT5 in portable mode...")
    # Use the command found in ps aux
    # root 140798 ... start.exe /exec terminal64.exe
    # root 140866 ... C:\Program Files\MetaTrader 5\terminal64.exe /portable
    
    # We need to run it via Wine start.
    cmd = "export WINEPREFIX=/root/.wine; nohup wine '/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe' /portable > /dev/null 2>&1 &"
    ssh.exec_command(cmd)
    
    print("MT5 restart command sent.")
    time.sleep(5)
    
    print("Checking if MT5 is running...")
    stdin, stdout, stderr = ssh.exec_command("ps aux | grep terminal64.exe | grep -v grep")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    run()
