
import paramiko
import os

VPS_HOST = "72.62.170.153"
VPS_USER = "root"
VPS_PASS = "gXRCBtbi21##"

def run():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS)
    
    print("Listing MQL5/Files...")
    stdin, stdout, stderr = ssh.exec_command("ls -l '/root/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files'")
    print(stdout.read().decode())
    
    print("Checking if DataExporter.mq5 is in Services...")
    stdin, stdout, stderr = ssh.exec_command("ls -l '/root/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Services'")
    print(stdout.read().decode())

    print("Checking if BG_Executor.mq5 is in Experts...")
    stdin, stdout, stderr = ssh.exec_command("ls -l '/root/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts'")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    run()
