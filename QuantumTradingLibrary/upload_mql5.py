
import paramiko
import os

VPS_HOST = "72.62.170.153"
VPS_USER = "root"
VPS_PASS = "gXRCBtbi21##"

def upload():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS)
    
    sftp = ssh.open_sftp()
    
    mt5_path = "/root/.wine/drive_c/Program Files/MetaTrader 5/MQL5"
    
    print("Uploading DataExporter.ex5 to Services...")
    sftp.put("DataExporter.ex5", f"{mt5_path}/Services/DataExporter.ex5")
    
    print("Uploading BG_Executor.ex5 to Experts...")
    sftp.put("BG_Executor.ex5", f"{mt5_path}/Experts/BG_Executor.ex5")
    
    print("Uploading mq5 files for reference...")
    sftp.put("DataExporter.mq5", f"{mt5_path}/Services/DataExporter.mq5")
    sftp.put("BG_Executor.mq5", f"{mt5_path}/Experts/BG_Executor.mq5")
    
    sftp.close()
    ssh.close()
    print("Upload complete!")

if __name__ == "__main__":
    upload()
