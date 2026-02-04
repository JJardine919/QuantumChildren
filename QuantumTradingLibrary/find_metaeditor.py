
import paramiko

VPS_HOST = "72.62.170.153"
VPS_USER = "root"
VPS_PASS = "gXRCBtbi21##"

def run():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VPS_HOST, username=VPS_USER, password=VPS_PASS)
    
    print("Finding metaeditor64.exe...")
    stdin, stdout, stderr = ssh.exec_command("find / -name metaeditor64.exe 2>/dev/null")
    print(stdout.read().decode())
    
    ssh.close()

if __name__ == "__main__":
    run()
