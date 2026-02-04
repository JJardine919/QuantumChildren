import paramiko
import os

HOST = '203.161.61.61'
USER = 'root'
PASS = 'tg1MNYK98Vt09no8uN'
LOCAL_MT5 = r'C:\Program Files\Blue Guardian MT5 Terminal'
REMOTE_MT5 = '/root/.wine/drive_c/Program Files/Blue Guardian MT5 Terminal'

def upload_directory(sftp, local_dir, remote_dir):
    """Upload directory recursively"""
    count = 0
    for root, dirs, files in os.walk(local_dir):
        # Create relative path
        rel_path = os.path.relpath(root, local_dir)
        if rel_path == '.':
            current_remote = remote_dir
        else:
            current_remote = f"{remote_dir}/{rel_path}".replace('\\', '/')

        # Create remote directory
        try:
            sftp.mkdir(current_remote)
        except:
            pass

        # Upload files
        for f in files:
            local_file = os.path.join(root, f)
            remote_file = f"{current_remote}/{f}"
            try:
                sftp.put(local_file, remote_file)
                count += 1
                if count % 50 == 0:
                    print(f"Uploaded {count} files...")
            except Exception as e:
                print(f"Error uploading {f}: {e}")

    return count

def main():
    print("Connecting to VPS...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS)

    # Create base directory
    stdin, stdout, stderr = client.exec_command(f'mkdir -p "{REMOTE_MT5}"')
    stdout.read()

    print(f"Uploading Blue Guardian MT5 Terminal...")
    print(f"From: {LOCAL_MT5}")
    print(f"To: {REMOTE_MT5}")

    sftp = client.open_sftp()
    count = upload_directory(sftp, LOCAL_MT5, REMOTE_MT5)
    sftp.close()

    print(f"\nUploaded {count} files total")

    # Verify
    stdin, stdout, stderr = client.exec_command(f'ls -la "{REMOTE_MT5}" | head -10')
    print("\nVerification:")
    print(stdout.read().decode())

    client.close()
    print("Done!")

if __name__ == '__main__':
    main()
