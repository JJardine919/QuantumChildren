# Server Deployment Guide

## Quick Deploy to VPS

### 1. SSH into your VPS
```bash
ssh root@203.161.61.61
```

### 2. Create directory
```bash
mkdir -p /opt/quantumchildren
cd /opt/quantumchildren
```

### 3. Upload files (from your local machine)
```bash
scp -r SERVER/* root@203.161.61.61:/opt/quantumchildren/
```

### 4. Install Python and pip
```bash
apt update && apt install -y python3 python3-pip
```

### 5. Install requirements
```bash
pip3 install flask gunicorn
```

### 6. Start server
```bash
chmod +x start_server.sh
./start_server.sh
```

### 7. Test it's working
```bash
curl http://localhost:8888/stats
```

## Keep Running After Logout

Option A - Use screen:
```bash
screen -S quantum
python3 collection_server.py
# Press Ctrl+A then D to detach
```

Option B - Use systemd service:
```bash
cat > /etc/systemd/system/quantumchildren.service << EOF
[Unit]
Description=QuantumChildren Collection Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/quantumchildren
ExecStart=/usr/bin/gunicorn -w 4 -b 0.0.0.0:8888 collection_server:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable quantumchildren
systemctl start quantumchildren
```

## Firewall

Make sure port 8888 is open:
```bash
ufw allow 8888
```

## Check Status

```bash
curl http://203.161.61.61:8888/stats
```
