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
pip3 install -r requirements.txt
```

### 6. Set Environment Variables

Create an env file on the VPS:
```bash
cat > /opt/quantumchildren/.env << 'EOF'
# Admin API key for protected endpoints (/backtest, /compile)
QC_ADMIN_KEY=your-admin-key-here

# Base44 Dashboard Webhook Forwarding
# Get the webhook URL from Base44: Dashboard > Code > Functions > ingestSignal
# Copy the URL and REMOVE the /ingestSignal from the end.
# Example: https://app--quantum-children.base44.app/api/apps/abc123/functions
BASE44_WEBHOOK_URL=

# Must match the QC_WEBHOOK_KEY secret stored in Base44 Dashboard > Secrets
BASE44_WEBHOOK_KEY=
EOF

chmod 600 /opt/quantumchildren/.env
```

### 7. Start server
```bash
chmod +x start_server.sh
./start_server.sh
```

### 8. Test it's working
```bash
curl http://localhost:8888/stats
```

## Keep Running After Logout

Option A - Use screen:
```bash
screen -S quantum
source /opt/quantumchildren/.env && export BASE44_WEBHOOK_URL BASE44_WEBHOOK_KEY QC_ADMIN_KEY
python3 collection_server.py
# Press Ctrl+A then D to detach
```

Option B - Use systemd service (RECOMMENDED):
```bash
cat > /etc/systemd/system/quantumchildren.service << EOF
[Unit]
Description=QuantumChildren Collection Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/quantumchildren
EnvironmentFile=/opt/quantumchildren/.env
ExecStart=/usr/bin/gunicorn -w 4 -b 0.0.0.0:8888 collection_server:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
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
# Server health
curl http://203.161.61.61:8888/stats

# Service status
systemctl status quantumchildren

# Logs
journalctl -u quantumchildren -f
```

## Updating

When you update `collection_server.py`:
```bash
scp collection_server.py root@203.161.61.61:/opt/quantumchildren/
ssh root@203.161.61.61 "systemctl restart quantumchildren"
```

## Base44 Integration

The collection server forwards every incoming signal and outcome to your Base44 app
in the background. This happens automatically when `BASE44_WEBHOOK_URL` and
`BASE44_WEBHOOK_KEY` are set in the environment.

- Forwarding is fire-and-forget (never slows down the node response)
- If Base44 is unreachable, the VPS keeps collecting normally
- All data is always stored locally in SQLite regardless of forwarding status

### To connect:
1. In Base44: Dashboard > Secrets > add `QC_WEBHOOK_KEY` with your key
2. In Base44: Dashboard > Code > Functions > ingestSignal > copy the URL
3. On VPS: Set `BASE44_WEBHOOK_URL` (the URL minus `/ingestSignal`) and `BASE44_WEBHOOK_KEY` (same key)
4. Restart the service: `systemctl restart quantumchildren`
