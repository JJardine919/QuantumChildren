# ETARE Redux Brain Deployment Guide

## Overview

This guide explains how to deploy the upgraded LSTM-based signal generator to your VPS Docker container.

**What Changed:**
- Old Brain: MLP (simple feed-forward network) - untrained
- New Brain: LSTM (recurrent network) - trained on 60 months of data

**Your Trained Champions:**
| Symbol | Fitness | Profit |
|--------|---------|--------|
| EURUSD | 0.0496 | $7,440 |
| GBPCHF | 0.0599 | $3,593 |
| NZDCHF | 0.0798 | $798 |
| GBPUSD | 0.0237 | $1,424 |

---

## Quick Deploy (Automated)

### Option A: PowerShell (Recommended)
```powershell
.\deploy_redux_brain.ps1
```

### Option B: Batch Script
```cmd
deploy_redux_brain.bat
```

---

## Manual Deployment

If the automated scripts don't work, follow these steps:

### Step 1: Export Champions (Already Done)
```bash
python export_champions.py --output-dir ./champions
```

### Step 2: Upload Files to VPS

```bash
# Create directory on VPS
ssh root@72.62.170.153 "mkdir -p /root/quantum-brain/champions /root/quantum-brain/01_Systems/System_03_ETARE /root/quantum-brain/06_Integration/HybridBridge"

# Upload Dockerfile
scp Dockerfile.redux root@72.62.170.153:/root/quantum-brain/Dockerfile

# Upload requirements
scp requirements.txt root@72.62.170.153:/root/quantum-brain/

# Upload champions
scp champions/* root@72.62.170.153:/root/quantum-brain/champions/

# Upload code
scp 01_Systems/System_03_ETARE/ETARE_module.py root@72.62.170.153:/root/quantum-brain/01_Systems/System_03_ETARE/
scp 06_Integration/HybridBridge/etare_signal_generator_redux.py root@72.62.170.153:/root/quantum-brain/06_Integration/HybridBridge/
```

### Step 3: Build Docker Image on VPS

```bash
ssh root@72.62.170.153

cd /root/quantum-brain
docker build -t quantum-brain-redux .
```

### Step 4: Stop Old Container & Start New

```bash
# Stop and remove old container
docker stop quantum-brain
docker rm quantum-brain

# Start new Redux container
docker run -d \
  --name quantum-brain \
  --restart unless-stopped \
  -v "/root/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files:/root/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files" \
  quantum-brain-redux
```

---

## Verification

### Check Container Status
```bash
ssh root@72.62.170.153 "docker ps"
```

### View Logs
```bash
ssh root@72.62.170.153 "docker logs -f quantum-brain"
```

Expected output:
```
[OK] Redux Signal Generator initialized
[OK] Signal file: /root/.wine/.../etare_signals.json
[OK] Champions loaded: ['EURUSD', 'GBPCHF', 'NZDCHF', 'GBPUSD']

============================================================
Signal Generation #0
============================================================
  [+] EURUSD     BUY    (67.3%)
  [=] GBPUSD     HOLD   (42.1%)
  [-] GBPCHF     SELL   (55.8%)
  ...
```

### Check Signal File
```bash
ssh root@72.62.170.153 "cat '/root/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/etare_signals.json'"
```

---

## Troubleshooting

### "No champion found" for a symbol
The signal generator will use HOLD for symbols without a trained champion. This is expected for symbols that haven't finished training yet.

### Docker build fails
Make sure requirements.txt exists and doesn't have Windows-only packages:
```bash
ssh root@72.62.170.153 "cd /root/quantum-brain && sed -i '/MetaTrader5/d' requirements.txt"
```

### Container keeps restarting
Check the logs for errors:
```bash
docker logs quantum-brain
```

### Signals not updating
1. Check if the container is running: `docker ps`
2. Check container logs: `docker logs quantum-brain`
3. Verify the signal file path matches MT5 executor expectations

---

## Adding More Champions

As training completes for more symbols:

1. Re-run the export on your local machine:
   ```bash
   python export_champions.py --output-dir ./champions
   ```

2. Upload the new champions:
   ```bash
   scp champions/champion_*.pth root@72.62.170.153:/root/quantum-brain/champions/
   scp champions/champions_manifest.json root@72.62.170.153:/root/quantum-brain/champions/
   ```

3. Restart the container to load new champions:
   ```bash
   ssh root@72.62.170.153 "docker restart quantum-brain"
   ```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE                           │
│  ┌─────────────────┐    ┌─────────────────────────────┐    │
│  │  Training       │    │  export_champions.py        │    │
│  │  (3 parallel)   │───▶│  Extracts best LSTM weights │    │
│  │  ETARE_Redux.py │    │  → champions/*.pth          │    │
│  └─────────────────┘    └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                               │
                               │ SCP Upload
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                       VPS (72.62.170.153)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Docker: quantum-brain                               │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  etare_signal_generator_redux.py            │    │   │
│  │  │  - Loads LSTM champions                     │    │   │
│  │  │  - Fetches M5 data                          │    │   │
│  │  │  - Generates BUY/SELL/HOLD signals          │    │   │
│  │  │  - Writes to etare_signals.json             │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                               │                             │
│                               ▼ JSON Signal File            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  MT5 via Wine                                       │   │
│  │  - ETARE_Executor.mq5 reads signals                 │   │
│  │  - Executes trades                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## File Inventory

| File | Purpose |
|------|---------|
| `export_champions.py` | Extracts best models from SQLite DB |
| `champions/*.pth` | PyTorch model weights |
| `champions/champions_manifest.json` | Metadata (fitness, profit) |
| `Dockerfile.redux` | Container build instructions |
| `etare_signal_generator_redux.py` | Main signal generation logic |
| `deploy_redux_brain.ps1` | Automated deployment (PowerShell) |
| `deploy_redux_brain.bat` | Automated deployment (Batch) |

---

*Last Updated: January 22, 2026*
