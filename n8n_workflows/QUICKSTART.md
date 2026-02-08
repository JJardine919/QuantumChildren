# Quick Start - 5 Minutes to Your Own Signal Network

## Option A: Just Receive Signals (Easiest)

**Time: 2 minutes**

1. Join our Telegram channel: [Coming Soon]
2. Done. You'll receive signals.

---

## Option B: Run Your Own n8n Node

**Time: 10 minutes**

### Step 1: Get n8n Running

**Cloud (easiest):**
- Sign up at https://n8n.io (free tier available)

**Self-hosted:**
```bash
npx n8n
```
Or with Docker:
```bash
docker run -p 5678:5678 n8nio/n8n
```

### Step 2: Import Workflow

1. Open n8n: http://localhost:5678
2. Create account (first time only)
3. Click "Workflows" -> "Import from File"
4. Select `QuantumChildren_MINIMAL_Receiver.json`

### Step 3: Configure Telegram

1. Message @BotFather on Telegram
2. Send `/newbot` and follow prompts
3. Copy your bot token
4. In n8n: Settings -> Credentials -> Create New -> Telegram
5. Paste your bot token

### Step 4: Set Your Chat ID

1. Message your bot on Telegram
2. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Find your `chat_id` in the response
4. In n8n: Settings -> Variables -> Add `TELEGRAM_CHAT_ID`

### Step 5: Activate

1. Toggle workflow to Active
2. Note your webhook URL: `http://your-server:5678/webhook/qc-signal`
3. Share this URL to receive signals

---

## Option C: Run Full Signal Generation

**Time: 30 minutes**

### Prerequisites
- Python 3.11+
- MetaTrader 5 terminal installed and logged in
- n8n running (see Option B)

### Step 1: Install Dependencies
```bash
pip install flask requests MetaTrader5
```

### Step 2: Start Webhook Server
```bash
cd n8n_workflows
python quantum_webhook_server.py --port 8889
```

### Step 3: Start Signal Emitter
In another terminal:
```bash
python signal_emitter.py --service --webhook http://localhost:5678/webhook/quantum-signal
```

### Step 4: Verify
- Visit http://localhost:8889 - should show server info
- Visit http://localhost:8889/signals/latest - should show signals
- Check your Telegram - should receive alerts

---

## Windows Quick Start

Double-click `START_NETWORK.bat` to run the webhook server with demo signals.

---

## What Happens Next

1. Your node receives/generates signals
2. Quality signals broadcast to your Telegram
3. Share your webhook URL with others
4. Network grows organically
5. More nodes = better signal consensus

---

## Troubleshooting

**No signals appearing:**
- Check n8n workflow is Active (green toggle)
- Verify Telegram credentials are correct
- Check webhook URL is accessible

**"Connection refused" errors:**
- Ensure n8n is running on correct port
- Check firewall allows port 5678

**Signals but no Telegram messages:**
- Verify `TELEGRAM_CHAT_ID` is set correctly
- Check Telegram bot token is valid
- Ensure you've messaged the bot first

---

## Get Help

- Check the full README.md for detailed documentation
- GitHub Issues: [Coming Soon]
- Telegram Support: [Coming Soon]
