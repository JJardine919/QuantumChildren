# Blue Guardian Integrated System v2.0

## ETARE + Quantum Compression + LLM Watchdog

---

## What's New in v2.0

| Component | Description | Performance |
|-----------|-------------|-------------|
| **ETARE System** | Evolutionary Trading Algorithm - 50 adaptive strategies | ~65% win rate target |
| **Quantum Compression** | Qiskit quantum features - 7 metrics | +14% accuracy boost |
| **LLM Watchdog** | Gemma 3 12B emergency shutoff | Intelligent risk management |
| **Archiver** | SQLite-based quantum feature storage | Fast retrieval + compression |

---

## What's In This Package

| File | Purpose |
|------|---------|
| `accounts_config.json` | **EDIT THIS** - Your account credentials |
| `bg_brain_integrated.py` | Main brain (ETARE + Quantum + LLM) |
| `archiver_service.py` | Quantum feature archiver |
| `llm_watchdog.py` | LLM emergency shutoff system |
| `BG_MultiExecutor.mq5` | Trade executor EA (v2.1) |
| `BG_DataExporter.mq5` | Market data exporter (v2.1) |
| `docker-compose.yml` | Full stack deployment |
| `Dockerfile.brain` | Brain container |
| `Dockerfile.archiver` | Archiver container |

---

## System Architecture

```
                      NAMECHEAP VPS
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │   ┌───────────────────────────────────────────┐     │
    │   │         DOCKER CONTAINERS                  │     │
    │   │                                           │     │
    │   │  ┌─────────────────────────────────────┐  │     │
    │   │  │     bg_brain_integrated             │  │     │
    │   │  │     ────────────────────            │  │     │
    │   │  │  ┌─────────┐  ┌─────────┐          │  │     │
    │   │  │  │  ETARE  │  │ QUANTUM │          │  │     │
    │   │  │  │  50 ind │  │ 7 feat  │          │  │     │
    │   │  │  └────┬────┘  └────┬────┘          │  │     │
    │   │  │       └─────┬──────┘               │  │     │
    │   │  │             ▼                      │  │     │
    │   │  │     ┌─────────────┐                │  │     │
    │   │  │     │  ENSEMBLE   │                │  │     │
    │   │  │     │  DECISION   │                │  │     │
    │   │  │     └──────┬──────┘                │  │     │
    │   │  │            ▼                       │  │     │
    │   │  │  ┌──────────────────┐              │  │     │
    │   │  │  │  LLM WATCHDOG    │ ←─ Gemma 3   │  │     │
    │   │  │  │  Emergency Stop  │    12B       │  │     │
    │   │  │  └──────────────────┘              │  │     │
    │   │  └─────────────────────────────────────┘  │     │
    │   │                                           │     │
    │   │  ┌─────────────────┐  ┌─────────────────┐ │     │
    │   │  │     OLLAMA      │  │    ARCHIVER     │ │     │
    │   │  │   (Gemma 3 12B) │  │ (Quantum Store) │ │     │
    │   │  └─────────────────┘  └─────────────────┘ │     │
    │   └───────────────────────────────────────────┘     │
    │                      │                              │
    │                      ▼ JSON Signals                 │
    │   ┌───────────────────────────────────────────┐     │
    │   │              MT5 TERMINALS                │     │
    │   │                                           │     │
    │   │  [INSTANT 1]   [INSTANT 2]   [COMPETITION]│     │
    │   │  DataExporter  DataExporter  DataExporter │     │
    │   │  Executor v2.1 Executor v2.1 Executor v2.1│     │
    │   └───────────────────────────────────────────┘     │
    └─────────────────────────────────────────────────────┘
```

---

## Step 1: Configure Your Accounts

Edit `accounts_config.json`:

```json
{
  "accounts": [
    {
      "name": "BG_INSTANT_1",
      "account_id": "YOUR_ACCOUNT_ID",      // <- CHANGE THIS
      "password": "YOUR_PASSWORD",          // <- CHANGE THIS
      "server": "BlueGuardian-Server",
      "symbol": "BTCUSD",
      "magic_number": 100001,
      "account_type": "instant_challenge",
      "max_lot_size": 0.5,
      "daily_dd_limit": 5.0,
      "max_dd_limit": 10.0
    }
  ],
  "vps": {
    "host": "YOUR_VPS_IP",                  // <- CHANGE THIS
    "user": "root",
    "password": "YOUR_VPS_PASSWORD"         // <- CHANGE THIS
  }
}
```

---

## Step 2: Deploy to VPS

### Option A: Docker Compose (Recommended)

```bash
# Upload files to VPS
scp -r BlueGuardian_Deploy/* root@YOUR_VPS_IP:/opt/blueguardian/

# SSH to VPS
ssh root@YOUR_VPS_IP

# Navigate to directory
cd /opt/blueguardian

# Pull LLM model first (important!)
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull gemma2:2b

# Start full stack
docker-compose up -d
```

### Option B: Manual Python

```bash
# Install dependencies
pip install -r requirements_brain.txt

# Install Ollama (for LLM watchdog)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gemma2:2b

# Run brain
python bg_brain_integrated.py --interval 30
```

---

## Step 3: Set Up MT5 Terminals

For EACH Blue Guardian account:

1. **Install MT5 Terminal** via Wine
   ```bash
   wine64 "MT5Setup.exe"
   ```

2. **Log into account**

3. **Start Data Exporter Service**
   - Navigate to: Services > BG_DataExporter
   - Settings:
     - Symbols: `BTCUSD`
     - BarsToExport: `500`
     - UpdateInterval: `30`
     - UseCommonFolder: `true`

4. **Attach BG_MultiExecutor EA** to BTCUSD chart
   - Settings:
     - AccountName: `BG_INSTANT_1` (match config)
     - MagicNumber: `100001` (match config)
     - MaxLotSize: `0.5`
     - DailyDDLimit: `4.0`
     - MaxDDLimit: `8.0`
     - TradeEnabled: `false` (initially!)

---

## Step 4: Verify Everything Works

### Check Brain Status
```bash
docker logs bg_brain_integrated --tail 100
```

Expected output:
```
BLUE GUARDIAN INTEGRATED BRAIN v2.0
ETARE + Quantum Compression + LLM Watchdog
==================================================
Quantum extractor initialized: 3 qubits, 1000 shots
ETARE system initialized: 50 individuals
LLM Watchdog initialized with model: gemma2:2b
Blue Guardian Brain initialized
Brain started, checking every 30s
[BG_INSTANT_1] NORMAL: All metrics within acceptable range
Signal written: BG_INSTANT_1 -> BUY (68.3%)
ETARE evolved to generation 1, best fitness: 0.0234
```

### Check Watchdog Status
```bash
docker exec bg_brain_integrated python -c "from llm_watchdog import get_watchdog; w=get_watchdog(); print(w.get_status())"
```

### Check Archiver Stats
```bash
docker exec bg_archiver python -c "from archiver_service import get_archiver; a=get_archiver('/app/archive/quantum_archive.db'); print(a.get_stats())"
```

### Check Signal Files
```bash
ls -la /opt/blueguardian/signals/
cat /opt/blueguardian/signals/signal_BG_INSTANT_1.json
```

Expected:
```json
{
  "timestamp": "2026-01-29T14:30:00",
  "symbol": "BTCUSD",
  "action": "BUY",
  "confidence": 0.683,
  "etare_action": "BUY",
  "quantum_features": {
    "quantum_entropy": 2.345,
    "dominant_state_prob": 0.187,
    "phase_coherence": 0.723,
    "entanglement_degree": 0.556
  },
  "watchdog_status": "NORMAL",
  "status": "OK"
}
```

---

## Step 5: Go Live

Once verified:

1. **In each MT5 terminal**, open EA settings
2. **Change** `TradeEnabled` = `true`
3. **Monitor** via logs and Telegram (if configured)

---

## Emergency Procedures

### LLM Watchdog Triggered

If you see:
```
CRITICAL for BG_INSTANT_1: Daily drawdown 4.2% at critical level
Emergency signal written: signal_BG_INSTANT_1.json
```

The system has automatically:
1. Written a BLOCK signal to the account
2. The EA will stop opening new trades
3. Existing positions remain (unless CLOSE_ALL triggered)

### Manual Emergency Stop

```bash
# Stop brain
docker stop bg_brain_integrated

# Or write manual block signal
echo '{"status":"BLOCKED","action":"CLOSE_ALL","block_reason":"Manual stop"}' > signal_BG_INSTANT_1.json
```

---

## Component Details

### ETARE (Evolutionary Trading)

- **Population**: 50 neural network individuals
- **Evolution**: Genetic crossover + mutation
- **Extinction Events**: Removes weak strategies periodically
- **Target**: 65% win rate through adaptive learning

### Quantum Compression

- **Qubits**: 3 (8 quantum states)
- **Shots**: 1000 measurements per feature extraction
- **Features**:
  1. Quantum Entropy - market uncertainty
  2. Dominant State Probability - trend strength
  3. Superposition Measure - volatility indicator
  4. Phase Coherence - trend consistency
  5. Entanglement Degree - pattern memory
  6. Quantum Variance - state dispersion
  7. Significant States - market complexity

### LLM Watchdog

- **Model**: Gemma 3 12B (via Ollama)
- **Checks**: Account metrics, quantum features, market context
- **Actions**:
  - NORMAL - trading allowed
  - CAUTION - reduced position sizes
  - WARNING - minimum trades only
  - CRITICAL - no new trades
  - EMERGENCY - close all positions

---

## Troubleshooting

### Brain Not Starting

```bash
docker-compose logs bg-brain
# Check for Python errors

# Restart
docker-compose restart bg-brain
```

### Ollama Not Responding

```bash
# Check if running
docker ps | grep ollama

# Restart
docker restart bg_ollama

# Re-pull model
docker exec bg_ollama ollama pull gemma2:2b
```

### No Quantum Features

```bash
# Check if Qiskit installed
docker exec bg_brain_integrated pip list | grep qiskit

# If missing, it falls back to pseudo-quantum
# This still works but less accurate
```

### EA Not Executing

1. Check signal file exists
2. Verify magic numbers match
3. Check `TradeEnabled` = true
4. Review EA journal logs in MT5

---

## Performance Expectations

| Metric | Target |
|--------|--------|
| Win Rate | 60-65% |
| Daily Trades | 1-5 |
| Max Drawdown | <8% |
| Profit Factor | >1.5 |
| Signal Latency | <500ms |

---

## Support

Monitor your system via:
- Docker logs
- MT5 EA journal
- Archive database stats

Good luck with your challenges!
