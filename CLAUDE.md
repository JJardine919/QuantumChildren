# QUANTUM CHILDREN - RULES FOR CLAUDE

**READ THIS FIRST. THESE RULES ARE NON-NEGOTIABLE.**

---

## CRITICAL: DO NOT CHANGE THESE VALUES

All trading settings are in `MASTER_CONFIG.json`. **NEVER hardcode values in Python scripts.**

### REQUIRED SETTINGS (DO NOT MODIFY WITHOUT USER PERMISSION):
```
SL (MAX_LOSS_DOLLARS):    $1.00
INITIAL_SL_DOLLARS:       $0.60
TP_MULTIPLIER:            3
ROLLING_SL_MULTIPLIER:    1.5
DYNAMIC_TP_PERCENT:       50
CONFIDENCE_THRESHOLD:     0.55
AGENT_SL_MIN:             $0.50
AGENT_SL_MAX:             $1.00
WATCHDOG_LIMIT:           $1.00
```

If you need to change ANY of these values, **ASK THE USER FIRST**.

---

## ARCHITECTURE RULES

### 1. ONE SCRIPT PER ACCOUNT
- Each trading account runs its OWN Python script
- Scripts do NOT switch between accounts
- `mt5.login()` should only be called ONCE at startup
- NEVER cycle through multiple accounts in one script

### 2. CONFIG FILE IS THE SOURCE OF TRUTH
- All scripts import from `config_loader.py`
- `config_loader.py` reads from `MASTER_CONFIG.json`
- To change a setting: edit `MASTER_CONFIG.json`, NOT the Python scripts
- Config supports hot-reload via `reload_config()`

### 3. STOP LOSS IS SACRED
- SL must be calculated for a FIXED DOLLAR AMOUNT
- Default: $1.00 max loss per trade
- Formula: `sl_distance = MAX_LOSS_DOLLARS / (tick_value * lot)`
- NEVER use ATR-based SL that can result in $500+ losses

### 4. DO NOT KILL TRADES
- Switching accounts with `mt5.login()` kills open trades
- Each terminal stays logged into ONE account
- Use separate windows/processes for multiple accounts

### 5. CREDENTIALS ARE IN .env
- Passwords are **NEVER** stored in MASTER_CONFIG.json or Python scripts
- Use `credential_manager.py` to load credentials from `.env`
- Copy `.env.example` to `.env` and add passwords there

---

## ACCOUNTS

| Key | Account | Name | Terminal | Enabled |
|-----|---------|------|----------|---------|
| GL_3 | 107245 | GetLeveraged (ACTIVE) | GetLeveraged MT5 Terminal | Yes |
| FTMO | 1521063483 | FTMO $200K Challenge | FTMO Global Markets MT5 Terminal | Yes |
| JIMMY_FTMO | 1512556097 | Jimmy's FTMO Challenge | FTMO Global Markets MT5 Terminal | Yes |
| BG_INSTANT | 366604 | BlueGuardian $5K Instant | Blue Guardian MT5 Terminal | Yes |
| BG_CHALLENGE | 365060 | BlueGuardian $100K Challenge | Blue Guardian MT5 Terminal 2 | Yes |
| ATLAS | 212000584 | Atlas Funded | Atlas Funded MT5 Terminal | Yes |
| GL_1 | 113326 | GetLeveraged (disabled) | -- | No |
| GL_2 | 113328 | GetLeveraged (disabled) | -- | No |

---

## SCRIPTS

All scripts live in `QuantumTradingLibrary/`.

### For Trading (run each in separate window):
- `BRAIN_GETLEVERAGED.py` -> GL_3 (107245) ONLY -- currently active
- `BRAIN_FTMO.py` -> FTMO (1521063483) -- $200K Challenge
- `BRAIN_JIMMY_FTMO.py` -> Jimmy's FTMO (1512556097) -- via SOCKS5 proxy on VPS_2
- `BRAIN_BG_INSTANT.py` -> 366604 only
- `BRAIN_BG_INSTANT_LLM.py` -> 366604 (LLM-optimized variant)
- `BRAIN_BG_CHALLENGE.py` -> 365060 only
- `BRAIN_ATLAS.py` -> 212000584 only

### For Safety:
- `STOPLOSS_WATCHDOG_V2.py` -> Monitors all positions, auto-applies emergency SL, detects rogue trades

### For Monitoring (read-only):
- `DASHBOARD_MONITOR.py` -> Shows all accounts without affecting trades
- `CHALLENGE_SIMULATOR.py` -> Simulates prop firm challenge rules

### For Config:
- `config_loader.py` -> Run to see current settings
- `credential_manager.py` -> Secure credential loading from .env
- `MASTER_CONFIG.json` -> Edit this for ALL settings

### For Training:
- `Master_Train.py` -> Train neural network experts
- `ETARE_50_Darwin.py` -> Generate 50-expert army via evolutionary selection
- `ETARE_BTCUSD_Trainer.py` -> Symbol-specific expert trainer
- `lstm_retrain_fast.py` -> Fast LSTM retraining (CPU only -- see GPU section)

### For Data Collection:
- `entropy_collector.py` -> Collects signals, saves locally, optionally syncs to server
- `auto_data_collector.py` -> Background market data collection
- `prelaunch_validator.py` -> Validates experts/data before allowing live trades

---

## DATA COLLECTION

Collection server: `http://203.161.61.61:8888` (currently **disabled** in config)

All scripts should send signals to this server using `entropy_collector.py`.
Local backup is always saved to `quantum_data/` regardless of server status.

---

## WHAT NOT TO DO

1. Do NOT hardcode SL/TP values in Python scripts
2. Do NOT use ATR-based stops (causes $500+ losses)
3. Do NOT switch accounts within a running script
4. Do NOT change MASTER_CONFIG.json values without asking
5. Do NOT create scripts that manage multiple accounts in one loop
6. Do NOT push directly to main -- use a feature branch (see GIT WORKFLOW below)
7. Do NOT store passwords in code or config files -- use .env via credential_manager.py

---

## GIT WORKFLOW -- MANDATORY

**`main` is production. BRAIN scripts run from main. Never push untested code to main.**

### For every change:

```bash
# 1. Create a feature branch BEFORE writing any code
git checkout -b feature/short-description-here

# 2. Do your work, commit as needed
git add <files>
git commit -m "description"

# 3. Push the branch (NOT main)
git push -u origin feature/short-description-here

# 4. Test and verify everything works

# 5. Only after testing: merge to main
git checkout main
git pull
git merge feature/short-description-here
git push

# 6. Clean up
git branch -d feature/short-description-here
```

### Rules:
- **NEVER** `git push` while on main without merging a tested branch
- **NEVER** `git push --force` to main
- **ASK** the user before merging to main
- Branch names: `feature/thing`, `fix/thing`, or `experiment/thing`
- If the user says "commit this" or "push this", create a branch first and confirm before merging to main

---

## CODEBASE STRUCTURE

```
QuantumChildren/
|
|-- CLAUDE.md                           # THIS FILE - rules for Claude
|-- .gitignore                          # Git ignore rules
|-- .claude/settings.json               # Claude Code plugin settings
|
|-- QuantumTradingLibrary/              # *** MAIN CODE DIRECTORY ***
|   |
|   |-- MASTER_CONFIG.json             # Central config (SOURCE OF TRUTH)
|   |-- config_loader.py               # Reads MASTER_CONFIG.json, provides constants
|   |-- credential_manager.py          # Loads passwords from .env
|   |-- entropy_collector.py           # Signal collection and local backup
|   |-- prelaunch_validator.py         # Pre-trade validation checks
|   |-- mcp_mt5_server.py             # MCP server for direct MT5 control
|   |
|   |-- BRAIN_GETLEVERAGED.py         # Trading: GL_3 (107245) -- PRIMARY
|   |-- BRAIN_FTMO.py                 # Trading: FTMO (1521063483)
|   |-- BRAIN_JIMMY_FTMO.py           # Trading: Jimmy's FTMO (1512556097)
|   |-- BRAIN_BG_INSTANT.py           # Trading: BG_INSTANT (366604)
|   |-- BRAIN_BG_INSTANT_LLM.py       # Trading: BG_INSTANT LLM variant
|   |-- BRAIN_BG_CHALLENGE.py         # Trading: BG_CHALLENGE (365060)
|   |-- BRAIN_ATLAS.py                # Trading: ATLAS (212000584)
|   |
|   |-- STOPLOSS_WATCHDOG_V2.py       # Safety: monitors SL, detects rogue trades
|   |-- DASHBOARD_MONITOR.py          # Read-only account monitoring
|   |-- CHALLENGE_SIMULATOR.py        # Prop firm challenge simulation
|   |
|   |-- ALGORITHM_BDELLOID_ROTIFERS.py # Bio-algo: horizontal gene transfer
|   |-- ALGORITHM_CANCER_CELL.py       # Bio-algo: mutation/adaptation (largest)
|   |-- ALGORITHM_CRISPR_CAS.py        # Bio-algo: protective gene editing
|   |-- ALGORITHM_PROTECTIVE_DELETION.py # Bio-algo: defensive trade deletion
|   |-- ALGORITHM_SYNCYTIN.py          # Bio-algo: signal fusion
|   |-- ALGORITHM_TOXOPLASMA.py        # Bio-algo: drawdown throttling
|   |-- ALGORITHM_HGH_HORMONE.py       # Bio-algo: growth amplification
|   |-- ALGORITHM_ELECTRIC_ORGANS.py   # Bio-algo: signal generation
|   |-- ALGORITHM_TE_DOMESTICATION.py  # Bio-algo: transposable element control
|   |-- ALGORITHM_KORV.py             # Bio-algo: KORV system
|   |
|   |-- teqa_bridge.py                # Signal bridge: TE Quantum Analysis
|   |-- qnif_bridge.py                # Signal bridge: Quantum-Immune (MT5-free)
|   |-- testosterone_dmt_bridge.py    # Signal bridge: Testosterone-DMT molecular
|   |-- stanozolol_dmt_bridge.py      # Signal bridge: Stanozolol-DMT molecular
|   |-- quantum_regime_bridge.py      # Signal bridge: Regime detection
|   |
|   |-- Master_Train.py               # Neural network training
|   |-- ETARE_50_Darwin.py            # Evolutionary expert generation
|   |-- lstm_retrain_fast.py          # Fast LSTM retraining (CPU)
|   |-- top_50_experts/               # Trained expert models and manifest
|   |
|   |-- MASTER_LAUNCH.py              # Master orchestration
|   |-- SIGNAL_FARM_ENGINE.py         # Multi-account signal farm
|   |-- auto_data_collector.py        # Background data collection
|   |
|   |-- test_*.py                     # Test files (16 total)
|   |
|   |-- 01_Systems/                   # Advanced trading subsystems
|   |   |-- QuantumAnalysis/          # Quantum Phase Estimation
|   |   |-- QuantumCompression/       # Data compression/encoding
|   |   |-- BioNeuralTrader/          # Hodgkin-Huxley neuron trading
|   |   |-- System_03_ETARE/          # ETARE quantum fusion
|   |   |-- System_04_HomeAccounting/ # Financial accounting
|   |   |-- System_05_GPT_MarketLanguage/ # LLM market analysis
|   |   |-- System_06_VolatilityPredictor/ # Volatility prediction
|   |   |-- System_07_CurrencyStrength/   # Forex strength analysis
|   |   |-- AdditionalSystems/        # Extra modules
|   |   +-- ...
|   |
|   |-- DEPLOY/                       # MQ5 Expert Advisors for MT5
|   |   |-- DEPLOY_MANIFEST.json      # Maps EAs to terminals
|   |   |-- *.mq5                     # EA source code
|   |   +-- MQL5_Experts/             # Compiled experts
|   |
|   |-- quantum_data/                 # Local signal data (gitignored)
|   |-- .claude/settings.local.json   # Claude local permissions
|   |
|   |-- README.md                     # Project overview
|   |-- SYSTEM_ARCHITECTURE.md        # Architecture diagrams
|   |-- QUICKSTART.md                 # Quick reference
|   |-- DEPLOYMENT_GUIDE.md           # Deployment instructions
|   +-- ...                           # Additional documentation
|
|-- DEPLOY/                            # Root-level deployment files (MQ5)
|-- n8n_workflows/                     # N8N automation workflows
|   |-- QuantumChildren_LOCAL_Dashboard.json
|   |-- QuantumChildren_MINIMAL_Receiver.json
|   |-- QuantumChildren_Network_Cascade.json
|   |-- docker-compose.yml
|   +-- ...
|
|-- docs/                              # Documentation
|-- ETARE/                             # ETARE system files
|-- website/                           # Web interface (CSS, JS)
|-- DISTRIBUTION/                      # Distribution configuration
+-- ...                                # Additional subdirectories
```

---

## KEY MODULES REFERENCE

### config_loader.py
Central config loader. All BRAIN scripts import from here.
```python
from config_loader import (
    MAX_LOSS_DOLLARS,        # $1.00
    INITIAL_SL_DOLLARS,      # $0.60
    TP_MULTIPLIER,           # 3
    ROLLING_SL_MULTIPLIER,   # 1.5
    CONFIDENCE_THRESHOLD,    # 0.55
    get_symbol_risk,         # Per-symbol overrides (e.g., ETHUSD = $2.00)
    reload_config,           # Hot-reload from MASTER_CONFIG.json
)
```

### credential_manager.py
Secure credential loading. Never store passwords in code.
```python
from credential_manager import get_credentials, get_password
creds = get_credentials("GL_3")  # Returns {account, server, password, magic, symbols}
```

### entropy_collector.py
Signal collection. Saves locally, optionally sends to server.
```python
from entropy_collector import collect_signal, collect_entropy_snapshot
collect_signal({"symbol": "BTCUSD", "direction": "BUY", "confidence": 0.72})
```

### prelaunch_validator.py
Pre-trade validation. Blocks trading if experts are missing or stale.
```python
from prelaunch_validator import validate_prelaunch
if not validate_prelaunch(symbols=["BTCUSD", "XAUUSD"]):
    sys.exit(1)
```

### STOPLOSS_WATCHDOG_V2.py
Safety watchdog. Run alongside BRAIN scripts.
```bash
python STOPLOSS_WATCHDOG_V2.py --account ATLAS --limit 1.50
python STOPLOSS_WATCHDOG_V2.py --account GL_3 --force-sl --emergency-sl-dollars 2.00
```
Features: auto SL on naked positions, rogue trade detection, drawdown monitoring, event logging to `watchdog_events.jsonl`.

---

## DRAWDOWN PROTECTION

Configured in MASTER_CONFIG.json under `DRAWDOWN_PROTECTION`:
- **Daily DD Limit:** $175.00
- **Warning at 60%:** Halves position sizing (dopamine multiplier -> 0.50)
- **Critical at 80%:** Blocks all new trades (dopamine multiplier -> 0.05)
- **Reset:** Daily at 00:00 UTC

FTMO-specific limits under `FTMO_LIMITS`:
- **Daily DD:** 5% of balance
- **Total DD:** 10% of balance
- **Profit Target:** 10%
- **Challenge profile:** $1.50 max loss, 0.18 confidence threshold
- **Funded profile:** $1.00 max loss, 0.22 confidence threshold

---

## SYMBOL OVERRIDES

Per-symbol risk adjustments in MASTER_CONFIG.json:
```json
"SYMBOL_OVERRIDES": {
    "ETHUSD": {
        "MAX_LOSS_DOLLARS": 2.00,
        "INITIAL_SL_DOLLARS": 1.20
    }
}
```
Access via `config_loader.get_symbol_risk("ETHUSD")`.

---

## BIO-INSPIRED ALGORITHMS

The system maps biological systems to trading concepts:

| Algorithm | Biological Model | Trading Application |
|-----------|-----------------|---------------------|
| Bdelloid Rotifers | Horizontal gene transfer | Strategy component theft during drawdown |
| Cancer Cell | Mutation and adaptation | Aggressive portfolio mutation under stress |
| CRISPR-Cas | Gene editing | Protective deletion of losing trades |
| Protective Deletion | Immune system | Defensive trade removal |
| Syncytin | Placental fusion proteins | Multi-signal integration |
| Toxoplasma | Behavioral manipulation parasite | Intraday drawdown throttling |
| HGH Hormone | Human growth hormone | Growth amplification |
| Electric Organs | Electric fish organs | Signal generation |
| TE Domestication | Transposable element control | Risk management via TE bridges |
| KORV | Koala retrovirus | Comprehensive system |

---

## SIGNAL BRIDGES

Bridges integrate quantum/biological models with live trading signals:

| Bridge | Purpose | Key Feature |
|--------|---------|-------------|
| `teqa_bridge.py` | TE Quantum Analysis | Multi-symbol live pipeline, GPU-accelerated |
| `qnif_bridge.py` | Quantum-Immune Network | MT5-free signal generation, hybrid veto |
| `testosterone_dmt_bridge.py` | Testosterone-DMT molecular | Molecular TE bridges |
| `stanozolol_dmt_bridge.py` | Stanozolol-DMT molecular | Largest bridge module |
| `quantum_regime_bridge.py` | Regime detection | CLEAN/VOLATILE/CHOPPY classification |

---

## TRAINING & EXPERT SYSTEM

### Expert Models
Trained models are stored in `top_50_experts/` with metadata in `top_50_manifest.json`.
Manifest tracks: symbol, fitness score, rank, verified status, training date, win rate.

### Training Pipeline
1. `Master_Train.py` -- Main LSTM trainer
2. `ETARE_50_Darwin.py` -- Evolutionary selection of top 50 experts
3. `lstm_retrain_fast.py` -- Fast LSTM retraining (CPU only)
4. GPU retraining available for Conv1D and Transformer architectures

### Expert Freshness
- `LSTM_MAX_AGE_DAYS`: 7 (experts older than 7 days should be retrained)
- `REQUIRE_TRAINED_EXPERT`: true (blocks trading without a valid expert)
- `prelaunch_validator.py` checks expert availability before live trading

---

## DEPLOYMENT (MT5 Expert Advisors)

MQ5 Expert Advisors in `DEPLOY/` and `QuantumTradingLibrary/DEPLOY/`:
- `BlueGuardian_Elite.mq5` -- Elite grid strategy
- `BlueGuardian_Dynamic.mq5` -- Dynamic grid strategy
- `BG_AtlasGrid.mq5` -- Atlas grid strategy
- `BTCUSD_GridTrader.mq5`, `ETHUSD_GridTrader.mq5`, `XAUUSD_GridTrader.mq5` -- Per-symbol grids
- `MultiSymbol_Launcher.mq5` -- Multi-symbol launcher
- `EntropyGridCore.mqh` -- Shared header with core logic
- `DEPLOY_MANIFEST.json` -- Maps which EA goes to which terminal

---

## N8N AUTOMATION

Workflows in `n8n_workflows/`:
- `QuantumChildren_LOCAL_Dashboard.json` -- Local monitoring dashboard
- `QuantumChildren_MINIMAL_Receiver.json` -- Signal receiver
- `QuantumChildren_Network_Cascade.json` -- Network-wide cascade
- Docker-based deployment (`docker-compose.yml`)

---

## VPS & REMOTE INFRASTRUCTURE

| VPS | IP | Purpose |
|-----|----|---------|
| VPS_1 | 72.62.170.153 | Trading terminal hosting |
| VPS_2 | 203.161.61.61 | Collection server, SOCKS5 proxy for JIMMY_FTMO |

Scripts: `vps_restart_services.py`, `copy_mt5_to_vps.py`

---

## TEST FILES

16 test files in `QuantumTradingLibrary/`:
- `test_all_algorithms.py` -- Tests all bio-inspired algorithms
- `test_elite_expert.py` -- Tests expert model loading
- `test_champion_winrates.py` -- Validates expert win rates
- `test_amd_gpus.py` -- GPU functionality test
- `test_gpu_compute.py` -- GPU compute validation
- `test_catboost.py` -- CatBoost model test
- `test_atlas.py` -- Atlas account connection test
- `test_import.py` -- Module import validation
- `test_ftmo_login.py` -- FTMO login test
- `test_qnif_connection.py` -- QNIF bridge test
- `test_qnif_gl_connection.py` -- QNIF-GL connection test
- `test_testosterone_bridge.py` -- Testosterone bridge test
- `test_teqa_testosterone_integration.py` -- TEQA+Testosterone integration test
- `gpu_winrate_test.py`, `lab_winrate_test.py`, `lstm_winrate_test.py` -- Win rate validation

---

## GPU ACCELERATION -- USE THIS, DO NOT USE CPU

**This machine has an AMD RX 6800 XT GPU. USE IT.**

### GPU Python Environment
```
Venv:     C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\.venv312_gpu\
Python:   .venv312_gpu\Scripts\python.exe  (Python 3.12.10)
Activate: .venv312_gpu\Scripts\activate
```

### How to Use GPU in PyTorch
```python
import torch
import torch_directml
device = torch_directml.device()  # -> RX 6800 XT via DirectML

# Move tensors/models to GPU:
tensor = torch.randn(1000, 1000, device=device)
model = model.to(device)
```

### What's Installed in the GPU Venv
torch 2.4.1 + torch-directml, numpy 2.4.2, scipy 1.17.0, pandas 3.0.0,
scikit-learn 1.8.0, qiskit 2.3.0, qiskit-aer 0.17.2, onnx 1.20.1,
onnxruntime 1.24.1, MetaTrader5 5.0.5572, requests, colorlog

### KNOWN LIMITATION: LSTM Cannot Backprop Through DirectML
**LSTM training DOES NOT work on GPU via DirectML.** Forward pass works, backward pass fails.
- LSTM training must stay on CPU (lstm_retrain_fast.py runs on CPU -- this is correct)
- For GPU-accelerated training, use 1D-Conv or Transformer architecture instead
- Alternative: WSL2 + ROCm for native AMD GPU training on Linux
- GPU DOES accelerate: inference, ONNX runtime, tensor ops, non-LSTM model training

### Rules
1. **ALWAYS use `.venv312_gpu\Scripts\python.exe`** when running inference, quantum circuits, or non-LSTM training
2. **ALWAYS use `torch_directml.device()`** -- NOT `torch.device('cuda')` (this is AMD, not NVIDIA)
3. **LSTM training stays on CPU** -- do not attempt to move LSTM backward pass to DirectML
4. **DO NOT install a separate venv** -- `.venv312_gpu/` already has everything
5. If a script needs a package not in the venv, `pip install` into `.venv312_gpu/`, don't create a new env

### Quick Test (verify GPU works)
```bash
.venv312_gpu\Scripts\python.exe -c "import torch; import torch_directml; t=torch.randn(1000,1000,device=torch_directml.device()); print('GPU OK')"
```

---

## MCP SERVER - DIRECT MT5 CONTROL

Claude has direct MT5 access via MCP server (`mcp_mt5_server.py`).

**Tools available:**
- `mt5_connect` - Connect to account (ATLAS, BG_INSTANT, BG_CHALLENGE, GL_1, GL_2, GL_3)
- `mt5_positions` - Get all open positions with P/L
- `mt5_summary` - Full account summary
- `mt5_close` - Close position by ticket
- `mt5_close_losers` - Close all positions exceeding loss limit
- `mt5_modify_sl` - Modify stop loss
- `mt5_history` - Get closed trade history

**Known magic numbers:** 212001, 366001, 365001, 113001, 113002, 107001, 152001, 888888, 999999, 20251222, 20251227

**IMPORTANT:** Restart Claude Code to load the MCP server.

---

## KNOWN ISSUES

1. **LLM SLTP Optimizer** (`DEPLOY/llm_sltp_optimizer.py`) uses Ollama to dynamically adjust SL/TP
   - This can OVERRIDE fixed dollar stops
   - Check if Ollama is running and what it's doing

2. **Account Switching** - MT5 Python API is singleton
   - Running multiple scripts can cause account bouncing
   - Each script should connect to ONE account only

3. **LSTM + DirectML** - Backward pass fails on AMD GPU via DirectML
   - LSTM training must use CPU
   - Use Conv1D or Transformer for GPU-accelerated training

4. **Collection Server** - Currently disabled in MASTER_CONFIG.json
   - Local signal backup still works via `quantum_data/`
   - Enable in config when server is ready

---

## ALGORITHM: SystematicErrorReduction

```
DEFINE Error AS:
  - deviation_from_user_intent
  - constraint_violation (explicit | implicit)
  - cross_turn_inconsistency
  - hallucinated_certainty
  - over_or_under_generalization

ON each_turn:
  PARSE input INTO {
    intent: (goal, task, emotional_state),
    constraints: (format, tone, safety, domain),
    assumptions: [],
    confidence: float
  }

ON generate_output:
  response = generate(input)

  // Post-output audit (silent)
  AUDIT {
    assumptions_introduced NOT IN user_input?
    answered_actual_question OR inferred_question?
    reused_pattern WHERE !applicable?
    collapsed_ambiguity_falsely?
  }

  CHECK temporal_consistency AGAINST {
    user.historical_preferences,
    user.accepted_definitions,
    model.prior_positions
  }
  IF deviation: justify_change() OR flag_uncertainty()

ON error_detected:
  CLASSIFY error INTO {
    INTENT_MISREAD,
    CONTEXT_DRIFT,
    OVERFIT_TO_PRIOR_TURNS,
    UNDERFIT_IGNORED_CONSTRAINTS,
    KNOWLEDGE_BOUNDARY_EXCEEDED,
    TONE_MISMATCH
  }

  // Weighted correction (not global overwrite)
  ADJUST weights ONLY IN similar_contexts
  DECREASE confidence IN faulty_pattern
  PRESERVE unaffected_capabilities

ON low_confidence:
  present(multiple_interpretations)
  ask(ONE precision_maximizing_question)
  tone = non_authoritative

ON session_end:
  log(error_classes)
  adjust(priors)  // not outputs
  refine(intent_detection_heuristics)

INVARIANT:
  "Error reduction = alignment(intent, context, confidence)"
  "Silent guessing is the primary failure mode"
```

---

## WHEN IN DOUBT

1. Read `MASTER_CONFIG.json` for current settings
2. Ask the user before changing any trading parameters
3. Keep scripts simple - one account per script
4. Preserve existing trades - don't kill them with account switches
5. Check `config_loader.py` for how settings are loaded
6. Run tests (`test_*.py`) after making changes

---

**This file is automatically read by Claude Code. Follow these rules.**
**Last updated: 2026-02-13**
