# QUANTUM CHILDREN - RULES FOR CLAUDE

**READ THIS FIRST. THESE RULES ARE NON-NEGOTIABLE.**

---

## STOP. BEFORE YOU DO ANYTHING:

**If you are about to modify ANY trading value (SL, TP, lot size, multiplier, etc.):**
1. **DO NOT** write the value directly in a Python script
2. **DO NOT** change MASTER_CONFIG.json without explicit user permission
3. **ALL** scripts must import values from `config_loader.py`
4. **ASK** the user first if you think something needs to change

**This has been broken multiple times by Claude sessions not knowing this.**

---

## CRITICAL: THESE VALUES ARE FIXED

```
SL (MAX_LOSS_DOLLARS):    $1.00      ← Final SL after rolling
SL (INITIAL_SL_DOLLARS):  $0.60      ← Starting SL (60 cents)
TP_MULTIPLIER:            3          ← TP = 3x SL
ROLLING_SL_MULTIPLIER:    1.5        ← SL rolls up by 1.5x
DYNAMIC_TP_PERCENT:       50         ← 50% partial TP
SET_DYNAMIC_TP:           true       ← Dynamic TP enabled
ROLLING_SL_ENABLED:       true       ← Rolling SL enabled
CONFIDENCE_THRESHOLD:     0.55       ← Min confidence to trade
ATR_MULTIPLIER:           0.0438     ← ATR mult for SL distance (lot adjusts for $1 risk)
AGENT_SL_MIN:             $0.50      ← Agent SL floor
AGENT_SL_MAX:             $1.00      ← Agent SL ceiling
REQUIRE_TRAINED_EXPERT:   true       ← Must have trained expert
```

**Source of truth:** `MASTER_CONFIG.json`
**How scripts get values:** `from config_loader import MAX_LOSS_DOLLARS, INITIAL_SL_DOLLARS, ...`

---

## FILES THAT MUST NOT HAVE HARDCODED TRADING VALUES:

These files should ONLY use values imported from config_loader:
- `BRAIN_ATLAS.py`
- `BRAIN_BG_INSTANT.py`
- `BRAIN_BG_CHALLENGE.py`
- `BRAIN_GETLEVERAGED.py`
- `STOPLOSS_WATCHDOG.py`

**If you see something like `MAX_LOSS_DOLLARS = 1.50` in these files, IT IS WRONG.**
**The correct pattern is:** `from config_loader import MAX_LOSS_DOLLARS`

---

## VERIFICATION COMMAND

Run this to check for hardcoded values that should not exist:
```bash
grep -n "MAX_LOSS_DOLLARS\s*=\|TP_MULTIPLIER\s*=" BRAIN_*.py
```
If this returns any lines with `= number`, those are bugs.

---

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

### 3. STOP LOSS IS SACRED
- SL must be calculated for a FIXED DOLLAR AMOUNT
- Default: $1.00 max loss per trade
- Formula: `sl_distance = MAX_LOSS_DOLLARS / (tick_value * lot)`
- NEVER use ATR-based SL that can result in $500+ losses

### 4. DO NOT KILL TRADES
- Switching accounts with `mt5.login()` kills open trades
- Each terminal stays logged into ONE account
- Use separate windows/processes for multiple accounts

---

## ACCOUNTS

| Key | Account | Name | Terminal |
|-----|---------|------|----------|
| GL_3 | 107245 | GetLeveraged (ACTIVE) | GetLeveraged MT5 Terminal |
| JIMMY_FTMO | 1512556097 | Jimmy's FTMO Challenge | FTMO Global Markets MT5 Terminal |
| BG_INSTANT | 366604 | BlueGuardian $5K Instant | Blue Guardian MT5 Terminal |
| BG_CHALLENGE | 365060 | BlueGuardian $100K Challenge | Blue Guardian MT5 Terminal 2 |
| ATLAS | 212000584 | Atlas Funded | Atlas Funded MT5 Terminal |

---

## SCRIPTS

### For Trading (run each in separate window):
- `BRAIN_GETLEVERAGED.py` → GL_3 (107245) ONLY — currently active
- `BRAIN_JIMMY_FTMO.py` → Jimmy's FTMO (1512556097) — via SOCKS5 proxy on VPS_2
- `BRAIN_BG_INSTANT.py` → 366604 only
- `BRAIN_BG_CHALLENGE.py` → 365060 only
- `BRAIN_ATLAS.py` → 212000584 only

### For Monitoring (read-only):
- `DASHBOARD_MONITOR.py` → Shows all accounts without affecting trades

### For Config:
- `config_loader.py` → Run to see current settings
- `MASTER_CONFIG.json` → Edit this for ALL settings

### For Safety (CRITICAL):
- `STOPLOSS_WATCHDOG.py` → Legacy watchdog (force-close only)
- `STOPLOSS_WATCHDOG_V2.py` → ENHANCED watchdog (RECOMMENDED)
  - Run: `python STOPLOSS_WATCHDOG_V2.py --account ATLAS --limit 1.50`
  - Auto-detects and applies SL to positions MISSING stop losses (SL=0.0)
  - Force-closes positions exceeding loss limit
  - Detects ROGUE trades (unknown magic numbers, manual trades)
  - Monitors total drawdown across all positions
  - Comprehensive event logging to `watchdog_events.jsonl`
  - Check interval: 3 seconds (configurable)
  - Options: `--force-sl`, `--emergency-sl-dollars 2.00`, `--drawdown 500`

### MCP Server Tools (for Claude):
- `mt5_scan_no_sl` → Scan all positions for missing SL and rogue magic numbers
- `mt5_force_sl` → Apply emergency SL to a specific position by ticket

---

## DATA COLLECTION

Collection server: `http://203.161.61.61:8888`

All scripts should send signals to this server using `entropy_collector.py`.

---

## PRE-LAUNCH VALIDATION

**IMPORTANT:** BRAIN scripts now enforce pre-launch validation before trading.

**Validator:** `prelaunch_validator.py`

**What it checks:**
1. Trained experts exist for trading symbols (in `top_50_experts/`)
2. Collection server is reachable (warning only)
3. Recent data collection activity (warning only)

**Usage in BRAIN scripts:**
```python
from prelaunch_validator import validate_prelaunch

if not validate_prelaunch(symbols=['BTCUSD', 'XAUUSD']):
    sys.exit(1)
```

**CLI usage:**
```bash
python prelaunch_validator.py --symbols BTCUSD XAUUSD ETHUSD
python prelaunch_validator.py --require-server  # Fail if server down
python prelaunch_validator.py --bypass          # Warn but don't block
```

**Pre-launch workflow:**
1. Start compression server (`quantum_server.py`)
2. Collect data (runs automatically with BRAIN scripts)
3. Train experts: `python Master_Train.py`
4. Validate: `python prelaunch_validator.py`
5. Launch trading: `python BRAIN_ATLAS.py`

---

## WHAT NOT TO DO

1. ❌ Do NOT hardcode SL/TP values in Python scripts
2. ❌ Do NOT use ATR-based stops (causes $500+ losses)
3. ❌ Do NOT switch accounts within a running script
4. ❌ Do NOT change MASTER_CONFIG.json values without asking
5. ❌ Do NOT create scripts that manage multiple accounts in one loop
6. ❌ Do NOT push directly to main -- use a feature branch (see GIT WORKFLOW below)

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

## WHEN IN DOUBT

1. Read `MASTER_CONFIG.json` for current settings
2. Ask the user before changing any trading parameters
3. Keep scripts simple - one account per script
4. Preserve existing trades - don't kill them with account switches

---

---

## GPU ACCELERATION — USE THIS, DO NOT USE CPU

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
device = torch_directml.device()  # → RX 6800 XT via DirectML

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
- LSTM training must stay on CPU (lstm_retrain_fast.py runs on CPU — this is correct)
- For GPU-accelerated training, use 1D-Conv or Transformer architecture instead
- Alternative: WSL2 + ROCm for native AMD GPU training on Linux
- GPU DOES accelerate: inference, ONNX runtime, tensor ops, non-LSTM model training

### Rules
1. **ALWAYS use `.venv312_gpu\Scripts\python.exe`** when running inference, quantum circuits, or non-LSTM training
2. **ALWAYS use `torch_directml.device()`** — NOT `torch.device('cuda')` (this is AMD, not NVIDIA)
3. **LSTM training stays on CPU** — do not attempt to move LSTM backward pass to DirectML
4. **DO NOT install a separate venv** — `.venv312_gpu/` already has everything
5. If a script needs a package not in the venv, `pip install` into `.venv312_gpu/`, don't create a new env

### Quick Test (verify GPU works)
```bash
.venv312_gpu\Scripts\python.exe -c "import torch; import torch_directml; t=torch.randn(1000,1000,device=torch_directml.device()); print('GPU OK')"
```

---

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
    assumptions_introduced ∉ user_input?
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

**This file is automatically read by Claude Code. Follow these rules.**
