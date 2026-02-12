# QUANTUM CHILDREN - RULES FOR CLAUDE

**READ THIS FIRST. THESE RULES ARE NON-NEGOTIABLE.**

## CRITICAL: DO NOT CHANGE THESE VALUES

All trading settings are in `MASTER_CONFIG.json`. **NEVER hardcode values in Python scripts.**

### REQUIRED SETTINGS (DO NOT MODIFY WITHOUT USER PERMISSION):
```
SL (MAX_LOSS_DOLLARS):    $1.00
TP_MULTIPLIER:            3
ROLLING_SL_MULTIPLIER:    1.5
DYNAMIC_TP_PERCENT:       50
CONFIDENCE_THRESHOLD:     0.70
AGENT_SL_MIN:             $0.50
AGENT_SL_MAX:             $1.00
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
| BG_INSTANT | 366604 | BlueGuardian $5K Instant | Blue Guardian MT5 Terminal |
| BG_CHALLENGE | 365060 | BlueGuardian $100K Challenge | Blue Guardian MT5 Terminal 2 |
| ATLAS | 212000584 | Atlas Funded | Atlas Funded MT5 Terminal |
| GL_1 | 113326 | GetLeveraged #1 | GetLeveraged MT5 |
| GL_2 | 113328 | GetLeveraged #2 | GetLeveraged MT5 |
| GL_3 | 107245 | GetLeveraged #3 | GetLeveraged MT5 |
| FTMO | 1521063483 | FTMO Challenge | FTMO-Demo2 (81A933A9AFC5DE3C23B15CAB19C63850) |

---

## SCRIPTS

### For Trading (run each in separate window):
- `BRAIN_BG_INSTANT.py` → 366604 only
- `BRAIN_BG_CHALLENGE.py` → 365060 only
- `BRAIN_ATLAS.py` → 212000584 only
- `BRAIN_GETLEVERAGED.py` → GetLeveraged accounts
- `BRAIN_FTMO.py` → 1521063483 only

### For Monitoring (read-only):
- `DASHBOARD_MONITOR.py` → Shows all accounts without affecting trades

### For Config:
- `config_loader.py` → Run to see current settings
- `MASTER_CONFIG.json` → Edit this for ALL settings

---

## DATA COLLECTION

Collection server: `http://203.161.61.61:8888`

All scripts should send signals to this server using `entropy_collector.py`.

---

## WHAT NOT TO DO

1. ❌ Do NOT hardcode SL/TP values in Python scripts
2. ❌ Do NOT use ATR-based stops (causes $500+ losses)
3. ❌ Do NOT switch accounts within a running script
4. ❌ Do NOT change MASTER_CONFIG.json values without asking
5. ❌ Do NOT create scripts that manage multiple accounts in one loop

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

---

## MCP SERVER - DIRECT MT5 CONTROL

Claude has direct MT5 access via MCP server.

**Tools available:**
- `mt5_connect` - Connect to account (ATLAS, BG_INSTANT, BG_CHALLENGE, GL_1, GL_2, GL_3)
- `mt5_positions` - Get all open positions with P/L
- `mt5_summary` - Full account summary
- `mt5_close` - Close position by ticket
- `mt5_close_losers` - Close all positions exceeding loss limit
- `mt5_modify_sl` - Modify stop loss
- `mt5_history` - Get closed trade history

**IMPORTANT:** Restart Claude Code to load the MCP server.

---

## KNOWN ISSUES

1. **LLM SLTP Optimizer** (`DEPLOY/llm_sltp_optimizer.py`) uses Ollama to dynamically adjust SL/TP
   - This can OVERRIDE fixed dollar stops
   - Check if Ollama is running and what it's doing

2. **Account Switching** - MT5 Python API is singleton
   - Running multiple scripts can cause account bouncing
   - Each script should connect to ONE account only

---

**This file is automatically read by Claude Code. Follow these rules.**
