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
SL (MAX_LOSS_DOLLARS):    $1.00      ← NEVER CHANGE
TP_MULTIPLIER:            3          ← NEVER CHANGE
ROLLING_SL_MULTIPLIER:    1.5        ← NEVER CHANGE
DYNAMIC_TP_PERCENT:       40         ← NEVER CHANGE
CONFIDENCE_THRESHOLD:     0.22       ← NEVER CHANGE
AGENT_SL_MIN:             $0.50      ← NEVER CHANGE
AGENT_SL_MAX:             $1.00      ← NEVER CHANGE
```

**Source of truth:** `MASTER_CONFIG.json`
**How scripts get values:** `from config_loader import MAX_LOSS_DOLLARS, TP_MULTIPLIER, ...`

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
| BG_INSTANT | 366604 | BlueGuardian $5K Instant | Blue Guardian MT5 Terminal |
| BG_CHALLENGE | 365060 | BlueGuardian $100K Challenge | Blue Guardian MT5 Terminal 2 |
| ATLAS | 212000584 | Atlas Funded | Atlas Funded MT5 Terminal |
| GL_1 | 113326 | GetLeveraged #1 | GetLeveraged MT5 |
| GL_2 | 113328 | GetLeveraged #2 | GetLeveraged MT5 |
| GL_3 | 107245 | GetLeveraged #3 | GetLeveraged MT5 |

---

## SCRIPTS

### For Trading (run each in separate window):
- `BRAIN_BG_INSTANT.py` → 366604 only
- `BRAIN_BG_CHALLENGE.py` → 365060 only
- `BRAIN_ATLAS.py` → 212000584 only
- `BRAIN_GETLEVERAGED.py` → GetLeveraged accounts

### For Monitoring (read-only):
- `DASHBOARD_MONITOR.py` → Shows all accounts without affecting trades

### For Config:
- `config_loader.py` → Run to see current settings
- `MASTER_CONFIG.json` → Edit this for ALL settings

### For Safety (CRITICAL):
- `STOPLOSS_WATCHDOG.py` → MONITORS and FORCE-CLOSES positions exceeding loss limit
  - Run: `python STOPLOSS_WATCHDOG.py --account ATLAS --limit 1.50`
  - This is a SAFETY NET - runs independently of trading scripts
  - Closes ANY position (Python, MQL5 EA, manual) that exceeds loss limit
  - Check interval: 5 seconds

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
