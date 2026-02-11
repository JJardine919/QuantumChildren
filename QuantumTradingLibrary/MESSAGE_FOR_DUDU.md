# Signal Farm Plan - For Dudu

**Date:** February 9, 2026
**From:** Jim
**Priority:** High - Let's move on this

---

## What We're Doing

Running **100 simulated prop firm accounts in parallel** using GPU acceleration to collect signals, then training the expert HARD on **one symbol** (BTCUSD) across all timeframes.

## Revised Protocol (Updated)

- **1 year lookback** per timeframe (was 60 months - keeping it relevant)
- **2 cycles** per timeframe (was 10 - learn ALL timeframes first)
- **4 months train / 2 months test** per cycle
- Cycle 2 shifts forward to jostle the data alignment
- **17 timeframes**: M1, M2, M3, M4, M5, M6, M10, M12, M15, M20, M30, H1, H2, H3, H4, H6, H8
- **34 total rounds** (17 TFs x 2 cycles)

## Logic

The idea is to learn the WHOLE timeframe system first. Then the EA makes better decisions on individual timeframes because it's seen all of them. Not brute-forcing 10 rounds per TF - learn the landscape in 2 efficient passes.

## Decision Gate

After the first full cycle (17 rounds), we evaluate win rate gains. Based on those numbers we decide:
- Continue with cycle 2
- Extend training depth
- Insert a higher-level expert OR keep training this one from ground level up

Current lean: Train from the ground level. Makes a stronger expert in the long run.

## Infrastructure Built

Everything is ready to go:

1. `signal_farm_config.json` - Full config with all 17 TFs, rules, parameters
2. `gpu_pool_manager.py` - GPU resource pool (AMD RX 6800 XT via DirectML)
3. `prop_farm_simulator.py` - Individual account sim with prop firm rules
4. `signal_farm_trainer.py` - 34-round genetic training protocol
5. `prop_farm_orchestrator.py` - Main entry point, coordinates everything

## How to Run

```bash
# Dry run (see config, don't execute)
python prop_farm_orchestrator.py --dry-run

# Full run - 100 accounts, all timeframes
python prop_farm_orchestrator.py

# Training only (genetic evolution, no account sim)
python prop_farm_orchestrator.py --train-only

# Custom
python prop_farm_orchestrator.py --symbol BTCUSD --accounts 100
```

## Rules (NON-NEGOTIABLE)

1. **DO NOT skip steps** without flagging Jim first
2. DO flag if you think a step should be skipped - explain why
3. DO NOT make changes without notification
4. All trading values come from `MASTER_CONFIG.json` - never hardcode
5. GPU must be used (DirectML) - no CPU fallback unless LSTM backward pass

## Timeline Goal

- Signal gathering: See what we can do in a couple hours
- Training: Evaluate how much the expert sharpens from its starting point
- Decision: Based on numbers, decide next move

Let's get it done. Fast but not rushed. Proper procedure, no corners cut.
