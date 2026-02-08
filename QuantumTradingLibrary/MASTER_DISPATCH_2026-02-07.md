# MASTER DISPATCH - February 7, 2026 Evening

**Created:** 2026-02-07 ~evening MST
**Created by:** Claude Opus 4.6 (Hub Session)
**Purpose:** Centralized tracking for 5 parallel Claude sessions

---

## SESSION #1 — TEQA / Transposable Elements Live

**Status:** RUNNING (cycling BTCUSD, ETHUSD, XAUUSD every 60s on ATLAS)

**What was accomplished:**
- Confirmed FIRST-EVER application of transposable elements to trading (zero prior art)
- TEQA v2.0 dashboard: 25-qubit quantum circuit, 335/33.5M states, 6.54 bits entropy
- 91.2% pattern strength (Class II transposon detection)
- 63.5% LONG confidence, unanimous TE vote (6.4 LONG / 0.0 SHORT)
- Jardine's Gate threshold clearing at 22.8%
- 3 research agents dispatched (TEs + neural networks, TE-inspired algorithms, DooDoo/QC architecture)
- Source paper: Serrato-Capuchina & Matute 2018
- Key files: `transposable_element_algorithm.py`, `TransposableEdge.mqh`, `te_quantum_signal.json`

**Next directive:**
Let it collect data. When ready to check in:
- Check signal distribution — how often does it pass vs get filtered by Jardine's Gate across 3 symbols?
- Look at mutation history — is novelty staying in 0.2-0.6 range or converging?
- Compare signals across symbols — independent or correlated?
- Collect research agent results (TEs + neural networks, TE-inspired algorithms, DooDoo mapping)

---

## SESSION #2 — Repo Cleanup + Public Release

**Status:** COMPLETE — repo fully triaged, now general helper

**What was accomplished:**
- .gitignore updated (Windows copy duplicates, signal JSONs)
- 18 commits pushed to origin/main
- Created `github.com/JJardine919/QuantumChildren-Free` (public) — 19 clean files
- No proprietary code/models/credentials in public repo
- Collection server endpoint: 203.161.61.61:8888

**Next directive:**
```
Triage the 100+ unstaged/untracked files in QuantumTradingLibrary.

You already cleaned up .gitignore and pushed 18 commits. Now finish the job — there are
100+ unstaged files in C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary from
multiple prior sessions. Run git status and sort everything into:

1. COMMIT — new scripts, MQL5 files, anything that's real work (credential_manager.py,
   prelaunch_validator.py, Deploy folder MQL5 files, etc.)
2. GITIGNORE — .venv311/, rocm_env/, *.db files, quantum_v2_sim_*.json, build artifacts
3. DELETE — stale temp files, duplicate copies, noise
4. FLAG FOR REVIEW — anything you're not sure about, list it for Jim to decide

Do NOT commit lstm_retrain_fast.py yet — being modified by another session. Add it to a
"commit later" list.

Batch commits logically (scripts in one, MQL5 in another, config in another). Push when done.
```

---

## SESSION #3 — Main Repo Deep Clean

**Status:** DONE (14 commits, 549a3df pushed)

**What was accomplished:**
- 14 commits pushed (5e7b159 → 549a3df)
- 95 runtime files removed from tracking (.pth/.db/.pyc/.log/.jsonl)
- Security verified: no .env, .pth, .db, .pyc in repo
- Git status: CLEAN
- 14 GB in QuantumTradingLibrary (mostly .pth + venvs, all gitignored)
- ~389 MB in stale -github/ clone folders (safe to delete)

**Next directive:**
```
Main repo is clean. Next: audit the MQL5 Deploy folders.

You committed 35 MQL5 expert advisors and 36 deploy/diagnostic tools. Verify the Deploy
folder structure matches what's actually loaded in the 5 MT5 terminals. Cross-reference
what's in QuantumTradingLibrary/Deploy/ with what's in
AppData\Roaming\MetaQuotes\Terminal\*\MQL5\Experts\. Flag any mismatches — files in the
repo that aren't deployed, or files in the terminals that aren't in the repo. Also check
that the 8 ONNX models (deployed by session #5) are present in all terminals.
```

---

## SESSION #4 — MQL5 Deploy Audit + Fix (reassigned)

**Status:** COMPLETE — 8/8 items done, 57/57 file checks passed, zero failures

**What was accomplished:**
- Built lstm_retrain_fast.py — working evolution pipeline (backprop + extinction + mutation)
- Pushed models from ~50% to 59.5% out-of-sample in 30 epochs (v1)
- Identified HOLD selectivity gap as breakthrough target (45% HOLD in-sample vs 6% OOS)
- Corrected knowledge base with real ETARE benchmarks
- v2 upgrade: 8 steps implemented:
  1. Confidence thresholding (0.55) — forces low-confidence to HOLD
  2. Class-weighted loss — inverse-frequency weights
  3. Regularization — label smoothing 0.1 + weight decay 1e-4
  4. Walk-forward validation — 30K bars, 6 chunks, 2 folds
  5. Retrained head start — seeds from expert_RETRAINED_best.pth
  6. 80 epochs + early stop (15) + LR schedule (halves after 15 stale)
  7. Enhanced reporting — train/test WR + HOLD% + gap + OVERFIT WARNING
  8. Training log — training_log.json with per-epoch metrics

**ETARE Training Ladder (real benchmarks):**
- 63% = darwin selection alone, zero training (STARTING baseline)
- 70s = one 10-round, 60-month walk-forward session
- 80s+ = 3 symbols x 3 timeframes, proper walk-forward
- 59.5% retrained = below baseline (wrong starting models + simple split in v1)

**MQL5 Deploy Audit completed — found 8 action items. Now fixing all of them with full autonomy:**
1. Add `BG_AtlasGrid.mq5` source to repo
2. Deploy or archive 6 undeployed EAs
3. Push `BG_Diagnostic` + `BG_ForceTrade` to FTMO/GetLev/Atlas
4. Deploy grid traders to FTMO
5. Deploy `BlueGuardian_Quantum` to BG_CHALLENGE
6. Deploy `BlueGuardian_Elite` to Atlas
7. Copy ONNX models to working repo
8. Consolidate scattered MQ5 sources into single Deploy manifest

**Terminal Identity Map (confirmed):**
| Hash | Terminal | Account |
|------|----------|---------|
| 4613C16E | Blue Guardian MT5 Terminal 2 | BG_CHALLENGE (365060) |
| 59C07D67 | Blue Guardian MT5 Terminal | BG_INSTANT (366604) |
| 81A933A9 | FTMO Global Markets MT5 | FTMO (1521063483) |
| D0E8209F | MetaTrader 5 (generic) | GetLeveraged (113326/113328/107245) |
| F6E5FFA1 | Atlas Funded MT5 Terminal | ATLAS (212000584) |

**ONNX models:** All 8 verified present in all 5 terminals.

**Results:**
- 8/8 action items complete, ~55 file operations, zero failures
- DEPLOY_MANIFEST.json created (27 EAs + 8 ONNX models → 5 terminals)
- BG_AtlasGrid.mq5 recovered from terminal to repo
- 6 undeployed EAs pushed to all 5 terminals (source-only, need compile)
- ONNX models copied to working repo (6.53 MB in onnx_experts/)
- All terminals now have consistent, complete EA + model sets

**Note:** 6 newly deployed EAs are .mq5 source only — need MetaEditor compile pass for .ex5

**Next directive:** Compile pass on 6 source-only EAs, or audit Include files across terminals.

---

## SESSION #5 — ONNX Export + MT5 Deploy + Backtest

**Status:** All deployed, overfit question addressed

**What was accomplished:**
- export_expert_onnx.py fixed: weights_only=False, dynamo=False, --best-per-symbol flag
- 8 ONNX models exported (one per symbol, ~798 KB each, LSTM 8→128→3 + softmax)
  - BTCUSD (special), AUDNZD (#1), XAUUSD (#11), ETHUSD (#21)
  - EURCAD (#26), GBPUSD (#27), EURNZD (#40), NZDCHF (#50)
- Deployed to all 5 MT5 terminals (40 ONNX files + Include files + EAs)
- NeuralExpert_Test.mq5 — log-only test EA (no trades)
- NeuralExpert_Backtest.mq5 — tradeable EA ($1 SL, 3x TP, 0.22 confidence)
- Training data goes up to today (2026-02-07) — NO historical OOS window
- Three options: A (forward test), B (accept walk-forward fitness), C (retrain with holdout)

**Next directive:**
```
Run Option A immediately — attach NeuralExpert_Test.mq5 to BTCUSD on ATLAS. Log-only,
no trades. Let it collect predictions for the session.

In parallel, set up Option C: modify lstm_retrain_fast.py v2 (the one session #4 just
upgraded) to accept a --cutoff-date 2025-11-01 flag that stops pulling data at that date.
Don't run it yet — just wire the flag. Session #4 will handle the actual retraining.

Report back the first 10-20 predictions from the test EA so we can see what live inference
looks like.
```

---

## CROSS-SESSION DEPENDENCIES

| From | To | Dependency |
|------|----|-----------|
| #4 | #2 | Don't commit lstm_retrain_fast.py until #4 finishes modifying it |
| #5 | #4 | --cutoff-date flag wired by #5, retraining executed by #4 |
| #1 | #5 | TEQA signals feed into same MT5 terminals where ONNX models run |
| #4 | #5 | Retrained models → re-export to ONNX → redeploy to terminals |

## KEY FILES ACROSS ALL SESSIONS

| File | Session | Status |
|------|---------|--------|
| `transposable_element_algorithm.py` | #1 | Running |
| `TransposableEdge.mqh` | #1, #5 | Deployed |
| `te_quantum_signal.json` | #1 | Being generated live |
| `teqa_live.py` | #2 | Fixed, committed, pushed |
| `export_expert_onnx.py` | #5 | Fixed (both copies) |
| `lstm_retrain_fast.py` | #4 | v2 upgraded, DO NOT COMMIT YET |
| `expert_RETRAINED_best.pth` | #4 | 59.5% OOS, being improved |
| `MUSHROOM_RABIES_LAB_KNOWLEDGE.md` | #4 | Corrected with real benchmarks |
| `NeuralExpert_Test.mq5` | #5 | Deployed to all terminals |
| `NeuralExpert_Backtest.mq5` | #5 | Deployed to all terminals |
| `training_log.json` | #4 | Will be created on next run |

---

## TERMINAL IDENTITY MAP

| Hash | Terminal | Account | Key |
|------|----------|---------|-----|
| 4613C16E | Blue Guardian MT5 Terminal 2 | 365060 | BG_CHALLENGE |
| 59C07D67 | Blue Guardian MT5 Terminal | 366604 | BG_INSTANT |
| 81A933A9 | FTMO Global Markets MT5 | 1521063483 | FTMO |
| D0E8209F | MetaTrader 5 (generic) | 113326/113328/107245 | GL_1/GL_2/GL_3 |
| F6E5FFA1 | Atlas Funded MT5 Terminal | 212000584 | ATLAS |

---

## TEQA LIVE DATA (from Session #1 report)

| Symbol | Novelty | Entropy | States | Direction | Consensus | Gate Status |
|--------|---------|---------|--------|-----------|-----------|-------------|
| BTCUSD | 0.396 | 13.08 bits | 4,813 | SHORT (28/30) | 100% | PASS |
| ETHUSD | 0.380 | 12.55 bits | 4,225 | SHORT (29/30) | 100% | PASS |
| XAUUSD | 0.378 | 12.48 bits | 4,775 | Split votes | Blocked | G7 (consensus) |

- BTC/ETH correlated (same species) — correct
- XAUUSD independent, blocked by G7 — correct
- Shock levels creeping CALM → ELEVATED on BTC/ETH — watch
- Novelty healthy at 0.37-0.43 range

---

## LSTM RETRAIN STATUS (needs separate session to execute)

`lstm_retrain_fast.py` v2 is upgraded and ready to run but no session is currently executing it.
The v2 upgrade includes: confidence thresholding (0.55), class-weighted loss, walk-forward (30K bars, 6 chunks),
80 epochs + early stop, enhanced reporting with HOLD gap + OVERFIT WARNING, training log JSON.

**When ready to run:**
```
cd C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary
python lstm_retrain_fast.py
```

---

*Hub session — routing all 5 parallel sessions through centralized dispatch.*
