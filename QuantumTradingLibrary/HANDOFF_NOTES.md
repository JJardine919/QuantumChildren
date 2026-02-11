# QUANTUM CHILDREN - HANDOFF NOTES
## Date: 2026-02-10
## For: Gemini (or any AI picking up where Claude left off)

---

## WHAT WAS DONE TODAY

### 1. Full System Audit (3 agents ran in parallel)
- **DooDoo** audited all 6 MT5 terminals, 18+ EAs, 7 accounts, full file inventory
- **Specs** audited all trading logic, signal generation, risk management, prop firm compliance
- **Biskits** audited infrastructure, Docker, builds, architecture

### 2. StealthMode Toggle (15 files modified)
Added `StealthMode` input parameter to ALL production EAs. When enabled:
- `request.comment = ""` (empty - no EA fingerprint)
- `request.magic = 0` (matches manual trades)

Files modified:
- DEPLOY/BG_AtlasGrid_Original.mq5
- DEPLOY/BG_AtlasGrid_JardinesGate.mq5
- DEPLOY/BlueGuardian_Dynamic.mq5
- DEPLOY/BlueGuardian_Elite.mq5
- BlueGuardian_Deploy/BG_AtlasStyle.mq5
- BlueGuardian_Deploy/BG_AggressiveCompetition.mq5
- BlueGuardian_Deploy/BG_MultiExecutor.mq5
- BlueGuardian_Deploy/BG_SimpleGrid.mq5
- BlueGuardian_Deploy/BG_Diagnostic.mq5
- BlueGuardian_Deploy/BG_ForceTrade.mq5
- GridMaster300K/GridCore.mqh (class member + SetStealthMode method)
- GetLeveraged_Grid/EntropyGridCore.mqh (class member + SetStealthMode method)
- DEPLOY/EntropyGridCore.mqh (class member + SetStealthMode method)
- XAUUSD_GridMaster/XAUUSD_GridCore.mqh (class member + SetStealthMode method)

NOTE: DEAL_REASON_EXPERT is stamped server-side by MT5 and CANNOT be hidden. Broker can always query it.

### 3. QNIF Nervous System Connection (6 files created/modified)
Created `qnif_bridge.py` - reads QNIF signal files and implements Hybrid Veto:
- QNIF HOLD = trade BLOCKED regardless of TEQA/LSTM opinion
- QNIF agrees with legacy = confidence boost
- QNIF disagrees = QNIF wins (Biological Consensus)
- Legacy HOLD but QNIF strong signal = QNIF can initiate

Wired into all 5 BRAIN scripts:
- BRAIN_ATLAS.py
- BRAIN_BG_INSTANT.py
- BRAIN_BG_CHALLENGE.py
- BRAIN_GETLEVERAGED.py
- BRAIN_FTMO.py

Pipeline: LSTM/Analysis -> TEQA filter -> QNIF Hybrid Veto -> Trade/Block

### 4. ETH Bio-Training (In Progress)
Worker #4 running: `train_crypto_bio.py --csv`
- Building VDJ memory cells for ETH (had 0 during sim)
- BTC already strong (80% pass rate, $13.6K profit in sim)
- ETH underperforming (0% pass, $1.2K) due to cold immune system

### 5. Gemini Bio-Fortification (In Progress - Gemini #1)
Integrating into QNIF_Master.py's process_pulse:
- CRISPR Defense (snip high-drawdown sequences before Brain layer)
- Electric Organs (amplify Pattern Energy when multi-symbol quantum state alignment)
- Syncytin Fusion (horizontal gene transfer between BTCUSD and XAUUSD models)

---

## WHAT STILL NEEDS TO BE DONE

### CRITICAL
1. ~~**BG_Executor.mq5 bugs**~~ **FIXED** (v2.0):
   - ✅ `GetFillingMode()` replaces hard-coded `ORDER_FILLING_IOC`
   - ✅ Array-based VirtualPosition tracking (20 slots) replaces single struct
   - ✅ `RecoverExistingPositions()` on startup rebuilds virtual SL/TP from existing positions
   - ✅ StealthMode added for consistency with all other EAs

2. **Add broker-side emergency stops** to all EAs:
   - Currently ALL positions sent with SL=0, TP=0 (hidden SL architecture)
   - If MT5 crashes, positions are completely unprotected
   - Add wide emergency SL (3-5x normal) as catastrophic backstop

3. **BlueGuardian_Elite confidence gate broken**:
   - `MathMax(adjustedThreshold, 0.50)` clamps threshold, making compression boost ineffective
   - Gate operates at 0.50 instead of intended lower value
   - File: `DEPLOY/BlueGuardian_Elite.mq5` around line 165-168

### HIGH PRIORITY
4. **GetLeveraged terminal is COMPLETELY BARE** - zero custom EAs deployed despite 3 accounts active
5. **Run DEPLOY_TO_MT5.bat** to compile and deploy all EAs to all 6 terminals
6. **Start STOPLOSS_WATCHDOG_V2** if not already running on all accounts
7. **Python BRAIN regime detection stuck at 0.880** - never reaches 0.95 threshold, effectively idle

### MEDIUM PRIORITY
8. **Grid traders uncompiled** on BG and FTMO terminals (only Atlas has .ex5s)
9. **No weekend position close logic** on any EA - prop firm risk
10. **WIZARD TEKS model bug** - RunModel() always uses handle[0], models 82_4_0 and 82_5_0 never called
11. **FinMaster RSI levels inverted** - Oversold=77.1, Overbought=28.4 (backwards)
12. **Start Docker/n8n** - `docker-compose up -d` in n8n_workflows folder
13. **Commit QNIF and new algorithm work to git** before it gets lost
14. **Base44 prop firm deal tracker** - data and schema researched, ready to build

---

## SYSTEM STATE

### MT5 Terminals (6 total)
| Terminal | Broker | Status |
|----------|--------|--------|
| 4613...E5 | Blue Guardian Challenge | Active, deployed |
| 59C0...86 | Blue Guardian Instant | Active, deployed |
| 81A9...50 | FTMO | Active, deployed |
| 99B2...37 | GetLeveraged | BARE - needs full deployment |
| D0E8...75 | Default MT5 | Active, deployed |
| F6E5...55 | Atlas Funded | Most complete deployment |

### Accounts (7 total)
| Key | Account | Broker | Symbols | Enabled |
|-----|---------|--------|---------|---------|
| BG_INSTANT | 366604 | BlueGuardian | BTCUSD | YES |
| BG_CHALLENGE | 365060 | BlueGuardian | BTCUSD | YES |
| ATLAS | 212000584 | Atlas Funded | BTCUSD, ETHUSD | YES |
| GL_1 | 113326 | GetLeveraged | BTCUSD, ETHUSD | YES |
| GL_2 | 113328 | GetLeveraged | BTCUSD, ETHUSD | YES |
| GL_3 | 107245 | GetLeveraged | BTCUSD, ETHUSD | YES |
| FTMO | 1521063483 | FTMO-Demo2 | BTCUSD, XAUUSD, ETHUSD | YES |

### What's Healthy
- EntropyGridCore.mqh - production quality
- Jardine's Gate Algorithm - genuinely creative 6-gate filter
- BlueGuardian_Dynamic - most robust EA
- Drawdown management - conservative (4.5% vs 5% allowed)
- MASTER_CONFIG.json centralized config
- Watchdog running on Atlas (16K+ cycles)

### What's Broken/Dormant
- Docker: installed, no containers running
- n8n: configured, not started
- Signal collection server: disabled in config
- 4 of 6 terminals idle
- Python BRAIN regime detection stuck at 0.880
- 100+ unstaged git changes

---

## PROP FIRM DEALS (Current as of Feb 2026)
- Blue Guardian: Code **FEB** = 40% off + 150% refund. Starting at $10.
- Instant Funding: Code **TRADER2026** = 35% off + free 90% profit split
- IC Funded: Code **PROPFIRMS25** for $37 entry
- RebelsFunding: $25 cheapest entry ($5K account)

---

## KEY FILES
- Config: `MASTER_CONFIG.json`
- Credentials: `.env`
- Rules: `CLAUDE.md`
- Deploy: `DEPLOY_TO_MT5.bat`
- GPU venv: `.venv312_gpu/` (ALWAYS use this one)
- New bridge: `qnif_bridge.py`
- Signal files: `qnif_signal_BTCUSD.json`, `te_quantum_signal.json`

---

## IMPORTANT RULES (from CLAUDE.md)
- ONE script per account - never cycle accounts
- SL is $1.00 max loss (SACRED - don't change without asking)
- ALL settings come from MASTER_CONFIG.json via config_loader.py
- GPU venv is `.venv312_gpu/` - ALWAYS use this, not the others
- LSTM training stays on CPU (DirectML backward pass fails)
- DO NOT switch accounts with mt5.login() - kills open trades
