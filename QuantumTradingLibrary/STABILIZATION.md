# STABILIZATION.md
## Quantum Children QNIF Trading System - Operator's Playbook

**Last Updated**: 2026-02-11
**Operator**: Jim
**System Version**: QNIF v2.0 (CRISPR-Cas9 + VDJ Immune Memory)

---

## 1. SYSTEM OVERVIEW

### Core Processes

**quantum_server.py** - Signal Compression Backbone
- Runs continuously on dedicated port
- Compresses WebSocket market data for all other processes
- **Critical**: If this dies, entire system goes blind
- Restart priority: FIRST

**BRAIN_ATLAS.py** - Main Trading Brain
- 60-second decision cycle
- LSTM inference (7-day model max age)
- VDJ immune memory (pattern recognition from losses)
- CRISPR-Cas9 adaptation (stores losing patterns, avoids repeats)
- Confidence threshold: 0.22 minimum to trade
- 15-minute cooldown after losses

**STOPLOSS_WATCHDOG_V2.py** - Position Safety Monitor
- 3-second scan cycle
- Force-closes any position at $1.50 loss
- Applies missing stop-loss orders ($0.60 initial)
- Sets take-profit at 3x risk
- **Never disable this process**

**MASTER_LAUNCH.py** - Process Orchestrator
- 30-second health checks on all processes
- Auto-restart on crash (max 5 per process)
- After 5 restarts, flags process as failed and alerts
- Monitors MT5 connection health

**state_persistence.py** - Memory System
- Saves ephemeral state every cycle
- Cooldown timers, VDJ memory cells, CRISPR spacer library
- Auto-restores on system restart
- **Location**: `state/` directory in project root

### Process Interaction Flow
```
Market Data → quantum_server.py (compress)
            → BRAIN_ATLAS.py (analyze, decide)
            → MT5 (execute trade)
            → STOPLOSS_WATCHDOG_V2.py (protect)

MASTER_LAUNCH.py monitors all above, restarts failures
state_persistence.py saves/loads memory across restarts
```

---

## 2. HEALTH DASHBOARD

### Every Morning (5 minutes)
1. Check MASTER_LAUNCH.py console - should show all processes "HEALTHY"
2. Verify last trade timestamp - should be within current session
3. Check state file age: `state/quantum_state.json` - should update every 60 seconds
4. Review overnight P&L - flag if >$50 loss occurred
5. Verify MT5 connection on all accounts (green checkmark in terminal)

### Every 4 Hours During Market
1. Check drawdown percentage (console output from BRAIN_ATLAS.py)
2. Verify STOPLOSS_WATCHDOG_V2.py is cycling (3-second updates)
3. Review CRISPR spacer count - should increase after losses, plateau when winning
4. Check VDJ memory cell diversity - healthy system shows 15-40 unique patterns
5. Scan for restart count - if any process >3 restarts, investigate

### Key Metrics to Watch
- **Trade frequency**: 8-20 trades/day normal, <5 = signal issue, >30 = confidence threshold too low
- **Win rate**: 35-55% healthy, <30% = model aging or market regime shift
- **Cooldown frequency**: Should trigger 3-8 times/day, >12 = choppy conditions
- **Process uptime**: All should stay up >23 hours, frequent crashes = investigate logs

---

## 3. WARNING SIGNS

### Yellow Flags (Monitor Closely)
- **Drawdown 60% ($105)**: Console shows "DD_WARNING" - reduce risk manually if continues
- **LSTM model age >5 days**: Inference quality degrades, retrain soon
- **Same pattern in CRISPR spacers >10 times**: Market changed, immune system outdated
- **VDJ memory cells <10**: System hasn't learned enough, may overtrade
- **Restart count 3-4 on any process**: Unstable, check logs for crash cause
- **Cooldown active >20% of trading hours**: Market unsuitable, consider pause

### Red Flags (Immediate Action Required)
- **Drawdown 80% ($140)**: Console shows "DD_CRITICAL" - BRAIN_ATLAS pauses trading automatically
- **Drawdown hits $175**: FULL STOP - system shuts down, manual restart required
- **Any position loss >$1.50**: STOPLOSS_WATCHDOG failed, manual close + investigate
- **LSTM model age >7 days**: Auto-pauses trading, retrain immediately
- **Restart count 5 on any process**: MASTER_LAUNCH flags as FAILED, manual intervention
- **quantum_server.py down >30 seconds**: Entire system blind, restart immediately

### Critical Thresholds (From MASTER_CONFIG.json)
```
MAX_LOSS_DOLLARS: $1.00           (per trade target)
INITIAL_SL_DOLLARS: $0.60         (auto-applied)
FORCE_CLOSE_THRESHOLD: $1.50      (emergency cutoff)
DAILY_DD_LIMIT_DOLLARS: $175      (full stop)
DD_WARNING: 60% ($105)            (yellow flag)
DD_CRITICAL: 80% ($140)           (auto-pause)
CONFIDENCE_THRESHOLD: 0.22        (minimum to trade)
LOSS_COOLDOWN_MINUTES: 15         (after each loss)
LSTM_MAX_AGE_DAYS: 7              (model expiry)
```

---

## 4. EMERGENCY PROCEDURES

### Runaway Process (Excessive Trading)
**Symptoms**: >5 trades in 10 minutes, ignoring cooldown, confidence <0.22 trades appearing
**Action**:
1. Kill BRAIN_ATLAS.py immediately (Ctrl+C or Task Manager)
2. Let STOPLOSS_WATCHDOG close any open positions
3. Check `state/quantum_state.json` - look for corrupted cooldown timers
4. Verify MASTER_CONFIG.json values unchanged
5. Restart BRAIN_ATLAS.py via MASTER_LAUNCH.py
6. Monitor first 3 trades closely

### Drawdown Limit Hit ($175)
**Action**:
1. System auto-stops trading (BRAIN_ATLAS pauses)
2. STOPLOSS_WATCHDOG continues protecting open positions
3. **Do NOT restart trading same day**
4. Review all trades in MT5 history - identify failure mode
5. Check CRISPR spacers - likely need reset or market regime changed
6. Retrain LSTM model before next day
7. Reset daily drawdown counter at midnight (auto) or manually edit state file

### MT5 Disconnect
**Symptoms**: BRAIN_ATLAS shows "MT5 connection lost", no new trades, watchdog can't apply SL
**Action**:
1. Check VPS internet connection (ping google.com)
2. Restart MT5 terminal - wait for full login
3. Verify all accounts connected (ATLAS, BG_INSTANT, BG_CHALLENGE, FTMO, GL_x)
4. MASTER_LAUNCH auto-reconnects processes within 30 seconds
5. If reconnect fails, manually restart MASTER_LAUNCH.py
6. Check open positions manually - apply SL if missing

### Full System Restart (Nuclear Option)
**When**: Multiple process failures, corrupted state, VPS reboot, major config change
**Action**:
1. Close all positions manually in MT5 (or let STOPLOSS_WATCHDOG do it)
2. Stop MASTER_LAUNCH.py (Ctrl+C)
3. Verify all child processes terminated (Task Manager - check for python.exe)
4. Backup `state/quantum_state.json` to `state/backup_YYYYMMDD_HHMM.json`
5. **Optional**: Delete state file to reset immune memory (only if corrupted)
6. Restart: `python MASTER_LAUNCH.py`
7. Wait 2 minutes - verify all processes HEALTHY
8. Monitor first 5 trades manually

### CRISPR Spacer Library Corruption
**Symptoms**: Trades repeating exact losing patterns, spacer count >500, same loss scenario 5+ times
**Action**:
1. Stop BRAIN_ATLAS.py
2. Edit `state/quantum_state.json` - locate `crispr_spacers` section
3. Delete entire section or reduce to last 50 entries
4. Save file, restart BRAIN_ATLAS.py
5. System rebuilds spacer library from fresh losses

---

## 5. DAILY CHECKLIST (5 Minutes)

### Pre-Market (Before Session Open)
- [ ] Verify VPS uptime >23 hours (if rebooted, investigate)
- [ ] Check all MT5 accounts logged in and connected
- [ ] Confirm MASTER_LAUNCH.py shows all processes HEALTHY
- [ ] Review previous day P&L - flag anomalies
- [ ] Verify LSTM model age <7 days (console output from BRAIN_ATLAS)
- [ ] Check `state/quantum_state.json` last modified <2 minutes ago
- [ ] Scan for Windows Update notifications - defer if scheduled

### Mid-Session (Around Noon)
- [ ] Check current drawdown % - should be <60%
- [ ] Verify trade count reasonable (4-12 trades by midday)
- [ ] Review any cooldown activations - should see 1-4 by now
- [ ] Check STOPLOSS_WATCHDOG cycle time - should be 3-4 seconds
- [ ] Scan process restart counts - all should be 0-1

### Post-Market (After Session Close)
- [ ] Review full day P&L across all accounts
- [ ] Check final drawdown % - reset expected at midnight
- [ ] Count total trades - log if <8 or >30
- [ ] Review CRISPR spacer additions - healthy system adds 2-8/day
- [ ] Verify VDJ memory cell count - should grow slowly (1-3/day)
- [ ] Check for any process crashes in logs
- [ ] **FTMO accounts**: Verify all positions closed (no overnight holds)

---

## 6. WEEKLY CHECKLIST (30 Minutes)

### Every Sunday Evening (Pre-Week Prep)
- [ ] Retrain LSTM model if >5 days old (run training script)
- [ ] Archive previous week's state file: `state/archive/week_YYYY_WW.json`
- [ ] Review CRISPR spacer library - delete patterns >30 days old
- [ ] Analyze week's win rate - should be 35-55%
- [ ] Check VDJ memory diversity - prune duplicates if >50 cells
- [ ] Update MASTER_CONFIG.json if market conditions changed
- [ ] Review all account balances - reconcile with expected based on trades
- [ ] Test emergency shutdown procedure (stop/restart all processes)

### Every Month (Performance Review)
- [ ] Full LSTM retraining from scratch (60+ days data)
- [ ] Reset CRISPR spacer library (fresh start)
- [ ] Reset VDJ memory cells (keep only top 20 performers)
- [ ] Review and adjust confidence threshold if needed (0.20-0.25 range)
- [ ] Analyze monthly drawdown pattern - identify worst days/hours
- [ ] Update this STABILIZATION.md with any new procedures learned

---

## 7. VPS-SPECIFIC MONITORING

### Disk Space (Check Weekly)
**Location**: C:\Users\jimjj\Music\QuantumChildren\
**Concern**: Log files, state backups, LSTM model versions grow over time
**Action**:
- Verify >10GB free on C: drive
- Archive old logs to external storage if <5GB free
- Delete LSTM models >30 days old (keep latest 2)
- Clean `state/archive/` folder - keep last 90 days

### RAM Usage (Check During High Volatility)
**Normal**: 2-4GB total for all processes
**Warning**: >6GB - likely memory leak
**Action**:
- Task Manager - sort by memory, identify bloated process
- Check for zombie python.exe instances (kill manually)
- Restart heavy process via MASTER_LAUNCH
- If persists, full system restart

### Windows Update (Critical)
**Policy**: DEFER all updates during trading hours
**Action**:
- Check for pending updates every Monday
- Schedule for Saturday 2am if available
- **Never allow auto-restart** - VPS must stay up 24/7
- After update, test full system (all processes + MT5 connection)

### Network Latency (Check if Trades Delayed)
**Test**: Ping MT5 broker server from VPS
**Acceptable**: <50ms to broker, <100ms to quantum_server
**Action if >100ms**:
- Check VPS network status (provider dashboard)
- Restart network adapter (VPS control panel)
- Contact VPS provider if persists >1 hour
- **Emergency**: Switch to backup VPS (have config ready)

### Process Priority (Set Once, Verify Monthly)
**Task Manager → Details → Right-click python.exe**:
- quantum_server.py: **High Priority** (signal backbone)
- STOPLOSS_WATCHDOG_V2.py: **Above Normal** (safety critical)
- BRAIN_ATLAS.py: **Normal** (CPU-intensive, let OS manage)
- MASTER_LAUNCH.py: **Normal** (orchestrator)

---

## 8. FTMO ACCOUNT RULES (Critical Compliance)

### Daily Drawdown Calculation
**FTMO Rule**: Daily DD includes unrealized P&L (open positions count)
**System Handling**:
- BRAIN_ATLAS calculates DD with open positions included
- STOPLOSS_WATCHDOG force-closes if any position risks DD breach
- **Manual Check Required**: Before taking final trade of day, verify total exposure <$150

### Weekend Position Management
**FTMO Rule**: No positions held over weekend (Friday close by 4:45pm ET)
**Action**:
- BRAIN_ATLAS auto-pauses new trades after 4:30pm Friday
- STOPLOSS_WATCHDOG closes all positions by 4:50pm Friday
- **Manual Verification**: Check MT5 at 5pm Friday - zero open positions on FTMO
- Monday pre-market: Verify clean slate before system resumes

### Copy Trading Detection
**FTMO Rule**: No copy trading allowed (instant fail)
**System Design**: Each account trades independently via isolated BRAIN_ATLAS instances
**Compliance**:
- Never run multiple accounts from same BRAIN_ATLAS process
- Verify trade timestamps differ by >5 seconds across accounts
- **Manual Audit Weekly**: Compare trade times across FTMO accounts - should NOT be identical

### Consistency Rule (FTMO Challenge Phase)
**Requirement**: No single day >40% of total profit
**Monitoring**:
- Track daily P&L in spreadsheet (not automated)
- If approaching 35% of total profit in one day, stop trading manually
- Let BRAIN_ATLAS cooldowns naturally distribute wins over time

### Max Daily Loss (FTMO Specific)
**FTMO $100k account**: $5,000 daily loss limit (vs our $175 system limit)
**Safety**: Our system limit much tighter, FTMO breach unlikely
**Action if FTMO DD warning**: Immediate manual stop, investigate aggressive trading

---

## QUICK REFERENCE - EMERGENCY CONTACTS

**VPS Provider**: [Your VPS support contact]
**MT5 Broker Support**: [Broker phone/chat]
**Backup Power**: [UPS runtime = X hours]
**This System Designer**: Biskits (20 years experience, #1 quantum architecture optimization)

---

## OPERATOR NOTES SECTION

**Use this space to log unusual events, modifications, lessons learned:**

```
YYYY-MM-DD: [Your observation here]

Example:
2026-02-11: Initial deployment. System stable after 48h. CRISPR learning curve observed - first 20 trades had 2 repeats, then adapted. VDJ memory built to 18 cells by end of week.

```

---

**End of Playbook**
*"Measure twice, cut once. Monitor always."* - Biskits
