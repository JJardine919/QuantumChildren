# QUANTUM CHILDREN SIGNAL FARM
## Comprehensive Investigation Report & Base44 Collaboration Brief

**Date:** February 7, 2026
**Prepared by:** Claude (Opus 4.6) with DooDoo MQL5 Specialist
**Purpose:** Architecture proposal for a self-sustaining signal collection and parameter evolution system

---

## THE CORE IDEA

Build a **Signal Farm** -- 50-100 simulation accounts running Quantum Children EAs across every market regime, collecting massive volumes of live signal data while simultaneously refining trading parameters through evolutionary pressure.

Each simulation account runs a challenge harder than any real prop firm challenge. If an EA configuration can beat OUR challenge, it can beat FTMO, BlueGuardian, Atlas, GetLeveraged -- all of them. The ones that survive get their parameters fed forward. The ones that fail provide equally valuable data about what doesn't work and why.

**Dual Output:**
1. **Data Collection** -- Thousands of concurrent signal streams feeding the BRAIN, enabling it to evolve beyond single-direction decision making
2. **Parameter Evolution** -- Continuous refinement across all account sizes, all pair combinations, all market conditions. Spencer's "survival of the fittest" applied to trading systems.

---

## WHAT WE HAVE TODAY (Current Infrastructure)

### Trading Engine: EntropyGridCore (The Farm Engine)

The **CEntropyGridManager** class (`DEPLOY/EntropyGridCore.mqh`, ~1200 lines) is the backbone. It runs entirely inside MT5 with zero Python dependency.

**Why this is the farm engine:**
- Self-contained: No external services needed (no Ollama, no Python, no collection server)
- Multi-symbol capable: `MultiSymbol_Launcher.mq5` runs BTCUSD + XAUUSD + ETHUSD from ONE chart
- Built-in regime detection via entropy confidence scoring
- Hidden SL/TP (anti-stop-hunting architecture)
- Grid mechanics with up to 5 positions per symbol
- Drawdown circuit breaker (4.5% daily, 9.0% max)
- Partial TP, breakeven, trailing stops -- all managed internally

**Current indicator stack:** ATR(14), EMA(8/21/200), RSI(14) on M5

### Signal Generation Pipelines (4 Active)

| Pipeline | Engine | Win Rate | Notes |
|----------|--------|----------|-------|
| ETARE (Full) | Qiskit quantum + CatBoost + Ollama LLM | ~71-75% (low entropy) | Most sophisticated, highest overhead |
| ETARE (Base, no entropy) | CatBoost iteration alone | **88%** | Discovered Feb 6 -- remarkable without entropy layer |
| EntropyGrid Confidence | MQL5 4-factor formula | Varies by regime | Self-contained, no external deps |
| Quantum Bridge | Archiver filter + fidelity + Ollama | ~62% medium entropy | Lightweight variant |

### Live Accounts (7 Active)

| Account | Broker | Size | Symbols |
|---------|--------|------|---------|
| BG_INSTANT (366604) | BlueGuardian | $5K Instant | BTCUSD |
| BG_CHALLENGE (365060) | BlueGuardian | $100K Challenge | BTCUSD |
| ATLAS (212000584) | Atlas Funded | Funded | BTCUSD, ETHUSD |
| GL_1/2/3 | GetLeveraged | Multiple | BTCUSD, ETHUSD |
| FTMO (1521063483) | FTMO | Challenge | BTC, XAU, ETH |

### Data Collection Infrastructure

- **Central server:** `http://203.161.61.61:8888` (entropy_collector.py)
- **Endpoints:** `/signal`, `/outcome`, `/entropy`, `/ping`
- **Local backup:** JSONL files in `quantum_data/` (never loses data)
- **n8n cascade network:** Docker stack with quality-tiered signal distribution

### Communication Layers (6 Active)

1. File-based signal bridge (Python JSON to MQL5)
2. MCP Server (Claude to MT5, JSON-RPC 2.0)
3. HTTP collection network (nodes to central server)
4. Ollama LLM API (Python to local inference)
5. n8n webhook cascade (signal distribution network)
6. VPS/remote deployment (SSH-based)

### Simulation Dashboard

- `quantum_monitor.html` -- Live WebSocket-connected dashboard
- Real-time entropy charts, trade metrics, neural network visualization
- Recently built (Feb 6, 2026) with live market data piped in

---

## WHAT WE NEED TO BUILD

### The Challenge Simulator

A standardized "super-challenge" that is **harder than any real prop firm**:

| Parameter | Our Challenge | Typical Prop Firm |
|-----------|--------------|-------------------|
| Max Daily Drawdown | 3.5% (vs 4-5% industry) | 4-5% |
| Max Total Drawdown | 7% (vs 8-12% industry) | 8-12% |
| Profit Target | 12% (vs 8-10% industry) | 8-10% |
| Time Limit | 21 days (aggressive) | 30-60 days |
| Minimum Trading Days | 10 | 5-10 |
| Max Position Hold | 24 hours (forces active management) | Unlimited |

**Principle:** If you can pass this, you can pass anything.

### The Farm Architecture

```
                    BASE44 DASHBOARD (Presentation Layer)
                    ===================================
                    - Live monitoring of all 100 accounts
                    - Regime coverage map
                    - Parameter evolution tracking
                    - Win/loss analytics per configuration
                    - Challenge pass/fail leaderboard
                    - Signal quality aggregation
                            |
                            | (webhook push / API pull)
                            |
                    CENTRAL COLLECTION SERVER
                    ========================
                    - Receives signals from all farm nodes
                    - Aggregates outcomes and performance
                    - Stores time-series signal data
                    - Feeds analytics to Base44
                    - Already exists: 203.161.61.61:8888
                            |
                            | (HTTP POST signals)
                            |
    =====================================================
    |           |           |           |           |
  MT5 #1     MT5 #2     MT5 #3    ...          MT5 #N
  Sim Acct   Sim Acct   Sim Acct              Sim Acct
  ---------------------  ---------------------
  | MultiSymbol_Launcher (3 symbols each)    |
  | EntropyGridCore engine                   |
  | Challenge rules enforced                 |
  | Signal reporter (to collection server)   |
  | Parameter set A  | Parameter set B  | ...|
  =====================================================
```

### What Each Simulation Account Runs

1. **One MT5 terminal** (can be on VPS instances for scale)
2. **MultiSymbol_Launcher EA** managing BTCUSD + XAUUSD + ETHUSD
3. **Challenge enforcer** -- custom logic that tracks P&L against our super-challenge rules
4. **Signal reporter** -- sends every signal, entry, exit, and outcome to the collection server
5. **Unique parameter set** -- each account gets a different configuration variant to test

### Parameter Dimensions to Evolve

Each simulation account runs a unique combination:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| CONFIDENCE_THRESHOLD | 0.15 - 0.40 | Signal sensitivity |
| ATR_MULTIPLIER | 0.03 - 0.08 | Volatility scaling |
| TP_MULTIPLIER | 2x - 5x | Profit target aggressiveness |
| INITIAL_SL_DOLLARS | $0.40 - $1.00 | Risk per trade |
| GRID_MAX_POSITIONS | 3 - 7 | Grid depth |
| GRID_SPACING_ATR | 0.3x - 0.8x | Grid level spacing |
| EMA periods | Various | Trend detection sensitivity |
| RSI thresholds | Various | Overbought/oversold sensitivity |
| PARTIAL_TP_PERCENT | 30% - 70% | How much to close early |
| TRAILING_ATR_MULT | 0.5x - 2.0x | Trailing stop tightness |

**100 accounts x 10 parameter dimensions = massive evolutionary search space**

### Data Collection Schema

Every signal farm node reports:

```json
{
    "account_id": "SIM_042",
    "parameter_set_id": "PS_042",
    "timestamp": "2026-02-07T14:30:00Z",
    "symbol": "BTCUSD",
    "regime": "LOW_ENTROPY",
    "signal": {
        "direction": "BUY",
        "confidence": 0.73,
        "entry_price": 98500.00,
        "sl_price": 98440.00,
        "tp_price": 98680.00
    },
    "outcome": {
        "exit_price": 98650.00,
        "pnl_dollars": 2.85,
        "duration_seconds": 1800,
        "exit_reason": "PARTIAL_TP"
    },
    "challenge_status": {
        "day": 8,
        "total_pnl_percent": 6.2,
        "daily_dd_percent": 0.8,
        "max_dd_percent": 2.1,
        "passed": false,
        "failed": false,
        "in_progress": true
    },
    "market_context": {
        "atr_14": 450.5,
        "ema_alignment": "BULLISH",
        "rsi_14": 58.3,
        "spread": 15.0,
        "hour_utc": 14
    }
}
```

---

## BASE44'S ROLE: THE COLLABORATION ASK

### What We Need From Base44

We're not asking Base44 to build the trading engine or the signal collection backend -- those exist. We need Base44 to be **the brain's eyes and the operator's control panel**. Base44 is uniquely positioned to build the layer that makes this data useful and actionable.

### Entity Design (Data Models)

We need Base44 to design and implement these entities:

**1. SimulationAccount**
- Tracks each of the 50-100 simulation accounts
- Fields: account_id, parameter_set (JSON), status (running/passed/failed/paused), start_date, current_day, pnl_percent, daily_dd, max_dd, symbols_active

**2. Signal**
- Every signal generated across the farm
- Fields: account_id, timestamp, symbol, direction, confidence, regime, entry_price, sl, tp, parameter_set_id
- This will be HIGH VOLUME -- potentially thousands per day

**3. TradeOutcome**
- Completed trade results
- Fields: signal_id (reference), exit_price, pnl_dollars, pnl_percent, duration, exit_reason, market_context (JSON)

**4. ChallengeRun**
- Each challenge attempt from start to completion
- Fields: account_id, parameter_set_id, start_date, end_date, result (passed/failed/in_progress), final_pnl, max_dd_hit, daily_dd_hit, days_traded, failure_reason

**5. ParameterSet**
- Each unique configuration being tested
- Fields: set_id, parameters (JSON), generation (for evolutionary tracking), parent_set_id (which config it evolved from), total_challenges, challenges_passed, pass_rate, avg_pnl

**6. RegimeCoverage**
- Tracks which regimes are being actively covered
- Fields: regime_type, symbol, timeframe, active_accounts_count, signals_today, avg_confidence

### Backend Functions (Webhook Endpoints)

**1. receiveSignal** -- Webhook endpoint that the collection server pushes to
- Receives signal data, writes to Signal entity
- Updates RegimeCoverage counters
- Triggers any real-time subscriptions for the dashboard

**2. receiveOutcome** -- Receives trade completion data
- Links to original signal
- Updates ChallengeRun progress
- Checks if challenge passed or failed
- If failed: marks ChallengeRun, pauses account
- If passed: flags for parameter promotion

**3. evolveParameters** -- Scheduled automation (daily or on challenge completion)
- Looks at which ParameterSets have the highest pass rates
- Generates new parameter combinations based on top performers
- This is the "survival of the fittest" engine
- Cross-breeds successful parameters, mutates slightly, creates next generation
- Outputs new parameter configs for the next batch of simulation accounts

**4. getDashboardData** -- Aggregation endpoint for the frontend
- Summary stats: total accounts, active challenges, pass rate, regime coverage
- Top performing parameter sets
- Current market regime distribution
- Signal volume and quality metrics

**5. getEvolutionHistory** -- Tracks parameter evolution over time
- Shows how parameters have evolved generation over generation
- Visualizes which traits are being selected for

### Dashboard Pages

**1. Farm Overview**
- Grid/table of all simulation accounts with status indicators
- Color-coded: green (passing), yellow (in progress), red (failed)
- Key metrics: total signals today, challenge pass rate, regime coverage %

**2. Challenge Leaderboard**
- Ranked list of parameter sets by pass rate and average PnL
- Shows generation number (how many evolution cycles)
- Click into any set to see its full history

**3. Regime Coverage Map**
- Visual representation of which market regimes are actively being tested
- Shows gaps in coverage
- Signal density per regime

**4. Evolution Timeline**
- Generational view of how parameters are evolving
- Which parameters are converging (the system is finding optimal values)
- Which parameters still have high variance (still exploring)

**5. Signal Explorer**
- Browse/filter individual signals across the farm
- Filter by symbol, regime, confidence, outcome
- Statistical analysis of signal quality by parameter set

**6. Live Monitor**
- Real-time feed of signals and trades as they happen
- Uses Base44's WebSocket subscriptions for push updates
- Similar to the existing quantum_monitor.html but scaled to 100 accounts

### Scheduled Automations

**1. Daily Evolution Run** (runs every 24 hours)
- Evaluates completed challenges
- Promotes winning parameters
- Generates next generation of parameter sets
- Restarts failed accounts with new configurations

**2. Health Check** (runs every 5 minutes)
- Pings collection server to verify it's receiving data
- Checks for accounts that have gone silent
- Alerts if regime coverage drops below threshold

**3. Data Aggregation** (runs every hour)
- Rolls up signal data into hourly/daily summaries
- Updates ParameterSet statistics
- Calculates regime coverage metrics

### What Base44 Does NOT Need To Build

- The MT5 Expert Advisors (we have these)
- The signal collection server (exists at 203.161.61.61:8888)
- The trading logic or regime detection (handled by EntropyGridCore)
- The MT5 terminal management (handled by our batch scripts/VPS)

---

## TECHNICAL INTEGRATION POINTS

### How Data Flows to Base44

```
MT5 EA generates signal
    -> entropy_collector.py sends to collection server (203.161.61.61:8888)
    -> Collection server enriches and forwards to Base44 webhook
    -> Base44 receiveSignal function writes to entities
    -> WebSocket subscription pushes to dashboard
```

### Base44 Webhook URLs (to be created)

```
POST https://app--quantum-farm.base44.app/api/apps/{app-id}/functions/receiveSignal
POST https://app--quantum-farm.base44.app/api/apps/{app-id}/functions/receiveOutcome
POST https://app--quantum-farm.base44.app/api/apps/{app-id}/functions/challengeUpdate
```

### Authentication

- Webhook endpoints should be unauthenticated (collection server is internal)
- Dashboard uses Base44's built-in auth for operator access
- Service role for backend aggregation functions

---

## SCALING PLAN

### Phase 1: Proof of Concept (Week 1)
- 5 simulation accounts
- 3 symbols (BTC, XAU, ETH)
- 5 different parameter sets
- Basic dashboard with signal feed
- Verify data pipeline end-to-end

### Phase 2: Scale Up (Week 2-3)
- 25 simulation accounts
- Expand parameter search space
- Challenge enforcer fully operational
- First evolution cycle
- Full dashboard operational

### Phase 3: Full Farm (Week 4+)
- 50-100 simulation accounts
- Multiple VPS instances for MT5 terminals
- Automated evolution cycles
- Complete regime coverage
- Parameter convergence analysis

---

## THE BIGGER PICTURE

### Why This Changes Everything

**Before:** We need millions of users to generate enough data for the BRAIN to make autonomous decisions across all market conditions.

**After:** We generate that data ourselves. 100 accounts running 24/5 across 3+ symbols, each producing dozens of signals per day = thousands of data points daily. Within weeks, we have what would take months to collect from organic users.

**The Evolution Angle (Spencer's Principle):**
Each generation of parameter sets is subjected to environmental pressure (the super-challenge). Only the fittest survive. Their "genes" (parameter values) get recombined and slightly mutated for the next generation. Over time, the system converges on configurations that are genuinely robust -- not overfit to one market condition, but proven across all regimes.

**The BRAIN Evolution:**
As this data floods in, the BRAIN stops being a single-direction signal follower. It starts seeing patterns across hundreds of concurrent experiments:
- "When BTCUSD enters regime X, parameter sets with higher TP multipliers survive"
- "In low entropy conditions, tighter grids outperform"
- "This parameter combination passes challenges 3x more often than average"

The BRAIN becomes a meta-learner over trading strategies, not just a signal generator.

### Business Model Impact

- Free tier remains viable -- we don't NEED user data to improve
- Premium offering: "Battle-tested parameters that passed our super-challenge"
- Credibility: "Our configurations have passed X challenges across Y market conditions"
- The simulation farm becomes a continuous R&D engine

---

## APPENDIX: BASE44 PLATFORM CAPABILITIES REFERENCE

For Base44's team -- here's what we know about your platform's strengths and how we've designed around constraints:

| Capability | How We Use It |
|-----------|---------------|
| MongoDB-compatible entities | Signal, TradeOutcome, ParameterSet storage |
| Webhook endpoints (backend functions) | Receiving data from collection server |
| Scheduled automations (5-min minimum) | Health checks, hourly aggregation, daily evolution |
| WebSocket subscriptions (Realtime SDK) | Live dashboard updates |
| React frontend + Tailwind/shadcn | Dashboard UI |
| AI agent system | Future: natural language querying of farm data |
| Row-level security | Multi-operator access control |
| Deno backend runtime | All server-side logic |

| Constraint | Our Mitigation |
|-----------|---------------|
| 3-min function timeout | Pre-aggregated data from collection server; Base44 just stores and serves |
| 5-min automation minimum | Health checks don't need sub-minute precision |
| No persistent processes | Collection server handles persistence; Base44 receives pre-processed data |
| 5,000 items per request | Pagination for signal history; dashboards use aggregated views |
| Integration credits | Webhooks don't consume credits; automations are limited to 3 scheduled tasks |

**The heavy lifting (MT5 terminals, signal collection, real-time data pipeline) stays on our infrastructure. Base44 is the intelligence layer -- the place where data becomes insight and parameters evolve into winners.**

---

*This document is the collaboration brief for the Base44 team. We're relying on their expertise to architect and implement the dashboard, data model, evolution engine, and monitoring system described above. The trading infrastructure exists and is proven. We need Base44 to build the layer that makes it all visible, analyzable, and self-improving.*
