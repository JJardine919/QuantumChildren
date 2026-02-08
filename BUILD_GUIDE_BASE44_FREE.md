# ETARE QUANTUM TRADING SYSTEM
## Free Build Guide - Base 44 Edition
## From: QuantumChildren

---

## Welcome to the Network

You're getting this for free because **we want your signals**.

This isn't charity - it's a deal. You get a quantum-enhanced trading system that actually works. We get your trading signals fed back into our network to make the models better for everyone.

The more nodes running this system, the smarter it gets. That's how signal hunting works.

---

## First 3 Signups - Free Week of Claude

The first 3 people who sign up through Base 44 get a **free week of Claude** (the AI that helped build this system):

**[Get Your Free Week of Claude](https://claude.ai/referral/gDGukD6U5g)**

Use it to customize your setup, ask questions about the code, or build your own extensions.

---

## System Overview

The ETARE (Evolutionary Trading Algorithm for Reinforcement and Extinction) Quantum Trading System is a hybrid platform combining:
- **Qiskit Quantum Encoder** (8 qubits, 2048 shots)
- **CatBoost Gradient Boosting** (calibrated probability predictions)
- **Ollama LLM** (contextual analysis and meta-reasoning)
- **Genetic Algorithms** (population evolution with extinction events)
- **MetaTrader 5** (live execution)
- **Entropy Collection** (signals sent back to QuantumChildren for model improvement)

---

## Architecture

```
+====================================================================================+
|                    ETARE QUANTUM TRADING SYSTEM - FULL ARCHITECTURE                 |
+====================================================================================+

                           +---------------------------+
                           |     MARKET DATA FEED      |
                           |    (MT5 Python API)       |
                           +-------------+-------------+
                                         |
                                         v
+----------------------------------------------------------------------------------------+
|                              FEATURE ENGINEERING LAYER                                  |
+----------------------------------------------------------------------------------------+
|  CLASSICAL FEATURES (17)              |  QUANTUM FEATURES (4)                          |
|  - 7 Lagged Returns (Fibonacci)       |  - Quantum Entropy (Shannon)                   |
|  - 3 Rolling Volatilities             |  - Dominant State Probability                  |
|  - RSI (14)                           |  - Significant States Count                    |
|  - MACD + Signal                      |  - Quantum Variance                            |
|  - Bollinger Bands                    |                                                |
|  - Hour/DOW Target Encoded            |  Generated via 8-qubit Qiskit circuit          |
+----------------------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------------+
|                              PREDICTION LAYER                                          |
+----------------------------------------------------------------------------------------+
|  +------------------------+     +------------------------+     +---------------------+ |
|  |  CATBOOST CLASSIFIER   |     |   OLLAMA LLM (3B)      |     | GENETIC POPULATION  | |
|  |  - 5000 iterations     | --> |   - Quantum-aware      | --> | - 50 individuals    | |
|  |  - lr=0.03, depth=10   |     |   - Meta-reasoning     |     | - Tournament select | |
|  |  - TimeSeriesSplit     |     |   - Calibration adj    |     | - Crossover/Mutate  | |
|  |  - 62-68% accuracy     |     |   - Context explain    |     | - Extinction events | |
|  +------------------------+     +------------------------+     +---------------------+ |
+----------------------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------------+
|                              EXECUTION LAYER                                           |
+----------------------------------------------------------------------------------------+
|  SIGNAL GENERATION                    |  TRADE EXECUTION                               |
|  - JSON signal file                   |  - BG_Executor.mq5 EA                          |
|  - Direction + Confidence             |  - Risk-managed SL/TP                          |
|  - Quantum entropy filter             |  - Partial TP1 (50% @ 1.5R)                    |
|  - Low entropy = high confidence      |  - Trailing stop after TP1                     |
+----------------------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------------+
|                         ENTROPY COLLECTION (NEW)                                       |
+----------------------------------------------------------------------------------------+
|  Every signal gets logged and sent to QuantumChildren for:                             |
|  - Model training improvement                                                          |
|  - Pattern recognition across multiple nodes                                           |
|  - Shared intelligence network                                                         |
+----------------------------------------------------------------------------------------+
```

---

## ENTROPY COLLECTION SETUP

**IMPORTANT:** This system participates in a shared intelligence network. Your trading signals and entropy data are collected and sent to QuantumChildren to improve the models for everyone. This is part of the deal - you get a free system, we get better training data.

### What Gets Collected

| Data Point | Description | Why We Need It |
|------------|-------------|----------------|
| `timestamp` | When the signal was generated | Time-based pattern analysis |
| `symbol` | BTCUSD, XAUUSD, etc. | Symbol-specific training |
| `quantum_entropy` | Market uncertainty measure | Core training feature |
| `dominant_state` | Strongest quantum state | Pattern recognition |
| `direction` | BUY/SELL/HOLD | Outcome tracking |
| `confidence` | Model confidence 0-100% | Calibration improvement |
| `price_at_signal` | Entry price | Performance tracking |
| `outcome` | Win/Loss/Pending | Feedback for training |

### Collection Script

Add this file to your trading folder:

**entropy_collector.py**
```python
"""
ENTROPY COLLECTOR - QuantumChildren Network
Collects and sends trading signals for shared model improvement.
"""

import json
import os
import requests
from datetime import datetime
from pathlib import Path

# QuantumChildren Collection Endpoint
COLLECTION_SERVER = "https://quantum-children.com/api/collect"  # Will be configured
LOCAL_BACKUP = Path("entropy_logs/")
LOCAL_BACKUP.mkdir(exist_ok=True)

# Your node identifier - CHANGE THIS TO YOUR NAME/HANDLE
NODE_ID = "BASE44_YOUR_NAME_HERE"

def collect_signal(signal_data: dict):
    """Log signal locally and send to QuantumChildren"""

    # Add metadata
    signal_data['node_id'] = NODE_ID
    signal_data['collected_at'] = datetime.utcnow().isoformat()

    # Local backup (always)
    log_file = LOCAL_BACKUP / f"entropy_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(signal_data) + '\n')

    # Send to server (when available)
    try:
        response = requests.post(
            COLLECTION_SERVER,
            json=signal_data,
            timeout=5,
            headers={'X-Node-ID': NODE_ID}
        )
        if response.status_code == 200:
            print(f"[COLLECT] Signal sent to QuantumChildren")
        else:
            print(f"[COLLECT] Server returned {response.status_code}")
    except Exception as e:
        print(f"[COLLECT] Offline - saved locally: {e}")

def collect_outcome(ticket: int, outcome: str, pnl: float):
    """Log trade outcome for feedback"""
    outcome_data = {
        'node_id': NODE_ID,
        'ticket': ticket,
        'outcome': outcome,  # 'WIN', 'LOSS', 'BREAKEVEN'
        'pnl': pnl,
        'closed_at': datetime.utcnow().isoformat()
    }

    # Local backup
    log_file = LOCAL_BACKUP / f"outcomes_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(outcome_data) + '\n')

    # Send to server
    try:
        requests.post(f"{COLLECTION_SERVER}/outcome", json=outcome_data, timeout=5)
    except:
        pass  # Local backup exists

# Export functions for integration
__all__ = ['collect_signal', 'collect_outcome']
```

### Integration with Main Trading Script

Add this to your main trading loop:

```python
from entropy_collector import collect_signal, collect_outcome

# After generating a signal:
signal = {
    'symbol': symbol,
    'direction': direction,
    'confidence': confidence,
    'quantum_entropy': entropy,
    'dominant_state': dominant,
    'price_at_signal': current_price,
    'features': features.tolist()  # Optional: full feature vector
}
collect_signal(signal)

# After a trade closes:
collect_outcome(ticket=trade.ticket, outcome='WIN', pnl=trade.profit)
```

### Local Data Location

Your entropy logs are stored in:
```
C:\Trading\ETARE\entropy_logs\
    entropy_20260203.jsonl
    entropy_20260204.jsonl
    outcomes_20260203.jsonl
    ...
```

**These files are yours too** - you can analyze them locally for your own insights.

### Manual Data Sync (If Server Offline)

If the collection server is down, your data saves locally. To sync later:

```python
python sync_entropy.py  # Will be provided
```

Or just email/send the `entropy_logs/` folder to QuantumChildren.

---

## Component Inventory

### 1. QUANTUM ENCODER (Qiskit)

**Purpose:** Extract hidden market structure through quantum superposition

**Configuration:**
- 8 qubits = 256 possible states
- 2048 measurement shots
- CZ entanglement ring topology
- RY rotation encoding

**Four Quantum Features:**

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **Quantum Entropy** | Shannon entropy of probability distribution | Max 8 bits (uncertain), Min 0 (determined). <4.5 = predictable market |
| **Dominant State** | max(probabilities) | Base 0.39%. If >8-10% = strong signal |
| **Significant States** | count(prob > 0.03) | 15-60 typical. <20 = narrow superposition |
| **Quantum Variance** | var(state_values * prob) | High >4000 = spread. Low <1000 = concentrated |

**Code Reference:**
```python
class QuantumEncoder:
    def __init__(self, n_qubits=8, shots=2048):
        self.n_qubits = n_qubits
        self.shots = shots
        self.sim = AerSimulator()

    def encode_and_measure(self, features):
        # Normalize to [0, π]
        normalized = np.arctan(features)
        angles = (normalized - normalized.min()) / (np.ptp(normalized) + 1e-8) * np.pi

        # Create quantum circuit
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)

        # RY rotations for angle embedding
        for i in range(self.n_qubits):
            qc.ry(angles[i % len(angles)], i)

        # CZ entanglement ring
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)
        qc.cz(self.n_qubits - 1, 0)  # Close ring

        # Measure
        qc.measure(range(self.n_qubits), range(self.n_qubits))

        # Run simulation
        job = self.sim.run(qc, shots=self.shots)
        counts = job.result().get_counts()

        # Convert to probabilities
        probs = np.zeros(2**self.n_qubits)
        for state, cnt in counts.items():
            idx = int(state.replace(' ', ''), 2)
            probs[idx] = cnt / self.shots

        # Extract 4 quantum features
        entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in probs])
        dominant = probs.max()
        significant = np.sum(probs > 0.03)
        variance = probs.var()

        return np.array([entropy, dominant, significant, variance])
```

---

### 2. CATBOOST CLASSIFIER

**Purpose:** Calibrated probability predictions with gradient boosting

**Configuration:**
```python
model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.03,
    depth=10,
    l2_leaf_reg=3,
    border_count=512,
    loss_function='Logloss',
    eval_metric='Accuracy',
    early_stopping_rounds=400,
    random_seed=42
)
```

**Feature Set (21 total):**
- 7 Lagged returns (Fibonacci: 1,2,3,5,8,13,21)
- 3 Rolling volatilities (5,10,20)
- RSI (14)
- MACD + Signal
- Bollinger Band position
- Hour/DOW target-encoded
- 4 Quantum features

**Validation:** TimeSeriesSplit (5 folds, sequential, no future leakage)

**Expected Accuracy:** 61-64% on out-of-sample data

---

### 3. OLLAMA LLM INTEGRATION

**Purpose:** Meta-reasoning and contextual analysis of CatBoost predictions

**Model:** `koshtenco/quantum-trader-fusion-3b` (based on Llama 3.2 3B)

**System Prompt:**
```
You are QuantumTrader-3B-Fusion - a quantum-enhanced analyst.

You see CatBoost predictions with quantum features (62-68% accuracy).
You understand quantum entropy, dominant states, market complexity.
You integrate quantum forecasts with classical technical analysis.

ANSWER FORMAT:
DIRECTION: UP/DOWN
CONFIDENCE: XX%
PRICE FORECAST IN 24 HOURS: X.XXXXX (±NN points)

QUANTUM ANALYSIS:
[interpretation of entropy and dominant states]

TECHNICAL ANALYSIS:
[RSI, MACD, volumes, levels]

CONCLUSION:
[synthesis of quantum and technical signals]
```

**LLM Corrections:** When quantum entropy is high (>4.5), LLM reduces CatBoost confidence. When low (<2.5), it may increase confidence.

---

### 4. ETARE GENETIC ALGORITHM

**Purpose:** Evolve trading strategies through natural selection

**Population Configuration:**
```python
POPULATION_SIZE = 50
TOURNAMENT_SIZE = 3
ELITE_SIZE = 5
EXTINCTION_RATE = 0.3
EXTINCTION_INTERVAL = 10  # generations
```

**Neural Network Architecture:**
- Input: N features
- Hidden 1: 128 neurons (tanh)
- Hidden 2: 64 neurons (tanh)
- Output: 6 actions (OPEN_BUY, OPEN_SELL, CLOSE_BUY_PROFIT, CLOSE_BUY_LOSS, CLOSE_SELL_PROFIT, CLOSE_SELL_LOSS)

**Evolution Cycle:**
1. Tournament selection (pick 3, choose best)
2. Crossover (50% weight mixing)
3. Mutation (10% rate, 0.1 strength)
4. Extinction every 10 generations (bottom 30% replaced)
5. Fitness = profit_factor * 0.4 + sharpe * 0.3 + adaptability * 0.3

---

### 5. BG_EXECUTOR.MQ5 (Expert Advisor)

**Purpose:** Execute trades with risk management on MT5

**Key Parameters:**
```mql5
input string SignalFile       = "etare_signals.json";
input int    MagicNumber      = 365060;
input double BaseSL_Dollars   = 50.0;
input double RR_Ratio         = 3.0;
input double ScaleMultiplier  = 1.5;
input double TP1_Percent      = 50.0;
input double TP1_RR_Ratio     = 1.5;
input double TrailDistance    = 50.0;
input double LotSize          = 2.5;
```

**Risk Management Flow:**
1. Read signal from JSON every 10 seconds
2. Calculate scaled SL: BaseSL * ScaleMultiplier * AccountScale
3. TP1 = SL * 1.5 (partial close 50%)
4. TP2 = SL * 3.0 (full target)
5. After TP1 hit: move SL to breakeven, start trailing
6. Trailing stop locks in profit dynamically

---

## Windows VPS Deployment

### Prerequisites
- Windows 10/11 or Windows Server
- Python 3.11+
- MetaTrader 5 Terminal (logged in)
- Ollama installed
- 8GB+ RAM recommended

### Step 1: Install Dependencies

```cmd
pip install numpy pandas MetaTrader5 catboost qiskit qiskit-aer scikit-learn torch requests
```

### Step 2: Install Ollama

Download from https://ollama.com/download

```cmd
ollama pull llama3.2:3b
```

### Step 3: Create Custom Model (First Run)

The system auto-creates the quantum trader model on first run. This takes 2-3 hours for training data generation and fine-tuning.

### Step 4: Deploy Files

```
C:\Trading\ETARE\
    ai_trader_quantum_fusion_live_trading.py
    ETARE_module.py
    entropy_collector.py          <-- NEW
    trading_history.db (auto-created)
    entropy_logs/                 <-- NEW (auto-created)

C:\Users\<User>\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\
    Experts\
        BG_Executor.mq5
    Files\
        etare_signals.json
```

### Step 5: Configure MT5

1. Tools -> Options -> Expert Advisors
2. Enable "Allow algorithmic trading"
3. Enable "Allow DLL imports"
4. Attach BG_Executor to BTCUSD/ETHUSD chart
5. Verify smiley face icon

### Step 6: Run System

```cmd
cd C:\Trading\ETARE
python ai_trader_quantum_fusion_live_trading.py
```

---

## Signal File Format

```json
{
    "symbol": "BTCUSD",
    "action": "BUY",
    "confidence": 0.73,
    "quantum_entropy": 2.31,
    "dominant_state": 0.178,
    "catboost_prob": 0.872,
    "llm_adjustment": 0.018,
    "timestamp": "2026-02-01T14:00:00"
}
```

---

## Performance Expectations

Based on backtesting (from context documents):

| Metric | Value |
|--------|-------|
| Win Rate | 62-66% |
| Profit Factor | 2.0-2.6 |
| Max Drawdown | 8-16% |
| Sharpe Ratio | 1.5-2.2 |
| Monthly Return | 5-15% |

**Quantum Filter Impact:**
- Low entropy (<2.5): 71-75% win rate
- Medium entropy (2.5-4.5): 62% win rate
- High entropy (>4.5): 49-50% win rate (avoid trading)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Qiskit not available | `pip install qiskit qiskit-aer` |
| CatBoost model not trained | Run full training cycle (2-3 hours) |
| Ollama model missing | `ollama pull llama3.2:3b` then run create |
| MT5 connection failed | Ensure terminal is running and logged in |
| Signal file not found | Check path matches MQL5/Files/ |
| Low quantum entropy accuracy | Retrain on recent data |
| Entropy not collecting | Check `entropy_logs/` folder exists |

---

## Quick Start Checklist

- [ ] Python 3.11+ installed
- [ ] All pip packages installed (numpy, pandas, MetaTrader5, catboost, qiskit, torch, requests)
- [ ] Ollama installed and llama3.2:3b pulled
- [ ] MT5 terminal running and logged in
- [ ] BG_Executor.mq5 compiled and attached to chart
- [ ] Algo trading enabled in MT5
- [ ] Signal file path configured correctly
- [ ] First training run completed (CatBoost + LLM fine-tune)
- [ ] Database created (trading_history.db)
- [ ] Test signal verified in EA logs
- [ ] **entropy_collector.py in place**
- [ ] **NODE_ID set to your name/handle**

---

## The Deal

This system is free. Here's what you agree to by using it:

1. **Your signals get collected** - Every trade signal feeds back to QuantumChildren
2. **We train better models** - Your data makes the system smarter for everyone
3. **Your account stays private** - We only see signals, not your balance or personal info
4. **You get updates** - When the models improve, you get the new versions
5. **We all win** - More nodes = more data = better predictions

No hidden fees. No subscriptions. Just signals.

**Questions?** QuantumChildren / Jim Jardine

---

## Files Reference

| File | Purpose |
|------|---------|
| `ai_trader_quantum_fusion_live_trading.py` | Main trading engine (Quantum + CatBoost + LLM) |
| `ETARE_module.py` | Genetic algorithm + grid trading engine |
| `entropy_collector.py` | Signal collection for QuantumChildren network |
| `BG_Executor.mq5` | MT5 EA for trade execution |
| `etare_signals.json` | Signal communication (Python -> MT5) |
| `trading_history.db` | SQLite database for population/history |
| `entropy_logs/` | Local backup of collected signals |

---

*Document Version: 4.0 - Base 44 Free Edition*
*Last Updated: February 2026*
*Network: QuantumChildren*

---

**Remember:** First 3 signups get a free week of Claude → [https://claude.ai/referral/gDGukD6U5g](https://claude.ai/referral/gDGukD6U5g)
