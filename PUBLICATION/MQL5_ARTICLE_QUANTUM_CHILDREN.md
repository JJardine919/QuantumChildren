# Quantum Children: A Bio-Inspired Adaptive Trading System Using Transposable Element Domestication

## Introduction

What if your trading algorithm could learn which signal combinations actually make money — not through curve-fitting or over-optimization, but through a process inspired by 3.5 billion years of biological evolution?

This article introduces **Quantum Children**, an open-source adaptive trading system that uses concepts from molecular biology — specifically transposable element (TE) domestication and neural somatic mosaicism — to build a self-improving signal processing engine. The system runs on MetaTrader 5 via a Python bridge, uses quantum-inspired circuit computation for signal generation, and implements a closed-loop feedback mechanism that gets smarter with every trade.

This is not a traditional indicator or EA. It is a framework that treats your market signals like DNA — and lets evolution decide which ones survive.

---

## The Problem with Static Signal Systems

Every trader has experienced this: you build an EA, backtest it, optimize the parameters, deploy it — and it works for three months before the market changes and your edge disappears. You re-optimize, redeploy, and the cycle repeats.

The fundamental problem is that static signal systems don't adapt. A moving average crossover doesn't know that crossovers stopped working last Tuesday. An RSI divergence doesn't know that divergences have been failing in the current regime.

What if the system could track which signal combinations preceded winning trades, amplify those combinations, and suppress the losers — automatically, in real-time, without re-optimization?

That's what TE domestication does.

---

## The Biological Metaphor (Keep Reading — This Is the Good Part)

In molecular biology, **transposable elements** (TEs) are pieces of DNA that can copy themselves and move around the genome. They make up about 45% of your DNA. For decades, scientists called them "junk DNA." They were wrong.

It turns out that some TEs get **domesticated** — the host organism finds them useful and keeps them around. Some famous examples:

- **Your immune system** (RAG1/RAG2) was literally built from a DNA transposon that got domesticated by vertebrates
- **Your memory** depends on Arc protein, a domesticated retrovirus that transfers RNA between neurons
- **The placenta** uses syncytin, a domesticated retroviral envelope protein

The pattern: billions of random TE insertions happen → most are neutral or harmful → a few accidentally do something useful → those get kept and amplified → they become essential.

**Sound familiar?** Replace "TE insertions" with "trading signals" and you have the core idea:

- Billions of signal combinations fire every day
- Most are noise
- A few actually precede profitable trades
- Those should be amplified
- The rest should be ignored

That's the Quantum Children algorithm.

---

## Architecture Overview

The system has four main components:

```
┌─────────────────────────────────────────────────────┐
│  TEQA v3.0 Engine (Python)                          │
│  33 signal families → quantum circuit → confidence  │
│  → domestication boost → signal JSON                │
└─────────────┬───────────────────────────────────────┘
              ↓ (JSON file)
┌─────────────────────────────────────────────────────┐
│  BRAIN Script (Python + MT5 API)                    │
│  Read signal → combine with LSTM → execute trade    │
│  → manage positions → poll outcomes                 │
└─────────────┬───────────────────────────────────────┘
              ↓ (MT5 deal history)
┌─────────────────────────────────────────────────────┐
│  Feedback Loop                                      │
│  Match closed trades to signals → update            │
│  domestication database → boost winning patterns    │
└─────────────┬───────────────────────────────────────┘
              ↓ (SQLite)
              → back to TEQA (get_boost on next cycle)
```

### Component 1: The 33 Signal Families

Instead of using 3-4 indicators, the system uses **33 signal generators** organized into families (inspired by TE classification):

**Momentum signals** (like retrotransposons — they amplify trends):
- RSI variants, MACD histogram, Stochastic momentum
- Rate of change, momentum oscillators

**Structural signals** (like DNA transposons — they detect structural changes):
- Support/resistance breaks, pivot points
- Chart pattern detection, order flow imbalance

**Volatility signals** (like neural TEs — brain-specific):
- ATR regimes, Bollinger Band squeeze/expansion
- Volatility clustering, GARCH-inspired measures

**Cross-instrument signals** (speciation):
- Correlation regime shifts between related instruments
- Relative strength, pair divergence

Each signal produces an activation strength (0 to 1) and a direction (long or short). On any given bar, some subset of these 33 signals will be "active" (strength > 0.3). The specific combination of active signals is the **TE activation pattern** — this is the fingerprint that gets tracked.

### Component 2: Quantum-Inspired Circuit Processing

The 33 signal activations are encoded into a **quantum-inspired circuit** with 33 qubits:

- 25 qubits for the main signal families (genome circuit)
- 8 qubits for brain-specific signals (neural circuit)

Each qubit's rotation angle is set by its signal's strength and direction. The qubits are entangled through CX/CZ gates, modeling signal interactions. Measurement produces a probability distribution over long/short/neutral states.

**Why quantum?** Because entanglement captures non-linear interactions between signals that you can't get from simple weighted sums. When Signal A and Signal B are both active, the entangled circuit produces a different output than just adding their individual contributions. This is particularly valuable for detecting regime changes where signal relationships shift.

The system runs on Qiskit's AerSimulator (no quantum hardware needed). On an AMD RX 6800 XT with DirectML, each circuit evaluation takes approximately 50ms.

### Component 3: The Neural Mosaic (7 Virtual Neurons)

Here's where it gets interesting. Instead of one model making one prediction, the system runs **7 virtual neurons**, each with a slightly different genome:

- Neuron 1 might amplify RSI and silence MACD
- Neuron 2 might invert Stochastic and amplify Bollinger
- Neuron 3 might rewire ATR's signal to the momentum qubit

These differences are created by random "L1 insertions" at startup — mimicking how real neurons in your brain each have slightly different DNA due to retrotransposon activity.

The 7 neurons each run the quantum circuit with their modified signals and vote on direction. The majority wins. This creates a **diverse committee** that is more robust than any single model.

**Every 5 cycles, evolution happens:**
- The worst-performing neuron dies
- The best-performing neuron reproduces (with mutations)
- If all neurons converge, forced diversity is injected

The neuron population gets smarter over time through natural selection.

### Component 4: The Domestication Loop (The Secret Sauce)

This is the core innovation. Every time a trade closes, the system:

1. **Looks up which signals were active** when the trade was opened
2. **Records whether the trade won or lost**
3. **Updates a running score** for that specific signal combination
4. **Domesticates** combinations that hit 20+ trades with 70%+ win rate

Domesticated patterns get a **confidence boost** on future signals. The boost follows a sigmoid curve:

```
boost = 1.0 + 0.30 × sigmoid(15 × (win_rate - 0.65))

Win Rate → Boost Factor
  55%    →  1.02 (barely registered)
  60%    →  1.07
  65%    →  1.15
  70%    →  1.23 (domestication threshold)
  80%    →  1.29
  90%    →  1.30 (saturation)
```

The boost is applied at the qubit rotation level — meaning domesticated signal combinations don't just get a flat bonus, they actually change the quantum interference pattern, producing fundamentally different (and historically more profitable) outputs.

**Hysteresis prevents oscillation:** A pattern needs 70% WR to become domesticated, but only loses status below 60%. This prevents flip-flopping at the boundary.

**Bayesian shrinkage prevents noise:** Raw win rates are adjusted with a Beta(10,10) prior, so a pattern that goes 7/10 early on doesn't get prematurely domesticated. It has to earn it over 20+ real trades.

**Expiry prevents staleness:** Patterns inactive for 30 days lose domestication. Markets change. Your edge should too.

---

## The BRAIN Scripts: MT5 Integration

Each trading account runs its own Python script that bridges the TEQA engine to MetaTrader 5:

```python
# Simplified BRAIN loop (actual code is more detailed)
import MetaTrader5 as mt5
from teqa_bridge import TEQABridge
from teqa_feedback import TradeOutcomePoller

# Connect to ONE account (never switch mid-session)
mt5.initialize()
mt5.login(account=212000584, password=get_credentials('ATLAS'))

bridge = TEQABridge()
feedback = TradeOutcomePoller(magic_number=21200)

while market_is_open():
    # 1. Read quantum signal
    signal = bridge.get_signal(symbol='XAUUSD')

    # 2. Combine with LSTM prediction
    action, confidence = combine_with_lstm(signal, lstm_model)

    # 3. Execute if confident
    if confidence > CONFIDENCE_THRESHOLD:
        sl_distance = MAX_LOSS_DOLLARS / (tick_value * lot)  # Fixed $1 risk
        mt5.order_send(
            symbol='XAUUSD',
            type=action,
            volume=lot,
            sl=price - sl_distance,  # or + for short
            tp=price + sl_distance * TP_MULTIPLIER,
            magic=21200,
        )

    # 4. Manage open positions (rolling SL, dynamic TP)
    manage_positions()

    # 5. Feed outcomes back to domestication
    feedback.poll()  # ← This closes the loop

    sleep(60)
```

### Risk Management

Risk is sacred in this system. Every trade is sized for a **fixed dollar stop loss** (default $1.00):

```
sl_distance = MAX_LOSS_DOLLARS / (tick_value * lot_size)
```

This means:
- On XAUUSD at 0.01 lot: SL is about 100 points ($1.00 risk)
- On BTCUSD at 0.01 lot: SL adjusts automatically for the tick value
- You NEVER risk more than $1.00 per trade regardless of instrument

TP is set at 3x the SL distance (default TP_MULTIPLIER = 3).

Rolling stop loss moves the SL to breakeven + profit after the trade moves 1.5x the original SL in your favor.

Dynamic TP takes 50% off the table at the halfway point to TP.

All of these values are loaded from a central `MASTER_CONFIG.json` — nothing is hardcoded in the scripts.

---

## The LSTM Layer

Alongside the TEQA quantum engine, each instrument has a trained LSTM (Long Short-Term Memory) neural network:

- **Architecture:** 8 input features → 128 hidden units (2 layers) → 3 outputs (BUY/SELL/HOLD)
- **Training:** Walk-forward validation on 30,000 bars, 6 chunks
- **Confidence thresholding:** Only acts on predictions above the configured threshold
- **Retraining:** Periodic retraining with new data via `Master_Train.py`

The TEQA signal and LSTM prediction are combined:
- If both agree on direction and both are confident → strong signal
- If they disagree → the TEQA domestication boost can tip the balance
- If neither is confident → no trade (HOLD)

This dual-signal approach means the system needs both a quantum-biological signal AND a neural network prediction to agree before risking capital.

---

## MQL5 EA Integration

The Python system generates signal JSON files that can be consumed by MQL5 Expert Advisors. For pure MQL5 implementations, the key concepts translate:

| Python Concept | MQL5 Equivalent |
|----------------|-----------------|
| Signal JSON file | Shared memory or file in `MQL5/Files/` |
| SQLite domestication DB | `DatabaseCreate()` / `DatabaseExecute()` (build 2650+) |
| Feedback polling | `OnTradeTransaction()` event handler |
| Magic number matching | `HistoryDealGetInteger(ticket, DEAL_MAGIC)` |
| Signal-trade matching | Store signal hash in order comment field |

The `OnTradeTransaction()` approach is actually superior to Python polling — it fires instantly when a position closes, eliminating the match-window ambiguity entirely.

```mql5
// MQL5 pseudocode for domestication feedback
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
    if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
    {
        ulong ticket = trans.deal;
        if(HistoryDealSelect(ticket))
        {
            long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
            if(magic == MY_MAGIC)
            {
                double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                              + HistoryDealGetDouble(ticket, DEAL_COMMISSION)
                              + HistoryDealGetDouble(ticket, DEAL_SWAP);

                string comment = HistoryDealGetString(ticket, DEAL_COMMENT);
                // comment contains "TE:a3f8b2c1" — the pattern hash

                string pattern_hash = StringSubstr(comment, 3);
                bool won = (profit > 0);

                RecordPattern(pattern_hash, won, profit);
            }
        }
    }
}
```

---

## Results and Live Performance

The system is currently deployed across multiple prop firm challenge accounts:

| Account | Firm | Status |
|---------|------|--------|
| Atlas Funded | Atlas | Live trading |
| BlueGuardian $5K | BlueGuardian | Instant funded |
| BlueGuardian $100K | BlueGuardian | Challenge phase |
| GetLeveraged x3 | GetLeveraged | Signal farm (A/B testing) |
| FTMO | FTMO | Challenge phase |

The domestication database is accumulating trade outcomes. Early observations:
- Certain TE combinations (particularly those involving volatility + momentum signals) domesticate faster than pure trend signals
- The neural evolution converges on neurons that amplify contrarian volatility signals in ranging markets
- Cross-instrument signals (speciation engine) provide the strongest domestication candidates when correlated instruments diverge

---

## How to Run It Yourself

### Prerequisites
- MetaTrader 5 terminal (any broker)
- Python 3.12 with: `torch`, `qiskit`, `MetaTrader5`, `numpy`, `pandas`
- Trained LSTM models (run `Master_Train.py` first)

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/[repo]/QuantumChildren.git
cd QuantumChildren/QuantumTradingLibrary

# 2. Configure your account in MASTER_CONFIG.json
# Edit the ACCOUNTS section with your broker details

# 3. Train LSTM models for your instruments
python Master_Train.py

# 4. Start the TEQA signal engine
python teqa_live.py --symbols XAUUSD BTCUSD ETHUSD

# 5. Start the BRAIN trading script (in a separate window)
python BRAIN_ATLAS.py

# 6. (Optional) Start the safety watchdog
python STOPLOSS_WATCHDOG_V2.py --account ATLAS --limit 1.50
```

### Configuration
All settings live in `MASTER_CONFIG.json`:
```json
{
    "MAX_LOSS_DOLLARS": 1.00,
    "TP_MULTIPLIER": 3,
    "CONFIDENCE_THRESHOLD": 0.22,
    "ROLLING_SL_MULTIPLIER": 1.5,
    "DYNAMIC_TP_PERCENT": 50,
    "ROLLING_SL_ENABLED": true,
    "SET_DYNAMIC_TP": true
}
```

**Rule #1:** Never hardcode trading values in scripts. Always use `config_loader.py`.

**Rule #2:** One script per account. Never switch accounts in a running script.

**Rule #3:** Stop loss is sacred. Fixed dollar amount, always.

---

## Open Source

Quantum Children is released under **GPL-3.0**. The complete source code, algorithm specifications, and trained model architecture are available on GitHub.

Website: [quantum-children.com](https://quantum-children.com)

---

## Conclusion

The TE domestication approach is fundamentally different from traditional indicator optimization:

- **Traditional:** Pick signals → optimize parameters → deploy → watch it degrade → re-optimize
- **Quantum Children:** Run 33 signals → let evolution find the winners → domestication amplifies them → the system improves with every trade

The market is a living system. Your trading algorithm should be one too.

---

*Quantum Children is free, open-source software provided for educational purposes. Trading involves substantial risk. Past performance does not guarantee future results. Use at your own risk.*

*Author: J. Jardine*
*License: GPL-3.0*
