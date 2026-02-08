# MUSHROOM RABIES LAB - KNOWLEDGE BASE
**Last Updated: 2026-02-07**
**Source: Live testing session with real MT5 data (30,000 bars M5)**

---

## 1. EXPERT ARCHITECTURE MAP

There are **4 distinct architectures** across the codebase. They are NOT interchangeable.

### Architecture A: Darwin Feedforward (BEST PERFORMERS)
- **Location:** `etare_darwin.db` (population table, JSON blobs)
- **Top Win Rate:** 73.6% (in-sample), 55.3% (out-of-sample live)
- **Weight Format:** JSON dict with keys:
  - `input_weights`: (128, 8) — features to hidden
  - `hidden_weights`: (128, 128) — hidden to hidden
  - `output_weights`: (3, 128) — hidden to output
  - `hidden_bias`: (128,)
  - `output_bias`: (3,)
- **Forward Pass:**
  ```
  x_norm = (x - mean) / std          # per-sample normalization
  hidden  = tanh(x @ input_weights.T + hidden_bias)
  hidden2 = tanh(hidden @ hidden_weights.T)
  output  = hidden2 @ output_weights.T + output_bias
  actions = argmax(output)            # 0=HOLD, 1=BUY, 2=SELL
  ```
- **Created By:** `ETARE_50_Darwin.py`, `etare_adaptive_darwin.py`, `etare_fast_baseline.py`
- **Fitness Formula (adaptive/fast):** `fitness = win_rate` (direct)
- **Fitness Formula (ETARE_50_Darwin):** `fitness = accuracy*0.3 + win_rate*0.4 + profit_factor*0.2 + drawdown_resistance*0.1`
- **Population Size:** 50 individuals in DB
- **Top 15 by fitness:**

| ID | Fitness | Profit | Notes |
|----|---------|--------|-------|
| 1  | 0.7364  | $967   | Best overall |
| 2  | 0.7364  | $967   | Clone of #1 |
| 3  | 0.7234  | $681   | |
| 4  | 0.7208  | $932   | |
| 5  | 0.7184  | $886   | |
| 6  | 0.7184  | $886   | Clone of #5 |
| 7  | 0.7144  | $688   | |
| 8  | 0.7144  | $688   | Clone of #7 |
| 9  | 0.7089  | $487   | |
| 10 | 0.7066  | $687   | |
| 11 | 0.7066  | $687   | Clone of #10 |
| 12 | 0.7007  | $305   | |
| 13 | 0.7004  | $353   | |
| 14 | 0.6976  | $545   | |
| 15 | 0.6963  | $654   | |

### Architecture B: LSTM 2-Layer (top_50_experts/)
- **Location:** `top_50_experts/*.pth` (PyTorch state_dict)
- **Top Win Rate:** ~48-49% (out-of-sample live)
- **Model Class:** `LSTMModel` in `mushroom_rabies_lab.py`
  - `nn.LSTM(input_size=8, hidden_size=128, num_layers=2, batch_first=True)`
  - `nn.Dropout(0.4)`
  - `nn.Linear(128, 3)`
- **Weight Keys in state_dict:**
  - `lstm.weight_ih_l0`: (512, 8) — 4 gates x 128 hidden x 8 input
  - `lstm.weight_hh_l0`: (512, 128)
  - `lstm.bias_ih_l0`: (512,)
  - `lstm.bias_hh_l0`: (512,)
  - `lstm.weight_ih_l1`: (512, 128)
  - `lstm.weight_hh_l1`: (512, 128)
  - `lstm.bias_ih_l1`: (512,)
  - `lstm.bias_hh_l1`: (512,)
  - `fc.weight`: (3, 128)
  - `fc.bias`: (3,)
- **LSTM Gate Layout:** [input_gate | forget_gate | cell_gate | output_gate], each 128 rows
- **Created By:** `etare_redux_v2.db` export, manifest in `top_50_manifest.json`
- **Fitness Range:** 0.06 to 0.27 (composite score, NOT win rate)
- **Source DB:** `etare_redux_v2.db` (no longer present, only v3 exists)
- **51 files** in top_50_experts/ (50 ranked + 1 BTCUSD special)

### Architecture C: StrikeBoss 3-Layer LSTM
- **Location:** `StrikeBoss/strikeboss_lstm_best.pth`
- **Win Rate:** 31% (12-month MT5 backtest on BTCUSD M10)
- **Model:** 3 separate LSTM layers with decreasing size
  - `lstm1`: (512, 8) ih / (512, 128) hh — hidden=128
  - `lstm2`: (256, 128) ih / (256, 64) hh — hidden=64
  - `lstm3`: (128, 64) ih / (128, 32) hh — hidden=32
  - `fc1`: (16, 32)
  - `fc2`: (2, 16) — **output_size=2** (BUY/SELL only, no HOLD)
- **Trained via:** Colab notebook (`StrikeBOSS_Training_Colab.ipynb`)
- **NOT compatible with mushroom_rabies_lab.py** (different layer naming, different output size)

### Architecture D: ETARE Redux v3 LSTM (etare_redux_v3.db)
- **Location:** `etare_redux_v3.db` (population_state table, PyTorch blobs)
- **Symbol:** XAUUSD only (1 batch of training completed)
- **Fitness Range:** 0.0 to 0.16
- **Same LSTM architecture as Architecture B**
- **Schema:** id, symbol, individual_index, weights (blob), fitness, total_profit

---

## 2. MUSHROOM RABIES LAB - TESTED RESULTS

### Lab Pipeline
```
Expert -> Rabies -> Mushrooms -> TEQA -> Compress -> Mutant
```

### Stage Details

**RABIES:** Amplify weights to create hyperexcitability
- LSTM: Amplify input gate (aggression x), cell gate (aggression x 0.8), sharpen output gate, suppress forget gate bias (0.3x)
- Feedforward: Amplify input_weights (aggression x), hidden_weights (aggression x 0.8), sharpen output_weights, suppress hidden_bias (0.3x)

**MUSHROOMS:** Stochastic perturbation
- Gaussian noise scaled by weight std x dose
- Activate dormant connections (abs < 0.01 threshold)
- Cross-layer bleed (0.05x)

**TEQA:** Transposable element insertion from `te_quantum_signal.json`
- Reads Jardines Gate (confidence, direction) and quantum metrics (novelty, vote ratio)
- Rank-1 perturbation on hidden/recurrent weights
- Directional nudge on output weights

**COMPRESS:** Quality check on mutant hidden state
- QuTiP quantum autoencoder (if available) or SVD fallback
- Classifies regime: CLEAN (>0.95), VOLATILE (>0.85), CHOPPY (<0.85)

### LSTM Expert Results (top_50_experts/, 30K bars M5 live MT5 data)

| Expert | Rabies | Dose | Before WR | After WR | Delta | Trades Before | Trades After |
|--------|--------|------|-----------|----------|-------|---------------|--------------|
| expert_BTCUSD_special | 2.0 | 0.30 | 47.7% | 49.2% | **+1.5%** | 2,581 | 4,301 |
| expert_rank11_XAUUSD | 1.5 | 0.15 | 49.3% | 54.5% | +5.3% | 3,184 | **11** (BROKEN) |
| expert_rank21_ETHUSD | 1.5 | 0.15 | 48.6% | 48.8% | +0.2% | 3,626 | 3,685 |

**XAUUSD result is INVALID** — collapsed to 11 trades (model went nearly all-HOLD).

### Darwin Feedforward Results (etare_darwin.db #1, 30K bars M5 live MT5 data)

**CORRECTED 2026-02-07:** Initial results used WRONG features (bb_middle/bb_std/ema_10/price_change instead of training features). Results below use correct training features: `rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr` with `adjust=False` EMA and true ATR.

~~INVALID (wrong features): 55.3% baseline, +1.7% standard, -4.5% heavy~~

| Expert | Rabies | Dose | Before WR | After WR | Delta | Trades Before | Trades After |
|--------|--------|------|-----------|----------|-------|---------------|--------------|
| darwin #1 (73.6% fitness) | 1.5 | 0.15 | 47.0% | **48.9%** | **+1.9%** | 3,990 | 3,786 |
| darwin #1 (73.6% fitness) | 2.0 | 0.30 | 47.0% | 45.1% | **-1.9%** | 3,990 | 4,889 |

### LSTM Retraining Results (2026-02-07)

Retrained top 10 LSTM experts from `top_50_experts/` using ETARE_50_Darwin.py evolution pipeline.
**30-bar sequences** (SEQ_LENGTH=30) so LSTM memory is engaged. 5000 bars BTCUSD M5, 80/20 split.

| Metric | In-Sample (train) | Out-of-Sample (test) |
|--------|-------------------|----------------------|
| Win Rate | **90.1%** | **59.5%** |
| Accuracy | 73.8% | 47.5% |
| HOLD % | 45% (selective) | 6% (indiscriminate) |
| Trades | 2,176 | 928 |

Best model saved: `expert_RETRAINED_best.pth`

### Key Findings

1. **63% out-of-sample is the STARTING baseline** — darwin selection alone (no backprop training) produces 63% with proper walk-forward validation. This is where training BEGINS, not ends.
2. **70%+ out-of-sample is achievable** with one proper 10-round, 60-month walk-forward training session on a single symbol.
3. **Multi-symbol x multi-timeframe walk-forward adds 15-20%** on top — 3 symbols (BTCUSD, XAUUSD, ETHUSD) x 3 timeframes (M1, M5, M15) with proper walk-forward where data is NEVER seen twice.
4. **The 59.5% retrained result is BELOW the untrained baseline** because it started from broken 50% experts (wrong models from DB) and used a simple 80/20 split instead of walk-forward. Do NOT use this as a benchmark.
5. **The overfit smoking gun is HOLD selectivity.** In-sample: 45% HOLD (selective sniping). Out-of-sample: 6% HOLD (fires on everything). Proper walk-forward prevents this by ensuring the model never evaluates on data it trained on.
4. **Walk-forward is mandatory.** Train on one period, test on data never seen, never repeat, change timeframe AND position. Simple 80/20 split is NOT sufficient — it's too easy to leak information.
5. **Standard dose (1.5/0.15) is the sweet spot for feedforward experts.** Heavy trip overdoses them.
6. **Heavy trip (2.0/0.30) works better on LSTMs** which have 4 gates to absorb noise. Feedforward nets have only 3 matrices — less room.
7. **XAUUSD LSTM was borderline (51% BUY)** — mutation paralyzed it. Avoid mutating experts near 50/50 decision boundaries.
8. **Trade count changes matter.** If trades drop dramatically (3184 -> 11), the win rate is meaningless. If trades spike (3990 -> 4889 on heavy trip), the model became indiscriminate.
9. **All compression results came back CHOPPY** — mutations introduce enough chaos that SVD can't find clean structure.
10. **Feature matching is CRITICAL.** Initial darwin test used wrong features (5 of 8 mismatched). Always verify features match training exactly.
11. **etare_darwin.db stores GeneticWeights (feedforward) but fitness was earned by LSTM model.** The LSTM weights were never saved to DB. Do NOT run feedforward inference on those weights and expect LSTM fitness numbers.
12. **LSTM experts need 30-bar sequences.** Previous tests passed single bars — LSTM memory was never engaged, effectively running as feedforward. Always use `create_sequences()` with SEQ_LENGTH=30.

---

## 3. EXPERT DATABASES

| Database | Tables | Contents | Status |
|----------|--------|----------|--------|
| `etare_darwin.db` | population, history | 50 feedforward experts, 73.6% top fitness | **ACTIVE - best experts** |
| `etare_redux_v3.db` | training_log, population_state | XAUUSD LSTM training (1 batch) | Active, early training |
| `etare_btcusd.db` | (unknown) | BTCUSD-specific training | Needs investigation |
| `teqa_domestication.db` | domesticated_patterns | TE pattern tracking | Empty (0 rows) |
| `etare_redux_v2.db` | (missing) | Source for top_50_experts | **FILE DELETED** |

---

## 4. SIMULATION RESULTS ON FILE

### Prop Firm Simulations

| File | Expert | Base WR | With Compression | Pass Rate | Avg PnL |
|------|--------|---------|-----------------|-----------|---------|
| `prop_sim_blueguardian_results.json` | expert_C7_E36_WR72 | 72% | 89.2% | 100% (500/500) | $571 |
| `prop_sim_blueguardian_results.json` | same, no compression | 72% | 76.3% | 99.6% | $543 |
| `prop_sim_365060_results.json` | BG 100K Challenge | — | 89.5% | 100% (1000/1000) | $11,525 |
| `prop_sim_blueguardian_stress_results.json` | Stress test | 68% degraded | 60.3% | — | — |

**NOTE:** These simulations used synthetic trade generation based on win rate probabilities, NOT actual expert model inference. The 89% numbers are Monte Carlo projections, not real backtests.

### Quantum V2 Simulations (actual signal-based)

| File | Symbol | Win Rate | Trades | Notes |
|------|--------|----------|--------|-------|
| `quantum_v2_sim_BTCUSD_20260204_142412.json` | BTCUSD | **78.9%** | 19 | Best result, only 19 trades |
| `quantum_v2_sim_BTCUSD_20260206_015557.json` | BTCUSD | 58.8% | 17 | |
| `quantum_v2_sim_BTCUSD_20260206_015625.json` | BTCUSD | 43.8% | 201 | More trades, lower WR |
| Most others | BTCUSD | 0-40% | varies | Many had 0 trades |

---

## 5. TEQA SIGNAL STATE

Current signal file: `te_quantum_signal.json`
- **Direction:** SHORT (direction = -1)
- **Confidence:** 52.1%
- **Novelty:** 0.433
- **Vote Ratio:** vote_long / (vote_long + vote_short)
- **Active Qubits:** normalized to /25.0
- **Measurement Entropy:** normalized to /25.0

This signal is injected into every mutation. If the signal changes, mutation behavior changes.

---

## 6. LAB COMPATIBILITY MATRIX

| Expert Source | Architecture | Lab Script | Compatible? | Notes |
|---------------|-------------|------------|-------------|-------|
| etare_darwin.db | Feedforward | `lab_darwin_winrate_test.py` | YES | Use standard dose (1.5/0.15) |
| top_50_experts/*.pth | 2-layer LSTM | `mushroom_rabies_lab.py` | YES | Heavy trip OK but experts are weak |
| StrikeBoss | 3-layer LSTM | Neither | NO | Different layer naming + output_size=2 |
| etare_redux_v3.db | 2-layer LSTM | `mushroom_rabies_lab.py` | YES (needs extraction) | Only XAUUSD, early training |

---

## 7. FEATURE ENGINEERING

**WARNING: Different training scripts use DIFFERENT features. You MUST match the features to the expert's training source.**

### Feature Set A: ETARE_50_Darwin.py (Darwin DB experts)
Used by: `etare_darwin.db` population, `ETARE_50_Darwin.py`

| # | Feature | Calculation | Notes |
|---|---------|-------------|-------|
| 1 | rsi | 14-period RSI | |
| 2 | macd | EMA(12) - EMA(26) | **adjust=False** |
| 3 | macd_signal | EMA(9) of MACD | **adjust=False** |
| 4 | bb_upper | SMA(20) + 2*STD(20) | Upper Bollinger |
| 5 | bb_lower | SMA(20) - 2*STD(20) | Lower Bollinger |
| 6 | momentum | close / close[10] | |
| 7 | roc | pct_change(10) * 100 | Rate of change |
| 8 | atr | rolling(14).mean() of True Range | **True ATR** (includes gaps) |

### Feature Set B: gpu_winrate_test.py / test_champion_winrates.py (LSTM experts)
Used by: `top_50_experts/*.pth`, LSTM win rate tests

| # | Feature | Calculation | Notes |
|---|---------|-------------|-------|
| 1 | rsi | 14-period RSI | |
| 2 | macd | EMA(12) - EMA(26) | adjust=True (default) |
| 3 | bb_middle | SMA(20) | |
| 4 | bb_std | STD(20) | |
| 5 | ema_10 | EMA(10) | |
| 6 | momentum | close / close[10] | |
| 7 | atr | rolling(14) high-low range | **NOT true ATR** |
| 8 | price_change | close.pct_change() | |

### Feature Set C: etare_adaptive_darwin.py (17 features)
Used by: `etare_adaptive_darwin.py`, `etare_fast_baseline.py`
- 17 features including volume metrics, multiple EMA periods
- **NOT compatible with 8-feature models**

Normalization for all: `(feature - mean) / (std + 1e-8)` per column.
Darwin models also apply per-sample normalization in forward pass (double-normalized).

---

## 8. DATA SOURCES

| Source | Access | Symbols | Bars Available |
|--------|--------|---------|----------------|
| MT5 (live terminals) | `mt5.copy_rates_from_pos()` | BTCUSD, XAUUSD, ETHUSD, AUDNZD, GBPUSD, EURCAD, etc. | 30,000+ M5 bars |
| Binance API | REST (no auth needed) | BTCUSDT, ETHUSDT | 3 months M5 (~25,920 bars) |

MT5 is preferred (real broker data with spread/gaps). Binance is fallback.

---

## 9. RECOMMENDED PARAMETERS

### For Darwin Feedforward Experts (55%+ live WR):
```
rabies = 1.5      # Standard
dose = 0.15       # Standard
teqa_strength = 0.2
DO NOT use heavy trip (2.0/0.3) — destroys signal
```

### For LSTM Experts (if retrained above 55%):
```
rabies = 1.5-2.0  # Can handle more aggression (4 gates absorb noise)
dose = 0.15-0.30  # Proportional to rabies
teqa_strength = 0.2
```

---

## 10. RESOLVED ISSUES

1. **Feature mismatch (RESOLVED 2026-02-07):** `lab_darwin_winrate_test.py` was using wrong features. Fixed to match `ETARE_50_Darwin.py` training features exactly (rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr with adjust=False and true ATR).
2. **Weight shape confirmation (RESOLVED 2026-02-07):** Verified `etare_darwin.db` weights are (128,8)/(128,128)/(3,128) format from `ETARE_50_Darwin.py`, NOT the 17-feature/6-action format from `etare_adaptive_darwin.py`.
3. **Action enum (RESOLVED 2026-02-07):** Darwin DB uses 3-action (0=HOLD, 1=BUY, 2=SELL), confirmed by output_size=3.

## 11. OPEN QUESTIONS / TODO

1. **Walk-forward retraining needed.** The 59.5% retrained experts are BELOW the untrained 63% baseline because they started from wrong models and used simple 80/20 split. Proper walk-forward training progression: 63% (baseline) -> 70s (one 10-round 60-month session) -> 80s+ (multi-symbol x multi-timeframe). Use ETARE_Redux.py logic: 10 batches x 3 cycles, never see data twice, change BOTH position and timeframe between runs.
2. **Multiple mutation runs:** Mushroom trip is stochastic. Run N mutations, keep the best. (Evolution of mutations — natural next step)
3. **StrikeBoss integration:** Needs adapter for 3-layer LSTM + 2-output format.
4. **QuTiP compression:** Not installed. SVD fallback only.
5. **TEQA signal staleness:** Current signal is SHORT/52.1%.
6. **ETARE_50_Darwin.py saves wrong thing.** `_save_to_db` saves GeneticWeights but fitness comes from LSTM. The LSTM model (which earns the fitness) is never persisted. Future training should save `model.state_dict()` not `weights.to_dict()`.
7. **HOLD selectivity is the key to high WR.** In-sample models HOLD 45% of the time (selective sniping). Out-of-sample they only HOLD 6% (indiscriminate). Solving the overfit = making HOLD selectivity generalize.

---

## 11. FILE REFERENCE

| File | Purpose |
|------|---------|
| `mushroom_rabies_lab.py` | Original lab (LSTM only) |
| `lab_darwin_winrate_test.py` | Lab adapted for feedforward + win rate test |
| `lab_winrate_test.py` | LSTM before/after win rate test |
| `te_quantum_signal.json` | TEQA signal input (read by lab) |
| `top_50_manifest.json` | Metadata for top_50_experts/ |
| `etare_darwin.db` | Darwin feedforward population |
| `etare_redux_v3.db` | LSTM training population |
| `lstm_winrate_test.py` | LSTM evaluation with 30-bar sequences (correct method) |
| `lstm_retrain_fast.py` | LSTM retraining pipeline (10 experts, evolution, extinction) |
| `ETARE_50_Darwin.py` | Source of etare_darwin.db — has LSTM+GeneticWeights dual architecture |
| `etare_full_darwin.py` | Pure feedforward darwin (17 features, 6 actions, no DB save) |
| `etare_baseline_darwin.py` | Pure LSTM darwin (17 features, no backprop, Darwin selection only) |
| `etare context.txt` | Original ETARE article by Koshtenko (reference architecture) |
| `MASTER_CONFIG.json` | Trading config (DO NOT MODIFY without permission) |
| `config_loader.py` | Config import for all BRAIN scripts |

---

**This file is the single source of truth for all Mushroom Rabies Lab work. Update it when new tests are run or new findings emerge.**
