# ALGORITHM: VDJ_RECOMBINATION
## Quantum-Inspired Adaptive Immune System for Trading Strategy Generation

```
Date:       2026-02-08
Authors:    DooDoo + Claude
Status:     COMPLETE SPECIFICATION -- READY FOR IMPLEMENTATION
Parent:     TEQA v3.0 Neural-TE Integration (teqa_v3_neural_te.py)
Biological: RAG1/RAG2 V(D)J Recombination (domesticated Transib transposon)
```

---

## 0. THE BIOLOGICAL FOUNDATION

In vertebrate biology, RAG1/RAG2 recombinase -- itself a domesticated Transib
transposon (already in our TE family #24) -- performs V(D)J recombination to
generate the adaptive immune system. The process:

1. **V(D)J RECOMBINATION**: RAG1/RAG2 selects one Variable (V) segment, one
   Diversity (D) segment, and one Joining (J) segment from gene pools, then
   cuts and recombines them. The 12/23 rule ensures correct pairing.

2. **JUNCTIONAL DIVERSITY**: At each junction, the TdT enzyme randomly adds
   N-nucleotides, creating trillions of unique antibody configurations from
   a limited gene pool.

3. **CLONAL SELECTION**: B cells expressing antibodies that bind antigen
   proliferate; non-binders undergo apoptosis.

4. **AFFINITY MATURATION**: Winning B cells undergo somatic hypermutation in
   germinal centers, producing slightly modified antibodies. Those with higher
   affinity survive; those with lower affinity die.

5. **MEMORY B CELLS**: The highest-affinity antibodies are permanently stored
   as memory cells for instant recall upon re-exposure.

This is evolution on fast-forward: generate diversity, test against reality,
amplify winners, kill losers, remember the best. It is the most powerful
adaptive algorithm known in biology.

---

## 1. THE TRADING MAPPING

```
BIOLOGY                          TRADING ANALOG
------------------------------   -----------------------------------------
V segments (~65 gene options)    Entry signal generators (from 33 TE families)
D segments (~27 gene options)    Regime/context modifiers
J segments (~6 gene options)     Exit strategies
RAG1/RAG2 recombination         Combinatorial strategy assembly
12/23 rule                       Compatibility constraints (not all combos valid)
N-nucleotide junctional add      Random parameter perturbation at junctions
Antibody                         Micro-strategy (complete V+D+J + params)
Antigen                          Current market conditions (OHLCV + regime)
B cell clone                     Strategy instance running on live/sim data
Clonal selection                 Fitness-weighted survival
Affinity maturation              Parameter mutation on winning strategies
Somatic hypermutation            Small parameter tweaks within winning strategies
Memory B cell                    Domesticated strategy in SQLite DB
Apoptosis                        Strategy deletion (fitness below threshold)
Germinal center                  Walk-forward optimization window
Bone marrow                      Strategy generation pool (always producing new)
Thymus (negative selection)      Risk management filter (kill dangerous strats)
```

---

## 2. SEGMENT POOL DEFINITIONS

### 2.1 V Segments: Entry Signal Generators

The V segment determines HOW we enter a trade. Each V segment is a function
that maps market data to a directional signal. The 33 TE families from
`teqa_v3_neural_te.py` are organized into entry signal families:

```python
V_SEGMENTS = {
    # ---- MOMENTUM FAMILY (7 segments) ----
    # These TEs detect momentum-based entries
    "V_momentum_fast": {
        "te_source":     "BEL_Pao",          # qubit 0
        "signal":        "momentum",
        "lookback":      [5, 8, 13],          # Fibonacci lookbacks
        "entry_rule":    "momentum > threshold",
        "direction":     "trend-following",
        "description":   "Fast momentum burst detection",
    },
    "V_momentum_slow": {
        "te_source":     "Penelope",          # qubit 7
        "signal":        "trend_duration",
        "lookback":      [21, 34, 55],
        "entry_rule":    "trend_bars > threshold",
        "direction":     "trend-following",
        "description":   "Sustained trend detection",
    },
    "V_momentum_macd": {
        "te_source":     "Ty3_gypsy",         # qubit 3
        "signal":        "macd",
        "lookback":      [12, 26],
        "entry_rule":    "macd_cross > 0",
        "direction":     "trend-following",
        "description":   "MACD crossover momentum",
    },
    "V_momentum_ema": {
        "te_source":     "CACTA",             # qubit 11
        "signal":        "ema_crossover",
        "lookback":      [8, 21],
        "entry_rule":    "ema_fast > ema_slow",
        "direction":     "trend-following",
        "description":   "EMA crossover trend",
    },
    "V_momentum_strength": {
        "te_source":     "DIRS1",             # qubit 1
        "signal":        "trend_strength",
        "lookback":      [8, 21],
        "entry_rule":    "trend_strength > threshold",
        "direction":     "trend-following",
        "description":   "Directional trend strength",
    },
    "V_momentum_orderflow": {
        "te_source":     "Mavericks_Polinton", # qubit 17
        "signal":        "order_flow",
        "lookback":      [10],
        "entry_rule":    "imbalance > threshold",
        "direction":     "trend-following",
        "description":   "Volume order flow imbalance",
    },
    "V_momentum_price": {
        "te_source":     "LINE",              # qubit 6
        "signal":        "price_change",
        "lookback":      [1, 2, 3],
        "entry_rule":    "abs(return) > threshold",
        "direction":     "trend-following",
        "description":   "Raw price change momentum",
    },

    # ---- STRUCTURE FAMILY (8 segments) ----
    # These TEs detect structural/statistical entries
    "V_structure_rsi": {
        "te_source":     "Ty1_copia",         # qubit 2
        "signal":        "rsi",
        "lookback":      [14],
        "entry_rule":    "rsi < 30 OR rsi > 70",
        "direction":     "mean-reverting",
        "description":   "RSI overbought/oversold",
    },
    "V_structure_bollinger": {
        "te_source":     "Ty5",               # qubit 4
        "signal":        "bollinger_position",
        "lookback":      [20],
        "entry_rule":    "bb_pos < -1 OR bb_pos > 1",
        "direction":     "mean-reverting",
        "description":   "Bollinger band extreme",
    },
    "V_structure_meanrev": {
        "te_source":     "RTE",               # qubit 8
        "signal":        "mean_reversion",
        "lookback":      [20],
        "entry_rule":    "deviation > threshold",
        "direction":     "mean-reverting",
        "description":   "Mean reversion signal",
    },
    "V_structure_sr": {
        "te_source":     "I_element",         # qubit 15
        "signal":        "support_resistance",
        "lookback":      [20],
        "entry_rule":    "near_level AND bounce",
        "direction":     "mean-reverting",
        "description":   "Support/resistance bounce",
    },
    "V_structure_fractal": {
        "te_source":     "Mariner_Tc1",       # qubit 16
        "signal":        "fractal_dim",
        "lookback":      [30],
        "entry_rule":    "fd_shift detected",
        "direction":     "adaptive",
        "description":   "Fractal dimension regime shift",
    },
    "V_structure_autocorr": {
        "te_source":     "Transib",           # qubit 24
        "signal":        "autocorrelation",
        "lookback":      [30],
        "entry_rule":    "autocorr > threshold",
        "direction":     "adaptive",
        "description":   "Autocorrelation structure",
    },
    "V_structure_micro": {
        "te_source":     "PIF_Harbinger",     # qubit 20
        "signal":        "microstructure",
        "lookback":      [20],
        "entry_rule":    "microstructure > threshold",
        "direction":     "adaptive",
        "description":   "Microstructure signal",
    },
    "V_structure_pattern": {
        "te_source":     "L1_Neuronal",       # qubit 25
        "signal":        "pattern_repetition",
        "lookback":      [5, 10, 45],
        "entry_rule":    "pattern_corr > threshold",
        "direction":     "adaptive",
        "description":   "Neural pattern recognition",
    },

    # ---- VOLATILITY FAMILY (7 segments) ----
    # These TEs detect volatility-based entries
    "V_vol_short": {
        "te_source":     "Alu",               # qubit 5
        "signal":        "short_volatility",
        "lookback":      [5, 20],
        "entry_rule":    "vol_ratio > threshold",
        "direction":     "neutral",
        "description":   "Short-term volatility expansion",
    },
    "V_vol_atr": {
        "te_source":     "VIPER_Ngaro",       # qubit 10
        "signal":        "atr_ratio",
        "lookback":      [14],
        "entry_rule":    "atr_expanding",
        "direction":     "neutral",
        "description":   "ATR ratio expansion",
    },
    "V_vol_compression": {
        "te_source":     "Crypton",           # qubit 12
        "signal":        "compression_ratio",
        "lookback":      [50],
        "entry_rule":    "compression > threshold",
        "direction":     "neutral",
        "description":   "Information compression breakout",
    },
    "V_vol_spread": {
        "te_source":     "P_element",         # qubit 19
        "signal":        "spread_analysis",
        "lookback":      [10],
        "entry_rule":    "spread_expansion > threshold",
        "direction":     "neutral",
        "description":   "Spread expansion detection",
    },
    "V_vol_breakout": {
        "te_source":     "SVA_Regulatory",    # qubit 28
        "signal":        "compression_breakout",
        "lookback":      [10, 30],
        "entry_rule":    "breakout_from_compression",
        "direction":     "adaptive",
        "description":   "SVA regime breakout",
    },
    "V_vol_mutation": {
        "te_source":     "Mutator",           # qubit 18
        "signal":        "mutation_rate",
        "lookback":      [20],
        "entry_rule":    "sign_change_rate > threshold",
        "direction":     "neutral",
        "description":   "Price mutation rate (choppiness)",
    },
    "V_vol_noise": {
        "te_source":     "Alu_Exonization",   # qubit 29
        "signal":        "noise_pattern",
        "lookback":      [20],
        "entry_rule":    "hidden_autocorr detected",
        "direction":     "adaptive",
        "description":   "Hidden structure in noise",
    },

    # ---- FLOW FAMILY (6 segments) ----
    # These TEs detect flow/volume/session entries
    "V_flow_volume": {
        "te_source":     "SINE",              # qubit 9
        "signal":        "tick_volume",
        "lookback":      [20],
        "entry_rule":    "volume_spike detected",
        "direction":     "neutral",
        "description":   "Volume spike entry",
    },
    "V_flow_profile": {
        "te_source":     "Helitron",          # qubit 13
        "signal":        "volume_profile",
        "lookback":      [20],
        "entry_rule":    "profile_skew > threshold",
        "direction":     "adaptive",
        "description":   "Volume profile shift",
    },
    "V_flow_candle": {
        "te_source":     "hobo",              # qubit 14
        "signal":        "candle_pattern",
        "lookback":      [1],
        "entry_rule":    "body_ratio > threshold",
        "direction":     "adaptive",
        "description":   "Candlestick pattern",
    },
    "V_flow_gap": {
        "te_source":     "piggyBac",          # qubit 21
        "signal":        "gap_analysis",
        "lookback":      [1],
        "entry_rule":    "gap_size > threshold",
        "direction":     "adaptive",
        "description":   "Gap fill/continuation",
    },
    "V_flow_session": {
        "te_source":     "pogo",              # qubit 22
        "signal":        "session_overlap",
        "lookback":      [0],
        "entry_rule":    "in_session_overlap",
        "direction":     "neutral",
        "description":   "London/NY session overlap",
    },
    "V_flow_diversity": {
        "te_source":     "Rag_like",          # qubit 23
        "signal":        "diversity_index",
        "lookback":      [30],
        "entry_rule":    "diversity > threshold",
        "direction":     "neutral",
        "description":   "Signal diversity index",
    },

    # ---- NEURAL/CROSS FAMILY (5 segments) ----
    # These TEs use neural-specific and cross-instrument signals
    "V_neural_mosaic": {
        "te_source":     "L1_Somatic",        # qubit 26
        "signal":        "multi_tf_variance",
        "lookback":      [5, 10, 20, 50],
        "entry_rule":    "tf_agreement OR tf_divergence",
        "direction":     "adaptive",
        "description":   "Multi-timeframe neural mosaic",
    },
    "V_neural_synapse": {
        "te_source":     "HERV_Synapse",      # qubit 27
        "signal":        "cross_correlation",
        "lookback":      [50],
        "entry_rule":    "corr_shift detected",
        "direction":     "adaptive",
        "description":   "Cross-instrument synaptic signal",
    },
    "V_neural_arc": {
        "te_source":     "Arc_Capsid",        # qubit 32
        "signal":        "successful_pattern_echo",
        "lookback":      [0],
        "entry_rule":    "echo_from_winner",
        "direction":     "adaptive",
        "description":   "Arc capsid signal echo from winners",
    },
    "V_neural_snr": {
        "te_source":     "piwiRNA_Neural",    # qubit 31
        "signal":        "signal_noise_ratio",
        "lookback":      [30],
        "entry_rule":    "snr > threshold",
        "direction":     "neutral",
        "description":   "Signal quality filter",
    },
    "V_neural_trim28": {
        "te_source":     "TRIM28_Silencer",   # qubit 30
        "signal":        "drawdown",
        "lookback":      [0],
        "entry_rule":    "drawdown < max_threshold",
        "direction":     "risk-off",
        "description":   "Risk suppression check",
    },
}
# TOTAL: 33 V segments (one per TE family)
```

### 2.2 D Segments: Regime/Context Modifiers

The D segment determines the CONTEXT in which the V signal operates. It
modifies the V signal's parameters based on the current market regime.

```python
D_SEGMENTS = {
    # ---- PRIMARY REGIMES (4 segments) ----
    "D_trending_up": {
        "regime":          "TRENDING_BULLISH",
        "detection":       "EMA(8) > EMA(21) > EMA(50) AND ADX > 25",
        "v_modifier":      "amplify trend-following V, suppress mean-reverting V",
        "param_shift":     {"momentum_mult": 1.3, "reversion_mult": 0.5},
        "exit_bias":       "trail",
        "description":     "Strong bullish trend context",
    },
    "D_trending_down": {
        "regime":          "TRENDING_BEARISH",
        "detection":       "EMA(8) < EMA(21) < EMA(50) AND ADX > 25",
        "v_modifier":      "amplify trend-following V (short side), suppress mean-reverting V",
        "param_shift":     {"momentum_mult": 1.3, "reversion_mult": 0.5},
        "exit_bias":       "trail",
        "description":     "Strong bearish trend context",
    },
    "D_ranging": {
        "regime":          "RANGE_BOUND",
        "detection":       "ADX < 20 AND BB_width < percentile(20)",
        "v_modifier":      "amplify mean-reverting V, suppress trend-following V",
        "param_shift":     {"momentum_mult": 0.5, "reversion_mult": 1.5},
        "exit_bias":       "fixed_target",
        "description":     "Range-bound mean reversion context",
    },
    "D_volatile": {
        "regime":          "HIGH_VOLATILITY",
        "detection":       "ATR(14) > 2 * ATR(50) OR shock_label in [SHOCK, EXTREME]",
        "v_modifier":      "amplify volatility V, widen all stops",
        "param_shift":     {"vol_mult": 1.5, "sl_mult": 1.5, "lot_mult": 0.5},
        "exit_bias":       "time_based",
        "description":     "High volatility regime",
    },

    # ---- TRANSITION REGIMES (4 segments) ----
    "D_breakout_forming": {
        "regime":          "BREAKOUT_FORMING",
        "detection":       "compression_ratio > 2.0 AND vol_increasing",
        "v_modifier":      "prepare for breakout, tighten entries",
        "param_shift":     {"entry_threshold_mult": 1.3, "patience_mult": 2.0},
        "exit_bias":       "trail",
        "description":     "Compression before breakout",
    },
    "D_trend_exhaustion": {
        "regime":          "TREND_EXHAUSTION",
        "detection":       "RSI > 75 AND trend_duration > 20 AND volume_declining",
        "v_modifier":      "suppress trend-following V, boost reversal V",
        "param_shift":     {"momentum_mult": 0.3, "reversion_mult": 1.8},
        "exit_bias":       "fixed_target",
        "description":     "Trend running out of steam",
    },
    "D_correlation_shift": {
        "regime":          "CORRELATION_BREAKDOWN",
        "detection":       "cross_corr_delta > 0.3 within 4 hours",
        "v_modifier":      "activate neural/cross signals, isolate instrument",
        "param_shift":     {"neural_mult": 1.5, "cross_trust": 0.3},
        "exit_bias":       "time_based",
        "description":     "Cross-instrument correlation shifting",
    },
    "D_session_transition": {
        "regime":          "SESSION_TRANSITION",
        "detection":       "within 30 min of major session open/close",
        "v_modifier":      "boost flow/volume V segments, widen stops",
        "param_shift":     {"flow_mult": 1.5, "sl_mult": 1.3},
        "exit_bias":       "time_based",
        "description":     "Major session transition period",
    },

    # ---- SPECIAL REGIMES (5 segments) ----
    "D_news_shock": {
        "regime":          "NEWS_SHOCK",
        "detection":       "volume > 5 * avg_volume AND ATR spike > 3x",
        "v_modifier":      "suppress ALL entries for N bars, then activate vol V",
        "param_shift":     {"cooldown_bars": 5, "vol_mult": 2.0},
        "exit_bias":       "time_based",
        "description":     "News-driven market shock",
    },
    "D_low_liquidity": {
        "regime":          "LOW_LIQUIDITY",
        "detection":       "spread > 2 * avg_spread OR volume < 0.3 * avg_volume",
        "v_modifier":      "suppress all entries or require higher confidence",
        "param_shift":     {"confidence_mult": 1.5, "lot_mult": 0.3},
        "exit_bias":       "fixed_target",
        "description":     "Low liquidity / wide spread",
    },
    "D_momentum_divergence": {
        "regime":          "MOMENTUM_DIVERGENCE",
        "detection":       "price making new highs BUT momentum indicator declining",
        "v_modifier":      "activate reversal V segments, suppress momentum V",
        "param_shift":     {"momentum_mult": 0.3, "reversion_mult": 1.5},
        "exit_bias":       "fixed_target",
        "description":     "Hidden divergence forming",
    },
    "D_accumulation": {
        "regime":          "ACCUMULATION",
        "detection":       "narrow range AND increasing volume AND decreasing volatility",
        "v_modifier":      "patient entries, widen lookback",
        "param_shift":     {"patience_mult": 2.0, "lookback_mult": 1.5},
        "exit_bias":       "trail",
        "description":     "Smart money accumulation phase",
    },
    "D_distribution": {
        "regime":          "DISTRIBUTION",
        "detection":       "range expansion failing AND decreasing volume at highs/lows",
        "v_modifier":      "activate reversal V segments",
        "param_shift":     {"reversion_mult": 1.5, "momentum_mult": 0.5},
        "exit_bias":       "fixed_target",
        "description":     "Distribution phase before reversal",
    },
}
# TOTAL: 13 D segments
```

### 2.3 J Segments: Exit Strategies

The J segment determines HOW we exit the trade. Each J segment is a
complete exit management strategy.

```python
J_SEGMENTS = {
    # ---- TRAILING FAMILY ----
    "J_trail_atr": {
        "exit_type":       "TRAILING_STOP",
        "method":          "ATR-based trailing stop",
        "params": {
            "atr_period":      14,
            "atr_multiplier":  2.0,
            "activation_rr":   1.0,    # Activate trailing after 1:1 R:R
        },
        "tp_method":       "none (let trail catch it)",
        "time_limit_bars": 0,          # No time limit
        "description":     "ATR trailing stop, unlimited upside",
    },
    "J_trail_chandelier": {
        "exit_type":       "TRAILING_STOP",
        "method":          "Chandelier exit (highest high - ATR*mult)",
        "params": {
            "lookback":        22,
            "atr_multiplier":  3.0,
            "activation_rr":   0.5,
        },
        "tp_method":       "none",
        "time_limit_bars": 0,
        "description":     "Chandelier trailing for trends",
    },
    "J_trail_parabolic": {
        "exit_type":       "TRAILING_STOP",
        "method":          "Parabolic SAR trailing",
        "params": {
            "af_start":        0.02,
            "af_increment":    0.02,
            "af_max":          0.20,
        },
        "tp_method":       "none",
        "time_limit_bars": 0,
        "description":     "Accelerating parabolic trail",
    },

    # ---- FIXED TARGET FAMILY ----
    "J_fixed_rr2": {
        "exit_type":       "FIXED_TARGET",
        "method":          "Fixed 2:1 reward-to-risk",
        "params": {
            "rr_ratio":        2.0,
        },
        "tp_method":       "SL_distance * rr_ratio",
        "time_limit_bars": 200,
        "description":     "Conservative 2:1 R:R target",
    },
    "J_fixed_rr3": {
        "exit_type":       "FIXED_TARGET",
        "method":          "Fixed 3:1 reward-to-risk",
        "params": {
            "rr_ratio":        3.0,
        },
        "tp_method":       "SL_distance * rr_ratio",
        "time_limit_bars": 300,
        "description":     "Standard 3:1 R:R target",
    },

    # ---- DYNAMIC FAMILY ----
    "J_dynamic_partial": {
        "exit_type":       "DYNAMIC",
        "method":          "Partial close at TP1, trail remainder",
        "params": {
            "tp1_rr":          1.5,    # First target
            "tp1_close_pct":   50,     # Close 50% at TP1
            "trail_remainder": True,
            "trail_atr_mult":  1.5,
        },
        "tp_method":       "partial + trail",
        "time_limit_bars": 0,
        "description":     "50% partial TP then trail (matches MASTER_CONFIG)",
    },
    "J_dynamic_structure": {
        "exit_type":       "DYNAMIC",
        "method":          "Exit at next S/R level",
        "params": {
            "sr_lookback":     100,
            "min_rr":          1.0,    # Must offer at least 1:1
        },
        "tp_method":       "next_sr_level",
        "time_limit_bars": 500,
        "description":     "Structure-based dynamic target",
    },
    "J_dynamic_reversal": {
        "exit_type":       "DYNAMIC",
        "method":          "Exit on reversal signal from any V segment",
        "params": {
            "reversal_confidence": 0.6,  # Reversal signal must exceed this
            "min_rr":              0.5,  # Emergency minimum
        },
        "tp_method":       "reversal_detection",
        "time_limit_bars": 0,
        "description":     "Exit when opposing signal fires",
    },

    # ---- TIME-BASED FAMILY ----
    "J_time_fixed": {
        "exit_type":       "TIME_BASED",
        "method":          "Close after N bars regardless of P/L",
        "params": {
            "max_bars":        60,     # 1 hour on M1
            "move_sl_be":      True,   # Move SL to breakeven at 50% time
        },
        "tp_method":       "none (time exit)",
        "time_limit_bars": 60,
        "description":     "Time-based exit for choppy regimes",
    },
    "J_time_session": {
        "exit_type":       "TIME_BASED",
        "method":          "Close before session end",
        "params": {
            "close_before_minutes": 30,  # Close 30 min before session close
            "session_end_hour":     21,  # UTC
        },
        "tp_method":       "none (session exit)",
        "time_limit_bars": 0,
        "description":     "Session-end forced exit",
    },
}
# TOTAL: 10 J segments
```

---

## 3. THE 12/23 RULE: COMPATIBILITY CONSTRAINTS

In biology, the 12/23 rule ensures that RAG only joins segments flanked by
compatible recombination signal sequences (RSS). A 12-spacer RSS can ONLY
join with a 23-spacer RSS. This prevents nonsensical gene joins.

In trading, not all V+D+J combinations make biological/strategic sense.

### 3.1 Compatibility Matrix

```python
# Each V segment has a 12-spacer or 23-spacer RSS tag
# Each D segment has BOTH a 12 and 23 tag (one on each side)
# Each J segment has a 12 or 23 tag
# V must connect to D's left side, D's right side must connect to J
# 12 only pairs with 23

V_RSS = {
    # Momentum V segments have 23-spacer (pair with D's 12-left)
    "V_momentum_fast":      23,
    "V_momentum_slow":      23,
    "V_momentum_macd":      23,
    "V_momentum_ema":       23,
    "V_momentum_strength":  23,
    "V_momentum_orderflow": 23,
    "V_momentum_price":     23,
    # Structure V segments have 12-spacer (pair with D's 23-left)
    "V_structure_rsi":      12,
    "V_structure_bollinger": 12,
    "V_structure_meanrev":  12,
    "V_structure_sr":       12,
    "V_structure_fractal":  12,
    "V_structure_autocorr": 12,
    "V_structure_micro":    12,
    "V_structure_pattern":  12,
    # Volatility V segments have 23-spacer
    "V_vol_short":          23,
    "V_vol_atr":            23,
    "V_vol_compression":    23,
    "V_vol_spread":         23,
    "V_vol_breakout":       23,
    "V_vol_mutation":       23,
    "V_vol_noise":          23,
    # Flow V segments have 12-spacer
    "V_flow_volume":        12,
    "V_flow_profile":       12,
    "V_flow_candle":        12,
    "V_flow_gap":           12,
    "V_flow_session":       12,
    "V_flow_diversity":     12,
    # Neural V segments have BOTH (universal adapter)
    "V_neural_mosaic":      0,   # 0 = pairs with either
    "V_neural_synapse":     0,
    "V_neural_arc":         0,
    "V_neural_snr":         0,
    "V_neural_trim28":      0,
}

D_RSS = {
    # D segment has left RSS and right RSS
    # (left_rss, right_rss)
    "D_trending_up":          (12, 23),  # Accepts 23-V, emits to 12-J
    "D_trending_down":        (12, 23),
    "D_ranging":              (23, 12),  # Accepts 12-V, emits to 23-J
    "D_volatile":             (12, 12),  # Accepts 23-V, emits to 23-J
    "D_breakout_forming":     (12, 23),
    "D_trend_exhaustion":     (23, 12),
    "D_correlation_shift":    (12, 23),
    "D_session_transition":   (23, 12),
    "D_news_shock":           (12, 12),
    "D_low_liquidity":        (23, 23),
    "D_momentum_divergence":  (23, 12),
    "D_accumulation":         (12, 23),
    "D_distribution":         (23, 12),
}

J_RSS = {
    # Trailing exits have 12-spacer (pair with D's 23-right)
    "J_trail_atr":          12,
    "J_trail_chandelier":   12,
    "J_trail_parabolic":    12,
    # Fixed exits have 23-spacer (pair with D's 12-right)
    "J_fixed_rr2":          23,
    "J_fixed_rr3":          23,
    # Dynamic exits have BOTH
    "J_dynamic_partial":    0,
    "J_dynamic_structure":  0,
    "J_dynamic_reversal":   0,
    # Time exits have 23-spacer
    "J_time_fixed":         23,
    "J_time_session":       23,
}

def rss_compatible(v_rss, d_left_rss, d_right_rss, j_rss):
    """
    Check 12/23 rule compatibility.
    V connects to D's left side, D's right side connects to J.
    12 pairs with 23. 0 (universal) pairs with anything.
    """
    # V-D junction
    if v_rss != 0 and d_left_rss != 0:
        if v_rss == d_left_rss:  # Same spacer = INCOMPATIBLE
            return False
    # D-J junction
    if d_right_rss != 0 and j_rss != 0:
        if d_right_rss == j_rss:  # Same spacer = INCOMPATIBLE
            return False
    return True
```

### 3.2 Valid Combination Count

```
Total raw combinations:  33 V * 13 D * 10 J = 4,290
After 12/23 rule filter: ~2,400 valid combinations (depends on RSS assignments)
With junctional diversity: 2,400 * ~100 parameter variations = ~240,000 unique antibodies
```

This parallels biology: 65V * 27D * 6J = 10,530 base combinations, but
junctional diversity creates billions of unique antibodies.

---

## 4. QUANTUM CIRCUIT DESIGN

### 4.1 Qubit Mapping

The VDJ recombination quantum circuit uses a purpose-built architecture
that is SEPARATE from but INFORMED by the main TEQA 33-qubit circuit.

```
VDJ QUANTUM CIRCUIT: 16 qubits
=================================

Qubits 0-5:   V segment selector (2^6 = 64, encodes 33 V segments)
Qubits 6-9:   D segment selector (2^4 = 16, encodes 13 D segments)
Qubits 10-13: J segment selector (2^4 = 16, encodes 10 J segments)
Qubit  14:    RSS compatibility flag
Qubit  15:    Junctional diversity seed
```

### 4.2 Recombination as Quantum Operations

```python
def build_vdj_circuit(te_activations: List[Dict], shock_level: float) -> QuantumCircuit:
    """
    Build the VDJ recombination quantum circuit.

    The circuit creates a superposition of all valid V+D+J combinations,
    weighted by current TE activation strengths, then collapses to select
    one specific antibody (micro-strategy).
    """
    qc = QuantumCircuit(16, 16)

    # ===== STEP 1: V SEGMENT SUPERPOSITION =====
    # Put V-selector qubits into weighted superposition based on
    # current TE activation strengths.
    # Strong TE activation = higher amplitude for that V segment.
    v_amplitudes = compute_v_amplitudes(te_activations)
    # Encode into qubits 0-5 using amplitude encoding
    # (Uses RY rotations to create non-uniform superposition)
    for i in range(6):
        angle = v_amplitudes[i] * math.pi
        qc.ry(angle, i)

    # ===== STEP 2: D SEGMENT SUPERPOSITION =====
    # Put D-selector qubits into superposition weighted by
    # regime detection strengths.
    d_amplitudes = compute_d_amplitudes(te_activations, shock_level)
    for i in range(4):
        angle = d_amplitudes[i] * math.pi
        qc.ry(angle, 6 + i)

    # ===== STEP 3: J SEGMENT SUPERPOSITION =====
    # Put J-selector qubits into superposition.
    # Bias toward certain exits based on D regime.
    j_amplitudes = compute_j_amplitudes()
    for i in range(4):
        angle = j_amplitudes[i] * math.pi
        qc.ry(angle, 10 + i)

    # ===== STEP 4: ENTANGLEMENT (12/23 RULE) =====
    # Entangle V-D and D-J to enforce compatibility.
    # CNOT gates create correlations that suppress incompatible combos.

    # V-D entanglement: V qubits control D qubits
    qc.cx(0, 6)   # V bit 0 influences D bit 0
    qc.cx(1, 7)   # V bit 1 influences D bit 1
    qc.cx(2, 8)   # ...
    # These create quantum correlations that bias toward compatible pairs

    # D-J entanglement: D qubits control J qubits
    qc.cx(6, 10)
    qc.cx(7, 11)
    qc.cx(8, 12)

    # ===== STEP 5: RSS COMPATIBILITY ORACLE =====
    # Qubit 14 flags whether the current superposition state
    # represents a valid 12/23 combination.
    # This is a multi-controlled gate that checks RSS rules.
    # States that violate 12/23 get their amplitude reduced.
    qc.h(14)  # Put flag in superposition
    # Phase kickback: incompatible states get phase-flipped
    # (This implements Grover-like amplitude amplification
    #  of valid combinations)
    apply_rss_oracle(qc, v_qubits=[0,1,2,3,4,5],
                     d_qubits=[6,7,8,9],
                     j_qubits=[10,11,12,13],
                     flag_qubit=14)

    # ===== STEP 6: JUNCTIONAL DIVERSITY =====
    # Qubit 15 adds quantum randomness to the junction points.
    # This seeds the parameter perturbation that creates
    # junctional diversity in the selected antibody.
    qc.h(15)     # Full superposition
    qc.cz(14, 15)  # Entangle with RSS flag
    # Additional rotation from shock level (stress = more diversity)
    qc.ry(shock_level * math.pi * 0.3, 15)

    # ===== STEP 7: MEASUREMENT =====
    qc.measure(range(16), range(16))

    return qc
```

### 4.3 Amplitude Computation

```python
def compute_v_amplitudes(te_activations: List[Dict]) -> List[float]:
    """
    Convert 33 TE activation strengths into 6-qubit amplitude encoding.

    Each V segment maps to one TE. The TE's activation strength determines
    how likely that V segment is to be selected. This is NOT uniform random --
    it is biased by current market conditions.

    The 6-qubit encoding uses the bit pattern to index into V segments:
    |000000> = V_momentum_fast, |000001> = V_momentum_slow, etc.
    """
    # Get all 33 V segment strengths from their TE sources
    v_strengths = []
    for v_name, v_def in V_SEGMENTS.items():
        te_name = v_def["te_source"]
        act = next((a for a in te_activations if a["te"] == te_name), None)
        strength = act["strength"] if act else 0.0
        v_strengths.append(strength)

    # Pad to 64 (2^6) with zeros for unused states
    while len(v_strengths) < 64:
        v_strengths.append(0.0)

    # Normalize to valid quantum amplitudes
    total = sum(s**2 for s in v_strengths) + 1e-10
    v_strengths = [s / math.sqrt(total) for s in v_strengths]

    # Convert to 6 rotation angles for RY gates
    # This is a simplified encoding; full amplitude encoding
    # would use more complex decomposition
    amplitudes = [0.0] * 6
    for bit in range(6):
        # Each qubit's rotation is determined by the average strength
        # of V segments that have this bit set in their index
        set_strengths = []
        for idx, s in enumerate(v_strengths[:33]):
            if (idx >> bit) & 1:
                set_strengths.append(s)
        if set_strengths:
            amplitudes[bit] = sum(set_strengths) / len(set_strengths)

    return amplitudes
```

### 4.4 Measurement Interpretation

```python
def interpret_vdj_measurement(bitstring: str) -> dict:
    """
    Interpret a 16-bit measurement result as a V+D+J selection.

    Returns:
        {
            "v_index": int,      # Which V segment (0-32)
            "d_index": int,      # Which D segment (0-12)
            "j_index": int,      # Which J segment (0-9)
            "rss_valid": bool,   # 12/23 rule satisfied
            "junction_seed": int # Seed for parameter perturbation
        }
    """
    # Qiskit returns bitstrings in reverse order
    bits = bitstring[::-1]

    v_index = int(bits[0:6], 2)    # Qubits 0-5
    d_index = int(bits[6:10], 2)   # Qubits 6-9
    j_index = int(bits[10:14], 2)  # Qubits 10-13
    rss_flag = int(bits[14])       # Qubit 14
    junction = int(bits[15])       # Qubit 15

    # Clamp to valid segment ranges
    v_list = list(V_SEGMENTS.keys())
    d_list = list(D_SEGMENTS.keys())
    j_list = list(J_SEGMENTS.keys())

    v_index = v_index % len(v_list)
    d_index = d_index % len(d_list)
    j_index = j_index % len(j_list)

    return {
        "v_index": v_index,
        "v_name":  v_list[v_index],
        "d_index": d_index,
        "d_name":  d_list[d_index],
        "j_index": j_index,
        "j_name":  j_list[j_index],
        "rss_valid": bool(rss_flag),
        "junction_seed": junction,
    }
```

---

## 5. JUNCTIONAL DIVERSITY: PARAMETER PERTURBATION

In biology, TdT enzyme adds random nucleotides at V-D and D-J junctions,
creating astronomical combinatorial diversity from limited gene segments.

In trading, this maps to random parameter perturbation at the junction
points between V, D, and J segments.

### 5.1 Mathematical Formulation

```
For an antibody A = (V_i, D_j, J_k):

V parameters:   theta_V = {lookback, threshold, activation_fn_params}
D parameters:   theta_D = {regime_thresholds, param_shift_factors}
J parameters:   theta_J = {exit_params, time_limits, trail_factors}

Junctional diversity adds N-nucleotides at each junction:

Junction V-D:
    theta_VD = theta_V + epsilon_VD
    where epsilon_VD ~ N(0, sigma_VD^2 * I)
    sigma_VD = base_sigma * (1 + shock_level * 0.5)

Junction D-J:
    theta_DJ = theta_D + epsilon_DJ
    where epsilon_DJ ~ N(0, sigma_DJ^2 * I)
    sigma_DJ = base_sigma * (1 + shock_level * 0.3)

Full antibody parameter vector:
    Theta_A = [theta_VD, theta_D, theta_DJ]

With base_sigma = 0.1 (10% perturbation), this creates:
    - Under calm markets: +-10% parameter jitter
    - Under shock: +-15-20% parameter jitter (more diversity)
```

### 5.2 Implementation

```python
def apply_junctional_diversity(
    antibody: dict,
    junction_seed: int,
    shock_level: float,
    base_sigma: float = 0.1
) -> dict:
    """
    Apply TdT-like random parameter perturbation at V-D and D-J junctions.

    This is the key diversity mechanism that turns 2,400 valid V+D+J combos
    into ~240,000+ unique micro-strategies.
    """
    rng = np.random.RandomState(junction_seed)

    v_def = V_SEGMENTS[antibody["v_name"]]
    d_def = D_SEGMENTS[antibody["d_name"]]
    j_def = J_SEGMENTS[antibody["j_name"]]

    # ---- V-D Junction: Perturb V entry parameters ----
    sigma_vd = base_sigma * (1 + shock_level * 0.5)
    perturbed_v = {}
    for lookback in v_def["lookback"]:
        # Perturb lookback period (integer, must stay positive)
        delta = int(rng.normal(0, sigma_vd * lookback))
        perturbed_v[f"lookback_{lookback}"] = max(2, lookback + delta)

    # Perturb D's param_shift factors
    perturbed_d = {}
    for key, val in d_def["param_shift"].items():
        delta = rng.normal(0, sigma_vd * abs(val))
        perturbed_d[key] = max(0.01, val + delta)

    # ---- D-J Junction: Perturb J exit parameters ----
    sigma_dj = base_sigma * (1 + shock_level * 0.3)
    perturbed_j = {}
    for key, val in j_def["params"].items():
        if isinstance(val, (int, float)):
            delta = rng.normal(0, sigma_dj * abs(val + 1e-10))
            if isinstance(val, int):
                perturbed_j[key] = max(1, int(val + delta))
            else:
                perturbed_j[key] = max(0.01, val + delta)
        else:
            perturbed_j[key] = val  # Non-numeric params unchanged

    antibody["perturbed_v_params"] = perturbed_v
    antibody["perturbed_d_params"] = perturbed_d
    antibody["perturbed_j_params"] = perturbed_j

    return antibody
```

---

## 6. CLONAL SELECTION: FITNESS FUNCTION

### 6.1 Antigen: Market Reality

The "antigen" that antibodies are tested against is the actual market data.
Each antibody (micro-strategy) is run through a walk-forward window and
scored on its performance.

### 6.2 Fitness Function

```python
def fitness_clonal_selection(antibody_result: dict) -> float:
    """
    Clonal selection fitness function.

    In biology: how strongly does the antibody bind the antigen?
    In trading: how well does the micro-strategy perform on recent data?

    The fitness is a composite score that prevents overfitting by
    incorporating multiple independent metrics.

    Formula:
        F(A) = w1 * posterior_WR + w2 * profit_factor_norm
             + w3 * sortino_norm + w4 * consistency
             - w5 * max_dd_penalty - w6 * trade_count_penalty

    Where:
        posterior_WR      = Bayesian win rate with Beta(10,10) prior
        profit_factor_norm = min(1.0, profit_factor / 3.0)
        sortino_norm      = min(1.0, sortino_ratio / 3.0)
        consistency       = 1 - stddev(per_trade_returns) / mean(per_trade_returns)
        max_dd_penalty    = max_drawdown / account_equity
        trade_count_penalty = max(0, (MIN_TRADES - n_trades) / MIN_TRADES)

    Weights:
        w1=0.25, w2=0.20, w3=0.20, w4=0.15, w5=0.10, w6=0.10
    """
    r = antibody_result  # Shorthand

    n_trades = r["n_trades"]
    n_wins = r["n_wins"]
    n_losses = n_trades - n_wins

    # ---- Component 1: Bayesian posterior win rate ----
    # Beta(10,10) prior: starts at 50%, needs real data to move
    ALPHA_PRIOR = 10
    BETA_PRIOR = 10
    posterior_wr = (ALPHA_PRIOR + n_wins) / (ALPHA_PRIOR + BETA_PRIOR + n_trades)

    # ---- Component 2: Profit factor ----
    avg_win = r["total_profit"] / max(1, n_wins) if r["total_profit"] > 0 else 0
    avg_loss = abs(r["total_loss"]) / max(1, n_losses) if r["total_loss"] < 0 else 0.01
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
    pf_norm = min(1.0, profit_factor / 3.0)  # Cap at PF=3 for normalization

    # ---- Component 3: Sortino ratio ----
    if len(r["trade_returns"]) > 1:
        mean_return = np.mean(r["trade_returns"])
        downside_returns = [ret for ret in r["trade_returns"] if ret < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-10
        sortino = mean_return / (downside_std + 1e-10)
        sortino_norm = min(1.0, max(0.0, sortino / 3.0))
    else:
        sortino_norm = 0.0

    # ---- Component 4: Consistency ----
    if len(r["trade_returns"]) > 1 and np.mean(r["trade_returns"]) != 0:
        consistency = 1.0 - min(1.0, np.std(r["trade_returns"]) /
                                (abs(np.mean(r["trade_returns"])) + 1e-10))
        consistency = max(0.0, consistency)
    else:
        consistency = 0.0

    # ---- Component 5: Maximum drawdown penalty ----
    max_dd_penalty = min(1.0, r["max_drawdown"] / (r.get("account_equity", 5000) * 0.10))

    # ---- Component 6: Trade count penalty ----
    MIN_TRADES = 20
    if n_trades < MIN_TRADES:
        trade_penalty = (MIN_TRADES - n_trades) / MIN_TRADES
    else:
        trade_penalty = 0.0

    # ---- COMPOSITE FITNESS ----
    fitness = (
        0.25 * posterior_wr +
        0.20 * pf_norm +
        0.20 * sortino_norm +
        0.15 * consistency -
        0.10 * max_dd_penalty -
        0.10 * trade_penalty
    )

    return float(np.clip(fitness, 0.0, 1.0))
```

### 6.3 Selection Thresholds

```python
# Clonal Selection Thresholds
APOPTOSIS_THRESHOLD = 0.25        # Below this: strategy dies (apoptosis)
SURVIVAL_THRESHOLD = 0.40         # Above this: strategy survives
PROLIFERATION_THRESHOLD = 0.55    # Above this: strategy proliferates (cloned)
MEMORY_THRESHOLD = 0.70           # Above this: becomes memory B cell (domesticated)

# Population management
MAX_ACTIVE_ANTIBODIES = 50        # Maximum strategies running simultaneously
MIN_ACTIVE_ANTIBODIES = 10        # Always maintain at least this many
GENERATION_SIZE = 20              # New antibodies per generation cycle
EVALUATION_WINDOW_BARS = 500      # Walk-forward window for evaluation (~8 hours on M1)
```

---

## 7. AFFINITY MATURATION: SOMATIC HYPERMUTATION

### 7.1 Biological Parallel

After clonal selection, winning B cells enter germinal centers where they
undergo somatic hypermutation -- small, random mutations in the antibody
variable region. Cells with improved binding survive; cells with worse
binding die. This iterative process sharpens the antibody's specificity.

### 7.2 Trading Implementation

```python
def affinity_maturation(
    winner: dict,
    n_mutants: int = 5,
    mutation_rate: float = 0.05,
    generation: int = 0
) -> List[dict]:
    """
    Somatic hypermutation of a winning antibody.

    Creates n_mutants variants of the winning strategy by making small
    parameter adjustments. These variants compete against each other
    and the parent. The best survives.

    The mutation_rate decreases as the generation increases, modeling
    how affinity maturation converges over time:

        effective_rate = mutation_rate / (1 + 0.1 * generation)

    This ensures early generations explore broadly while later
    generations fine-tune.

    Mathematical formulation:
        For each parameter theta_i in the winner:
            theta_i_mutant = theta_i * (1 + N(0, effective_rate))

        For discrete parameters (lookback periods, bar counts):
            theta_i_mutant = theta_i + Binomial(1, effective_rate) * Uniform(-2, +2)
    """
    effective_rate = mutation_rate / (1.0 + 0.1 * generation)
    mutants = []

    for m in range(n_mutants):
        mutant = deepcopy(winner)
        mutant["parent_id"] = winner["antibody_id"]
        mutant["antibody_id"] = f"{winner['antibody_id']}_m{generation}_{m}"
        mutant["generation"] = generation + 1
        mutant["mutation_type"] = "somatic_hypermutation"

        # Mutate V junction parameters
        for key, val in mutant["perturbed_v_params"].items():
            if isinstance(val, float):
                mutant["perturbed_v_params"][key] = val * (1 + np.random.normal(0, effective_rate))
            elif isinstance(val, int):
                if np.random.random() < effective_rate:
                    mutant["perturbed_v_params"][key] = max(2, val + np.random.randint(-2, 3))

        # Mutate D modifier parameters
        for key, val in mutant["perturbed_d_params"].items():
            if isinstance(val, (int, float)):
                mutant["perturbed_d_params"][key] = max(0.01, val * (1 + np.random.normal(0, effective_rate)))

        # Mutate J exit parameters
        for key, val in mutant["perturbed_j_params"].items():
            if isinstance(val, float):
                mutant["perturbed_j_params"][key] = max(0.01, val * (1 + np.random.normal(0, effective_rate)))
            elif isinstance(val, int):
                if np.random.random() < effective_rate:
                    mutant["perturbed_j_params"][key] = max(1, val + np.random.randint(-2, 3))

        mutants.append(mutant)

    return mutants
```

### 7.3 Convergence Criterion

```
An antibody is considered "mature" when:
    1. It has survived >= 3 rounds of affinity maturation
    2. Its fitness has not improved by more than 1% in the last round
    3. Its parameter values have stabilized (variance < 0.02)

Mature antibodies are promoted to memory B cell status.
```

---

## 8. MEMORY B CELLS: DOMESTICATION INTO THE DATABASE

### 8.1 Integration with TEDomesticationTracker

Memory B cells are the permanently stored, battle-tested strategies that
provide instant response upon re-exposure to a recognized market pattern.

```python
def promote_to_memory_cell(
    antibody: dict,
    domestication_tracker: TEDomesticationTracker,
    db_path: str = "vdj_memory_cells.db"
):
    """
    Promote a mature antibody to memory B cell status.

    This writes the complete strategy specification to the VDJ memory database
    AND registers the associated TE pattern in the TEDomesticationTracker so
    that the main TEQA engine's get_boost() can benefit from this discovery.
    """
    # Step 1: Save full antibody specification to VDJ DB
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_cells (
                antibody_id     TEXT PRIMARY KEY,
                v_segment       TEXT NOT NULL,
                d_segment       TEXT NOT NULL,
                j_segment       TEXT NOT NULL,
                v_params        TEXT,           -- JSON of perturbed params
                d_params        TEXT,           -- JSON of perturbed params
                j_params        TEXT,           -- JSON of perturbed params
                fitness         REAL NOT NULL,
                posterior_wr    REAL,
                profit_factor   REAL,
                sortino         REAL,
                n_trades        INTEGER,
                n_wins          INTEGER,
                generation      INTEGER,
                parent_id       TEXT,
                maturation_rounds INTEGER,
                created_at      TEXT,
                last_activated  TEXT,
                activation_count INTEGER DEFAULT 0,
                active          INTEGER DEFAULT 1,
                te_combo        TEXT             -- TE family hash for domestication link
            )
        """)

        # Generate TE combo from V segment's source TE
        te_combo = antibody["v_name"].replace("V_", "")
        active_tes = [V_SEGMENTS[antibody["v_name"]]["te_source"]]

        conn.execute("""
            INSERT OR REPLACE INTO memory_cells VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            antibody["antibody_id"],
            antibody["v_name"],
            antibody["d_name"],
            antibody["j_name"],
            json.dumps(antibody.get("perturbed_v_params", {})),
            json.dumps(antibody.get("perturbed_d_params", {})),
            json.dumps(antibody.get("perturbed_j_params", {})),
            antibody["fitness"],
            antibody.get("posterior_wr", 0.5),
            antibody.get("profit_factor", 0.0),
            antibody.get("sortino", 0.0),
            antibody.get("n_trades", 0),
            antibody.get("n_wins", 0),
            antibody.get("generation", 0),
            antibody.get("parent_id", ""),
            antibody.get("maturation_rounds", 0),
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            0,
            "+".join(sorted(active_tes)),
        ))
        conn.commit()

    # Step 2: Register the TE pattern in TEDomesticationTracker
    # This cross-links VDJ memory with TEQA domestication for boost
    domestication_tracker.record_pattern(
        active_tes=active_tes,
        won=True,
        profit=antibody.get("total_profit", 0.0)
    )
```

### 8.2 Memory Cell Recall

```python
def recall_memory_cells(
    current_activations: List[Dict],
    current_regime: str,
    db_path: str = "vdj_memory_cells.db"
) -> List[dict]:
    """
    When the market presents a recognized pattern (antigen re-exposure),
    memory B cells activate instantly -- no need to go through the full
    VDJ recombination process.

    This is like the secondary immune response: faster and stronger.

    Matching criteria:
        1. V segment's source TE is currently active (strength > 0.5)
        2. D segment's regime matches current detected regime
        3. Memory cell fitness > MEMORY_THRESHOLD
        4. Memory cell not expired (last_activated within 30 days)
    """
    recalled = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM memory_cells
            WHERE active = 1 AND fitness >= ?
            ORDER BY fitness DESC
        """, (MEMORY_THRESHOLD,)).fetchall()

    for row in rows:
        v_name = row["v_segment"]
        d_name = row["d_segment"]

        # Check V segment activation
        te_source = V_SEGMENTS[v_name]["te_source"]
        te_act = next((a for a in current_activations if a["te"] == te_source), None)
        if te_act is None or te_act["strength"] < 0.5:
            continue

        # Check D segment regime match
        d_regime = D_SEGMENTS[d_name]["regime"]
        if d_regime != current_regime:
            continue

        # Check expiry
        last_act = datetime.fromisoformat(row["last_activated"])
        if (datetime.now() - last_act).days > DOMESTICATION_EXPIRY_DAYS:
            continue

        # RECALL: This memory cell matches current conditions
        recalled.append({
            "antibody_id":   row["antibody_id"],
            "v_segment":     v_name,
            "d_segment":     d_name,
            "j_segment":     row["j_segment"],
            "v_params":      json.loads(row["v_params"]),
            "d_params":      json.loads(row["d_params"]),
            "j_params":      json.loads(row["j_params"]),
            "fitness":       row["fitness"],
            "source":        "MEMORY_CELL",
            "n_trades":      row["n_trades"],
        })

    return recalled
```

---

## 9. THE FULL ALGORITHM: ALGORITHM_VDJ_RECOMBINATION

### 9.1 Complete Pseudocode

```
ALGORITHM VDJ_RECOMBINATION
============================================================

INPUT:
    bars:             OHLCV market data (N x 5 numpy array)
    symbol:           trading instrument (e.g., "BTCUSD")
    account_equity:   current account balance
    drawdown:         current drawdown fraction
    te_activations:   from TEQAv3Engine.analyze()
    shock_level:      from GenomicShockDetector
    shock_label:      "CALM" | "NORMAL" | "ELEVATED" | "SHOCK" | "EXTREME"

OUTPUT:
    selected_strategy: the chosen antibody (micro-strategy) with all params
    action:           BUY | SELL | HOLD
    confidence:       0.0 - 1.0
    lot_multiplier:   position sizing multiplier

STATE (persistent across calls):
    active_antibodies: List[Antibody]    -- currently running strategies
    memory_cells:      SQLite DB         -- domesticated strategies
    generation:        int               -- current generation number
    bone_marrow_pool:  List[Antibody]    -- newly generated, untested

============================================================

PROCEDURE:

    # ============================================
    # PHASE 0: TRIM28 CHECK (Emergency Suppression)
    # ============================================
    IF shock_label == "EXTREME":
        RETURN (HOLD, 0.0, 1.0, "TRIM28_SUPPRESSION")
    END IF

    # ============================================
    # PHASE 1: MEMORY CELL RECALL (Secondary Response)
    # ============================================
    # First, check if any memory B cells match current conditions.
    # This is the FAST path -- no recombination needed.

    current_regime = detect_regime(bars, te_activations)
    memory_matches = recall_memory_cells(te_activations, current_regime)

    IF memory_matches IS NOT EMPTY:
        # Pick highest-fitness memory cell
        best_memory = max(memory_matches, key=lambda m: m.fitness)

        # Update last_activated timestamp
        update_memory_activation(best_memory.antibody_id)

        # Execute the memory cell's strategy directly
        signal = execute_antibody_signal(best_memory, bars, te_activations)

        IF signal.passes_gates():
            RETURN (signal.action, signal.confidence * 1.2, signal.lot_mult,
                    f"MEMORY_RECALL:{best_memory.antibody_id}")
        END IF
    END IF

    # ============================================
    # PHASE 2: BONE MARROW -- Generate New Antibodies
    # ============================================
    # The bone marrow continuously generates new naive B cells
    # (untested micro-strategies) via V(D)J recombination.

    IF len(bone_marrow_pool) < GENERATION_SIZE:

        # Build VDJ quantum circuit
        vdj_circuit = build_vdj_circuit(te_activations, shock_level)

        # Execute circuit with multiple shots
        # Each shot produces one candidate V+D+J combination
        results = quantum_engine.execute_circuit(vdj_circuit, shots=GENERATION_SIZE * 4)

        # Interpret measurements into antibody specifications
        FOR bitstring, count IN results.counts:
            selection = interpret_vdj_measurement(bitstring)

            # Apply 12/23 rule filter
            IF NOT rss_compatible(selection):
                CONTINUE    # This combination is sterile
            END IF

            # Apply junctional diversity
            antibody = create_antibody(selection)
            antibody = apply_junctional_diversity(
                antibody,
                junction_seed=hash(bitstring),
                shock_level=shock_level
            )

            bone_marrow_pool.append(antibody)
        END FOR
    END IF

    # ============================================
    # PHASE 3: THYMIC SELECTION (Negative Selection)
    # ============================================
    # In biology, the thymus kills T cells that would attack self.
    # In trading, we kill strategies that violate risk management rules.

    FOR antibody IN bone_marrow_pool:
        IF violates_risk_rules(antibody):
            bone_marrow_pool.remove(antibody)  # Negative selection (death)
        END IF
    END FOR

    def violates_risk_rules(antibody):
        """
        Thymic negative selection: kill strategies that would attack 'self'
        (i.e., blow up the account).

        Kill if:
            1. Exit strategy has no stop loss mechanism
            2. Position size multiplier > 3x
            3. V+D combo produces undefined risk (no loss bound)
            4. Strategy requires data we don't have access to
        """
        j_def = J_SEGMENTS[antibody["j_name"]]
        if j_def["exit_type"] == "TIME_BASED" and not j_def["params"].get("move_sl_be"):
            return True   # Time-based exit without SL = unlimited risk
        if antibody.get("perturbed_d_params", {}).get("lot_mult", 1) > 3.0:
            return True   # Excessive position size
        return False

    # ============================================
    # PHASE 4: ANTIGEN EXPOSURE (Walk-Forward Test)
    # ============================================
    # Move naive B cells from bone marrow to active circulation.
    # Expose them to the antigen (recent market data).

    # Graduate from bone marrow to active pool
    new_graduates = bone_marrow_pool[:MAX_ACTIVE_ANTIBODIES - len(active_antibodies)]
    bone_marrow_pool = bone_marrow_pool[len(new_graduates):]
    active_antibodies.extend(new_graduates)

    # Run each active antibody against recent market data (walk-forward)
    FOR antibody IN active_antibodies:
        antibody.result = simulate_walk_forward(
            antibody,
            bars[-EVALUATION_WINDOW_BARS:],
            symbol
        )
        antibody.fitness = fitness_clonal_selection(antibody.result)
    END FOR

    # ============================================
    # PHASE 5: CLONAL SELECTION
    # ============================================
    # Antibodies that bind antigen (profitable) survive and proliferate.
    # Non-binders undergo apoptosis (deletion).

    survivors = []
    proliferators = []
    memory_candidates = []

    FOR antibody IN active_antibodies:
        IF antibody.fitness < APOPTOSIS_THRESHOLD:
            # APOPTOSIS: This strategy dies
            log(f"APOPTOSIS: {antibody.antibody_id} fitness={antibody.fitness:.3f}")
            CONTINUE

        ELSE IF antibody.fitness < SURVIVAL_THRESHOLD:
            # ANERGY: Strategy survives but doesn't grow
            survivors.append(antibody)

        ELSE IF antibody.fitness < PROLIFERATION_THRESHOLD:
            # SURVIVAL: Strategy survives and gets tested more
            survivors.append(antibody)

        ELSE IF antibody.fitness < MEMORY_THRESHOLD:
            # PROLIFERATION: Clone this strategy (with mutations)
            proliferators.append(antibody)
            survivors.append(antibody)

        ELSE:
            # MEMORY B CELL CANDIDATE: Exceptional performance
            memory_candidates.append(antibody)
            survivors.append(antibody)
        END IF
    END FOR

    active_antibodies = survivors

    # ============================================
    # PHASE 6: AFFINITY MATURATION (Germinal Center)
    # ============================================
    # Proliferating antibodies undergo somatic hypermutation.

    FOR antibody IN proliferators:
        mutants = affinity_maturation(
            antibody,
            n_mutants=5,
            mutation_rate=0.05,
            generation=antibody.generation
        )

        # Test mutants against same antigen (recent data)
        FOR mutant IN mutants:
            mutant.result = simulate_walk_forward(mutant, bars[-EVALUATION_WINDOW_BARS:], symbol)
            mutant.fitness = fitness_clonal_selection(mutant.result)
        END FOR

        # Keep the best mutant (if better than parent)
        best_mutant = max(mutants, key=lambda m: m.fitness)
        IF best_mutant.fitness > antibody.fitness:
            active_antibodies.append(best_mutant)
            antibody.maturation_rounds += 1
        END IF
    END FOR

    # ============================================
    # PHASE 7: MEMORY B CELL PROMOTION
    # ============================================
    # Exceptional antibodies become permanent memory cells.

    FOR antibody IN memory_candidates:
        IF antibody.fitness >= MEMORY_THRESHOLD AND antibody.maturation_rounds >= 3:
            promote_to_memory_cell(antibody, domestication_tracker)
            log(f"MEMORY B CELL: {antibody.antibody_id} fitness={antibody.fitness:.3f} "
                f"WR={antibody.posterior_wr:.1%}")
        END IF
    END FOR

    # ============================================
    # PHASE 8: CONSENSUS SIGNAL GENERATION
    # ============================================
    # All surviving antibodies vote on market direction.
    # This mirrors the polyclonal immune response --
    # multiple antibodies recognizing different epitopes of the same antigen.

    votes = []
    FOR antibody IN active_antibodies:
        signal = execute_antibody_signal(antibody, bars, te_activations)
        votes.append({
            "direction": signal.direction,
            "confidence": signal.confidence,
            "fitness": antibody.fitness,
            "antibody_id": antibody.antibody_id,
        })
    END FOR

    # Fitness-weighted vote
    total_fitness = sum(v.fitness for v in votes)
    IF total_fitness > 0:
        weighted_direction = sum(
            v.direction * v.confidence * v.fitness for v in votes
        ) / total_fitness

        IF weighted_direction > 0.1:
            action = "BUY"
        ELSE IF weighted_direction < -0.1:
            action = "SELL"
        ELSE:
            action = "HOLD"
        END IF

        confidence = abs(weighted_direction)

        # Lot multiplier from memory cell boost
        best_fitness = max(v.fitness for v in votes)
        lot_mult = 1.0 + 0.3 * max(0, best_fitness - SURVIVAL_THRESHOLD)
    ELSE:
        action = "HOLD"
        confidence = 0.0
        lot_mult = 1.0
    END IF

    # ============================================
    # PHASE 9: GENERATION ADVANCEMENT
    # ============================================
    generation += 1

    RETURN (action, confidence, lot_mult,
            f"VDJ_GEN{generation} active={len(active_antibodies)} "
            f"memory={count_memory_cells()}")

END ALGORITHM
```

---

## 10. INTEGRATION WITH EXISTING TEQA SYSTEM

### 10.1 Architecture Diagram

```
                    +---------------------------+
                    |     MT5 OHLCV DATA        |
                    +---------------------------+
                                |
                                v
                    +---------------------------+
                    |   TEActivationEngine      |
                    |   (33 TE families)        |
                    +---------------------------+
                         |              |
              +----------+              +-----------+
              |                                     |
              v                                     v
    +--------------------+            +----------------------------+
    | TEQAv3Engine       |            | VDJ_RECOMBINATION          |
    | (main pipeline)    |            | (strategy generation)      |
    |                    |            |                            |
    | Steps 1-9:        |            | Phases 0-9:               |
    | TE Activation     |            | Memory recall             |
    | Genomic Shock     |            | Bone marrow generation    |
    | Neural Mosaic     |            | Thymic selection          |
    | Split Quantum     |<---------->| Walk-forward testing      |
    | Consensus Vote    |  feedback  | Clonal selection          |
    | Domestication     |  loop      | Affinity maturation       |
    | Gate Checks       |            | Memory B cell promotion   |
    +--------------------+            | Consensus signal          |
              |                       +----------------------------+
              |                                     |
              v                                     v
    +--------------------+            +----------------------------+
    | te_quantum_signal  |            | vdj_memory_cells.db        |
    | .json              |            | (strategy memory)          |
    +--------------------+            +----------------------------+
              |                                     |
              +------------------+------------------+
                                 |
                                 v
                    +---------------------------+
                    |   TEDomesticationTracker   |
                    |   teqa_domestication.db    |
                    +---------------------------+
                                 |
                                 v
                    +---------------------------+
                    |   TEQA Bridge / BRAIN     |
                    |   Trading Execution       |
                    +---------------------------+
```

### 10.2 VDJ Engine Class Interface

```python
class VDJRecombinationEngine:
    """
    V(D)J Recombination Engine for adaptive strategy generation.

    Integrates with the existing TEQAv3Engine by:
        1. Consuming te_activations from TEActivationEngine
        2. Using shock_level from GenomicShockDetector
        3. Writing to TEDomesticationTracker via promote_to_memory_cell()
        4. Reading from TEDomesticationTracker via get_boost()

    The VDJ engine runs ALONGSIDE the main TEQA pipeline, not replacing it.
    TEQA provides the real-time directional signal; VDJ provides the
    strategy selection (which V+D+J combo to use for the current trade).
    """

    def __init__(
        self,
        teqa_engine: TEQAv3Engine,
        memory_db_path: str = None,
        max_active: int = MAX_ACTIVE_ANTIBODIES,
        generation_size: int = GENERATION_SIZE,
        eval_window: int = EVALUATION_WINDOW_BARS,
    ):
        self.teqa = teqa_engine
        self.quantum = TEQAQuantumEngine(shots=4096)
        self.domestication = teqa_engine.domestication
        self.memory_db = memory_db_path or str(
            Path(__file__).parent / "vdj_memory_cells.db"
        )

        self.max_active = max_active
        self.generation_size = generation_size
        self.eval_window = eval_window

        self.active_antibodies: List[dict] = []
        self.bone_marrow_pool: List[dict] = []
        self.generation = 0

        self._init_memory_db()

    def run_cycle(
        self,
        bars: np.ndarray,
        symbol: str,
        te_activations: List[Dict],
        shock_level: float,
        shock_label: str,
        drawdown: float = 0.0,
    ) -> dict:
        """
        Execute one full VDJ recombination cycle.

        Called from the TEQA live runner alongside TEQAv3Engine.analyze().

        Returns dict with:
            - action: BUY/SELL/HOLD
            - confidence: 0.0-1.0
            - lot_mult: position size multiplier
            - strategy_id: which antibody was selected
            - generation: current generation
            - population_stats: active/memory/bone_marrow counts
        """
        # Implementation follows the 9-phase algorithm above
        ...
```

### 10.3 Integration Point in teqa_live.py

```python
# In run_once() after engine.analyze():

# Create VDJ engine (once, at startup)
vdj_engine = VDJRecombinationEngine(
    teqa_engine=engine,
    memory_db_path=str(script_dir / "vdj_memory_cells.db"),
)

# After TEQA analysis:
result = engine.analyze(bars=bars, symbol=symbol, ...)

# Run VDJ cycle using TEQA's activations
vdj_result = vdj_engine.run_cycle(
    bars=bars,
    symbol=symbol,
    te_activations=result["te_activations"],
    shock_level=result["shock_score"],
    shock_label=result["shock_label"],
    drawdown=drawdown,
)

# Merge VDJ strategy selection with TEQA signal
# TEQA provides: direction, confidence
# VDJ provides: strategy (which V+D+J), lot multiplier, memory cell boost
if vdj_result["action"] != "HOLD":
    signal_json["vdj"] = {
        "strategy_id":   vdj_result["strategy_id"],
        "v_segment":     vdj_result["v_segment"],
        "d_segment":     vdj_result["d_segment"],
        "j_segment":     vdj_result["j_segment"],
        "source":        vdj_result["source"],  # "MEMORY_CELL" or "ACTIVE"
        "fitness":       vdj_result["fitness"],
        "generation":    vdj_result["generation"],
        "lot_mult":      vdj_result["lot_mult"],
        "population":    vdj_result["population_stats"],
    }
```

---

## 11. MATHEMATICAL SUMMARY

### 11.1 Recombination Combinatorics

```
Total V segments:         |V| = 33
Total D segments:         |D| = 13
Total J segments:         |J| = 10

Raw combinations:         |V| x |D| x |J| = 4,290

12/23 rule valid combos:  C_valid = sum over all (v,d,j):
                            1 if rss_compatible(v_rss, d_left, d_right, j_rss)
                            0 otherwise
                          Approx: C_valid ~ 2,400

Junctional diversity:     For each valid combo, parameter perturbation creates
                          approximately K unique variants where:
                          K = product over all continuous params of:
                            ceil(param_range / (base_sigma * param_value))
                          Approx: K ~ 100 per combo

Total unique antibodies:  C_valid * K ~ 240,000
```

### 11.2 Fitness Function

```
F(A) = 0.25 * P_WR(A) + 0.20 * PF_n(A) + 0.20 * S_n(A)
     + 0.15 * C(A)     - 0.10 * DD(A)   - 0.10 * TC(A)

Where:
    P_WR(A) = (alpha + w) / (alpha + beta + n)          Bayesian posterior WR
    PF_n(A) = min(1, avg_win / avg_loss / 3)            Normalized profit factor
    S_n(A)  = min(1, max(0, mean_ret / dd_std / 3))     Normalized Sortino
    C(A)    = max(0, 1 - std(rets) / |mean(rets)|)      Return consistency
    DD(A)   = min(1, max_dd / (equity * 0.10))           Drawdown penalty
    TC(A)   = max(0, (20 - n) / 20)                      Trade count penalty

    alpha = 10, beta = 10 (Beta prior)
```

### 11.3 Affinity Maturation Rate

```
effective_rate(g) = mu_0 / (1 + lambda * g)

Where:
    mu_0   = 0.05  (base mutation rate, 5%)
    lambda = 0.1   (convergence coefficient)
    g      = generation number

At generation 0:  rate = 5.0%
At generation 10: rate = 2.5%
At generation 20: rate = 1.7%
At generation 50: rate = 0.8%
```

### 11.4 Population Dynamics

```
dN/dt = bone_marrow_rate + proliferation_rate - apoptosis_rate

Where:
    bone_marrow_rate     = GENERATION_SIZE / cycle_time
    proliferation_rate   = |{A : F(A) > 0.55}| * n_mutants
    apoptosis_rate       = |{A : F(A) < 0.25}|

Steady state:
    N_ss ~ MAX_ACTIVE_ANTIBODIES = 50

Memory cell accumulation:
    dM/dt = |{A : F(A) > 0.70 AND maturation >= 3}| / cycle_time
    M is bounded by expiry: cells expire after 30 days of non-activation
```

### 11.5 Quantum Amplitude Encoding

```
For V segment selection with 6 qubits:

    |psi_V> = sum_{i=0}^{63} alpha_i |i>

    Where alpha_i = sqrt(s_i) / sqrt(sum_j s_j^2)
    And s_i = TE_activation_strength[i] if i < 33, else 0

    Implemented via RY rotations:
    For qubit k in {0,...,5}:
        theta_k = pi * mean({s_i : bit k of i is 1, i < 33})
        RY(theta_k) |0> on qubit k

The entanglement layer (CNOT gates) then creates correlations
between V, D, and J selections that encode the 12/23 rule as
quantum interference: incompatible states destructively interfere,
reducing their measurement probability.
```

---

## 12. PERFORMANCE CONSIDERATIONS

### 12.1 Computational Budget

```
VDJ quantum circuit (16 qubits):     ~50ms per execution
Walk-forward simulation per antibody: ~200ms on 500 bars
Affinity maturation (5 mutants):      ~1000ms per winner

Per cycle:
    Generation:     16-qubit circuit * 1 execution     =    50ms
    Evaluation:     50 antibodies * 200ms               = 10,000ms
    Maturation:     ~10 winners * 1000ms                = 10,000ms
    Total:                                               ~ 20 seconds

This runs ALONGSIDE the TEQA quantum circuit (~10-15 seconds),
so total cycle time is ~20 seconds (VDJ runs in parallel during
TEQA's neural mosaic computation).
```

### 12.2 Memory Management

```
Active antibodies:    50 * ~2KB each  =  ~100KB
Bone marrow pool:     20 * ~2KB each  =   ~40KB
Memory cells (DB):    Grows at ~1-2 cells/day, ~4KB each
                      After 1 year: ~500 cells * 4KB = ~2MB
                      With 30-day expiry: ~60 active cells at any time
```

---

## 13. FILE DELIVERABLES

```
QuantumTradingLibrary/
    vdj_recombination.py              -- Main VDJ engine (this spec implemented)
    vdj_segments.py                   -- V, D, J segment pool definitions
    vdj_quantum_circuit.py            -- 16-qubit VDJ quantum circuit
    vdj_fitness.py                    -- Clonal selection fitness function
    vdj_memory_cells.db               -- SQLite memory B cell database
    docs/plans/
        2026-02-08-algorithm-vdj-recombination.md  -- This document
```

---

## 14. THE RAG1/RAG2 CONNECTION

This entire system is inspired by the fact that **RAG1 is already in our
TE family list as Transib (qubit 24)** and **Rag_like (qubit 23)**. The
RAG recombinase IS a domesticated transposon. So the VDJ system is
literally what happens when Transib/RAG gets domesticated by the trading
system -- it goes from being a "selfish genetic element" (a simple
autocorrelation/diversity detector) to being the ENGINE that generates
the adaptive immune repertoire of trading strategies.

The biological parallel is exact:
- Transib was a selfish DNA transposon that jumped around genomes
- ~500 million years ago, it was "domesticated" into RAG1/RAG2
- Instead of jumping randomly, it now performs precise V(D)J recombination
- This gave vertebrates the adaptive immune system

In TEQA:
- Transib (qubit 24) is currently a simple autocorrelation signal
- Rag_like (qubit 23) is a diversity index detector
- Through VDJ_RECOMBINATION, these TEs become the ENGINE that generates,
  tests, selects, and remembers trading strategies
- This gives the trading system an adaptive "immune" response to markets

The domestication is complete. The transposon has been harnessed.

---

```
END OF SPECIFICATION

ALGORITHM_VDJ_RECOMBINATION v1.0
Total V segments: 33
Total D segments: 13
Total J segments: 10
Raw combinations: 4,290
Valid (12/23 rule): ~2,400
With junctional diversity: ~240,000+
Fitness dimensions: 6 (posterior WR, PF, Sortino, consistency, DD, trade count)
Quantum circuit: 16 qubits
Integration: TEQAv3Engine + TEDomesticationTracker + TEQABridge
```
