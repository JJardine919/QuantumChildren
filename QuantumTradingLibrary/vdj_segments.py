"""
VDJ Segment Pool Definitions
==============================
Defines all V (entry), D (context/regime), and J (exit) segments for the
VDJ Recombination adaptive strategy generation engine.

Maps the 33 TE families from teqa_v3_neural_te.py into entry signal
generators, regime modifiers, and exit strategies.

Includes the 12/23 RSS compatibility rule that constrains valid V+D+J
combinations (analogous to vertebrate V(D)J recombination).

Authors: DooDoo + Claude
Date:    2026-02-09
Parent:  ALGORITHM_VDJ_RECOMBINATION v1.0
"""

# ============================================================
# V SEGMENTS: Entry Signal Generators (33 total, one per TE)
# ============================================================

V_SEGMENTS = {
    # ---- MOMENTUM FAMILY (7 segments) ----
    "V_momentum_fast": {
        "te_source": "BEL_Pao",
        "signal": "momentum",
        "lookback": [5, 8, 13],
        "entry_rule": "momentum > threshold",
        "direction": "trend-following",
    },
    "V_momentum_slow": {
        "te_source": "Penelope",
        "signal": "trend_duration",
        "lookback": [21, 34, 55],
        "entry_rule": "trend_bars > threshold",
        "direction": "trend-following",
    },
    "V_momentum_macd": {
        "te_source": "Ty3_gypsy",
        "signal": "macd",
        "lookback": [12, 26],
        "entry_rule": "macd_cross > 0",
        "direction": "trend-following",
    },
    "V_momentum_ema": {
        "te_source": "CACTA",
        "signal": "ema_crossover",
        "lookback": [8, 21],
        "entry_rule": "ema_fast > ema_slow",
        "direction": "trend-following",
    },
    "V_momentum_strength": {
        "te_source": "DIRS1",
        "signal": "trend_strength",
        "lookback": [8, 21],
        "entry_rule": "trend_strength > threshold",
        "direction": "trend-following",
    },
    "V_momentum_orderflow": {
        "te_source": "Mavericks_Polinton",
        "signal": "order_flow",
        "lookback": [10],
        "entry_rule": "imbalance > threshold",
        "direction": "trend-following",
    },
    "V_momentum_price": {
        "te_source": "LINE",
        "signal": "price_change",
        "lookback": [1, 2, 3],
        "entry_rule": "abs(return) > threshold",
        "direction": "trend-following",
    },

    # ---- STRUCTURE FAMILY (8 segments) ----
    "V_structure_rsi": {
        "te_source": "Ty1_copia",
        "signal": "rsi",
        "lookback": [14],
        "entry_rule": "rsi < 30 OR rsi > 70",
        "direction": "mean-reverting",
    },
    "V_structure_bollinger": {
        "te_source": "Ty5",
        "signal": "bollinger_position",
        "lookback": [20],
        "entry_rule": "bb_pos < -1 OR bb_pos > 1",
        "direction": "mean-reverting",
    },
    "V_structure_meanrev": {
        "te_source": "RTE",
        "signal": "mean_reversion",
        "lookback": [20],
        "entry_rule": "deviation > threshold",
        "direction": "mean-reverting",
    },
    "V_structure_sr": {
        "te_source": "I_element",
        "signal": "support_resistance",
        "lookback": [20],
        "entry_rule": "near_level AND bounce",
        "direction": "mean-reverting",
    },
    "V_structure_fractal": {
        "te_source": "Mariner_Tc1",
        "signal": "fractal_dim",
        "lookback": [30],
        "entry_rule": "fd_shift detected",
        "direction": "adaptive",
    },
    "V_structure_autocorr": {
        "te_source": "Transib",
        "signal": "autocorrelation",
        "lookback": [30],
        "entry_rule": "autocorr > threshold",
        "direction": "adaptive",
    },
    "V_structure_micro": {
        "te_source": "PIF_Harbinger",
        "signal": "microstructure",
        "lookback": [20],
        "entry_rule": "microstructure > threshold",
        "direction": "adaptive",
    },
    "V_structure_pattern": {
        "te_source": "L1_Neuronal",
        "signal": "pattern_repetition",
        "lookback": [5, 10, 45],
        "entry_rule": "pattern_corr > threshold",
        "direction": "adaptive",
    },

    # ---- VOLATILITY FAMILY (7 segments) ----
    "V_vol_short": {
        "te_source": "Alu",
        "signal": "short_volatility",
        "lookback": [5, 20],
        "entry_rule": "vol_ratio > threshold",
        "direction": "neutral",
    },
    "V_vol_atr": {
        "te_source": "VIPER_Ngaro",
        "signal": "atr_ratio",
        "lookback": [14],
        "entry_rule": "atr_expanding",
        "direction": "neutral",
    },
    "V_vol_compression": {
        "te_source": "Crypton",
        "signal": "compression_ratio",
        "lookback": [50],
        "entry_rule": "compression > threshold",
        "direction": "neutral",
    },
    "V_vol_spread": {
        "te_source": "P_element",
        "signal": "spread_analysis",
        "lookback": [10],
        "entry_rule": "spread_expansion > threshold",
        "direction": "neutral",
    },
    "V_vol_breakout": {
        "te_source": "SVA_Regulatory",
        "signal": "compression_breakout",
        "lookback": [10, 30],
        "entry_rule": "breakout_from_compression",
        "direction": "adaptive",
    },
    "V_vol_mutation": {
        "te_source": "Mutator",
        "signal": "mutation_rate",
        "lookback": [20],
        "entry_rule": "sign_change_rate > threshold",
        "direction": "neutral",
    },
    "V_vol_noise": {
        "te_source": "Alu_Exonization",
        "signal": "noise_pattern",
        "lookback": [20],
        "entry_rule": "hidden_autocorr detected",
        "direction": "adaptive",
    },

    # ---- FLOW FAMILY (6 segments) ----
    "V_flow_volume": {
        "te_source": "SINE",
        "signal": "tick_volume",
        "lookback": [20],
        "entry_rule": "volume_spike detected",
        "direction": "neutral",
    },
    "V_flow_profile": {
        "te_source": "Helitron",
        "signal": "volume_profile",
        "lookback": [20],
        "entry_rule": "profile_skew > threshold",
        "direction": "adaptive",
    },
    "V_flow_candle": {
        "te_source": "hobo",
        "signal": "candle_pattern",
        "lookback": [1],
        "entry_rule": "body_ratio > threshold",
        "direction": "adaptive",
    },
    "V_flow_gap": {
        "te_source": "piggyBac",
        "signal": "gap_analysis",
        "lookback": [1],
        "entry_rule": "gap_size > threshold",
        "direction": "adaptive",
    },
    "V_flow_session": {
        "te_source": "pogo",
        "signal": "session_overlap",
        "lookback": [0],
        "entry_rule": "in_session_overlap",
        "direction": "neutral",
    },
    "V_flow_diversity": {
        "te_source": "Rag_like",
        "signal": "diversity_index",
        "lookback": [30],
        "entry_rule": "diversity > threshold",
        "direction": "neutral",
    },

    # ---- NEURAL/CROSS FAMILY (5 segments) ----
    "V_neural_mosaic": {
        "te_source": "L1_Somatic",
        "signal": "multi_tf_variance",
        "lookback": [5, 10, 20, 50],
        "entry_rule": "tf_agreement OR tf_divergence",
        "direction": "adaptive",
    },
    "V_neural_synapse": {
        "te_source": "HERV_Synapse",
        "signal": "cross_correlation",
        "lookback": [50],
        "entry_rule": "corr_shift detected",
        "direction": "adaptive",
    },
    "V_neural_arc": {
        "te_source": "Arc_Capsid",
        "signal": "successful_pattern_echo",
        "lookback": [0],
        "entry_rule": "echo_from_winner",
        "direction": "adaptive",
    },
    "V_neural_snr": {
        "te_source": "piwiRNA_Neural",
        "signal": "signal_noise_ratio",
        "lookback": [30],
        "entry_rule": "snr > threshold",
        "direction": "neutral",
    },
    "V_neural_trim28": {
        "te_source": "TRIM28_Silencer",
        "signal": "drawdown",
        "lookback": [0],
        "entry_rule": "drawdown < max_threshold",
        "direction": "risk-off",
    },
}


# ============================================================
# D SEGMENTS: Regime/Context Modifiers (13 total)
# ============================================================

D_SEGMENTS = {
    # ---- PRIMARY REGIMES (4) ----
    "D_trending_up": {
        "regime": "TRENDING_BULLISH",
        "detection": "EMA(8) > EMA(21) > EMA(50) AND ADX > 25",
        "v_modifier": "amplify trend-following V, suppress mean-reverting V",
        "param_shift": {"momentum_mult": 1.3, "reversion_mult": 0.5},
        "exit_bias": "trail",
    },
    "D_trending_down": {
        "regime": "TRENDING_BEARISH",
        "detection": "EMA(8) < EMA(21) < EMA(50) AND ADX > 25",
        "v_modifier": "amplify trend-following V (short side), suppress mean-reverting V",
        "param_shift": {"momentum_mult": 1.3, "reversion_mult": 0.5},
        "exit_bias": "trail",
    },
    "D_ranging": {
        "regime": "RANGE_BOUND",
        "detection": "ADX < 20 AND BB_width < percentile(20)",
        "v_modifier": "amplify mean-reverting V, suppress trend-following V",
        "param_shift": {"momentum_mult": 0.5, "reversion_mult": 1.5},
        "exit_bias": "fixed_target",
    },
    "D_volatile": {
        "regime": "HIGH_VOLATILITY",
        "detection": "ATR(14) > 2 * ATR(50) OR shock_label in [SHOCK, EXTREME]",
        "v_modifier": "amplify volatility V, widen all stops",
        "param_shift": {"vol_mult": 1.5, "sl_mult": 1.5, "lot_mult": 0.5},
        "exit_bias": "time_based",
    },

    # ---- TRANSITION REGIMES (4) ----
    "D_breakout_forming": {
        "regime": "BREAKOUT_FORMING",
        "detection": "compression_ratio > 2.0 AND vol_increasing",
        "v_modifier": "prepare for breakout, tighten entries",
        "param_shift": {"entry_threshold_mult": 1.3, "patience_mult": 2.0},
        "exit_bias": "trail",
    },
    "D_trend_exhaustion": {
        "regime": "TREND_EXHAUSTION",
        "detection": "RSI > 75 AND trend_duration > 20 AND volume_declining",
        "v_modifier": "suppress trend-following V, boost reversal V",
        "param_shift": {"momentum_mult": 0.3, "reversion_mult": 1.8},
        "exit_bias": "fixed_target",
    },
    "D_correlation_shift": {
        "regime": "CORRELATION_BREAKDOWN",
        "detection": "cross_corr_delta > 0.3 within 4 hours",
        "v_modifier": "activate neural/cross signals, isolate instrument",
        "param_shift": {"neural_mult": 1.5, "cross_trust": 0.3},
        "exit_bias": "time_based",
    },
    "D_session_transition": {
        "regime": "SESSION_TRANSITION",
        "detection": "within 30 min of major session open/close",
        "v_modifier": "boost flow/volume V segments, widen stops",
        "param_shift": {"flow_mult": 1.5, "sl_mult": 1.3},
        "exit_bias": "time_based",
    },

    # ---- SPECIAL REGIMES (5) ----
    "D_news_shock": {
        "regime": "NEWS_SHOCK",
        "detection": "volume > 5 * avg_volume AND ATR spike > 3x",
        "v_modifier": "suppress ALL entries for N bars, then activate vol V",
        "param_shift": {"cooldown_bars": 5, "vol_mult": 2.0},
        "exit_bias": "time_based",
    },
    "D_low_liquidity": {
        "regime": "LOW_LIQUIDITY",
        "detection": "spread > 2 * avg_spread OR volume < 0.3 * avg_volume",
        "v_modifier": "suppress all entries or require higher confidence",
        "param_shift": {"confidence_mult": 1.5, "lot_mult": 0.3},
        "exit_bias": "fixed_target",
    },
    "D_momentum_divergence": {
        "regime": "MOMENTUM_DIVERGENCE",
        "detection": "price making new highs BUT momentum indicator declining",
        "v_modifier": "activate reversal V segments, suppress momentum V",
        "param_shift": {"momentum_mult": 0.3, "reversion_mult": 1.5},
        "exit_bias": "fixed_target",
    },
    "D_accumulation": {
        "regime": "ACCUMULATION",
        "detection": "narrow range AND increasing volume AND decreasing volatility",
        "v_modifier": "patient entries, widen lookback",
        "param_shift": {"patience_mult": 2.0, "lookback_mult": 1.5},
        "exit_bias": "trail",
    },
    "D_distribution": {
        "regime": "DISTRIBUTION",
        "detection": "range expansion failing AND decreasing volume at highs/lows",
        "v_modifier": "activate reversal V segments",
        "param_shift": {"reversion_mult": 1.5, "momentum_mult": 0.5},
        "exit_bias": "fixed_target",
    },
}


# ============================================================
# J SEGMENTS: Exit Strategies (10 total)
# ============================================================

J_SEGMENTS = {
    # ---- TRAILING FAMILY ----
    "J_trail_atr": {
        "exit_type": "TRAILING_STOP",
        "method": "ATR-based trailing stop",
        "params": {
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "activation_rr": 1.0,
        },
        "tp_method": "none",
        "time_limit_bars": 0,
    },
    "J_trail_chandelier": {
        "exit_type": "TRAILING_STOP",
        "method": "Chandelier exit (highest high - ATR*mult)",
        "params": {
            "lookback": 22,
            "atr_multiplier": 3.0,
            "activation_rr": 0.5,
        },
        "tp_method": "none",
        "time_limit_bars": 0,
    },
    "J_trail_parabolic": {
        "exit_type": "TRAILING_STOP",
        "method": "Parabolic SAR trailing",
        "params": {
            "af_start": 0.02,
            "af_increment": 0.02,
            "af_max": 0.20,
        },
        "tp_method": "none",
        "time_limit_bars": 0,
    },

    # ---- FIXED TARGET FAMILY ----
    "J_fixed_rr2": {
        "exit_type": "FIXED_TARGET",
        "method": "Fixed 2:1 reward-to-risk",
        "params": {
            "rr_ratio": 2.0,
        },
        "tp_method": "SL_distance * rr_ratio",
        "time_limit_bars": 200,
    },
    "J_fixed_rr3": {
        "exit_type": "FIXED_TARGET",
        "method": "Fixed 3:1 reward-to-risk",
        "params": {
            "rr_ratio": 3.0,
        },
        "tp_method": "SL_distance * rr_ratio",
        "time_limit_bars": 300,
    },

    # ---- DYNAMIC FAMILY ----
    "J_dynamic_partial": {
        "exit_type": "DYNAMIC",
        "method": "Partial close at TP1, trail remainder",
        "params": {
            "tp1_rr": 1.5,
            "tp1_close_pct": 50,
            "trail_remainder": True,
            "trail_atr_mult": 1.5,
        },
        "tp_method": "partial + trail",
        "time_limit_bars": 0,
    },
    "J_dynamic_structure": {
        "exit_type": "DYNAMIC",
        "method": "Exit at next S/R level",
        "params": {
            "sr_lookback": 100,
            "min_rr": 1.0,
        },
        "tp_method": "next_sr_level",
        "time_limit_bars": 500,
    },
    "J_dynamic_reversal": {
        "exit_type": "DYNAMIC",
        "method": "Exit on reversal signal from any V segment",
        "params": {
            "reversal_confidence": 0.6,
            "min_rr": 0.5,
        },
        "tp_method": "reversal_detection",
        "time_limit_bars": 0,
    },

    # ---- TIME-BASED FAMILY ----
    "J_time_fixed": {
        "exit_type": "TIME_BASED",
        "method": "Close after N bars regardless of P/L",
        "params": {
            "max_bars": 60,
            "move_sl_be": True,
        },
        "tp_method": "none",
        "time_limit_bars": 60,
    },
    "J_time_session": {
        "exit_type": "TIME_BASED",
        "method": "Close before session end",
        "params": {
            "close_before_minutes": 30,
            "session_end_hour": 21,
        },
        "tp_method": "none",
        "time_limit_bars": 0,
    },
}


# ============================================================
# 12/23 RSS COMPATIBILITY RULE
# ============================================================

V_RSS = {
    # Momentum V segments: 23-spacer
    "V_momentum_fast": 23,
    "V_momentum_slow": 23,
    "V_momentum_macd": 23,
    "V_momentum_ema": 23,
    "V_momentum_strength": 23,
    "V_momentum_orderflow": 23,
    "V_momentum_price": 23,
    # Structure V segments: 12-spacer
    "V_structure_rsi": 12,
    "V_structure_bollinger": 12,
    "V_structure_meanrev": 12,
    "V_structure_sr": 12,
    "V_structure_fractal": 12,
    "V_structure_autocorr": 12,
    "V_structure_micro": 12,
    "V_structure_pattern": 12,
    # Volatility V segments: 23-spacer
    "V_vol_short": 23,
    "V_vol_atr": 23,
    "V_vol_compression": 23,
    "V_vol_spread": 23,
    "V_vol_breakout": 23,
    "V_vol_mutation": 23,
    "V_vol_noise": 23,
    # Flow V segments: 12-spacer
    "V_flow_volume": 12,
    "V_flow_profile": 12,
    "V_flow_candle": 12,
    "V_flow_gap": 12,
    "V_flow_session": 12,
    "V_flow_diversity": 12,
    # Neural V segments: 0 = universal adapter
    "V_neural_mosaic": 0,
    "V_neural_synapse": 0,
    "V_neural_arc": 0,
    "V_neural_snr": 0,
    "V_neural_trim28": 0,
}

D_RSS = {
    # (left_rss, right_rss) -- V connects to left, right connects to J
    "D_trending_up": (12, 23),
    "D_trending_down": (12, 23),
    "D_ranging": (23, 12),
    "D_volatile": (12, 12),
    "D_breakout_forming": (12, 23),
    "D_trend_exhaustion": (23, 12),
    "D_correlation_shift": (12, 23),
    "D_session_transition": (23, 12),
    "D_news_shock": (12, 12),
    "D_low_liquidity": (23, 23),
    "D_momentum_divergence": (23, 12),
    "D_accumulation": (12, 23),
    "D_distribution": (23, 12),
}

J_RSS = {
    # Trailing exits: 12-spacer
    "J_trail_atr": 12,
    "J_trail_chandelier": 12,
    "J_trail_parabolic": 12,
    # Fixed exits: 23-spacer
    "J_fixed_rr2": 23,
    "J_fixed_rr3": 23,
    # Dynamic exits: 0 = universal
    "J_dynamic_partial": 0,
    "J_dynamic_structure": 0,
    "J_dynamic_reversal": 0,
    # Time exits: 23-spacer
    "J_time_fixed": 23,
    "J_time_session": 23,
}


def rss_compatible(v_name: str, d_name: str, j_name: str) -> bool:
    """
    Check 12/23 rule compatibility for a V+D+J combination.

    V connects to D's left side, D's right side connects to J.
    12 pairs with 23 only. 0 (universal) pairs with anything.
    Same spacer = INCOMPATIBLE (just like biology).
    """
    v_rss = V_RSS.get(v_name, 0)
    d_left, d_right = D_RSS.get(d_name, (0, 0))
    j_rss = J_RSS.get(j_name, 0)

    # V-D junction: must be complementary (12<->23)
    if v_rss != 0 and d_left != 0:
        if v_rss == d_left:
            return False

    # D-J junction: must be complementary (12<->23)
    if d_right != 0 and j_rss != 0:
        if d_right == j_rss:
            return False

    return True


def count_valid_combinations() -> int:
    """Count total valid V+D+J combinations after 12/23 rule."""
    count = 0
    for v in V_SEGMENTS:
        for d in D_SEGMENTS:
            for j in J_SEGMENTS:
                if rss_compatible(v, d, j):
                    count += 1
    return count


# Pre-computed segment name lists for index lookups
V_NAMES = list(V_SEGMENTS.keys())
D_NAMES = list(D_SEGMENTS.keys())
J_NAMES = list(J_SEGMENTS.keys())

N_V = len(V_NAMES)  # 33
N_D = len(D_NAMES)  # 13
N_J = len(J_NAMES)  # 10
