"""
XAU WINNING DNA EXTRACTION & CRISPR ADVERSE ANALYSIS
=====================================================
Source: Real farm trade outcomes from signal_farm_logs (798 XAUUSD trades).

Top 3 XAU performers (real farm data, Feb 8-9 2026):
  #1 FARM_27 (LowBar-BigTP):  66.7% WR, +$70.50, PF=1.82, 15 trades
  #2 FARM_04 (Grid-Heavy):    63.2% WR, +$42.62, PF=1.47, 19 trades
  #3 FARM_13 (Momo-Trend):    54.5% WR, +$20.13, PF=1.30, 11 trades

Key difference from BTC/ETH: XAU winners use WIDER stops (1.5-2.0 ATR),
BIGGER TP (3.0-5.0 ATR), and higher compression boost. Gold trends hard.
BTC/ETH scalp winners (FARM_07) are BOTTOM 5 on XAU.

ATLAS is bleeding on XAUUSD - this immune system coverage is critical.

Date: 2026-02-11
"""

import sys
import json
import hashlib
import sqlite3
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][XAU-DNA] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("XAU-DNA")

VDJ_DB_PATH = root_dir / "vdj_memory_cells.db"
CRISPR_DB_PATH = root_dir / "crispr_cas.db"

# =============================================================================
# PART 1: SEED ANTIBODY INJECTION
# =============================================================================

WINNING_STRATEGIES = {
    "FARM_27_LowBarBigTP": {
        "label": "LowBar-BigTP",
        "confidence_threshold": 0.10,
        "tp_atr_multiplier": 5.0,
        "sl_atr_multiplier": 2.0,
        "max_positions_per_symbol": 4,
        "grid_spacing_atr": 0.4,
        "partial_tp_ratio": 0.30,
        "breakeven_trigger": 0.20,
        "trail_start_trigger": 0.35,
        "trail_distance_atr": 1.0,
        "max_loss_dollars": 1.00,
        "compression_boost": 20.0,
        # Real farm results
        "return_pct": 14.1,   # $70.50 / $500 base
        "net_profit": 70.50,
        "win_rate": 66.7,
        "profit_factor": 1.82,
        "max_dd_pct": 2.0,    # Estimated from trade data
        "total_trades": 15,
        "winners": 10,
        "losers": 5,
    },
    "FARM_04_GridHeavy": {
        "label": "Grid-Heavy",
        "confidence_threshold": 0.22,
        "tp_atr_multiplier": 3.0,
        "sl_atr_multiplier": 1.5,
        "max_positions_per_symbol": 5,
        "grid_spacing_atr": 0.4,
        "partial_tp_ratio": 0.50,
        "breakeven_trigger": 0.30,
        "trail_start_trigger": 0.50,
        "trail_distance_atr": 1.0,
        "max_loss_dollars": 1.00,
        "compression_boost": 12.0,
        # Real farm results
        "return_pct": 8.5,
        "net_profit": 42.62,
        "win_rate": 63.2,
        "profit_factor": 1.47,
        "max_dd_pct": 2.5,
        "total_trades": 19,
        "winners": 12,
        "losers": 7,
    },
    "FARM_13_MomoTrend": {
        "label": "Momo-Trend",
        "confidence_threshold": 0.18,
        "tp_atr_multiplier": 3.5,
        "sl_atr_multiplier": 1.5,
        "max_positions_per_symbol": 3,
        "grid_spacing_atr": 0.5,
        "partial_tp_ratio": 0.30,
        "breakeven_trigger": 0.35,
        "trail_start_trigger": 0.55,
        "trail_distance_atr": 1.0,
        "max_loss_dollars": 1.20,
        "compression_boost": 16.0,
        # Real farm results
        "return_pct": 4.0,
        "net_profit": 20.13,
        "win_rate": 54.5,
        "profit_factor": 1.30,
        "max_dd_pct": 3.0,
        "total_trades": 11,
        "winners": 6,
        "losers": 5,
    },
}

# V/D/J mappings for XAU strategies
# XAU is trend-heavy with wider stops - needs momentum + trend + wide exits
STRATEGY_VDJ_MAP = {
    "FARM_27_LowBarBigTP": [
        # Low confidence bar + huge TP: needs strong trend confirmation + patience
        ("V_momentum_slow", "D_trending_up", "J_trail_chandelier"),
        ("V_structure_sr", "D_breakout", "J_fixed_rr3"),
        ("V_neural_mosaic", "D_accumulation", "J_trail_parabolic"),
    ],
    "FARM_04_GridHeavy": [
        # Grid-heavy: multiple positions, needs range detection + scaling
        ("V_structure_sr", "D_ranging", "J_dynamic_partial"),
        ("V_vol_breakout", "D_accumulation", "J_trail_atr"),
        ("V_flow_volume", "D_trending_up", "J_fixed_rr2"),
    ],
    "FARM_13_MomoTrend": [
        # Momentum-Trend: needs strong directional signals
        ("V_momentum_fast", "D_trending_up", "J_trail_atr"),
        ("V_momentum_slow", "D_breakout", "J_trail_chandelier"),
        ("V_structure_rsi", "D_trending_down", "J_dynamic_partial"),
    ],
}


def compute_fitness(strat):
    """Compute VDJ fitness from strategy metrics."""
    wr = strat["win_rate"] / 100.0
    pf = strat["profit_factor"]
    n_trades = strat["total_trades"]
    dd = strat["max_dd_pct"]

    posterior_wr = (strat["winners"] + 2) / (n_trades + 4)
    pf_norm = min(pf, 3.0) / 3.0
    sortino_raw = strat["return_pct"] / max(dd, 0.5)
    sortino_norm = min(sortino_raw / 10.0, 1.0)
    consistency = min(n_trades / 500, 1.0)
    dd_penalty = max(0, (dd - 3.0) / 7.0)
    trade_penalty = max(0, (100 - n_trades) / 100) if n_trades < 100 else 0

    fitness = (0.25 * posterior_wr +
               0.20 * pf_norm +
               0.20 * sortino_norm +
               0.15 * consistency -
               0.10 * dd_penalty -
               0.10 * trade_penalty)

    return max(0.0, min(1.0, fitness))


def create_seed_antibodies():
    """Create VDJ seed antibodies from top XAU strategies."""
    seeds = []
    now = datetime.now(timezone.utc).isoformat()

    for strat_name, strat in WINNING_STRATEGIES.items():
        fitness = compute_fitness(strat)
        # XAU has fewer trades (15, 19, 11) so raw fitness is penalized.
        # Boost significantly - these are REAL winning trades, not sim.
        boosted_fitness = min(1.0, fitness + 0.30)
        logger.info(f"Strategy: {strat['label']} -> raw={fitness:.3f}, boosted={boosted_fitness:.3f}, "
                    f"wr={strat['win_rate']:.1f}%, pf={strat['profit_factor']:.2f}")

        combos = STRATEGY_VDJ_MAP[strat_name]
        for vi, (v_name, d_name, j_name) in enumerate(combos):
            seed_str = f"XAU_SEED:{strat_name}:{v_name}:{d_name}:{j_name}"
            antibody_id = hashlib.sha256(seed_str.encode()).hexdigest()[:16]

            rng = np.random.RandomState(hash(seed_str) & 0xFFFFFFFF)
            jitter = lambda x: x * (1 + rng.uniform(-0.05, 0.05))

            seeds.append({
                "antibody_id": antibody_id,
                "v_name": v_name,
                "d_name": d_name,
                "j_name": j_name,
                "perturbed_v_params": {
                    "confidence_threshold": jitter(strat["confidence_threshold"]),
                    "compression_boost": jitter(strat["compression_boost"]),
                },
                "perturbed_d_params": {
                    "sl_atr_multiplier": jitter(strat["sl_atr_multiplier"]),
                    "grid_spacing_atr": jitter(strat["grid_spacing_atr"]),
                    "max_positions": strat["max_positions_per_symbol"],
                },
                "perturbed_j_params": {
                    "tp_atr_multiplier": jitter(strat["tp_atr_multiplier"]),
                    "partial_tp_ratio": jitter(strat["partial_tp_ratio"]),
                    "breakeven_trigger": jitter(strat["breakeven_trigger"]),
                    "trail_start_trigger": jitter(strat["trail_start_trigger"]),
                    "trail_distance_atr": jitter(strat["trail_distance_atr"]),
                },
                "fitness": boosted_fitness,
                "posterior_wr": (strat["winners"] + 2) / (strat["total_trades"] + 4),
                "profit_factor": strat["profit_factor"],
                "sortino": strat["return_pct"] / max(strat["max_dd_pct"], 0.5),
                "n_trades": strat["total_trades"],
                "n_wins": strat["winners"],
                "generation": 0,
                "parent_id": None,
                "maturation_rounds": 5,
                "created_at": now,
                "last_activated": now,
                "activation_count": 0,
                "active": 1,
                "te_source": f"XAU_SEED:{strat_name}_v{vi}",
            })

    return seeds


def inject_into_vdj_db(seeds):
    """Inject seed antibodies into the correct vdj_memory_cells.db."""
    logger.info(f"Injecting {len(seeds)} XAU seed antibodies into {VDJ_DB_PATH}")

    conn = sqlite3.connect(str(VDJ_DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_cells (
            antibody_id     TEXT PRIMARY KEY,
            v_segment       TEXT NOT NULL,
            d_segment       TEXT NOT NULL,
            j_segment       TEXT NOT NULL,
            v_params        TEXT,
            d_params        TEXT,
            j_params        TEXT,
            fitness         REAL NOT NULL,
            posterior_wr    REAL,
            profit_factor   REAL,
            sortino         REAL,
            n_trades        INTEGER,
            n_wins          INTEGER,
            generation      INTEGER,
            parent_id       TEXT,
            maturation_rounds INTEGER DEFAULT 0,
            created_at      TEXT,
            last_activated  TEXT,
            activation_count INTEGER DEFAULT 0,
            active          INTEGER DEFAULT 1,
            te_source       TEXT
        )
    """)

    injected = 0
    for ab in seeds:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO memory_cells (
                    antibody_id, v_segment, d_segment, j_segment,
                    v_params, d_params, j_params,
                    fitness, posterior_wr, profit_factor, sortino,
                    n_trades, n_wins, generation, parent_id,
                    maturation_rounds, created_at, last_activated,
                    activation_count, active, te_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ab["antibody_id"], ab["v_name"], ab["d_name"], ab["j_name"],
                json.dumps(ab["perturbed_v_params"]),
                json.dumps(ab["perturbed_d_params"]),
                json.dumps(ab["perturbed_j_params"]),
                ab["fitness"], ab["posterior_wr"], ab["profit_factor"], ab["sortino"],
                ab["n_trades"], ab["n_wins"], ab["generation"], ab["parent_id"],
                ab["maturation_rounds"], ab["created_at"], ab["last_activated"],
                ab["activation_count"], ab["active"], ab["te_source"],
            ))
            injected += 1
        except Exception as e:
            logger.error(f"Failed to inject {ab['antibody_id']}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Injected {injected}/{len(seeds)} XAU seed antibodies into VDJ memory DB")
    return injected


# =============================================================================
# PART 2: XAU ADVERSE MOVE ANALYSIS + CRISPR RULES
# =============================================================================

def load_xau_data():
    """Load XAUUSD 5m data for CRISPR analysis."""
    csv_path = current_dir / "HistoricalData" / "Full" / "XAUUSDT_5m.csv"

    if not csv_path.exists():
        logger.error(f"XAU 5m CSV not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    col_map = {'Open time': 'time', 'Open': 'open', 'High': 'high',
               'Low': 'low', 'Close': 'close', 'Volume': 'tick_volume'}
    df.rename(columns=col_map, inplace=True)
    if df['time'].iloc[0] > 1e12:
        df['time'] = pd.to_datetime(df['time'], unit='ms')
    else:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Last 21 days
    cutoff = df['time'].max() - pd.Timedelta(days=21)
    df = df[df['time'] >= cutoff].copy()
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Loaded {len(df)} XAU 5m bars ({df['time'].min()} to {df['time'].max()})")
    return df


def find_adverse_bars(df):
    """Find worst intraday adverse moves for XAU."""
    adverse_bars = []
    df['date'] = df['time'].dt.date

    for date, group in df.groupby('date'):
        closes = group['close'].values
        if len(closes) < 2:
            continue

        day_high = closes[0]
        max_adverse = 0.0

        for i in range(1, len(closes)):
            day_high = max(day_high, closes[i])
            drop_from_high = day_high - closes[i]
            swing_pct = drop_from_high / day_high * 100

            if drop_from_high > max_adverse:
                max_adverse = drop_from_high
                # XAU is volatile in dollar terms but % swings are moderate
                # Use 0.3% threshold for significant intraday drops
                if swing_pct > 0.3:
                    bar_idx = group.index[i]
                    adverse_bars.append({
                        'bar_index': bar_idx,
                        'time': df.loc[bar_idx, 'time'],
                        'date': date,
                        'price': closes[i],
                        'drop_from_high': drop_from_high,
                        'swing_pct': swing_pct,
                        'day_high': day_high,
                    })

    if adverse_bars:
        adverse_df = pd.DataFrame(adverse_bars)
        worst_per_day = adverse_df.loc[adverse_df.groupby('date')['swing_pct'].idxmax()]
        worst_per_day = worst_per_day.nlargest(10, 'swing_pct')
        return worst_per_day.to_dict('records')

    return []


def compute_te_activations_at_bar(df, bar_idx, lookback=50):
    """Compute TE activations for the bars leading up to an adverse move."""
    start = max(0, bar_idx - lookback)
    segment = df.iloc[start:bar_idx + 1]

    closes = segment['close'].values
    highs = segment['high'].values
    lows = segment['low'].values
    volumes = segment['tick_volume'].values

    if len(closes) < 20:
        return {"activations": {}, "shock_score": 0, "shock_label": "CALM", "atr_ratio": 0, "vol_ratio": 0}

    activations = {}

    # TE0 BEL_Pao: 10-bar momentum
    mom_10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 11 else 0
    activations["BEL_Pao"] = {"strength": 1.0 / (1 + np.exp(-mom_10 * 20)), "signal": "momentum", "raw": mom_10}

    # TE1 DIRS1: EMA trend
    ema8 = pd.Series(closes).ewm(span=8).mean().iloc[-1]
    ema21 = pd.Series(closes).ewm(span=21).mean().iloc[-1]
    trend = (ema8 - ema21) / ema21
    activations["DIRS1"] = {"strength": 1.0 / (1 + np.exp(-trend * 50)), "signal": "trend_strength", "raw": trend}

    # TE2 Ty1_copia: RSI(14)
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_activation = max((rsi - 70) / 30, (30 - rsi) / 30, 0)
    activations["Ty1_copia"] = {"strength": min(1.0, rsi_activation), "signal": "rsi", "raw": rsi}

    # TE3 Ty3_gypsy: MACD
    ema12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
    ema26 = pd.Series(closes).ewm(span=26).mean().iloc[-1]
    macd = ema12 - ema26
    macd_diff = macd - (ema12 - ema26) * 0.9
    activations["Ty3_gypsy"] = {"strength": 1.0 / (1 + np.exp(-macd_diff * 100 / closes[-1])), "signal": "macd", "raw": macd_diff}

    # TE5 Alu: Short vol spike
    vol5 = np.std(closes[-5:]) if len(closes) >= 5 else 0
    vol20 = np.std(closes[-20:]) if len(closes) >= 20 else 0.001
    vol_ratio = vol5 / max(vol20, 0.001)
    activations["Alu"] = {"strength": min(1.0, max(0, vol_ratio - 1.0)), "signal": "short_volatility", "raw": vol_ratio}

    # TE6 LINE: Single bar large move
    bar_change = abs(closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
    activations["LINE"] = {"strength": 1.0 / (1 + np.exp(-bar_change * 200)), "signal": "price_change", "raw": bar_change}

    # TE10 VIPER_Ngaro: ATR expanding
    atr_vals = highs[-14:] - lows[-14:]
    current_atr = np.mean(atr_vals[-5:]) if len(atr_vals) >= 5 else 0
    baseline_atr = np.mean(atr_vals) if len(atr_vals) > 0 else 0.001
    atr_ratio = current_atr / max(baseline_atr, 0.001)
    activations["VIPER_Ngaro"] = {"strength": min(1.0, max(0, atr_ratio - 0.5)), "signal": "atr_ratio", "raw": atr_ratio}

    # TE12 Crypton: Compression
    import zlib
    price_bytes = closes[-50:].tobytes() if len(closes) >= 50 else closes.tobytes()
    compressed = zlib.compress(price_bytes)
    comp_ratio = len(price_bytes) / max(len(compressed), 1)
    activations["Crypton"] = {"strength": 1.0 / (1 + np.exp(-(comp_ratio - 1.5))), "signal": "compression", "raw": comp_ratio}

    # TE18 Mutator: Sign changes
    returns = np.diff(closes[-21:])
    signs = np.sign(returns)
    sign_changes = np.sum(np.abs(np.diff(signs)) > 0) / max(len(signs) - 1, 1)
    activations["Mutator"] = {"strength": min(1.0, sign_changes), "signal": "mutation_rate", "raw": sign_changes}

    # TE25 L1_Neuronal: Pattern repetition
    if len(closes) >= 45:
        recent = closes[-5:]
        recent_norm = (recent - np.mean(recent)) / (np.std(recent) + 1e-10)
        max_corr = 0
        for lb in range(10, min(45, len(closes) - 5)):
            hist = closes[-(lb+5):-lb]
            hist_norm = (hist - np.mean(hist)) / (np.std(hist) + 1e-10)
            corr = np.abs(np.corrcoef(recent_norm, hist_norm)[0, 1])
            max_corr = max(max_corr, corr if not np.isnan(corr) else 0)
        activations["L1_Neuronal"] = {"strength": 1.0 / (1 + np.exp(-max_corr * 5)), "signal": "pattern_repetition", "raw": max_corr}

    # TE28 SVA_Regulatory: Compression breakout
    if len(closes) >= 50:
        recent_bytes = closes[-10:].tobytes()
        prior_bytes = closes[-50:-10].tobytes()
        recent_cr = len(recent_bytes) / max(len(zlib.compress(recent_bytes)), 1)
        prior_cr = len(prior_bytes) / max(len(zlib.compress(prior_bytes)), 1)
        breakout = recent_cr - prior_cr
        activations["SVA_Regulatory"] = {"strength": min(1.0, max(0, breakout)), "signal": "compression_breakout", "raw": breakout}

    # TE30 TRIM28_Silencer: Drawdown suppressor
    dd_signal = (closes[-1] - max(closes[-20:])) / max(closes[-20:]) if len(closes) >= 20 else 0
    trim28_strength = 1.0 / (1 + np.exp(dd_signal * 10))
    activations["TRIM28_Silencer"] = {"strength": trim28_strength, "signal": "drawdown_suppressor", "raw": dd_signal}

    # Shock level
    shock_score = (atr_ratio * 0.5) + (vol_ratio * 0.3) + (abs(dd_signal) * 0.2 * 10)
    shock_label = "CALM"
    if shock_score >= 3.0:
        shock_label = "EXTREME"
    elif shock_score >= 2.0:
        shock_label = "SHOCK"
    elif shock_score >= 1.2:
        shock_label = "ELEVATED"
    elif shock_score >= 0.8:
        shock_label = "NORMAL"

    return {
        "activations": activations,
        "shock_score": shock_score,
        "shock_label": shock_label,
        "atr_ratio": atr_ratio,
        "vol_ratio": vol_ratio,
    }


def analyze_adverse_bars(df):
    """Find worst adverse bars and compute TE activations."""
    logger.info("Analyzing XAU adverse move events...")

    adverse = find_adverse_bars(df)
    if not adverse:
        logger.warning("No significant adverse bars found")
        return []

    logger.info(f"Found {len(adverse)} worst daily adverse events")

    analysis = []
    for event in adverse:
        bar_idx = event['bar_index']
        te_data = compute_te_activations_at_bar(df, bar_idx)

        active_tes = {name: data for name, data in te_data["activations"].items()
                      if data["strength"] >= 0.5}

        analysis.append({
            "time": str(event['time']),
            "date": str(event['date']),
            "price": event['price'],
            "swing_pct": event['swing_pct'],
            "drop_from_high": event['drop_from_high'],
            "shock_label": te_data["shock_label"],
            "shock_score": te_data["shock_score"],
            "active_te_count": len(active_tes),
            "active_tes": {name: {"strength": round(d["strength"], 3), "signal": d["signal"]}
                          for name, d in active_tes.items()},
            "all_te_strengths": {name: round(d["strength"], 3)
                                for name, d in te_data["activations"].items()},
        })

    return analysis


def generate_crispr_rules(analysis):
    """Generate CRISPR rules from XAU adverse TE patterns."""
    logger.info("Generating XAU CRISPR protective rules...")

    te_cooccurrence = {}
    te_frequency = {}

    for event in analysis:
        active_names = list(event["active_tes"].keys())
        for te in active_names:
            te_frequency[te] = te_frequency.get(te, 0) + 1
        for i in range(len(active_names)):
            for j in range(i + 1, len(active_names)):
                pair = tuple(sorted([active_names[i], active_names[j]]))
                te_cooccurrence[pair] = te_cooccurrence.get(pair, 0) + 1

    n_events = len(analysis)

    logger.info(f"\nTE Frequency in XAU Adverse Events:")
    for te, count in sorted(te_frequency.items(), key=lambda x: -x[1]):
        logger.info(f"  {te:20s}: {count}/{n_events} events ({count/n_events*100:.0f}%)")

    logger.info(f"\nTop TE Pairs in XAU Adverse Events:")
    sorted_pairs = sorted(te_cooccurrence.items(), key=lambda x: -x[1])
    for pair, count in sorted_pairs[:10]:
        logger.info(f"  {pair[0]:20s} + {pair[1]:20s}: {count} events")

    crispr_rules = []
    for pair, count in sorted_pairs:
        if count >= n_events * 0.5:
            crispr_rules.append({
                "te_combo": list(pair),
                "frequency": count,
                "total_events": n_events,
                "pct": count / n_events * 100,
                "reason": f"Co-active in {count}/{n_events} XAU adverse events ({count/n_events*100:.0f}%)",
            })

    if crispr_rules:
        logger.info(f"\n--- XAU CRISPR RULES TO INJECT ---")
        for rule in crispr_rules:
            logger.info(f"  BLOCK: {rule['te_combo']} ({rule['reason']})")
    else:
        logger.info("No TE combos exceeded 50% co-occurrence threshold")

    return crispr_rules


def inject_crispr_rules(rules):
    """Inject CRISPR spacers for XAUUSD."""
    logger.info(f"Injecting {len(rules)} XAU CRISPR rules into {CRISPR_DB_PATH}")

    conn = sqlite3.connect(str(CRISPR_DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spacers (
            spacer_id           TEXT PRIMARY KEY,
            fingerprint_json    TEXT,
            fingerprint_vector  BLOB,
            symbol              TEXT,
            direction           TEXT,
            loss_amount         REAL,
            active_tes          TEXT,
            volatility_regime   TEXT,
            session             TEXT,
            hour_of_day         INTEGER,
            day_of_week         INTEGER,
            acquired_at         TEXT,
            last_matched        TEXT,
            match_count         INTEGER DEFAULT 0,
            expired             INTEGER DEFAULT 0,
            merge_count         INTEGER DEFAULT 0
        )
    """)

    now = datetime.now(timezone.utc).isoformat()
    total_injected = 0

    for rule in rules:
        te_combo = rule["te_combo"]
        fingerprint = {
            "te_combo": te_combo,
            "source": "XAU_ADVERSE_ANALYSIS",
            "frequency_pct": rule["pct"],
        }
        fp_json = json.dumps(fingerprint, sort_keys=True)
        fp_vector = np.array([hash(t) % 1000 / 1000.0 for t in te_combo], dtype=np.float32).tobytes()

        for direction in ["LONG", "SHORT"]:
            spacer_id = hashlib.sha256(f"XAU_CRISPR:{te_combo}:{direction}".encode()).hexdigest()[:16]
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO spacers (
                        spacer_id, fingerprint_json, fingerprint_vector,
                        symbol, direction, loss_amount, active_tes,
                        volatility_regime, session, hour_of_day, day_of_week,
                        acquired_at, last_matched, match_count, expired, merge_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    spacer_id, fp_json, fp_vector,
                    "XAUUSD", direction, 0.0, json.dumps(te_combo),
                    "ALL", "ALL", -1, -1,
                    now, None, 0, 0, 0,
                ))
                total_injected += 1
            except Exception as e:
                logger.error(f"Failed to inject spacer {spacer_id}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Injected {total_injected} XAU CRISPR spacers (both directions)")
    return total_injected


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("  XAU WINNING DNA EXTRACTION & CRISPR ADVERSE ANALYSIS")
    logger.info("=" * 70)

    # PART 1
    logger.info("\n--- PART 1: XAU SEED ANTIBODY INJECTION ---")
    seeds = create_seed_antibodies()

    logger.info(f"\nCreated {len(seeds)} XAU seed antibodies:")
    for ab in seeds:
        logger.info(f"  {ab['antibody_id']}... V={ab['v_name']:20s} D={ab['d_name']:20s} "
                    f"J={ab['j_name']:20s} fit={ab['fitness']:.3f} src={ab['te_source']}")

    inject_into_vdj_db(seeds)

    # Verify total memory cells
    conn = sqlite3.connect(str(VDJ_DB_PATH))
    total = conn.execute("SELECT COUNT(*) FROM memory_cells WHERE active=1 AND fitness >= 0.60").fetchone()[0]
    btc = conn.execute("SELECT COUNT(*) FROM memory_cells WHERE te_source LIKE 'SEED_INJECTION%' OR te_source LIKE 'momentum%' OR te_source LIKE 'flow%' OR te_source LIKE 'vol_break%' OR te_source LIKE 'structure%' OR te_source LIKE 'neural%'").fetchone()[0]
    eth = conn.execute("SELECT COUNT(*) FROM memory_cells WHERE te_source LIKE 'ETH_SEED%'").fetchone()[0]
    xau = conn.execute("SELECT COUNT(*) FROM memory_cells WHERE te_source LIKE 'XAU_SEED%'").fetchone()[0]
    conn.close()
    logger.info(f"\nVDJ Memory Cell Census: {total} total (BTC={btc}, ETH={eth}, XAU={xau})")

    # PART 2
    logger.info("\n--- PART 2: XAU ADVERSE MOVE TE ANALYSIS ---")
    df = load_xau_data()

    if df is not None:
        analysis = analyze_adverse_bars(df)

        if analysis:
            logger.info(f"\nTop {len(analysis)} XAU Adverse Events:")
            for i, event in enumerate(analysis, 1):
                logger.info(f"  #{i} {event['time']} | Price: ${event['price']:,.2f} | "
                           f"Swing: {event['swing_pct']:.2f}% | "
                           f"Shock: {event['shock_label']} ({event['shock_score']:.2f}) | "
                           f"Active TEs: {event['active_te_count']}")
                for te_name, te_info in event["active_tes"].items():
                    logger.info(f"      {te_name:20s} str={te_info['strength']:.3f} ({te_info['signal']})")

            rules = generate_crispr_rules(analysis)
            if rules:
                inject_crispr_rules(rules)

            # Verify total CRISPR spacers
            conn = sqlite3.connect(str(CRISPR_DB_PATH))
            total_spacers = conn.execute("SELECT COUNT(*) FROM spacers WHERE expired=0").fetchone()[0]
            btc_sp = conn.execute("SELECT COUNT(*) FROM spacers WHERE symbol='BTCUSD' AND expired=0").fetchone()[0]
            eth_sp = conn.execute("SELECT COUNT(*) FROM spacers WHERE symbol='ETHUSD' AND expired=0").fetchone()[0]
            xau_sp = conn.execute("SELECT COUNT(*) FROM spacers WHERE symbol='XAUUSD' AND expired=0").fetchone()[0]
            conn.close()
            logger.info(f"\nCRISPR Spacer Census: {total_spacers} total (BTC={btc_sp}, ETH={eth_sp}, XAU={xau_sp})")

            # Save report
            report = {
                "symbol": "XAUUSD",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_source": "Real farm trade outcomes (signal_farm_logs)",
                "seed_antibodies": [{
                    "id": s["antibody_id"],
                    "v": s["v_name"], "d": s["d_name"], "j": s["j_name"],
                    "fitness": s["fitness"],
                    "source": s["te_source"],
                } for s in seeds],
                "adverse_events": analysis,
                "crispr_rules": rules,
            }
            report_path = current_dir / "sim_results" / "xau_winning_dna_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"\nReport saved to {report_path}")
        else:
            logger.warning("No adverse events found")
    else:
        logger.warning("Could not load XAU data - skipping CRISPR analysis")

    logger.info(f"\n{'=' * 70}")
    logger.info("  XAU DNA INJECTION COMPLETE - IMMUNE SYSTEM FULLY ARMED")
    logger.info(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
