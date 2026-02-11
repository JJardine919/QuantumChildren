"""
WINNING DNA EXTRACTION & CRISPR BREACH ANALYSIS
=================================================
Two jobs:
  1. Extract top 3 BTC strategy parameters -> VDJ seed antibodies
  2. Analyze daily DD breach timestamps -> identify active TE combos -> CRISPR rules

Top 3 BTC performers (3-week sim):
  #1 FARM_07 (Scalp-Volume):  +59.5%, PF=1.45, 907 trades
  #2 FARM_05 (Tight-Stop):   +50.0%, PF=1.39, 723 trades
  #3 FARM_10 (Swing-Loose):  +30.7%, PF=1.65, 182 trades

Date: 2026-02-10
"""

import sys
import json
import hashlib
import sqlite3
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Path setup
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][DNA-INJECT] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DNA-INJECT")


# =============================================================================
# PART 1: SEED ANTIBODY INJECTION
# =============================================================================

# Top 3 winning FARM parameter sets
WINNING_STRATEGIES = {
    "FARM_07_ScalpVolume": {
        "label": "Scalp-Volume",
        "confidence_threshold": 0.20,
        "tp_atr_multiplier": 1.8,
        "sl_atr_multiplier": 0.7,
        "max_positions_per_symbol": 4,
        "grid_spacing_atr": 0.25,
        "partial_tp_ratio": 0.60,
        "breakeven_trigger": 0.20,
        "trail_start_trigger": 0.35,
        "trail_distance_atr": 0.5,
        "max_loss_dollars": 0.60,
        "compression_boost": 14.0,
        # Sim results
        "return_pct": 59.53,
        "net_profit": 2976.59,
        "win_rate": 44.0,
        "profit_factor": 1.45,
        "max_dd_pct": 7.3,
        "total_trades": 907,
        "winners": 399,
        "losers": 508,
    },
    "FARM_05_TightStop": {
        "label": "Tight-Stop",
        "confidence_threshold": 0.28,
        "tp_atr_multiplier": 2.0,
        "sl_atr_multiplier": 0.8,
        "max_positions_per_symbol": 3,
        "grid_spacing_atr": 0.5,
        "partial_tp_ratio": 0.60,
        "breakeven_trigger": 0.20,
        "trail_start_trigger": 0.40,
        "trail_distance_atr": 0.6,
        "max_loss_dollars": 0.60,
        "compression_boost": 10.0,
        # Sim results
        "return_pct": 50.02,
        "net_profit": 2500.77,
        "win_rate": 41.8,
        "profit_factor": 1.39,
        "max_dd_pct": 6.88,
        "total_trades": 723,
        "winners": 302,
        "losers": 421,
    },
    "FARM_10_SwingLoose": {
        "label": "Swing-Loose",
        "confidence_threshold": 0.20,
        "tp_atr_multiplier": 4.5,
        "sl_atr_multiplier": 2.0,
        "max_positions_per_symbol": 3,
        "grid_spacing_atr": 0.8,
        "partial_tp_ratio": 0.45,
        "breakeven_trigger": 0.25,
        "trail_start_trigger": 0.40,
        "trail_distance_atr": 1.3,
        "max_loss_dollars": 1.00,
        "compression_boost": 12.0,
        # Sim results
        "return_pct": 30.72,
        "net_profit": 1535.86,
        "win_rate": 50.5,
        "profit_factor": 1.65,
        "max_dd_pct": 6.25,
        "total_trades": 182,
        "winners": 92,
        "losers": 90,
    },
}

# Map FARM strategy archetypes to VDJ gene segments
# Based on how each strategy trades: what it detects (V), what regime it thrives in (D), how it exits (J)
STRATEGY_TO_VDJ = {
    "FARM_07_ScalpVolume": [
        # High-frequency scalping with momentum + volume
        {"v": "V_momentum_fast",    "d": "D_trending_up",    "j": "J_trail_atr"},
        {"v": "V_flow_volume",      "d": "D_trending_down",  "j": "J_dynamic_partial"},
        {"v": "V_momentum_fast",    "d": "D_breakout",       "j": "J_trail_parabolic"},
    ],
    "FARM_05_TightStop": [
        # Tight risk, moderate TP, structure-based entries
        {"v": "V_structure_rsi",    "d": "D_trending_up",    "j": "J_fixed_rr2"},
        {"v": "V_vol_breakout",     "d": "D_breakout",       "j": "J_trail_atr"},
        {"v": "V_structure_sr",     "d": "D_ranging",        "j": "J_dynamic_partial"},
    ],
    "FARM_10_SwingLoose": [
        # Wide stops, patient swings, high PF
        {"v": "V_momentum_slow",    "d": "D_trending_up",    "j": "J_trail_chandelier"},
        {"v": "V_structure_sr",     "d": "D_accumulation",   "j": "J_fixed_rr3"},
        {"v": "V_neural_mosaic",    "d": "D_trending_down",  "j": "J_trail_chandelier"},
    ],
}


def make_antibody_id(v_name, d_name, j_name, params):
    raw = f"{v_name}|{d_name}|{j_name}|{json.dumps(params, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def farm_to_vdj_params(strategy_key, vdj_combo):
    """Convert FARM strategy parameters to VDJ perturbed params."""
    s = WINNING_STRATEGIES[strategy_key]

    # V params: entry signal lookbacks derived from strategy speed
    v_params = {}
    if "fast" in vdj_combo["v"] or "flow" in vdj_combo["v"]:
        v_params = {"lookback_5": 5, "lookback_8": 8, "lookback_13": 13}
    elif "slow" in vdj_combo["v"] or "mosaic" in vdj_combo["v"]:
        v_params = {"lookback_21": 21, "lookback_34": 34, "lookback_55": 55}
    elif "rsi" in vdj_combo["v"]:
        v_params = {"lookback_14": 14}
    elif "breakout" in vdj_combo["v"] or "sr" in vdj_combo["v"]:
        v_params = {"lookback_20": 20, "lookback_50": 50}
    else:
        v_params = {"lookback_10": 10, "lookback_20": 20}

    # D params: regime modifiers from strategy aggressiveness
    d_params = {
        "momentum_mult": 1.0 + (1.0 - s["confidence_threshold"]) * 0.5,  # Lower conf = more momentum
        "reversion_mult": s["confidence_threshold"],  # Higher conf = more reversion allowed
        "sl_mult": s["sl_atr_multiplier"],
        "lot_mult": min(1.5, s["max_positions_per_symbol"] / 3.0),
    }

    # J params: exit strategy from TP/SL ratios
    j_params = {
        "atr_multiplier": s["trail_distance_atr"],
        "activation_rr": s["breakeven_trigger"] * 4,  # Scale to R:R
        "tp1_rr": s["tp_atr_multiplier"] / s["sl_atr_multiplier"],  # Effective R:R
        "partial_close_pct": s["partial_tp_ratio"],
    }

    return v_params, d_params, j_params


def compute_fitness(s):
    """Compute VDJ fitness from FARM sim results."""
    # Bayesian win rate (Beta(3,3) prior)
    posterior_wr = (3 + s["winners"]) / (6 + s["total_trades"])

    # Profit factor normalized
    pf_norm = min(1.0, s["profit_factor"] / 3.0)

    # Sortino estimate from win rate and PF
    avg_win = s["net_profit"] / max(1, s["winners"]) if s["winners"] > 0 else 0
    avg_loss = abs(s["net_profit"] - avg_win * s["winners"]) / max(1, s["losers"]) if s["losers"] > 0 else 1
    sortino_est = min(1.0, max(0.0, (avg_win / max(avg_loss, 0.01)) / 3.0))

    # Consistency
    consistency = min(1.0, max(0.0, (s["win_rate"] / 100.0 - 0.30) / 0.40))

    # DD penalty
    dd_penalty = min(1.0, s["max_dd_pct"] / 10.0)

    # Trade count penalty
    trade_penalty = max(0.0, (20 - s["total_trades"]) / 20.0) if s["total_trades"] < 20 else 0.0

    fitness = (
        0.25 * posterior_wr +
        0.20 * pf_norm +
        0.20 * sortino_est +
        0.15 * consistency -
        0.10 * dd_penalty -
        0.10 * trade_penalty
    )
    fitness = max(0.0, min(1.0, fitness))

    return fitness, posterior_wr, sortino_est


def create_seed_antibodies():
    """Create seed antibodies from top 3 winning strategies."""
    seeds = []
    now = datetime.now().isoformat()

    for strategy_key, combos in STRATEGY_TO_VDJ.items():
        s = WINNING_STRATEGIES[strategy_key]
        fitness, posterior_wr, sortino_est = compute_fitness(s)

        logger.info(f"Strategy: {s['label']} -> fitness={fitness:.3f}, wr={posterior_wr:.3f}")

        for i, combo in enumerate(combos):
            v_params, d_params, j_params = farm_to_vdj_params(strategy_key, combo)

            antibody = {
                "antibody_id": make_antibody_id(combo["v"], combo["d"], combo["j"], j_params),
                "v_name": combo["v"],
                "d_name": combo["d"],
                "j_name": combo["j"],
                "perturbed_v_params": v_params,
                "perturbed_d_params": d_params,
                "perturbed_j_params": j_params,
                "fitness": fitness,
                "posterior_wr": posterior_wr,
                "profit_factor": s["profit_factor"],
                "sortino": sortino_est * 3.0,  # Denormalize
                "n_trades": s["total_trades"],
                "n_wins": s["winners"],
                "total_profit": s["net_profit"],
                "total_loss": -(s["net_profit"] / s["profit_factor"]) if s["profit_factor"] > 0 else 0,
                "max_drawdown": s["max_dd_pct"] * 50.0,  # Rough $ estimate from $5K balance
                "trade_returns": [],  # Not available from summary
                "generation": 0,
                "parent_id": "",
                "maturation_rounds": 5,  # Mark as mature (survived 3-week sim)
                "created_at": now,
                "last_activated": now,
                "activation_count": 0,
                "active": 1,
                "source": f"SEED_INJECTION:{strategy_key}_v{i}",
                "te_source": combo["v"].replace("V_", ""),
            }
            seeds.append(antibody)

    return seeds


def inject_into_vdj_db(seeds, db_path=None):
    """Inject seed antibodies into vdj_memory_cells.db."""
    if db_path is None:
        db_path = current_dir / "vdj_memory_cells.db"

    logger.info(f"Injecting {len(seeds)} seed antibodies into {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Ensure table exists
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
    logger.info(f"Injected {injected}/{len(seeds)} seed antibodies into VDJ memory DB")
    return injected


# =============================================================================
# PART 2: DAILY DD BREACH ANALYSIS -> TE ACTIVATION -> CRISPR RULES
# =============================================================================

def load_btc_data():
    """Load BTC 5m CSV data."""
    csv_path = current_dir / "HistoricalData" / "Full" / "BTCUSDT_5m.csv"
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
    logger.info(f"Loaded {len(df)} BTC 5m bars ({df['time'].min()} to {df['time'].max()})")
    return df


def find_dd_breach_bars(df, daily_dd_limit=175.0):
    """
    Find bars where daily P/L drawdown would have breached the $175 limit.
    Simulates a simple position following close prices and tracks intraday DD.
    """
    breach_bars = []
    df['date'] = df['time'].dt.date

    for date, group in df.groupby('date'):
        closes = group['close'].values
        if len(closes) < 2:
            continue

        # Track intraday P/L swing (max price - min price seen so far)
        day_high = closes[0]
        day_low = closes[0]
        max_adverse = 0.0

        for i in range(1, len(closes)):
            day_high = max(day_high, closes[i])
            day_low = min(day_low, closes[i])

            # Adverse move = max drop from any intraday high
            drop_from_high = day_high - closes[i]

            # Scale to dollar risk (assume 0.01 lot BTC, rough $1/point)
            # With tight stops (0.7-0.8 ATR), $175 breach ~= large intraday swing
            dollar_adverse = drop_from_high * 0.01  # Scale factor for simulation

            if drop_from_high > max_adverse:
                max_adverse = drop_from_high

                # Check if this bar's volatility swing exceeds threshold
                # Use raw price swing as proxy - the sim showed DD breaches
                # happened on days with $175+ swings
                atr_14 = np.std(closes[max(0, i-14):i+1]) * 2 if i >= 14 else np.std(closes[:i+1]) * 2
                swing_pct = drop_from_high / closes[i] * 100

                if swing_pct > 0.5:  # >0.5% intraday drop = significant for BTC
                    bar_idx = group.index[i]
                    breach_bars.append({
                        'bar_index': bar_idx,
                        'time': df.loc[bar_idx, 'time'],
                        'date': date,
                        'price': closes[i],
                        'drop_from_high': drop_from_high,
                        'swing_pct': swing_pct,
                        'day_high': day_high,
                        'atr_est': atr_14,
                    })

    # Keep only the worst bar per day (highest swing)
    if breach_bars:
        breach_df = pd.DataFrame(breach_bars)
        worst_per_day = breach_df.loc[breach_df.groupby('date')['swing_pct'].idxmax()]
        # Top 10 worst days
        worst_per_day = worst_per_day.nlargest(10, 'swing_pct')
        return worst_per_day.to_dict('records')

    return []


def compute_te_activations_at_bar(df, bar_idx, lookback=50):
    """
    Compute TE activations for the bars leading up to a DD breach.
    Uses the raw signal computations that each TE family monitors.
    """
    start = max(0, bar_idx - lookback)
    segment = df.iloc[start:bar_idx + 1]

    closes = segment['close'].values
    highs = segment['high'].values
    lows = segment['low'].values
    opens = segment['open'].values
    volumes = segment['tick_volume'].values

    if len(closes) < 20:
        return {"activations": {}, "shock_score": 0, "shock_label": "CALM", "atr_ratio": 0, "vol_ratio": 0}

    # Compute the signals each TE family monitors
    activations = {}

    # --- CLASS I (Retrotransposons) ---

    # TE0 BEL_Pao: 10-bar momentum
    mom_10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 11 else 0
    activations["BEL_Pao"] = {"strength": 1.0 / (1 + np.exp(-mom_10 * 20)), "signal": "momentum", "raw": mom_10}

    # TE1 DIRS1: EMA8 vs EMA21 trend strength
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

    # TE3 Ty3_gypsy: MACD divergence
    ema12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
    ema26 = pd.Series(closes).ewm(span=26).mean().iloc[-1]
    macd = ema12 - ema26
    signal_line = pd.Series(closes).ewm(span=12).mean().iloc[-1] - pd.Series(closes).ewm(span=26).mean().iloc[-1]
    macd_diff = macd - signal_line * 0.9  # Approximate signal line
    activations["Ty3_gypsy"] = {"strength": 1.0 / (1 + np.exp(-macd_diff * 100 / closes[-1])), "signal": "macd", "raw": macd_diff}

    # TE5 Alu: Short volatility spike
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

    # --- CLASS II (DNA Transposons) ---

    # TE12 Crypton: Compression ratio
    import zlib
    price_bytes = closes[-50:].tobytes() if len(closes) >= 50 else closes.tobytes()
    compressed = zlib.compress(price_bytes)
    comp_ratio = len(price_bytes) / max(len(compressed), 1)
    activations["Crypton"] = {"strength": 1.0 / (1 + np.exp(-(comp_ratio - 1.5))), "signal": "compression", "raw": comp_ratio}

    # TE18 Mutator: Sign changes in returns
    returns = np.diff(closes[-21:])
    signs = np.sign(returns)
    sign_changes = np.sum(np.abs(np.diff(signs)) > 0) / max(len(signs) - 1, 1)
    activations["Mutator"] = {"strength": min(1.0, sign_changes), "signal": "mutation_rate", "raw": sign_changes}

    # --- NEURAL TEs ---

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

    # TE30 TRIM28_Silencer: Drawdown (inverse - active during calm)
    dd_signal = (closes[-1] - max(closes[-20:])) / max(closes[-20:]) if len(closes) >= 20 else 0
    trim28_strength = 1.0 / (1 + np.exp(dd_signal * 10))  # Inverse: more DD = stronger suppression
    activations["TRIM28_Silencer"] = {"strength": trim28_strength, "signal": "drawdown_suppressor", "raw": dd_signal}

    # Compute shock level
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


def analyze_dd_breaches(df):
    """Find DD breach bars and compute TE activations at each."""
    logger.info("Analyzing daily drawdown breach events...")

    breach_bars = find_dd_breach_bars(df)
    if not breach_bars:
        logger.warning("No significant DD breach bars found")
        return []

    logger.info(f"Found {len(breach_bars)} worst daily swing events")

    breach_analysis = []
    for breach in breach_bars:
        bar_idx = breach['bar_index']
        te_data = compute_te_activations_at_bar(df, bar_idx)

        # Find active TEs (strength >= 0.5)
        active_tes = {name: data for name, data in te_data["activations"].items()
                      if data["strength"] >= 0.5}

        breach_analysis.append({
            "time": str(breach['time']),
            "date": str(breach['date']),
            "price": breach['price'],
            "swing_pct": breach['swing_pct'],
            "drop_from_high": breach['drop_from_high'],
            "shock_label": te_data["shock_label"],
            "shock_score": te_data["shock_score"],
            "active_te_count": len(active_tes),
            "active_tes": {name: {"strength": round(d["strength"], 3), "signal": d["signal"]}
                          for name, d in active_tes.items()},
            "all_te_strengths": {name: round(d["strength"], 3)
                                for name, d in te_data["activations"].items()},
        })

    return breach_analysis


def generate_crispr_rules(breach_analysis):
    """
    From DD breach TE analysis, identify recurring harmful TE combos
    and generate CRISPR deletion rules.
    """
    logger.info("Generating CRISPR protective deletion rules...")

    # Count TE co-occurrences across all breach events
    te_cooccurrence = {}
    te_frequency = {}

    for breach in breach_analysis:
        active_names = list(breach["active_tes"].keys())

        for te in active_names:
            te_frequency[te] = te_frequency.get(te, 0) + 1

        # Count pairs
        for i in range(len(active_names)):
            for j in range(i + 1, len(active_names)):
                pair = tuple(sorted([active_names[i], active_names[j]]))
                te_cooccurrence[pair] = te_cooccurrence.get(pair, 0) + 1

    # Sort by frequency
    sorted_pairs = sorted(te_cooccurrence.items(), key=lambda x: x[1], reverse=True)
    sorted_singles = sorted(te_frequency.items(), key=lambda x: x[1], reverse=True)

    # Generate rules for pairs that appear in >50% of breach events
    threshold = len(breach_analysis) * 0.5
    harmful_combos = []

    for pair, count in sorted_pairs:
        if count >= threshold:
            harmful_combos.append({
                "te_combo": list(pair),
                "occurrences": count,
                "breach_pct": count / len(breach_analysis) * 100,
                "rule": "BLOCK",
                "reason": f"Co-active in {count}/{len(breach_analysis)} DD breach events ({count/len(breach_analysis)*100:.0f}%)",
            })

    return harmful_combos, sorted_singles, sorted_pairs


def inject_crispr_rules(harmful_combos, db_path=None):
    """Inject CRISPR spacers for harmful TE combinations."""
    if db_path is None:
        db_path = current_dir / "crispr_cas.db"

    logger.info(f"Injecting {len(harmful_combos)} CRISPR rules into {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spacers (
            spacer_id TEXT PRIMARY KEY,
            fingerprint_json TEXT NOT NULL,
            fingerprint_vector BLOB NOT NULL,
            symbol TEXT NOT NULL,
            direction INTEGER NOT NULL,
            loss_amount REAL DEFAULT 0.0,
            active_tes TEXT DEFAULT '[]',
            volatility_regime TEXT DEFAULT 'MEDIUM',
            session TEXT DEFAULT 'OVERLAP',
            hour_of_day INTEGER DEFAULT 12,
            day_of_week INTEGER DEFAULT 2,
            acquired_at TEXT NOT NULL,
            last_matched TEXT,
            match_count INTEGER DEFAULT 0,
            expired INTEGER DEFAULT 0,
            merge_count INTEGER DEFAULT 1
        )
    """)

    now = datetime.now().isoformat()
    injected = 0

    for combo in harmful_combos:
        te_names = combo["te_combo"]

        # Create synthetic fingerprint vector focused on the TE combo
        # Use a deterministic vector so the same combo always matches
        combo_str = "|".join(sorted(te_names))
        seed_hash = hashlib.md5(combo_str.encode()).digest()
        np.random.seed(int.from_bytes(seed_hash[:4], 'big'))
        synthetic_vec = np.random.randn(26).astype(np.float64)
        vec_bytes = synthetic_vec.tobytes()
        spacer_id = hashlib.md5(vec_bytes).hexdigest()[:16]

        fp_json = json.dumps({
            "type": "TE_COMBO_CRISPR",
            "te_combo": te_names,
            "reason": combo["reason"],
            "breach_pct": combo["breach_pct"],
        })

        for direction in [1, -1]:  # Block both LONG and SHORT
            dir_spacer_id = f"{spacer_id}_{'+' if direction == 1 else '-'}"

            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO spacers (
                        spacer_id, fingerprint_json, fingerprint_vector,
                        symbol, direction, loss_amount, active_tes,
                        volatility_regime, session, hour_of_day, day_of_week,
                        acquired_at, expired, merge_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1)
                """, (
                    dir_spacer_id, fp_json, vec_bytes,
                    "BTCUSD", direction, 175.0,  # DD limit amount
                    json.dumps(te_names),
                    "HIGH", "OVERLAP", 14, 2,
                    now,
                ))
                injected += 1
            except Exception as e:
                logger.error(f"Failed to inject CRISPR rule {dir_spacer_id}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Injected {injected} CRISPR spacers (both directions)")
    return injected


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("  WINNING DNA EXTRACTION & CRISPR BREACH ANALYSIS")
    logger.info("=" * 70)

    # --- PART 1: Seed Antibody Injection ---
    logger.info("\n--- PART 1: SEED ANTIBODY INJECTION ---")
    seeds = create_seed_antibodies()

    logger.info(f"\nCreated {len(seeds)} seed antibodies:")
    for ab in seeds:
        logger.info(f"  {ab['antibody_id'][:8]}... "
                    f"V={ab['v_name']:20s} D={ab['d_name']:18s} J={ab['j_name']:18s} "
                    f"fit={ab['fitness']:.3f} src={ab['source']}")

    injected = inject_into_vdj_db(seeds)

    # --- PART 2: DD Breach TE Analysis ---
    logger.info("\n--- PART 2: DAILY DD BREACH TE ANALYSIS ---")
    df = load_btc_data()
    breach_analysis = analyze_dd_breaches(df)

    if breach_analysis:
        logger.info(f"\nTop {len(breach_analysis)} DD Breach Events:")
        for i, b in enumerate(breach_analysis, 1):
            active_names = list(b["active_tes"].keys())
            logger.info(f"  #{i} {b['time']} | Price: ${b['price']:,.0f} | "
                       f"Swing: {b['swing_pct']:.2f}% | Shock: {b['shock_label']} "
                       f"({b['shock_score']:.2f}) | Active TEs: {len(active_names)}")
            for te_name, te_data in b["active_tes"].items():
                logger.info(f"      {te_name:20s} str={te_data['strength']:.3f} ({te_data['signal']})")

        # Generate CRISPR rules
        harmful_combos, te_freq, te_pairs = generate_crispr_rules(breach_analysis)

        logger.info(f"\nTE Frequency in DD Breaches:")
        for te, count in te_freq[:10]:
            logger.info(f"  {te:20s}: {count}/{len(breach_analysis)} events ({count/len(breach_analysis)*100:.0f}%)")

        logger.info(f"\nTop TE Pairs in DD Breaches:")
        for pair, count in te_pairs[:10]:
            logger.info(f"  {pair[0]:20s} + {pair[1]:20s}: {count} events")

        if harmful_combos:
            logger.info(f"\n--- CRISPR RULES TO INJECT ---")
            for combo in harmful_combos:
                logger.info(f"  BLOCK: {combo['te_combo']} ({combo['reason']})")

            crispr_count = inject_crispr_rules(harmful_combos)
        else:
            logger.info("No TE combos exceeded 50% co-occurrence threshold for CRISPR rules")

    # --- Save Report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "seed_antibodies": len(seeds),
        "seed_details": [{
            "id": ab["antibody_id"],
            "v": ab["v_name"], "d": ab["d_name"], "j": ab["j_name"],
            "fitness": ab["fitness"],
            "source": ab["source"],
        } for ab in seeds],
        "dd_breach_events": len(breach_analysis),
        "breach_details": breach_analysis,
    }

    report_path = current_dir / "sim_results" / "winning_dna_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved to {report_path}")

    logger.info("\n" + "=" * 70)
    logger.info("  COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
