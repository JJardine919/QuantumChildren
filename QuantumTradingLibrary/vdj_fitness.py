"""
VDJ Fitness Function -- Clonal Selection & Affinity Maturation
===============================================================
6-component composite fitness function for antibody evaluation,
clonal selection thresholds, and somatic hypermutation.

Fitness formula:
    F(A) = 0.25 * posterior_WR + 0.20 * profit_factor_norm
         + 0.20 * sortino_norm + 0.15 * consistency
         - 0.10 * max_dd_penalty - 0.10 * trade_count_penalty

Selection thresholds:
    APOPTOSIS:     F < 0.25  (strategy dies)
    SURVIVAL:      F >= 0.40 (survives, no growth)
    PROLIFERATION: F >= 0.55 (cloned with mutations)
    MEMORY:        F >= 0.70 (promoted to memory B cell)

Affinity maturation:
    effective_rate = base_rate / (1 + 0.1 * generation)
    Converges over time: early generations explore, later ones refine.

Authors: DooDoo + Claude
Date:    2026-02-09
Parent:  ALGORITHM_VDJ_RECOMBINATION v1.0
"""

import logging
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

# Bayesian prior for win rate (Beta distribution)
PRIOR_ALPHA = 10
PRIOR_BETA = 10

# Fitness weights
W_POSTERIOR_WR = 0.25
W_PROFIT_FACTOR = 0.20
W_SORTINO = 0.20
W_CONSISTENCY = 0.15
W_MAX_DD = 0.10
W_TRADE_COUNT = 0.10

# Minimum trades for reliable fitness
MIN_TRADES = 20

# Clonal selection thresholds
APOPTOSIS_THRESHOLD = 0.25
SURVIVAL_THRESHOLD = 0.40
PROLIFERATION_THRESHOLD = 0.55
MEMORY_THRESHOLD = 0.70

# Population management
MAX_ACTIVE_ANTIBODIES = 50
MIN_ACTIVE_ANTIBODIES = 10
GENERATION_SIZE = 20
EVALUATION_WINDOW_BARS = 500

# Affinity maturation
BASE_MUTATION_RATE = 0.05
CONVERGENCE_COEFF = 0.1
N_MUTANTS_PER_WINNER = 5

# Memory cell management
DOMESTICATION_EXPIRY_DAYS = 30


# ============================================================
# FITNESS FUNCTION
# ============================================================

def fitness_clonal_selection(antibody_result: Dict) -> float:
    """
    6-component composite fitness function for clonal selection.

    In biology: how strongly does the antibody bind the antigen?
    In trading: how well does the micro-strategy perform on recent data?

    Args:
        antibody_result: dict with keys:
            n_trades, n_wins, total_profit, total_loss,
            trade_returns (list of floats), max_drawdown,
            account_equity (optional, default 5000)

    Returns:
        Fitness score clipped to [0.0, 1.0]
    """
    r = antibody_result
    n_trades = r.get("n_trades", 0)
    n_wins = r.get("n_wins", 0)
    n_losses = n_trades - n_wins
    trade_returns = r.get("trade_returns", [])

    # Component 1: Bayesian posterior win rate
    posterior_wr = (PRIOR_ALPHA + n_wins) / (PRIOR_ALPHA + PRIOR_BETA + n_trades)

    # Component 2: Profit factor (normalized)
    total_profit = r.get("total_profit", 0.0)
    total_loss = r.get("total_loss", 0.0)
    avg_win = total_profit / max(1, n_wins) if total_profit > 0 else 0
    avg_loss = abs(total_loss) / max(1, n_losses) if total_loss < 0 else 0.01
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
    pf_norm = min(1.0, profit_factor / 3.0)

    # Component 3: Sortino ratio (normalized)
    if len(trade_returns) > 1:
        mean_return = np.mean(trade_returns)
        downside_returns = [ret for ret in trade_returns if ret < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-10
        sortino = mean_return / (downside_std + 1e-10)
        sortino_norm = min(1.0, max(0.0, sortino / 3.0))
    else:
        sortino_norm = 0.0

    # Component 4: Consistency (low variance relative to mean)
    if len(trade_returns) > 1 and np.mean(trade_returns) != 0:
        consistency = 1.0 - min(1.0, np.std(trade_returns) /
                                (abs(np.mean(trade_returns)) + 1e-10))
        consistency = max(0.0, consistency)
    else:
        consistency = 0.0

    # Component 5: Maximum drawdown penalty
    max_dd = r.get("max_drawdown", 0.0)
    equity = r.get("account_equity", 5000)
    max_dd_penalty = min(1.0, max_dd / (equity * 0.10))

    # Component 6: Trade count penalty (insufficient data)
    if n_trades < MIN_TRADES:
        trade_penalty = (MIN_TRADES - n_trades) / MIN_TRADES
    else:
        trade_penalty = 0.0

    # Composite fitness
    fitness = (
        W_POSTERIOR_WR * posterior_wr
        + W_PROFIT_FACTOR * pf_norm
        + W_SORTINO * sortino_norm
        + W_CONSISTENCY * consistency
        - W_MAX_DD * max_dd_penalty
        - W_TRADE_COUNT * trade_penalty
    )

    return float(np.clip(fitness, 0.0, 1.0))


def compute_detailed_metrics(antibody_result: Dict) -> Dict:
    """
    Compute all fitness sub-components for diagnostics.
    Returns dict with each component's value plus the composite.
    """
    r = antibody_result
    n_trades = r.get("n_trades", 0)
    n_wins = r.get("n_wins", 0)
    n_losses = n_trades - n_wins
    trade_returns = r.get("trade_returns", [])

    posterior_wr = (PRIOR_ALPHA + n_wins) / (PRIOR_ALPHA + PRIOR_BETA + n_trades)

    total_profit = r.get("total_profit", 0.0)
    total_loss = r.get("total_loss", 0.0)
    avg_win = total_profit / max(1, n_wins) if total_profit > 0 else 0
    avg_loss = abs(total_loss) / max(1, n_losses) if total_loss < 0 else 0.01
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

    sortino = 0.0
    if len(trade_returns) > 1:
        mean_return = np.mean(trade_returns)
        downside = [ret for ret in trade_returns if ret < 0]
        dd_std = np.std(downside) if len(downside) > 1 else 1e-10
        sortino = mean_return / (dd_std + 1e-10)

    return {
        "fitness": fitness_clonal_selection(antibody_result),
        "posterior_wr": posterior_wr,
        "profit_factor": profit_factor,
        "sortino": sortino,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": r.get("max_drawdown", 0.0),
    }


# ============================================================
# CLONAL SELECTION
# ============================================================

def classify_antibody(fitness: float) -> str:
    """
    Classify an antibody into one of four categories based on fitness.

    Returns: "APOPTOSIS", "ANERGY", "SURVIVAL", "PROLIFERATION", or "MEMORY"
    """
    if fitness < APOPTOSIS_THRESHOLD:
        return "APOPTOSIS"
    elif fitness < SURVIVAL_THRESHOLD:
        return "ANERGY"
    elif fitness < PROLIFERATION_THRESHOLD:
        return "SURVIVAL"
    elif fitness < MEMORY_THRESHOLD:
        return "PROLIFERATION"
    else:
        return "MEMORY"


def clonal_selection(antibodies: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Perform clonal selection on a population of antibodies.

    Each antibody dict must have a "fitness" key.

    Returns dict with keys: "dead", "anergic", "survivors",
                             "proliferators", "memory_candidates"
    """
    result = {
        "dead": [],
        "anergic": [],
        "survivors": [],
        "proliferators": [],
        "memory_candidates": [],
    }

    for ab in antibodies:
        fate = classify_antibody(ab.get("fitness", 0.0))
        if fate == "APOPTOSIS":
            result["dead"].append(ab)
        elif fate == "ANERGY":
            result["anergic"].append(ab)
        elif fate == "SURVIVAL":
            result["survivors"].append(ab)
        elif fate == "PROLIFERATION":
            result["proliferators"].append(ab)
            result["survivors"].append(ab)
        elif fate == "MEMORY":
            result["memory_candidates"].append(ab)
            result["survivors"].append(ab)

    log.info(
        "[VDJ-FIT] Clonal selection: %d dead, %d anergic, %d survive, %d proliferate, %d memory",
        len(result["dead"]), len(result["anergic"]),
        len(result["survivors"]),
        len(result["proliferators"]),
        len(result["memory_candidates"]),
    )

    return result


# ============================================================
# AFFINITY MATURATION (Somatic Hypermutation)
# ============================================================

def effective_mutation_rate(generation: int) -> float:
    """
    Compute the effective mutation rate for a given generation.

    rate(g) = base_rate / (1 + convergence_coeff * g)

    At generation 0:  5.0%
    At generation 10: 2.5%
    At generation 20: 1.7%
    At generation 50: 0.8%
    """
    return BASE_MUTATION_RATE / (1.0 + CONVERGENCE_COEFF * generation)


def affinity_maturation(
    winner: Dict,
    n_mutants: int = N_MUTANTS_PER_WINNER,
    generation: int = 0,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    """
    Somatic hypermutation of a winning antibody.

    Creates n_mutants variants by making small parameter adjustments.
    The mutation rate decreases with generation, modeling convergence.

    Args:
        winner: antibody dict with perturbed_v_params, perturbed_d_params,
                perturbed_j_params keys
        n_mutants: number of variants to create
        generation: current generation (controls mutation rate)
        rng: random state for reproducibility

    Returns:
        List of mutant antibody dicts.
    """
    if rng is None:
        rng = np.random.RandomState()

    eff_rate = effective_mutation_rate(generation)
    mutants = []

    for m in range(n_mutants):
        mutant = deepcopy(winner)
        mutant["parent_id"] = winner.get("antibody_id", "")
        mutant["antibody_id"] = f"{winner.get('antibody_id', 'ab')}__m{generation}_{m}"
        mutant["generation"] = generation + 1
        mutant["mutation_type"] = "somatic_hypermutation"

        # Mutate V junction parameters
        for key, val in mutant.get("perturbed_v_params", {}).items():
            if isinstance(val, float):
                mutant["perturbed_v_params"][key] = val * (1 + rng.normal(0, eff_rate))
            elif isinstance(val, int):
                if rng.random() < eff_rate:
                    mutant["perturbed_v_params"][key] = max(2, val + rng.randint(-2, 3))

        # Mutate D modifier parameters
        for key, val in mutant.get("perturbed_d_params", {}).items():
            if isinstance(val, (int, float)):
                mutant["perturbed_d_params"][key] = max(
                    0.01, val * (1 + rng.normal(0, eff_rate))
                )

        # Mutate J exit parameters
        for key, val in mutant.get("perturbed_j_params", {}).items():
            if isinstance(val, float):
                mutant["perturbed_j_params"][key] = max(
                    0.01, val * (1 + rng.normal(0, eff_rate))
                )
            elif isinstance(val, int):
                if rng.random() < eff_rate:
                    mutant["perturbed_j_params"][key] = max(1, val + rng.randint(-2, 3))

        # Reset fitness (must be re-evaluated)
        mutant["fitness"] = 0.0
        mutants.append(mutant)

    return mutants


def check_maturity(antibody: Dict, min_rounds: int = 3, improvement_threshold: float = 0.01) -> bool:
    """
    Check if an antibody has reached maturity (converged).

    Mature when:
        1. Survived >= min_rounds of affinity maturation
        2. Fitness improvement in last round < improvement_threshold
    """
    rounds = antibody.get("maturation_rounds", 0)
    last_improvement = antibody.get("last_fitness_improvement", 1.0)
    return rounds >= min_rounds and last_improvement < improvement_threshold


# ============================================================
# THYMIC NEGATIVE SELECTION
# ============================================================

def thymic_selection(antibody: Dict) -> bool:
    """
    Thymic negative selection: kill strategies that would 'attack self'
    (i.e., blow up the account).

    Returns True if antibody PASSES (is safe), False if it should die.

    Kill if:
        1. J segment is time-based without SL mechanism
        2. Position size multiplier > 3x
        3. No bounded loss scenario
    """
    from vdj_segments import J_SEGMENTS

    j_name = antibody.get("j_name", "")
    j_def = J_SEGMENTS.get(j_name, {})

    # Time-based exit without SL = unlimited risk
    if j_def.get("exit_type") == "TIME_BASED":
        if not j_def.get("params", {}).get("move_sl_be", False):
            return False

    # Excessive position size
    d_params = antibody.get("perturbed_d_params", {})
    if d_params.get("lot_mult", 1.0) > 3.0:
        return False

    return True
