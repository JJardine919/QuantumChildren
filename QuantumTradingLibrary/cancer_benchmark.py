"""
CANCER CELL BENCHMARK -- Capture baseline metrics before optimization.
=====================================================================
Runs two modes (random seeds vs expert parents) and records all metrics
to cancer_benchmarks.json for comparison against future iterations.

Usage: python cancer_benchmark.py
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import cancer_cell as cc
from cancer_cell import (CancerCellEngine, ExpertCellLoader,
                         MITOSIS_POPULATION_SIZE, MITOSIS_PARENT_COUNT,
                         MITOSIS_GENERATIONS, TELOMERASE_PROMOTION_WR,
                         IMMUNE_CHECKPOINT_WR, VERSION)

BENCHMARK_FILE = Path(__file__).parent / "cancer_benchmarks.json"
SYMBOLS = ["BTCUSD", "XAUUSD", "ETHUSD"]
BARS_COUNT = 2000
POP_SIZE = 200
GENERATIONS = 5


def load_bars():
    bars_by_symbol = {}
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            for sym in SYMBOLS:
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, BARS_COUNT)
                if rates is not None and len(rates) > 0:
                    bars_by_symbol[sym] = np.array([
                        [r[1], r[2], r[3], r[4], r[5]] for r in rates
                    ])
                    log.info("Loaded %d bars for %s", len(rates), sym)
            mt5.shutdown()
    except Exception as e:
        log.warning("MT5 not available: %s", e)

    for sym in SYMBOLS:
        if sym not in bars_by_symbol:
            n = BARS_COUNT
            close = np.cumsum(np.random.randn(n) * 0.001) + 100
            high = close + abs(np.random.randn(n) * 0.0005)
            low = close - abs(np.random.randn(n) * 0.0005)
            opn = close + np.random.randn(n) * 0.0002
            vol = np.random.randint(100, 10000, n).astype(float)
            bars_by_symbol[sym] = np.column_stack([opn, high, low, close, vol])
    return bars_by_symbol


def collect_run_metrics(engine, symbols, bars_by_symbol, parent_strategies,
                        use_experts, label):
    """Run simulation and collect ALL metrics from ALL survivors (not just promoted)."""
    cc.MITOSIS_POPULATION_SIZE = POP_SIZE
    cc.MITOSIS_GENERATIONS = GENERATIONS

    t0 = time.time()
    result = engine.run(symbols, bars_by_symbol,
                        parent_strategies=parent_strategies,
                        use_experts=use_experts,
                        return_all_survivors=True)
    elapsed = time.time() - t0

    validated, survivors = result
    total_mutants = POP_SIZE * cc.MITOSIS_PARENT_COUNT * len(symbols)

    # Collect from ALL survivors (not just promoted)
    all_fitness = [m.fitness_score for m in survivors if m.fitness_score > 0]
    all_wr = [m.win_rate for m in survivors if m.total_trades > 0]
    all_pf = [m.profit_factor for m in survivors if m.total_trades > 0]
    all_trades = [m.total_trades for m in survivors]
    all_telomere = [m.telomere_length for m in survivors]
    all_drivers = [m.driver_count for m in survivors]
    all_sharpe = [m.sharpe_ratio for m in survivors if m.total_trades > 0]
    all_drawdown = [m.max_drawdown for m in survivors if m.total_trades > 0]
    metastasis_count = sum(len(m.metastasized_to) for m in survivors)

    metrics = {
        "label": label,
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "symbols": symbols,
            "bars_per_symbol": BARS_COUNT,
            "population_per_parent": POP_SIZE,
            "generations": GENERATIONS,
            "parent_count": MITOSIS_PARENT_COUNT,
            "telomerase_wr_threshold": TELOMERASE_PROMOTION_WR,
            "immune_wr_threshold": IMMUNE_CHECKPOINT_WR,
        },
        "performance": {
            "elapsed_seconds": round(elapsed, 1),
            "mutants_per_second": round(total_mutants / max(elapsed, 0.1), 1),
        },
        "population": {
            "total_generated": total_mutants,
            "survivors_alive": len(survivors),
            "validated_promoted": len(validated),
            "survival_rate_pct": round(len(survivors) / max(total_mutants, 1) * 100, 2),
            "promotion_rate_pct": round(len(validated) / max(total_mutants, 1) * 100, 4),
        },
        "fitness": {
            "avg": round(float(np.mean(all_fitness)), 4) if all_fitness else 0,
            "max": round(float(np.max(all_fitness)), 4) if all_fitness else 0,
            "min": round(float(np.min(all_fitness)), 4) if all_fitness else 0,
            "std": round(float(np.std(all_fitness)), 4) if all_fitness else 0,
            "p75": round(float(np.percentile(all_fitness, 75)), 4) if all_fitness else 0,
            "p90": round(float(np.percentile(all_fitness, 90)), 4) if all_fitness else 0,
        },
        "win_rate": {
            "avg_pct": round(float(np.mean(all_wr)) * 100, 2) if all_wr else 0,
            "max_pct": round(float(np.max(all_wr)) * 100, 2) if all_wr else 0,
            "min_pct": round(float(np.min(all_wr)) * 100, 2) if all_wr else 0,
            "p75_pct": round(float(np.percentile(all_wr, 75)) * 100, 2) if all_wr else 0,
            "p90_pct": round(float(np.percentile(all_wr, 90)) * 100, 2) if all_wr else 0,
            "above_55_pct": round(sum(1 for w in all_wr if w >= 0.55) / max(len(all_wr), 1) * 100, 2),
            "above_60_pct": round(sum(1 for w in all_wr if w >= 0.60) / max(len(all_wr), 1) * 100, 2),
            "above_65_pct": round(sum(1 for w in all_wr if w >= 0.65) / max(len(all_wr), 1) * 100, 2),
        },
        "profit_factor": {
            "avg": round(float(np.mean(all_pf)), 4) if all_pf else 0,
            "max": round(float(np.max(all_pf)), 4) if all_pf else 0,
            "p75": round(float(np.percentile(all_pf, 75)), 4) if all_pf else 0,
        },
        "sharpe": {
            "avg": round(float(np.mean(all_sharpe)), 4) if all_sharpe else 0,
            "max": round(float(np.max(all_sharpe)), 4) if all_sharpe else 0,
        },
        "drawdown": {
            "avg": round(float(np.mean(all_drawdown)), 4) if all_drawdown else 0,
            "max": round(float(np.max(all_drawdown)), 4) if all_drawdown else 0,
        },
        "trades": {
            "avg_per_mutant": round(float(np.mean(all_trades)), 1) if all_trades else 0,
            "total": int(np.sum(all_trades)) if all_trades else 0,
        },
        "biology": {
            "avg_telomere_bp": round(float(np.mean(all_telomere)), 0) if all_telomere else 0,
            "avg_drivers": round(float(np.mean(all_drivers)), 1) if all_drivers else 0,
            "metastasis_events": metastasis_count,
            "promotions": len(validated),
        },
    }

    return metrics, validated


def print_table(benchmarks):
    """Print comparison table."""
    print()
    print("=" * 90)
    print("CANCER CELL BENCHMARKS")
    print("=" * 90)

    headers = ["Metric", *[b["label"] for b in benchmarks]]
    col_w = [30] + [25] * len(benchmarks)

    def row(label, key_path):
        vals = []
        for b in benchmarks:
            obj = b
            for k in key_path.split("."):
                obj = obj.get(k, {}) if isinstance(obj, dict) else 0
            vals.append(str(obj))
        parts = [label.ljust(col_w[0])]
        for i, v in enumerate(vals):
            parts.append(v.rjust(col_w[i + 1]))
        print("".join(parts))

    def sep():
        print("-" * sum(col_w))

    # Header
    parts = [headers[0].ljust(col_w[0])]
    for i, h in enumerate(headers[1:]):
        parts.append(h.rjust(col_w[i + 1]))
    print("".join(parts))
    sep()

    # Performance
    row("Elapsed (sec)", "performance.elapsed_seconds")
    row("Mutants/sec", "performance.mutants_per_second")
    sep()

    # Population
    row("Total generated", "population.total_generated")
    row("Survivors alive", "population.survivors_alive")
    row("Survival rate (%)", "population.survival_rate_pct")
    row("Validated (promoted)", "population.validated_promoted")
    row("Promotion rate (%)", "population.promotion_rate_pct")
    sep()

    # Fitness
    row("Fitness avg", "fitness.avg")
    row("Fitness max", "fitness.max")
    row("Fitness std", "fitness.std")
    row("Fitness p75", "fitness.p75")
    row("Fitness p90", "fitness.p90")
    sep()

    # Win Rate
    row("WR avg (%)", "win_rate.avg_pct")
    row("WR max (%)", "win_rate.max_pct")
    row("WR p75 (%)", "win_rate.p75_pct")
    row("WR p90 (%)", "win_rate.p90_pct")
    row("WR >= 55% (%pop)", "win_rate.above_55_pct")
    row("WR >= 60% (%pop)", "win_rate.above_60_pct")
    row("WR >= 65% (%pop)", "win_rate.above_65_pct")
    sep()

    # Profit Factor
    row("PF avg", "profit_factor.avg")
    row("PF max", "profit_factor.max")
    row("PF p75", "profit_factor.p75")
    sep()

    # Sharpe / Drawdown
    row("Sharpe avg", "sharpe.avg")
    row("Sharpe max", "sharpe.max")
    row("Drawdown avg", "drawdown.avg")
    row("Drawdown max", "drawdown.max")
    sep()

    # Trades
    row("Trades avg/mutant", "trades.avg_per_mutant")
    row("Trades total", "trades.total")
    sep()

    # Biology
    row("Avg telomere (bp)", "biology.avg_telomere_bp")
    row("Avg drivers", "biology.avg_drivers")
    row("Metastasis events", "biology.metastasis_events")
    row("Promotions", "biology.promotions")
    print("=" * sum(col_w))


def clear_db():
    """Truncate all tables for a clean run."""
    import sqlite3
    db_path = str(cc.DB_PATH)
    try:
        with sqlite3.connect(db_path) as conn:
            for tbl in ["mutants", "clusters", "metastasis_log",
                        "promotion_log", "immune_checkpoint", "run_history"]:
                conn.execute(f"DELETE FROM {tbl}")
            conn.commit()
    except Exception:
        pass


def main():
    bars_by_symbol = load_bars()
    benchmarks = []

    # --- Run 1: Random Seeds ---
    clear_db()
    log.info("=" * 60)
    log.info("BENCHMARK RUN 1: RANDOM SEEDS")
    log.info("=" * 60)
    engine1 = CancerCellEngine()
    m1, _ = collect_run_metrics(engine1, SYMBOLS, bars_by_symbol,
                                parent_strategies=None,
                                use_experts=False,
                                label="Random Seeds")
    benchmarks.append(m1)

    # --- Run 2: Expert Parents ---
    clear_db()
    log.info("=" * 60)
    log.info("BENCHMARK RUN 2: EXPERT PARENTS")
    log.info("=" * 60)
    engine2 = CancerCellEngine()
    m2, _ = collect_run_metrics(engine2, SYMBOLS, bars_by_symbol,
                                parent_strategies=None,
                                use_experts=True,
                                label="Expert Parents")
    benchmarks.append(m2)

    # Print table
    print_table(benchmarks)

    # Save to file
    existing = []
    if BENCHMARK_FILE.exists():
        try:
            existing = json.loads(BENCHMARK_FILE.read_text())
        except:
            existing = []

    existing.append({
        "benchmark_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "runs": benchmarks,
    })

    BENCHMARK_FILE.write_text(json.dumps(existing, indent=2))
    print(f"\nBenchmarks saved to: {BENCHMARK_FILE}")


if __name__ == "__main__":
    main()
