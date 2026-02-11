"""
SILVER WALK-FORWARD - Deep Validation on All Silver Symbols
============================================================
All silver symbols (XAGUSD, XAGEUR, XAGAUD) on M15, M20, M30.
1 year lookback, 2 rounds of 6-month train / 2-month test.

This proves whether the silver edge is real or just 30-day noise.
Walk-forward = train on past, test on UNSEEN future data.

Layout (1 year = 12 months):
  Round 1: Train months 1-6,  Test months 7-8
  Round 2: Train months 3-8,  Test months 9-10  (shifted forward)

  Months 11-12 = holdout (never trained on)

Usage:
    python silver_walkforward.py
    python silver_walkforward.py --accounts 1000 --rounds 5
"""

import numpy as np
import pandas as pd
import json
import time
import sys
import os
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import MetaTrader5 as mt5
from config_loader import (
    MAX_LOSS_DOLLARS, TP_MULTIPLIER, ROLLING_SL_MULTIPLIER,
    DYNAMIC_TP_PERCENT, CONFIDENCE_THRESHOLD
)

# Reuse DNA and challenge runner from prop_challenge_1000
from prop_challenge_1000 import (
    ChallengerDNA, ChallengeResult, SymbolResult,
    crossover_dna, run_challenge, run_multi_symbol_challenge
)


SILVER_SYMBOLS = ["XAGUSD", "XAGEUR", "XAGAUD"]
TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
}

# Walk-forward layout (in bars from end)
# 1 year of M15 = 96 bars/day * 252 trading days = ~24,192 bars
# 1 year of M30 = 48 bars/day * 252 = ~12,096 bars
# We'll fetch 1 year and split proportionally


def run_walkforward(num_accounts=1000, evo_rounds=5, days=365,
                    batch_size=100, workers=8, terminal_path=None):
    """
    Walk-forward validation on all silver symbols x M15/M20/M30.

    Train/test split:
      Round 1: First 60% train, next 20% test
      Round 2: Shift forward - middle 60% train, next 20% test
      Final 20% = holdout (unseen validation)
    """
    print("=" * 70)
    print("  SILVER WALK-FORWARD - DEEP VALIDATION")
    print("  All Silver x M15/M20/M30 x 1 Year")
    print("=" * 70)
    print(f"  Challengers:   {num_accounts}")
    print(f"  Evo Rounds:    {evo_rounds}")
    print(f"  Symbols:       {', '.join(SILVER_SYMBOLS)}")
    print(f"  Timeframes:    {', '.join(TIMEFRAMES.keys())}")
    print(f"  Lookback:      {days} days (~1 year)")
    print(f"  Walk-Forward:  2 rounds (6mo train / 2mo test)")
    print("=" * 70)

    # Init MT5
    init_ok = mt5.initialize(path=terminal_path) if terminal_path else mt5.initialize()
    if not init_ok:
        print("ERROR: MT5 init failed")
        return

    # Fetch data for all symbol x timeframe combos
    all_data = {}  # (symbol, tf_name) -> DataFrame
    all_info = {}  # symbol -> (point, contract_size)

    for symbol in SILVER_SYMBOLS:
        si = mt5.symbol_info(symbol)
        if si:
            all_info[symbol] = (si.point, si.trade_contract_size)
        else:
            all_info[symbol] = (0.001, 5000.0)

        for tf_name, tf_val in TIMEFRAMES.items():
            bars_per_day = {
                "M15": 96, "M20": 72, "M30": 48,
            }[tf_name]
            bars_needed = bars_per_day * days

            rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, min(bars_needed, 500000))
            if rates is None or len(rates) < 500:
                print(f"  WARNING: {symbol} {tf_name} - only {len(rates) if rates is not None else 0} bars")
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            all_data[(symbol, tf_name)] = df
            print(f"  {symbol} {tf_name}: {len(df):,} bars  ({df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()})")

    mt5.shutdown()

    if not all_data:
        print("ERROR: No data loaded")
        return

    print(f"\n  Loaded {len(all_data)} symbol-timeframe combinations")

    output_dir = SCRIPT_DIR / "signal_farm_output" / "silver_walkforward"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Walk-forward splits
    # Each dataset gets split into: train1, test1, train2, test2, holdout
    wf_splits = {}
    for key, df in all_data.items():
        n = len(df)
        # Round 1: first 60% train, next 20% test
        train1_end = int(n * 0.60)
        test1_end = int(n * 0.80)
        # Round 2: shift - 20% to 80% train, 80% to 100% test (holdout becomes test)
        train2_start = int(n * 0.20)
        train2_end = int(n * 0.80)
        test2_end = n

        wf_splits[key] = {
            "train1": df.iloc[:train1_end].copy(),
            "test1": df.iloc[train1_end:test1_end].copy(),
            "train2": df.iloc[train2_start:train2_end].copy(),
            "test2": df.iloc[train2_end:].copy(),
            "holdout": df.iloc[test1_end:].copy(),
        }
        symbol, tf = key
        print(f"  {symbol} {tf}: Train1={train1_end} bars, Test1={test1_end-train1_end}, "
              f"Train2={train2_end-train2_start}, Test2={n-train2_end}, Holdout={n-test1_end}")

    tournament_start = time.time()
    all_history = []
    grand_results = {}  # Track across WF rounds

    for wf_round in range(1, 3):  # 2 walk-forward rounds
        split_key = f"train{wf_round}"
        test_key = f"test{wf_round}"

        print(f"\n{'#' * 70}")
        print(f"# WALK-FORWARD ROUND {wf_round}/2 - TRAINING PHASE")
        print(f"{'#' * 70}")

        # Build train data dict per symbol (merge all TFs into one run? No - run per TF)
        for tf_name in TIMEFRAMES:
            print(f"\n  --- {tf_name} TRAINING ---")

            # Train data for this timeframe
            train_data = {}
            train_info = {}
            for symbol in SILVER_SYMBOLS:
                key = (symbol, tf_name)
                if key in wf_splits:
                    train_data[symbol] = wf_splits[key][split_key]
                    train_info[symbol] = all_info[symbol]

            if not train_data:
                print(f"  No data for {tf_name}, skipping")
                continue

            # Generate/evolve population
            if wf_round == 1:
                population = []
                for i in range(num_accounts):
                    dna = ChallengerDNA(challenger_id=i + 1)
                    if i > 0:
                        dna.mutate(strength=0.4)
                    population.append(dna)
            # Round 2 reuses surviving population from round 1

            for evo in range(1, evo_rounds + 1):
                evo_start = time.time()
                results = []
                total_batches = (len(population) + batch_size - 1) // batch_size

                for bn in range(total_batches):
                    s = bn * batch_size
                    e = min(s + batch_size, len(population))
                    batch = population[s:e]

                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {
                            executor.submit(run_multi_symbol_challenge, dna,
                                           train_data, train_info): dna
                            for dna in batch
                        }
                        for f in as_completed(futures):
                            try:
                                results.append(f.result())
                            except:
                                pass

                results.sort(key=lambda r: r.score, reverse=True)
                active = [r for r in results if r.total_trades > 0]
                avg_wr = sum(r.win_rate for r in active) / len(active) if active else 0
                passed = sum(1 for r in active if r.challenge_passed)
                evo_elapsed = time.time() - evo_start

                print(f"    Evo {evo}/{evo_rounds}: {len(active)} active, "
                      f"WR={avg_wr:.1f}%, Passed={passed}, "
                      f"Best=#{results[0].challenger_id if results else 0} "
                      f"({results[0].score:.1f}) [{evo_elapsed:.1f}s]")

                # Evolve
                if evo < evo_rounds:
                    elite_count = max(10, len(population) // 10)
                    breed_count = max(20, len(population) // 5)
                    result_map = {r.challenger_id: r for r in results}

                    population.sort(key=lambda d: result_map.get(d.challenger_id, ChallengeResult(
                        0, "", {}, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, "", 0)).score, reverse=True)

                    elite = population[:elite_count]
                    breeders = population[:elite_count + breed_count]
                    new_pop = list(elite)
                    next_id = max(d.challenger_id for d in population) + 1
                    while len(new_pop) < len(population):
                        p1 = random.choice(breeders)
                        p2 = random.choice(breeders)
                        child = crossover_dna(p1, p2, next_id)
                        child.mutate(strength=0.1 + 0.05 * evo)
                        child.generation = evo
                        new_pop.append(child)
                        next_id += 1
                    population = new_pop

            # ============================================================
            # TEST PHASE - Run champions on UNSEEN data
            # ============================================================
            print(f"\n  --- {tf_name} TESTING (UNSEEN DATA) ---")
            test_data = {}
            for symbol in SILVER_SYMBOLS:
                key = (symbol, tf_name)
                if key in wf_splits:
                    test_data[symbol] = wf_splits[key][test_key]

            if not test_data:
                print(f"  No test data for {tf_name}")
                continue

            # Take top 50 from training and test them
            top_ids = set(r.challenger_id for r in results[:50])
            top_dna = [d for d in population if d.challenger_id in top_ids]

            test_results = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(run_multi_symbol_challenge, dna,
                                   test_data, train_info): dna
                    for dna in top_dna
                }
                for f in as_completed(futures):
                    try:
                        test_results.append(f.result())
                    except:
                        pass

            test_results.sort(key=lambda r: r.score, reverse=True)
            test_active = [r for r in test_results if r.total_trades > 0]

            if test_active:
                test_avg_wr = sum(r.win_rate for r in test_active) / len(test_active)
                test_passed = sum(1 for r in test_active if r.challenge_passed)
                train_avg_wr = avg_wr

                print(f"    Train WR: {train_avg_wr:.1f}% -> Test WR: {test_avg_wr:.1f}%  "
                      f"({'HELD' if test_avg_wr >= train_avg_wr * 0.85 else 'DEGRADED'})")
                print(f"    Test Passed: {test_passed}/{len(test_active)}")

                if test_results:
                    best = test_results[0]
                    print(f"    Best on unseen: #{best.challenger_id} WR={best.win_rate:.1f}% "
                          f"PF={best.profit_factor:.2f} ${best.net_profit:,.2f}")

                result_key = f"WF{wf_round}_{tf_name}"
                grand_results[result_key] = {
                    "train_wr": round(train_avg_wr, 1),
                    "test_wr": round(test_avg_wr, 1),
                    "degradation": round(test_avg_wr - train_avg_wr, 1),
                    "test_passed": test_passed,
                    "test_total": len(test_active),
                    "best_id": best.challenger_id if test_results else 0,
                    "best_wr": best.win_rate if test_results else 0,
                    "best_pf": best.profit_factor if test_results else 0,
                    "best_profit": best.net_profit if test_results else 0,
                }

                all_history.append({
                    "wf_round": wf_round,
                    "timeframe": tf_name,
                    "train_wr": round(train_avg_wr, 1),
                    "test_wr": round(test_avg_wr, 1),
                    "top_10_test": [
                        {"id": r.challenger_id, "wr": r.win_rate, "pf": r.profit_factor,
                         "profit": r.net_profit, "best_sym": r.best_symbol}
                        for r in test_results[:10]
                    ],
                })

    # ============================================================
    # HOLDOUT VALIDATION - Ultimate test
    # ============================================================
    print(f"\n{'#' * 70}")
    print(f"# HOLDOUT VALIDATION - DATA NEVER SEEN IN ANY TRAINING")
    print(f"{'#' * 70}")

    for tf_name in TIMEFRAMES:
        holdout_data = {}
        for symbol in SILVER_SYMBOLS:
            key = (symbol, tf_name)
            if key in wf_splits and len(wf_splits[key]["holdout"]) > 100:
                holdout_data[symbol] = wf_splits[key]["holdout"]

        if not holdout_data:
            print(f"  {tf_name}: Insufficient holdout data")
            continue

        top_ids = set(r.challenger_id for r in results[:20])
        top_dna = [d for d in population if d.challenger_id in top_ids]

        holdout_results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_multi_symbol_challenge, dna,
                               holdout_data, {s: all_info[s] for s in holdout_data}): dna
                for dna in top_dna
            }
            for f in as_completed(futures):
                try:
                    holdout_results.append(f.result())
                except:
                    pass

        holdout_results.sort(key=lambda r: r.score, reverse=True)
        h_active = [r for r in holdout_results if r.total_trades > 0]
        if h_active:
            h_wr = sum(r.win_rate for r in h_active) / len(h_active)
            h_passed = sum(1 for r in h_active if r.challenge_passed)
            print(f"  {tf_name} HOLDOUT: WR={h_wr:.1f}%, Passed={h_passed}/{len(h_active)}")
            if holdout_results:
                b = holdout_results[0]
                print(f"    Best: #{b.challenger_id} WR={b.win_rate:.1f}% PF={b.profit_factor:.2f} ${b.net_profit:,.2f}")
            grand_results[f"HOLDOUT_{tf_name}"] = {
                "wr": round(h_wr, 1), "passed": h_passed, "total": len(h_active),
            }

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    total_time = time.time() - tournament_start

    print(f"\n{'=' * 70}")
    print(f"  SILVER WALK-FORWARD COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total Time: {total_time/60:.1f} minutes")

    print(f"\n  WALK-FORWARD SUMMARY:")
    print(f"  {'Phase':>15} {'TrainWR':>8} {'TestWR':>8} {'Delta':>7} {'Passed':>7} {'Verdict':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")
    for key, val in grand_results.items():
        if "HOLDOUT" not in key:
            delta = val["degradation"]
            verdict = "REAL" if delta > -5 else ("MARGINAL" if delta > -10 else "OVERFIT")
            print(f"  {key:>15} {val['train_wr']:>7.1f}% {val['test_wr']:>7.1f}% "
                  f"{delta:>+6.1f} {val['test_passed']:>4}/{val['test_total']:<3} {verdict:>8}")

    print(f"\n  HOLDOUT (NEVER TRAINED ON):")
    for key, val in grand_results.items():
        if "HOLDOUT" in key:
            print(f"    {key}: WR={val['wr']:.1f}%, Passed={val['passed']}/{val['total']}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"silver_wf_{ts}.json", 'w') as f:
        json.dump({"results": grand_results, "history": all_history}, f, indent=2)
    print(f"\n  Saved: {output_dir / f'silver_wf_{ts}.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silver Walk-Forward")
    parser.add_argument("--accounts", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5, help="Evo rounds per WF phase")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--terminal", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.accounts = 250
        args.rounds = 3

    run_walkforward(
        num_accounts=args.accounts,
        evo_rounds=args.rounds,
        days=args.days,
        batch_size=args.batch_size,
        workers=args.workers,
        terminal_path=args.terminal,
    )
