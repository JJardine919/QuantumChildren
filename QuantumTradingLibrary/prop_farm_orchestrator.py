"""
Prop Farm Orchestrator - Main Entry Point
==========================================
Coordinates the full signal farm operation:
  1. Spin up 100 simulated prop firm accounts
  2. Distribute across GPU pool
  3. Run 34-round training protocol (17 TFs x 2 cycles)
  4. Collect signals from all accounts
  5. Aggregate results and track win rates
  6. Decision gates after each cycle

This is the script you run to kick everything off.

Usage:
    python prop_farm_orchestrator.py
    python prop_farm_orchestrator.py --symbol BTCUSD --accounts 100
    python prop_farm_orchestrator.py --dry-run
    python prop_farm_orchestrator.py --train-only
    python prop_farm_orchestrator.py --sim-only

DO NOT SKIP STEPS. Flag user if steps should be skipped.
"""

import json
import time
import sys
import os
import argparse
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_loader import (
    MAX_LOSS_DOLLARS, TP_MULTIPLIER, ROLLING_SL_MULTIPLIER,
    DYNAMIC_TP_PERCENT, CONFIDENCE_THRESHOLD
)
from gpu_pool_manager import GPUPool, get_device, get_device_name
from prop_farm_simulator import PropFarmAccount, result_to_json, signals_to_json


# Load farm config
FARM_CONFIG_PATH = SCRIPT_DIR / "signal_farm_config.json"
with open(FARM_CONFIG_PATH) as f:
    FARM_CONFIG = json.load(f)


class SignalFarmOrchestrator:
    """
    Main orchestrator for the signal farm.

    Manages:
    - 100 simulated prop firm accounts
    - GPU pool for parallel computation
    - 34-round training protocol
    - Signal collection and aggregation
    - Decision gates and win rate tracking
    """

    def __init__(self, symbol="BTCUSD", num_accounts=100, dry_run=False):
        self.symbol = symbol
        self.num_accounts = num_accounts
        self.dry_run = dry_run

        # GPU pool
        gpu_config = FARM_CONFIG["GPU"]
        self.gpu_pool = GPUPool(max_concurrent=gpu_config["MAX_WORKERS_PER_GPU"])

        # Output directory
        self.output_dir = SCRIPT_DIR / FARM_CONFIG["SIGNALS"]["SIGNAL_OUTPUT_DIR"]
        self.output_dir.mkdir(exist_ok=True)

        # Simulation accounts
        sim_config = FARM_CONFIG["SIMULATION"]
        self.accounts = [
            PropFarmAccount(
                account_id=i + 1,
                symbol=symbol,
                balance=sim_config["ACCOUNT_BALANCE"],
                max_daily_dd_pct=sim_config["MAX_DAILY_DD_PCT"],
                max_total_dd_pct=sim_config["MAX_TOTAL_DD_PCT"],
                profit_target_pct=sim_config["PROFIT_TARGET_PCT"],
            )
            for i in range(num_accounts)
        ]

        # Results tracking
        self.all_results = []
        self.all_signals = []
        self.round_stats = []
        self._lock = threading.Lock()

    def print_banner(self):
        """Print startup banner."""
        protocol = FARM_CONFIG["PROTOCOL"]
        sim = FARM_CONFIG["SIMULATION"]

        print("=" * 70)
        print("  PROP FARM ORCHESTRATOR")
        print("  Signal Collection & Training Infrastructure")
        print("=" * 70)
        print(f"  Symbol:          {self.symbol}")
        print(f"  Accounts:        {self.num_accounts}")
        print(f"  GPU:             {get_device_name()}")
        print(f"  GPU Pool:        {self.gpu_pool}")
        print(f"  Protocol:        {protocol['NAME']}")
        print(f"  Timeframes:      {protocol['TOTAL_TIMEFRAMES']}")
        print(f"  Cycles/TF:       {protocol['CYCLES_PER_TIMEFRAME']}")
        print(f"  Total Rounds:    {protocol['TOTAL_ROUNDS']}")
        print(f"  Lookback:        {protocol['LOOKBACK_MONTHS']} months")
        print(f"  Train/Test:      {protocol['TRAIN_MONTHS']}mo / {protocol['TEST_MONTHS']}mo")
        print("-" * 70)
        print(f"  Account Balance: ${sim['ACCOUNT_BALANCE']:,.0f}")
        print(f"  Max Daily DD:    {sim['MAX_DAILY_DD_PCT']}%")
        print(f"  Max Total DD:    {sim['MAX_TOTAL_DD_PCT']}%")
        print(f"  Profit Target:   {sim['PROFIT_TARGET_PCT']}%")
        print("-" * 70)
        print(f"  TRADING SETTINGS (from MASTER_CONFIG):")
        print(f"    Max Loss:        ${MAX_LOSS_DOLLARS}")
        print(f"    TP Multiplier:   {TP_MULTIPLIER}x")
        print(f"    Rolling SL:      {ROLLING_SL_MULTIPLIER}x")
        print(f"    Dynamic TP:      {DYNAMIC_TP_PERCENT}%")
        print(f"    Confidence:      {CONFIDENCE_THRESHOLD}")
        print("=" * 70)

    def run_parallel_simulation(self, data, timeframe, cycle, phase,
                                 batch_size=None):
        """
        Run all accounts against the same data in parallel batches.

        Args:
            data: DataFrame with OHLCV data
            timeframe: timeframe name
            cycle: cycle number
            phase: "train" or "test"
            batch_size: accounts per batch (default from config)

        Returns:
            list of AccountResult
        """
        if batch_size is None:
            batch_size = FARM_CONFIG["SIMULATION"]["PARALLEL_BATCH_SIZE"]

        results = []
        total_batches = (self.num_accounts + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start = batch_num * batch_size
            end = min(start + batch_size, self.num_accounts)
            batch_accounts = self.accounts[start:end]

            batch_results = []

            def _run_account(account):
                return account.run_simulation(
                    df=data,
                    timeframe=timeframe,
                    cycle=cycle,
                    phase=phase,
                    contract_size=1.0,
                    point=0.01,
                )

            # Run batch in parallel threads
            with ThreadPoolExecutor(max_workers=min(8, len(batch_accounts))) as executor:
                futures = {
                    executor.submit(_run_account, acc): acc
                    for acc in batch_accounts
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        acc = futures[future]
                        print(f"    Account {acc.account_id} error: {e}")

            results.extend(batch_results)

            # Progress
            win_rates = [r.win_rate for r in batch_results if r.total_trades > 0]
            avg_wr = sum(win_rates) / len(win_rates) if win_rates else 0
            print(f"    Batch {batch_num+1}/{total_batches}: "
                  f"{len(batch_results)} accounts, avg WR={avg_wr:.1f}%")

        return results

    def collect_signals(self, results):
        """Collect and aggregate signals from all account results."""
        total_signals = 0
        for result in results:
            total_signals += result.signals_count
            with self._lock:
                self.all_signals.extend(result.signals[:100])  # Cap per account

        return total_signals

    def aggregate_results(self, results, timeframe, cycle, phase):
        """Aggregate results across all accounts for a round."""
        if not results:
            return {}

        active_results = [r for r in results if r.total_trades > 0]
        if not active_results:
            return {"warning": "No accounts had trades"}

        win_rates = [r.win_rate for r in active_results]
        profits = [r.net_profit for r in active_results]
        dd_pcts = [r.max_dd_pct for r in active_results]
        trade_counts = [r.total_trades for r in active_results]

        passed = sum(1 for r in active_results if r.challenge_passed)
        failed = sum(1 for r in active_results if r.challenge_failed)

        stats = {
            "timeframe": timeframe,
            "cycle": cycle,
            "phase": phase,
            "accounts_active": len(active_results),
            "accounts_total": len(results),
            "avg_win_rate": round(sum(win_rates) / len(win_rates), 1),
            "best_win_rate": round(max(win_rates), 1),
            "worst_win_rate": round(min(win_rates), 1),
            "median_win_rate": round(sorted(win_rates)[len(win_rates)//2], 1),
            "avg_profit": round(sum(profits) / len(profits), 2),
            "best_profit": round(max(profits), 2),
            "worst_profit": round(min(profits), 2),
            "avg_dd_pct": round(sum(dd_pcts) / len(dd_pcts), 2),
            "max_dd_pct": round(max(dd_pcts), 2),
            "avg_trades": round(sum(trade_counts) / len(trade_counts), 0),
            "challenge_passed": passed,
            "challenge_failed": failed,
            "pass_rate": round(passed / len(active_results) * 100, 1) if active_results else 0,
        }

        return stats

    def run_full_protocol(self):
        """
        Execute the complete 34-round protocol.

        For each of 17 timeframes:
          - Fetch 1 year of data
          - Run 2 cycles (4-month train + 2-month test each)
          - Cycle 2 jostles forward
          - Evaluate all 100 accounts per round
          - Collect signals
          - Check decision gates
        """
        import MetaTrader5 as mt5
        import pandas as pd

        if not mt5.initialize():
            print("ERROR: MT5 initialization failed!")
            return

        protocol = FARM_CONFIG["PROTOCOL"]
        timeframes = FARM_CONFIG["TIMEFRAMES"]

        TF_MAP = {
            "M1":  mt5.TIMEFRAME_M1,  "M2":  mt5.TIMEFRAME_M2,
            "M3":  mt5.TIMEFRAME_M3,  "M4":  mt5.TIMEFRAME_M4,
            "M5":  mt5.TIMEFRAME_M5,  "M6":  mt5.TIMEFRAME_M6,
            "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
            "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20,
            "M30": mt5.TIMEFRAME_M30, "H1":  mt5.TIMEFRAME_H1,
            "H2":  mt5.TIMEFRAME_H2,  "H3":  mt5.TIMEFRAME_H3,
            "H4":  mt5.TIMEFRAME_H4,  "H6":  mt5.TIMEFRAME_H6,
            "H8":  mt5.TIMEFRAME_H8,
        }

        overall_start = time.time()
        round_num = 0

        for tf_info in timeframes:
            tf_name = tf_info["name"]
            tf_const = TF_MAP[tf_name]
            bars_per_day = tf_info["bars_per_day"]
            bars_needed = bars_per_day * 30 * protocol["LOOKBACK_MONTHS"]
            bars_needed = min(bars_needed, 500000)

            print(f"\n{'#' * 70}")
            print(f"# TIMEFRAME: {tf_name} ({tf_info['order']}/{len(timeframes)})")
            print(f"{'#' * 70}")

            # Fetch data
            print(f"  Fetching {bars_needed:,} bars of {self.symbol} {tf_name}...")
            rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, bars_needed)
            if rates is None or len(rates) < 500:
                print(f"  ERROR: Not enough data for {tf_name}")
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            print(f"  Got {len(df):,} bars: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

            bars_per_month = bars_per_day * 30
            train_bars = bars_per_month * protocol["TRAIN_MONTHS"]
            test_bars = bars_per_month * protocol["TEST_MONTHS"]

            for cycle in range(1, protocol["CYCLES_PER_TIMEFRAME"] + 1):
                round_num += 1

                # Cycle 2 shifts forward by 1 month
                offset = bars_per_month * (cycle - 1)

                train_start = offset
                train_end = train_start + train_bars
                test_start = train_end
                test_end = test_start + test_bars

                if test_end > len(df):
                    available = len(df) - train_start
                    train_bars_actual = int(available * 0.67)
                    test_bars_actual = available - train_bars_actual
                    train_end = train_start + train_bars_actual
                    test_start = train_end
                    test_end = test_start + test_bars_actual

                if train_end - train_start < 200 or test_end - test_start < 100:
                    print(f"  SKIP: Not enough data for {tf_name} cycle {cycle}")
                    continue

                train_df = df.iloc[train_start:train_end].copy()
                test_df = df.iloc[test_start:test_end].copy()

                print(f"\n  --- Round {round_num}/{protocol['TOTAL_ROUNDS']}: "
                      f"{tf_name} Cycle {cycle} ---")
                print(f"  Train: {len(train_df):,} bars | "
                      f"Test: {len(test_df):,} bars")

                round_start = time.time()

                # Run simulation on TRAIN data
                print(f"\n  TRAINING PHASE ({len(train_df):,} bars, {self.num_accounts} accounts):")
                train_results = self.run_parallel_simulation(
                    train_df, tf_name, cycle, "train"
                )
                train_stats = self.aggregate_results(train_results, tf_name, cycle, "train")

                # Run simulation on TEST data
                print(f"\n  TESTING PHASE ({len(test_df):,} bars, {self.num_accounts} accounts):")
                test_results = self.run_parallel_simulation(
                    test_df, tf_name, cycle, "test"
                )
                test_stats = self.aggregate_results(test_results, tf_name, cycle, "test")

                # Collect signals
                train_sigs = self.collect_signals(train_results)
                test_sigs = self.collect_signals(test_results)

                round_elapsed = time.time() - round_start

                # Print round summary
                print(f"\n  ROUND {round_num} SUMMARY:")
                print(f"    Train: avg WR={train_stats.get('avg_win_rate', 0):.1f}% "
                      f"best={train_stats.get('best_win_rate', 0):.1f}% "
                      f"pass={train_stats.get('challenge_passed', 0)}/{self.num_accounts}")
                print(f"    Test:  avg WR={test_stats.get('avg_win_rate', 0):.1f}% "
                      f"best={test_stats.get('best_win_rate', 0):.1f}% "
                      f"pass={test_stats.get('challenge_passed', 0)}/{self.num_accounts}")
                print(f"    Signals: {train_sigs + test_sigs:,} collected")
                print(f"    Time: {round_elapsed:.1f}s")

                # Save round stats
                self.round_stats.append({
                    "round": round_num,
                    "timeframe": tf_name,
                    "cycle": cycle,
                    "train": train_stats,
                    "test": test_stats,
                    "signals_collected": train_sigs + test_sigs,
                    "elapsed_s": round(round_elapsed, 1),
                })

                # Save checkpoint
                self._save_checkpoint(round_num)

                # Store results
                self.all_results.extend(train_results)
                self.all_results.extend(test_results)

            # Decision gate after first pass through all TFs
            if round_num == len(timeframes):
                self._check_decision_gate()

        total_elapsed = time.time() - overall_start

        mt5.shutdown()

        # Final report
        self._print_final_report(total_elapsed)
        self._save_final_report(total_elapsed)

    def _save_checkpoint(self, round_num):
        """Save checkpoint after each round."""
        path = self.output_dir / "orchestrator_checkpoint.json"
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "round": round_num,
            "total_results": len(self.all_results),
            "total_signals": len(self.all_signals),
            "round_stats": self.round_stats,
        }
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _check_decision_gate(self):
        """Check decision gate after first cycle."""
        if len(self.round_stats) < 2:
            return

        test_wrs = [
            s["test"].get("avg_win_rate", 0)
            for s in self.round_stats
            if "test" in s and isinstance(s["test"], dict)
        ]
        if not test_wrs:
            return

        first_wr = test_wrs[0]
        last_wr = test_wrs[-1]
        avg_wr = sum(test_wrs) / len(test_wrs)

        print(f"\n{'*' * 70}")
        print(f"  DECISION GATE - First Cycle Complete")
        print(f"{'*' * 70}")
        print(f"  First TF avg WR:  {first_wr:.1f}%")
        print(f"  Latest TF avg WR: {last_wr:.1f}%")
        print(f"  Overall avg WR:   {avg_wr:.1f}%")
        print(f"  Trend:            {last_wr - first_wr:+.1f}%")

        gate = FARM_CONFIG["DECISION_GATES"]["AFTER_FIRST_CYCLE"]
        if last_wr - first_wr < gate["MIN_WIN_RATE_IMPROVEMENT"]:
            print(f"  FLAG: Improvement below {gate['MIN_WIN_RATE_IMPROVEMENT']}%")
            print(f"  Recommend: Review before continuing")
        else:
            print(f"  PASS: Win rate improving")
        print(f"{'*' * 70}")

    def _print_final_report(self, total_time):
        """Print comprehensive final report."""
        print(f"\n{'=' * 70}")
        print(f"  FINAL REPORT - PROP FARM ORCHESTRATOR")
        print(f"{'=' * 70}")
        print(f"  Symbol:         {self.symbol}")
        print(f"  Accounts:       {self.num_accounts}")
        print(f"  Total Time:     {total_time/60:.1f} minutes")
        print(f"  Rounds:         {len(self.round_stats)}")
        print(f"  Total Results:  {len(self.all_results)}")
        print(f"  Total Signals:  {len(self.all_signals):,}")

        if self.round_stats:
            print(f"\n  ROUND-BY-ROUND:")
            print(f"  {'Round':>5} {'TF':>5} {'Cyc':>3} {'Train%':>7} {'Test%':>7} {'Pass':>5} {'Sigs':>8} {'Time':>6}")
            print(f"  {'-'*5} {'-'*5} {'-'*3} {'-'*7} {'-'*7} {'-'*5} {'-'*8} {'-'*6}")

            for s in self.round_stats:
                train_wr = s["train"].get("avg_win_rate", 0) if isinstance(s["train"], dict) else 0
                test_wr = s["test"].get("avg_win_rate", 0) if isinstance(s["test"], dict) else 0
                passed = s["test"].get("challenge_passed", 0) if isinstance(s["test"], dict) else 0
                print(f"  {s['round']:>5} {s['timeframe']:>5} {s['cycle']:>3} "
                      f"{train_wr:>6.1f}% {test_wr:>6.1f}% "
                      f"{passed:>5} {s['signals_collected']:>8,} {s['elapsed_s']:>5.1f}s")

            # Overall averages
            test_wrs = [
                s["test"].get("avg_win_rate", 0)
                for s in self.round_stats
                if isinstance(s.get("test"), dict)
            ]
            if test_wrs:
                print(f"\n  Overall avg test WR: {sum(test_wrs)/len(test_wrs):.1f}%")
                print(f"  Best test WR:        {max(test_wrs):.1f}%")

        print(f"\n  GPU Pool Final State: {self.gpu_pool}")
        print(f"{'=' * 70}")

    def _save_final_report(self, total_time):
        """Save final report to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"orchestrator_report_{timestamp}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "symbol": self.symbol,
            "num_accounts": self.num_accounts,
            "total_time_minutes": round(total_time / 60, 1),
            "total_results": len(self.all_results),
            "total_signals": len(self.all_signals),
            "gpu": get_device_name(),
            "gpu_pool": self.gpu_pool.status(),
            "round_stats": self.round_stats,
            "protocol": FARM_CONFIG["PROTOCOL"],
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n  Report saved: {path}")

        # Save signals summary
        if self.all_signals:
            sig_path = self.output_dir / f"signals_{timestamp}.json"
            sig_data = signals_to_json(self.all_signals[:10000])  # Cap at 10K
            with open(sig_path, 'w') as f:
                json.dump(sig_data, f)
            print(f"  Signals saved: {sig_path} ({len(sig_data)} entries)")


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prop Farm Orchestrator - Signal Collection & Training"
    )
    parser.add_argument("--symbol", default="BTCUSD",
                       help="Symbol to focus on (default: BTCUSD)")
    parser.add_argument("--accounts", type=int, default=100,
                       help="Number of simulated accounts (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show configuration without running")
    parser.add_argument("--train-only", action="store_true",
                       help="Run genetic training only (signal_farm_trainer.py)")
    parser.add_argument("--sim-only", action="store_true",
                       help="Run account simulation only (no genetic training)")
    args = parser.parse_args()

    if args.train_only:
        # Delegate to signal_farm_trainer
        from signal_farm_trainer import run_signal_farm_training
        run_signal_farm_training(symbol=args.symbol, dry_run=args.dry_run)
        return

    orchestrator = SignalFarmOrchestrator(
        symbol=args.symbol,
        num_accounts=args.accounts,
        dry_run=args.dry_run,
    )

    orchestrator.print_banner()

    if args.dry_run:
        print("\n  [DRY RUN] Would execute the above. Exiting.")
        return

    orchestrator.run_full_protocol()


if __name__ == "__main__":
    main()
