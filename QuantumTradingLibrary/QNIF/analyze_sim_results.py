"""
Simulation Results Analyzer
============================
Built by Biskits - Analyze QNIF 3-week simulation results.

When you've got 55 accounts and 2 symbols worth of data, you need a tool
to make sense of it. This is that tool.

Usage:
    python analyze_sim_results.py
    python analyze_sim_results.py --file sim_results/summary_20260210_040841.json
    python analyze_sim_results.py --top 10
    python analyze_sim_results.py --export-csv
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd


def load_latest_results(results_dir: Path) -> Dict[str, Any]:
    """Load the most recent simulation results."""
    summary_files = sorted(results_dir.glob("summary_*.json"), reverse=True)

    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {results_dir}")

    latest = summary_files[0]
    print(f"Loading results from: {latest.name}")

    with open(latest, 'r') as f:
        summary = json.load(f)

    # Load detailed results
    timestamp = latest.stem.replace('summary_', '')
    btc_file = results_dir / f"btc_sim_{timestamp}.json"
    eth_file = results_dir / f"eth_sim_{timestamp}.json"

    btc_results = []
    eth_results = []

    if btc_file.exists():
        with open(btc_file, 'r') as f:
            btc_results = json.load(f)

    if eth_file.exists():
        with open(eth_file, 'r') as f:
            eth_results = json.load(f)

    return {
        'summary': summary,
        'btc_results': btc_results,
        'eth_results': eth_results,
        'timestamp': timestamp
    }


def print_detailed_analysis(data: Dict[str, Any]):
    """Print comprehensive analysis."""
    summary = data['summary']
    btc_results = data['btc_results']
    eth_results = data['btc_results']

    print("\n" + "="*80)
    print("  QNIF 3-WEEK SIMULATION - DETAILED ANALYSIS")
    print("="*80)
    print(f"  Timestamp: {data['timestamp']}")
    print(f"  Days Simulated: {summary.get('simulation_days', 'N/A')}")
    print("="*80 + "\n")

    # BTC Analysis
    print("BTCUSD PERFORMANCE")
    print("-" * 80)
    btc = summary.get('btc', {})
    print(f"  Accounts Tested:    {btc.get('accounts', 0)}")
    print(f"  Challenges Passed:  {btc.get('passed', 0)} ({btc.get('passed', 0)/btc.get('accounts', 1)*100:.1f}%)")
    print(f"  Challenges Failed:  {btc.get('failed', 0)}")
    print(f"  Total Profit/Loss:  ${btc.get('total_profit', 0):,.2f}")
    print(f"  Average Win Rate:   {btc.get('avg_win_rate', 0):.1f}%")
    print(f"  Total Trades:       {btc.get('total_trades', 0)}")
    print()

    # ETH Analysis
    print("ETHUSD PERFORMANCE")
    print("-" * 80)
    eth = summary.get('eth', {})
    print(f"  Accounts Tested:    {eth.get('accounts', 0)}")
    print(f"  Challenges Passed:  {eth.get('passed', 0)} ({eth.get('passed', 0)/eth.get('accounts', 1)*100:.1f}%)")
    print(f"  Challenges Failed:  {eth.get('failed', 0)}")
    print(f"  Total Profit/Loss:  ${eth.get('total_profit', 0):,.2f}")
    print(f"  Average Win Rate:   {eth.get('avg_win_rate', 0):.1f}%")
    print(f"  Total Trades:       {eth.get('total_trades', 0)}")
    print()

    # Combined Stats
    total_accounts = btc.get('accounts', 0) + eth.get('accounts', 0)
    total_passed = btc.get('passed', 0) + eth.get('passed', 0)
    total_profit = btc.get('total_profit', 0) + eth.get('total_profit', 0)

    print("COMBINED STATISTICS")
    print("-" * 80)
    print(f"  Total Simulations:  {total_accounts}")
    print(f"  Overall Pass Rate:  {total_passed/total_accounts*100:.1f}%" if total_accounts > 0 else "  N/A")
    print(f"  Total P/L:          ${total_profit:,.2f}")
    print(f"  Avg P/L per Acct:   ${total_profit/total_accounts:,.2f}" if total_accounts > 0 else "  N/A")
    print()


def print_top_performers(data: Dict[str, Any], top_n: int = 10):
    """Print top performing accounts."""
    btc_results = data['btc_results']
    eth_results = data['eth_results']

    all_results = btc_results + eth_results

    # Sort by net profit
    sorted_by_profit = sorted(all_results, key=lambda x: x.get('net_profit', 0), reverse=True)[:top_n]

    # Sort by win rate
    sorted_by_wr = sorted(all_results, key=lambda x: x.get('win_rate', 0), reverse=True)[:top_n]

    # Sort by profit factor
    sorted_by_pf = sorted(all_results, key=lambda x: x.get('profit_factor', 0), reverse=True)[:top_n]

    print("TOP PERFORMERS BY NET PROFIT")
    print("-" * 80)
    print(f"{'Rank':<6}{'Symbol':<10}{'Account':<12}{'Net P/L':<15}{'Win Rate':<12}{'PF':<8}{'Status'}")
    print("-" * 80)
    for i, r in enumerate(sorted_by_profit, 1):
        status = 'PASS' if r.get('challenge_passed') else 'FAIL'
        print(f"{i:<6}{r.get('symbol', 'N/A'):<10}#{r.get('account_id', 0):<11}"
              f"${r.get('net_profit', 0):>9,.2f}   "
              f"{r.get('win_rate', 0):>5.1f}%     "
              f"{r.get('profit_factor', 0):>5.2f}  "
              f"{status}")
    print()

    print("TOP PERFORMERS BY WIN RATE")
    print("-" * 80)
    print(f"{'Rank':<6}{'Symbol':<10}{'Account':<12}{'Win Rate':<12}{'Net P/L':<15}{'Trades':<10}{'Status'}")
    print("-" * 80)
    for i, r in enumerate(sorted_by_wr, 1):
        status = 'PASS' if r.get('challenge_passed') else 'FAIL'
        print(f"{i:<6}{r.get('symbol', 'N/A'):<10}#{r.get('account_id', 0):<11}"
              f"{r.get('win_rate', 0):>5.1f}%     "
              f"${r.get('net_profit', 0):>9,.2f}   "
              f"{r.get('total_trades', 0):<10}"
              f"{status}")
    print()

    print("TOP PERFORMERS BY PROFIT FACTOR")
    print("-" * 80)
    print(f"{'Rank':<6}{'Symbol':<10}{'Account':<12}{'PF':<8}{'Net P/L':<15}{'Win Rate':<12}{'Status'}")
    print("-" * 80)
    for i, r in enumerate(sorted_by_pf, 1):
        status = 'PASS' if r.get('challenge_passed') else 'FAIL'
        pf = r.get('profit_factor', 0)
        pf_str = f"{pf:.2f}" if pf < 999 else "INF"
        print(f"{i:<6}{r.get('symbol', 'N/A'):<10}#{r.get('account_id', 0):<11}"
              f"{pf_str:>5}   "
              f"${r.get('net_profit', 0):>9,.2f}   "
              f"{r.get('win_rate', 0):>5.1f}%     "
              f"{status}")
    print()


def print_failure_analysis(data: Dict[str, Any]):
    """Analyze why accounts failed."""
    btc_results = data['btc_results']
    eth_results = data['eth_results']

    all_results = btc_results + eth_results

    failed = [r for r in all_results if r.get('challenge_failed')]

    if not failed:
        print("FAILURE ANALYSIS")
        print("-" * 80)
        print("  No failures! All accounts passed or are still running.")
        print()
        return

    print("FAILURE ANALYSIS")
    print("-" * 80)
    print(f"  Total Failures: {len(failed)}/{len(all_results)} ({len(failed)/len(all_results)*100:.1f}%)")
    print()

    # Group by fail reason
    fail_reasons = {}
    for r in failed:
        reason = r.get('fail_reason', 'Unknown')
        if reason not in fail_reasons:
            fail_reasons[reason] = 0
        fail_reasons[reason] += 1

    print("  Failure Reasons:")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"    {reason}: {count} ({count/len(failed)*100:.1f}%)")
    print()

    # Worst performers
    sorted_failures = sorted(failed, key=lambda x: x.get('net_profit', 0))[:5]

    print("  Worst Performers:")
    print(f"  {'Symbol':<10}{'Account':<12}{'Loss':<15}{'Reason'}")
    print("  " + "-" * 70)
    for r in sorted_failures:
        print(f"  {r.get('symbol', 'N/A'):<10}#{r.get('account_id', 0):<11}"
              f"${r.get('net_profit', 0):>9,.2f}   "
              f"{r.get('fail_reason', 'Unknown')}")
    print()


def export_to_csv(data: Dict[str, Any], output_path: Path):
    """Export results to CSV for further analysis."""
    btc_results = data['btc_results']
    eth_results = data['eth_results']

    all_results = btc_results + eth_results

    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)

    print(f"Exported {len(all_results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="QNIF Simulation Results Analyzer")
    parser.add_argument('--file', type=str, default=None,
                       help='Specific summary file to analyze')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top performers to show (default: 10)')
    parser.add_argument('--export-csv', type=str, default=None,
                       help='Export results to CSV file')
    parser.add_argument('--failures', action='store_true',
                       help='Show detailed failure analysis')

    args = parser.parse_args()

    # Determine results directory
    results_dir = Path(__file__).parent / "sim_results"

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        print("Run qnif_3week_sim.py first to generate results.")
        return 1

    # Load results
    try:
        if args.file:
            # Load specific file
            with open(args.file, 'r') as f:
                summary = json.load(f)
            # Extract timestamp and load detailed results
            # (simplified - assumes file naming convention)
            data = {'summary': summary, 'btc_results': [], 'eth_results': [], 'timestamp': 'manual'}
        else:
            # Load latest
            data = load_latest_results(results_dir)
    except Exception as e:
        print(f"ERROR: Failed to load results: {e}")
        return 1

    # Print analysis
    print_detailed_analysis(data)
    print_top_performers(data, top_n=args.top)

    if args.failures:
        print_failure_analysis(data)

    # Export if requested
    if args.export_csv:
        output_path = Path(args.export_csv)
        export_to_csv(data, output_path)

    print("="*80)
    print("  Analysis complete.")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
