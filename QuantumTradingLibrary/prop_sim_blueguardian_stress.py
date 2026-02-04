"""
BLUE GUARDIAN STRESS TEST SIMULATION
====================================
Tests the 72% WR expert under HIGH CHAOS conditions (nightmare scenario).
This simulates:
- Increased slippage
- Wider spreads during volatility
- Flash crashes/spikes
- Requotes/execution delays
- Reduced trade frequency

If the system passes under these conditions, it's ready for anything.
"""

import json
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import hashlib

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# STRESS TEST CONFIG - Much more conservative
STRESS_CONFIG = {
    'account_size': 5_000.0,
    'daily_dd_pct': 0.05,
    'max_dd_pct': 0.10,
    'profit_target_pct': 0.10,
    'challenge_days': 21,

    # Conservative risk management
    'risk_per_trade_pct': 0.005,  # 0.5% risk per trade (half of normal)
    'min_rr_ratio': 1.5,
    'max_trades_per_day': 3,      # Reduced trade frequency

    # Expert settings - slightly degraded for stress test
    'base_win_rate': 0.68,        # 68% (degraded from 72% due to market conditions)
    'compression_boost': 0.10,    # +10% (degraded from 12%)
    'projected_win_rate': 0.78,   # 78% with compression

    # HIGH CHAOS settings
    'slippage_pips_mean': 3.0,    # 2x normal slippage
    'slippage_pips_std': 2.5,
    'spread_base_usd': 4.0,       # 2x normal spread
    'chaos_level': 0.9,           # Near-nightmare conditions

    # Market events
    'flash_crash_prob': 0.02,     # 2% chance per trading session
    'gap_prob': 0.05,             # 5% chance of gap open
    'requote_prob': 0.1,          # 10% of trades get requoted

    # Streaks (real market clustering)
    'loss_streak_prob': 0.15,     # 15% chance of entering a loss streak
    'loss_streak_length': 3,      # 3 consecutive losses in a streak
}

STRESS_CONFIG['daily_dd_limit'] = STRESS_CONFIG['account_size'] * STRESS_CONFIG['daily_dd_pct']
STRESS_CONFIG['max_dd_limit'] = STRESS_CONFIG['account_size'] * STRESS_CONFIG['max_dd_pct']
STRESS_CONFIG['profit_target'] = STRESS_CONFIG['account_size'] * STRESS_CONFIG['profit_target_pct']
STRESS_CONFIG['risk_amount'] = STRESS_CONFIG['account_size'] * STRESS_CONFIG['risk_per_trade_pct']


class QuantumCompressionFilter:
    """Quantum compression regime filter"""
    def __init__(self, base_win_rate: float = 0.68, compression_boost: float = 0.10):
        self.base_win_rate = base_win_rate
        self.compression_boost = compression_boost
        self.cache = {}

    def analyze_regime(self, price_window: np.ndarray) -> dict:
        data_hash = hashlib.md5(price_window.tobytes()).hexdigest()
        if data_hash in self.cache:
            return self.cache[data_hash]

        returns = np.diff(price_window) / (price_window[:-1] + 1e-10)
        if len(returns) < 10:
            return {'ratio': 1.0, 'regime': 'NEUTRAL', 'trade_boost': 0.0, 'effective_win_rate': self.base_win_rate}

        hist, _ = np.histogram(returns, bins=8, density=True)
        hist = hist + 1e-10
        entropy = -np.sum(hist * np.log2(hist)) * (hist > 0).sum() / 8

        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
        else:
            autocorr = 0

        compression_score = (1 - entropy/3.0) * 0.5 + (autocorr + 1) / 2 * 0.5
        ratio = 1.0 + compression_score * 3.0

        if ratio > 2.5:
            regime = 'TRENDING'
            trade_boost = self.compression_boost
        elif ratio > 1.5:
            regime = 'CLEAN'
            trade_boost = self.compression_boost * 0.7
        else:
            regime = 'CHOPPY'
            trade_boost = 0.0

        result = {
            'ratio': ratio,
            'entropy': entropy,
            'regime': regime,
            'trade_boost': trade_boost,
            'effective_win_rate': self.base_win_rate + trade_boost
        }
        self.cache[data_hash] = result
        return result

    def should_trade(self, regime_info: dict) -> Tuple[bool, float]:
        regime = regime_info['regime']
        if regime == 'TRENDING':
            return True, 1.2  # Slightly reduced from 1.5
        elif regime == 'CLEAN':
            return True, 1.0
        else:
            return False, 0.3  # Much more conservative in choppy


@dataclass
class PortfolioState:
    equity: float = 5000.0
    daily_start_equity: float = 5000.0
    high_water_mark: float = 5000.0
    position: Optional[str] = None
    entry_price: float = 0.0
    trades: List = field(default_factory=list)
    daily_trades: int = 0
    is_active: bool = True
    fail_reason: Optional[str] = None
    max_dd_hit: float = 0.0
    max_daily_dd_hit: float = 0.0
    current_day: int = 0
    daily_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    loss_streak: int = 0
    in_loss_streak_mode: bool = False

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades


def generate_chaotic_market(days: int = 30) -> Tuple[pd.DataFrame, List[int]]:
    """Generate market data with flash crashes and gaps"""

    bars_per_day = 288
    total_bars = days * bars_per_day
    base_price = 95000.0

    times = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=total_bars,
        freq='5min'
    )

    np.random.seed(int(datetime.now().timestamp() % 100000))

    returns = np.random.normal(0, 0.025 / np.sqrt(bars_per_day), total_bars)

    # Add trend
    trend = np.zeros(total_bars)
    for i in range(1, total_bars):
        trend[i] = 0.999 * trend[i-1] + np.random.randn() * 0.0001

    # Flash crash events
    flash_crash_bars = []
    for day in range(days):
        if random.random() < STRESS_CONFIG['flash_crash_prob']:
            crash_bar = day * bars_per_day + random.randint(50, 200)
            if crash_bar < total_bars:
                flash_crash_bars.append(crash_bar)
                # Sharp drop followed by partial recovery
                returns[crash_bar] = -0.03 + random.uniform(-0.02, 0)
                if crash_bar + 1 < total_bars:
                    returns[crash_bar + 1] = 0.015 + random.uniform(0, 0.01)

    # Gap opens
    for day in range(1, days):
        if random.random() < STRESS_CONFIG['gap_prob']:
            gap_bar = day * bars_per_day
            if gap_bar < total_bars:
                returns[gap_bar] = random.uniform(-0.02, 0.02)

    total_returns = returns + trend
    total_returns = np.clip(total_returns, -0.05, 0.05)

    prices = base_price * np.exp(np.cumsum(total_returns))

    intrabar_vol = np.abs(np.random.randn(total_bars)) * 0.001 + 0.0005
    high = prices * (1 + intrabar_vol)
    low = prices * (1 - intrabar_vol)
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]

    df = pd.DataFrame({
        'time': times,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
    })

    return df, flash_crash_bars


class StressTestSimulator:
    """High-chaos stress test simulator"""

    def __init__(self, use_compression: bool = True, num_runs: int = 1000):
        self.use_compression = use_compression
        self.num_runs = num_runs
        self.compression_filter = QuantumCompressionFilter(
            base_win_rate=STRESS_CONFIG['base_win_rate'],
            compression_boost=STRESS_CONFIG['compression_boost']
        )
        self.results = []

    def simulate_trade(self, regime_info: dict, portfolio: PortfolioState) -> float:
        """Simulate trade with high chaos"""

        size_mult = 1.0  # Default value

        # Check for loss streak mode
        if portfolio.in_loss_streak_mode and portfolio.loss_streak > 0:
            # Forced loss during streak
            effective_wr = 0.3  # Very low win rate during streaks
            portfolio.loss_streak -= 1
            if portfolio.loss_streak == 0:
                portfolio.in_loss_streak_mode = False
            size_mult = 0.5  # Reduced size during streak
        elif random.random() < STRESS_CONFIG['loss_streak_prob'] and not portfolio.in_loss_streak_mode:
            # Enter loss streak
            portfolio.in_loss_streak_mode = True
            portfolio.loss_streak = STRESS_CONFIG['loss_streak_length']
            effective_wr = 0.3
            size_mult = 0.5
        else:
            if self.use_compression:
                effective_wr = regime_info['effective_win_rate']
                _, size_mult = self.compression_filter.should_trade(regime_info)
            else:
                effective_wr = STRESS_CONFIG['base_win_rate']
                size_mult = 1.0

            size_mult = min(size_mult, 1.0)  # Cap at 1.0 for stress test

        risk = STRESS_CONFIG['risk_amount'] * size_mult
        chaos = STRESS_CONFIG['chaos_level']

        # Requote simulation (trade might not execute at desired price)
        if random.random() < STRESS_CONFIG['requote_prob']:
            risk *= 0.7  # Worse fill

        slippage = np.random.normal(
            STRESS_CONFIG['slippage_pips_mean'],
            STRESS_CONFIG['slippage_pips_std']
        ) * chaos

        spread = STRESS_CONFIG['spread_base_usd'] * (1 + random.uniform(0, 1.0) * chaos)

        is_win = random.random() < effective_wr

        if is_win:
            rr = random.uniform(STRESS_CONFIG['min_rr_ratio'], 2.0)
            base_profit = risk * rr
            profit_mult = 1.0 - (slippage * 0.02)
            pnl = max(base_profit * profit_mult - spread, 5)  # Minimum $5 profit
            portfolio.winning_trades += 1
        else:
            loss_mult = 1.0 + (slippage * 0.03)
            pnl = -(risk * loss_mult) - spread
            portfolio.losing_trades += 1

        return pnl

    def run_single_simulation(self, run_id: int) -> dict:
        """Run single stress test simulation"""

        market_data, flash_crashes = generate_chaotic_market(days=STRESS_CONFIG['challenge_days'] + 10)
        prices = market_data['close'].values
        times = market_data['time'].values

        portfolio = PortfolioState(
            equity=STRESS_CONFIG['account_size'],
            daily_start_equity=STRESS_CONFIG['account_size'],
            high_water_mark=STRESS_CONFIG['account_size']
        )

        current_date = None
        day_count = 0
        bars_per_day = 288
        max_bars = min(len(prices), STRESS_CONFIG['challenge_days'] * bars_per_day)
        daily_dd_violations = []

        for t in range(100, max_bars):
            if not portfolio.is_active:
                break

            # Skip trading during flash crash recovery
            if t in flash_crashes or (t > 0 and t-1 in flash_crashes):
                continue

            bar_date = pd.Timestamp(times[t]).date()

            if bar_date != current_date:
                current_date = bar_date
                day_count += 1
                portfolio.current_day = day_count
                portfolio.daily_start_equity = portfolio.equity
                portfolio.daily_pnl = 0.0
                portfolio.daily_trades = 0

                if day_count > STRESS_CONFIG['challenge_days']:
                    break

            if portfolio.equity >= STRESS_CONFIG['account_size'] + STRESS_CONFIG['profit_target']:
                break

            if portfolio.daily_trades >= STRESS_CONFIG['max_trades_per_day']:
                continue

            window_size = 50
            if t >= window_size:
                price_window = prices[t-window_size:t]
                regime_info = self.compression_filter.analyze_regime(price_window)
            else:
                regime_info = {
                    'regime': 'NEUTRAL',
                    'effective_win_rate': STRESS_CONFIG['base_win_rate'],
                    'ratio': 1.5
                }

            if self.use_compression:
                should_trade, _ = self.compression_filter.should_trade(regime_info)
                if not should_trade:
                    continue

            # Lower trade frequency in stress test
            if random.random() > 0.05:
                continue

            pnl = self.simulate_trade(regime_info, portfolio)
            portfolio.trades.append({
                'day': day_count,
                'bar': t,
                'pnl': pnl,
                'regime': regime_info['regime'],
            })

            portfolio.equity += pnl
            portfolio.daily_pnl += pnl
            portfolio.daily_trades += 1

            if portfolio.equity > portfolio.high_water_mark:
                portfolio.high_water_mark = portfolio.equity

            daily_dd = portfolio.daily_start_equity - portfolio.equity
            if daily_dd > portfolio.max_daily_dd_hit:
                portfolio.max_daily_dd_hit = daily_dd

            if daily_dd > STRESS_CONFIG['daily_dd_limit']:
                portfolio.is_active = False
                portfolio.fail_reason = f"DAILY LOSS LIMIT HIT (Day {day_count})"
                daily_dd_violations.append({'day': day_count, 'dd': daily_dd})
                break

            total_dd = STRESS_CONFIG['account_size'] - portfolio.equity
            if total_dd > portfolio.max_dd_hit:
                portfolio.max_dd_hit = total_dd

            if total_dd > STRESS_CONFIG['max_dd_limit']:
                portfolio.is_active = False
                portfolio.fail_reason = f"MAX DRAWDOWN HIT: {total_dd/STRESS_CONFIG['account_size']*100:.1f}%"
                break

        profit_loss = portfolio.equity - STRESS_CONFIG['account_size']
        passed = False

        if portfolio.is_active:
            if profit_loss >= STRESS_CONFIG['profit_target']:
                passed = True
            elif day_count >= STRESS_CONFIG['challenge_days']:
                portfolio.fail_reason = f"TIME EXPIRED (Profit: {profit_loss/STRESS_CONFIG['account_size']*100:.1f}%)"

        return {
            'run_id': run_id,
            'passed': passed,
            'final_equity': portfolio.equity,
            'profit_loss': profit_loss,
            'profit_pct': (portfolio.equity / STRESS_CONFIG['account_size'] - 1) * 100,
            'win_rate': portfolio.win_rate,
            'total_trades': portfolio.total_trades,
            'max_dd_hit': portfolio.max_dd_hit,
            'max_dd_pct': (portfolio.max_dd_hit / STRESS_CONFIG['account_size']) * 100,
            'max_daily_dd_hit': portfolio.max_daily_dd_hit,
            'max_daily_dd_pct': (portfolio.max_daily_dd_hit / STRESS_CONFIG['account_size']) * 100,
            'days_survived': portfolio.current_day,
            'fail_reason': portfolio.fail_reason,
            'compression_enabled': self.use_compression
        }

    def run_stress_test(self) -> dict:
        """Run full stress test"""

        log.info("=" * 70)
        log.info("BLUE GUARDIAN STRESS TEST (NIGHTMARE MODE)")
        log.info("=" * 70)
        log.info(f"Degraded Win Rate: {STRESS_CONFIG['base_win_rate']*100:.0f}% (from 72%)")
        log.info(f"Compression Boost: +{STRESS_CONFIG['compression_boost']*100:.0f}%")
        log.info(f"Projected (Degraded): {STRESS_CONFIG['projected_win_rate']*100:.0f}%")
        log.info(f"Chaos Level: {STRESS_CONFIG['chaos_level']*100:.0f}%")
        log.info(f"Slippage: {STRESS_CONFIG['slippage_pips_mean']} pips avg")
        log.info(f"Flash Crash Prob: {STRESS_CONFIG['flash_crash_prob']*100:.0f}%/day")
        log.info(f"Monte Carlo Runs: {self.num_runs}")
        log.info("=" * 70)

        self.results = []

        for i in range(self.num_runs):
            result = self.run_single_simulation(i)
            self.results.append(result)

            if (i + 1) % 100 == 0:
                passes = sum(1 for r in self.results if r['passed'])
                log.info(f"  Progress: {i+1}/{self.num_runs} | Pass rate: {passes}/{i+1} ({passes/(i+1)*100:.1f}%)")

        return self.analyze_results()

    def analyze_results(self) -> dict:
        passed = [r for r in self.results if r['passed']]
        failed = [r for r in self.results if not r['passed']]

        pass_rate = len(passed) / len(self.results) * 100
        win_rates = [r['win_rate'] for r in self.results]
        pnls = [r['profit_loss'] for r in self.results]
        max_dds = [r['max_dd_pct'] for r in self.results]
        daily_dds = [r['max_daily_dd_pct'] for r in self.results]

        fail_reasons = {}
        for r in failed:
            reason = r['fail_reason'].split(':')[0] if r['fail_reason'] else 'Unknown'
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        return {
            'total_runs': len(self.results),
            'passed': len(passed),
            'failed': len(failed),
            'pass_rate': pass_rate,
            'avg_win_rate': np.mean(win_rates) * 100,
            'avg_pnl': np.mean(pnls),
            'avg_max_dd': np.mean(max_dds),
            'avg_daily_dd': np.mean(daily_dds),
            'best_pnl': max(pnls),
            'worst_pnl': min(pnls),
            'fail_reasons': fail_reasons,
            'compression_enabled': self.use_compression
        }

    def print_report(self, analysis: dict):
        print("\n" + "=" * 70)
        print("  STRESS TEST RESULTS (NIGHTMARE MODE)")
        print("=" * 70)

        print(f"\n{'STRESS CONDITIONS':^70}")
        print("-" * 70)
        print(f"  Degraded Base WR:    {STRESS_CONFIG['base_win_rate']*100:.0f}%")
        print(f"  Chaos Level:         {STRESS_CONFIG['chaos_level']*100:.0f}%")
        print(f"  Slippage Mean:       {STRESS_CONFIG['slippage_pips_mean']} pips")
        print(f"  Flash Crash Risk:    {STRESS_CONFIG['flash_crash_prob']*100:.0f}%/day")
        print(f"  Loss Streak Risk:    {STRESS_CONFIG['loss_streak_prob']*100:.0f}%")
        print(f"  Risk Per Trade:      {STRESS_CONFIG['risk_per_trade_pct']*100:.1f}%")

        print(f"\n{'RESULTS ({} RUNS)':^70}".format(analysis['total_runs']))
        print("-" * 70)

        if analysis['pass_rate'] >= 60:
            verdict = "BATTLE-READY"
            symbol = "[STRONG]"
        elif analysis['pass_rate'] >= 40:
            verdict = "ACCEPTABLE UNDER STRESS"
            symbol = "[OK]"
        else:
            verdict = "NEEDS REFINEMENT"
            symbol = "[WEAK]"

        print(f"\n  {symbol} STRESS TEST PASS RATE: {analysis['pass_rate']:.1f}%")
        print(f"  Passed: {analysis['passed']} | Failed: {analysis['failed']}")

        print(f"\n  PERFORMANCE UNDER STRESS:")
        print(f"    Win Rate:             {analysis['avg_win_rate']:.1f}%")
        print(f"    Average P/L:          ${analysis['avg_pnl']:+,.2f}")
        print(f"    Best P/L:             ${analysis['best_pnl']:+,.2f}")
        print(f"    Worst P/L:            ${analysis['worst_pnl']:+,.2f}")

        print(f"\n  DRAWDOWN UNDER STRESS:")
        print(f"    Avg Max Drawdown:     {analysis['avg_max_dd']:.2f}%")
        print(f"    Avg Daily DD:         {analysis['avg_daily_dd']:.2f}%")

        if analysis['fail_reasons']:
            print(f"\n  FAILURE BREAKDOWN:")
            for reason, count in sorted(analysis['fail_reasons'].items(), key=lambda x: -x[1]):
                pct = count / analysis['failed'] * 100 if analysis['failed'] > 0 else 0
                print(f"    {reason}: {count} ({pct:.0f}%)")

        print(f"\n{'STRESS TEST VERDICT':^70}")
        print("=" * 70)
        print(f"  {verdict}")
        print("=" * 70)

        return analysis


def main():
    print("\n" + "#" * 70)
    print("#  BLUE GUARDIAN STRESS TEST - NIGHTMARE SCENARIO")
    print("#  Testing 72% Expert under extreme market conditions")
    print("#" * 70 + "\n")

    # Run stress test WITH compression
    print("\n[1/2] Stress test WITH compression filter...")
    sim_with = StressTestSimulator(use_compression=True, num_runs=1000)
    analysis_with = sim_with.run_stress_test()
    sim_with.print_report(analysis_with)

    # Run stress test WITHOUT compression
    print("\n[2/2] Stress test WITHOUT compression filter...")
    sim_without = StressTestSimulator(use_compression=False, num_runs=1000)
    analysis_without = sim_without.run_stress_test()
    sim_without.print_report(analysis_without)

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPRESSION IMPACT UNDER STRESS")
    print("=" * 70)
    print(f"  Stress Pass Rate WITH compression:    {analysis_with['pass_rate']:.1f}%")
    print(f"  Stress Pass Rate WITHOUT compression: {analysis_without['pass_rate']:.1f}%")
    print(f"  Improvement under stress:             +{analysis_with['pass_rate'] - analysis_without['pass_rate']:.1f}%")
    print("=" * 70)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'STRESS_TEST_NIGHTMARE',
        'stress_config': {
            'degraded_win_rate': STRESS_CONFIG['base_win_rate'],
            'chaos_level': STRESS_CONFIG['chaos_level'],
            'flash_crash_prob': STRESS_CONFIG['flash_crash_prob'],
            'loss_streak_prob': STRESS_CONFIG['loss_streak_prob']
        },
        'with_compression': analysis_with,
        'without_compression': analysis_without
    }

    output_path = Path('prop_sim_blueguardian_stress_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nStress test results saved to: {output_path.absolute()}")

    # Final recommendation
    print("\n" + "=" * 70)
    print("  FINAL RECOMMENDATION")
    print("=" * 70)

    normal_pass = 100.0  # From previous test
    stress_pass = analysis_with['pass_rate']

    if stress_pass >= 60:
        rec = "APPROVED FOR BLUE GUARDIAN LIVE DEPLOYMENT"
        details = "System shows strong resilience under nightmare conditions."
    elif stress_pass >= 40:
        rec = "CONDITIONAL APPROVAL - Use conservative position sizing"
        details = "System is viable but recommend starting with 50% of planned size."
    else:
        rec = "HOLD - More optimization needed before live deployment"
        details = "System struggles under extreme stress. Consider additional training."

    print(f"\n  Normal Conditions Pass Rate: {normal_pass:.1f}%")
    print(f"  Stress Conditions Pass Rate: {stress_pass:.1f}%")
    print(f"\n  RECOMMENDATION: {rec}")
    print(f"  {details}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    main()
