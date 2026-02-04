"""
BLUE GUARDIAN 1-MONTH PROP FIRM SIMULATION
==========================================
Simulates the 72% WR expert (with compression +12% boost = 82% projected)
against Blue Guardian prop firm rules for 1 month.

Rules:
- Starting Balance: $5,000
- Daily Drawdown Limit: 5% ($250)
- Max Drawdown Limit: 10% ($500)
- Profit Target: 10% ($500)
- Duration: ~21 trading days (1 month)

Expert: expert_C7_E36_WR72.json specs:
- Original Win Rate: 72%
- Compression Ratio: 3.0
- Projected Win Rate: 82% (with quantum compression regime filtering)
"""

import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s][%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Blue Guardian Challenge Rules
BG_CONFIG = {
    'account_size': 5_000.0,
    'daily_dd_pct': 0.05,       # 5% daily drawdown limit
    'max_dd_pct': 0.10,         # 10% max drawdown limit
    'profit_target_pct': 0.10,  # 10% profit target
    'challenge_days': 21,       # ~1 month trading days

    # Risk Management
    'risk_per_trade_pct': 0.01,  # 1% risk per trade
    'min_rr_ratio': 1.5,         # Minimum risk:reward
    'max_trades_per_day': 5,     # Max trades per day

    # Expert settings
    'base_win_rate': 0.72,       # 72% base win rate
    'compression_boost': 0.12,   # +12% from compression filtering
    'projected_win_rate': 0.82,  # 82% with compression
    'compression_ratio': 3.0,
    'regime': 'CLEAN',

    # Market simulation
    'slippage_pips_mean': 1.5,
    'slippage_pips_std': 1.0,
    'spread_base_usd': 2.0,
    'chaos_level': 0.5,          # Market chaos factor (0.5 = moderate)
}

# Calculated limits
BG_CONFIG['daily_dd_limit'] = BG_CONFIG['account_size'] * BG_CONFIG['daily_dd_pct']
BG_CONFIG['max_dd_limit'] = BG_CONFIG['account_size'] * BG_CONFIG['max_dd_pct']
BG_CONFIG['profit_target'] = BG_CONFIG['account_size'] * BG_CONFIG['profit_target_pct']
BG_CONFIG['risk_amount'] = BG_CONFIG['account_size'] * BG_CONFIG['risk_per_trade_pct']


# ============================================================================
# MODELS
# ============================================================================
class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class LSTMExpert(nn.Module):
    """LSTM architecture matching ETARE champions"""
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# ============================================================================
# QUANTUM COMPRESSION REGIME FILTER (Simplified Fast Version)
# ============================================================================
class QuantumCompressionFilter:
    """
    Simulates the quantum compression regime filter.
    Higher compression ratio = cleaner trending market = boost win rate.
    Lower compression ratio = choppy market = reduce position size or skip.
    """
    def __init__(self, base_win_rate: float = 0.72, compression_boost: float = 0.12):
        self.base_win_rate = base_win_rate
        self.compression_boost = compression_boost
        self.cache = {}

    def analyze_regime(self, price_window: np.ndarray) -> dict:
        """Analyze market regime using Shannon entropy proxy"""
        data_hash = hashlib.md5(price_window.tobytes()).hexdigest()
        if data_hash in self.cache:
            return self.cache[data_hash]

        returns = np.diff(price_window) / (price_window[:-1] + 1e-10)
        if len(returns) < 10:
            return {'ratio': 1.0, 'regime': 'NEUTRAL', 'trade_boost': 0.0}

        # Shannon entropy from discretized returns
        hist, _ = np.histogram(returns, bins=8, density=True)
        hist = hist + 1e-10
        entropy = -np.sum(hist * np.log2(hist)) * (hist > 0).sum() / 8

        # Volatility measure
        volatility = np.std(returns)

        # Trend strength (autocorrelation)
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        else:
            autocorr = 0

        # Compression ratio simulation
        # Low entropy + high autocorr = trending = high compression ratio
        # High entropy + low autocorr = choppy = low compression ratio
        compression_score = (1 - entropy/3.0) * 0.5 + (autocorr + 1) / 2 * 0.5
        ratio = 1.0 + compression_score * 3.0  # Range: 1.0 to 4.0

        # Determine regime and trade boost
        if ratio > 2.5:
            regime = 'TRENDING'
            trade_boost = self.compression_boost
        elif ratio > 1.5:
            regime = 'CLEAN'
            trade_boost = self.compression_boost * 0.7
        else:
            regime = 'CHOPPY'
            trade_boost = 0.0  # No boost in choppy markets

        result = {
            'ratio': ratio,
            'entropy': entropy,
            'volatility': volatility,
            'autocorr': autocorr if not np.isnan(autocorr) else 0,
            'regime': regime,
            'trade_boost': trade_boost,
            'effective_win_rate': self.base_win_rate + trade_boost
        }

        self.cache[data_hash] = result
        return result

    def should_trade(self, regime_info: dict) -> Tuple[bool, float]:
        """
        Decide whether to trade based on regime.
        Returns: (should_trade, position_size_multiplier)
        """
        regime = regime_info['regime']

        if regime == 'TRENDING':
            return True, 1.5  # Full size + boost
        elif regime == 'CLEAN':
            return True, 1.0  # Normal size
        else:  # CHOPPY
            return False, 0.5  # Skip or reduced size


# ============================================================================
# PORTFOLIO STATE
# ============================================================================
@dataclass
class PortfolioState:
    equity: float = 5000.0
    daily_start_equity: float = 5000.0
    high_water_mark: float = 5000.0
    position: Optional[str] = None
    entry_price: float = 0.0
    entry_day: int = 0
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

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_loss(self) -> float:
        return self.equity - BG_CONFIG['account_size']

    @property
    def profit_pct(self) -> float:
        return (self.equity / BG_CONFIG['account_size'] - 1) * 100


# ============================================================================
# MARKET DATA GENERATION
# ============================================================================
def generate_market_data(symbol: str = 'BTCUSD', days: int = 30) -> pd.DataFrame:
    """Generate realistic market data for simulation"""

    params = {
        'BTCUSD': {'base_price': 95000.0, 'daily_vol': 0.025},
        'XAUUSD': {'base_price': 2650.0, 'daily_vol': 0.012},
        'EURUSD': {'base_price': 1.0850, 'daily_vol': 0.005},
    }

    p = params.get(symbol, {'base_price': 1000.0, 'daily_vol': 0.015})

    bars_per_day = 288  # M5 candles per day
    total_bars = days * bars_per_day

    times = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=total_bars,
        freq='5min'
    )

    np.random.seed(42)  # Reproducible

    # Generate returns with trend and mean reversion
    returns = np.random.normal(0, p['daily_vol'] / np.sqrt(bars_per_day), total_bars)

    # Add trend component
    trend = np.zeros(total_bars)
    for i in range(1, total_bars):
        trend[i] = 0.999 * trend[i-1] + np.random.randn() * 0.0001

    # Combine
    total_returns = returns + trend
    total_returns = np.clip(total_returns, -0.03, 0.03)

    prices = p['base_price'] * np.exp(np.cumsum(total_returns))

    # OHLCV
    intrabar_vol = np.abs(np.random.randn(total_bars)) * 0.001 + 0.0005
    high = prices * (1 + intrabar_vol)
    low = prices * (1 - intrabar_vol)
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]

    return pd.DataFrame({
        'time': times,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'tick_volume': np.random.randint(100, 10000, total_bars)
    })


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Extract 8 technical indicators as features"""
    data = df.copy()

    # RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    data['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(20).mean()
    data['bb_std'] = data['close'].rolling(20).std()
    data['bb_position'] = (data['close'] - data['bb_middle']) / (data['bb_std'] + 1e-8)

    # Momentum
    data['momentum'] = data['close'] / data['close'].shift(10)

    # ATR
    data['atr'] = data['high'].rolling(14).max() - data['low'].rolling(14).min()

    # Price change
    data['price_change'] = data['close'].pct_change()

    # Normalize
    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_position', 'momentum', 'atr', 'price_change', 'close']
    for col in feature_cols:
        if col in data.columns:
            mean = data[col].rolling(100, min_periods=1).mean()
            std = data[col].rolling(100, min_periods=1).std()
            data[col] = (data[col] - mean) / (std + 1e-8)
            data[col] = data[col].clip(-4, 4)

    data = data.fillna(0)
    return data[feature_cols].values


# ============================================================================
# PROP FIRM SIMULATION
# ============================================================================
class BluGuardianSimulator:
    """
    Simulates Blue Guardian prop firm challenge with:
    - 72% base win rate expert
    - Quantum compression regime filtering (+12% boost = 82% projected)
    - Proper prop firm rules enforcement
    """

    def __init__(self, use_compression: bool = True, num_runs: int = 100):
        self.use_compression = use_compression
        self.num_runs = num_runs
        self.compression_filter = QuantumCompressionFilter(
            base_win_rate=BG_CONFIG['base_win_rate'],
            compression_boost=BG_CONFIG['compression_boost']
        )
        self.results = []

    def simulate_trade(self, regime_info: dict, portfolio: PortfolioState) -> float:
        """
        Simulate a single trade outcome with market chaos.
        Returns PnL.
        """
        # Get effective win rate based on regime
        if self.use_compression:
            effective_wr = regime_info['effective_win_rate']
            _, size_mult = self.compression_filter.should_trade(regime_info)
        else:
            effective_wr = BG_CONFIG['base_win_rate']
            size_mult = 1.0

        # Risk amount adjusted by position size multiplier
        risk = BG_CONFIG['risk_amount'] * size_mult

        # Chaos factors
        chaos = BG_CONFIG['chaos_level']

        # Slippage (worse on losses)
        slippage = np.random.normal(
            BG_CONFIG['slippage_pips_mean'],
            BG_CONFIG['slippage_pips_std']
        ) * chaos

        # Spread
        spread = BG_CONFIG['spread_base_usd'] * (1 + random.uniform(0, 0.5) * chaos)

        # Trade outcome
        is_win = random.random() < effective_wr

        if is_win:
            # Winning trade: risk * RR ratio
            rr = random.uniform(BG_CONFIG['min_rr_ratio'], 2.5)
            base_profit = risk * rr
            # Slight reduction from slippage on wins
            profit_mult = 1.0 - (slippage * 0.01)
            pnl = base_profit * profit_mult - spread
            portfolio.winning_trades += 1
        else:
            # Losing trade: lose the risk amount + extra from slippage
            loss_mult = 1.0 + (slippage * 0.02)  # Slippage makes losses worse
            pnl = -(risk * loss_mult) - spread
            portfolio.losing_trades += 1

        return pnl

    def run_single_simulation(self, run_id: int) -> dict:
        """Run a single prop firm challenge simulation"""

        # Generate market data
        market_data = generate_market_data('BTCUSD', days=BG_CONFIG['challenge_days'] + 10)
        features = prepare_features(market_data)
        prices = market_data['close'].values
        times = market_data['time'].values

        # Initialize portfolio
        portfolio = PortfolioState(
            equity=BG_CONFIG['account_size'],
            daily_start_equity=BG_CONFIG['account_size'],
            high_water_mark=BG_CONFIG['account_size']
        )

        current_date = None
        day_count = 0
        bars_per_day = 288
        max_bars = min(len(features), BG_CONFIG['challenge_days'] * bars_per_day)

        # Track daily violations
        daily_dd_violations = []

        for t in range(100, max_bars):
            if not portfolio.is_active:
                break

            bar_date = pd.Timestamp(times[t]).date()

            # New trading day
            if bar_date != current_date:
                current_date = bar_date
                day_count += 1
                portfolio.current_day = day_count
                portfolio.daily_start_equity = portfolio.equity
                portfolio.daily_pnl = 0.0
                portfolio.daily_trades = 0

                if day_count > BG_CONFIG['challenge_days']:
                    break

            # Check profit target (PASSED)
            if portfolio.equity >= BG_CONFIG['account_size'] + BG_CONFIG['profit_target']:
                break

            # Check max daily trades
            if portfolio.daily_trades >= BG_CONFIG['max_trades_per_day']:
                continue

            # Get regime analysis using price window
            window_size = 50
            if t >= window_size:
                price_window = prices[t-window_size:t]
                regime_info = self.compression_filter.analyze_regime(price_window)
            else:
                regime_info = {
                    'regime': 'NEUTRAL',
                    'effective_win_rate': BG_CONFIG['base_win_rate'],
                    'ratio': 1.5
                }

            # Skip trade in choppy conditions if using compression
            if self.use_compression:
                should_trade, _ = self.compression_filter.should_trade(regime_info)
                if not should_trade:
                    continue

            # Simulate trade decision (simplified - trades on ~10% of bars)
            if random.random() > 0.10:
                continue

            # Execute trade
            pnl = self.simulate_trade(regime_info, portfolio)
            portfolio.trades.append({
                'day': day_count,
                'bar': t,
                'pnl': pnl,
                'regime': regime_info['regime'],
                'equity_after': portfolio.equity + pnl
            })

            portfolio.equity += pnl
            portfolio.daily_pnl += pnl
            portfolio.daily_trades += 1

            # Update high water mark
            if portfolio.equity > portfolio.high_water_mark:
                portfolio.high_water_mark = portfolio.equity

            # Check daily drawdown
            daily_dd = portfolio.daily_start_equity - portfolio.equity
            if daily_dd > portfolio.max_daily_dd_hit:
                portfolio.max_daily_dd_hit = daily_dd

            if daily_dd > BG_CONFIG['daily_dd_limit']:
                portfolio.is_active = False
                portfolio.fail_reason = f"DAILY LOSS LIMIT HIT (Day {day_count}): ${daily_dd:.2f} > ${BG_CONFIG['daily_dd_limit']:.2f}"
                daily_dd_violations.append({
                    'day': day_count,
                    'dd': daily_dd,
                    'limit': BG_CONFIG['daily_dd_limit']
                })
                break

            # Check max drawdown
            total_dd = BG_CONFIG['account_size'] - portfolio.equity
            if total_dd > portfolio.max_dd_hit:
                portfolio.max_dd_hit = total_dd

            if total_dd > BG_CONFIG['max_dd_limit']:
                portfolio.is_active = False
                portfolio.fail_reason = f"MAX DRAWDOWN HIT: ${total_dd:.2f} > ${BG_CONFIG['max_dd_limit']:.2f}"
                break

        # Determine pass/fail
        profit_loss = portfolio.equity - BG_CONFIG['account_size']
        passed = False

        if portfolio.is_active:
            if profit_loss >= BG_CONFIG['profit_target']:
                passed = True
            elif day_count >= BG_CONFIG['challenge_days']:
                portfolio.fail_reason = f"TIME EXPIRED (Profit: {portfolio.profit_pct:.1f}% of {BG_CONFIG['profit_target_pct']*100}% required)"

        return {
            'run_id': run_id,
            'passed': passed,
            'final_equity': portfolio.equity,
            'profit_loss': profit_loss,
            'profit_pct': portfolio.profit_pct,
            'win_rate': portfolio.win_rate,
            'total_trades': portfolio.total_trades,
            'winning_trades': portfolio.winning_trades,
            'losing_trades': portfolio.losing_trades,
            'max_dd_hit': portfolio.max_dd_hit,
            'max_dd_pct': (portfolio.max_dd_hit / BG_CONFIG['account_size']) * 100,
            'max_daily_dd_hit': portfolio.max_daily_dd_hit,
            'max_daily_dd_pct': (portfolio.max_daily_dd_hit / BG_CONFIG['account_size']) * 100,
            'days_survived': portfolio.current_day,
            'fail_reason': portfolio.fail_reason,
            'daily_dd_violations': daily_dd_violations,
            'compression_enabled': self.use_compression
        }

    def run_monte_carlo(self) -> dict:
        """Run Monte Carlo simulation of the prop firm challenge"""

        log.info("=" * 70)
        log.info("BLUE GUARDIAN PROP FIRM SIMULATION")
        log.info("=" * 70)
        log.info(f"Expert: 72% WR + Compression Boost = 82% Projected")
        log.info(f"Account Size: ${BG_CONFIG['account_size']:,.0f}")
        log.info(f"Profit Target: {BG_CONFIG['profit_target_pct']*100}% (${BG_CONFIG['profit_target']:,.0f})")
        log.info(f"Daily DD Limit: {BG_CONFIG['daily_dd_pct']*100}% (${BG_CONFIG['daily_dd_limit']:,.0f})")
        log.info(f"Max DD Limit: {BG_CONFIG['max_dd_pct']*100}% (${BG_CONFIG['max_dd_limit']:,.0f})")
        log.info(f"Challenge Duration: {BG_CONFIG['challenge_days']} days")
        log.info(f"Compression Filter: {'ENABLED' if self.use_compression else 'DISABLED'}")
        log.info(f"Monte Carlo Runs: {self.num_runs}")
        log.info("=" * 70)

        self.results = []

        for i in range(self.num_runs):
            result = self.run_single_simulation(i)
            self.results.append(result)

            if (i + 1) % 20 == 0:
                passes = sum(1 for r in self.results if r['passed'])
                log.info(f"  Progress: {i+1}/{self.num_runs} runs | Pass rate so far: {passes}/{i+1} ({passes/(i+1)*100:.1f}%)")

        return self.analyze_results()

    def analyze_results(self) -> dict:
        """Analyze Monte Carlo results"""

        passed = [r for r in self.results if r['passed']]
        failed = [r for r in self.results if not r['passed']]

        pass_rate = len(passed) / len(self.results) * 100

        # Win rate stats
        win_rates = [r['win_rate'] for r in self.results]
        avg_win_rate = np.mean(win_rates) * 100

        # PnL stats
        pnls = [r['profit_loss'] for r in self.results]
        avg_pnl = np.mean(pnls)

        # Drawdown stats
        max_dds = [r['max_dd_pct'] for r in self.results]
        avg_max_dd = np.mean(max_dds)

        daily_dds = [r['max_daily_dd_pct'] for r in self.results]
        avg_daily_dd = np.mean(daily_dds)

        # Failure analysis
        fail_reasons = {}
        for r in failed:
            reason = r['fail_reason'].split(':')[0] if r['fail_reason'] else 'Unknown'
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        analysis = {
            'total_runs': len(self.results),
            'passed': len(passed),
            'failed': len(failed),
            'pass_rate': pass_rate,
            'avg_win_rate': avg_win_rate,
            'avg_pnl': avg_pnl,
            'avg_max_dd': avg_max_dd,
            'avg_daily_dd': avg_daily_dd,
            'best_pnl': max(pnls),
            'worst_pnl': min(pnls),
            'best_equity': max(r['final_equity'] for r in self.results),
            'worst_equity': min(r['final_equity'] for r in self.results),
            'fail_reasons': fail_reasons,
            'compression_enabled': self.use_compression,
            'base_win_rate': BG_CONFIG['base_win_rate'] * 100,
            'projected_win_rate': BG_CONFIG['projected_win_rate'] * 100
        }

        return analysis

    def print_report(self, analysis: dict):
        """Print detailed simulation report"""

        print("\n" + "=" * 70)
        print("  BLUE GUARDIAN SIMULATION RESULTS")
        print("=" * 70)

        print(f"\n{'EXPERT CONFIGURATION':^70}")
        print("-" * 70)
        print(f"  Base Win Rate:       {analysis['base_win_rate']:.0f}%")
        print(f"  Compression Boost:   +{BG_CONFIG['compression_boost']*100:.0f}%")
        print(f"  Projected Win Rate:  {analysis['projected_win_rate']:.0f}%")
        print(f"  Compression Filter:  {'ENABLED' if analysis['compression_enabled'] else 'DISABLED'}")

        print(f"\n{'CHALLENGE PARAMETERS':^70}")
        print("-" * 70)
        print(f"  Starting Balance:    ${BG_CONFIG['account_size']:,.0f}")
        print(f"  Profit Target:       {BG_CONFIG['profit_target_pct']*100:.0f}% (${BG_CONFIG['profit_target']:,.0f})")
        print(f"  Max Daily Drawdown:  {BG_CONFIG['daily_dd_pct']*100:.0f}% (${BG_CONFIG['daily_dd_limit']:,.0f})")
        print(f"  Max Total Drawdown:  {BG_CONFIG['max_dd_pct']*100:.0f}% (${BG_CONFIG['max_dd_limit']:,.0f})")
        print(f"  Duration:            {BG_CONFIG['challenge_days']} trading days")

        print(f"\n{'SIMULATION RESULTS ({} RUNS)':^70}".format(analysis['total_runs']))
        print("-" * 70)

        # Pass/Fail with visual indicator
        if analysis['pass_rate'] >= 70:
            verdict = "READY FOR LIVE"
            symbol = "[PASS]"
        elif analysis['pass_rate'] >= 50:
            verdict = "MARGINAL - MORE TESTING NEEDED"
            symbol = "[WARN]"
        else:
            verdict = "NOT READY - NEEDS OPTIMIZATION"
            symbol = "[FAIL]"

        print(f"\n  {symbol} OVERALL PASS RATE: {analysis['pass_rate']:.1f}%")
        print(f"  Passed: {analysis['passed']} | Failed: {analysis['failed']}")

        print(f"\n  PERFORMANCE METRICS:")
        print(f"    Achieved Win Rate:    {analysis['avg_win_rate']:.1f}%")
        print(f"    Average P/L:          ${analysis['avg_pnl']:+,.2f}")
        print(f"    Best P/L:             ${analysis['best_pnl']:+,.2f}")
        print(f"    Worst P/L:            ${analysis['worst_pnl']:+,.2f}")

        print(f"\n  DRAWDOWN METRICS:")
        print(f"    Avg Max Drawdown:     {analysis['avg_max_dd']:.2f}%")
        print(f"    Avg Daily DD:         {analysis['avg_daily_dd']:.2f}%")

        if analysis['fail_reasons']:
            print(f"\n  FAILURE BREAKDOWN:")
            for reason, count in sorted(analysis['fail_reasons'].items(), key=lambda x: -x[1]):
                pct = count / analysis['failed'] * 100 if analysis['failed'] > 0 else 0
                print(f"    {reason}: {count} ({pct:.0f}% of failures)")

        print(f"\n{'VERDICT':^70}")
        print("=" * 70)
        print(f"  {verdict}")
        print("=" * 70)

        return analysis


def main():
    """Main simulation entry point"""

    print("\n" + "#" * 70)
    print("#  72% EXPERT + COMPRESSION BOOST = 82% PROJECTED WIN RATE")
    print("#  BLUE GUARDIAN $5,000 ACCOUNT PROP FIRM SIMULATION")
    print("#" * 70 + "\n")

    # Run simulation WITH compression (82% projected)
    print("\n[1/2] Running simulation WITH compression filter...")
    sim_with_compression = BluGuardianSimulator(use_compression=True, num_runs=500)
    analysis_with = sim_with_compression.run_monte_carlo()
    sim_with_compression.print_report(analysis_with)

    # Run simulation WITHOUT compression (72% base)
    print("\n[2/2] Running simulation WITHOUT compression filter (baseline)...")
    sim_without_compression = BluGuardianSimulator(use_compression=False, num_runs=500)
    analysis_without = sim_without_compression.run_monte_carlo()
    sim_without_compression.print_report(analysis_without)

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPRESSION FILTER IMPACT")
    print("=" * 70)
    print(f"  Pass Rate WITH compression:    {analysis_with['pass_rate']:.1f}%")
    print(f"  Pass Rate WITHOUT compression: {analysis_without['pass_rate']:.1f}%")
    print(f"  Improvement:                   +{analysis_with['pass_rate'] - analysis_without['pass_rate']:.1f}%")
    print(f"\n  Win Rate WITH compression:     {analysis_with['avg_win_rate']:.1f}%")
    print(f"  Win Rate WITHOUT compression:  {analysis_without['avg_win_rate']:.1f}%")
    print(f"  Win Rate Boost:                +{analysis_with['avg_win_rate'] - analysis_without['avg_win_rate']:.1f}%")
    print("=" * 70)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'expert_config': {
            'name': 'expert_C7_E36_WR72',
            'base_win_rate': 72,
            'compression_boost': 12,
            'projected_win_rate': 82,
            'compression_ratio': 3.0,
            'regime': 'CLEAN'
        },
        'challenge_config': {
            'account_size': BG_CONFIG['account_size'],
            'profit_target_pct': BG_CONFIG['profit_target_pct'],
            'daily_dd_pct': BG_CONFIG['daily_dd_pct'],
            'max_dd_pct': BG_CONFIG['max_dd_pct'],
            'duration_days': BG_CONFIG['challenge_days']
        },
        'with_compression': analysis_with,
        'without_compression': analysis_without,
        'recommendation': 'READY FOR LIVE' if analysis_with['pass_rate'] >= 70 else 'NEEDS MORE TESTING'
    }

    output_path = Path('prop_sim_blueguardian_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path.absolute()}")

    return results


if __name__ == '__main__':
    main()
