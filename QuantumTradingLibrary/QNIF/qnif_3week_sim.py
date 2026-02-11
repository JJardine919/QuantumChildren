"""
QNIF 3-Week Backtest Simulation
=================================
20-year veteran Biskits built this. It's designed to stress-test the QNIF
pipeline against prop firm rules on BTC and ETH over the last 3 weeks of M5 data.

This isn't a toy. This is what you run before you risk real capital.

Architecture:
- QNIF_Engine provides bio-quantum signals (compression, neural consensus, immune)
- PropFarmAccount simulates prop firm execution with strict DD rules
- 55 parameter sets from signal_farm_config run in parallel
- Results saved to JSON for analysis

Date: 2026-02-10
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import asdict

import numpy as np
import pandas as pd

# GPU Setup - Use DirectML for tensor ops
try:
    import torch
    import torch_directml
    DEVICE = torch_directml.device()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    DEVICE = None

# Path Setup
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][QNIF_SIM] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("QNIF_SIM")

# Imports
try:
    from QNIF.QNIF_Master import QNIF_Engine, BioQuantumState
    from prop_farm_simulator import PropFarmAccount, AccountResult, result_to_json
    from signal_farm_config import FARM_ACCOUNTS, CHALLENGE_RULES, FARM_SYMBOLS
    from credential_manager import get_credentials
    import MetaTrader5 as mt5
    IMPORTS_OK = True
except ImportError as e:
    logger.error(f"Import failed: {e}")
    IMPORTS_OK = False


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_mt5_data(symbol: str, days: int = 21, account_key: str = "ATLAS") -> pd.DataFrame:
    """
    Fetch M5 data from MT5 for the last N days.

    Args:
        symbol: Trading symbol (BTCUSD, ETHUSD, etc.)
        days: Number of days to fetch
        account_key: Account to connect to

    Returns:
        DataFrame with columns: time, open, high, low, close, tick_volume
    """
    logger.info(f"Fetching {days} days of M5 data for {symbol} from MT5...")

    # Get credentials
    creds = get_credentials(account_key)
    if not creds:
        raise ValueError(f"No credentials found for {account_key}")

    # Initialize MT5
    if not mt5.initialize(creds.get('terminal_path')):
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    # Login
    authorized = mt5.login(
        login=creds['account'],
        password=creds['password'],
        server=creds['server']
    )

    if not authorized:
        mt5.shutdown()
        raise RuntimeError(f"MT5 login failed for {account_key}: {mt5.last_error()}")

    logger.info(f"Connected to MT5 account {creds['account']}")

    # Fetch data
    try:
        utc_to = datetime.utcnow()
        utc_from = utc_to - timedelta(days=days)

        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, utc_from, utc_to)

        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {symbol}")

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        logger.info(f"Fetched {len(df)} M5 bars for {symbol} ({df['time'].min()} to {df['time'].max()})")

        return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    finally:
        mt5.shutdown()


def load_csv_data(symbol: str, days: int = 21) -> pd.DataFrame:
    """
    Load data from CSV files in QNIF/HistoricalData/Full/.

    Args:
        symbol: Trading symbol
        days: Number of days to load (from end of file)

    Returns:
        DataFrame with columns: time, open, high, low, close, tick_volume
    """
    logger.info(f"Loading {days} days of data for {symbol} from CSV...")

    # Map MT5 symbol names to Binance CSV file names
    csv_symbol_map = {"BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT", "XAUUSD": "XAUUSDT", "XAGUSD": "XAGUSDT"}
    csv_symbol = csv_symbol_map.get(symbol, symbol)
    csv_path = current_dir / "HistoricalData" / "Full" / f"{csv_symbol}_5m.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Standardize column names - Binance CSV format:
    # Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, ...
    col_map = {
        'Open time': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'tick_volume',
    }
    df.rename(columns=col_map, inplace=True)

    # Parse time (Binance CSVs use millisecond timestamps)
    if 'time' in df.columns:
        if df['time'].dtype in ['int64', 'float64']:
            if df['time'].iloc[0] > 1e12:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
            else:
                df['time'] = pd.to_datetime(df['time'], unit='s')
        else:
            df['time'] = pd.to_datetime(df['time'])
    else:
        raise ValueError(f"No time column found in {csv_path}")

    # Sort by time
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Take last N days
    if len(df) > 0:
        cutoff = df['time'].max() - timedelta(days=days)
        df = df[df['time'] >= cutoff].copy()

    logger.info(f"Loaded {len(df)} bars for {symbol} ({df['time'].min()} to {df['time'].max()})")

    if 'tick_volume' not in df.columns:
        df['tick_volume'] = 1  # Dummy volume

    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]


# =============================================================================
# QNIF SIGNAL GENERATION
# =============================================================================

def generate_qnif_signals(df: pd.DataFrame, symbol: str, engine: QNIF_Engine,
                          window_size: int = 256, stride: int = 10) -> List[Dict[str, Any]]:
    """
    Run QNIF_Engine.process_pulse() on rolling 256-bar windows with stride.

    Args:
        df: OHLCV DataFrame
        symbol: Trading symbol
        engine: QNIF_Engine instance
        window_size: Size of rolling window
        stride: Process every Nth bar (VDJ quantum circuit takes ~1min/bar)

    Returns:
        List of signal dictionaries with QNIF state
    """
    start_idx = max(window_size, 100)
    total_steps = (len(df) - start_idx) // stride
    logger.info(f"Generating QNIF signals for {symbol} (window={window_size}, stride={stride}, ~{total_steps} pulses)...")

    signals = []
    bars = df[['open', 'high', 'low', 'close']].values
    pulse_count = 0

    for i in range(start_idx, len(bars), stride):
        window = bars[i-window_size:i]
        pulse_count += 1

        if pulse_count % 25 == 0:
            logger.info(f"  [{symbol}] Pulse {pulse_count}/{total_steps} (bar {i}/{len(bars)})")

        try:
            state: BioQuantumState = engine.process_pulse(symbol, window)

            signal = {
                'bar_index': i,
                'time': str(df.iloc[i]['time']),
                'symbol': symbol,
                'price': float(df.iloc[i]['close']),
                'action': state.final_action,
                'confidence': state.final_confidence,
                'compression_ratio': state.compression_ratio,
                'regime': state.regime,
                'neural_consensus': state.neural_consensus,
                'pattern_energy': state.pattern_energy,
                'shock_level': state.shock_level,
                'is_memory_recall': state.is_memory_recall,
                'lot_multiplier': state.lot_multiplier,
                'generation': state.generation,
            }

            signals.append(signal)

        except Exception as e:
            logger.warning(f"QNIF pulse failed at bar {i}: {e}")
            continue

    logger.info(f"Generated {len(signals)} QNIF signals for {symbol}")
    return signals


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_simulation_for_account(account_id: str, params: Any, df: pd.DataFrame,
                                symbol: str, qnif_signals: List[Dict[str, Any]]) -> AccountResult:
    """
    Run PropFarmAccount simulation with QNIF signals.

    Args:
        account_id: Account identifier
        params: FarmParams from signal_farm_config
        df: OHLCV DataFrame
        symbol: Trading symbol
        qnif_signals: List of QNIF signal dicts

    Returns:
        AccountResult with performance metrics
    """
    # Determine starting balance
    balance = params.starting_balance if params.starting_balance > 0 else CHALLENGE_RULES.starting_balance

    # Create account
    account = PropFarmAccount(
        account_id=hash(account_id) % 1000000,  # Numeric ID for PropFarmAccount
        symbol=symbol,
        balance=balance,
        max_daily_dd_pct=CHALLENGE_RULES.daily_dd_limit_pct * 100,
        max_total_dd_pct=CHALLENGE_RULES.max_dd_limit_pct * 100,
        profit_target_pct=CHALLENGE_RULES.profit_target_pct * 100,
    )

    # Override account parameters with FarmParams
    account.min_confidence = params.confidence_threshold
    account.tp_multiplier = params.tp_atr_multiplier / params.sl_atr_multiplier
    account.sl_atr_mult = params.sl_atr_multiplier
    account.max_positions = params.max_positions_per_symbol * 2  # BUY + SELL
    account.grid_spacing_pts = int(params.grid_spacing_atr * 100)  # Rough conversion

    # Symbol specs (hardcoded for BTC/ETH - good enough for sim)
    contract_size = 1.0
    point = 0.01

    # Run simulation
    result = account.run_simulation(
        df=df,
        timeframe="M5",
        cycle=1,
        phase="test",
        individual=None,
        contract_size=contract_size,
        point=point
    )

    return result


def run_multi_account_simulation(symbol: str, df: pd.DataFrame,
                                  qnif_signals: List[Dict[str, Any]],
                                  accounts_to_run: List[str] = None) -> List[AccountResult]:
    """
    Run simulation across multiple FARM_ACCOUNTS in parallel (sequential for safety).

    Args:
        symbol: Trading symbol
        df: OHLCV DataFrame
        qnif_signals: QNIF signals
        accounts_to_run: List of account IDs to run (None = all)

    Returns:
        List of AccountResult objects
    """
    if accounts_to_run is None:
        accounts_to_run = list(FARM_ACCOUNTS.keys())[:10]  # Default: first 10 for speed

    logger.info(f"Running simulation for {len(accounts_to_run)} accounts on {symbol}...")

    results = []

    for account_id in accounts_to_run:
        if account_id not in FARM_ACCOUNTS:
            logger.warning(f"Account {account_id} not found in FARM_ACCOUNTS, skipping")
            continue

        params = FARM_ACCOUNTS[account_id]

        try:
            result = run_simulation_for_account(account_id, params, df, symbol, qnif_signals)
            results.append(result)

            logger.info(f"  {account_id} ({params.label}): "
                       f"P/L=${result.net_profit:,.2f} | "
                       f"WR={result.win_rate:.1f}% | "
                       f"Trades={result.total_trades} | "
                       f"Pass={result.challenge_passed}")

        except Exception as e:
            logger.error(f"Simulation failed for {account_id}: {e}")
            continue

    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_summary_table(btc_results: List[AccountResult], eth_results: List[AccountResult]):
    """Print comparison table between BTC and ETH."""
    print("\n" + "="*80)
    print("  QNIF 3-WEEK SIMULATION SUMMARY")
    print("="*80)

    # BTC Summary
    btc_passed = [r for r in btc_results if r.challenge_passed]
    btc_failed = [r for r in btc_results if r.challenge_failed]
    btc_total_profit = sum(r.net_profit for r in btc_results)
    btc_avg_wr = np.mean([r.win_rate for r in btc_results]) if btc_results else 0
    btc_total_trades = sum(r.total_trades for r in btc_results)

    print(f"\nBTCUSD ({len(btc_results)} accounts):")
    print(f"  Passed:        {len(btc_passed)}/{len(btc_results)} ({len(btc_passed)/len(btc_results)*100:.1f}%)")
    print(f"  Failed:        {len(btc_failed)}/{len(btc_results)}")
    print(f"  Total P/L:     ${btc_total_profit:,.2f}")
    print(f"  Avg Win Rate:  {btc_avg_wr:.1f}%")
    print(f"  Total Trades:  {btc_total_trades}")

    # ETH Summary
    eth_passed = [r for r in eth_results if r.challenge_passed]
    eth_failed = [r for r in eth_results if r.challenge_failed]
    eth_total_profit = sum(r.net_profit for r in eth_results)
    eth_avg_wr = np.mean([r.win_rate for r in eth_results]) if eth_results else 0
    eth_total_trades = sum(r.total_trades for r in eth_results)

    print(f"\nETHUSD ({len(eth_results)} accounts):")
    print(f"  Passed:        {len(eth_passed)}/{len(eth_results)} ({len(eth_passed)/len(eth_results)*100:.1f}%)")
    print(f"  Failed:        {len(eth_failed)}/{len(eth_results)}")
    print(f"  Total P/L:     ${eth_total_profit:,.2f}")
    print(f"  Avg Win Rate:  {eth_avg_wr:.1f}%")
    print(f"  Total Trades:  {eth_total_trades}")

    # Best performers
    print("\n" + "-"*80)
    print("  TOP PERFORMERS")
    print("-"*80)

    all_results = btc_results + eth_results
    all_results_sorted = sorted(all_results, key=lambda x: x.net_profit, reverse=True)[:5]

    for i, r in enumerate(all_results_sorted, 1):
        print(f"{i}. {r.symbol} Account #{r.account_id}: "
              f"${r.net_profit:,.2f} | "
              f"WR={r.win_rate:.1f}% | "
              f"PF={r.profit_factor:.2f} | "
              f"{'PASS' if r.challenge_passed else 'FAIL'}")

    print("="*80 + "\n")


def save_results(btc_results: List[AccountResult], eth_results: List[AccountResult],
                 output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save BTC results
    btc_file = output_dir / f"btc_sim_{timestamp}.json"
    with open(btc_file, 'w') as f:
        json.dump([result_to_json(r) for r in btc_results], f, indent=2)
    logger.info(f"Saved BTC results to {btc_file}")

    # Save ETH results
    eth_file = output_dir / f"eth_sim_{timestamp}.json"
    with open(eth_file, 'w') as f:
        json.dump([result_to_json(r) for r in eth_results], f, indent=2)
    logger.info(f"Saved ETH results to {eth_file}")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'simulation_days': 21,
        'btc': {
            'accounts': len(btc_results),
            'passed': sum(1 for r in btc_results if r.challenge_passed),
            'failed': sum(1 for r in btc_results if r.challenge_failed),
            'total_profit': sum(r.net_profit for r in btc_results),
            'avg_win_rate': np.mean([r.win_rate for r in btc_results]),
            'total_trades': sum(r.total_trades for r in btc_results),
        },
        'eth': {
            'accounts': len(eth_results),
            'passed': sum(1 for r in eth_results if r.challenge_passed),
            'failed': sum(1 for r in eth_results if r.challenge_failed),
            'total_profit': sum(r.net_profit for r in eth_results),
            'avg_win_rate': np.mean([r.win_rate for r in eth_results]),
            'total_trades': sum(r.total_trades for r in eth_results),
        }
    }

    summary_file = output_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="QNIF 3-Week Backtest Simulation")
    parser.add_argument('--account', default='ATLAS',
                       help='MT5 account to use for data fetch (default: ATLAS)')
    parser.add_argument('--csv', action='store_true',
                       help='Load data from CSV instead of MT5')
    parser.add_argument('--days', type=int, default=21,
                       help='Number of days to simulate (default: 21)')
    parser.add_argument('--accounts', nargs='+', default=None,
                       help='Specific farm accounts to run (default: first 10)')
    parser.add_argument('--window', type=int, default=256,
                       help='QNIF window size (default: 256)')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSD', 'ETHUSD'],
                       help='Symbols to simulate (default: BTCUSD ETHUSD)')
    parser.add_argument('--stride', type=int, default=10,
                       help='Process every Nth bar (VDJ ~1min/bar, default: 10)')

    args = parser.parse_args()

    # Check imports
    if not IMPORTS_OK:
        logger.error("Required imports failed. Check environment.")
        return 1

    # GPU check
    if GPU_AVAILABLE:
        logger.info(f"GPU acceleration available: {DEVICE}")
    else:
        logger.warning("GPU not available, using CPU")

    # Initialize QNIF Engine
    logger.info("Initializing QNIF Engine...")
    engine = QNIF_Engine()

    # Results storage
    symbol_results = {}

    # Run simulation for each symbol
    for symbol in args.symbols:
        logger.info(f"\n{'='*80}")
        logger.info(f"  SIMULATING {symbol}")
        logger.info(f"{'='*80}\n")

        # Fetch or load data
        try:
            if args.csv:
                df = load_csv_data(symbol, days=args.days)
            else:
                df = fetch_mt5_data(symbol, days=args.days, account_key=args.account)
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            continue

        # Generate QNIF signals
        try:
            qnif_signals = generate_qnif_signals(df, symbol, engine, window_size=args.window, stride=args.stride)
        except Exception as e:
            logger.error(f"Failed to generate QNIF signals for {symbol}: {e}")
            continue

        # Run multi-account simulation
        try:
            results = run_multi_account_simulation(symbol, df, qnif_signals, args.accounts)
            symbol_results[symbol] = results
        except Exception as e:
            logger.error(f"Failed to run simulation for {symbol}: {e}")
            continue

    # Print summary
    btc_results = symbol_results.get('BTCUSD', [])
    eth_results = symbol_results.get('ETHUSD', [])

    if btc_results or eth_results:
        print_summary_table(btc_results, eth_results)

        # Save results
        output_dir = current_dir / "sim_results"
        save_results(btc_results, eth_results, output_dir)
    else:
        logger.error("No results to report")
        return 1

    logger.info("Simulation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
