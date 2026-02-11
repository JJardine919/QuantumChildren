"""
Test Simulation Setup
=====================
Biskits built this to catch problems BEFORE you waste time running a full sim.

20 years of experience taught me: validate your environment before you start the job.

This script checks:
1. All imports work
2. QNIF Engine initializes
3. PropFarmAccount works
4. Signal farm config loads
5. MT5 or CSV data is accessible
6. GPU is available (optional)

Run this FIRST.

Usage:
    python test_sim_setup.py
    python test_sim_setup.py --csv
    python test_sim_setup.py --quick
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Path Setup
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

print("="*80)
print("  QNIF SIMULATION SETUP TEST")
print("  Built by Biskits - Validating environment before you start")
print("="*80)
print()

# =============================================================================
# TEST 1: IMPORTS
# =============================================================================

print("[1/8] Testing imports...")
import_errors = []

try:
    import numpy as np
    print("  [OK] numpy")
except ImportError as e:
    import_errors.append(("numpy", str(e)))

try:
    import pandas as pd
    print("  [OK] pandas")
except ImportError as e:
    import_errors.append(("pandas", str(e)))

try:
    import MetaTrader5 as mt5
    print("  [OK] MetaTrader5")
except ImportError as e:
    import_errors.append(("MetaTrader5", str(e)))

try:
    from prop_farm_simulator import PropFarmAccount, AccountResult
    print("  [OK] prop_farm_simulator")
except ImportError as e:
    import_errors.append(("prop_farm_simulator", str(e)))

try:
    from signal_farm_config import FARM_ACCOUNTS, CHALLENGE_RULES
    print("  [OK] signal_farm_config")
except ImportError as e:
    import_errors.append(("signal_farm_config", str(e)))

try:
    from credential_manager import get_credentials
    print("  [OK] credential_manager")
except ImportError as e:
    import_errors.append(("credential_manager", str(e)))

if import_errors:
    print()
    print("  [FAIL] IMPORT ERRORS:")
    for module, error in import_errors:
        print(f"    {module}: {error}")
    print()
    print("  Fix these before running simulation.")
    sys.exit(1)
else:
    print("  [OK] All required imports OK")

print()

# =============================================================================
# TEST 2: GPU AVAILABILITY
# =============================================================================

print("[2/8] Checking GPU availability...")

try:
    import torch
    import torch_directml
    device = torch_directml.device()
    print(f"  [OK] GPU available: {device}")
except ImportError:
    print("  [WARN] GPU not available (torch_directml not installed)")
    print("    Simulation will run on CPU (slower but functional)")
except Exception as e:
    print(f"  [WARN] GPU check failed: {e}")

print()

# =============================================================================
# TEST 3: QNIF ENGINE
# =============================================================================

print("[3/8] Testing QNIF Engine initialization...")

try:
    from QNIF.QNIF_Master import QNIF_Engine, BioQuantumState
    engine = QNIF_Engine()
    print("  [OK] QNIF_Engine initialized")

    # Try a dummy pulse
    dummy_bars = np.random.randn(256, 4) + 50000  # Fake OHLC
    state = engine.process_pulse("TESTBTC", dummy_bars)
    print(f"  [OK] process_pulse() works (action={state.final_action}, conf={state.final_confidence:.3f})")

except ImportError as e:
    print(f"  [WARN] QNIF Engine import failed: {e}")
    print("    Some QNIF components may not be installed")
    print("    Simulation will fall back to PropFarmAccount's internal signals")
except Exception as e:
    print(f"  [WARN] QNIF Engine test failed: {e}")
    print("    Simulation may still work with fallback signals")

print()

# =============================================================================
# TEST 4: PROP FARM ACCOUNT
# =============================================================================

print("[4/8] Testing PropFarmAccount...")

try:
    # Create test account
    account = PropFarmAccount(
        account_id=999,
        symbol="BTCUSD",
        balance=5000.0,
        max_daily_dd_pct=3.5,
        max_total_dd_pct=7.0,
        profit_target_pct=12.0
    )
    print("  [OK] PropFarmAccount created")

    # Generate fake data
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='5min')
    fake_data = pd.DataFrame({
        'time': dates,
        'open': np.random.randn(1000).cumsum() + 50000,
        'high': np.random.randn(1000).cumsum() + 50100,
        'low': np.random.randn(1000).cumsum() + 49900,
        'close': np.random.randn(1000).cumsum() + 50000,
        'tick_volume': np.random.randint(100, 1000, 1000)
    })

    # Run quick sim
    result = account.run_simulation(fake_data, timeframe="M5", cycle=1, phase="test")
    print(f"  [OK] Simulation ran (P/L=${result.net_profit:.2f}, Trades={result.total_trades})")

except Exception as e:
    print(f"  [FAIL] PropFarmAccount test failed: {e}")
    print("    This is a critical error - simulation will not work")
    sys.exit(1)

print()

# =============================================================================
# TEST 5: SIGNAL FARM CONFIG
# =============================================================================

print("[5/8] Testing signal farm config...")

try:
    account_count = len(FARM_ACCOUNTS)
    print(f"  [OK] Loaded {account_count} farm accounts")

    # Sample a few accounts
    sample_keys = list(FARM_ACCOUNTS.keys())[:3]
    for key in sample_keys:
        params = FARM_ACCOUNTS[key]
        print(f"    {key}: {params.label} (conf={params.confidence_threshold}, tp={params.tp_atr_multiplier})")

    print(f"  [OK] Challenge rules: ${CHALLENGE_RULES.starting_balance} balance, "
          f"{CHALLENGE_RULES.profit_target_pct*100:.0f}% target, "
          f"{CHALLENGE_RULES.max_dd_limit_pct*100:.0f}% max DD")

except Exception as e:
    print(f"  [FAIL] Signal farm config failed: {e}")
    sys.exit(1)

print()

# =============================================================================
# TEST 6: CREDENTIALS
# =============================================================================

print("[6/8] Testing credentials...")

try:
    creds = get_credentials('ATLAS')
    if creds:
        print(f"  [OK] ATLAS credentials loaded (account={creds['account']})")
    else:
        print("  [WARN] ATLAS credentials not found")
        print("    You'll need to use --csv mode")
except Exception as e:
    print(f"  [WARN] Credential check failed: {e}")
    print("    You'll need to use --csv mode")

print()

# =============================================================================
# TEST 7: MT5 CONNECTION (optional)
# =============================================================================

print("[7/8] Testing MT5 connection (optional)...")

parser = argparse.ArgumentParser()
parser.add_argument('--csv', action='store_true', help='Skip MT5 test')
parser.add_argument('--quick', action='store_true', help='Skip all optional tests')
args = parser.parse_args()

if args.csv or args.quick:
    print("  [SKIP] Skipped (CSV mode)")
else:
    try:
        creds = get_credentials('ATLAS')
        if not creds:
            print("  [SKIP] Skipped (no credentials)")
        else:
            if not mt5.initialize(creds.get('terminal_path')):
                print(f"  [WARN] MT5 initialize failed: {mt5.last_error()}")
            else:
                authorized = mt5.login(
                    login=creds['account'],
                    password=creds['password'],
                    server=creds['server']
                )

                if not authorized:
                    print(f"  [WARN] MT5 login failed: {mt5.last_error()}")
                else:
                    # Try to fetch 1 bar
                    utc_to = datetime.utcnow()
                    utc_from = utc_to - timedelta(days=1)
                    rates = mt5.copy_rates_range("BTCUSD", mt5.TIMEFRAME_M5, utc_from, utc_to)

                    if rates is None or len(rates) == 0:
                        print("  [WARN] MT5 data fetch failed (no data)")
                    else:
                        print(f"  [OK] MT5 connection OK (fetched {len(rates)} bars)")

                mt5.shutdown()
    except Exception as e:
        print(f"  [WARN] MT5 test error: {e}")
        print("    You can still use --csv mode")

print()

# =============================================================================
# TEST 8: CSV DATA (optional)
# =============================================================================

print("[8/8] Testing CSV data availability...")

csv_dir = current_dir / "HistoricalData" / "Full"

if not csv_dir.exists():
    print(f"  [SKIP] CSV directory not found: {csv_dir}")
else:
    btc_csv = csv_dir / "BTCUSD_M5.csv"
    eth_csv = csv_dir / "ETHUSD_M5.csv"

    found_files = []

    if btc_csv.exists():
        print(f"  [OK] Found BTCUSD_M5.csv")
        found_files.append("BTCUSD")
    else:
        print(f"  [SKIP] BTCUSD_M5.csv not found")

    if eth_csv.exists():
        print(f"  [OK] Found ETHUSD_M5.csv")
        found_files.append("ETHUSD")
    else:
        print(f"  [SKIP] ETHUSD_M5.csv not found")

    if not found_files:
        print("  [WARN] No CSV files found - you'll need MT5 mode or download data first")
    else:
        print(f"  [OK] CSV mode available for: {', '.join(found_files)}")

print()

# =============================================================================
# FINAL VERDICT
# =============================================================================

print("="*80)
print("  SETUP TEST COMPLETE")
print("="*80)
print()
print("VERDICT:")
print("  [OK] Core functionality is ready")
print("  [OK] You can run qnif_3week_sim.py")
print()
print("RECOMMENDATIONS:")
if import_errors:
    print("  - Fix import errors listed above")
else:
    print("  - All imports OK")

if args.csv:
    print("  - Using CSV mode (MT5 not tested)")
else:
    print("  - Test MT5 connection if you see warnings above")

print()
print("QUICK START:")
print("  python qnif_3week_sim.py --csv                    # CSV mode (offline)")
print("  python qnif_3week_sim.py --account ATLAS          # MT5 mode")
print("  python qnif_3week_sim.py --days 7 --quick         # Quick 7-day test")
print()
print("="*80)
print()
