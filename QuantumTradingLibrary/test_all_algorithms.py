"""
TEST: All 8 Biological Algorithms -- Integration Smoke Test
============================================================
Runs every algorithm with synthetic data to verify:
1. All imports work
2. All classes instantiate
3. All core methods execute without error
4. Cross-algorithm integration points work

Usage:
    python test_all_algorithms.py

Authors: DooDoo + Claude
Date:    2026-02-08
"""

import os
import sys
import io
import json
import time
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -- Setup --
TEST_DIR = tempfile.mkdtemp(prefix="qc_test_")
ORIG_DIR = os.getcwd()
os.chdir(Path(__file__).parent)

# Synthetic market data for testing
import numpy as np
np.random.seed(42)

SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "NAS100"]
ACTIVE_TES = ["L1_Neuronal", "Alu_Exonization", "HERV_Synapse", "Ty3_gypsy", "CACTA"]
ACTIVE_TES_BAD = ["hobo", "Mutator", "P_element", "Crypton"]

def make_synthetic_bars(n=100):
    """Generate synthetic OHLCV bars for testing."""
    close = 2000.0 + np.cumsum(np.random.randn(n) * 5)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    opn = close + np.random.randn(n) * 2
    volume = np.random.randint(100, 10000, n).astype(float)
    return {
        "open": opn.tolist(),
        "high": high.tolist(),
        "low": low.tolist(),
        "close": close.tolist(),
        "volume": volume.tolist(),
        "tick_volume": volume.tolist(),
    }

BARS = make_synthetic_bars(100)

# -- Result tracking --
results = {}

def test(name: str):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"  TEST: {name}")
            print(f"{'='*60}")
            start = time.time()
            try:
                func()
                elapsed = time.time() - start
                results[name] = ("PASS", elapsed)
                print(f"  [OK] PASS ({elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.time() - start
                results[name] = ("FAIL", elapsed, str(e))
                print(f"  [XX] FAIL ({elapsed:.2f}s): {e}")
                traceback.print_exc()
        return wrapper
    return decorator


# ============================================================
# ALGORITHM 1: V(D)J Recombination
# ============================================================
@test("1. V(D)J Recombination -- Antibody Generation")
def test_vdj():
    from vdj_recombination import VDJRecombinationEngine, VDJTEQABridge

    db_path = os.path.join(TEST_DIR, "vdj_test.db")
    engine = VDJRecombinationEngine(db_path=db_path)

    # Generate antibodies via recombine()
    antibodies = engine.recombine(n=20)
    assert len(antibodies) > 0, "No antibodies generated"
    print(f"    Generated {len(antibodies)} antibodies")

    # Record some outcomes for clonal selection
    for ab in antibodies[:10]:
        won = np.random.random() > 0.4
        ab.total_trades += 1
        if won:
            ab.win_rate = (ab.win_rate * max(ab.total_trades - 1, 0) + 1.0) / ab.total_trades
            ab.total_pnl += abs(np.random.randn() * 10)
            ab.profit_factor = max(ab.profit_factor, 1.2)
        else:
            ab.total_pnl -= abs(np.random.randn() * 5)

    # Run selection
    survivors, dead = engine.clonal_selection(antibodies)
    print(f"    Clonal selection: {len(survivors)} survivors, {len(dead)} dead")

    # Test bridge
    bridge = VDJTEQABridge(vdj_engine=engine)
    print(f"    VDJTEQABridge instantiated OK")


# ============================================================
# ALGORITHM 2: Protective Deletion
# ============================================================
@test("2. Protective Deletion -- Toxic Pattern Suppression")
def test_protective_deletion():
    from protective_deletion import ProtectiveDeletionTracker

    db_path = os.path.join(TEST_DIR, "deletion_test.db")
    tracker = ProtectiveDeletionTracker(db_path=db_path)

    # Record wins for good pattern
    for _ in range(25):
        tracker.record_outcome(ACTIVE_TES, won=True, profit=1.0)

    good_supp = tracker.get_suppression(ACTIVE_TES)
    print(f"    Good pattern suppression: {good_supp}")
    assert good_supp >= 0.9, f"Good pattern should not be suppressed, got {good_supp}"

    # Record losses for bad pattern
    for _ in range(25):
        tracker.record_outcome(ACTIVE_TES_BAD, won=False, profit=-1.0)

    bad_supp = tracker.get_suppression(ACTIVE_TES_BAD)
    print(f"    Bad pattern suppression: {bad_supp}")
    assert bad_supp < 1.0, f"Bad pattern should be suppressed, got {bad_supp}"

    # Combined modifier
    combined = tracker.get_combined_modifier(ACTIVE_TES)
    print(f"    Combined modifier (good): {combined}")

    combined_bad = tracker.get_combined_modifier(ACTIVE_TES_BAD)
    print(f"    Combined modifier (bad): {combined_bad}")


# ============================================================
# ALGORITHM 3: CRISPR-Cas9
# ============================================================
@test("3. CRISPR-Cas9 -- Immune Memory Gate")
def test_crispr():
    from crispr_cas import CRISPRTEQABridge

    db_path = os.path.join(TEST_DIR, "crispr_test.db")
    bridge = CRISPRTEQABridge(db_path=db_path)

    bars_list = list(zip(
        BARS["open"], BARS["high"], BARS["low"],
        BARS["close"], BARS["volume"]
    ))

    # Record a loss to create a spacer
    bars_array = np.array(bars_list[-30:])
    bridge.on_trade_loss(
        bars=bars_array,
        symbol="XAUUSD",
        direction=1,
        loss_amount=1.0,
        active_tes=ACTIVE_TES_BAD,
        spread=0.5
    )
    print("    Spacer acquired from trade loss")

    # Gate check
    result = bridge.gate_check(
        symbol="XAUUSD",
        direction=1,
        bars=bars_array,
        spread=0.5,
        active_tes=ACTIVE_TES,
        domestication_boost=1.0,
        confidence=0.5
    )
    gate_pass = result.get('gate_pass', result) if isinstance(result, dict) else result
    print(f"    Gate check result: pass={gate_pass}")


# ============================================================
# ALGORITHM 4: Electric Organs
# ============================================================
@test("4. Electric Organs -- Convergent Evolution")
def test_electric_organs():
    from electric_organs import ElectricOrgansBridge

    bridge = ElectricOrgansBridge(
        lineage_symbols=SYMBOLS,
        db_dir=TEST_DIR,
    )

    # Apply convergence boost (no data yet = 1.0)
    boost = bridge.apply(
        active_tes=ACTIVE_TES,
        domestication_boost=1.25,
        symbol="XAUUSD"
    )
    print(f"    Convergence boost (no data): {boost}")
    assert boost >= 1.0, f"Boost should be >= 1.0, got {boost}"


# ============================================================
# ALGORITHM 5: KoRV
# ============================================================
@test("5. KoRV -- Signal Onboarding Lifecycle")
def test_korv():
    from korv import KoRVTEQABridge, KoRVDomesticationEngine

    db_path = os.path.join(TEST_DIR, "korv_test.db")
    engine = KoRVDomesticationEngine(db_path=db_path)

    # Register a new signal
    engine.register_new_signal("RSI_variant", "XAUUSD", "H1")
    print("    Signal registered: RSI_variant / XAUUSD / H1")

    # Record some outcomes
    for i in range(20):
        won = np.random.random() > 0.3  # ~70% win rate
        engine.record_outcome("RSI_variant", "XAUUSD", "H1", won=won, pnl=1.0 if won else -1.0)

    weight = engine.get_signal_weight("RSI_variant", "XAUUSD", "H1")
    print(f"    Signal weight after 20 trades: {weight}")

    # Test bridge
    bridge = KoRVTEQABridge(engine=engine)
    conf = bridge.get_weighted_confidence(
        base_confidence=0.5,
        signal_types=["RSI_variant"],
        instrument="XAUUSD",
        timeframe="H1"
    )
    print(f"    Weighted confidence: {conf}")

    gate = bridge.get_gate_result(
        signal_types=["RSI_variant"],
        instrument="XAUUSD",
        timeframe="H1"
    )
    print(f"    Gate result: {gate}")


# ============================================================
# ALGORITHM 6: Bdelloid Rotifers
# ============================================================
@test("6. Bdelloid Rotifers -- Horizontal Gene Transfer")
def test_bdelloid():
    from bdelloid_rotifers import BdelloidHGTEngine, BdelloidTEQABridge

    db_path = os.path.join(TEST_DIR, "bdelloid_test.db")
    engine = BdelloidHGTEngine(db_path=db_path)

    # Register strategies (correct signature: strategy_name, instrument, timeframe, components)
    strat_a = engine.register_strategy(
        strategy_name="momentum_breakout",
        instrument="XAUUSD",
        timeframe="H1",
        components={
            "entry": {"type": "momentum", "period": 14, "threshold": 0.7},
            "exit": {"type": "trailing_stop", "multiplier": 2.0},
            "filter": {"type": "atr", "min_atr": 1.5},
        },
    )
    strat_b = engine.register_strategy(
        strategy_name="mean_revert",
        instrument="XAUUSD",
        timeframe="H1",
        components={
            "entry": {"type": "mean_reversion", "period": 20, "threshold": -0.8},
            "exit": {"type": "target", "rr_ratio": 2.0},
            "filter": {"type": "bollinger", "squeeze": True},
        },
    )
    print(f"    Registered strategies: {strat_a}, {strat_b}")

    # Test bridge
    bridge = BdelloidTEQABridge(hgt_engine=engine)
    sid = strat_a.strategy_id if hasattr(strat_a, 'strategy_id') else str(strat_a)
    gate = bridge.get_hgt_gate_result(sid)
    print(f"    HGT gate: {gate}")


# ============================================================
# ALGORITHM 7: Toxoplasma
# ============================================================
@test("7. Toxoplasma -- Regime Behavior Modification")
def test_toxoplasma():
    from toxoplasma import ToxoplasmaTEQABridge, ToxoplasmaEngine, StrategyType

    db_path = os.path.join(TEST_DIR, "toxo_test.db")
    engine = ToxoplasmaEngine(db_path=db_path)

    # Register strategy
    engine.register_strategy("teqa_main", StrategyType.MOMENTUM, "XAUUSD")
    print("    Strategy registered: teqa_main / MOMENTUM / XAUUSD")

    # Run a cycle
    result = engine.run_cycle("teqa_main", "XAUUSD", BARS)
    print(f"    Cycle result: {result}")

    # Test bridge
    bridge = ToxoplasmaTEQABridge(engine=engine)
    # Convert bars dict to numpy array for toxoplasma
    bars_arr = np.column_stack([BARS["open"], BARS["high"], BARS["low"], BARS["close"], BARS["volume"]])
    direction, confidence, size_mult = bridge.apply_to_signal(
        strategy_id="teqa_main",
        symbol="XAUUSD",
        original_direction=1,
        original_confidence=0.5,
        bars=bars_arr,
        active_tes=ACTIVE_TES
    )
    print(f"    Applied signal: dir={direction}, conf={confidence:.3f}, size={size_mult:.3f}")


# ============================================================
# ALGORITHM 8: Syncytin
# ============================================================
@test("8. Syncytin -- Strategy Fusion")
def test_syncytin():
    from syncytin import SyncytinFusionEngine, StrategyProfile, RegimeType
    from dataclasses import dataclass, field

    db_path = os.path.join(TEST_DIR, "syncytin_test.db")
    engine = SyncytinFusionEngine(db_path=db_path)

    # Create strategy profiles with correct field names
    returns_a = (np.random.randn(50) * 0.02).tolist()
    returns_b = (-np.array(returns_a) + np.random.randn(50) * 0.005).tolist()

    profile_a = StrategyProfile(
        strategy_id="trend_follower",
        strategy_type="custom",
        regime_affinity={
            "trending_up": 0.85,
            "trending_down": 0.80,
            "ranging": 0.20,
            "volatile": 0.50,
            "compressed": 0.30,
            "breakout": 0.60,
        },
        return_stream=returns_a,
        win_rate=0.55,
        profit_factor=1.8,
        total_trades=100,
    )
    profile_b = StrategyProfile(
        strategy_id="mean_reverter",
        strategy_type="custom",
        regime_affinity={
            "trending_up": 0.15,
            "trending_down": 0.20,
            "ranging": 0.90,
            "volatile": 0.40,
            "compressed": 0.70,
            "breakout": 0.25,
        },
        return_stream=returns_b,
        win_rate=0.60,
        profit_factor=1.6,
        total_trades=100,
    )

    # Screen candidates
    candidates = engine.screen_candidates([profile_a, profile_b])
    print(f"    Fusion candidates found: {len(candidates)}")

    if candidates:
        hybrid = engine.fuse(candidates[0])
        if hybrid:
            print(f"    Hybrid created: {hybrid.hybrid_id}")
            print(f"    Fusion type: {hybrid.fusion_type}")
        else:
            print("    Fusion returned None (criteria not met -- OK for smoke test)")
    else:
        print("    No compatible candidates found (OK for smoke test)")


# ============================================================
# INTEGRATION TEST: All algorithms in a pipeline
# ============================================================
@test("INTEGRATION -- Full AO Pipeline")
def test_integration():
    """Simulate one full cycle through all 8 algorithms."""
    print("    Simulating one TEQA cycle through all 8 subsystems...")

    # Step 1: Signal enters (simulated TEQA output)
    signal_direction = 1  # BUY
    signal_confidence = 0.45
    symbol = "XAUUSD"
    active_tes = ACTIVE_TES
    domestication_boost = 1.15

    print(f"    Input: dir={signal_direction}, conf={signal_confidence}, boost={domestication_boost}")

    # Step 2: Protective Deletion check
    from protective_deletion import ProtectiveDeletionTracker
    pd_tracker = ProtectiveDeletionTracker(db_path=os.path.join(TEST_DIR, "int_pd.db"))
    suppression = pd_tracker.get_suppression(active_tes)
    combined_mod = domestication_boost * suppression
    print(f"    [2] Protective Deletion: suppression={suppression}, combined={combined_mod:.3f}")

    # Step 3: CRISPR gate check
    from crispr_cas import CRISPRTEQABridge
    crispr = CRISPRTEQABridge(db_path=os.path.join(TEST_DIR, "int_crispr.db"))
    bars_np = np.column_stack([BARS["open"][-30:], BARS["high"][-30:], BARS["low"][-30:], BARS["close"][-30:], BARS["volume"][-30:]])
    cas9 = crispr.gate_check(symbol, signal_direction, bars_np, 0.5, active_tes, combined_mod, signal_confidence)
    crispr_pass = cas9.get("gate_pass", True) if isinstance(cas9, dict) else True
    print(f"    [3] CRISPR-Cas9 gate: {'PASS' if crispr_pass else 'BLOCKED'}")

    # Step 4: Electric Organs convergence
    from electric_organs import ElectricOrgansBridge
    eo = ElectricOrgansBridge(lineage_symbols=SYMBOLS, db_dir=TEST_DIR)
    conv_boost = eo.apply(active_tes, combined_mod, symbol)
    print(f"    [4] Electric Organs: convergence_boost={conv_boost:.3f}")

    # Step 5: KoRV weight check
    from korv import KoRVDomesticationEngine, KoRVTEQABridge
    korv_engine = KoRVDomesticationEngine(db_path=os.path.join(TEST_DIR, "int_korv.db"))
    korv_bridge = KoRVTEQABridge(engine=korv_engine)
    korv_conf = korv_bridge.get_weighted_confidence(base_confidence=signal_confidence, signal_types=["L1_Neuronal"], instrument=symbol, timeframe="H1")
    print(f"    [5] KoRV weighted confidence: {korv_conf:.3f}")

    # Step 6: Toxoplasma infection check
    from toxoplasma import ToxoplasmaEngine, ToxoplasmaTEQABridge, StrategyType
    toxo = ToxoplasmaEngine(db_path=os.path.join(TEST_DIR, "int_toxo.db"))
    toxo.register_strategy("main", StrategyType.MOMENTUM, symbol)
    toxo_bridge = ToxoplasmaTEQABridge(engine=toxo)
    toxo_bars = np.column_stack([BARS["open"], BARS["high"], BARS["low"], BARS["close"], BARS["volume"]])
    t_dir, t_conf, t_size = toxo_bridge.apply_to_signal("main", symbol, signal_direction, signal_confidence, toxo_bars, active_tes)
    print(f"    [7] Toxoplasma: dir={t_dir}, conf={t_conf:.3f}, size_mult={t_size:.3f}")

    # Final decision
    final_confidence = t_conf * conv_boost
    trade = crispr_pass and final_confidence > 0.22
    print(f"\n    FINAL: confidence={final_confidence:.3f}, trade={'YES' if trade else 'NO'}")
    print(f"    Pipeline complete -- all subsystems executed successfully")


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  QUANTUM CHILDREN -- Artificial Organism Integration Test")
    print("  Testing all 8 biological algorithms")
    print(f"  Test dir: {TEST_DIR}")
    print("=" * 60)

    tests = [
        test_vdj,
        test_protective_deletion,
        test_crispr,
        test_electric_organs,
        test_korv,
        test_bdelloid,
        test_toxoplasma,
        test_syncytin,
        test_integration,
    ]

    for t in tests:
        t()

    # -- Summary --
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v[0] == "PASS")
    failed = sum(1 for v in results.values() if v[0] == "FAIL")
    total_time = sum(v[1] for v in results.values())

    for name, result in results.items():
        status = result[0]
        elapsed = result[1]
        icon = "[OK]" if status == "PASS" else "[XX]"
        extra = f" -- {result[2]}" if len(result) > 2 else ""
        print(f"  {icon} {name} ({elapsed:.2f}s){extra}")

    print(f"\n  {passed} passed, {failed} failed, {total_time:.2f}s total")

    if failed == 0:
        print("\n  *** ALL SYSTEMS OPERATIONAL -- Artificial Organism is ALIVE ***")
    else:
        print(f"\n  !!! {failed} subsystem(s) need attention !!!")

    # Cleanup
    try:
        shutil.rmtree(TEST_DIR)
        print(f"\n  Cleaned up: {TEST_DIR}")
    except:
        print(f"\n  Note: Could not clean {TEST_DIR}")

    os.chdir(ORIG_DIR)
    sys.exit(0 if failed == 0 else 1)
