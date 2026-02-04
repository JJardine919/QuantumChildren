"""
Quantum Hardware vs Simulator Comparison
=========================================
Runs the same quantum feature extraction circuit on:
1. AerSimulator (classical simulation)
2. Real IBM Quantum hardware

Compares entropy and other quantum features to see if there's
a measurable difference in win rate.

Usage:
    python quantum_hardware_comparison.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# IBM Quantum Runtime
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    print("Warning: qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime")

# MT5 for market data
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
N_QUBITS = 3
N_SHOTS = 1000
QUANTUM_WINDOW = 50
TEST_SAMPLES = 100  # Number of price windows to compare

# ============================================================================
# QUANTUM CIRCUIT (EXACT SAME AS YOUR PRODUCTION CODE)
# ============================================================================
def create_quantum_circuit(features: np.ndarray, num_qubits: int = 3) -> QuantumCircuit:
    """Create quantum circuit - identical to quantum_lstm_system.py"""
    qc = QuantumCircuit(num_qubits, num_qubits)

    # RY rotations for encoding
    for i in range(num_qubits):
        feature_idx = i % len(features)
        angle = np.clip(np.pi * features[feature_idx], -2*np.pi, 2*np.pi)
        qc.ry(angle, i)

    # CNOT entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    # Measurements
    qc.measure(range(num_qubits), range(num_qubits))
    return qc

def compute_quantum_metrics(counts: dict, shots: int, num_qubits: int = 3) -> dict:
    """Compute 7 quantum features - identical to quantum_lstm_system.py"""
    probabilities = {state: count/shots for state, count in counts.items()}

    # 1. Shannon entropy
    quantum_entropy = -sum(p * np.log2(p) if p > 0 else 0
                          for p in probabilities.values())

    # 2. Dominant state probability
    dominant_state_prob = max(probabilities.values())

    # 3. Superposition measure
    threshold = 0.05
    significant_states = sum(1 for p in probabilities.values() if p > threshold)
    superposition_measure = significant_states / (2 ** num_qubits)

    # 4. Phase coherence
    state_values = [int(state, 2) for state in probabilities.keys()]
    max_value = 2 ** num_qubits - 1
    phase_coherence = 1.0 - (np.std(state_values) / max_value) if len(state_values) > 1 else 0.5

    # 5. Entanglement degree
    bit_correlations = []
    for i in range(num_qubits - 1):
        correlation = 0.0
        for state, prob in probabilities.items():
            if len(state) > i + 1:
                if state[-(i+1)] == state[-(i+2)]:
                    correlation += prob
        bit_correlations.append(correlation)
    entanglement_degree = np.mean(bit_correlations) if bit_correlations else 0.5

    # 6. Quantum variance
    mean_state = sum(int(state, 2) * prob for state, prob in probabilities.items())
    quantum_variance = sum((int(state, 2) - mean_state)**2 * prob
                          for state, prob in probabilities.items())

    # 7. Significant states count
    num_significant_states = float(significant_states)

    return {
        'quantum_entropy': quantum_entropy,
        'dominant_state_prob': dominant_state_prob,
        'superposition_measure': superposition_measure,
        'phase_coherence': phase_coherence,
        'entanglement_degree': entanglement_degree,
        'quantum_variance': quantum_variance,
        'num_significant_states': num_significant_states
    }

# ============================================================================
# SIMULATOR EXTRACTION
# ============================================================================
def extract_features_simulator(price_window: np.ndarray) -> dict:
    """Extract quantum features using AerSimulator (your current production method)"""
    returns = np.diff(price_window) / (price_window[:-1] + 1e-10)
    features = np.array([
        np.mean(returns),
        np.std(returns),
        np.max(returns) - np.min(returns)
    ])
    features = np.tanh(features)

    simulator = AerSimulator(method='statevector')
    qc = create_quantum_circuit(features, N_QUBITS)
    compiled = transpile(qc, simulator, optimization_level=2)
    job = simulator.run(compiled, shots=N_SHOTS)
    counts = job.result().get_counts()

    return compute_quantum_metrics(counts, N_SHOTS, N_QUBITS)

# ============================================================================
# REAL IBM QUANTUM EXTRACTION
# ============================================================================
def extract_features_ibm_hardware(price_window: np.ndarray, sampler, backend) -> dict:
    """Extract quantum features using real IBM Quantum hardware"""
    returns = np.diff(price_window) / (price_window[:-1] + 1e-10)
    features = np.array([
        np.mean(returns),
        np.std(returns),
        np.max(returns) - np.min(returns)
    ])
    features = np.tanh(features)

    qc = create_quantum_circuit(features, N_QUBITS)
    compiled = transpile(qc, backend, optimization_level=3)

    # Run on real hardware
    job = sampler.run([compiled], shots=N_SHOTS)
    result = job.result()

    # Extract counts from SamplerV2 result
    pub_result = result[0]
    counts_raw = pub_result.data.c.get_counts()

    # Normalize counts format
    counts = {}
    for bitstring, count in counts_raw.items():
        # Ensure consistent format
        key = format(int(bitstring, 2) if isinstance(bitstring, str) else bitstring, f'0{N_QUBITS}b')
        counts[key] = count

    return compute_quantum_metrics(counts, N_SHOTS, N_QUBITS)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_test_data(symbol: str = "BTCUSD", n_windows: int = TEST_SAMPLES) -> list:
    """Load price windows for testing"""
    if MT5_AVAILABLE and mt5.initialize():
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_windows + QUANTUM_WINDOW + 50)
        mt5.shutdown()

        if rates is not None and len(rates) > QUANTUM_WINDOW:
            df = pd.DataFrame(rates)
            windows = []
            for i in range(QUANTUM_WINDOW, min(len(df), n_windows + QUANTUM_WINDOW)):
                window = df['close'].iloc[i-QUANTUM_WINDOW:i].values
                future_price = df['close'].iloc[i] if i < len(df) else None
                current_price = df['close'].iloc[i-1]
                windows.append({
                    'prices': window,
                    'current_price': current_price,
                    'future_price': future_price,
                    'actual_direction': 'UP' if future_price and future_price > current_price else 'DOWN'
                })
            return windows

    # Fallback: synthetic data
    print("Using synthetic test data (MT5 not available)")
    windows = []
    base_price = 100000
    for i in range(n_windows):
        noise = np.random.randn(QUANTUM_WINDOW) * 100
        trend = np.linspace(0, np.random.randn() * 500, QUANTUM_WINDOW)
        prices = base_price + noise.cumsum() + trend
        future_move = np.random.randn() * 200
        windows.append({
            'prices': prices,
            'current_price': prices[-1],
            'future_price': prices[-1] + future_move,
            'actual_direction': 'UP' if future_move > 0 else 'DOWN'
        })
        base_price = prices[-1]
    return windows

# ============================================================================
# TRADING DECISION LOGIC
# ============================================================================
def make_trading_decision(features: dict, entropy_threshold: float = 2.5) -> dict:
    """
    Simplified trading decision based on quantum features.
    Returns whether we would trade and the predicted direction.
    """
    entropy = features['quantum_entropy']
    dominant = features['dominant_state_prob']
    coherence = features['phase_coherence']

    # High entropy = don't trade (market is random)
    if entropy > entropy_threshold:
        return {'trade': False, 'direction': None, 'reason': 'high_entropy'}

    # Low entropy + high dominant = trade
    if entropy < 2.0 and dominant > 0.2:
        # Direction based on coherence and variance
        if coherence > 0.6:
            direction = 'UP' if features['quantum_variance'] < 0.01 else 'DOWN'
        else:
            direction = 'DOWN' if features['quantum_variance'] > 0.01 else 'UP'
        return {'trade': True, 'direction': direction, 'reason': 'low_entropy_high_dominant'}

    return {'trade': False, 'direction': None, 'reason': 'uncertain'}

# ============================================================================
# MAIN COMPARISON
# ============================================================================
def run_comparison():
    print("=" * 80)
    print("QUANTUM HARDWARE vs SIMULATOR COMPARISON")
    print("=" * 80)
    print(f"Qubits: {N_QUBITS}")
    print(f"Shots: {N_SHOTS}")
    print(f"Test samples: {TEST_SAMPLES}")
    print("=" * 80)

    # Check IBM availability
    if not IBM_AVAILABLE:
        print("\nERROR: qiskit-ibm-runtime not installed")
        print("Install with: pip install qiskit-ibm-runtime")
        return

    # Initialize IBM Quantum
    print("\nConnecting to IBM Quantum...")
    try:
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=N_QUBITS)
        print(f"Selected backend: {backend.name}")
        print(f"Queue depth: {backend.status().pending_jobs}")

        sampler = Sampler(backend)
    except Exception as e:
        print(f"ERROR connecting to IBM Quantum: {e}")
        print("\nMake sure you have saved your IBM Quantum credentials:")
        print("  from qiskit_ibm_runtime import QiskitRuntimeService")
        print("  QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
        return

    # Load test data
    print("\nLoading test data...")
    test_windows = load_test_data("BTCUSD", TEST_SAMPLES)
    print(f"Loaded {len(test_windows)} test windows")

    # Results storage
    results = []

    print("\n" + "=" * 80)
    print("RUNNING COMPARISON (this will take a while due to quantum hardware queue)")
    print("=" * 80)

    for idx, window_data in enumerate(test_windows):
        print(f"\n[{idx+1}/{len(test_windows)}] Processing...")

        prices = window_data['prices']
        actual_direction = window_data['actual_direction']

        # Run on simulator
        start_sim = time.time()
        sim_features = extract_features_simulator(prices)
        sim_time = time.time() - start_sim

        # Run on real hardware
        start_hw = time.time()
        try:
            hw_features = extract_features_ibm_hardware(prices, sampler, backend)
            hw_time = time.time() - start_hw
            hw_success = True
        except Exception as e:
            print(f"  Hardware error: {e}")
            hw_features = None
            hw_time = 0
            hw_success = False

        # Trading decisions
        sim_decision = make_trading_decision(sim_features)
        hw_decision = make_trading_decision(hw_features) if hw_features else {'trade': False, 'direction': None}

        # Check correctness
        sim_correct = sim_decision['direction'] == actual_direction if sim_decision['trade'] else None
        hw_correct = hw_decision['direction'] == actual_direction if hw_decision['trade'] else None

        result = {
            'index': idx,
            'actual_direction': actual_direction,
            'sim_entropy': sim_features['quantum_entropy'],
            'sim_dominant': sim_features['dominant_state_prob'],
            'sim_coherence': sim_features['phase_coherence'],
            'sim_trade': sim_decision['trade'],
            'sim_direction': sim_decision['direction'],
            'sim_correct': sim_correct,
            'sim_time': sim_time,
            'hw_entropy': hw_features['quantum_entropy'] if hw_features else None,
            'hw_dominant': hw_features['dominant_state_prob'] if hw_features else None,
            'hw_coherence': hw_features['phase_coherence'] if hw_features else None,
            'hw_trade': hw_decision['trade'],
            'hw_direction': hw_decision['direction'],
            'hw_correct': hw_correct,
            'hw_time': hw_time,
            'hw_success': hw_success
        }
        results.append(result)

        # Progress output
        entropy_diff = abs(sim_features['quantum_entropy'] - hw_features['quantum_entropy']) if hw_features else 0
        print(f"  Simulator: entropy={sim_features['quantum_entropy']:.3f}, trade={sim_decision['trade']}, time={sim_time:.2f}s")
        if hw_features:
            print(f"  Hardware:  entropy={hw_features['quantum_entropy']:.3f}, trade={hw_decision['trade']}, time={hw_time:.2f}s")
            print(f"  Entropy diff: {entropy_diff:.4f}")

        # Save intermediate results
        if (idx + 1) % 10 == 0:
            save_results(results, f"quantum_comparison_partial_{idx+1}.json")

    # Final analysis
    analyze_results(results)
    save_results(results, "quantum_comparison_final.json")

def analyze_results(results: list):
    """Analyze and display comparison results"""
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)

    # Filter successful hardware runs
    hw_success = [r for r in results if r['hw_success']]

    print(f"\nTotal samples: {len(results)}")
    print(f"Hardware successful: {len(hw_success)}")

    if not hw_success:
        print("No successful hardware runs to analyze")
        return

    # Entropy comparison
    sim_entropies = [r['sim_entropy'] for r in hw_success]
    hw_entropies = [r['hw_entropy'] for r in hw_success]
    entropy_diffs = [abs(r['sim_entropy'] - r['hw_entropy']) for r in hw_success]

    print(f"\n--- ENTROPY COMPARISON ---")
    print(f"Simulator mean entropy: {np.mean(sim_entropies):.4f} (+/- {np.std(sim_entropies):.4f})")
    print(f"Hardware mean entropy:  {np.mean(hw_entropies):.4f} (+/- {np.std(hw_entropies):.4f})")
    print(f"Mean absolute difference: {np.mean(entropy_diffs):.4f}")
    print(f"Max difference: {np.max(entropy_diffs):.4f}")

    # Trading decisions comparison
    sim_trades = sum(1 for r in hw_success if r['sim_trade'])
    hw_trades = sum(1 for r in hw_success if r['hw_trade'])

    print(f"\n--- TRADING DECISIONS ---")
    print(f"Simulator trades: {sim_trades}/{len(hw_success)} ({sim_trades/len(hw_success)*100:.1f}%)")
    print(f"Hardware trades:  {hw_trades}/{len(hw_success)} ({hw_trades/len(hw_success)*100:.1f}%)")

    # Agreement
    agreements = sum(1 for r in hw_success if r['sim_trade'] == r['hw_trade'] and r['sim_direction'] == r['hw_direction'])
    print(f"Decision agreement: {agreements}/{len(hw_success)} ({agreements/len(hw_success)*100:.1f}%)")

    # Win rates
    sim_correct = [r for r in hw_success if r['sim_correct'] is not None]
    hw_correct = [r for r in hw_success if r['hw_correct'] is not None]

    print(f"\n--- WIN RATES ---")
    if sim_correct:
        sim_wins = sum(1 for r in sim_correct if r['sim_correct'])
        print(f"Simulator win rate: {sim_wins}/{len(sim_correct)} ({sim_wins/len(sim_correct)*100:.1f}%)")

    if hw_correct:
        hw_wins = sum(1 for r in hw_correct if r['hw_correct'])
        print(f"Hardware win rate:  {hw_wins}/{len(hw_correct)} ({hw_wins/len(hw_correct)*100:.1f}%)")

    # Cases where hardware differed and was correct
    different_decisions = [r for r in hw_success if r['sim_trade'] != r['hw_trade'] or r['sim_direction'] != r['hw_direction']]
    if different_decisions:
        print(f"\n--- DIFFERENT DECISIONS ---")
        print(f"Cases where hardware decision differed: {len(different_decisions)}")

        hw_better = sum(1 for r in different_decisions if r['hw_correct'] and not r['sim_correct'])
        sim_better = sum(1 for r in different_decisions if r['sim_correct'] and not r['hw_correct'])
        print(f"Hardware was better: {hw_better}")
        print(f"Simulator was better: {sim_better}")

    # Timing
    sim_times = [r['sim_time'] for r in hw_success]
    hw_times = [r['hw_time'] for r in hw_success]

    print(f"\n--- TIMING ---")
    print(f"Simulator avg time: {np.mean(sim_times):.3f}s")
    print(f"Hardware avg time:  {np.mean(hw_times):.3f}s")
    print(f"Hardware is {np.mean(hw_times)/np.mean(sim_times):.1f}x slower")

def save_results(results: list, filename: str):
    """Save results to JSON"""
    output_dir = Path("quantum_comparison_results")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {filepath}")

# ============================================================================
# QUICK TEST (NO HARDWARE)
# ============================================================================
def quick_test_simulator_only():
    """Quick test using only simulator - for verification before hardware run"""
    print("=" * 80)
    print("QUICK TEST (Simulator only)")
    print("=" * 80)

    test_windows = load_test_data("BTCUSD", 20)

    for idx, window_data in enumerate(test_windows[:5]):
        prices = window_data['prices']
        features = extract_features_simulator(prices)
        decision = make_trading_decision(features)

        print(f"\n[{idx+1}] Entropy: {features['quantum_entropy']:.3f}, "
              f"Dominant: {features['dominant_state_prob']:.3f}, "
              f"Trade: {decision['trade']}, Direction: {decision['direction']}")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test_simulator_only()
    else:
        print("\nThis will run comparisons on real IBM Quantum hardware.")
        print("Each sample may take 1-10 minutes depending on queue.")
        print(f"Total samples: {TEST_SAMPLES}")
        print("\nTo quick test (simulator only): python quantum_hardware_comparison.py --quick")
        print("\nContinue with hardware comparison? (yes/no): ", end="")

        confirm = input().strip().lower()
        if confirm in ['yes', 'y']:
            run_comparison()
        else:
            print("Cancelled. Run with --quick for simulator-only test.")
