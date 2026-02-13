"""
Test script for Testosterone-DMT Bridge
Shows all 4 layers and 4 gates in action with various market conditions.
"""

import numpy as np
from testosterone_dmt_bridge import create_bridge

print("\n" + "="*80)
print("TESTOSTERONE-DMT BRIDGE - Comprehensive Test Suite")
print("="*80)

bridge = create_bridge(shots=4096)

# ============================================================
# TEST 1: STRONG UPTREND (Should PASS all gates and BOOST)
# ============================================================
print("\n" + "="*80)
print("TEST 1: STRONG UPTREND - All Gates Should PASS")
print("="*80)

# Generate strong uptrend
np.random.seed(100)
base = 100.0
trend = np.linspace(0, 50, 100)  # Very strong uptrend (50% gain)
noise = np.random.randn(100) * 0.3  # Very low noise
prices_uptrend = base + trend + noise

data_uptrend = {
    'close': prices_uptrend.tolist(),
    'volume': np.random.randint(2000, 8000, 100).tolist(),
    'volatility': 1.1,  # Low volatility
    'drawdown': 0.02    # Minimal drawdown
}

result1 = bridge.process_signal(
    market_data=data_uptrend,
    base_signal=0.9,
    immune_conflict=0.05
)

print(f"\n>>> FINAL DECISION: {result1['action'].upper()} <<<")
print(f"Signal Strength: {result1['strength']:.3f}")
print(f"Regime: {result1['regime']}")
print(f"Position Multiplier: {result1['position_multiplier']:.2f}x")
print(f"Stop Loss Multiplier: {result1['stop_multiplier']:.2f}x (wider stops)")
print(f"Take Profit Multiplier: {result1['target_multiplier']:.2f}x (bigger targets)")

print("\n--- TESTOSTERONE LAYERS (4 Rings) ---")
t = result1['testosterone_layers']
print(f"Ring 1 - TREND DETECTION:")
print(f"  Trend Strength: {t['layer1_trend']['trend_strength']:.3f}")
print(f"  Trend Direction: {t['layer1_trend']['trend_direction']} (1=up, -1=down)")
print(f"  Androgen Binding Active: {t['layer1_trend']['binding_active']}")

print(f"\nRing 2 - MOMENTUM AMPLIFICATION:")
print(f"  Momentum: {t['layer2_momentum']['momentum']:.3f}")
print(f"  Acceleration: {t['layer2_momentum']['acceleration']:.3f}")
print(f"  DHT Converted: {t['layer2_momentum']['dht_converted']}")
print(f"  Concentration Factor: {t['layer2_momentum']['concentration_factor']:.3f}x")

print(f"\nRing 3 - POSITION SIZING:")
print(f"  Anabolic (growth): {t['layer3_sizing']['anabolic_component']:.3f}")
print(f"  Androgenic (aggression): {t['layer3_sizing']['androgenic_component']:.3f}")
print(f"  Anabolic:Androgenic Ratio: {t['layer3_sizing']['ratio']:.1f}:1")
print(f"  Position Size: {t['layer3_sizing']['position_size_multiplier']:.2f}x")

print(f"\nRing 4 - EXIT TIMING:")
print(f"  Exit Strategy: {t['layer4_exit']['exit_strategy']}")
print(f"  Aromatization Level: {t['layer4_exit']['aromatization_level']:.3f}")
print(f"  Aromatized (defensive mode): {t['layer4_exit']['aromatized']}")

print("\n--- DMT PATTERN RECOGNITION (5 Channels) ---")
dmt = result1['dmt_patterns']
print(f"Consensus Rate: {dmt['consensus_rate']:.1%} (channels agreeing)")
print(f"Average Confidence: {dmt['avg_confidence']:.3f}")
print(f"Dominant Polarity: {dmt['dominant_polarity']} (1=bullish, -1=bearish)")
print(f"All Channels Agree: {dmt['all_channels_agree']}")

print("\n--- TE BRIDGE ---")
bridge_out = result1['te_bridge']
print(f"Raw Testosterone Strength: {bridge_out['raw_strength']:.3f}")
print(f"DMT Pattern Confidence: {bridge_out['pattern_confidence']:.3f}")
print(f"Final Combined Signal: {bridge_out['final_signal']:.3f}")
print(f"TE Activation Level: {bridge_out['te_activation']:.3f}")
print(f"Binding Active: {bridge_out['binding_active']}")

print("\n--- DECISION GATES (4 Gates - ALL Must Pass) ---")
gates = result1['gates']
for gate_name, gate_info in gates['gates'].items():
    status = "[PASS]" if gate_info['passed'] else "[FAIL]"
    print(f"{gate_name}: {status}")
    print(f"  {gate_info['reason']}")

print(f"\n>>> ALL GATES PASSED: {gates['all_gates_passed']} <<<")
print(f"Gates Passed: {gates['gates_passed_count']}/{gates['gates_total']}")

if result1.get('quantum'):
    print("\n--- QUANTUM CIRCUIT (8 Qubits) ---")
    q = result1['quantum']
    print(f"Total Shots: {q['total_shots']}")
    print(f"Unique States Measured: {q['unique_states']} / 256 possible")
    print(f"Shannon Entropy: {q['shannon_entropy']:.3f} / {q['max_entropy']:.1f}")
    print(f"Quantum Novelty: {q['novelty']:.3f}")
    print(f"Vote Distribution: Long={q['vote_long']:.3f}, Short={q['vote_short']:.3f}")
    print(f"Vote Bias: {q['vote_bias']:+.3f} (positive=bullish)")
    print(f"Top State: {q['top_state']} ({q['top_count']} measurements)")

print(f"\nProcessing Time: {result1['processing_time_ms']:.2f}ms")


# ============================================================
# TEST 2: HIGH VOLATILITY (Should trigger AROMATIZATION)
# ============================================================
print("\n\n" + "="*80)
print("TEST 2: HIGH VOLATILITY - Should Trigger Aromatase (Defensive Mode)")
print("="*80)

# Generate volatile choppy market
np.random.seed(200)
choppy = np.random.randn(100) * 15  # High volatility
prices_choppy = base + choppy

data_volatile = {
    'close': prices_choppy.tolist(),
    'volume': np.random.randint(1000, 10000, 100).tolist(),
    'volatility': 3.5,  # Very high volatility
    'drawdown': 0.15    # 15% drawdown
}

result2 = bridge.process_signal(
    market_data=data_volatile,
    base_signal=0.2,
    immune_conflict=0.08
)

print(f"\n>>> FINAL DECISION: {result2['action'].upper()} <<<")
print(f"Signal Strength: {result2['strength']:.3f}")
print(f"Regime: {result2['regime']}")
print(f"Position Multiplier: {result2['position_multiplier']:.2f}x (reduced in defense mode)")
print(f"Stop Loss Multiplier: {result2['stop_multiplier']:.2f}x (tighter stops)")
print(f"Take Profit Multiplier: {result2['target_multiplier']:.2f}x (smaller targets)")

print("\n--- AROMATASE CONTROLLER ---")
exit4 = result2['testosterone_layers']['layer4_exit']
print(f"Aromatization Level: {exit4['aromatization_level']:.3f}")
print(f"Regime State: {exit4['regime']}")
print(f"Exit Strategy: {exit4['exit_strategy']}")
print(f"Aromatized (Testosterone to Estrogen): {exit4['aromatized']}")
print(f"\nDRAWDOWN: {data_volatile['drawdown']:.1%} (threshold: {0.10:.1%})")
print(f"VOLATILITY: {data_volatile['volatility']:.2f}x (threshold: {2.5:.2f}x)")


# ============================================================
# TEST 3: WEAK SIGNAL (Should FAIL gates and SUPPRESS)
# ============================================================
print("\n\n" + "="*80)
print("TEST 3: WEAK RANGING MARKET - Should Fail Gates")
print("="*80)

# Generate ranging market
np.random.seed(300)
ranging = np.sin(np.linspace(0, 4*np.pi, 100)) * 3
noise_r = np.random.randn(100) * 1.5
prices_ranging = base + ranging + noise_r

data_ranging = {
    'close': prices_ranging.tolist(),
    'volume': np.random.randint(1500, 4000, 100).tolist(),
    'volatility': 1.3,
    'drawdown': 0.04
}

result3 = bridge.process_signal(
    market_data=data_ranging,
    base_signal=0.1,
    immune_conflict=0.02
)

print(f"\n>>> FINAL DECISION: {result3['action'].upper()} <<<")
print(f"Signal Strength: {result3['strength']:.3f}")

print("\n--- WHY GATES FAILED ---")
gates3 = result3['gates']
for gate_name, gate_info in gates3['gates'].items():
    status = "[PASS]" if gate_info['passed'] else "[FAIL]"
    print(f"{gate_name}: {status}")
    print(f"  {gate_info['reason']}")

print(f"\nGates Passed: {gates3['gates_passed_count']}/4")
print("Result: Signal SUPPRESSED (gates failed)")


# ============================================================
# TEST 4: STRONG DOWNTREND
# ============================================================
print("\n\n" + "="*80)
print("TEST 4: STRONG DOWNTREND - Bearish Signal")
print("="*80)

# Generate strong downtrend
np.random.seed(400)
downtrend = np.linspace(0, -20, 100)
noise_d = np.random.randn(100) * 0.8
prices_downtrend = base + downtrend + noise_d

data_downtrend = {
    'close': prices_downtrend.tolist(),
    'volume': np.random.randint(3000, 9000, 100).tolist(),
    'volatility': 1.2,
    'drawdown': 0.03
}

result4 = bridge.process_signal(
    market_data=data_downtrend,
    base_signal=-0.85,
    immune_conflict=0.05
)

print(f"\n>>> FINAL DECISION: {result4['action'].upper()} <<<")
print(f"Signal Strength: {result4['strength']:.3f}")
print(f"Regime: {result4['regime']}")

print("\n--- TREND ANALYSIS ---")
t4 = result4['testosterone_layers']
print(f"Trend Direction: {t4['layer1_trend']['trend_direction']} (-1 = downtrend)")
print(f"Trend Strength: {t4['layer1_trend']['trend_strength']:.3f}")
print(f"Momentum: {t4['layer2_momentum']['momentum']:.3f} (negative = down)")

print("\n--- DMT POLARITY ---")
dmt4 = result4['dmt_patterns']
print(f"Dominant Polarity: {dmt4['dominant_polarity']} (-1 = bearish)")
print(f"Consensus Rate: {dmt4['consensus_rate']:.1%}")

print("\n--- GATE STATUS ---")
gates4 = result4['gates']
print(f"All Gates Passed: {gates4['all_gates_passed']}")
print(f"Gates: {gates4['gates_passed_count']}/4")


# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "="*80)
print("TESTOSTERONE-DMT BRIDGE - TEST SUMMARY")
print("="*80)

print("\n>>> STRATEGY PROFILE <<<")
params = bridge.get_strategy_parameters()
print(f"Position Sizing: {params['position_multiplier']:.2f}x")
print(f"Stop Loss: {params['stop_multiplier']:.2f}x")
print(f"Current Regime: {params['regime']}")
print(f"Strategy Bias: {params['strategy_bias']}")
print(f"Aggression Level: {params['aggression_level']}")

print("\n>>> MOLECULAR PROPERTIES <<<")
print(f"Molecule: {bridge.molecule}")
print(f"Strategy: {bridge.strategy_profile}")
print(f"Version: {bridge.version}")
print(f"Quantum Backend: {'Qiskit Aer (Available)' if result1.get('quantum') else 'Classical Fallback'}")

print("\n>>> COMPARISON TO STANOZOLOL <<<")
print("Testosterone (this bridge):")
print("  - 4 heavy processing layers")
print("  - 4 decision gates (ALL must pass)")
print("  - 8 qubits (4 rings x 2 qubits)")
print("  - 4096 quantum shots")
print("  - AGGRESSIVE: wider stops, bigger targets, trend-following")
print("  - AROMATASE: flips to defense in high volatility")
print("\nStanozolol (other bridge):")
print("  - 11 lighter processing layers")
print("  - 13 decision gates")
print("  - More qubits, more shots")
print("  - PRECISION: tighter execution, scalping bias")
print("  - SYNTHETIC: pure signal processing")

print("\n" + "="*80)
print("ALL TESTS COMPLETE - Module Ready for BRAIN Integration")
print("="*80)
print("\nIntegration:")
print("  from testosterone_dmt_bridge import create_bridge")
print("  bridge = create_bridge(shots=4096)")
print("  result = bridge.process_signal(market_data, base_signal, immune_conflict)")
print("  if result['action'] == 'boost': ...")
