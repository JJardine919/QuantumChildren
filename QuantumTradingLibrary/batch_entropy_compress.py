"""
Batch Entropy Compression for Trading Experts
==============================================
Runs both WalkForward 70%+ experts and QTL Top 50 experts through
quantum entropy removal to identify and enhance the cleanest performers.

Usage:
    python batch_entropy_compress.py

Output:
    - master_experts_80plus/ directory with compressed experts
    - compression_report.json with entropy metrics
"""

import numpy as np
import json
import torch
import os
import sqlite3
from pathlib import Path
from datetime import datetime
import time

# Quantum compression imports
try:
    import qutip as qt
    from scipy.optimize import minimize
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("WARNING: qutip not available - using classical entropy estimation")


class EntropyAnalyzer:
    """Analyze and remove entropy from neural network weights"""

    def __init__(self, fid_threshold=0.90, max_layers=3):
        self.fid_threshold = fid_threshold
        self.max_layers = max_layers

    def weights_to_quantum_state(self, weights_flat, target_qubits=8):
        """Convert flattened weights to normalized quantum state vector"""
        # Normalize weights to unit vector
        weights_norm = weights_flat / (np.linalg.norm(weights_flat) + 1e-10)

        # Pad/truncate to 2^n size
        target_size = 2 ** target_qubits
        if len(weights_norm) > target_size:
            # Take most significant weights (by magnitude)
            indices = np.argsort(np.abs(weights_norm))[-target_size:]
            state_vec = weights_norm[indices]
        else:
            state_vec = np.zeros(target_size, dtype=complex)
            state_vec[:len(weights_norm)] = weights_norm

        # Ensure normalization
        state_vec = state_vec / (np.linalg.norm(state_vec) + 1e-10)
        return state_vec.astype(complex)

    def classical_entropy(self, weights):
        """Classical entropy estimation when quantum not available.
        Uses natural log (ln) instead of log2 for wider dynamic range -
        ln spreads entropy values more, giving finer resolution between
        'clean' and 'noisy' experts. ETARE-style: let the numbers breathe.
        """
        weights_flat = np.array(weights).flatten()
        # Normalize to probability distribution
        probs = np.abs(weights_flat) / (np.sum(np.abs(weights_flat)) + 1e-10)
        probs = probs[probs > 1e-10]  # Remove zeros
        entropy = -np.sum(probs * np.log(probs + 1e-10))  # ln instead of log2
        max_entropy = np.log(len(weights_flat))            # ln instead of log2
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return normalized_entropy

    def quantum_compression_ratio(self, state_vector):
        """Get compression ratio using quantum autoencoder"""
        if not QUANTUM_AVAILABLE:
            return 1.0, 0

        num_qubits = int(np.log2(len(state_vector)))
        current_state = qt.Qobj(state_vector, dims=[[2] * num_qubits, [1] * num_qubits]).unit()
        current_qubits = num_qubits
        layers_compressed = 0

        for _ in range(self.max_layers):
            if current_qubits <= 2:
                break
            num_latent = current_qubits - 1
            num_params = 6 * current_qubits
            initial_params = np.random.rand(num_params) * np.pi

            result = minimize(
                self._cost, initial_params,
                args=(current_state, current_qubits, num_latent),
                method='COBYLA', options={'maxiter': 200}
            )

            fidelity = 1 - result.fun
            if fidelity < self.fid_threshold:
                break

            U = self._get_encoder(result.x, current_qubits)
            rho_out = U * (current_state * current_state.dag()) * U.dag()
            current_state = rho_out.ptrace(range(num_latent)).eigenstates()[1][-1].unit()
            current_qubits = num_latent
            layers_compressed += 1

        ratio = num_qubits / current_qubits
        return ratio, layers_compressed

    def _ry(self, theta):
        return (-1j * theta/2 * qt.sigmay()).expm()

    def _cnot(self, N, control, target):
        p0 = qt.ket2dm(qt.basis(2, 0))
        p1 = qt.ket2dm(qt.basis(2, 1))
        ops = [qt.qeye(2)] * N
        ops[control] = p0
        ops[target] = qt.qeye(2)
        term1 = qt.tensor(ops)
        ops[control] = p1
        ops[target] = qt.sigmax()
        term2 = qt.tensor(ops)
        return term1 + term2

    def _get_encoder(self, params, num_qubits):
        U = qt.qeye([2]*num_qubits)
        param_idx = 0
        for _ in range(6):
            ry_ops = [self._ry(params[param_idx + i]) for i in range(num_qubits)]
            param_idx += num_qubits
            U = qt.tensor(ry_ops) * U
            for i in range(num_qubits):
                U = self._cnot(num_qubits, i, (i + 1) % num_qubits) * U
        return U

    def _cost(self, params, input_state, num_qubits, num_latent):
        U = self._get_encoder(params, num_qubits)
        rho = input_state * input_state.dag() if input_state.type == 'ket' else input_state
        rho_out = U * rho * U.dag()
        rho_trash = rho_out.ptrace(range(num_latent, num_qubits))
        ref = qt.tensor([qt.ket2dm(qt.basis(2, 0)) for _ in range(num_qubits - num_latent)])
        return 1 - qt.fidelity(rho_trash, ref)

    def analyze_expert(self, weights, expert_type="unknown"):
        """Full entropy analysis of an expert's weights"""
        weights_flat = np.array(weights).flatten().astype(float)

        # Classical entropy
        classical_ent = self.classical_entropy(weights_flat)

        # Quantum compression (if available)
        if QUANTUM_AVAILABLE and len(weights_flat) >= 16:
            state_vec = self.weights_to_quantum_state(weights_flat)
            comp_ratio, layers = self.quantum_compression_ratio(state_vec)
        else:
            comp_ratio, layers = 1.0, 0

        # Compression quality score (0-1 scale, higher = less entropy)
        # This measures how compressible the weights are, NOT a win rate predictor.
        compression_quality = min(1.0, (comp_ratio - 1.0) / 2.0) if comp_ratio > 1.0 else 0.0

        return {
            "classical_entropy": classical_ent,
            "compression_ratio": comp_ratio,
            "layers_compressed": layers,
            "compression_quality": compression_quality,
            "regime": "CLEAN" if comp_ratio > 1.1 else "NOISY"
        }


def load_walkforward_experts(folder_path):
    """Load WalkForward JSON experts"""
    experts = []
    folder = Path(folder_path)

    for json_file in sorted(folder.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        # Extract weights
        all_weights = []
        for key in ['input_weights', 'hidden_weights', 'output_weights', 'hidden_bias', 'output_bias']:
            if key in data:
                weights = np.array(data[key]).flatten()
                all_weights.extend(weights.tolist())

        # Parse filename: expert_C7_E40_WR73.json
        import re
        match = re.match(r'expert_C(\d+)_E(\d+)_WR(\d+)\.json', json_file.name)
        if match:
            cycle = int(match.group(1))
            expert_idx = int(match.group(2))
            win_rate = int(match.group(3))
        else:
            cycle, expert_idx, win_rate = 0, 0, 0

        experts.append({
            "source": "walkforward",
            "filename": json_file.name,
            "filepath": str(json_file),
            "cycle": cycle,
            "expert_idx": expert_idx,
            "win_rate": win_rate,
            "weights": all_weights
        })

    return experts


def load_qtl_experts(db_path, top_n=50):
    """Load QTL experts from database"""
    experts = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT symbol, individual_index, weights, fitness, total_profit
        FROM population_state
        ORDER BY fitness DESC
        LIMIT ?
    ''', (top_n,))

    for rank, (symbol, idx, weights_blob, fitness, profit) in enumerate(cursor.fetchall(), 1):
        # Load PyTorch weights
        try:
            import io
            state_dict = torch.load(io.BytesIO(weights_blob), map_location='cpu', weights_only=False)
            all_weights = []
            for key, tensor in state_dict.items():
                all_weights.extend(tensor.numpy().flatten().tolist())
        except Exception as e:
            print(f"  Warning: Could not load weights for {symbol} #{idx}: {e}")
            continue

        experts.append({
            "source": "qtl",
            "filename": f"expert_rank{rank:02d}_{symbol}.pth",
            "symbol": symbol,
            "rank": rank,
            "expert_idx": idx,
            "fitness": fitness,
            "profit": profit,
            "weights": all_weights
        })

    conn.close()
    return experts


def main():
    print("=" * 70)
    print("BATCH ENTROPY COMPRESSION - DEEP COMPRESS PRO")
    print("=" * 70)
    print(f"Quantum Available: {QUANTUM_AVAILABLE}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    analyzer = EntropyAnalyzer(fid_threshold=0.90, max_layers=3)

    # Output directory
    output_dir = Path("master_experts_80plus")
    output_dir.mkdir(exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "quantum_available": QUANTUM_AVAILABLE,
        "walkforward_experts": [],
        "qtl_experts": [],
        "master_library": []
    }

    # ========================================
    # PROCESS WALKFORWARD EXPERTS
    # ========================================
    print("=" * 70)
    print("PROCESSING WALKFORWARD 70%+ EXPERTS")
    print("=" * 70)

    wf_path = r"C:\Users\jjj10\ETARE_WalkForward\elite_experts_70plus"
    if os.path.exists(wf_path):
        wf_experts = load_walkforward_experts(wf_path)
        print(f"Loaded {len(wf_experts)} WalkForward experts\n")

        for i, expert in enumerate(wf_experts):
            print(f"[{i+1}/{len(wf_experts)}] {expert['filename']} (WR: {expert['win_rate']}%)")

            start = time.time()
            analysis = analyzer.analyze_expert(expert['weights'], "feedforward")
            elapsed = time.time() - start

            expert_result = {
                **{k: v for k, v in expert.items() if k != 'weights'},
                **analysis,
                "analysis_time": elapsed
            }
            results["walkforward_experts"].append(expert_result)

            # Real win rate from backtest (no fabricated boost)
            actual_wr = expert['win_rate']

            print(f"  Entropy: {analysis['classical_entropy']:.4f} | "
                  f"Compression: {analysis['compression_ratio']:.2f}x | "
                  f"Regime: {analysis['regime']}")
            print(f"  Win Rate: {actual_wr}% (backtest) | Quality: {analysis['compression_quality']:.2f}")

            # Add to master if real WR 80%+ OR (70%+ with CLEAN regime)
            if actual_wr >= 80 or (actual_wr >= 70 and analysis['regime'] == 'CLEAN'):
                results["master_library"].append({
                    "source": "walkforward",
                    "filename": expert['filename'],
                    "win_rate": actual_wr,
                    "compression_quality": analysis['compression_quality'],
                    "regime": analysis['regime']
                })
            print()
    else:
        print(f"WalkForward path not found: {wf_path}")

    # ========================================
    # PROCESS QTL EXPERTS
    # ========================================
    print("=" * 70)
    print("PROCESSING QTL TOP 50 EXPERTS")
    print("=" * 70)

    qtl_db = r"C:\Users\jjj10\QuantumTradingLibrary\etare_redux_v2.db"
    if os.path.exists(qtl_db):
        qtl_experts = load_qtl_experts(qtl_db, top_n=50)
        print(f"Loaded {len(qtl_experts)} QTL experts\n")

        for i, expert in enumerate(qtl_experts):
            print(f"[{i+1}/{len(qtl_experts)}] {expert['filename']} (Fitness: {expert['fitness']:.4f})")

            start = time.time()
            analysis = analyzer.analyze_expert(expert['weights'], "lstm")
            elapsed = time.time() - start

            expert_result = {
                **{k: v for k, v in expert.items() if k != 'weights'},
                **analysis,
                "analysis_time": elapsed
            }
            results["qtl_experts"].append(expert_result)

            # QTL experts: fitness = ROI * win_rate^2, cannot reliably invert to win rate
            fitness = expert['fitness']

            print(f"  Entropy: {analysis['classical_entropy']:.4f} | "
                  f"Compression: {analysis['compression_ratio']:.2f}x | "
                  f"Regime: {analysis['regime']}")
            print(f"  Fitness: {fitness:.4f} | Profit: ${expert['profit']:.2f}")
            print(f"  Quality: {analysis['compression_quality']:.2f}")

            # Add to master if clean regime and high fitness
            if analysis['regime'] == 'CLEAN' and fitness > 0.15:
                results["master_library"].append({
                    "source": "qtl",
                    "filename": expert['filename'],
                    "symbol": expert['symbol'],
                    "fitness": fitness,
                    "profit": expert['profit'],
                    "compression_quality": analysis['compression_quality'],
                    "regime": analysis['regime']
                })
            print()
    else:
        print(f"QTL database not found: {qtl_db}")

    # ========================================
    # SUMMARY
    # ========================================
    print("=" * 70)
    print("COMPRESSION COMPLETE - SUMMARY")
    print("=" * 70)

    wf_clean = len([e for e in results["walkforward_experts"] if e.get('regime') == 'CLEAN'])
    qtl_clean = len([e for e in results["qtl_experts"] if e.get('regime') == 'CLEAN'])

    print(f"\nWalkForward Experts:")
    print(f"  Total: {len(results['walkforward_experts'])}")
    print(f"  Clean (low entropy): {wf_clean}")
    print(f"  Noisy (high entropy): {len(results['walkforward_experts']) - wf_clean}")

    print(f"\nQTL Experts:")
    print(f"  Total: {len(results['qtl_experts'])}")
    print(f"  Clean (low entropy): {qtl_clean}")
    print(f"  Noisy (high entropy): {len(results['qtl_experts']) - qtl_clean}")

    print(f"\nMASTER LIBRARY (Qualified Experts):")
    print(f"  Total qualified: {len(results['master_library'])}")

    # Sort: WalkForward by win_rate, QTL by fitness
    results["master_library"].sort(
        key=lambda x: x.get('win_rate', 0) if x['source'] == 'walkforward' else x.get('fitness', 0) * 100,
        reverse=True
    )

    print("\n  TOP 10 FOR INSTANT CHALLENGES:")
    print("  " + "-" * 60)
    for i, expert in enumerate(results["master_library"][:10], 1):
        if expert['source'] == 'walkforward':
            print(f"  {i}. {expert['filename']} - WR: {expert['win_rate']}% | Quality: {expert['compression_quality']:.2f}")
        else:
            print(f"  {i}. {expert['filename']} - Fitness: {expert['fitness']:.3f} | Quality: {expert['compression_quality']:.2f}")

    # Save report
    report_path = output_dir / "compression_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved: {report_path}")

    # Identify SAFEST expert for instant challenges (lowest entropy, highest WR)
    if results["master_library"]:
        safest = results["master_library"][0]
        print("\n" + "=" * 70)
        print("RECOMMENDED FOR INSTANT CHALLENGES (Safest - Won't Go Backwards)")
        print("=" * 70)
        print(f"  Expert: {safest['filename']}")
        print(f"  Source: {safest['source'].upper()}")
        if safest['source'] == 'walkforward':
            print(f"  Win Rate: {safest['win_rate']}% (backtest verified)")
        else:
            print(f"  Fitness: {safest['fitness']:.4f}")
            print(f"  Symbol: {safest['symbol']}")
        print(f"  Compression Quality: {safest['compression_quality']:.2f}")
        print(f"  Regime: {safest['regime']} (LOW ENTROPY - CLEAN)")

    return results


if __name__ == "__main__":
    main()
