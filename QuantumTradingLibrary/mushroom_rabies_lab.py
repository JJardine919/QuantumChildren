"""
MUSHROOM RABIES LAB
====================
Takes a trained LSTM expert and subjects it to:
  1. RABIES    - Hyperexcite the neural weights (amplify gate activations)
  2. MUSHROOMS - Stochastic perturbation (novel random pathways)
  3. TEQA      - Inject transposable element quantum signals into hidden state
  4. COMPRESS  - Blast through quantum autoencoder (only coherent signals survive)

Pipeline: Expert -> Rabies -> Mushrooms -> TEQA -> Quantum Compress -> Mutant Expert

"You don't find the edge. You grow it."

Author: DooDoo + Claude
Date: 2026-02-07
"""

import sys
import io
import os
import json
import copy
import time
import logging
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import qutip as qt
    from scipy.optimize import minimize
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][LAB] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('mushroom_rabies_lab.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.absolute()


# ============================================================
# LSTM MODEL (must match training architecture)
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=3, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, (h_n, c_n) = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out, h_n, c_n


# ============================================================
# STAGE 1: RABIES INJECTION
# ============================================================

class RabiesInjection:
    """
    Rabies attacks the nervous system causing hyperexcitability.

    In LSTM terms:
    - AMPLIFY input gate weights (more sensitive to every signal)
    - AMPLIFY cell gate weights (stronger memories)
    - REDUCE forget gate bias (less forgetting = hypervigilance)
    - SHARPEN output gate (more decisive outputs)

    LSTM weight layout: [input_gate | forget_gate | cell_gate | output_gate]
    Each gate has hidden_size rows, so 512 = 4 * 128
    """

    def __init__(self, aggression: float = 1.5, forget_suppression: float = 0.3):
        self.aggression = aggression          # Weight amplification factor
        self.forget_suppression = forget_suppression  # How much to suppress forgetting

    def inject(self, state_dict: dict) -> dict:
        """Inject rabies into LSTM weights."""
        mutant = copy.deepcopy(state_dict)
        logger.info(f"[RABIES] Injecting with aggression={self.aggression:.1f}")

        for key in mutant:
            if 'lstm' not in key:
                continue

            tensor = mutant[key]
            h = tensor.shape[0] // 4  # hidden_size

            if 'weight' in key:
                # Gate indices: [0:h]=input, [h:2h]=forget, [2h:3h]=cell, [3h:4h]=output
                input_gate = tensor[0:h]
                forget_gate = tensor[h:2*h]
                cell_gate = tensor[2*h:3*h]
                output_gate = tensor[3*h:4*h]

                # RABIES: Hyperexcite input gate (more sensitive to everything)
                input_gate *= self.aggression
                # RABIES: Amplify cell gate (stronger pattern memories)
                cell_gate *= self.aggression * 0.8
                # RABIES: Sharpen output gate (more decisive)
                output_gate *= 1.0 + (self.aggression - 1.0) * 0.5

                mutant[key] = torch.cat([input_gate, forget_gate, cell_gate, output_gate], dim=0)

                stats = f"mean={mutant[key].float().mean():.4f} std={mutant[key].float().std():.4f}"
                logger.info(f"  [{key}] amplified | {stats}")

            elif 'bias' in key:
                # Suppress forget gate bias (less forgetting = hypervigilance)
                forget_bias = tensor[h:2*h]
                forget_bias *= self.forget_suppression
                mutant[key] = tensor
                logger.info(f"  [{key}] forget gate suppressed by {self.forget_suppression:.1f}x")

        return mutant


# ============================================================
# STAGE 2: MUSHROOM TRIP (Psilocybin Perturbation)
# ============================================================

class MushroomTrip:
    """
    Psilocybin increases neural connectivity and randomness.
    Creates novel pathways between neurons that didn't exist before.

    In LSTM terms:
    - Add Gaussian noise scaled by weight magnitude (proportional chaos)
    - ACTIVATE dormant connections (weights near zero get a random boost)
    - Create cross-layer interference (layer 0 bleeds into layer 1)
    """

    def __init__(self, dose: float = 0.15, dormant_threshold: float = 0.01,
                 dormant_boost: float = 0.1, cross_bleed: float = 0.05):
        self.dose = dose                    # Noise intensity relative to weight std
        self.dormant_threshold = dormant_threshold  # Weights below this are "dormant"
        self.dormant_boost = dormant_boost  # Random activation of dormant neurons
        self.cross_bleed = cross_bleed      # Cross-layer bleed-through

    def trip(self, state_dict: dict) -> dict:
        """Send the model on a mushroom trip."""
        mutant = copy.deepcopy(state_dict)
        logger.info(f"[MUSHROOM] Dose={self.dose:.2f}, dormant_boost={self.dormant_boost:.2f}")

        # Phase 1: Proportional Gaussian noise on all weights
        for key in mutant:
            tensor = mutant[key]
            if tensor.ndim == 0:
                continue
            std = tensor.float().std().item()
            noise = torch.randn_like(tensor.float()) * std * self.dose
            mutant[key] = tensor.float() + noise
            logger.info(f"  [{key}] noise injected (std_scale={std * self.dose:.4f})")

        # Phase 2: Activate dormant connections
        dormant_activated = 0
        for key in mutant:
            if 'weight' not in key:
                continue
            tensor = mutant[key]
            dormant_mask = tensor.abs() < self.dormant_threshold
            n_dormant = dormant_mask.sum().item()
            if n_dormant > 0:
                activation = torch.randn_like(tensor) * self.dormant_boost
                tensor[dormant_mask] = activation[dormant_mask]
                mutant[key] = tensor
                dormant_activated += n_dormant
        logger.info(f"  Dormant neurons activated: {dormant_activated}")

        # Phase 3: Cross-layer bleed (layer 0 hidden-hidden bleeds into layer 1)
        key_l0 = 'lstm.weight_hh_l0'
        key_l1 = 'lstm.weight_hh_l1'
        if key_l0 in mutant and key_l1 in mutant:
            bleed = mutant[key_l0].float() * self.cross_bleed
            mutant[key_l1] = mutant[key_l1].float() + bleed
            logger.info(f"  Cross-layer bleed: L0->L1 at {self.cross_bleed:.2f}x")

        return mutant


# ============================================================
# STAGE 3: TEQA INJECTION (Transposable Element Insertion)
# ============================================================

class TEQAInjection:
    """
    Transposable elements modify existing genetic material by inserting
    themselves into the genome. They don't add new genes - they alter
    existing ones.

    In LSTM terms:
    - Read the TEQA signal (confidence, novelty, vote bias, etc.)
    - Inject as a bias modification on the LSTM's hidden-to-hidden weights
    - The TE "inserts" itself into the network's memory pathway
    - Stronger TEQA signals = larger insertion
    """

    def __init__(self, signal_path: str = None, insertion_strength: float = 0.2,
                 rabies_aggression: float = 1.0):
        if signal_path is None:
            self.signal_path = SCRIPT_DIR / 'te_quantum_signal.json'
        else:
            self.signal_path = Path(signal_path)
        self.insertion_strength = insertion_strength
        self.rabies_aggression = rabies_aggression

    def _load_signal(self) -> Optional[dict]:
        if not self.signal_path.exists():
            logger.warning("[TEQA] No signal file found")
            return None
        with open(self.signal_path) as f:
            return json.load(f)

    def inject(self, state_dict: dict) -> Tuple[dict, dict]:
        """Inject TEQA signals into the model weights. Returns (mutant, teqa_info)."""
        signal = self._load_signal()
        if signal is None:
            return state_dict, {}

        mutant = copy.deepcopy(state_dict)

        jg = signal['jardines_gate']
        q = signal['quantum']

        # Extract TE features as a modulation vector
        confidence = jg['confidence']
        direction = jg['direction']  # 1=LONG, -1=SHORT
        novelty = q['novelty']
        vote_ratio = q['vote_long'] / (q['vote_long'] + q['vote_short'] + 1e-8)
        entropy = q['measurement_entropy']
        n_active = q['n_active_qubits'] / 25.0  # Normalized

        # Create TE modulation vector (6 features -> expand to hidden_size)
        te_vector = np.array([confidence, direction * confidence, novelty,
                             vote_ratio, entropy / 25.0, n_active])

        effective_strength = self.insertion_strength * self.rabies_aggression
        logger.info(f"[TEQA] Inserting TE vector: conf={confidence:.3f} dir={'LONG' if direction==1 else 'SHORT'} "
                    f"novelty={novelty:.3f} effective_strength={effective_strength:.3f} "
                    f"(base={self.insertion_strength:.2f} x rabies={self.rabies_aggression:.1f})")

        # Insert into hidden-to-hidden weights as a rank-1 perturbation
        # This is the "transposon insertion" — it modifies the recurrent pathway
        for layer in range(2):
            key = f'lstm.weight_hh_l{layer}'
            if key not in mutant:
                continue

            W = mutant[key].float()
            h = W.shape[0] // 4  # hidden_size = 128

            # Create insertion matrix: outer product of TE vector (expanded) with
            # a random projection of the existing weights
            # This ensures the TE "fits" the existing genetic context
            np.random.seed(int(confidence * 10000) % 2**31)

            # Project TE vector to hidden_size dimensions
            projection = np.random.randn(len(te_vector), h).astype(np.float32)
            projection /= np.linalg.norm(projection, axis=1, keepdims=True) + 1e-8
            te_hidden = np.dot(te_vector, projection)  # shape: (hidden_size,)

            # Scale by PER-GATE weight statistics (not overall W std)
            # This ensures TEQA insertion scales proportionally even after rabies amplification
            cell_gate = W[2*h:3*h]  # (128, 128)
            input_gate = W[0:h]      # (128, 128)

            cell_std = cell_gate.std().item()
            input_std = input_gate.std().item()

            te_hidden_tensor = torch.tensor(te_hidden, dtype=torch.float32)

            # Rank-1 update: each gate uses its OWN std so insertion is proportional
            # to gate magnitude, not the overall matrix average
            cell_bias = te_hidden_tensor * cell_std * effective_strength
            input_bias = te_hidden_tensor * input_std * effective_strength

            cell_gate += cell_bias.unsqueeze(0).expand_as(cell_gate) * direction
            input_gate += input_bias.unsqueeze(0).expand_as(input_gate) * confidence

            mutant[key] = W
            logger.info(f"  [{key}] TE inserted into cell+input gates")

        teqa_info = {
            'confidence': confidence,
            'direction': direction,
            'novelty': novelty,
            'vote_ratio': vote_ratio,
            'entropy': entropy,
            'active_qubits': n_active,
        }

        return mutant, teqa_info


# ============================================================
# STAGE 4: QUANTUM COMPRESSION
# ============================================================

class QuantumCompressor:
    """
    Blast the mutated hidden state through a quantum autoencoder.
    Uses the Strike Boss compression engine (QuTiP).

    Takes the LSTM's hidden state activations, encodes them as a
    quantum state, compresses, and measures fidelity.

    High fidelity = the mutation produced a coherent signal (clean regime)
    Low fidelity = the mutation is noisy/chaotic (choppy regime)

    The compressed state IS the final signal.
    """

    def __init__(self, fidelity_threshold: float = 0.85, max_layers: int = 6):
        self.fidelity_threshold = fidelity_threshold
        self.max_layers = max_layers

    def _ry(self, theta):
        return (-1j * theta/2 * qt.sigmay()).expm()

    def _cnot(self, N, control, target):
        p0 = qt.ket2dm(qt.basis(2, 0))
        p1 = qt.ket2dm(qt.basis(2, 1))
        X = qt.sigmax()
        ops_0 = [qt.qeye(2)] * N
        ops_0[control] = p0
        ops_1 = [qt.qeye(2)] * N
        ops_1[control] = p1
        ops_1[target] = X
        return qt.tensor(ops_0) + qt.tensor(ops_1)

    def _get_encoder(self, params, n_qubits):
        U = qt.qeye([2] * n_qubits)
        idx = 0
        for _ in range(self.max_layers):
            ry_ops = [self._ry(params[idx + i]) for i in range(n_qubits)]
            idx += n_qubits
            U = qt.tensor(ry_ops) * U
            for i in range(n_qubits):
                U = self._cnot(n_qubits, i, (i+1) % n_qubits) * U
        return U

    def _cost(self, params, state, n_qubits, n_latent):
        n_trash = n_qubits - n_latent
        U = self._get_encoder(params, n_qubits)
        rho = state * state.dag() if state.type == 'ket' else state
        rho_out = U * rho * U.dag()
        rho_trash = rho_out.ptrace(range(n_latent, n_qubits))
        ref = qt.tensor([qt.ket2dm(qt.basis(2, 0)) for _ in range(n_trash)])
        return 1 - qt.fidelity(rho_trash, ref)

    def compress(self, hidden_state: np.ndarray) -> dict:
        """
        Compress the LSTM hidden state through quantum autoencoder.

        Args:
            hidden_state: numpy array from LSTM (128 dims)

        Returns:
            dict with fidelity, regime, compressed state, compression ratio
        """
        if not QUTIP_AVAILABLE:
            logger.warning("[COMPRESS] QuTiP not available, using classical fallback")
            return self._classical_fallback(hidden_state)

        # Encode 128-dim hidden state into quantum state vector
        # Use first 2^n elements that fit (128 = 2^7, perfect)
        n_qubits = 7  # 2^7 = 128
        state_vector = hidden_state[:2**n_qubits].astype(np.complex128)

        # Normalize to valid quantum state
        norm = np.linalg.norm(state_vector)
        if norm < 1e-10:
            return {'fidelity': 0.0, 'regime': 'DEAD', 'ratio': 0, 'iterations': 0}
        state_vector /= norm

        state = qt.Qobj(state_vector, dims=[[2]*n_qubits, [1]*n_qubits])

        logger.info(f"[COMPRESS] Encoding {n_qubits}-qubit state...")

        # Recursive compression
        current_state = state
        current_qubits = n_qubits
        total_iterations = 0
        min_latent = 3  # Don't compress below 3 qubits (8 dims)

        while current_qubits > min_latent:
            n_latent = current_qubits - 1
            n_params = self.max_layers * current_qubits
            init_params = np.random.rand(n_params) * np.pi

            result = minimize(
                self._cost, init_params,
                args=(current_state, current_qubits, n_latent),
                method='COBYLA',
                options={'maxiter': 300}
            )

            fid = 1 - result.fun
            total_iterations += 1

            if fid < self.fidelity_threshold:
                logger.info(f"  Compression stopped at {current_qubits}q (fid={fid:.3f} < {self.fidelity_threshold})")
                break

            # Extract compressed state
            U = self._get_encoder(result.x, current_qubits)
            rho = current_state * current_state.dag()
            rho_out = U * rho * U.dag()
            latent_state = rho_out.ptrace(range(n_latent))
            current_state = latent_state.eigenstates()[1][-1].unit()
            current_qubits = n_latent

            logger.info(f"  {current_qubits+1}q -> {current_qubits}q | fidelity={fid:.4f}")

        ratio = n_qubits / current_qubits

        # Classify regime from compression fidelity
        final_fid = fid if 'fid' in dir() else 1.0
        if final_fid >= 0.95:
            regime = 'CLEAN'
        elif final_fid >= 0.85:
            regime = 'VOLATILE'
        else:
            regime = 'CHOPPY'

        compressed_vector = current_state.full().flatten()

        logger.info(f"[COMPRESS] {n_qubits}q -> {current_qubits}q | ratio={ratio:.1f}x | "
                    f"fidelity={final_fid:.4f} | regime={regime}")

        return {
            'fidelity': float(final_fid),
            'regime': regime,
            'ratio': float(ratio),
            'iterations': total_iterations,
            'compressed_qubits': current_qubits,
            'compressed_state': compressed_vector.tolist(),
            'original_qubits': n_qubits,
        }

    def _classical_fallback(self, hidden_state: np.ndarray) -> dict:
        """Fallback when QuTiP not available — use SVD compression."""
        h = hidden_state.reshape(8, 16) if len(hidden_state) == 128 else hidden_state.reshape(-1, 8)
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        energy = np.cumsum(S**2) / np.sum(S**2)
        n_components = np.searchsorted(energy, self.fidelity_threshold) + 1
        ratio = len(S) / n_components

        if energy[0] >= 0.95:
            regime = 'CLEAN'
        elif energy[0] >= 0.85:
            regime = 'VOLATILE'
        else:
            regime = 'CHOPPY'

        logger.info(f"[COMPRESS-CLASSICAL] SVD ratio={ratio:.1f}x | regime={regime}")
        return {
            'fidelity': float(energy[min(n_components-1, len(energy)-1)]),
            'regime': regime,
            'ratio': float(ratio),
            'iterations': 0,
            'compressed_state': (U[:, :n_components] @ np.diag(S[:n_components])).flatten().tolist(),
        }


# ============================================================
# THE LAB: Full Pipeline
# ============================================================

class MushroomRabiesLab:
    """
    The full pipeline:
    Expert -> Rabies -> Mushrooms -> TEQA -> Inference -> Compress -> Mutant
    """

    def __init__(self,
                 rabies_aggression: float = 1.5,
                 mushroom_dose: float = 0.15,
                 teqa_strength: float = 0.2,
                 compress_fidelity: float = 0.85):
        self.rabies = RabiesInjection(aggression=rabies_aggression)
        self.mushrooms = MushroomTrip(dose=mushroom_dose)
        self.teqa = TEQAInjection(insertion_strength=teqa_strength,
                                   rabies_aggression=rabies_aggression)
        self.compressor = QuantumCompressor(fidelity_threshold=compress_fidelity)

    def load_expert(self, path: str) -> Tuple[nn.Module, dict]:
        """Load an expert model and return (model, state_dict)."""
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        model = LSTMModel(input_size=8, hidden_size=128, output_size=3, num_layers=2)
        model.load_state_dict(state_dict)
        model.eval()
        return model, state_dict

    def run_pipeline(self, expert_path: str, test_input: np.ndarray = None) -> dict:
        """
        Run the full Mushroom Rabies pipeline.

        Args:
            expert_path: Path to .pth expert model
            test_input: Optional test input (seq_len, 8) for inference

        Returns:
            Full report dict
        """
        start_time = time.time()
        expert_name = Path(expert_path).stem

        print()
        print("=" * 60)
        print("  MUSHROOM RABIES LAB")
        print(f"  Subject: {expert_name}")
        print("=" * 60)

        # Load original expert
        logger.info(f"Loading expert: {expert_path}")
        original_model, original_state = self.load_expert(expert_path)

        # ---- STAGE 1: RABIES ----
        print("\n  [1/4] RABIES INJECTION...")
        rabid_state = self.rabies.inject(original_state)

        # ---- STAGE 2: MUSHROOMS ----
        print("  [2/4] MUSHROOM TRIP...")
        tripping_state = self.mushrooms.trip(rabid_state)

        # ---- STAGE 3: TEQA ----
        print("  [3/4] TEQA INSERTION...")
        mutant_state, teqa_info = self.teqa.inject(tripping_state)

        # Create mutant model
        mutant_model = LSTMModel(input_size=8, hidden_size=128, output_size=3, num_layers=2)
        mutant_model.load_state_dict({k: v.float() if v.is_floating_point() else v
                                      for k, v in mutant_state.items()})
        mutant_model.eval()

        # ---- INFERENCE: Run test data through mutant ----
        if test_input is None:
            # Generate synthetic test data if none provided
            np.random.seed(42)
            test_input = np.random.randn(30, 8).astype(np.float32)  # 30 bars, 8 features

        input_tensor = torch.FloatTensor(test_input).unsqueeze(0)

        with torch.no_grad():
            # Original expert
            orig_out, orig_h, orig_c = original_model(input_tensor)
            orig_probs = torch.softmax(orig_out, dim=1)[0]

            # Mutant expert
            mut_out, mut_h, mut_c = mutant_model(input_tensor)
            mut_probs = torch.softmax(mut_out, dim=1)[0]

        actions = ['HOLD', 'BUY', 'SELL']
        orig_action = actions[orig_probs.argmax().item()]
        orig_conf = orig_probs.max().item()
        mut_action = actions[mut_probs.argmax().item()]
        mut_conf = mut_probs.max().item()

        print(f"\n  Original: {orig_action} ({orig_conf:.1%})")
        print(f"  Mutant:   {mut_action} ({mut_conf:.1%})")

        # ---- STAGE 4: QUANTUM COMPRESSION ----
        print("\n  [4/4] QUANTUM COMPRESSION...")
        hidden_state = mut_h[-1, 0].numpy()  # Last layer hidden state
        compress_result = self.compressor.compress(hidden_state)

        elapsed = time.time() - start_time

        # Build report
        report = {
            'expert': expert_name,
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'stages': {
                'rabies': {
                    'aggression': self.rabies.aggression,
                    'forget_suppression': self.rabies.forget_suppression,
                },
                'mushrooms': {
                    'dose': self.mushrooms.dose,
                    'dormant_boost': self.mushrooms.dormant_boost,
                    'cross_bleed': self.mushrooms.cross_bleed,
                },
                'teqa': teqa_info,
            },
            'original': {
                'action': orig_action,
                'confidence': float(orig_conf),
                'probabilities': orig_probs.tolist(),
            },
            'mutant': {
                'action': mut_action,
                'confidence': float(mut_conf),
                'probabilities': mut_probs.tolist(),
            },
            'compression': compress_result,
            'mutation_delta': {
                'action_changed': orig_action != mut_action,
                'confidence_delta': float(mut_conf - orig_conf),
                'prob_shift': (mut_probs - orig_probs).tolist(),
            },
        }

        # Save mutant model
        mutant_path = SCRIPT_DIR / 'top_50_experts' / f'{expert_name}_MUTANT.pth'
        torch.save({k: v.float() if v.is_floating_point() else v
                    for k, v in mutant_state.items()}, str(mutant_path))
        report['mutant_path'] = str(mutant_path)

        # Save report
        report_path = SCRIPT_DIR / 'teqa_analytics' / f'lab_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        print()
        print("=" * 60)
        print("  LAB RESULTS")
        print("=" * 60)
        print(f"  Expert:      {expert_name}")
        print(f"  Original:    {orig_action} ({orig_conf:.1%}) [{orig_probs[0]:.2f} | {orig_probs[1]:.2f} | {orig_probs[2]:.2f}]")
        print(f"  Mutant:      {mut_action} ({mut_conf:.1%}) [{mut_probs[0]:.2f} | {mut_probs[1]:.2f} | {mut_probs[2]:.2f}]")
        print(f"  Compression: {compress_result['regime']} (fidelity={compress_result['fidelity']:.4f})")
        if 'ratio' in compress_result:
            print(f"  Ratio:       {compress_result['ratio']:.1f}x")
        print(f"  Elapsed:     {elapsed:.1f}s")
        print(f"  Mutant saved: {mutant_path.name}")
        print(f"  Report saved: {report_path.name}")
        if teqa_info:
            d = "LONG" if teqa_info.get('direction', 0) == 1 else "SHORT"
            print(f"  TEQA:        {d} ({teqa_info.get('confidence', 0):.1%}) novelty={teqa_info.get('novelty', 0):.3f}")
        print("=" * 60)

        return report


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Mushroom Rabies Lab')
    parser.add_argument('--expert', default='expert_BTCUSD_special.pth',
                       help='Expert filename in top_50_experts/')
    parser.add_argument('--rabies', type=float, default=1.5,
                       help='Rabies aggression factor (default: 1.5)')
    parser.add_argument('--dose', type=float, default=0.15,
                       help='Mushroom dose (default: 0.15)')
    parser.add_argument('--teqa-strength', type=float, default=0.2,
                       help='TEQA insertion strength (default: 0.2)')
    parser.add_argument('--fidelity', type=float, default=0.85,
                       help='Compression fidelity threshold (default: 0.85)')
    args = parser.parse_args()

    expert_path = SCRIPT_DIR / 'top_50_experts' / args.expert
    if not expert_path.exists():
        logger.error(f"Expert not found: {expert_path}")
        sys.exit(1)

    lab = MushroomRabiesLab(
        rabies_aggression=args.rabies,
        mushroom_dose=args.dose,
        teqa_strength=args.teqa_strength,
        compress_fidelity=args.fidelity,
    )

    report = lab.run_pipeline(str(expert_path))


if __name__ == '__main__':
    main()
