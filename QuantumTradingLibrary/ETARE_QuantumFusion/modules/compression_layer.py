import numpy as np
from scipy.optimize import minimize
import qutip as qt
import os
import time

class QuantumCompressionLayer:
    """
    Compression layer for the Strike Boss Fusion Engine.
    Uses recursive quantum autoencoders to compress market state vectors.
    The resulting compression ratio serves as a Regime Metric (Complexity/Entropy).
    """
    def __init__(self, fid_threshold=0.95, max_layers=5):
        self.fid_threshold = fid_threshold
        self.max_layers = max_layers

    def _ry(self, theta):
        return (-1j * theta/2 * qt.sigmay()).expm()

    def _cnot(self, N, control, target):
        p0 = qt.ket2dm(qt.basis(2, 0))
        p1 = qt.ket2dm(qt.basis(2, 1))
        I = qt.qeye(2)
        X = qt.sigmax()
        ops = [qt.qeye(2)] * N
        ops[control] = p0
        ops[target] = I
        term1 = qt.tensor(ops)
        ops[control] = p1
        ops[target] = X
        term2 = qt.tensor(ops)
        return term1 + term2

    def _get_encoder(self, params, num_qubits):
        U = qt.qeye([2]*num_qubits)
        param_idx = 0
        for _ in range(6):  # 6 layers of RY/CNOT for expressivity
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

    def analyze_regime(self, state_vector):
        """
        Compresses a 2^n state vector and returns the compression ratio.
        Higher ratio = Trending/Clean (Low Entropy)
        Lower ratio = Choppy/Complex (High Entropy)
        """
        num_qubits = int(np.log2(len(state_vector)))
        current_state = qt.Qobj(state_vector, dims=[[2] * num_qubits, [1] * num_qubits]).unit()
        current_qubits = num_qubits
        layers_compressed = 0
        
        for i in range(self.max_layers):
            num_latent = current_qubits - 1
            num_params = 6 * current_qubits
            initial_params = np.random.rand(num_params) * np.pi
            
            result = minimize(self._cost, initial_params, args=(current_state, current_qubits, num_latent),
                              method='COBYLA', options={'maxiter': 500})
            
            fidelity = 1 - result.fun
            if fidelity < self.fid_threshold:
                break
                
            U = self._get_encoder(result.x, current_qubits)
            rho_out = U * (current_state * current_state.dag()) * U.dag()
            current_state = rho_out.ptrace(range(num_latent)).eigenstates()[1][-1].unit()
            current_qubits = num_latent
            layers_compressed += 1
            
        ratio = num_qubits / current_qubits
        return {
            "ratio": ratio,
            "layers": layers_compressed,
            "final_qubits": current_qubits,
            "regime": "TRENDING" if ratio > 1.3 else "CHOPPY"
        }

if __name__ == "__main__":
    # Quick sanity test
    dummy_state = np.random.rand(256).astype(complex)
    dummy_state /= np.linalg.norm(dummy_state)
    layer = QuantumCompressionLayer(fid_threshold=0.90)
    print("Testing with random noise...")
    print(layer.analyze_regime(dummy_state))