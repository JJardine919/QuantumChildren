import numpy as np
from scipy.optimize import minimize
import qutip as qt
import os
import time
import sys

# --- Copied Logic from deep_quantum_compress_pro.py ---

class HeadlessCompressor:
    def __init__(self):
        pass

    def ry(self, theta):
        return (-1j * theta/2 * qt.sigmay()).expm()

    def cnot(self, N, control, target):
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

    def get_encoder(self, params, num_qubits):
        # Simple layered ansatz: alternating RY and CNOT ring
        U = qt.qeye([2]*num_qubits)
        param_idx = 0
        for layer in range(6):  # 6 layers for higher accuracy
            # RY on each qubit
            ry_ops = [self.ry(params[param_idx + i]) for i in range(num_qubits)]
            param_idx += num_qubits
            U = qt.tensor(ry_ops) * U
            # CNOT ring
            for i in range(num_qubits):
                U = self.cnot(num_qubits, i, (i + 1) % num_qubits) * U
        return U

    def cost(self, params, input_state, num_qubits, num_latent):
        num_trash = num_qubits - num_latent
        U = self.get_encoder(params, num_qubits)
        rho = input_state * input_state.dag() if input_state.type == 'ket' else input_state
        rho_out = U * rho * U.dag()
        rho_trash = rho_out.ptrace(range(num_latent, num_qubits))
        ref = qt.tensor([qt.ket2dm(qt.basis(2, 0)) for _ in range(num_trash)])
        fid = qt.fidelity(rho_trash, ref)
        return 1 - fid

    def recursive_compress_quantum(self, state_vector, num_qubits, min_latent=1, fid_threshold=0.99, max_iterations=5):
        # original_hash = self.compute_sha256(state_vector)
        current_state = qt.Qobj(state_vector, dims=[[2] * num_qubits, [1] * num_qubits]).unit()
        current_qubits = num_qubits
        params_list = []
        latent_states = []
        iterations = 0
        
        print(f"  Starting compression: {num_qubits} qubits (Threshold: {fid_threshold})...")
        
        while current_qubits > min_latent and iterations < max_iterations:
            num_latent = current_qubits - 1  # Reduce by 1 qubit each time
            # num_trash = 1
            num_params = 6 * current_qubits  # For 6 layers of RY
            initial_params = np.random.rand(num_params) * np.pi
            
            # print(f"    Optimizing layer {iterations+1} ({current_qubits} -> {num_latent} qubits)...")
            
            result = minimize(self.cost, initial_params, args=(current_state, current_qubits, num_latent),
                              method='COBYLA', options={'maxiter': 1000}) # Increased maxiter slightly for headless safety
            
            current_fidelity = 1 - result.fun
            print(f"    Layer {iterations+1}: Fidelity = {current_fidelity:.6f}")

            if result.fun > (1 - fid_threshold):
                print(f"    Stopping: Fidelity loss too high ({current_fidelity:.6f} < {fid_threshold})")
                break  # Stop if compression loss is too high
                
            U = self.get_encoder(result.x, current_qubits)
            compressed_rho = U * (current_state * current_state.dag()) * U.dag()
            latent_state = compressed_rho.ptrace(range(num_latent)).eigenstates()[1][-1]  # Take highest eigenvector
            
            params_list.append(result.x)
            latent_states.append(latent_state)
            current_state = latent_state.unit()
            current_qubits = num_latent
            iterations += 1
            
        ratio = num_qubits / current_qubits
        return ratio, iterations, 0.0

def run_test():
    compressor = HeadlessCompressor()
    
    # Define files
    base_dir = r"C:\Users\jjj10\QuantumTradingLibrary\04_Data\QuantumStates"
    files = {
        "UPTREND": "btc_uptrend_state.npy",
        "CHOPPY": "btc_choppy_state.npy",
        "DOWNTREND": "btc_downtrend_state.npy"
    }
    
    expectations = {
        "UPTREND": "HIGH COMPRESSIBILITY (Ratio > 1.6, ideally)", # Ratio = orig/compressed. If orig=8, comp=5, ratio=1.6
        "CHOPPY": "LOW COMPRESSIBILITY (Ratio close to 1.0)",
        "DOWNTREND": "HIGH COMPRESSIBILITY"
    }
    
    # Note: My ratio calculation in code is num_qubits / current_qubits.
    # If it compresses 8 -> 5, ratio is 1.6.
    # If it compresses 8 -> 7, ratio is 1.14.
    
    print("\n" + "="*60)
    print("HEADLESS COMPRESSION TEST START")
    print("="*60)
    
    for name, filename in files.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
            
        print(f"\nProcessing {name} ({filename})...")
        try:
            state_vector = np.load(path)
            num_qubits = int(np.log2(len(state_vector)))
            
            start_time = time.time()
            ratio, iterations, fidelity = compressor.recursive_compress_quantum(state_vector, num_qubits)
            elapsed = time.time() - start_time
            
            print(f"  Result: {iterations} layers compressed")
            print(f"  Final Size: {num_qubits - iterations} qubits (Original: {num_qubits})")
            print(f"  Compression Ratio: {ratio:.2f}x")
            print(f"  Time: {elapsed:.2f}s")
            
            # Validation Logic
            is_compressible = ratio > 1.3 # Arbitrary threshold for "good" compression
            
            status = "UNKNOWN"
            if name in ["UPTREND", "DOWNTREND"]:
                if is_compressible: status = "PASS ✅ (Compressed well)"
                else: status = "FAIL ❌ (Did not compress)"
            elif name == "CHOPPY":
                if not is_compressible: status = "PASS ✅ (Resisted compression)"
                else: status = "FAIL ❌ (Compressed too much)"
                
            print(f"  Status: {status}")
            
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_test()