import tkinter as tk
from tkinter import filedialog, messagebox
import qutip as qt
import numpy as np
from scipy.optimize import minimize
import time
import os
import hashlib  # For classical hash verification on serialized data

class StrikeBossCompressionEngine:
    def __init__(self, root):
        self.root = root
        self.root.title("Strike Boss Compression Engine")
        self.root.configure(bg="#0f0f0f")
        self.root.geometry("600x450")

        # Stored file path
        self.npy_file_path = None

        # Header
        header = tk.Label(root, text="STRIKE BOSS COMPRESSION", fg="#10b981", bg="#0f0f0f", font=("Arial", 22, "bold"))
        header.pack(pady=10)
        subheader = tk.Label(root, text="Recursive quantum autoencoder - Sustainability Protocol V1", fg="#10b981", bg="#0f0f0f", font=("Arial", 10))
        subheader.pack()

        # Compress section
        compress_frame = tk.Frame(root, bg="#1a1a1a", bd=1, relief="flat")
        compress_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)
        tk.Label(compress_frame, text="COMPRESS QUANTUM STATE", fg="#10b981", bg="#1a1a1a", font=("Arial", 10, "bold")).pack(pady=5)
        self.compress_btn_text = tk.StringVar()
        self.compress_btn_text.set("Select .npy source...")
        self.compress_btn = tk.Button(compress_frame, textvariable=self.compress_btn_text, command=self.select_npy_file, bg="#0f0f0f", fg="#ffffff", bd=0, width=25, height=3)
        self.compress_btn.pack(pady=10)
        tk.Button(compress_frame, text="RUN COMPRESSION", command=self.compress_quantum, bg="#10b981", fg="#000000", font=("Arial", 10, "bold"), bd=0).pack(pady=10)

        # Decompress section
        decompress_frame = tk.Frame(root, bg="#1a1a1a", bd=1, relief="flat")
        decompress_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)
        tk.Label(decompress_frame, text="DECOMPRESS ARCHIVE", fg="#10b981", bg="#1a1a1a", font=("Arial", 10, "bold")).pack(pady=5)
        self.decompress_btn = tk.Button(decompress_frame, text="Select .dqcp archive", command=self.decompress_quantum, bg="#0f0f0f", fg="#ffffff", bd=0, width=25, height=3)
        self.decompress_btn.pack(pady=10)
        tk.Button(decompress_frame, text="RUN DECOMPRESSION", command=self.decompress_quantum, bg="#10b981", fg="#000000", font=("Arial", 10, "bold"), bd=0).pack(pady=10)

        # Stats
        stats_frame = tk.Frame(root, bg="#0f0f0f")
        stats_frame.pack(fill="x", pady=10)
        self.states_processed = tk.Label(stats_frame, text="0\nProcessed", fg="#10b981", bg="#0f0f0f")
        self.states_processed.grid(row=0, column=0, padx=25)
        self.max_compression = tk.Label(stats_frame, text="0x\nRatio", fg="#10b981", bg="#0f0f0f")
        self.max_compression.grid(row=0, column=1, padx=25)
        self.speed = tk.Label(stats_frame, text="0s\nLatency", fg="#10b981", bg="#0f0f0f")
        self.speed.grid(row=0, column=2, padx=25)
        self.total_saved = tk.Label(stats_frame, text="0 qubits\nSaved", fg="#10b981", bg="#0f0f0f")
        self.total_saved.grid(row=0, column=3, padx=25)


        # Track stats
        self.processed_count = 0
        self.max_ratio = 0
        self.total_saved_qubits = 0

    def select_npy_file(self):
        initial_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "04_Data", "QuantumStates")
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select a Quantum State Vector File",
            filetypes=[("NPY files", "*.npy")]
        )
        if file_path:
            self.npy_file_path = file_path
            # Provide feedback to the user
            self.compress_btn_text.set(os.path.basename(file_path))
            self.root.update_idletasks() # Force GUI update
    
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

    def compute_sha256(self, data):
        return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()

    def recursive_compress_quantum(self, state_vector, num_qubits, min_latent=1, fid_threshold=0.90, max_iterations=5):
        original_hash = self.compute_sha256(state_vector)
        current_state = qt.Qobj(state_vector, dims=[[2] * num_qubits, [1] * num_qubits]).unit()
        current_qubits = num_qubits
        params_list = []
        latent_states = []
        iterations = 0
        while current_qubits > min_latent and iterations < max_iterations:
            num_latent = current_qubits - 1  # Reduce by 1 qubit each time
            num_trash = 1
            num_params = 6 * current_qubits  # For 6 layers of RY
            initial_params = np.random.rand(num_params) * np.pi
            result = minimize(self.cost, initial_params, args=(current_state, current_qubits, num_latent),
                              method='COBYLA', options={'maxiter': 500})
            if result.fun > (1 - fid_threshold):
                break  # Stop if compression loss is too high
            U = self.get_encoder(result.x, current_qubits)
            compressed_rho = U * (current_state * current_state.dag()) * U.dag()
            latent_state = compressed_rho.ptrace(range(num_latent)).eigenstates()[1][-1]  # Take highest eigenvector for pure approx
            params_list.append(result.x)
            latent_states.append(latent_state)
            current_state = latent_state.unit()
            current_qubits = num_latent
            iterations += 1
        ratio = num_qubits / current_qubits
        return current_state, params_list[::-1], ratio, iterations, original_hash  # Reverse params for decompression

    def compress_quantum(self):
        if not self.npy_file_path:
            messagebox.showerror("Error", "Please select an .npy file first.")
            return
            
        file_path = self.npy_file_path
        start_time = time.time()
        try:
            state_vector = np.load(file_path)
        except Exception as e:
            messagebox.showerror("File Load Error", f"Failed to load numpy file:\n{e}")
            return

        num_qubits = int(np.log2(len(state_vector)))
        if 2**num_qubits != len(state_vector):
            messagebox.showerror("Error", "Invalid state vector size (must be 2^n)")
            return

        try:
            compressed_state, params_list, ratio, iterations, original_hash = self.recursive_compress_quantum(state_vector, num_qubits)
        except Exception as e:
            messagebox.showerror("Compression Error", f"An error occurred during compression:\n{e}")
            return

        # Save as .dqcp.npz (numpy archive with state and params)
        output_path = file_path.replace('.npy', '.dqcp')
        try:
            np.savez(output_path, state=compressed_state.full().flatten(), params=np.array(params_list, dtype=object), hash=original_hash)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save compressed file:\n{e}")
            return


        # Verify decompression fidelity using quantum fidelity metric
        original_state_qobj = qt.Qobj(state_vector, dims=[[2] * num_qubits, [1] * num_qubits]).unit()
        decompressed_state = self.decompress_helper(compressed_state, params_list, num_qubits)
        
        fidelity = qt.fidelity(original_state_qobj, decompressed_state)
        
        # INTERPRETATION LOGIC (Instead of Error)
        regime_type = "UNKNOWN"
        if fidelity >= 0.95:
            regime_type = "TRENDING (Low Entropy - Clean)"
        elif fidelity >= 0.85:
            regime_type = "VOLATILE (Medium Entropy)"
        else:
            regime_type = "CHOPPY (High Entropy - Complex)"

        # Update stats
        self.processed_count += 1
        self.max_ratio = max(self.max_ratio, ratio)
        self.total_saved_qubits += num_qubits - int(np.log2(len(compressed_state.full())))
        elapsed = time.time() - start_time
        
        self.states_processed.config(text=f"{self.processed_count}\nStates Processed")
        self.max_compression.config(text=f"{ratio:.1f}x\nMax Compression")
        self.speed.config(text=f"{elapsed:.1f}s\nSpeed")
        self.total_saved.config(text=f"{self.total_saved_qubits} qubits\nTotal Saved")

        message_type = messagebox.showinfo
        if fidelity < 0.90:
            message_type = messagebox.showwarning
            
        message_type("Analysis Result", 
            f"Compressed to {output_path}.npz\n\n"
            f"Regime: {regime_type}\n"
            f"Fidelity: {fidelity:.6f}\n"
            f"Ratio: {ratio:.2f}x\n"
            f"Iterations: {iterations}"
        )

    def decompress_helper(self, latent_state, params_list, target_qubits):
        current_state = latent_state.unit()
        current_qubits = int(np.log2(current_state.shape[0]))
        for params in params_list:
            num_latent = current_qubits
            current_qubits += 1  # Add back one trash qubit
            U = self.get_encoder(params, current_qubits)
            trash = qt.basis(2, 0)
            extended_state = qt.tensor(current_state, trash)
            current_state = (U.dag() * extended_state).unit()
        return current_state

    def decompress_quantum(self):
        initial_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "04_Data", "QuantumStates")
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("DQCP files", "*.dqcp.npz"), ("All DQCP", "*.dqcp*")]
        )
        if not file_path:
            return
        start_time = time.time()
        data = np.load(file_path, allow_pickle=True)
        compressed_state_vec = data['state']
        params_list = data['params']
        original_hash = data['hash']

        num_qubits = int(np.log2(len(compressed_state_vec))) + len(params_list)  # Infer original size
        compressed_state = qt.Qobj(compressed_state_vec, dims=[[2] * int(np.log2(len(compressed_state_vec))), [1] * int(np.log2(len(compressed_state_vec)))]).unit()

        decompressed_state = self.decompress_helper(compressed_state, params_list, num_qubits)
        decompressed_vec = decompressed_state.full().flatten()
        decompressed_hash = self.compute_sha256(decompressed_vec)

        if decompressed_hash != original_hash:
            messagebox.showerror("Error", "Decompression failed verification!")
            return

        # Remove .npz and .dqcp extensions, add _decompressed.npy
        output_path = file_path.replace(".dqcp.npz", "").replace(".dqcp", "") + "_decompressed.npy"
        np.save(output_path, decompressed_vec)

        # Update stats (similar to compress)
        self.processed_count += 1
        elapsed = time.time() - start_time
        self.states_processed.config(text=f"{self.processed_count}\nStates Processed")
        self.speed.config(text=f"{elapsed:.1f}s\nSpeed")

        messagebox.showinfo("Success", f"Decompressed to {output_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StrikeBossCompressionEngine(root)
    root.mainloop()