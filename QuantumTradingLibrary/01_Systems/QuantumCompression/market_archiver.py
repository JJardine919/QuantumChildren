"""
QUANTUM MARKET ARCHIVER
=======================
Autonomously captures market states, converts them to quantum manifolds,
and compresses them into permanent archives (.dqcp) for AI training.

Part of the DeepCompress.Pro system.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import qutip as qt
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime

# Import local utilities
utils_path = Path(__file__).parent / "utils"
sys.path.insert(0, str(utils_path))
from signal_processing import NoiseReducer

class MarketArchiver:
    def __init__(self, symbol="BTCUSD", timeframe=mt5.TIMEFRAME_M5):
        self.symbol = symbol
        self.timeframe = timeframe
        self.reducer = NoiseReducer()
        self.archive_dir = Path("C:/Users/jjj10/QuantumTradingLibrary/04_Data/Archive")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        if not mt5.initialize():
            print("ERROR: MT5 initialization failed")
            sys.exit(1)

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
        U = qt.qeye([2]*num_qubits)
        param_idx = 0
        # Use 2 layers for production archiver for better mapping
        for layer in range(2):
            ry_ops = [self.ry(params[param_idx + i]) for i in range(num_qubits)]
            param_idx += num_qubits
            U = qt.tensor(ry_ops) * U
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

    def compress_state(self, state_vector, num_qubits, fid_threshold=0.999):
        current_state = qt.Qobj(state_vector, dims=[[2] * num_qubits, [1] * num_qubits]).unit()
        current_qubits = num_qubits
        params_list = []
        
        # Try to compress as much as possible until fidelity drops
        while current_qubits > 1:
            num_latent = current_qubits - 1
            num_params = 2 * current_qubits # 2 layers
            initial_params = np.random.rand(num_params) * np.pi
            
            result = minimize(self.cost, initial_params, args=(current_state, current_qubits, num_latent),
                            method='COBYLA', options={'maxiter': 500})
            
            if result.fun > (1 - fid_threshold):
                break
                
            U = self.get_encoder(result.x, current_qubits)
            compressed_rho = U * (current_state * current_state.dag()) * U.dag()
            latent_state = compressed_rho.ptrace(range(num_latent)).eigenstates()[1][-1]
            params_list.append(result.x)
            current_state = latent_state.unit()
            current_qubits = num_latent
            
        return current_state, params_list, num_qubits/current_qubits

    def capture_and_archive(self):
        print(f"\n[ARCHIVER] Capturing market state for {self.symbol}...")
        
        # 1. Get Data
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 256)
        if rates is None or len(rates) < 256:
            print("ERROR: Failed to get enough bars")
            return
            
        prices = np.array([r['close'] for r in rates])
        
        # 2. Denoise
        print("[ARCHIVER] Denoising manifold...")
        denoised = self.reducer.midas_style_denoise(prices)
        
        # 3. Encode (Amplitude Encoding)
        # Shift and normalize
        data = denoised - denoised.min() + 1e-6
        norm = np.linalg.norm(data)
        state_vector = (data / norm).astype(complex)
        
        # 4. Compress
        print("[ARCHIVER] Running Recursive Quantum Compression (DeepCompress)...")
        num_qubits = 8 # 256 states
        compressed_state, params, ratio = self.compress_state(state_vector, num_qubits)
        
        # 5. Archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_state_{self.symbol}_{timestamp}.dqcp.npz"
        filepath = self.archive_dir / filename
        
        np.savez(filepath, 
                 state=compressed_state.full().flatten(), 
                 params=np.array(params, dtype=object),
                 ratio=ratio,
                 symbol=self.symbol,
                 timestamp=timestamp)
        
        print(f"[SUCCESS] State archived to {filename}")
        print(f"Ratio: {ratio:.2f}x | Saved Qubits: {num_qubits - int(np.log2(len(compressed_state.full())))}")
        
        return filepath

if __name__ == "__main__":
    archiver = MarketArchiver()
    archiver.capture_and_archive()
    mt5.shutdown()
