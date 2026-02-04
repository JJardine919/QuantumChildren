import sys
import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# Add parent directory to path to allow imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_adapter import BaseAdapter

class QPEAdapter(BaseAdapter):
    """
    Adapter for the Quantum Phase Estimation (QPE) system.
    Predicts market direction based on quantum probability distributions.
    """
    
    def __init__(self, num_qubits=22, shots=3000):
        super().__init__("QPE_Analysis")
        self.num_qubits = num_qubits
        self.shots = shots
        self.simulator = AerSimulator()

    def _qpe_dlog(self, a, N, num_qubits):
        qr = QuantumRegister(num_qubits + 1)
        cr = ClassicalRegister(num_qubits)
        qc = QuantumCircuit(qr, cr)
        
        for q in range(num_qubits):
            qc.h(q)
        qc.x(num_qubits)
        
        for q in range(num_qubits):
            qc.cp(2 * np.pi * (a**(2**q) % N) / N, q, num_qubits)
        
        qc.barrier()
        for i in range(num_qubits):
            qc.h(i)
            for j in range(i):
                qc.cp(-np.pi / float(2 ** (i - j)), j, i)
        
        for i in range(num_qubits // 2):
            qc.swap(i, num_qubits - 1 - i)
        
        qc.measure(range(num_qubits), range(num_qubits))
        return qc

    def get_signal(self, symbol, timeframe, lookback=256):
        """
        Retrieves data and generates a signal using Quantum Phase Estimation.
        """
        # Ensure MT5 is initialized
        if not mt5.initialize():
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "MT5 Init Failed"}

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback)
        if rates is None or len(rates) < lookback:
            return {"name": self.name, "signal": 0.0, "confidence": 0.0, "error": "Insufficient Data"}

        df = pd.DataFrame(rates)
        
        # QPE math parameters (constants from original Price_Qiskit.py)
        a = 70000000
        N = 17000000
        
        qc = self._qpe_dlog(a, N, self.num_qubits)
        compiled_circuit = transpile(qc, self.simulator)
        job = self.simulator.run(compiled_circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Calculate signal and confidence
        total_shots = sum(counts.values())
        sorted_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Use top 10 states for weighted prediction (logic from predict_horizon)
        horizon_len = 10
        weighted_ones = 0
        weighted_zeros = 0
        
        for state, count in sorted_states[:10]:
            weight = count / total_shots
            # Analyze the 'predominant' bit of the top states
            ones = state.count('1')
            zeros = state.count('0')
            if ones > zeros:
                weighted_ones += weight
            else:
                weighted_zeros += weight
        
        signal = 1.0 if weighted_ones > weighted_zeros else -1.0
        confidence = max(weighted_ones, weighted_zeros)
        
        # Probability concentration (the original system's metric)
        top_prob = sorted_states[0][1] / total_shots
        
        return {
            "name": self.name,
            "signal": signal,
            "confidence": confidence,
            "metadata": {
                "top_state_prob": top_prob,
                "weighted_ones": weighted_ones,
                "weighted_zeros": weighted_zeros
            }
        }

if __name__ == "__main__":
    # Test on EURUSD H1
    adapter = QPEAdapter()
    print("Testing QPE Adapter...")
    res = adapter.get_signal("EURUSD", mt5.TIMEFRAME_H1)
    print(res)
    mt5.shutdown()
