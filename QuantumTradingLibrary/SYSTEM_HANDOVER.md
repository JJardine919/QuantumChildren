# QuantumTradingLibrary - System Handover Sheet

## 📌 Overview
This document serves as a high-level summary for developers or AI agents joining the **QuantumTradingLibrary** project. It explains the core architecture, recent changes, and how the different systems interact.

---

## 🏗️ Core Architecture
The project is organized into modular systems located in `01_Systems/`. Each system handles a specific aspect of market analysis or trading.

### Key Systems:
1.  **Quantum Analysis (`01_Systems/QuantumAnalysis/`)**: Uses Qiskit to simulate quantum circuits for trend prediction.
2.  **Bio-Neural Trader (`01_Systems/BioNeuralTrader/`)**: Implements biologically-inspired neural networks (Hodgkin-Huxley) for price prediction.
3.  **ETARE (`01_Systems/System_03_ETARE/`)**: An evolutionary trading engine that breeds strategies using genetic algorithms.
4.  **GPT Market Language (`01_Systems/System_05_GPT_MarketLanguage/`)**: Translates price movements into a "language" (BIP39 words) and uses Transformer models to predict next "words."
5.  **Quantum Compression (`01_Systems/QuantumCompression/`)**: 
    *   **Purpose**: Encodes market data into quantum states and uses a recursive quantum autoencoder to compress them.
    *   **Theory**: High compression ratios indicate "simple" market regimes (trends), while low ratios indicate "complex" or "noisy" regimes (choppy).

---

## 📂 Recent Organizational Changes (Jan 20, 2026)
We have standardized the directory structure to separate code from data:
- **Code**: All logic stays in `01_Systems/`.
- **Data**: All raw market data and generated states are moved to `04_Data/`.
  - `04_Data/MarketData/`: CSV files from MetaTrader 5.
  - `04_Data/QuantumStates/`: `.npy` (quantum state vectors) and `.dqcp.npz` (compressed archives).
- **Utils**: System-specific utilities (like signal processing) are now in `utils/` subdirectories with `__init__.py` for proper package imports.

---

## 🛠️ Key Files & Their Roles (Quantum Compression System)
If you are working on the **Quantum Compression** logic, these are the critical files:

1.  **`01_Systems/QuantumCompression/deep_quantum_compress_pro.py`**:
    *   The GUI tool for compressing and decompressing quantum states.
    *   Uses `QuTiP` for quantum object manipulation and `COBYLA` for optimization.
    *   **Fidelity**: Ensures 99.99%+ fidelity during decompression.

2.  **`01_Systems/QuantumCompression/create_market_quantum_states.py`**:
    *   Fetches data from MT5, denoises it, and converts it to `.npy` quantum state vectors.
    *   Outputs to `04_Data/QuantumStates/`.

3.  **`01_Systems/QuantumCompression/utils/signal_processing.py`**:
    *   Contains the `NoiseReducer` class.
    *   Uses **Wavelet Transforms** (db8/db4) and **CEEMDAN** to remove market noise before quantum encoding.

4.  **`01_Systems/QuantumCompression/utils/fetch_mt5_256bars_btcusd_m5.py`**:
    *   Utility to specifically pull 256-bar windows for different regimes (Uptrend, Choppy, Pullback).

---

## 🚀 How to Run the Compression Pipeline
1.  **Generate Data**: Run `fetch_mt5_256bars_btcusd_m5.py` to get CSVs.
2.  **Convert to States**: Run `create_market_quantum_states.py` to generate `.npy` vectors.
3.  **Compress**: Run `python 01_Systems/QuantumCompression/deep_quantum_compress_pro.py`.
    *   Click "Choose .npy file" (defaults to `04_Data/QuantumStates/`).
    *   Click "COMPRESS".
    *   Check the compression ratio (Trending should be < 0.6, Choppy > 0.8).

---

## 📝 Next Steps
- **Validation**: Verify if the compression ratios consistently separate trending vs. choppy markets.
- **Integration**: Feed the compression ratio as a "Regime Feature" into the **Bio-Neural** or **ETARE** systems.
- **Automation**: Create a monitor that runs this compression in real-time to alert on regime shifts.

---
*Last Updated: January 20, 2026*
