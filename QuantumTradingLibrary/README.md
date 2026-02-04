# Quantum Trading Library

A comprehensive collection of advanced trading systems combining quantum computing, bio-inspired neural networks, and machine learning for financial market analysis and automated trading.

## Project Structure

```
QuantumTradingLibrary/
├── 01_Systems/              # All trading systems (base copies - DO NOT MODIFY)
│   ├── QuantumAnalysis/     # Quantum Phase Estimation for market analysis
│   └── BioNeuralTrader/     # Hodgkin-Huxley bio-inspired neural networks
├── 02_Documentation/        # System documentation and research papers
├── 03_Config/              # Configuration files for deployment
├── 04_Data/                # Market data and datasets
├── 05_Results/             # Analysis results and visualizations
├── 06_Integration/         # Integration modules (future)
└── 07_Testing/             # Test scripts and backtesting
```

## Systems Catalog

### 1. Quantum Analysis System
**Location:** `01_Systems/QuantumAnalysis/`
**Status:** Organized ✓
**Hardware:** CPU + Quantum Simulator (Qiskit)

**Components:**
- `Price_Qiskit.py` - Core quantum phase estimation analysis
- `Price_Qiskit_Visual.py` - Visual analysis with charts

**Key Features:**
- Quantum Phase Estimation (QPE) for market state analysis
- Binary price sequence encoding
- Event horizon prediction (forecast 10+ candles)
- Probability distribution analysis
- Integration with MetaTrader 5

**Accuracy:** 66-100% trend direction, 5.13% probability concentration

### 2. Bio-Neural Trading System
**Location:** `01_Systems/BioNeuralTrader/`
**Status:** Organized ✓
**Hardware:** GPU-accelerated (PyTorch) - Optimized for 16x AMD Radeon RX 6800

**Components:**
- `BioTraderLearn.py` - Training system with Hodgkin-Huxley neurons
- `BioTraderPredictor.py` - GUI predictor interface

**Key Features:**
- Hodgkin-Huxley neuron model for realistic brain-like processing
- STDP (Spike-Timing-Dependent Plasticity) learning
- Plasma influence modeling
- 100+ market features (SMA, EMA, RSI, MACD, Bollinger, ATR, momentum)
- PyTorch deep learning with adaptive training
- Real-time prediction with GUI

## Hardware Configuration

**Primary System:**
- 16x AMD Radeon RX 6800 GPUs (GPU acceleration ready)
- VPS deployment target
- MT5 integration

## Dependencies

**Core Requirements:**
- Python 3.8+
- MetaTrader5
- Qiskit + Qiskit-Aer (Quantum computing)
- PyTorch (GPU version recommended)
- NumPy, Pandas, Matplotlib
- scikit-learn
- PyCryptodome (for hashing)

See `requirements.txt` for complete list.

## Usage Philosophy

**IMPORTANT:** This is a library of BASE SYSTEMS. Do not modify the original files in `01_Systems/`.

1. **Current Phase:** Collection and organization
2. **Next Phase:** Add 4-5 more systems
3. **Review Phase:** Analyze all systems together
4. **Integration Phase:** Build unified platform

## Deployment Strategy

**Target:** VPS with full automation
**Features:**
- Real-time prediction updates
- Multi-symbol support
- Backtesting framework
- Model persistence and retraining
- Fully automated trading (with risk management)

## System Comparison

| Feature | Quantum Analysis | Bio-Neural Trader |
|---------|-----------------|------------------|
| **Approach** | Quantum state superposition | Biological neural modeling |
| **Method** | QPE + Discrete logarithm | Hodgkin-Huxley + STDP |
| **Input** | Binary price sequences | 100+ market features |
| **Output** | Probability distribution | Direct price prediction |
| **Strength** | Trend direction | Pattern learning |
| **Hardware** | CPU/Simulator | GPU-accelerated |
| **Training** | Not required | Adaptive learning |

## Next Steps

1. ✓ Organize existing systems
2. ⏳ Add remaining systems (4-5 more)
3. ⏳ Document all systems
4. ⏳ Comparative analysis
5. ⏳ Design integration architecture
6. ⏳ Build unified platform

## Notes

- Keep all base copies pristine
- Document everything thoroughly
- Test individually before integration
- Educational resources and papers in `02_Documentation/`
- Results from all tests in `05_Results/`

---

**Created:** 2026-01-09
**Last Updated:** 2026-01-09
**Status:** Phase 1 - Organization (2/6+ systems collected)
