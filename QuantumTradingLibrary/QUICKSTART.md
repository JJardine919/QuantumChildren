# Quick Start Guide - Quantum Trading Library

## Welcome!

This library is your collection of advanced trading systems. Currently in the **ORGANIZATION PHASE**.

## Current Status

✓ **2 Systems Organized**
1. Quantum Analysis System (QPE + Discrete Logarithm)
2. Bio-Neural Trader (Hodgkin-Huxley + STDP)

⏳ **Awaiting 4+ More Systems**

## Quick Navigation

```
QuantumTradingLibrary/
├── README.md                    # ← Start here (project overview)
├── QUICKSTART.md               # ← You are here
├── requirements.txt            # ← Install dependencies
│
├── 01_Systems/                 # ← All trading systems
│   ├── QuantumAnalysis/       # ✓ Quantum Phase Estimation
│   ├── BioNeuralTrader/       # ✓ Bio-inspired neural nets
│   ├── System_03_Pending/     # ⏳ Awaiting...
│   ├── System_04_Pending/     # ⏳ Awaiting...
│   ├── System_05_Pending/     # ⏳ Awaiting...
│   └── System_06_Pending/     # ⏳ Awaiting...
│
├── 02_Documentation/          # ← Detailed docs
│   └── SYSTEMS_CATALOG.md     # ← System comparison & specs
│
├── 03_Config/                 # ← Configuration files
│   ├── config.yaml           # ← Master config template
│   └── vps_setup.sh          # ← VPS deployment script
│
├── 04_Data/                   # ← Market data storage
├── 05_Results/                # ← Analysis results & charts
├── 06_Integration/            # ← Future integration code
└── 07_Testing/                # ← Test scripts & backtests
```

## Phase 1: Adding Your Systems (CURRENT)

### For Each System You Bring:

1. **Copy files to appropriate directory:**
   ```bash
   # If it's system 3:
   Copy to: 01_Systems/System_03_Pending/

   # If it's additional:
   Copy to: 01_Systems/AdditionalSystems/SystemName/
   ```

2. **Tell me about it:**
   - What does it do?
   - What technology (Qiskit, TensorFlow, custom, etc.)?
   - What are the file names?
   - Any special requirements?

3. **I'll organize it:**
   - Create documentation
   - Add to catalog
   - Note integration points
   - Update requirements if needed

## Testing Individual Systems

### Quantum Analysis System
```bash
# Navigate to the system
cd 01_Systems/QuantumAnalysis

# Basic analysis
python Price_Qiskit.py

# Visual analysis
python Price_Qiskit_Visual.py
```

**Inputs:**
- Symbol (default: EURUSD)
- Event horizon offset (candles back)
- Horizon length (forecast length)

**Outputs:**
- Probability matrix
- Binary horizon prediction
- Visualizations (in `quantum_trading_results/`)

### Bio-Neural Trader

#### Training:
```bash
cd 01_Systems/BioNeuralTrader

# Train the model
python BioTraderLearn.py
```
- Takes 10-30 minutes
- Saves charts to `./charts/`
- Shows correlation and MSE

#### Prediction GUI:
```bash
python BioTraderPredictor.py
```
- Select symbol and timeframe
- Adjust history/forecast bars
- Click "Update" for predictions

## Installation

### 1. Prerequisites
- Python 3.10+
- MetaTrader 5 installed
- (Optional) AMD GPU with ROCm for acceleration

### 2. Create Virtual Environment
```bash
cd QuantumTradingLibrary
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure MT5
Edit `03_Config/config.yaml`:
```yaml
mt5:
  login: YOUR_MT5_LOGIN
  password: "YOUR_PASSWORD"
  server: "YOUR_BROKER_SERVER"
```

### 5. Test Connection
```python
import MetaTrader5 as mt5

if mt5.initialize():
    print("✓ MT5 connected!")
    print(f"Account: {mt5.account_info().login}")
    mt5.shutdown()
else:
    print("✗ MT5 connection failed")
```

## VPS Deployment (Future)

When ready to deploy on VPS:
```bash
# Copy vps_setup.sh to your VPS
scp 03_Config/vps_setup.sh user@vps-ip:~/

# SSH into VPS
ssh user@vps-ip

# Run setup script
chmod +x vps_setup.sh
./vps_setup.sh
```

## Common Tasks

### View System Documentation
```bash
# Quantum system docs
cat 01_Systems/QuantumAnalysis/README.md

# Bio-neural docs
cat 01_Systems/BioNeuralTrader/README.md

# Systems comparison
cat 02_Documentation/SYSTEMS_CATALOG.md
```

### Update Requirements
After adding a new system:
```bash
pip freeze > requirements_updated.txt
# Review and merge into requirements.txt
```

### Check GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Run Quantum Analysis
```python
from Price_Qiskit import analyze_from_point

# Analyze 20 candles back, forecast 10 ahead
analyze_from_point(
    timepoint_offset=20,
    horizon_length=10
)
```

### Train Bio-Neural Model
```python
from BioTraderLearn import EnhancedPlasmaBrainTrader, MarketFeatures

# Create trader
trader = EnhancedPlasmaBrainTrader(
    input_size=100,
    hidden_size=64,
    output_size=1
)

# Train and predict
# (See BioTraderLearn.py for full example)
```

## Troubleshooting

### MT5 Connection Issues
```python
import MetaTrader5 as mt5
print(mt5.last_error())  # Shows error details
```

### Qiskit Installation Issues
```bash
pip install qiskit qiskit-aer --upgrade
```

### PyTorch GPU Issues
```bash
# For AMD GPUs (ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Reduce `num_qubits` in quantum analysis (try 18-20)
- Reduce `shots` (try 1000-2000)
- Reduce `hidden_size` in bio-neural (try 32)

## Next Steps

**Current Phase:** Organization (2/6+ systems)

1. ⏳ **You:** Bring remaining systems (4+)
2. ⏳ **We:** Organize and document each
3. ⏳ **Review:** Analyze all systems together
4. ⏳ **Design:** Plan integration architecture
5. ⏳ **Build:** Create unified platform
6. ⏳ **Test:** Backtest and paper trade
7. ⏳ **Deploy:** VPS with full automation

## Resources

### Documentation
- `README.md` - Project overview
- `02_Documentation/SYSTEMS_CATALOG.md` - System specs & comparison
- Individual system READMEs

### Configuration
- `03_Config/config.yaml` - Master configuration
- `03_Config/vps_setup.sh` - VPS deployment

### Support
- Check system READMEs first
- Review error messages carefully
- Test individual systems before integration

## Important Notes

⚠️ **DO NOT modify files in `01_Systems/`**
These are base copies. Make changes in `06_Integration/` or create copies for experiments.

⚠️ **Always test with paper trading first**
Never run live trading without extensive backtesting and paper trading.

⚠️ **Keep your MT5 credentials secure**
Never commit config files with real credentials to version control.

## Quick Reference

| Task | Command |
|------|---------|
| **Test quantum system** | `python 01_Systems/QuantumAnalysis/Price_Qiskit.py` |
| **Train bio-neural** | `python 01_Systems/BioNeuralTrader/BioTraderLearn.py` |
| **GUI predictor** | `python 01_Systems/BioNeuralTrader/BioTraderPredictor.py` |
| **View system docs** | `cat 01_Systems/*/README.md` |
| **Check requirements** | `cat requirements.txt` |
| **View catalog** | `cat 02_Documentation/SYSTEMS_CATALOG.md` |

---

**Ready?** Let's add your next system! Just let me know what you're bringing and I'll organize it.

**Questions?** Check the detailed READMEs in each system directory.

**Status:** Phase 1 - Organization (2/6+ systems collected) ✓
