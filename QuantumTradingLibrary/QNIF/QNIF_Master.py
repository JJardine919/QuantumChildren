"""
QNIF_Master.py
Quantum Neural-Immune Fusion - Master Controller

Integrates:
1. ETARE Quantum Fusion (Compression/Filter)
2. TEQA v3.0 (Neural Mosaic/Sensors)
3. Algorithm VDJ (Immune/Strategy)
4. AOI Pipeline (CRISPR Defense, Syncytin Fusion, Electric Organs)
5. Quantum Archiver (Tier 1 Pre-computed Features)

"The market is a biological organism. QNIF is its immune system."

Date: 2026-02-09
"""

import sys
import logging
import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Add root to path
current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.resolve()
sys.path.append(str(root_dir))

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][QNIF] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("QNIF")

# --- Import Components ---
try:
    from ETARE_QuantumFusion.modules.compression_layer import QuantumCompressionLayer
    from teqa_v3_neural_te import TEQAv3Engine
    from vdj_recombination import VDJRecombinationEngine
    from quantum_regime_bridge import QuantumRegimeBridge
    
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import Error: {e}")
    COMPONENTS_AVAILABLE = False

@dataclass
class BioQuantumState:
    """Unified state flowing through the 5-layer architecture."""
    symbol: str = "UNKNOWN"
    
    # Layer 1: Skin (Perception & Filter)
    compression_ratio: float = 1.0
    regime: str = "UNKNOWN"
    source_tier: int = 3 # 1=Archiver, 2=QuTiP, 3=Classical
    is_tradeable: bool = False

    # Layer 2: Genome (Sensing & Energy)
    active_tes: List[str] = field(default_factory=list)
    shock_level: float = 0.0
    shock_label: str = "CALM"
    pattern_energy: float = 0.0 # The "91.2% Energy Source" (Amplitude Sq)
    
    # Layer 3: Brain (Cognition & Consensus)
    neural_consensus: float = 0.0 
    neural_confidence: float = 0.0
    consensus_pass: bool = False
    
    # Layer 4: Immune (Adaptation & Defense)
    selected_antibody: Dict[str, Any] = field(default_factory=dict)
    is_memory_recall: bool = False
    crispr_blocked: bool = False # CRISPR defense
    fusion_active: bool = False  # Syncytin fusion
    
    # Layer 5: Gate (Execution)
    final_action: str = "HOLD"
    final_confidence: float = 0.0
    lot_multiplier: float = 1.0
    
    # Meta
    generation: int = 0
    timestamp: str = ""

class QNIF_Engine:
    def __init__(self, memory_db: str = None):
        logger.info("Initializing QNIF Engine (Bio-Quantum Synthesis)...")
        
        if not COMPONENTS_AVAILABLE:
            logger.error("Critical components missing. Check environment.")
            return

        # Tier 1 Archiver (Pre-computed)
        try:
            self.bridge = QuantumRegimeBridge()
            logger.info("Tier 1: Archiver Bridge [READY]")
        except Exception as e:
            logger.warning(f"Archiver Bridge Failed: {e}")
            self.bridge = None

        # Tier 2 Skin (Live Compression)
        self.compression = QuantumCompressionLayer(fid_threshold=0.90)
        
        # Layer 2/3: Brain/Genome (Sensing/Mosaic)
        self.teqa = TEQAv3Engine()
        
        # Layer 4: Immune (Recombination)
        self.vdj = VDJRecombinationEngine(memory_db_path=memory_db)
        
        logger.info("QNIF Engine: Bio-Immune Layers [OK]")

    def process_pulse(self, symbol: str, bars: np.ndarray) -> BioQuantumState:
        """Process a single pulse through the integrated architecture."""
        state = BioQuantumState(symbol=symbol)
        prices = bars[:, 3] if bars.ndim == 2 else bars
        
        # --- LAYER 1: SKIN (Archiver + Compression) ---
        # 1. Try Archiver (Tier 1)
        if self.bridge:
            try:
                regime, confidence = self.bridge.get_regime(prices, symbol)
                state.regime = regime.value
                state.compression_ratio = confidence
                state.source_tier = 1
                state.is_tradeable = state.compression_ratio > 1.05
            except Exception: pass

        # 2. Try Live Compression (Tier 2) if Tier 1 failed or missed
        if state.source_tier != 1:
            try:
                size = 256
                p = prices[-size:] if len(prices) >= size else prices
                p = (p - np.mean(p)) / (np.linalg.norm(p - np.mean(p)) + 1e-10)
                state_vector = p.astype(complex)
                
                comp_res = self.compression.analyze_regime(state_vector)
                state.compression_ratio = comp_res.get('ratio', 1.0)
                state.regime = comp_res.get('regime', 'UNKNOWN')
                state.source_tier = 2
                state.is_tradeable = state.compression_ratio > 1.05
            except Exception:
                state.is_tradeable = True

        # --- LAYER 2: GENOME (Energy Sensing) ---
        te_activations = []
        if self.teqa:
            teqa_res = self.teqa.analyze(bars, symbol)
            state.active_tes = [a['te'] for a in teqa_res.get('te_activations', [])]
            state.shock_level = teqa_res.get('shock_score', 0.0)
            state.shock_label = teqa_res.get('shock_label', 'CALM')
            
            # The "91.2% Energy Source" - high amplitude quantum state
            state.pattern_energy = teqa_res.get('amplitude_sq', 0.0)
            
            # Layer 3 results (Neural Mosaic)
            state.neural_consensus = teqa_res.get('direction', 0.0)
            state.neural_confidence = teqa_res.get('confidence', 0.0)
            state.consensus_pass = teqa_res.get('neural_consensus_pass', False)
            te_activations = teqa_res.get('te_activations', [])

        # --- LAYER 4: IMMUNE (Adaptation & Strategy) ---
        if self.vdj:
            vdj_res = self.vdj.run_cycle(
                bars=bars,
                symbol=symbol,
                te_activations=te_activations,
                shock_level=state.shock_level,
                shock_label=state.shock_label
            )
            state.selected_antibody = vdj_res
            state.is_memory_recall = vdj_res.get('source') == 'MEMORY_CELL'
            state.generation = vdj_res.get('generation', 0)
            
            # --- LAYER 5: GATE (Execution) ---
            vdj_action = vdj_res.get('action', 'HOLD')
            
            # Bio-Quantum Decision Logic
            # Require agreement between Brain (Consensus) and Immune (VDJ)
            # Boost confidence by the "Pattern Energy" (The 91% source)
            if state.consensus_pass:
                if (state.neural_consensus > 0 and vdj_action == 'BUY') or \
                   (state.neural_consensus < 0 and vdj_action == 'SELL'):
                    
                    state.final_action = vdj_action
                    
                    # Pattern Energy amplification: if state amplitude is high, confidence spikes
                    energy_boost = 1.0 + (state.pattern_energy * 0.5) # e.g. 91% energy = 1.45x boost
                    state.final_confidence = state.neural_confidence * energy_boost
                    
                    state.lot_multiplier = vdj_res.get('lot_mult', 1.0)
                    
                    # Apply Domestication Boost if available
                    dom_boost = teqa_res.get('domestication_boost', 1.0)
                    state.lot_multiplier *= dom_boost
                else:
                    state.final_action = "HOLD"
            else:
                state.final_action = "HOLD"
        
        state.timestamp = teqa_res.get('timestamp', '')
        return state

def run_test():
    engine = QNIF_Engine()
    
    # Simulated Trend for testing
    t = np.linspace(0, 10, 300)
    prices = 100 + 5*t + np.random.normal(0, 0.2, 300)
    mock_bars = np.zeros((300, 5))
    mock_bars[:, 3] = prices # Close
    
    print("\n--- PULSING QNIF ENGINE ---")
    state = engine.process_pulse("BTCUSD", mock_bars)
    
    print(f"SYMBOL:           {state.symbol}")
    print(f"TIER:             {state.source_tier} ({state.regime})")
    print(f"COMPRESSION:      {state.compression_ratio:.4f}")
    print(f"PATTERN ENERGY:   {state.pattern_energy*100:.1f}%")
    print(f"ACTIVE TEs:       {len(state.active_tes)}")
    print(f"SHOCK LEVEL:      {state.shock_level:.2f} ({state.shock_label})")
    print(f"NEURAL CONSENSUS: {state.neural_consensus:.2f}")
    print(f"IMMUNE ACTION:    {state.selected_antibody.get('action', 'HOLD')}")
    print(f"----------------------------")
    print(f"FINAL DECISION:   {state.final_action}")
    print(f"FINAL CONFIDENCE: {state.final_confidence:.4f}")
    print(f"LOT MULTIPLIER:   {state.lot_multiplier:.2f}")
    print(f"MEMORY RECALL:    {state.is_memory_recall}")
    print(f"----------------------------")

if __name__ == "__main__":
    run_test()
