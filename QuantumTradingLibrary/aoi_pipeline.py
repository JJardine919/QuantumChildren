"""
AOI PIPELINE - Artificial Organism Intelligence Integration Layer
=================================================================
Wraps all 8 biological algorithms into a single callable pipeline
for BRAIN scripts and the AOI Observer.

Pipeline order (matches biological defense hierarchy):
  1. VDJ Recombination     - Generate strategy antibodies
  2. Protective Deletion   - Suppress toxic patterns
  3. CRISPR-Cas9           - Immune memory gate (block known losers)
  4. Electric Organs        - Cross-instrument convergence boost
  5. KoRV                  - New signal onboarding weights
  6. Bdelloid Rotifers     - Horizontal gene transfer (strategy sharing)
  7. Toxoplasma            - Regime behavior modification detection
  8. Syncytin              - Strategy fusion (hybrid organisms)

Usage:
    from aoi_pipeline import AOIPipeline

    pipeline = AOIPipeline(symbols=['BTCUSD', 'ETHUSD'])
    result = pipeline.process(
        symbol='BTCUSD',
        direction=1,
        confidence=0.65,
        bars=bars_array,
        active_tes=['LINE', 'Alu', 'Ty3_gypsy'],
        domestication_boost=1.3,
    )

Author: James Jardine / Quantum Children
Date: 2026-02-08
"""

import logging
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

log = logging.getLogger(__name__)

# ============================================================
# ALGORITHM IMPORTS (each wrapped in try/except for resilience)
# ============================================================

# 1. VDJ Recombination
try:
    from vdj_recombination import VDJRecombinationEngine, VDJTEQABridge
    VDJ_AVAILABLE = True
except ImportError as e:
    log.warning(f"VDJ Recombination not available: {e}")
    VDJ_AVAILABLE = False

# 2. Protective Deletion
try:
    from protective_deletion import ProtectiveDeletionTracker
    PROTECTIVE_DELETION_AVAILABLE = True
except ImportError as e:
    log.warning(f"Protective Deletion not available: {e}")
    PROTECTIVE_DELETION_AVAILABLE = False

# 3. CRISPR-Cas9
try:
    from crispr_cas import CRISPRTEQABridge
    CRISPR_AVAILABLE = True
except ImportError as e:
    log.warning(f"CRISPR-Cas9 not available: {e}")
    CRISPR_AVAILABLE = False

# 4. Electric Organs
try:
    from electric_organs import ElectricOrgansBridge
    ELECTRIC_ORGANS_AVAILABLE = True
except ImportError as e:
    log.warning(f"Electric Organs not available: {e}")
    ELECTRIC_ORGANS_AVAILABLE = False

# 5. KoRV
try:
    from korv import KoRVDomesticationEngine, KoRVTEQABridge
    KORV_AVAILABLE = True
except ImportError as e:
    log.warning(f"KoRV not available: {e}")
    KORV_AVAILABLE = False

# 6. Bdelloid Rotifers
try:
    from bdelloid_rotifers import BdelloidHGTEngine, BdelloidTEQABridge
    BDELLOID_AVAILABLE = True
except ImportError as e:
    log.warning(f"Bdelloid Rotifers not available: {e}")
    BDELLOID_AVAILABLE = False

# 7. Toxoplasma
try:
    from toxoplasma import ToxoplasmaEngine, ToxoplasmaTEQABridge
    TOXOPLASMA_AVAILABLE = True
except ImportError as e:
    log.warning(f"Toxoplasma not available: {e}")
    TOXOPLASMA_AVAILABLE = False

# 8. Syncytin
try:
    from syncytin import SyncytinFusionEngine, StrategyProfile
    SYNCYTIN_AVAILABLE = True
except ImportError as e:
    log.warning(f"Syncytin not available: {e}")
    SYNCYTIN_AVAILABLE = False

# TEQA core (for TE activations and domestication)
try:
    from teqa_v3_neural_te import TEActivationEngine, TEDomesticationTracker
    TEQA_CORE_AVAILABLE = True
except ImportError as e:
    log.warning(f"TEQA core not available: {e}")
    TEQA_CORE_AVAILABLE = False


# ============================================================
# RESULT DATA STRUCTURES
# ============================================================

@dataclass
class AlgorithmResult:
    """Result from a single biological algorithm."""
    name: str
    available: bool
    ran: bool = False
    elapsed_ms: float = 0.0
    gate_pass: bool = True
    confidence_modifier: float = 1.0
    direction_override: Optional[int] = None
    details: Dict = field(default_factory=dict)
    error: str = ""


@dataclass
class AOIResult:
    """Combined result from the full AOI pipeline."""
    timestamp: str = ""
    symbol: str = ""
    original_direction: int = 0
    original_confidence: float = 0.0
    final_direction: int = 0
    final_confidence: float = 0.0
    final_gate_pass: bool = True
    active_tes: List[str] = field(default_factory=list)
    domestication_boost: float = 1.0
    n_algorithms_ran: int = 0
    n_algorithms_available: int = 0
    total_elapsed_ms: float = 0.0
    algorithms: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "original_direction": self.original_direction,
            "original_confidence": self.original_confidence,
            "final_direction": self.final_direction,
            "final_confidence": self.final_confidence,
            "final_gate_pass": self.final_gate_pass,
            "active_tes": self.active_tes,
            "domestication_boost": self.domestication_boost,
            "n_algorithms_ran": self.n_algorithms_ran,
            "n_algorithms_available": self.n_algorithms_available,
            "total_elapsed_ms": self.total_elapsed_ms,
            "algorithms": self.algorithms,
        }


# ============================================================
# AOI PIPELINE
# ============================================================

class AOIPipeline:
    """
    The Artificial Organism Intelligence pipeline.

    Initializes all 8 biological algorithms and provides a single
    process() method that runs the full defense/adaptation cascade.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        db_dir: str = None,
    ):
        self.symbols = symbols or ["BTCUSD", "ETHUSD"]
        if db_dir is None:
            self.db_dir = str(Path(__file__).parent)
        else:
            self.db_dir = db_dir

        # Track availability
        self.available = {}

        # Initialize TEQA core
        self.te_engine = None
        self.domestication_tracker = None
        if TEQA_CORE_AVAILABLE:
            self.te_engine = TEActivationEngine()
            self.domestication_tracker = TEDomesticationTracker()

        # Initialize all 8 algorithms
        self._init_algorithms()

        avail_count = sum(1 for v in self.available.values() if v)
        log.info(f"AOI Pipeline initialized: {avail_count}/8 algorithms available")

    def _init_algorithms(self):
        """Initialize each algorithm with error isolation."""

        # 1. VDJ Recombination
        self.vdj_engine = None
        self.vdj_bridge = None
        try:
            if VDJ_AVAILABLE:
                self.vdj_engine = VDJRecombinationEngine()
                self.vdj_bridge = VDJTEQABridge(vdj_engine=self.vdj_engine)
                self.available["vdj"] = True
            else:
                self.available["vdj"] = False
        except Exception as e:
            log.warning(f"VDJ init failed: {e}")
            self.available["vdj"] = False

        # 2. Protective Deletion
        self.protective_deletion = None
        try:
            if PROTECTIVE_DELETION_AVAILABLE:
                self.protective_deletion = ProtectiveDeletionTracker(
                    domestication_tracker=self.domestication_tracker,
                )
                self.available["protective_deletion"] = True
            else:
                self.available["protective_deletion"] = False
        except Exception as e:
            log.warning(f"Protective Deletion init failed: {e}")
            self.available["protective_deletion"] = False

        # 3. CRISPR-Cas9
        self.crispr = None
        try:
            if CRISPR_AVAILABLE:
                self.crispr = CRISPRTEQABridge()
                self.available["crispr"] = True
            else:
                self.available["crispr"] = False
        except Exception as e:
            log.warning(f"CRISPR init failed: {e}")
            self.available["crispr"] = False

        # 4. Electric Organs
        self.electric_organs = None
        try:
            if ELECTRIC_ORGANS_AVAILABLE:
                self.electric_organs = ElectricOrgansBridge(
                    lineage_symbols=self.symbols,
                    db_dir=self.db_dir,
                )
                self.available["electric_organs"] = True
            else:
                self.available["electric_organs"] = False
        except Exception as e:
            log.warning(f"Electric Organs init failed: {e}")
            self.available["electric_organs"] = False

        # 5. KoRV
        self.korv_engine = None
        self.korv = None
        try:
            if KORV_AVAILABLE:
                self.korv_engine = KoRVDomesticationEngine()
                self.korv = KoRVTEQABridge(engine=self.korv_engine)
                self.available["korv"] = True
            else:
                self.available["korv"] = False
        except Exception as e:
            log.warning(f"KoRV init failed: {e}")
            self.available["korv"] = False

        # 6. Bdelloid Rotifers
        self.bdelloid_engine = None
        self.bdelloid_bridge = None
        try:
            if BDELLOID_AVAILABLE:
                self.bdelloid_engine = BdelloidHGTEngine()
                self.bdelloid_bridge = BdelloidTEQABridge(
                    hgt_engine=self.bdelloid_engine,
                )
                self.available["bdelloid"] = True
            else:
                self.available["bdelloid"] = False
        except Exception as e:
            log.warning(f"Bdelloid init failed: {e}")
            self.available["bdelloid"] = False

        # 7. Toxoplasma
        self.toxoplasma_engine = None
        self.toxoplasma = None
        try:
            if TOXOPLASMA_AVAILABLE:
                self.toxoplasma_engine = ToxoplasmaEngine()
                self.toxoplasma = ToxoplasmaTEQABridge(engine=self.toxoplasma_engine)
                self.available["toxoplasma"] = True
            else:
                self.available["toxoplasma"] = False
        except Exception as e:
            log.warning(f"Toxoplasma init failed: {e}")
            self.available["toxoplasma"] = False

        # 8. Syncytin
        self.syncytin = None
        try:
            if SYNCYTIN_AVAILABLE:
                self.syncytin = SyncytinFusionEngine()
                self.available["syncytin"] = True
            else:
                self.available["syncytin"] = False
        except Exception as e:
            log.warning(f"Syncytin init failed: {e}")
            self.available["syncytin"] = False

    def get_te_activations(self, bars: np.ndarray) -> Tuple[List[str], float, List[Dict]]:
        """
        Run TEQA TE activation engine on bars data.

        Returns:
            (active_te_names, domestication_boost, full_activations)
        """
        if self.te_engine is None:
            return [], 1.0, []

        activations = self.te_engine.compute_all_activations(bars)

        # Extract active TEs (strength > 0.5)
        active_tes = [a["te"] for a in activations if a["strength"] > 0.5]

        # Get domestication boost
        boost = 1.0
        if self.domestication_tracker and active_tes:
            boost = self.domestication_tracker.get_boost(active_tes)

        return active_tes, boost, activations

    def process(
        self,
        symbol: str,
        direction: int,
        confidence: float,
        bars: np.ndarray,
        active_tes: List[str] = None,
        domestication_boost: float = 1.0,
        strategy_id: str = "default",
        timeframe: str = "M1",
    ) -> AOIResult:
        """
        Run the full AOI pipeline.

        Args:
            symbol: Trading instrument (e.g. 'BTCUSD')
            direction: 1 (long), -1 (short), 0 (neutral)
            confidence: Base confidence score (0.0-1.0)
            bars: OHLCV numpy array (N x 5+)
            active_tes: List of active TE family names
            domestication_boost: Boost from TEDomesticationTracker
            strategy_id: Strategy identifier for Toxoplasma/Bdelloid
            timeframe: Timeframe string for KoRV

        Returns:
            AOIResult with full pipeline output
        """
        t0 = time.perf_counter()

        if active_tes is None:
            active_tes = []

        result = AOIResult(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            original_direction=direction,
            original_confidence=confidence,
            active_tes=active_tes,
            domestication_boost=domestication_boost,
            n_algorithms_available=sum(1 for v in self.available.values() if v),
        )

        # Running state - modified by each algorithm
        curr_direction = direction
        curr_confidence = confidence
        gate_pass = True

        # === 1. VDJ Recombination (Gate G11) ===
        algo = self._run_vdj(bars)
        result.algorithms["vdj"] = self._algo_to_dict(algo)
        if algo.ran and not algo.gate_pass:
            gate_pass = False
        if algo.ran:
            curr_confidence *= algo.confidence_modifier

        # === 2. Protective Deletion ===
        algo = self._run_protective_deletion(active_tes)
        result.algorithms["protective_deletion"] = self._algo_to_dict(algo)
        if algo.ran:
            curr_confidence *= algo.confidence_modifier

        # === 3. CRISPR-Cas9 (Gate G12) ===
        algo = self._run_crispr(symbol, curr_direction, bars, active_tes,
                                domestication_boost, curr_confidence)
        result.algorithms["crispr"] = self._algo_to_dict(algo)
        if algo.ran and not algo.gate_pass:
            gate_pass = False

        # === 4. Electric Organs ===
        algo = self._run_electric_organs(active_tes, domestication_boost, symbol)
        result.algorithms["electric_organs"] = self._algo_to_dict(algo)
        if algo.ran:
            curr_confidence *= algo.confidence_modifier

        # === 5. KoRV ===
        algo = self._run_korv(curr_confidence, active_tes, symbol, timeframe)
        result.algorithms["korv"] = self._algo_to_dict(algo)
        if algo.ran:
            curr_confidence = algo.details.get("weighted_confidence", curr_confidence)
            if not algo.gate_pass:
                gate_pass = False

        # === 6. Bdelloid Rotifers ===
        algo = self._run_bdelloid(strategy_id)
        result.algorithms["bdelloid"] = self._algo_to_dict(algo)
        if algo.ran:
            curr_confidence *= algo.confidence_modifier
            if not algo.gate_pass:
                gate_pass = False

        # === 7. Toxoplasma ===
        algo = self._run_toxoplasma(strategy_id, symbol, curr_direction,
                                     curr_confidence, bars, active_tes)
        result.algorithms["toxoplasma"] = self._algo_to_dict(algo)
        if algo.ran:
            curr_direction = algo.direction_override or curr_direction
            curr_confidence = algo.details.get("modified_confidence", curr_confidence)

        # === 8. Syncytin ===
        algo = self._run_syncytin()
        result.algorithms["syncytin"] = self._algo_to_dict(algo)

        # Finalize
        elapsed = (time.perf_counter() - t0) * 1000
        result.final_direction = curr_direction
        result.final_confidence = max(0.0, min(1.0, curr_confidence))
        result.final_gate_pass = gate_pass
        result.total_elapsed_ms = round(elapsed, 2)
        result.n_algorithms_ran = sum(
            1 for a in result.algorithms.values() if a.get("ran", False)
        )

        return result

    # ============================================================
    # INDIVIDUAL ALGORITHM RUNNERS
    # ============================================================

    def _run_vdj(self, bars: np.ndarray) -> AlgorithmResult:
        r = AlgorithmResult(name="vdj", available=self.available.get("vdj", False))
        if not r.available or self.vdj_bridge is None:
            return r
        try:
            t0 = time.perf_counter()
            gate = self.vdj_bridge.get_vdj_gate_result(bars)
            r.elapsed_ms = (time.perf_counter() - t0) * 1000
            r.ran = True
            r.gate_pass = gate.get("gate_pass", True)
            r.confidence_modifier = gate.get("confidence", 1.0) if r.gate_pass else 0.8
            r.details = gate
        except Exception as e:
            r.error = str(e)
            log.warning(f"VDJ error: {e}")
        return r

    def _run_protective_deletion(self, active_tes: List[str]) -> AlgorithmResult:
        r = AlgorithmResult(name="protective_deletion",
                           available=self.available.get("protective_deletion", False))
        if not r.available or self.protective_deletion is None:
            return r
        try:
            t0 = time.perf_counter()
            modifier = self.protective_deletion.get_combined_modifier(active_tes)
            r.elapsed_ms = (time.perf_counter() - t0) * 1000
            r.ran = True
            r.confidence_modifier = modifier
            r.details = {
                "combined_modifier": modifier,
                "suppression": self.protective_deletion.get_suppression(active_tes),
            }
        except Exception as e:
            r.error = str(e)
            log.warning(f"Protective Deletion error: {e}")
        return r

    def _run_crispr(self, symbol: str, direction: int, bars: np.ndarray,
                    active_tes: List[str], domestication_boost: float,
                    confidence: float) -> AlgorithmResult:
        r = AlgorithmResult(name="crispr",
                           available=self.available.get("crispr", False))
        if not r.available or self.crispr is None:
            return r
        try:
            t0 = time.perf_counter()
            gate = self.crispr.gate_check(
                symbol=symbol,
                direction=direction,
                bars=bars,
                active_tes=active_tes,
                domestication_boost=domestication_boost,
                confidence=confidence,
            )
            r.elapsed_ms = (time.perf_counter() - t0) * 1000
            r.ran = True
            r.gate_pass = gate.get("gate_pass", True)
            r.details = gate
        except Exception as e:
            r.error = str(e)
            log.warning(f"CRISPR error: {e}")
        return r

    def _run_electric_organs(self, active_tes: List[str],
                             domestication_boost: float,
                             symbol: str) -> AlgorithmResult:
        r = AlgorithmResult(name="electric_organs",
                           available=self.available.get("electric_organs", False))
        if not r.available or self.electric_organs is None:
            return r
        try:
            t0 = time.perf_counter()
            boosted = self.electric_organs.apply(
                active_tes=active_tes,
                domestication_boost=domestication_boost,
                symbol=symbol,
            )
            r.elapsed_ms = (time.perf_counter() - t0) * 1000
            r.ran = True
            # The return is the new boost factor. Confidence modifier is ratio.
            if domestication_boost > 0:
                r.confidence_modifier = boosted / domestication_boost
            else:
                r.confidence_modifier = 1.0
            r.details = {
                "original_boost": domestication_boost,
                "convergent_boost": boosted,
            }
        except Exception as e:
            r.error = str(e)
            log.warning(f"Electric Organs error: {e}")
        return r

    def _run_korv(self, confidence: float, active_tes: List[str],
                  instrument: str, timeframe: str) -> AlgorithmResult:
        r = AlgorithmResult(name="korv",
                           available=self.available.get("korv", False))
        if not r.available or self.korv is None:
            return r
        try:
            t0 = time.perf_counter()
            # Use active TEs as signal types
            signal_types = active_tes[:5] if active_tes else ["default"]
            weighted = self.korv.get_weighted_confidence(
                base_confidence=confidence,
                signal_types=signal_types,
                instrument=instrument,
                timeframe=timeframe,
            )
            gate = self.korv.get_gate_result(
                signal_types=signal_types,
                instrument=instrument,
                timeframe=timeframe,
            )
            r.elapsed_ms = (time.perf_counter() - t0) * 1000
            r.ran = True
            r.gate_pass = gate.get("gate_pass", True)
            r.details = {
                "weighted_confidence": weighted,
                "gate": gate,
            }
        except Exception as e:
            r.error = str(e)
            log.warning(f"KoRV error: {e}")
        return r

    def _run_bdelloid(self, strategy_id: str) -> AlgorithmResult:
        r = AlgorithmResult(name="bdelloid",
                           available=self.available.get("bdelloid", False))
        if not r.available or self.bdelloid_bridge is None:
            return r
        try:
            t0 = time.perf_counter()
            gate = self.bdelloid_bridge.get_hgt_gate_result(strategy_id=strategy_id)
            r.elapsed_ms = (time.perf_counter() - t0) * 1000
            r.ran = True
            r.gate_pass = gate.get("gate_pass", True)
            r.confidence_modifier = gate.get("confidence_modifier", 1.0)
            r.details = gate
        except Exception as e:
            r.error = str(e)
            log.warning(f"Bdelloid error: {e}")
        return r

    def _run_toxoplasma(self, strategy_id: str, symbol: str,
                        direction: int, confidence: float,
                        bars: np.ndarray, active_tes: List[str]) -> AlgorithmResult:
        r = AlgorithmResult(name="toxoplasma",
                           available=self.available.get("toxoplasma", False))
        if not r.available or self.toxoplasma is None:
            return r
        if direction == 0:
            # Toxoplasma needs a directional signal
            r.ran = True
            r.details = {"skipped": "neutral direction"}
            return r
        try:
            t0 = time.perf_counter()
            mod_dir, mod_conf, infection_score = self.toxoplasma.apply_to_signal(
                strategy_id=strategy_id,
                symbol=symbol,
                original_direction=direction,
                original_confidence=confidence,
                bars=bars,
                active_tes=active_tes,
            )
            r.elapsed_ms = (time.perf_counter() - t0) * 1000
            r.ran = True
            r.direction_override = mod_dir
            r.details = {
                "modified_direction": mod_dir,
                "modified_confidence": mod_conf,
                "infection_score": infection_score,
            }
        except Exception as e:
            r.error = str(e)
            log.warning(f"Toxoplasma error: {e}")
        return r

    def _run_syncytin(self) -> AlgorithmResult:
        r = AlgorithmResult(name="syncytin",
                           available=self.available.get("syncytin", False))
        if not r.available or self.syncytin is None:
            return r
        try:
            t0 = time.perf_counter()
            # Syncytin operates on strategy pools, not individual signals
            # Report current fusion state
            n_hybrids = len(self.syncytin.hybrids) if hasattr(self.syncytin, 'hybrids') else 0
            n_pool = len(self.syncytin.strategy_pool) if hasattr(self.syncytin, 'strategy_pool') else 0
            r.elapsed_ms = (time.perf_counter() - t0) * 1000
            r.ran = True
            r.details = {
                "n_hybrids": n_hybrids,
                "n_strategy_pool": n_pool,
                "status": "monitoring",
            }
        except Exception as e:
            r.error = str(e)
            log.warning(f"Syncytin error: {e}")
        return r

    @staticmethod
    def _algo_to_dict(algo: AlgorithmResult) -> dict:
        return {
            "name": algo.name,
            "available": algo.available,
            "ran": algo.ran,
            "elapsed_ms": round(algo.elapsed_ms, 2),
            "gate_pass": algo.gate_pass,
            "confidence_modifier": round(algo.confidence_modifier, 4),
            "direction_override": algo.direction_override,
            "details": algo.details,
            "error": algo.error,
        }

    def get_status(self) -> Dict:
        """Get pipeline status summary."""
        return {
            "algorithms": {
                name: avail for name, avail in self.available.items()
            },
            "n_available": sum(1 for v in self.available.values() if v),
            "n_total": 8,
            "teqa_core": TEQA_CORE_AVAILABLE,
            "symbols": self.symbols,
        }
