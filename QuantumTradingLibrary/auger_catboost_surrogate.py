"""
AUGER CATBOOST SURROGATE - ML Surrogate for Auger Cascade Simulation
=====================================================================
Trains CatBoost to predict treatment outcomes (DSB yield, cell kill,
therapeutic ratio) from cascade quantum features -- replacing the
expensive Monte Carlo + Qiskit simulation with instant ML inference.

Same pattern as ai_trader_quantum_fusion.py:
  Trading:  quantum features + technicals -> CatBoost -> UP/DOWN
  Physics:  quantum features + shell stats -> CatBoost -> DSB/cell_kill

Run with GPU venv:
    .venv312_gpu\\Scripts\\python.exe auger_catboost_surrogate.py --generate 2000
    .venv312_gpu\\Scripts\\python.exe auger_catboost_surrogate.py --train
    .venv312_gpu\\Scripts\\python.exe auger_catboost_surrogate.py --predict I-125

Authors: DooDoo + Claude
Date:    2026-02-11
"""

import json
import logging
import math
import os
import random
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# CatBoost
from catboost import CatBoostRegressor, Pool

# Sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Local: Auger simulation engine
from auger_cascade_core import (
    RADIONUCLIDE_DATA,
    SHELL_ORDER,
    SHELL_MAX_OCCUPANCY,
    SHELL_ENERGIES_BY_NUCLIDE,
    FLUORESCENCE_YIELDS,
    DEA_RESONANCES,
    DecayCascade,
    DamageSite,
    RepairConfig,
    DNATarget,
    ShellQuantumEncoder,
    cascade_branching,
    warburg_quantum_batch,
    angiogenesis,
    lethal_dsb_formation,
    repair_knockout,
    dna_repair_checkpoint,
    run_auger_treatment,
)

log = logging.getLogger("AUGER_SURROGATE")

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
TRAINING_DATA_PATH = BASE_DIR / "auger_surrogate_data.csv"
MODEL_DIR = BASE_DIR / "auger_surrogate_models"
CASCADE_MODEL_PATH = MODEL_DIR / "cascade_dsb_model.cbm"
RUN_MODEL_PATH = MODEL_DIR / "run_cellkill_model.cbm"
THERAPEUTIC_MODEL_PATH = MODEL_DIR / "run_therapeutic_model.cbm"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_cascade_features(cascade: DecayCascade,
                              nuclide: str) -> Dict[str, Any]:
    """
    Extract ML features from a single cascade.

    Features mirror the quantum encoder output from the trading system:
      Trading:  RSI, MACD, ATR, ... + quantum_entropy, dominant_state_prob, ...
      Physics:  shell stats, energy bands + quantum_entropy, dominant_state_prob, ...
    """
    energies = cascade.electron_energies
    shells = cascade.electron_shells
    n_electrons = len(energies)

    features = {
        "nuclide": nuclide,
        "n_electrons": n_electrons,
        "total_energy_eV": cascade.total_energy_eV,
        "charge_state": cascade.final_charge_state,
    }

    # --- Energy statistics ---
    if n_electrons > 0:
        e_arr = np.array(energies)
        features["energy_mean"] = float(np.mean(e_arr))
        features["energy_median"] = float(np.median(e_arr))
        features["energy_std"] = float(np.std(e_arr))
        features["energy_min"] = float(np.min(e_arr))
        features["energy_max"] = float(np.max(e_arr))
        features["energy_skew"] = float(
            ((e_arr - e_arr.mean()) ** 3).mean() / (e_arr.std() ** 3 + 1e-15)
        )
        # Energy band fractions
        features["frac_dea"] = float(np.sum(e_arr < 20) / n_electrons)
        features["frac_mid"] = float(np.sum((e_arr >= 20) & (e_arr < 100)) / n_electrons)
        features["frac_high"] = float(np.sum(e_arr >= 100) / n_electrons)
        # DEA resonance proximity score
        dea_score = 0.0
        for e in energies:
            if e < 25:
                for res in DEA_RESONANCES:
                    diff = abs(e - res["energy_eV"])
                    if diff < 2.0:
                        dea_score += res["sigma"] / (1.0 + diff)
        features["dea_resonance_score"] = dea_score
    else:
        for k in ["energy_mean", "energy_median", "energy_std", "energy_min",
                   "energy_max", "energy_skew", "frac_dea", "frac_mid",
                   "frac_high", "dea_resonance_score"]:
            features[k] = 0.0

    # --- Shell distribution ---
    shell_groups = {"K": 0, "L": 0, "M": 0, "N": 0, "O": 0}
    for s in shells:
        group = s[0] if s else "O"
        shell_groups[group] = shell_groups.get(group, 0) + 1
    total_s = max(1, sum(shell_groups.values()))
    for g in ["K", "L", "M", "N", "O"]:
        features[f"shell_frac_{g}"] = shell_groups[g] / total_s

    # --- Transition statistics ---
    n_transitions = len(cascade.transitions)
    features["n_transitions"] = n_transitions
    if n_transitions > 0:
        trans_energies = [t.energy_eV for t in cascade.transitions]
        features["trans_energy_mean"] = float(np.mean(trans_energies))
        features["trans_energy_std"] = float(np.std(trans_energies))
        # Overlap integral stats from quantum encoding
        overlaps = [t.overlap_integral for t in cascade.transitions if t.overlap_integral > 0]
        features["overlap_mean"] = float(np.mean(overlaps)) if overlaps else 0.0
        features["dos_mean"] = float(np.mean([t.density_of_states for t in cascade.transitions]))
        features["corr_mean"] = float(np.mean([t.correlation_factor for t in cascade.transitions]))
    else:
        for k in ["trans_energy_mean", "trans_energy_std", "overlap_mean",
                   "dos_mean", "corr_mean"]:
            features[k] = 0.0

    # --- Quantum features (shell occupancy encoding) ---
    # Same circuit topology as trading QuantumEncoder
    encoder = ShellQuantumEncoder(n_qubits=min(15, 10), n_shots=1024)
    occupancies = []
    for shell in SHELL_ORDER:
        max_occ = SHELL_MAX_OCCUPANCY.get(shell, 2)
        ejected = sum(1 for s in shells if s == shell)
        remaining = max(0, max_occ - ejected)
        occupancies.append(remaining / max_occ)
    qf = encoder.encode(np.array(occupancies))
    features["q_entropy"] = qf["entropy"]
    features["q_dominant"] = qf["dominant_state_prob"]
    features["q_significant"] = qf["significant_states"]
    features["q_variance"] = qf["variance"]
    features["q_coherence"] = qf["coherence_score"]

    # --- Fluorescence yield weighted average ---
    if shells:
        omega_vals = [FLUORESCENCE_YIELDS.get(s, 0.01) for s in shells]
        features["fluor_yield_avg"] = float(np.mean(omega_vals))
    else:
        features["fluor_yield_avg"] = 0.5

    return features


def extract_run_features(cascades: List[DecayCascade],
                          damage_sites: List[DamageSite],
                          nuclide: str,
                          repair: RepairConfig) -> Dict[str, Any]:
    """
    Extract run-level aggregate features for cell_kill / therapeutic ratio prediction.
    """
    n_cascades = len(cascades)

    # Aggregate cascade stats
    all_electrons = [c.total_electrons for c in cascades]
    all_energies = [c.total_energy_eV for c in cascades]
    all_charges = [c.final_charge_state for c in cascades]

    features = {
        "nuclide": nuclide,
        "n_decays": n_cascades,
        "p53_functional": int(repair.p53_functional),
        "brca_functional": int(repair.brca_functional),
    }

    if n_cascades > 0:
        features["electrons_mean"] = float(np.mean(all_electrons))
        features["electrons_std"] = float(np.std(all_electrons))
        features["energy_mean"] = float(np.mean(all_energies))
        features["energy_std"] = float(np.std(all_energies))
        features["charge_mean"] = float(np.mean(all_charges))
        features["charge_std"] = float(np.std(all_charges))

        # Budget distribution
        budgets = [c.simulation_budget for c in cascades]
        features["frac_full"] = budgets.count("full") / n_cascades
        features["frac_pruned"] = budgets.count("pruned") / n_cascades
    else:
        for k in ["electrons_mean", "electrons_std", "energy_mean", "energy_std",
                   "charge_mean", "charge_std", "frac_full", "frac_pruned"]:
            features[k] = 0.0

    # Damage stats
    n_ssb = sum(1 for s in damage_sites if s.damage_type == "SSB")
    n_dsb = sum(1 for s in damage_sites if s.damage_type == "DSB")
    n_base = sum(1 for s in damage_sites if s.damage_type == "base_lesion")
    n_total = len(damage_sites)

    features["total_damage_sites"] = n_total
    features["ssb_count"] = n_ssb
    features["dsb_count"] = n_dsb
    features["base_lesion_count"] = n_base
    features["dsb_per_decay"] = n_dsb / max(1, n_cascades)
    features["ssb_per_decay"] = n_ssb / max(1, n_cascades)

    # DEA damage fraction
    n_dea = sum(1 for s in damage_sites if s.mechanism == "DEA")
    features["dea_damage_frac"] = n_dea / max(1, n_total)

    # Lethal fraction (pre-repair)
    n_lethal_pre = sum(1 for s in damage_sites if s.is_lethal)
    features["lethal_frac_pre_repair"] = n_lethal_pre / max(1, n_total)

    # Aggregate quantum features across cascades
    encoder = ShellQuantumEncoder(n_qubits=10, n_shots=512)
    q_entropies = []
    q_dominants = []
    q_coherences = []

    # Sample up to 50 cascades for speed
    sample_cascades = cascades[:50] if len(cascades) > 50 else cascades
    for c in sample_cascades:
        occupancies = []
        for shell in SHELL_ORDER:
            max_occ = SHELL_MAX_OCCUPANCY.get(shell, 2)
            ejected = sum(1 for s in c.electron_shells if s == shell)
            remaining = max(0, max_occ - ejected)
            occupancies.append(remaining / max_occ)
        qf = encoder.encode(np.array(occupancies))
        q_entropies.append(qf["entropy"])
        q_dominants.append(qf["dominant_state_prob"])
        q_coherences.append(qf["coherence_score"])

    features["q_entropy_mean"] = float(np.mean(q_entropies)) if q_entropies else 0.0
    features["q_dominant_mean"] = float(np.mean(q_dominants)) if q_dominants else 0.0
    features["q_coherence_mean"] = float(np.mean(q_coherences)) if q_coherences else 0.0

    return features


# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

def generate_training_data(n_samples: int = 2000,
                            save_path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """
    Generate training data by running many cascade simulations
    with varied parameters.

    Each sample = one cascade with features + targets.
    """
    print("=" * 70)
    print("GENERATING TRAINING DATA FOR CATBOOST SURROGATE")
    print("=" * 70)
    print(f"  Target samples:  {n_samples}")
    print(f"  Output:          {save_path}")
    print()

    nuclides = list(RADIONUCLIDE_DATA.keys())
    # How many decays per batch (we extract per-cascade features)
    decays_per_batch = 50
    n_batches = max(1, n_samples // decays_per_batch)

    all_cascade_rows = []
    all_run_rows = []
    t_start = time.time()

    for batch_i in range(n_batches):
        nuclide = nuclides[batch_i % len(nuclides)]
        shell_energies = SHELL_ENERGIES_BY_NUCLIDE.get(nuclide)

        # Vary repair configs
        p53 = random.choice([True, False])
        brca = random.choice([True, False])
        cancer_repair = RepairConfig(
            p53_functional=p53, brca_functional=brca,
            atm_functional=random.choice([True, False]),
            label=f"gen_p53{'+' if p53 else '-'}_brca{'+' if brca else '-'}",
        )
        healthy_repair = RepairConfig(
            p53_functional=True, brca_functional=True,
            atm_functional=True, label="healthy",
        )

        # Phase 1: Cascade
        cascades = cascade_branching(nuclide, decays_per_batch)

        # Phase 3: Quantum encoding
        cascades = warburg_quantum_batch(cascades, shell_energies)

        # Phase 4: Angiogenesis
        cascades = angiogenesis(cascades)

        # Phase 6: DNA damage
        dna_target = DNATarget(length_bp=10000)
        damage_sites, dsb_per_decay = lethal_dsb_formation(cascades, dna_target)

        # Phase 2: Repair knockout
        healthy_damage, cancer_damage, therapeutic_ratio = repair_knockout(
            damage_sites, healthy_repair, cancer_repair
        )

        # Phase 7: Repair checkpoint
        surviving, cell_survival = dna_repair_checkpoint(damage_sites, cancer_repair)

        # --- Extract cascade-level features + targets ---
        # Assign damage to cascades
        damage_by_cascade = {}
        for site in damage_sites:
            cid = site.cascade_id
            if cid not in damage_by_cascade:
                damage_by_cascade[cid] = {"ssb": 0, "dsb": 0, "base": 0, "lethal": 0, "total": 0}
            damage_by_cascade[cid]["total"] += 1
            if site.damage_type == "SSB":
                damage_by_cascade[cid]["ssb"] += 1
            elif site.damage_type == "DSB":
                damage_by_cascade[cid]["dsb"] += 1
            elif site.damage_type == "base_lesion":
                damage_by_cascade[cid]["base"] += 1
            if site.is_lethal:
                damage_by_cascade[cid]["lethal"] += 1

        for cascade in cascades:
            features = extract_cascade_features(cascade, nuclide)
            # Add repair config as features
            features["p53_functional"] = int(p53)
            features["brca_functional"] = int(brca)

            # Targets
            dmg = damage_by_cascade.get(cascade.cascade_id,
                                         {"ssb": 0, "dsb": 0, "base": 0, "lethal": 0, "total": 0})
            features["target_dsb"] = dmg["dsb"]
            features["target_ssb"] = dmg["ssb"]
            features["target_lethal"] = dmg["lethal"]
            features["target_total_damage"] = dmg["total"]

            all_cascade_rows.append(features)

        # --- Extract run-level features + targets ---
        run_features = extract_run_features(cascades, damage_sites, nuclide, cancer_repair)
        run_features["target_cell_kill"] = 1.0 - cell_survival
        run_features["target_therapeutic_ratio"] = therapeutic_ratio
        run_features["target_dsb_per_decay"] = dsb_per_decay
        all_run_rows.append(run_features)

        elapsed = time.time() - t_start
        total_samples = len(all_cascade_rows)
        pct = total_samples / n_samples * 100
        if (batch_i + 1) % 5 == 0 or batch_i == 0:
            print(f"  Batch {batch_i+1}/{n_batches} | {total_samples} cascade samples "
                  f"| {len(all_run_rows)} run samples | {pct:.0f}% | {elapsed:.1f}s")

        if total_samples >= n_samples:
            break

    # Convert to DataFrames
    cascade_df = pd.DataFrame(all_cascade_rows[:n_samples])
    run_df = pd.DataFrame(all_run_rows)

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cascade_df.to_csv(save_path, index=False)

    run_save_path = save_path.parent / "auger_surrogate_run_data.csv"
    run_df.to_csv(run_save_path, index=False)

    elapsed = time.time() - t_start
    print()
    print(f"  CASCADE data: {len(cascade_df)} rows x {len(cascade_df.columns)} cols")
    print(f"  RUN data:     {len(run_df)} rows x {len(run_df.columns)} cols")
    print(f"  Saved to:     {save_path}")
    print(f"  Saved to:     {run_save_path}")
    print(f"  Total time:   {elapsed:.1f}s")
    print()

    return cascade_df


# ============================================================================
# TRAINING
# ============================================================================

CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "MAE",
    "random_seed": 42,
    "verbose": 200,
    "l2_leaf_reg": 3.0,
    "min_data_in_leaf": 5,
}

CASCADE_FEATURE_COLS = [
    "n_electrons", "total_energy_eV", "charge_state",
    "energy_mean", "energy_median", "energy_std", "energy_min", "energy_max",
    "energy_skew", "frac_dea", "frac_mid", "frac_high", "dea_resonance_score",
    "shell_frac_K", "shell_frac_L", "shell_frac_M", "shell_frac_N", "shell_frac_O",
    "n_transitions", "trans_energy_mean", "trans_energy_std",
    "overlap_mean", "dos_mean", "corr_mean",
    "q_entropy", "q_dominant", "q_significant", "q_variance", "q_coherence",
    "fluor_yield_avg",
    "p53_functional", "brca_functional",
]

CASCADE_CAT_FEATURES = ["nuclide"]

CASCADE_TARGETS = ["target_dsb", "target_ssb", "target_lethal", "target_total_damage"]


def train_cascade_model(df: pd.DataFrame,
                         target: str = "target_dsb") -> CatBoostRegressor:
    """
    Train CatBoost to predict per-cascade DSB count from features.

    Same pattern as ai_trader_quantum_fusion.py but regression instead of
    classification, and k-fold CV instead of walk-forward (not time series).
    """
    print(f"\n{'='*70}")
    print(f"TRAINING CASCADE MODEL: {target}")
    print(f"{'='*70}")

    # Prepare features
    feature_cols = CASCADE_FEATURE_COLS + CASCADE_CAT_FEATURES
    available = [c for c in feature_cols if c in df.columns]
    cat_idx = [available.index(c) for c in CASCADE_CAT_FEATURES if c in available]

    X = df[available].copy()
    y = df[target].values.astype(float)

    print(f"  Features:  {len(available)}")
    print(f"  Samples:   {len(X)}")
    print(f"  Target:    {target}")
    print(f"  y range:   [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.3f}")
    print()

    # K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = CatBoostRegressor(
            **CATBOOST_PARAMS,
            cat_features=cat_idx if cat_idx else None,
        )
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=0,
        )

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        fold_scores.append({"mae": mae, "rmse": rmse, "r2": r2})
        print(f"  Fold {fold_i+1}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    avg_mae = np.mean([s["mae"] for s in fold_scores])
    avg_r2 = np.mean([s["r2"] for s in fold_scores])
    print(f"\n  CV Average: MAE={avg_mae:.4f}  R2={avg_r2:.4f}")

    # Train final model on all data
    print("\n  Training final model on all data...")
    final_model = CatBoostRegressor(
        **CATBOOST_PARAMS,
        cat_features=cat_idx if cat_idx else None,
    )
    final_model.fit(X, y, verbose=200)

    # Feature importance
    importance = final_model.get_feature_importance()
    importance_df = pd.DataFrame({
        "feature": available,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    print("\n  TOP 15 FEATURES:")
    for _, row in importance_df.head(15).iterrows():
        bar = "#" * int(row["importance"] / importance_df["importance"].max() * 30)
        print(f"    {row['feature']:<25} {row['importance']:>7.2f}  {bar}")

    return final_model, importance_df, fold_scores


def train_all_models(cascade_data_path: Path = TRAINING_DATA_PATH):
    """Train all surrogate models and save them."""

    print("\n" + "#" * 70)
    print("# AUGER CATBOOST SURROGATE TRAINING")
    print("#" * 70)

    # Load data
    if not cascade_data_path.exists():
        print(f"No training data found at {cascade_data_path}")
        print("Run with --generate first.")
        sys.exit(1)

    df = pd.read_csv(cascade_data_path)
    print(f"\nLoaded {len(df)} cascade samples, {len(df.columns)} columns")

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_importance = {}
    all_scores = {}

    # --- CASCADE-LEVEL MODELS ---
    for target in CASCADE_TARGETS:
        if target not in df.columns:
            print(f"  Skipping {target} (not in data)")
            continue

        model, importance, scores = train_cascade_model(df, target)

        # Save model
        model_path = MODEL_DIR / f"cascade_{target.replace('target_', '')}_model.cbm"
        model.save_model(str(model_path))
        print(f"\n  Model saved: {model_path}")

        all_importance[target] = importance.to_dict("records")
        all_scores[target] = {
            "avg_mae": float(np.mean([s["mae"] for s in scores])),
            "avg_rmse": float(np.mean([s["rmse"] for s in scores])),
            "avg_r2": float(np.mean([s["r2"] for s in scores])),
        }

    # --- RUN-LEVEL MODELS ---
    run_data_path = cascade_data_path.parent / "auger_surrogate_run_data.csv"
    if run_data_path.exists():
        run_df = pd.read_csv(run_data_path)
        print(f"\nLoaded {len(run_df)} run samples for run-level models")

        for target in ["target_cell_kill", "target_therapeutic_ratio", "target_dsb_per_decay"]:
            if target not in run_df.columns or len(run_df) < 10:
                print(f"  Skipping {target} (insufficient data: {len(run_df)} rows)")
                continue

            # Run-level features
            run_feature_cols = [c for c in run_df.columns
                                if c not in ["target_cell_kill", "target_therapeutic_ratio",
                                             "target_dsb_per_decay", "nuclide"]]
            cat_idx_run = []

            X_run = run_df[run_feature_cols].copy()
            y_run = run_df[target].values.astype(float)

            print(f"\n  RUN MODEL: {target}")
            print(f"    Features: {len(run_feature_cols)}, Samples: {len(X_run)}")
            print(f"    y range: [{y_run.min():.4f}, {y_run.max():.4f}]")

            if len(X_run) < 10:
                print("    Too few samples for CV, training on all data...")
                model = CatBoostRegressor(
                    iterations=500, learning_rate=0.05, depth=6,
                    loss_function="RMSE", random_seed=42, verbose=100,
                )
                model.fit(X_run, y_run, verbose=100)
            else:
                n_splits = min(5, len(X_run))
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

                for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_run)):
                    X_t, X_v = X_run.iloc[train_idx], X_run.iloc[val_idx]
                    y_t, y_v = y_run[train_idx], y_run[val_idx]

                    fold_model = CatBoostRegressor(
                        iterations=1000, learning_rate=0.05, depth=6,
                        loss_function="RMSE", random_seed=42, verbose=0,
                    )
                    fold_model.fit(X_t, y_t, eval_set=(X_v, y_v), verbose=0)
                    y_p = fold_model.predict(X_v)
                    mae = mean_absolute_error(y_v, y_p)
                    r2 = r2_score(y_v, y_p) if len(y_v) > 1 else 0.0
                    print(f"    Fold {fold_i+1}: MAE={mae:.4f} R2={r2:.4f}")

                # Final model on all data
                model = CatBoostRegressor(
                    iterations=1000, learning_rate=0.05, depth=6,
                    loss_function="RMSE", random_seed=42, verbose=100,
                )
                model.fit(X_run, y_run, verbose=100)

            model_path = MODEL_DIR / f"run_{target.replace('target_', '')}_model.cbm"
            model.save_model(str(model_path))
            print(f"    Saved: {model_path}")

    # Save feature importance + scores
    summary = {"scores": all_scores, "importance": all_importance}
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Models saved to: {MODEL_DIR}")
    print(f"  Feature importance: {FEATURE_IMPORTANCE_PATH}")
    print()

    # Print summary table
    print(f"  {'Target':<30} {'MAE':>10} {'R2':>10}")
    print(f"  {'-'*50}")
    for target, sc in all_scores.items():
        print(f"  {target:<30} {sc['avg_mae']:>10.4f} {sc['avg_r2']:>10.4f}")

    return all_scores


# ============================================================================
# INFERENCE (INSTANT PREDICTION)
# ============================================================================

class AugerSurrogate:
    """
    Fast ML surrogate for Auger cascade predictions.

    Replaces: run_auger_treatment() (seconds) with instant CatBoost inference (ms).

    Usage:
        surrogate = AugerSurrogate()
        result = surrogate.predict("I-125", n_decays=100)
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self._load_models()
        self.encoder = ShellQuantumEncoder(n_qubits=10, n_shots=1024)

    def _load_models(self):
        for name in ["dsb", "ssb", "lethal", "total_damage"]:
            path = self.model_dir / f"cascade_{name}_model.cbm"
            if path.exists():
                model = CatBoostRegressor()
                model.load_model(str(path))
                self.models[f"cascade_{name}"] = model
                log.info("Loaded cascade model: %s", name)

        for name in ["cell_kill", "therapeutic_ratio", "dsb_per_decay"]:
            path = self.model_dir / f"run_{name}_model.cbm"
            if path.exists():
                model = CatBoostRegressor()
                model.load_model(str(path))
                self.models[f"run_{name}"] = model
                log.info("Loaded run model: %s", name)

    def predict_cascade(self, cascade: DecayCascade,
                         nuclide: str,
                         repair: Optional[RepairConfig] = None) -> Dict[str, float]:
        """Predict damage from a single cascade (instant)."""
        if repair is None:
            repair = RepairConfig(p53_functional=False, brca_functional=False)

        features = extract_cascade_features(cascade, nuclide)
        features["p53_functional"] = int(repair.p53_functional)
        features["brca_functional"] = int(repair.brca_functional)

        feature_cols = CASCADE_FEATURE_COLS + CASCADE_CAT_FEATURES
        available = [c for c in feature_cols if c in features]
        row = pd.DataFrame([{c: features[c] for c in available}])

        predictions = {}
        for name in ["dsb", "ssb", "lethal", "total_damage"]:
            model = self.models.get(f"cascade_{name}")
            if model:
                pred = float(model.predict(row)[0])
                predictions[name] = max(0.0, pred)

        return predictions

    def predict(self, nuclide: str = "I-125",
                n_decays: int = 100,
                p53: bool = False,
                brca: bool = False) -> Dict[str, Any]:
        """
        Predict full treatment outcome using ML surrogate.

        This is the FAST path:
          1. Run Phase 1 cascade (still Monte Carlo -- fast, <1s)
          2. Run Phase 3 quantum encoding (fast, classical sim)
          3. Use CatBoost to predict damage instead of Phase 6+7

        Speedup: ~10-50x vs full simulation (skips damage mapping + repair).
        """
        t_start = time.time()
        shell_energies = SHELL_ENERGIES_BY_NUCLIDE.get(nuclide, {})
        repair = RepairConfig(p53_functional=p53, brca_functional=brca)

        # Phase 1: Cascade (still Monte Carlo -- this is fast)
        cascades = cascade_branching(nuclide, n_decays)

        # Phase 3: Quantum encoding
        cascades = warburg_quantum_batch(cascades, shell_energies)

        # Phase 4: Angiogenesis
        cascades = angiogenesis(cascades)

        # ML SURROGATE: predict per-cascade damage
        cascade_predictions = []
        total_dsb = 0.0
        total_ssb = 0.0
        total_lethal = 0.0

        for cascade in cascades:
            pred = self.predict_cascade(cascade, nuclide, repair)
            cascade_predictions.append(pred)
            total_dsb += pred.get("dsb", 0)
            total_ssb += pred.get("ssb", 0)
            total_lethal += pred.get("lethal", 0)

        # Aggregate predictions
        dsb_per_decay = total_dsb / max(1, n_decays)

        # Cell kill from linear-quadratic model
        alpha, beta = 0.35, 0.035
        equiv_dose = total_lethal * 0.5
        cell_survival = math.exp(-alpha * equiv_dose - beta * equiv_dose ** 2)
        cell_kill = 1.0 - cell_survival

        # Therapeutic ratio estimate (cancer vs healthy)
        # Healthy tissue has full repair -> ~70% of DSBs repaired
        healthy_lethal = total_lethal * 0.30  # HR repairs 70%
        cancer_lethal = total_lethal
        therapeutic_ratio = cancer_lethal / max(1, healthy_lethal)

        duration = time.time() - t_start

        result = {
            "nuclide": nuclide,
            "n_decays": n_decays,
            "method": "catboost_surrogate",
            "predicted_dsb": round(total_dsb, 1),
            "predicted_ssb": round(total_ssb, 1),
            "dsb_per_decay": round(dsb_per_decay, 3),
            "predicted_lethal": round(total_lethal, 1),
            "cell_kill": round(cell_kill, 6),
            "cell_survival": round(cell_survival, 6),
            "therapeutic_ratio": round(therapeutic_ratio, 1),
            "n_cascades": len(cascades),
            "avg_electrons": round(np.mean([c.total_electrons for c in cascades]), 1),
            "duration_sec": round(duration, 3),
        }

        return result

    def benchmark(self, nuclide: str = "I-125", n_decays: int = 50) -> Dict:
        """
        Run both full simulation and surrogate, compare results and speed.
        """
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Full Simulation vs CatBoost Surrogate")
        print(f"{'='*70}")
        print(f"  Nuclide: {nuclide}, Decays: {n_decays}")
        print()

        # Full simulation
        print("  [1/2] Running FULL simulation...")
        t1 = time.time()
        full = run_auger_treatment(nuclide, n_decays, save_to_db=False)
        t_full = time.time() - t1

        # Surrogate prediction
        print("  [2/2] Running SURROGATE prediction...")
        t2 = time.time()
        surr = self.predict(nuclide, n_decays)
        t_surr = time.time() - t2

        speedup = t_full / max(0.001, t_surr)

        print(f"\n  {'Metric':<25} {'Full Sim':>12} {'Surrogate':>12} {'Error':>10}")
        print(f"  {'-'*60}")

        comparisons = [
            ("DSBs", full["total_dsb"], surr["predicted_dsb"]),
            ("SSBs", full["total_ssb"], surr["predicted_ssb"]),
            ("DSB/decay", full["dsb_per_decay"], surr["dsb_per_decay"]),
            ("Cell kill", full["cell_kill"], surr["cell_kill"]),
            ("Therapeutic ratio", full["therapeutic_ratio"], surr["therapeutic_ratio"]),
        ]

        for label, v_full, v_surr in comparisons:
            if v_full != 0:
                err = abs(v_surr - v_full) / (abs(v_full) + 1e-10) * 100
            else:
                err = 0.0
            print(f"  {label:<25} {v_full:>12.3f} {v_surr:>12.3f} {err:>9.1f}%")

        print(f"\n  TIME:      Full={t_full:.2f}s  Surrogate={t_surr:.3f}s  "
              f"Speedup={speedup:.1f}x")
        print(f"{'='*70}")

        return {
            "full": full,
            "surrogate": surr,
            "speedup": speedup,
            "time_full": t_full,
            "time_surrogate": t_surr,
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Auger CatBoost Surrogate - Train ML to replace Monte Carlo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auger_catboost_surrogate.py --generate 2000
  python auger_catboost_surrogate.py --train
  python auger_catboost_surrogate.py --predict I-125 --decays 100
  python auger_catboost_surrogate.py --benchmark
        """,
    )
    parser.add_argument("--generate", type=int, metavar="N",
                        help="Generate N training samples from simulation")
    parser.add_argument("--train", action="store_true",
                        help="Train CatBoost surrogate models")
    parser.add_argument("--predict", type=str, metavar="NUCLIDE",
                        choices=["I-125", "In-111", "Tl-201"],
                        help="Run surrogate prediction for a nuclide")
    parser.add_argument("--decays", type=int, default=100,
                        help="Number of decays for prediction (default: 100)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark surrogate vs full simulation")
    parser.add_argument("--p53", choices=["functional", "mutant"],
                        default="mutant")
    parser.add_argument("--brca", choices=["functional", "mutant"],
                        default="mutant")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if args.generate:
        generate_training_data(n_samples=args.generate)

    elif args.train:
        train_all_models()

    elif args.predict:
        surrogate = AugerSurrogate()
        result = surrogate.predict(
            args.predict, args.decays,
            p53=(args.p53 == "functional"),
            brca=(args.brca == "functional"),
        )
        print(f"\n{'='*70}")
        print("SURROGATE PREDICTION")
        print(f"{'='*70}")
        for k, v in result.items():
            print(f"  {k:<25} {v}")
        print(f"{'='*70}")

    elif args.benchmark:
        surrogate = AugerSurrogate()
        surrogate.benchmark("I-125", n_decays=args.decays)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
