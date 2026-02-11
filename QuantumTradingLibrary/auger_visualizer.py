"""
AUGER VISUALIZER - Cascade Tree + Electron Spectrum + Damage Map
================================================================
Visualization suite for Auger electron cancer treatment simulation.
Renders cascade branching trees, electron energy spectra, DNA damage
maps, and comparative radionuclide analysis.

Run with GPU venv:
    .venv312_gpu\\Scripts\\python.exe auger_visualizer.py
    .venv312_gpu\\Scripts\\python.exe auger_visualizer.py --from-db <run_id>
    .venv312_gpu\\Scripts\\python.exe auger_visualizer.py --live --decays 50

Authors: DooDoo + Claude
Date:    2026-02-11
Version: AUGER-VIZ-1.0
"""

import argparse
import logging
import math
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib import colormaps

from auger_cascade_core import (
    run_auger_treatment, compare_radionuclides,
    DecayCascade, DamageSite, AugerTransition, AugerDatabase,
    SHELL_ORDER, SHELL_ENERGIES_TE, SHELL_ENERGIES_CD, SHELL_ENERGIES_HG,
    SHELL_ENERGIES_BY_NUCLIDE, SHELL_MAX_OCCUPANCY,
    FLUORESCENCE_YIELDS, DEA_RESONANCES, DB_PATH,
)

log = logging.getLogger("AUGER-VIZ")

OUTPUT_DIR = Path(__file__).parent / "auger_viz"

# Color schemes
SHELL_COLORS = {
    "K": "#FF1744",   # Red - innermost
    "L": "#FF9100",   # Orange
    "M": "#FFEA00",   # Yellow
    "N": "#00E676",   # Green
    "O": "#00B0FF",   # Blue - outermost
}
DAMAGE_COLORS = {
    "SSB": "#FFA726",
    "DSB": "#EF5350",
    "base_lesion": "#66BB6A",
    "abasic": "#AB47BC",
}
MECHANISM_COLORS = {
    "direct_ionization": "#42A5F5",
    "DEA": "#EF5350",
    "coulomb": "#FF7043",
    "radical": "#66BB6A",
}


def _shell_color(shell_name: str) -> str:
    """Get color for a shell based on its principal quantum number."""
    return SHELL_COLORS.get(shell_name[0], "#9E9E9E")


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. CASCADE TREE VISUALIZATION
# ============================================================================

def plot_cascade_tree(cascade: DecayCascade,
                      shell_energies: Dict[str, float],
                      save_path: Optional[Path] = None,
                      title_suffix: str = "") -> Path:
    """
    Render a single Auger cascade as a branching tree diagram.

    Vertical axis = shell level (K at top, O at bottom)
    Horizontal branching = vacancy multiplication
    Node color = shell type (K=red, L=orange, M=yellow, N=green, O=blue)
    Edge label = emitted electron energy
    Edge width = proportional to log(energy)
    """
    _ensure_output_dir()

    fig, ax = plt.subplots(1, 1, figsize=(16, 10), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    # Build tree structure from transitions
    # Each transition: vacancy -> (filler_vacancy, ejected_vacancy) + emitted electron
    nodes = []   # (x, y, shell_name, label)
    edges = []   # (x1, y1, x2, y2, energy, color)
    electrons = []  # (x, y, energy, shell)

    # Y-axis: shell depth (K=0, O=16)
    shell_y = {s: i for i, s in enumerate(SHELL_ORDER)}

    # Track positions with BFS
    # Start from initial vacancy
    queue = [(cascade.initial_vacancy, 0, 0.0)]  # (shell, depth, x_pos)
    x_counter = [0.0]  # mutable counter for x spreading
    node_positions = {}
    visited_transitions = set()

    # First pass: assign positions
    level_counts = defaultdict(int)
    for trans in cascade.transitions:
        level_counts[trans.vacancy_shell] += 1

    # Layout the tree
    root_y = shell_y.get(cascade.initial_vacancy, 0)
    nodes.append((0.0, root_y, cascade.initial_vacancy,
                  f"{cascade.initial_vacancy}\nvacancy"))

    # Process transitions in order
    x_spread = 2.0
    current_x_by_level = defaultdict(float)
    branch_x = {}
    branch_x[cascade.initial_vacancy + "_0"] = 0.0

    for t_idx, trans in enumerate(cascade.transitions):
        vy = shell_y.get(trans.vacancy_shell, 0)
        fy = shell_y.get(trans.filler_shell, vy + 1)
        ey = shell_y.get(trans.ejected_shell, vy + 2)

        # Get parent x position
        parent_key = trans.vacancy_shell + f"_{t_idx}"
        parent_x = branch_x.get(parent_key, current_x_by_level[vy])

        # Spread children
        filler_x = parent_x - x_spread / (t_idx + 2)
        ejected_x = parent_x + x_spread / (t_idx + 2)

        # Store child positions
        branch_key_f = trans.filler_shell + f"_{t_idx}_f"
        branch_key_e = trans.ejected_shell + f"_{t_idx}_e"
        branch_x[branch_key_f] = filler_x
        branch_x[branch_key_e] = ejected_x

        # Filler vacancy node
        nodes.append((filler_x, fy, trans.filler_shell,
                       f"{trans.filler_shell}\nfills {trans.vacancy_shell}"))

        # Ejected electron node
        nodes.append((ejected_x, ey, trans.ejected_shell,
                       f"{trans.ejected_shell}\n{trans.energy_eV:.0f} eV"))

        # Edge: vacancy -> filler (electron moves up to fill hole)
        edges.append((parent_x, vy, filler_x, fy,
                       0, _shell_color(trans.filler_shell), "dashed"))

        # Edge: vacancy -> ejected (Auger electron emitted)
        e_width = max(0.5, min(4.0, math.log10(max(1, trans.energy_eV))))
        edges.append((parent_x, vy, ejected_x, ey,
                       trans.energy_eV, _shell_color(trans.ejected_shell), "solid"))

        # Electron emission marker
        electrons.append((ejected_x, ey, trans.energy_eV, trans.ejected_shell))

        current_x_by_level[fy] = filler_x
        current_x_by_level[ey] = ejected_x

    # Draw shell level bands
    for shell_name, y_val in shell_y.items():
        color = _shell_color(shell_name)
        ax.axhline(y=y_val, color=color, alpha=0.08, linewidth=20, zorder=0)
        ax.text(-10, y_val, shell_name, fontsize=8, color=color,
                alpha=0.5, va='center', ha='right', fontfamily='monospace')

    # Draw edges
    for x1, y1, x2, y2, energy, color, style in edges:
        ls = '--' if style == "dashed" else '-'
        lw = max(0.5, min(3.5, math.log10(max(1, energy)) * 0.8)) if energy > 0 else 0.8
        alpha = 0.9 if style == "solid" else 0.4
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
                alpha=alpha, linestyle=ls, zorder=1)
        if energy > 0 and style == "solid":
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, f"{energy:.0f}",
                    fontsize=6, color=color, alpha=0.7,
                    ha='center', va='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='#0D1117', edgecolor=color, alpha=0.5))

    # Draw nodes
    for x, y, shell, label in nodes:
        color = _shell_color(shell)
        circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8, zorder=3)
        ax.add_patch(circle)
        # Smaller inner circle
        inner = plt.Circle((x, y), 0.15, color='#0D1117', alpha=0.6, zorder=4)
        ax.add_patch(inner)

    # Draw electron emission markers (star/burst)
    for x, y, energy, shell in electrons:
        color = _shell_color(shell)
        size = max(30, min(200, energy / 50))
        ax.scatter(x, y, marker='*', s=size, c=color,
                   edgecolors='white', linewidths=0.5, zorder=5, alpha=0.9)

    # Title and labels
    title = (f"Auger Cascade Tree: {cascade.radionuclide} Decay"
             f" | {cascade.total_electrons} electrons"
             f" | {cascade.total_energy_eV:.0f} eV total"
             f" | charge +{cascade.final_charge_state}")
    if title_suffix:
        title += f" | {title_suffix}"

    ax.set_title(title, fontsize=13, color='white', pad=15, fontfamily='monospace')
    ax.set_ylabel("Electron Shell (inner -> outer)",
                  fontsize=10, color='#8B949E', fontfamily='monospace')
    ax.set_xlabel("Cascade Branching", fontsize=10, color='#8B949E',
                  fontfamily='monospace')

    # Invert Y so K is at top
    ax.invert_yaxis()
    ax.set_yticks(range(len(SHELL_ORDER)))
    ax.set_yticklabels(SHELL_ORDER, fontsize=7, color='#8B949E',
                       fontfamily='monospace')
    ax.tick_params(axis='x', colors='#8B949E', labelsize=7)

    # Legend
    legend_patches = [
        mpatches.Patch(color=SHELL_COLORS["K"], label="K shell (innermost)"),
        mpatches.Patch(color=SHELL_COLORS["L"], label="L shells"),
        mpatches.Patch(color=SHELL_COLORS["M"], label="M shells"),
        mpatches.Patch(color=SHELL_COLORS["N"], label="N shells"),
        mpatches.Patch(color=SHELL_COLORS["O"], label="O shells (outermost)"),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=7,
              facecolor='#161B22', edgecolor='#30363D', labelcolor='#8B949E')

    ax.set_xlim(-12, 12)

    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / f"cascade_tree_{cascade.cascade_id}.png"
    fig.savefig(save_path, dpi=150, facecolor='#0D1117',
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_cascade_tree_grid(cascades: List[DecayCascade],
                           shell_energies: Dict[str, float],
                           n_show: int = 6,
                           save_path: Optional[Path] = None) -> Path:
    """
    Plot a grid of cascade trees showing variety in cascade patterns.
    Picks the most diverse cascades by electron count.
    """
    _ensure_output_dir()

    # Select diverse cascades
    sorted_c = sorted(cascades, key=lambda c: c.total_electrons)
    step = max(1, len(sorted_c) // n_show)
    selected = sorted_c[::step][:n_show]

    n_cols = min(3, len(selected))
    n_rows = math.ceil(len(selected) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             facecolor='#0D1117')
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, cascade in enumerate(selected):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.set_facecolor('#0D1117')

        shell_y = {s: i for i, s in enumerate(SHELL_ORDER)}

        # Draw shell bands
        for s, y in shell_y.items():
            ax.axhline(y=y, color=_shell_color(s), alpha=0.06, linewidth=12)

        # Plot transitions as connected points
        for t_idx, trans in enumerate(cascade.transitions):
            vy = shell_y.get(trans.vacancy_shell, 0)
            ey = shell_y.get(trans.ejected_shell, vy + 2)
            x_offset = (t_idx - len(cascade.transitions) / 2) * 0.8

            color = _shell_color(trans.ejected_shell)
            lw = max(0.5, min(3, math.log10(max(1, trans.energy_eV)) * 0.7))
            ax.plot([x_offset, x_offset + 0.3], [vy, ey],
                    color=color, linewidth=lw, alpha=0.8)
            ax.scatter(x_offset + 0.3, ey, marker='*',
                       s=max(20, trans.energy_eV / 100),
                       c=color, edgecolors='white', linewidths=0.3, zorder=5)

        ax.invert_yaxis()
        ax.set_title(f"{cascade.total_electrons}e- | "
                     f"{cascade.total_energy_eV:.0f}eV | "
                     f"+{cascade.final_charge_state}",
                     fontsize=9, color='white', fontfamily='monospace')
        ax.set_yticks(range(0, len(SHELL_ORDER), 3))
        ax.set_yticklabels([SHELL_ORDER[i] for i in range(0, len(SHELL_ORDER), 3)],
                           fontsize=6, color='#8B949E', fontfamily='monospace')
        ax.tick_params(axis='x', labelsize=5, colors='#8B949E')

    # Hide unused axes
    for idx in range(len(selected), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Auger Cascade Diversity ({cascades[0].radionuclide}, "
                 f"{len(cascades)} decays)",
                 fontsize=14, color='white', fontfamily='monospace', y=1.02)

    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / f"cascade_grid_{cascades[0].radionuclide}.png"
    fig.savefig(save_path, dpi=150, facecolor='#0D1117',
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return save_path


# ============================================================================
# 2. ELECTRON ENERGY SPECTRUM
# ============================================================================

def plot_electron_spectrum(cascades: List[DecayCascade],
                           save_path: Optional[Path] = None,
                           log_scale: bool = True) -> Path:
    """
    Plot the electron energy spectrum as a histogram with DEA resonance
    overlay and shell-origin color coding.
    """
    _ensure_output_dir()

    all_energies = []
    all_shells = []
    for c in cascades:
        all_energies.extend(c.electron_energies)
        all_shells.extend(c.electron_shells)

    if not all_energies:
        log.warning("No electron energies to plot")
        return OUTPUT_DIR / "empty.png"

    energies = np.array(all_energies)
    nuclide = cascades[0].radionuclide

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0D1117')
    for ax in axes.flat:
        ax.set_facecolor('#0D1117')
        ax.tick_params(colors='#8B949E', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#30363D')

    # --- Panel 1: Full spectrum histogram ---
    ax1 = axes[0][0]
    bins = np.logspace(np.log10(max(0.1, energies.min())),
                       np.log10(energies.max() + 1), 60) if log_scale else 60
    counts, bin_edges, patches = ax1.hist(energies, bins=bins,
                                           color='#58A6FF', alpha=0.7,
                                           edgecolor='#1F6FEB', linewidth=0.5)

    # Color patches by energy range
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge < 20:
            patch.set_facecolor('#EF5350')   # DEA range - red
            patch.set_alpha(0.9)
        elif left_edge < 100:
            patch.set_facecolor('#FFA726')   # Medium - orange
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor('#42A5F5')   # High - blue
            patch.set_alpha(0.7)

    if log_scale:
        ax1.set_xscale('log')

    # Mark DEA resonance positions
    for res in DEA_RESONANCES:
        ax1.axvline(x=res["energy_eV"], color='#FF1744', linewidth=1.2,
                    linestyle='--', alpha=0.6)
        ax1.text(res["energy_eV"], ax1.get_ylim()[1] * 0.85,
                 f'{res["name"].split("_")[0]}\n{res["energy_eV"]:.1f}eV',
                 fontsize=5, color='#FF1744', rotation=90,
                 ha='right', va='top', fontfamily='monospace')

    # DEA effective range shading
    ax1.axvspan(0.1, 20, alpha=0.08, color='#FF1744', label='DEA range (0-20 eV)')

    ax1.set_xlabel("Electron Energy (eV)", fontsize=10, color='#8B949E',
                   fontfamily='monospace')
    ax1.set_ylabel("Count", fontsize=10, color='#8B949E', fontfamily='monospace')
    ax1.set_title("Auger Electron Energy Spectrum", fontsize=11,
                  color='white', fontfamily='monospace')
    ax1.legend(fontsize=7, facecolor='#161B22', edgecolor='#30363D',
               labelcolor='#8B949E')

    # --- Panel 2: Low-energy detail (DEA range) ---
    ax2 = axes[0][1]
    low_E = energies[energies < 25]
    if len(low_E) > 0:
        ax2.hist(low_E, bins=50, color='#EF5350', alpha=0.8,
                 edgecolor='#C62828', linewidth=0.5)

        # Mark each DEA resonance with band
        res_colors = ['#FF1744', '#FF6D00', '#FFD600', '#00E676', '#2979FF']
        for i, res in enumerate(DEA_RESONANCES):
            width = 1.5  # Resonance width ~ 1.5 eV
            ax2.axvspan(res["energy_eV"] - width / 2,
                        res["energy_eV"] + width / 2,
                        alpha=0.15, color=res_colors[i % len(res_colors)])
            ax2.axvline(x=res["energy_eV"], color=res_colors[i % len(res_colors)],
                        linewidth=1.5, linestyle='-', alpha=0.8)
            ax2.text(res["energy_eV"], ax2.get_ylim()[1] * 0.9 - i * ax2.get_ylim()[1] * 0.08,
                     f'{res["name"]}\n({res["energy_eV"]:.2f} eV, '
                     f'sigma={res["sigma"]})',
                     fontsize=5.5, color=res_colors[i % len(res_colors)],
                     ha='left', va='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#0D1117',
                               edgecolor=res_colors[i % len(res_colors)], alpha=0.7))

    ax2.set_xlabel("Electron Energy (eV)", fontsize=10, color='#8B949E',
                   fontfamily='monospace')
    ax2.set_ylabel("Count", fontsize=10, color='#8B949E', fontfamily='monospace')
    ax2.set_title("Low-Energy Detail (DEA Resonances)", fontsize=11,
                  color='white', fontfamily='monospace')
    ax2.set_xlim(-0.5, 25)

    # --- Panel 3: Shell origin breakdown ---
    ax3 = axes[1][0]
    shell_groups = defaultdict(list)
    for e, s in zip(all_energies, all_shells):
        shell_groups[s[0]].append(e)  # Group by principal shell letter

    labels_ordered = ["K", "L", "M", "N", "O"]
    bottom = np.zeros(50)
    bin_edges_linear = np.linspace(0, min(5000, energies.max()), 51)

    for shell_letter in labels_ordered:
        shell_e = shell_groups.get(shell_letter, [])
        if not shell_e:
            continue
        hist_vals, _ = np.histogram(shell_e, bins=bin_edges_linear)
        color = SHELL_COLORS.get(shell_letter, '#9E9E9E')
        ax3.bar(bin_edges_linear[:-1], hist_vals, width=np.diff(bin_edges_linear),
                bottom=bottom, color=color, alpha=0.8,
                label=f"{shell_letter} shell ({len(shell_e)})")
        bottom += hist_vals

    ax3.set_xlabel("Electron Energy (eV)", fontsize=10, color='#8B949E',
                   fontfamily='monospace')
    ax3.set_ylabel("Count", fontsize=10, color='#8B949E', fontfamily='monospace')
    ax3.set_title("Spectrum by Shell Origin", fontsize=11,
                  color='white', fontfamily='monospace')
    ax3.legend(fontsize=7, facecolor='#161B22', edgecolor='#30363D',
               labelcolor='#8B949E')

    # --- Panel 4: Cumulative energy distribution ---
    ax4 = axes[1][1]
    sorted_E = np.sort(energies)
    cumulative = np.cumsum(sorted_E) / sum(sorted_E) * 100
    ax4.plot(sorted_E, cumulative, color='#58A6FF', linewidth=2, alpha=0.9)
    ax4.fill_between(sorted_E, cumulative, alpha=0.1, color='#58A6FF')

    # Mark key percentiles
    for pct in [25, 50, 75, 90]:
        idx = np.searchsorted(cumulative, pct)
        if idx < len(sorted_E):
            ax4.axhline(y=pct, color='#30363D', linewidth=0.5, linestyle='--')
            ax4.axvline(x=sorted_E[idx], color='#FFA726', linewidth=0.8,
                        linestyle='--', alpha=0.5)
            ax4.text(sorted_E[idx], pct + 2,
                     f'{pct}%: {sorted_E[idx]:.0f} eV',
                     fontsize=7, color='#FFA726', fontfamily='monospace')

    ax4.set_xscale('log')
    ax4.set_xlabel("Electron Energy (eV)", fontsize=10, color='#8B949E',
                   fontfamily='monospace')
    ax4.set_ylabel("Cumulative Energy (%)", fontsize=10, color='#8B949E',
                   fontfamily='monospace')
    ax4.set_title("Cumulative Energy Distribution", fontsize=11,
                  color='white', fontfamily='monospace')

    # Suptitle
    n_dea = sum(1 for e in energies if e < 20)
    fig.suptitle(
        f"{nuclide} Electron Spectrum | "
        f"{len(energies)} electrons from {len(cascades)} decays | "
        f"Mean={np.mean(energies):.0f} eV | Median={np.median(energies):.0f} eV | "
        f"DEA range: {n_dea} ({n_dea/len(energies)*100:.0f}%)",
        fontsize=12, color='white', fontfamily='monospace', y=1.02
    )

    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / f"electron_spectrum_{nuclide}.png"
    fig.savefig(save_path, dpi=150, facecolor='#0D1117',
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return save_path


# ============================================================================
# 3. DNA DAMAGE MAP
# ============================================================================

def plot_damage_map(damage_sites: List[DamageSite],
                    dna_length_bp: int = 10000,
                    save_path: Optional[Path] = None) -> Path:
    """
    Plot DNA damage along the double helix.
    Top strand = sense, bottom strand = antisense.
    Color = damage type. Size = electron energy.
    Vertical lines connect opposing-strand SSBs that form DSBs.
    """
    _ensure_output_dir()

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), facecolor='#0D1117',
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    for ax in axes:
        ax.set_facecolor('#0D1117')
        ax.tick_params(colors='#8B949E', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#30363D')

    # --- Panel 1: Damage map along DNA ---
    ax1 = axes[0]

    # Draw DNA backbone
    ax1.axhline(y=0.5, color='#58A6FF', linewidth=2, alpha=0.3, label='sense strand')
    ax1.axhline(y=-0.5, color='#FF6D00', linewidth=2, alpha=0.3, label='antisense strand')

    # Draw base pair ticks
    tick_spacing = max(1, dna_length_bp // 200)
    for bp in range(0, dna_length_bp, tick_spacing):
        ax1.plot([bp, bp], [-0.15, 0.15], color='#30363D', linewidth=0.3, alpha=0.3)

    # Plot damage sites
    dsb_pairs = []
    for site in damage_sites:
        y = 0.5 if site.strand == "sense" else -0.5
        color = DAMAGE_COLORS.get(site.damage_type, '#9E9E9E')
        size = max(15, min(100, site.electron_energy_eV / 5))
        marker = 'X' if site.damage_type == "DSB" else 'o'
        alpha = 1.0 if site.is_lethal else 0.7

        ax1.scatter(site.position_bp, y, c=color, s=size, marker=marker,
                    alpha=alpha, edgecolors='white' if site.is_lethal else 'none',
                    linewidths=1.0 if site.is_lethal else 0, zorder=5)

        # If DSB, draw cross-strand connection
        if site.damage_type == "DSB":
            ax1.plot([site.position_bp, site.position_bp], [-0.5, 0.5],
                     color='#EF5350', linewidth=1.5, alpha=0.5, linestyle='-')

    # Cluster highlighting
    positions = [s.position_bp for s in damage_sites]
    if positions:
        pos_arr = np.array(sorted(positions))
        # Find clusters (gaps < 20 bp)
        clusters = []
        cluster_start = pos_arr[0]
        cluster_end = pos_arr[0]
        for p in pos_arr[1:]:
            if p - cluster_end < 20:
                cluster_end = p
            else:
                if cluster_end - cluster_start > 5:
                    clusters.append((cluster_start, cluster_end))
                cluster_start = p
                cluster_end = p
        if cluster_end - cluster_start > 5:
            clusters.append((cluster_start, cluster_end))

        for cs, ce in clusters:
            n_in_cluster = sum(1 for s in damage_sites if cs <= s.position_bp <= ce)
            if n_in_cluster >= 3:
                ax1.axvspan(cs - 2, ce + 2, alpha=0.08, color='#FF1744')
                ax1.text(cs, 1.0, f'{n_in_cluster} hits',
                         fontsize=6, color='#FF1744', fontfamily='monospace')

    ax1.set_xlim(-50, dna_length_bp + 50)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlabel("Base Pair Position", fontsize=10, color='#8B949E',
                   fontfamily='monospace')
    ax1.set_title("DNA Damage Map", fontsize=12, color='white',
                  fontfamily='monospace')

    # Legend
    legend_patches = [
        mpatches.Patch(color=DAMAGE_COLORS["SSB"], label=f'SSB ({sum(1 for s in damage_sites if s.damage_type=="SSB")})'),
        mpatches.Patch(color=DAMAGE_COLORS["DSB"], label=f'DSB ({sum(1 for s in damage_sites if s.damage_type=="DSB")})'),
        mpatches.Patch(color=DAMAGE_COLORS["base_lesion"], label=f'Base Lesion ({sum(1 for s in damage_sites if s.damage_type=="base_lesion")})'),
    ]
    ax1.legend(handles=legend_patches, fontsize=7, facecolor='#161B22',
               edgecolor='#30363D', labelcolor='#8B949E', loc='upper right')

    # --- Panel 2: Damage density ---
    ax2 = axes[1]
    bin_width = max(10, dna_length_bp // 100)
    bins_bp = np.arange(0, dna_length_bp + bin_width, bin_width)

    ssb_pos = [s.position_bp for s in damage_sites if s.damage_type == "SSB"]
    dsb_pos = [s.position_bp for s in damage_sites if s.damage_type == "DSB"]

    if ssb_pos:
        ax2.hist(ssb_pos, bins=bins_bp, color=DAMAGE_COLORS["SSB"],
                 alpha=0.6, label='SSB')
    if dsb_pos:
        ax2.hist(dsb_pos, bins=bins_bp, color=DAMAGE_COLORS["DSB"],
                 alpha=0.8, label='DSB', bottom=np.histogram(ssb_pos, bins=bins_bp)[0] if ssb_pos else 0)

    ax2.set_xlabel("Base Pair Position", fontsize=9, color='#8B949E',
                   fontfamily='monospace')
    ax2.set_ylabel("Damage Count", fontsize=9, color='#8B949E',
                   fontfamily='monospace')
    ax2.set_title("Damage Density Along DNA", fontsize=10, color='white',
                  fontfamily='monospace')
    ax2.legend(fontsize=7, facecolor='#161B22', edgecolor='#30363D',
               labelcolor='#8B949E')

    # --- Panel 3: Mechanism breakdown ---
    ax3 = axes[2]
    mech_counts = Counter(s.mechanism for s in damage_sites)
    if mech_counts:
        labels = list(mech_counts.keys())
        values = list(mech_counts.values())
        colors = [MECHANISM_COLORS.get(m, '#9E9E9E') for m in labels]
        bars = ax3.barh(labels, values, color=colors, alpha=0.8,
                        edgecolor='#30363D')
        for bar, val in zip(bars, values):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     str(val), fontsize=8, color='#8B949E', va='center',
                     fontfamily='monospace')

    ax3.set_xlabel("Count", fontsize=9, color='#8B949E', fontfamily='monospace')
    ax3.set_title("Damage by Mechanism", fontsize=10, color='white',
                  fontfamily='monospace')
    ax3.tick_params(axis='y', labelsize=7, colors='#8B949E')

    n_lethal = sum(1 for s in damage_sites if s.is_lethal)
    fig.suptitle(
        f"DNA Damage Profile | {len(damage_sites)} sites | "
        f"SSB={sum(1 for s in damage_sites if s.damage_type=='SSB')} | "
        f"DSB={sum(1 for s in damage_sites if s.damage_type=='DSB')} | "
        f"Lethal={n_lethal}",
        fontsize=12, color='white', fontfamily='monospace', y=1.02
    )

    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "damage_map.png"
    fig.savefig(save_path, dpi=150, facecolor='#0D1117',
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return save_path


# ============================================================================
# 4. RADIONUCLIDE COMPARISON CHART
# ============================================================================

def plot_comparison(results: Dict[str, Dict],
                    save_path: Optional[Path] = None) -> Path:
    """
    Multi-panel comparison across radionuclides.
    """
    _ensure_output_dir()

    nuclides = list(results.keys())
    n = len(nuclides)
    nuc_colors = {'I-125': '#58A6FF', 'In-111': '#F78166', 'Tl-201': '#7EE787'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#0D1117')
    for ax in axes.flat:
        ax.set_facecolor('#0D1117')
        ax.tick_params(colors='#8B949E', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#30363D')

    # 1. Electrons per decay
    ax = axes[0][0]
    vals = [results[n]["avg_electrons_per_decay"] for n in nuclides]
    colors = [nuc_colors.get(n, '#9E9E9E') for n in nuclides]
    bars = ax.bar(nuclides, vals, color=colors, alpha=0.8, edgecolor='#30363D')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', fontsize=9, color='white',
                fontfamily='monospace')
    ax.set_title("Avg Electrons / Decay", fontsize=11, color='white',
                 fontfamily='monospace')

    # 2. DSB per decay
    ax = axes[0][1]
    vals = [results[n]["dsb_per_decay"] for n in nuclides]
    bars = ax.bar(nuclides, vals, color=colors, alpha=0.8, edgecolor='#30363D')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=9, color='white',
                fontfamily='monospace')
    ax.set_title("DSBs / Decay", fontsize=11, color='white', fontfamily='monospace')

    # 3. Therapeutic ratio
    ax = axes[0][2]
    vals = [results[n]["therapeutic_ratio"] for n in nuclides]
    bars = ax.bar(nuclides, vals, color=colors, alpha=0.8, edgecolor='#30363D')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.1f}x', ha='center', fontsize=9, color='white',
                fontfamily='monospace')
    ax.set_title("Therapeutic Ratio", fontsize=11, color='white',
                 fontfamily='monospace')

    # 4. Energy spectrum overlay
    ax = axes[1][0]
    for nuc in nuclides:
        spectra = results[nuc].get("electron_spectrum", [])
        if spectra:
            energies = [e for e, _ in spectra]
            bins = np.logspace(np.log10(max(0.1, min(energies))),
                               np.log10(max(energies) + 1), 40)
            ax.hist(energies, bins=bins, alpha=0.5,
                    color=nuc_colors.get(nuc, '#9E9E9E'),
                    label=nuc, edgecolor='none')
    ax.set_xscale('log')
    ax.axvspan(0.1, 20, alpha=0.06, color='#FF1744')
    ax.set_title("Energy Spectra Overlay", fontsize=11, color='white',
                 fontfamily='monospace')
    ax.set_xlabel("Energy (eV)", fontsize=9, color='#8B949E', fontfamily='monospace')
    ax.legend(fontsize=8, facecolor='#161B22', edgecolor='#30363D',
              labelcolor='#8B949E')

    # 5. DEA fraction
    ax = axes[1][1]
    dea_fracs = []
    mid_fracs = []
    high_fracs = []
    for nuc in nuclides:
        spectra = results[nuc].get("electron_spectrum", [])
        energies = [e for e, _ in spectra]
        total = len(energies) if energies else 1
        dea_fracs.append(sum(1 for e in energies if e < 20) / total * 100)
        mid_fracs.append(sum(1 for e in energies if 20 <= e < 100) / total * 100)
        high_fracs.append(sum(1 for e in energies if e >= 100) / total * 100)

    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, dea_fracs, w, color='#EF5350', alpha=0.8, label='<20eV (DEA)')
    ax.bar(x, mid_fracs, w, color='#FFA726', alpha=0.8, label='20-100eV')
    ax.bar(x + w, high_fracs, w, color='#42A5F5', alpha=0.8, label='>100eV')
    ax.set_xticks(x)
    ax.set_xticklabels(nuclides, fontsize=9, color='#8B949E')
    ax.set_ylabel("%", fontsize=9, color='#8B949E', fontfamily='monospace')
    ax.set_title("Energy Band Distribution (%)", fontsize=11, color='white',
                 fontfamily='monospace')
    ax.legend(fontsize=7, facecolor='#161B22', edgecolor='#30363D',
              labelcolor='#8B949E')

    # 6. Damage type breakdown
    ax = axes[1][2]
    ssb_vals = [results[n]["total_ssb"] for n in nuclides]
    dsb_vals = [results[n]["total_dsb"] for n in nuclides]
    base_vals = [results[n]["total_base_lesions"] for n in nuclides]

    ax.bar(x - w, ssb_vals, w, color=DAMAGE_COLORS["SSB"], alpha=0.8, label='SSB')
    ax.bar(x, dsb_vals, w, color=DAMAGE_COLORS["DSB"], alpha=0.8, label='DSB')
    ax.bar(x + w, base_vals, w, color=DAMAGE_COLORS["base_lesion"], alpha=0.8,
           label='Base Lesion')
    ax.set_xticks(x)
    ax.set_xticklabels(nuclides, fontsize=9, color='#8B949E')
    ax.set_title("Damage Type Counts", fontsize=11, color='white',
                 fontfamily='monospace')
    ax.legend(fontsize=7, facecolor='#161B22', edgecolor='#30363D',
              labelcolor='#8B949E')

    fig.suptitle(
        "Auger Radionuclide Comparison: I-125 vs In-111 vs Tl-201",
        fontsize=14, color='white', fontfamily='monospace', y=1.02
    )

    plt.tight_layout()
    if save_path is None:
        save_path = OUTPUT_DIR / "radionuclide_comparison.png"
    fig.savefig(save_path, dpi=150, facecolor='#0D1117',
                edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    return save_path


# ============================================================================
# MAIN: Run simulation + generate all visualizations
# ============================================================================

def generate_all_visualizations(radionuclide: str = "I-125",
                                 n_decays: int = 100,
                                 compare: bool = True) -> List[Path]:
    """Run simulation(s) and generate all visualization panels."""
    _ensure_output_dir()
    output_files = []

    print("=" * 70)
    print("AUGER VISUALIZER - Generating all panels")
    print("=" * 70)

    # --- Single nuclide simulation ---
    print(f"\n[1/5] Running {radionuclide} simulation ({n_decays} decays)...")
    results = run_auger_treatment(
        radionuclide=radionuclide,
        n_decays=n_decays,
        save_to_db=True,
    )

    cascades = results["cascades"]
    damage_sites = results["damage_sites"]
    shell_energies = SHELL_ENERGIES_BY_NUCLIDE.get(radionuclide, SHELL_ENERGIES_TE)

    # --- Cascade tree (single best cascade) ---
    print("[2/5] Rendering cascade tree (single)...")
    best_cascade = max(cascades, key=lambda c: c.total_electrons)
    p = plot_cascade_tree(best_cascade, shell_energies)
    output_files.append(p)
    print(f"      -> {p}")

    # --- Cascade tree grid ---
    print("[3/5] Rendering cascade tree grid...")
    p = plot_cascade_tree_grid(cascades, shell_energies)
    output_files.append(p)
    print(f"      -> {p}")

    # --- Electron spectrum ---
    print("[4/5] Rendering electron spectrum...")
    p = plot_electron_spectrum(cascades)
    output_files.append(p)
    print(f"      -> {p}")

    # --- DNA damage map ---
    print("[5/5] Rendering DNA damage map...")
    p = plot_damage_map(damage_sites, dna_length_bp=results["dna_length_bp"])
    output_files.append(p)
    print(f"      -> {p}")

    # --- Comparative (all three nuclides) ---
    if compare:
        print("\n[BONUS] Running comparative analysis...")
        comp_results = {}
        for nuc in ["I-125", "In-111", "Tl-201"]:
            if nuc == radionuclide:
                comp_results[nuc] = results
            else:
                print(f"  Simulating {nuc}...")
                comp_results[nuc] = run_auger_treatment(
                    radionuclide=nuc,
                    n_decays=n_decays,
                    save_to_db=True,
                )

        p = plot_comparison(comp_results)
        output_files.append(p)
        print(f"      -> {p}")

        # Also render spectra for other nuclides
        for nuc in ["In-111", "Tl-201"]:
            if nuc != radionuclide:
                p = plot_electron_spectrum(comp_results[nuc]["cascades"])
                output_files.append(p)

    print("\n" + "=" * 70)
    print(f"VISUALIZATION COMPLETE: {len(output_files)} files generated")
    print(f"Output directory: {OUTPUT_DIR}")
    for f in output_files:
        print(f"  {f.name}")
    print("=" * 70)

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Auger Cascade Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auger_visualizer.py
  python auger_visualizer.py --nuclide I-125 --decays 200
  python auger_visualizer.py --no-compare
  python auger_visualizer.py --show
        """,
    )
    parser.add_argument("--nuclide", default="I-125",
                        choices=["I-125", "In-111", "Tl-201"])
    parser.add_argument("--decays", type=int, default=100)
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip comparative analysis")
    parser.add_argument("--show", action="store_true",
                        help="Open images after generation")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    files = generate_all_visualizations(
        radionuclide=args.nuclide,
        n_decays=args.decays,
        compare=not args.no_compare,
    )

    if args.show:
        import os
        for f in files:
            os.startfile(str(f))


if __name__ == "__main__":
    main()
