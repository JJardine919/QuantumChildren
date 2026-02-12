"""
Quantum Radar — TE Energy Visualization
=========================================
Reads the latest TEQA analytics and domestication DB to generate
an interactive HTML radar showing all 33 TE families.

Usage:
    python quantum_radar.py                  # One-shot: generate HTML
    python quantum_radar.py --live           # Auto-refresh every 60s
    python quantum_radar.py --serve          # Serve on localhost:8050

Authors: DooDoo + Claude
"""

import json
import os
import sqlite3
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

# ── TE Family Definitions (mirrors teqa_v3_neural_te.py) ──

TE_FAMILIES = [
    # Original 25 (Class I: Retrotransposons)
    {"name": "BEL_Pao",       "idx": 0,  "cls": "I",  "signal": "momentum",           "desc": "LTR retrotransposon"},
    {"name": "DIRS1",         "idx": 1,  "cls": "I",  "signal": "trend_strength",      "desc": "Tyrosine recombinase"},
    {"name": "Ty1_copia",     "idx": 2,  "cls": "I",  "signal": "rsi",                 "desc": "LTR Ty1/copia"},
    {"name": "Ty3_gypsy",     "idx": 3,  "cls": "I",  "signal": "macd",                "desc": "LTR Ty3/gypsy"},
    {"name": "Ty5",           "idx": 4,  "cls": "I",  "signal": "bollinger_position",  "desc": "LTR Ty5 (heterochromatin)"},
    {"name": "Alu",           "idx": 5,  "cls": "I",  "signal": "short_volatility",    "desc": "SINE Alu element"},
    {"name": "LINE",          "idx": 6,  "cls": "I",  "signal": "price_change",        "desc": "LINE-1 autonomous"},
    {"name": "Penelope",      "idx": 7,  "cls": "I",  "signal": "trend_duration",      "desc": "Penelope-like element"},
    {"name": "RTE",           "idx": 8,  "cls": "I",  "signal": "mean_reversion",      "desc": "Non-LTR RTE"},
    {"name": "SINE",          "idx": 9,  "cls": "I",  "signal": "tick_volume",          "desc": "Short interspersed element"},
    {"name": "VIPER_Ngaro",   "idx": 10, "cls": "I",  "signal": "atr_ratio",           "desc": "VIPER/Ngaro element"},
    # Original 25 (Class II: DNA Transposons)
    {"name": "CACTA",         "idx": 11, "cls": "II", "signal": "ema_crossover",       "desc": "En/Spm superfamily"},
    {"name": "Crypton",       "idx": 12, "cls": "II", "signal": "compression_ratio",   "desc": "Tyrosine recombinase"},
    {"name": "Helitron",      "idx": 13, "cls": "II", "signal": "volume_profile",      "desc": "Rolling-circle transposon"},
    {"name": "hobo",          "idx": 14, "cls": "II", "signal": "candle_pattern",      "desc": "hAT superfamily"},
    {"name": "I_element",     "idx": 15, "cls": "II", "signal": "support_resistance",  "desc": "I-element (Drosophila)"},
    {"name": "Mariner_Tc1",   "idx": 16, "cls": "II", "signal": "fractal_dim",         "desc": "Tc1/mariner superfamily"},
    {"name": "Mavericks_Polinton", "idx": 17, "cls": "II", "signal": "order_flow",     "desc": "Self-synthesizing"},
    {"name": "Mutator",       "idx": 18, "cls": "II", "signal": "mutation_rate",       "desc": "Mutator/MuDR"},
    {"name": "P_element",     "idx": 19, "cls": "II", "signal": "spread_analysis",     "desc": "P-element (hybrid dysgenesis)"},
    {"name": "PIF_Harbinger", "idx": 20, "cls": "II", "signal": "microstructure",      "desc": "PIF/Harbinger"},
    {"name": "piggyBac",      "idx": 21, "cls": "II", "signal": "gap_analysis",        "desc": "piggyBac (TTAA target)"},
    {"name": "pogo",          "idx": 22, "cls": "II", "signal": "session_overlap",     "desc": "Tc1/pogo"},
    {"name": "Rag_like",      "idx": 23, "cls": "II", "signal": "diversity_index",     "desc": "RAG-like (V(D)J origin)"},
    {"name": "Transib",       "idx": 24, "cls": "II", "signal": "autocorrelation",     "desc": "Transib superfamily"},
    # Neural 8 (v3.0)
    {"name": "L1_Neuronal",   "idx": 25, "cls": "N",  "signal": "pattern_repetition",  "desc": "L1 brain-specific — hippocampal memory"},
    {"name": "L1_Somatic",    "idx": 26, "cls": "N",  "signal": "multi_tf_variance",   "desc": "L1 somatic mosaicism — neural diversity"},
    {"name": "HERV_Synapse",  "idx": 27, "cls": "N",  "signal": "cross_correlation",   "desc": "HERV syncytin — cross-symbol correlation"},
    {"name": "SVA_Regulatory","idx": 28, "cls": "N",  "signal": "compression_breakout", "desc": "SVA composite — regime change detection"},
    {"name": "Alu_Exonization","idx": 29,"cls": "N",  "signal": "noise_pattern",       "desc": "Alu exonization — feature creation from noise"},
    {"name": "TRIM28_Silencer","idx": 30,"cls": "N",  "signal": "drawdown",            "desc": "TRIM28/KAP1 — TE repressor / risk management"},
    {"name": "piwiRNA_Neural","idx": 31, "cls": "N",  "signal": "signal_noise_ratio",  "desc": "Neural piRNA — TE quality control"},
    {"name": "Arc_Capsid",    "idx": 32, "cls": "N",  "signal": "successful_pattern_echo", "desc": "Arc protein (Ty3/gypsy) — inter-neuron transfer"},
]

BASE_DIR = Path(__file__).parent
ANALYTICS_DIR = BASE_DIR / "teqa_analytics"
DOMESTICATION_DB = BASE_DIR / "teqa_domestication.db"
OUTPUT_HTML = BASE_DIR / "quantum_radar.html"


def get_latest_analytics():
    """Read the most recent full_log JSON."""
    logs = sorted(ANALYTICS_DIR.glob("full_log_*.json"))
    if not logs:
        return None
    with open(logs[-1]) as f:
        return json.load(f)


def get_latest_report():
    """Read the most recent text report to extract per-TE activation data."""
    reports = sorted(ANALYTICS_DIR.glob("report_*.txt"))
    if not reports:
        return None
    with open(reports[-1]) as f:
        return f.read()


def get_domestication_data():
    """Query the domestication DB for all tracked patterns."""
    if not DOMESTICATION_DB.exists():
        return {"total": 0, "domesticated": 0, "patterns": []}

    conn = sqlite3.connect(str(DOMESTICATION_DB))
    cursor = conn.cursor()

    total = cursor.execute("SELECT COUNT(*) FROM domesticated_patterns").fetchone()[0]
    dom_count = cursor.execute("SELECT COUNT(*) FROM domesticated_patterns WHERE domesticated=1").fetchone()[0]

    cursor.execute("""
        SELECT pattern_hash, te_combo, win_count, loss_count, posterior_wr,
               domesticated, boost_factor, first_seen, last_seen
        FROM domesticated_patterns
        ORDER BY domesticated DESC, posterior_wr DESC
        LIMIT 50
    """)
    patterns = []
    for row in cursor.fetchall():
        patterns.append({
            "hash": row[0],
            "combo": row[1],
            "wins": row[2],
            "losses": row[3],
            "posterior_wr": round(row[4], 3) if row[4] else 0,
            "domesticated": bool(row[5]),
            "boost": round(row[6], 2) if row[6] else 1.0,
            "first_seen": row[7],
            "last_seen": row[8],
        })
    conn.close()

    # Count per-TE appearances in tracked combos
    te_stats = {}
    for p in patterns:
        for te in p["combo"].split("+"):
            if te not in te_stats:
                te_stats[te] = {"appearances": 0, "wins": 0, "losses": 0}
            te_stats[te]["appearances"] += 1
            te_stats[te]["wins"] += p["wins"]
            te_stats[te]["losses"] += p["losses"]

    return {
        "total": total,
        "domesticated": dom_count,
        "patterns": patterns,
        "te_stats": te_stats,
    }


def generate_html():
    """Generate the Quantum Radar HTML visualization."""
    analytics = get_latest_analytics()
    dom_data = get_domestication_data()

    active_tes = analytics.get("active_tes", []) if analytics else []
    shock_label = analytics.get("shock_label", "UNKNOWN") if analytics else "NO DATA"
    shock_score = analytics.get("shock_score", 0) if analytics else 0
    direction = analytics.get("direction", 0) if analytics else 0
    confidence = analytics.get("confidence", 0) if analytics else 0
    consensus = analytics.get("consensus_score", 0) if analytics else 0
    dom_boost = analytics.get("domestication_boost", 1.0) if analytics else 1.0
    vote_counts = analytics.get("vote_counts", {"long": 0, "short": 0, "neutral": 0}) if analytics else {"long": 0, "short": 0, "neutral": 0}
    gates = analytics.get("gates", {}) if analytics else {}
    timestamp = analytics.get("timestamp", "N/A") if analytics else "N/A"
    symbol = analytics.get("symbol", "N/A") if analytics else "N/A"
    elapsed = analytics.get("elapsed_ms", 0) if analytics else 0
    evo = analytics.get("evolution", {}) if analytics else {}

    # Build TE data for JS
    te_data = []
    for te in TE_FAMILIES:
        is_active = te["name"] in active_tes
        dom_te_stats = dom_data.get("te_stats", {}).get(te["name"], {})
        te_data.append({
            "name": te["name"],
            "idx": te["idx"],
            "cls": te["cls"],
            "signal": te["signal"],
            "desc": te["desc"],
            "active": is_active,
            "appearances": dom_te_stats.get("appearances", 0),
            "wins": dom_te_stats.get("wins", 0),
            "losses": dom_te_stats.get("losses", 0),
        })

    dir_str = "LONG" if direction > 0 else ("SHORT" if direction < 0 else "NEUTRAL")
    dir_color = "#00ff88" if direction > 0 else ("#ff4466" if direction < 0 else "#888888")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quantum Radar — TE Energy Visualization</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    min-height: 100vh;
    overflow-x: hidden;
  }}
  .header {{
    text-align: center;
    padding: 20px;
    border-bottom: 1px solid #1a1a2e;
  }}
  .header h1 {{
    font-size: 28px;
    color: #00ccff;
    letter-spacing: 4px;
    text-transform: uppercase;
  }}
  .header .subtitle {{
    color: #666;
    font-size: 12px;
    margin-top: 4px;
  }}
  .dashboard {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    padding: 16px;
    max-width: 1600px;
    margin: 0 auto;
  }}
  .panel {{
    background: #111118;
    border: 1px solid #1a1a2e;
    border-radius: 8px;
    padding: 16px;
  }}
  .panel h2 {{
    font-size: 13px;
    color: #00ccff;
    letter-spacing: 2px;
    margin-bottom: 12px;
    text-transform: uppercase;
    border-bottom: 1px solid #1a1a2e;
    padding-bottom: 6px;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 13px;
  }}
  .stat-label {{ color: #666; }}
  .stat-value {{ color: #e0e0e0; font-weight: bold; }}
  .stat-value.green {{ color: #00ff88; }}
  .stat-value.red {{ color: #ff4466; }}
  .stat-value.yellow {{ color: #ffcc00; }}
  .stat-value.cyan {{ color: #00ccff; }}

  /* Radar container */
  .radar-container {{
    grid-column: 1 / -1;
    display: flex;
    justify-content: center;
    padding: 20px;
  }}
  .radar-svg {{
    max-width: 900px;
    width: 100%;
  }}

  /* TE Grid */
  .te-grid {{
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 8px;
  }}
  .te-card {{
    background: #0d0d14;
    border: 1px solid #1a1a2e;
    border-radius: 6px;
    padding: 10px;
    font-size: 11px;
    transition: all 0.3s;
  }}
  .te-card.active {{
    border-color: #00ccff;
    box-shadow: 0 0 12px rgba(0, 204, 255, 0.15);
  }}
  .te-card.class-i {{ border-left: 3px solid #ff6644; }}
  .te-card.class-ii {{ border-left: 3px solid #44aaff; }}
  .te-card.class-n {{ border-left: 3px solid #aa44ff; }}
  .te-card .te-name {{
    font-weight: bold;
    font-size: 12px;
    color: #fff;
  }}
  .te-card .te-signal {{ color: #666; font-size: 10px; }}
  .te-card .te-status {{
    margin-top: 4px;
    display: flex;
    gap: 6px;
  }}
  .badge {{
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 9px;
    font-weight: bold;
    text-transform: uppercase;
  }}
  .badge.active {{ background: #002211; color: #00ff88; border: 1px solid #00ff88; }}
  .badge.inactive {{ background: #110000; color: #553333; border: 1px solid #332222; }}
  .badge.class-i {{ background: #1a0800; color: #ff6644; }}
  .badge.class-ii {{ background: #001133; color: #44aaff; }}
  .badge.class-n {{ background: #110033; color: #aa44ff; }}

  /* Domestication panel */
  .dom-table {{
    width: 100%;
    font-size: 11px;
    border-collapse: collapse;
  }}
  .dom-table th {{
    text-align: left;
    color: #666;
    padding: 4px 6px;
    border-bottom: 1px solid #1a1a2e;
  }}
  .dom-table td {{
    padding: 4px 6px;
    border-bottom: 1px solid #0d0d14;
  }}
  .dom-table .domestic {{ color: #00ff88; font-weight: bold; }}
  .dom-table .wild {{ color: #666; }}

  /* Gate checks */
  .gate {{ display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 12px; }}
  .gate-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    display: inline-block;
  }}
  .gate-dot.pass {{ background: #00ff88; box-shadow: 0 0 6px #00ff88; }}
  .gate-dot.fail {{ background: #ff4466; box-shadow: 0 0 6px #ff4466; }}

  .direction-badge {{
    display: inline-block;
    padding: 4px 16px;
    border-radius: 4px;
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 3px;
  }}

  .footer {{
    text-align: center;
    padding: 16px;
    color: #333;
    font-size: 10px;
    border-top: 1px solid #1a1a2e;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Quantum Radar</h1>
  <div class="subtitle">TEQA v3.0 — 33 Transposable Element Families — Neural-TE Integration</div>
  <div class="subtitle" style="margin-top:8px; color: #444;">
    {symbol} | {timestamp[:19]} | {elapsed:.0f}ms
  </div>
</div>

<div class="dashboard">

  <!-- Signal Panel -->
  <div class="panel">
    <h2>Signal Output</h2>
    <div style="text-align:center; margin: 12px 0;">
      <span class="direction-badge" style="background:{dir_color}22; color:{dir_color}; border: 1px solid {dir_color};">
        {dir_str}
      </span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Confidence</span>
      <span class="stat-value cyan">{confidence:.4f}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Concordance</span>
      <span class="stat-value">{analytics.get('concordance', 0) if analytics else 0:.4f}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Amplitude²</span>
      <span class="stat-value">{analytics.get('amplitude_sq', 0) if analytics else 0:.6f}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Domestication Boost</span>
      <span class="stat-value {'green' if dom_boost > 1.0 else ''}">{dom_boost:.2f}x</span>
    </div>
  </div>

  <!-- Neural Mosaic Panel -->
  <div class="panel">
    <h2>Neural Mosaic (7 Neurons)</h2>
    <div class="stat-row">
      <span class="stat-label">Consensus</span>
      <span class="stat-value {'green' if consensus >= 0.7 else 'yellow'}">{consensus:.3f} {'PASS' if consensus >= 0.7 else 'FAIL'}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Votes LONG</span>
      <span class="stat-value green">{vote_counts.get('long', 0)}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Votes SHORT</span>
      <span class="stat-value red">{vote_counts.get('short', 0)}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Votes NEUTRAL</span>
      <span class="stat-value">{vote_counts.get('neutral', 0)}</span>
    </div>
    <div class="stat-row" style="margin-top:8px;">
      <span class="stat-label">Evolution Gen</span>
      <span class="stat-value">{evo.get('generation', 'N/A')}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Avg Accuracy</span>
      <span class="stat-value">{evo.get('avg_accuracy', 0):.1%}</span>
    </div>
  </div>

  <!-- Genomic Shock & Gates -->
  <div class="panel">
    <h2>Genomic Shock & Gates</h2>
    <div class="stat-row">
      <span class="stat-label">Shock Level</span>
      <span class="stat-value {'green' if shock_label == 'CALM' else ('yellow' if shock_label in ['NORMAL', 'ELEVATED'] else 'red')}">{shock_label} ({shock_score:.2f})</span>
    </div>
    <div style="margin-top: 10px;">
      {"".join(f'<div class="gate"><span class="gate-dot {"pass" if v else "fail"}"></span><span>{k}</span></div>' for k, v in gates.items())}
    </div>
    <div style="margin-top: 10px; border-top: 1px solid #1a1a2e; padding-top: 8px;">
      <div class="stat-row">
        <span class="stat-label">Active Class I</span>
        <span class="stat-value" style="color:#ff6644">{analytics.get('n_active_class_i', 0) if analytics else 0}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Active Class II</span>
        <span class="stat-value" style="color:#44aaff">{analytics.get('n_active_class_ii', 0) if analytics else 0}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Active Neural</span>
        <span class="stat-value" style="color:#aa44ff">{analytics.get('n_active_neural', 0) if analytics else 0}</span>
      </div>
    </div>
  </div>

  <!-- Radar SVG -->
  <div class="radar-container">
    <svg class="radar-svg" viewBox="-500 -500 1000 1000" xmlns="http://www.w3.org/2000/svg">
      <!-- Background rings -->
      <circle cx="0" cy="0" r="400" fill="none" stroke="#1a1a2e" stroke-width="0.5"/>
      <circle cx="0" cy="0" r="300" fill="none" stroke="#1a1a2e" stroke-width="0.5"/>
      <circle cx="0" cy="0" r="200" fill="none" stroke="#1a1a2e" stroke-width="0.5"/>
      <circle cx="0" cy="0" r="100" fill="none" stroke="#1a1a2e" stroke-width="0.5"/>

      <!-- Ring labels -->
      <text x="5" y="-395" fill="#333" font-size="10" font-family="monospace">1.0</text>
      <text x="5" y="-295" fill="#333" font-size="10" font-family="monospace">0.75</text>
      <text x="5" y="-195" fill="#333" font-size="10" font-family="monospace">0.50</text>
      <text x="5" y="-95" fill="#333" font-size="10" font-family="monospace">0.25</text>

      <!-- Spokes and TE dots -->
"""

    import math
    n = len(TE_FAMILIES)
    for i, te in enumerate(te_data):
        angle = (2 * math.pi * i / n) - math.pi / 2
        # Spoke line
        x_end = math.cos(angle) * 420
        y_end = math.sin(angle) * 420
        html += f'      <line x1="0" y1="0" x2="{x_end:.1f}" y2="{y_end:.1f}" stroke="#111122" stroke-width="0.5"/>\n'

        # TE dot (radius = activation level * 400, or 50 if inactive)
        is_active = te["active"]
        r = 350 if is_active else 80
        x = math.cos(angle) * r
        y = math.sin(angle) * r

        cls_color = "#ff6644" if te["cls"] == "I" else ("#44aaff" if te["cls"] == "II" else "#aa44ff")
        dot_r = 12 if is_active else 5
        opacity = 1.0 if is_active else 0.3
        glow = f'filter="url(#glow)"' if is_active else ""

        html += f'      <circle cx="{x:.1f}" cy="{y:.1f}" r="{dot_r}" fill="{cls_color}" opacity="{opacity}" {glow}/>\n'

        # Label
        label_r = 440
        lx = math.cos(angle) * label_r
        ly = math.sin(angle) * label_r
        anchor = "start" if -math.pi/2 < angle < math.pi/2 else "end"
        label_color = "#aaa" if is_active else "#333"
        font_size = 11 if is_active else 9
        # Rotate label to be readable
        rot = math.degrees(angle)
        if rot > 90 or rot < -90:
            rot += 180
        html += f'      <text x="{lx:.1f}" y="{ly:.1f}" fill="{label_color}" font-size="{font_size}" font-family="monospace" text-anchor="{anchor}" dominant-baseline="middle" transform="rotate({rot:.1f},{lx:.1f},{ly:.1f})">{te["name"]}</text>\n'

    # Active TE polygon
    if active_tes:
        points = []
        for i, te in enumerate(te_data):
            angle = (2 * math.pi * i / n) - math.pi / 2
            r = 350 if te["active"] else 0
            x = math.cos(angle) * r
            y = math.sin(angle) * r
            points.append(f"{x:.1f},{y:.1f}")
        poly = " ".join(points)
        html += f'      <polygon points="{poly}" fill="#00ccff" fill-opacity="0.06" stroke="#00ccff" stroke-width="1" stroke-opacity="0.4"/>\n'

    html += """
      <!-- Glow filter -->
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="4" result="blur"/>
          <feMerge>
            <feMergeNode in="blur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>

      <!-- Center label -->
      <text x="0" y="-10" fill="#00ccff" font-size="16" font-family="monospace" text-anchor="middle" font-weight="bold">TEQA v3.0</text>
      <text x="0" y="10" fill="#666" font-size="11" font-family="monospace" text-anchor="middle">33 QUBITS</text>
      <text x="0" y="28" fill="#444" font-size="10" font-family="monospace" text-anchor="middle">"""
    html += f'{len(active_tes)} ACTIVE'
    html += """</text>
    </svg>
  </div>

  <!-- TE Family Grid -->
  <div class="te-grid">
"""

    for te in te_data:
        cls_class = f"class-{'i' if te['cls'] == 'I' else ('ii' if te['cls'] == 'II' else 'n')}"
        active_class = "active" if te["active"] else ""
        cls_label = "Retro" if te["cls"] == "I" else ("DNA" if te["cls"] == "II" else "Neural")

        html += f"""    <div class="te-card {cls_class} {active_class}">
      <div class="te-name">{te["name"]}</div>
      <div class="te-signal">{te["signal"]} → {te["desc"]}</div>
      <div class="te-status">
        <span class="badge {'active' if te['active'] else 'inactive'}">{'ACTIVE' if te['active'] else 'SILENT'}</span>
        <span class="badge {cls_class}">{cls_label}</span>
      </div>
"""
        if te["appearances"] > 0:
            html += f'      <div style="margin-top:4px; font-size:10px; color:#555;">Tracked: {te["appearances"]}x | W:{te["wins"]} L:{te["losses"]}</div>\n'
        html += "    </div>\n"

    html += "  </div>\n\n"

    # Domestication table
    html += """  <!-- Domestication Tracker -->
  <div class="panel" style="grid-column: 1 / -1;">
    <h2>TE Domestication Tracker</h2>
    <div class="stat-row" style="margin-bottom:8px;">
      <span class="stat-label">Tracked Patterns</span>
      <span class="stat-value">""" + str(dom_data["total"]) + """</span>
    </div>
    <div class="stat-row" style="margin-bottom:12px;">
      <span class="stat-label">Domesticated</span>
      <span class="stat-value green">""" + str(dom_data["domesticated"]) + """</span>
    </div>
    <table class="dom-table">
      <tr>
        <th>Hash</th>
        <th>TE Combination</th>
        <th>W</th>
        <th>L</th>
        <th>Posterior WR</th>
        <th>Boost</th>
        <th>Status</th>
      </tr>
"""

    for p in dom_data["patterns"][:20]:
        status_class = "domestic" if p["domesticated"] else "wild"
        status_text = "DOMESTICATED" if p["domesticated"] else "wild"
        combo_short = p["combo"][:80] + ("..." if len(p["combo"]) > 80 else "")
        html += f"""      <tr>
        <td style="color:#444;">{p["hash"][:12]}</td>
        <td style="font-size:10px;">{combo_short}</td>
        <td style="color:#00ff88;">{p["wins"]}</td>
        <td style="color:#ff4466;">{p["losses"]}</td>
        <td>{p["posterior_wr"]:.3f}</td>
        <td>{p["boost"]:.2f}x</td>
        <td class="{status_class}">{status_text}</td>
      </tr>\n"""

    html += """    </table>
  </div>

</div>

<div class="footer">
  Quantum Children — Artificial Organism Intelligence | DOI: 10.5281/zenodo.18526575 | GPL-3.0
  <br>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
</div>

</body>
</html>"""

    return html


def main():
    live = "--live" in sys.argv
    serve = "--serve" in sys.argv

    html = generate_html()
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[RADAR] Generated: {OUTPUT_HTML}")

    if serve:
        import http.server
        import threading
        port = 8050
        handler = http.server.SimpleHTTPRequestHandler
        os.chdir(str(BASE_DIR))
        server = http.server.HTTPServer(("", port), handler)
        print(f"[RADAR] Serving on http://localhost:{port}/quantum_radar.html")
        webbrowser.open(f"http://localhost:{port}/quantum_radar.html")
        server.serve_forever()
    elif live:
        webbrowser.open(str(OUTPUT_HTML))
        print("[RADAR] Live mode — refreshing every 60s (Ctrl+C to stop)")
        while True:
            time.sleep(60)
            html = generate_html()
            with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[RADAR] Refreshed: {datetime.now().strftime('%H:%M:%S')}")
    else:
        webbrowser.open(str(OUTPUT_HTML))


if __name__ == "__main__":
    main()
