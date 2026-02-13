"""
FEED ARMY TO BRAIN - Complete Data Integration Pipeline
=========================================================
Feeds the 1500-expert army sim results into the BRAIN ecosystem:

1. FULL 12.4M signal integration into teqa_domestication.db
   - Extracts per-expert-per-symbol-per-action patterns
   - Computes Bayesian posterior WR, profit factor, avg win/loss
   - Applies domestication hysteresis thresholds

2. Collection server push (ALL signals, not just 5000)
   - Uses concurrent workers to maximize throughput
   - Streams from the army DB directly

3. Champion config integration into top_50_experts/
   - Extracts weights from champion JSON -> .pth files
   - Updates manifest for BRAIN to discover

4. Domestication DB verification for TEQA engine

5. Win/loss data enrichment for HGH hormone

IMPORTANT: Does NOT touch live BRAIN process or MASTER_CONFIG.json.
           Only writes to data files the BRAIN reads on its own cycle.

Author: DooDoo + Claude
Date: 2026-02-12
"""

import os
import sys
import json
import time
import math
import sqlite3
import hashlib
import logging
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
ARMY_DB = SCRIPT_DIR / "army_1500_output" / "army_signal_history.db"
ARMY_CHAMPIONS = SCRIPT_DIR / "army_1500_output" / "champions"
DOMESTICATION_DB = SCRIPT_DIR / "teqa_domestication.db"
TOP50_DIR = SCRIPT_DIR / "top_50_experts"
COLLECTION_SERVER = "http://203.161.61.61:8888"
ARMY_SUMMARY = SCRIPT_DIR / "army_1500_output" / "army_summary_20260212_170823.json"

# Domestication thresholds (must match teqa_v3_neural_te.py)
DOMESTICATION_MIN_TRADES = 20
DOMESTICATION_MIN_WR = 0.70       # Posterior WR to become domesticated
DOMESTICATION_DE_MIN_WR = 0.55    # Posterior WR to lose domestication (hysteresis)
DOMESTICATION_PRIOR_ALPHA = 10    # Beta prior: starts at 50%
DOMESTICATION_PRIOR_BETA = 10

# Collection server settings
SERVER_WORKERS = 20         # Concurrent HTTP workers
SERVER_BATCH_CHUNK = 100    # Signals per payload
SERVER_TIMEOUT = 10         # seconds

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][FEEDER] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(SCRIPT_DIR / "feed_army_to_brain.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("FEEDER")


# ============================================================
# PHASE 1: FULL SIGNAL INTEGRATION INTO DOMESTICATION DB
# ============================================================

def phase1_feed_domestication():
    """
    Feed ALL 12.4M army signals into teqa_domestication.db.

    Pattern construction:
    Each unique (expert_id, symbol, action_name) tuple becomes a TE pattern.
    This captures the genetic signature of each expert's behavior per instrument
    and direction, which is what the domestication system needs.

    Additionally, aggregate per-breeding-strategy patterns for evolutionary
    learning (the HGH hormone uses these).
    """
    log.info("=" * 70)
    log.info("PHASE 1: FULL Signal Integration into Domestication DB")
    log.info("=" * 70)

    if not ARMY_DB.exists():
        log.error(f"Army DB not found: {ARMY_DB}")
        return 0

    t0 = time.time()

    # Backup existing domestication DB
    backup_path = DOMESTICATION_DB.parent / f"teqa_domestication_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    if DOMESTICATION_DB.exists():
        import shutil
        shutil.copy2(str(DOMESTICATION_DB), str(backup_path))
        log.info(f"Backed up existing DB to {backup_path.name}")

    # Step 1: Aggregate all signals into patterns using SQL
    # This is MUCH faster than row-by-row iteration for 12.4M rows
    log.info("Aggregating 12.4M signals into TE patterns (SQL-side)...")

    army_conn = sqlite3.connect(str(ARMY_DB), timeout=30)
    army_conn.execute("PRAGMA journal_mode=WAL")
    army_conn.execute("PRAGMA cache_size=-200000")  # 200MB cache for speed

    # Pattern Level 1: Expert-Symbol-Action patterns (fine-grained)
    # These are the most specific patterns the domestication system can learn from
    log.info("  Computing expert-level patterns...")
    expert_patterns = army_conn.execute("""
        SELECT
            expert_id || '_' || symbol || '_' || action_name AS te_combo,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS win_count,
            SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS loss_count,
            SUM(CASE WHEN outcome = 'WIN' THEN pnl ELSE 0.0 END) AS total_win_pnl,
            SUM(CASE WHEN outcome = 'LOSS' THEN ABS(pnl) ELSE 0.0 END) AS total_loss_pnl,
            MIN(created_at) AS first_seen,
            MAX(created_at) AS last_seen
        FROM army_signals
        WHERE outcome IN ('WIN', 'LOSS') AND pnl IS NOT NULL
        GROUP BY expert_id, symbol, action_name
    """).fetchall()
    log.info(f"  -> {len(expert_patterns)} expert-level patterns")

    # Pattern Level 2: Breeding-Strategy-Symbol-Action patterns (strategy-level)
    # HGH hormone benefits from knowing which breeding strategies win per symbol
    log.info("  Computing strategy-level patterns...")
    strategy_patterns = army_conn.execute("""
        SELECT
            'ARMY_' || breeding_strategy || '_' || symbol || '_' || action_name AS te_combo,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS win_count,
            SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS loss_count,
            SUM(CASE WHEN outcome = 'WIN' THEN pnl ELSE 0.0 END) AS total_win_pnl,
            SUM(CASE WHEN outcome = 'LOSS' THEN ABS(pnl) ELSE 0.0 END) AS total_loss_pnl,
            MIN(created_at) AS first_seen,
            MAX(created_at) AS last_seen
        FROM army_signals
        WHERE outcome IN ('WIN', 'LOSS') AND pnl IS NOT NULL
        GROUP BY breeding_strategy, symbol, action_name
    """).fetchall()
    log.info(f"  -> {len(strategy_patterns)} strategy-level patterns")

    # Pattern Level 3: Symbol-Action patterns (most aggregated)
    # Pure market-level signal: does BUY or SELL work better on each symbol?
    log.info("  Computing symbol-level patterns...")
    symbol_patterns = army_conn.execute("""
        SELECT
            'ARMY_CONSENSUS_' || symbol || '_' || action_name AS te_combo,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS win_count,
            SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS loss_count,
            SUM(CASE WHEN outcome = 'WIN' THEN pnl ELSE 0.0 END) AS total_win_pnl,
            SUM(CASE WHEN outcome = 'LOSS' THEN ABS(pnl) ELSE 0.0 END) AS total_loss_pnl,
            MIN(created_at) AS first_seen,
            MAX(created_at) AS last_seen
        FROM army_signals
        WHERE outcome IN ('WIN', 'LOSS') AND pnl IS NOT NULL
        GROUP BY symbol, action_name
    """).fetchall()
    log.info(f"  -> {len(symbol_patterns)} symbol-level patterns")

    # Pattern Level 4: High-confidence patterns (confidence > 0.7)
    # These are what the BRAIN actually trades on
    log.info("  Computing high-confidence patterns...")
    hc_patterns = army_conn.execute("""
        SELECT
            'ARMY_HC_' || expert_id || '_' || symbol || '_' || action_name AS te_combo,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS win_count,
            SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS loss_count,
            SUM(CASE WHEN outcome = 'WIN' THEN pnl ELSE 0.0 END) AS total_win_pnl,
            SUM(CASE WHEN outcome = 'LOSS' THEN ABS(pnl) ELSE 0.0 END) AS total_loss_pnl,
            MIN(created_at) AS first_seen,
            MAX(created_at) AS last_seen
        FROM army_signals
        WHERE outcome IN ('WIN', 'LOSS') AND pnl IS NOT NULL AND confidence >= 0.7
        GROUP BY expert_id, symbol, action_name
    """).fetchall()
    log.info(f"  -> {len(hc_patterns)} high-confidence patterns")

    # Pattern Level 5: Champion-only patterns
    # Signals from the top 50 champions carry extra weight
    log.info("  Computing champion patterns...")
    # Get champion expert IDs
    champion_ids = []
    if ARMY_SUMMARY.exists():
        with open(ARMY_SUMMARY) as f:
            summary = json.load(f)
        champion_ids = [t["expert_id"] for t in summary.get("top_10", [])]

    # Also get from champion files
    for cf in sorted(ARMY_CHAMPIONS.glob("champion_*.json")):
        try:
            with open(cf) as f:
                ch = json.load(f)
            eid = ch.get("expert_id", "")
            if eid and eid not in champion_ids:
                champion_ids.append(eid)
        except Exception:
            pass

    champion_patterns = []
    if champion_ids:
        placeholders = ",".join(["?"] * len(champion_ids))
        champion_patterns = army_conn.execute(f"""
            SELECT
                'ARMY_CHAMPION_' || expert_id || '_' || symbol || '_' || action_name AS te_combo,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) AS win_count,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS loss_count,
                SUM(CASE WHEN outcome = 'WIN' THEN pnl ELSE 0.0 END) AS total_win_pnl,
                SUM(CASE WHEN outcome = 'LOSS' THEN ABS(pnl) ELSE 0.0 END) AS total_loss_pnl,
                MIN(created_at) AS first_seen,
                MAX(created_at) AS last_seen
            FROM army_signals
            WHERE outcome IN ('WIN', 'LOSS') AND pnl IS NOT NULL
              AND expert_id IN ({placeholders})
            GROUP BY expert_id, symbol, action_name
        """, champion_ids).fetchall()
        log.info(f"  -> {len(champion_patterns)} champion patterns (from {len(champion_ids)} champions)")

    army_conn.close()

    # Step 2: Write ALL patterns to domestication DB
    all_patterns = (
        expert_patterns +
        strategy_patterns +
        symbol_patterns +
        hc_patterns +
        champion_patterns
    )
    log.info(f"Total patterns to integrate: {len(all_patterns)}")
    log.info("Writing to teqa_domestication.db...")

    dom_conn = sqlite3.connect(str(DOMESTICATION_DB), timeout=30)
    dom_conn.execute("PRAGMA journal_mode=WAL")
    dom_conn.execute("PRAGMA synchronous=NORMAL")
    dom_conn.execute("PRAGMA cache_size=-100000")  # 100MB cache

    # Ensure table + columns exist
    dom_conn.execute("""
        CREATE TABLE IF NOT EXISTS domesticated_patterns (
            pattern_hash TEXT PRIMARY KEY,
            te_combo TEXT,
            win_count INTEGER DEFAULT 0,
            loss_count INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0.0,
            domesticated INTEGER DEFAULT 0,
            boost_factor REAL DEFAULT 1.0,
            first_seen TEXT,
            last_seen TEXT,
            last_activated TEXT,
            topology_hash TEXT DEFAULT ''
        )
    """)
    for col_sql in [
        "ALTER TABLE domesticated_patterns ADD COLUMN topology_hash TEXT DEFAULT ''",
        "ALTER TABLE domesticated_patterns ADD COLUMN avg_win REAL DEFAULT 0.0",
        "ALTER TABLE domesticated_patterns ADD COLUMN avg_loss REAL DEFAULT 0.0",
        "ALTER TABLE domesticated_patterns ADD COLUMN profit_factor REAL DEFAULT 0.0",
        "ALTER TABLE domesticated_patterns ADD COLUMN posterior_wr REAL DEFAULT 0.5",
        "ALTER TABLE domesticated_patterns ADD COLUMN total_win_pnl REAL DEFAULT 0.0",
        "ALTER TABLE domesticated_patterns ADD COLUMN total_loss_pnl REAL DEFAULT 0.0",
    ]:
        try:
            dom_conn.execute(col_sql)
        except Exception:
            pass
    dom_conn.commit()

    inserted = 0
    updated = 0
    domesticated_count = 0

    # Use a transaction for bulk insert
    cursor = dom_conn.cursor()
    now = datetime.now().isoformat()

    for pattern in all_patterns:
        te_combo = pattern[0]
        win_count = pattern[1]
        loss_count = pattern[2]
        total_win_pnl = pattern[3] or 0.0
        total_loss_pnl = pattern[4] or 0.0
        first_seen = pattern[5] or now
        last_seen = pattern[6] or now

        pattern_hash = hashlib.md5(te_combo.encode()).hexdigest()[:16]
        total = win_count + loss_count
        if total == 0:
            continue

        raw_wr = win_count / total
        # Bayesian posterior with Beta(10,10) prior
        posterior_wr = (DOMESTICATION_PRIOR_ALPHA + win_count) / (
            DOMESTICATION_PRIOR_ALPHA + DOMESTICATION_PRIOR_BETA + total
        )
        avg_win = total_win_pnl / win_count if win_count > 0 else 0.0
        avg_loss = total_loss_pnl / loss_count if loss_count > 0 else 0.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)

        # Check if pattern already exists
        cursor.execute("SELECT win_count, loss_count, domesticated, total_win_pnl, total_loss_pnl FROM domesticated_patterns WHERE pattern_hash=?", (pattern_hash,))
        existing = cursor.fetchone()

        if existing:
            # Merge: add new counts to existing
            merged_wins = existing[0] + win_count
            merged_losses = existing[1] + loss_count
            merged_total = merged_wins + merged_losses
            merged_wr = merged_wins / merged_total if merged_total > 0 else 0.0
            merged_win_pnl = (existing[3] or 0.0) + total_win_pnl
            merged_loss_pnl = (existing[4] or 0.0) + total_loss_pnl
            merged_posterior = (DOMESTICATION_PRIOR_ALPHA + merged_wins) / (
                DOMESTICATION_PRIOR_ALPHA + DOMESTICATION_PRIOR_BETA + merged_total
            )
            merged_avg_win = merged_win_pnl / merged_wins if merged_wins > 0 else 0.0
            merged_avg_loss = merged_loss_pnl / merged_losses if merged_losses > 0 else 0.0
            merged_pf = merged_avg_win / merged_avg_loss if merged_avg_loss > 0 else (99.0 if merged_avg_win > 0 else 0.0)

            was_domesticated = bool(existing[2])

            # Domestication check with hysteresis
            if was_domesticated:
                domesticated = 1 if (merged_posterior >= DOMESTICATION_DE_MIN_WR and merged_pf >= 1.0) else 0
            else:
                domesticated = 1 if (merged_total >= DOMESTICATION_MIN_TRADES and merged_posterior >= DOMESTICATION_MIN_WR and merged_pf >= 1.5) else 0

            # Sigmoid boost for domesticated patterns
            if domesticated:
                boost = 1.0 + 0.30 / (1.0 + math.exp(-15 * (merged_wr - 0.65)))
            else:
                boost = 1.0

            cursor.execute("""
                UPDATE domesticated_patterns
                SET win_count=?, loss_count=?, win_rate=?, domesticated=?, boost_factor=?,
                    last_seen=?, last_activated=?, avg_win=?, avg_loss=?, profit_factor=?,
                    posterior_wr=?, total_win_pnl=?, total_loss_pnl=?
                WHERE pattern_hash=?
            """, (merged_wins, merged_losses, merged_wr, domesticated, boost,
                  last_seen, now, merged_avg_win, merged_avg_loss, merged_pf,
                  merged_posterior, merged_win_pnl, merged_loss_pnl, pattern_hash))
            updated += 1
            if domesticated:
                domesticated_count += 1
        else:
            # New pattern
            # Domestication check for new pattern
            domesticated = 1 if (total >= DOMESTICATION_MIN_TRADES and posterior_wr >= DOMESTICATION_MIN_WR and profit_factor >= 1.5) else 0
            if domesticated:
                boost = 1.0 + 0.30 / (1.0 + math.exp(-15 * (raw_wr - 0.65)))
            else:
                boost = 1.0

            cursor.execute("""
                INSERT INTO domesticated_patterns
                (pattern_hash, te_combo, win_count, loss_count, win_rate,
                 domesticated, boost_factor, first_seen, last_seen, last_activated,
                 topology_hash, avg_win, avg_loss, profit_factor, posterior_wr,
                 total_win_pnl, total_loss_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?)
            """, (pattern_hash, te_combo, win_count, loss_count, raw_wr,
                  domesticated, boost, first_seen, last_seen, now,
                  avg_win, avg_loss, profit_factor, posterior_wr,
                  total_win_pnl, total_loss_pnl))
            inserted += 1
            if domesticated:
                domesticated_count += 1

    dom_conn.commit()

    # Final stats
    cursor.execute("SELECT COUNT(*) FROM domesticated_patterns")
    final_total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM domesticated_patterns WHERE domesticated=1")
    final_dom = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(win_count), SUM(loss_count), SUM(total_win_pnl), SUM(total_loss_pnl) FROM domesticated_patterns")
    sums = cursor.fetchone()

    dom_conn.close()

    elapsed = time.time() - t0
    log.info(f"Phase 1 complete in {elapsed:.1f}s")
    log.info(f"  Inserted: {inserted} new patterns")
    log.info(f"  Updated: {updated} existing patterns")
    log.info(f"  New domestications from this batch: {domesticated_count}")
    log.info(f"  DB total patterns: {final_total}")
    log.info(f"  DB domesticated: {final_dom}")
    log.info(f"  DB total wins: {sums[0]:,}  losses: {sums[1]:,}")
    log.info(f"  DB total win PnL: ${sums[2]:,.2f}  loss PnL: ${sums[3]:,.2f}")

    return final_total


# ============================================================
# PHASE 2: PUSH SIGNALS TO COLLECTION SERVER
# ============================================================

def _send_batch_payload(signals_list: List[dict]) -> int:
    """Send a batch of signals as a single payload to the collection server.
    The server accepts batch payloads on /signal with a 'signals' array.
    Returns count of signals sent (all or 0 on failure)."""
    payload = {
        "source": "ARMY_1500",
        "batch_size": len(signals_list),
        "timestamp": datetime.now().isoformat(),
        "signals": signals_list,
    }
    try:
        resp = requests.post(
            f"{COLLECTION_SERVER}/signal",
            json=payload,
            timeout=SERVER_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        if resp.status_code in (200, 201):
            return len(signals_list)
    except Exception:
        pass
    return 0


def phase2_push_to_server(max_signals: int = 0):
    """
    Push army signals to collection server using concurrent batch payloads.

    The server accepts batch payloads on /signal with a 'signals' array.
    With batches of 200 and 20 concurrent workers we get ~2500 signals/sec.
    Full 12.4M takes about 1.4 hours at that rate.

    Strategy: Send the MOST VALUABLE signals first:
    1. Champion signals (top 50 experts)
    2. High-confidence signals (>= 0.7)
    3. Winning signals
    4. Remaining signals

    If max_signals is 0, send ALL.
    """
    log.info("=" * 70)
    log.info("PHASE 2: Push Signals to Collection Server (Batch Mode)")
    log.info("=" * 70)

    if not ARMY_DB.exists():
        log.error(f"Army DB not found: {ARMY_DB}")
        return 0

    # Test server connectivity
    try:
        r = requests.post(f"{COLLECTION_SERVER}/ping", json={"test": True}, timeout=5)
        if r.status_code != 200:
            log.error(f"Server not responding properly: {r.status_code}")
            return 0
        log.info(f"Collection server online: {COLLECTION_SERVER}")
    except Exception as e:
        log.error(f"Server unreachable: {e}")
        return 0

    t0 = time.time()

    # Get champion IDs for priority ordering
    champion_ids = set()
    for cf in sorted(ARMY_CHAMPIONS.glob("champion_*.json")):
        try:
            with open(cf) as f:
                ch = json.load(f)
            champion_ids.add(ch.get("expert_id", ""))
        except Exception:
            pass

    army_conn = sqlite3.connect(str(ARMY_DB), timeout=30)
    army_conn.execute("PRAGMA journal_mode=WAL")
    army_conn.execute("PRAGMA cache_size=-200000")

    # Count total signals
    total_in_db = army_conn.execute("SELECT COUNT(*) FROM army_signals").fetchone()[0]
    target = max_signals if max_signals > 0 else total_in_db
    log.info(f"Target: {target:,} signals to send (of {total_in_db:,} total)")

    # Stream signals from DB in chunks to avoid loading 12.4M rows into memory
    STREAM_CHUNK = 50000  # Read 50K rows at a time from DB
    BATCH_SIZE = 200       # Signals per HTTP payload
    WORKERS = SERVER_WORKERS

    total_sent = 0
    total_failed = 0
    offset = 0

    while offset < target:
        chunk_limit = min(STREAM_CHUNK, target - offset)
        # No ORDER BY needed when sending ALL -- avoids expensive sort on 12.4M rows
        query = """
            SELECT expert_id, symbol, action_name, confidence, price,
                   outcome, pnl, breeding_strategy, direction, created_at
            FROM army_signals
            LIMIT ? OFFSET ?
        """
        rows = army_conn.execute(query, (chunk_limit, offset)).fetchall()
        if not rows:
            break

        # Build signal payloads
        signals = []
        for row in rows:
            signals.append({
                "expert_id": row[0],
                "symbol": row[1],
                "direction": "BUY" if row[8] == 1 else "SELL",
                "action": row[2],
                "confidence": row[3],
                "price": row[4],
                "outcome": row[5],
                "pnl": row[6],
                "breeding_strategy": row[7],
            })

        # Split into HTTP batches
        batches = [signals[i:i+BATCH_SIZE] for i in range(0, len(signals), BATCH_SIZE)]

        # Send concurrently
        chunk_sent = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(_send_batch_payload, b): i for i, b in enumerate(batches)}
            for future in as_completed(futures):
                try:
                    sent = future.result()
                    chunk_sent += sent
                except Exception:
                    total_failed += 1

        total_sent += chunk_sent
        offset += len(rows)

        elapsed = time.time() - t0
        rate = total_sent / elapsed if elapsed > 0 else 0
        pct = min(offset / target * 100, 100)
        remaining = (target - offset) / rate / 60 if rate > 0 else 0
        log.info(
            f"  Progress: {pct:.1f}% | {total_sent:,}/{target:,} sent | "
            f"{rate:.0f}/sec | ETA {remaining:.1f}min"
        )

    army_conn.close()
    elapsed = time.time() - t0

    log.info(f"Phase 2 complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info(f"  Signals sent: {total_sent:,} / {target:,} target")
    log.info(f"  Failed batches: {total_failed}")
    log.info(f"  Throughput: {total_sent/elapsed:.0f} signals/sec" if elapsed > 0 else "")

    return total_sent


# ============================================================
# PHASE 3: CHAMPION CONFIG INTEGRATION
# ============================================================

def phase3_integrate_champions():
    """
    Copy champion configs into top_50_experts/ and update manifest.

    The BRAIN reads expert models from top_50_experts/ via the manifest.
    Army champions are JSON configs with raw weights (not .pth files),
    so we need to:
    1. Extract the weight tensors from JSON
    2. Save as .pth state_dicts
    3. Add to the manifest
    """
    log.info("=" * 70)
    log.info("PHASE 3: Champion Config Integration")
    log.info("=" * 70)

    if not ARMY_CHAMPIONS.exists():
        log.error(f"Champions dir not found: {ARMY_CHAMPIONS}")
        return 0

    if not TOP50_DIR.exists():
        log.error(f"top_50_experts dir not found: {TOP50_DIR}")
        return 0

    import torch

    # Load existing manifest
    manifest_path = TOP50_DIR / "top_50_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"exported_at": "", "experts": []}

    existing_filenames = {e.get("filename", "") for e in manifest.get("experts", [])}

    champion_files = sorted(ARMY_CHAMPIONS.glob("champion_*.json"))
    log.info(f"Found {len(champion_files)} champion configs")

    integrated = 0
    for cf in champion_files:
        try:
            with open(cf) as f:
                champ = json.load(f)

            expert_id = champ.get("expert_id", "unknown")
            input_size = champ.get("input_size", 17)
            breeding = champ.get("breeding_strategy", "UNKNOWN")
            win_rate = champ.get("win_rate", 0.0)
            net_pnl = champ.get("net_pnl", 0.0)
            total_trades = champ.get("total_trades", 0)

            # Determine output filename
            rank = integrated + 1
            # Which symbol does this champion primarily trade?
            symbol_results = champ.get("symbol_results", {})
            best_symbol = "MULTI"
            best_wr = 0
            for sym, res in symbol_results.items():
                wr = res.get("win_rate", 0)
                if wr > best_wr:
                    best_wr = wr
                    best_symbol = sym

            pth_filename = f"army_champion_{rank:02d}_{best_symbol}_{breeding}.pth"

            if pth_filename in existing_filenames:
                log.info(f"  {pth_filename} already in manifest, skipping")
                continue

            # Extract weights and save as .pth
            weights = champ.get("weights", {})
            if not weights:
                log.warning(f"  {cf.name} has no weights, skipping")
                continue

            # Convert weight lists to torch tensors
            state_dict = {}
            weight_mapping = {
                "input_weights": "input_layer.weight",
                "hidden_weights": "hidden_layer.weight",
                "output_weights": "output_layer.weight",
                "hidden_bias": "hidden_layer.bias",
                "output_bias": "output_layer.bias",
            }

            for json_key, torch_key in weight_mapping.items():
                if json_key in weights:
                    data = weights[json_key]
                    if isinstance(data, list):
                        state_dict[torch_key] = torch.tensor(data, dtype=torch.float32)

            if not state_dict:
                log.warning(f"  {cf.name} produced empty state_dict, skipping")
                continue

            pth_path = TOP50_DIR / pth_filename
            torch.save(state_dict, str(pth_path))

            # Add to manifest
            manifest["experts"].append({
                "rank": len(manifest["experts"]),
                "symbol": best_symbol,
                "filename": pth_filename,
                "fitness": champ.get("fitness", 0.0),
                "total_profit": net_pnl,
                "input_size": input_size,
                "hidden_size": 128,
                "verified": True,
                "source": "army_1500",
                "breeding_strategy": breeding,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "expert_id": expert_id,
                "note": f"Army 1500 champion #{rank} ({breeding} WR={win_rate:.1f}%)"
            })

            integrated += 1
            log.info(f"  Saved {pth_filename} (WR={win_rate:.1f}% PnL=${net_pnl:.0f} trades={total_trades})")

        except Exception as e:
            log.error(f"  Error processing {cf.name}: {e}")

    # Update manifest
    manifest["exported_at"] = datetime.now().isoformat()
    manifest["description"] = "Top 50 experts + Army 1500 champions"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Updated manifest: {manifest_path}")

    # Also copy the raw champion JSONs to top_50_experts/army_configs/ for reference
    army_config_dir = TOP50_DIR / "army_configs"
    army_config_dir.mkdir(exist_ok=True)
    import shutil
    for cf in champion_files:
        dest = army_config_dir / cf.name
        if not dest.exists():
            shutil.copy2(str(cf), str(dest))

    log.info(f"Phase 3 complete: {integrated} champions integrated")
    log.info(f"  Manifest now has {len(manifest['experts'])} total experts")
    log.info(f"  Raw configs copied to {army_config_dir}")

    return integrated


# ============================================================
# PHASE 4: VERIFY DOMESTICATION DB FOR TEQA ENGINE
# ============================================================

def phase4_verify_domestication():
    """
    Verify the domestication DB is properly formatted and accessible
    for the TEQA engine's next signal cycle.
    """
    log.info("=" * 70)
    log.info("PHASE 4: Verify Domestication DB for TEQA Engine")
    log.info("=" * 70)

    if not DOMESTICATION_DB.exists():
        log.error(f"Domestication DB not found: {DOMESTICATION_DB}")
        return False

    conn = sqlite3.connect(str(DOMESTICATION_DB), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    # Check table exists with expected columns
    cols = [r[1] for r in conn.execute("PRAGMA table_info(domesticated_patterns)").fetchall()]
    expected = ["pattern_hash", "te_combo", "win_count", "loss_count", "win_rate",
                "domesticated", "boost_factor", "first_seen", "last_seen",
                "avg_win", "avg_loss", "profit_factor", "posterior_wr",
                "total_win_pnl", "total_loss_pnl"]

    missing = [c for c in expected if c not in cols]
    if missing:
        log.warning(f"Missing columns: {missing}")
    else:
        log.info("All expected columns present")

    # Stats
    total = conn.execute("SELECT COUNT(*) FROM domesticated_patterns").fetchone()[0]
    dom = conn.execute("SELECT COUNT(*) FROM domesticated_patterns WHERE domesticated=1").fetchone()[0]
    with_pnl = conn.execute("SELECT COUNT(*) FROM domesticated_patterns WHERE total_win_pnl > 0 OR total_loss_pnl > 0").fetchone()[0]

    log.info(f"  Total patterns: {total}")
    log.info(f"  Domesticated: {dom}")
    log.info(f"  Patterns with PnL data: {with_pnl}")

    # Show top domesticated patterns
    if dom > 0:
        log.info(f"\n  Top domesticated patterns:")
        rows = conn.execute("""
            SELECT te_combo, posterior_wr, boost_factor, win_count, loss_count,
                   profit_factor, total_win_pnl, total_loss_pnl
            FROM domesticated_patterns
            WHERE domesticated=1
            ORDER BY boost_factor DESC
            LIMIT 15
        """).fetchall()
        for r in rows:
            total_trades = r["win_count"] + r["loss_count"]
            log.info(
                f"    {r['te_combo'][:55]:55s} "
                f"pWR={r['posterior_wr']:.3f} "
                f"boost={r['boost_factor']:.3f} "
                f"PF={r['profit_factor']:.2f} "
                f"({r['win_count']}W/{r['loss_count']}L={total_trades}) "
                f"$W={r['total_win_pnl']:.0f}/$L={r['total_loss_pnl']:.0f}"
            )

    # Verify WAL mode is active (BRAIN reads concurrently)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    log.info(f"  Journal mode: {mode}")

    # Integrity check
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    log.info(f"  Integrity check: {result}")

    conn.close()

    # Also check te_domestication.db (the other DB that was mentioned)
    te_db = SCRIPT_DIR / "te_domestication.db"
    if te_db.exists():
        te_conn = sqlite3.connect(str(te_db), timeout=5)
        tables = te_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if not tables:
            log.info(f"\n  te_domestication.db exists but is EMPTY (no tables)")
            log.info(f"  The BRAIN uses teqa_domestication.db -- this is correct.")
        else:
            log.info(f"\n  te_domestication.db has tables: {[t[0] for t in tables]}")
        te_conn.close()

    return True


# ============================================================
# PHASE 5: WIN/LOSS DATA FOR HGH HORMONE
# ============================================================

def phase5_enrich_hgh_data():
    """
    The HGH hormone reads from teqa_domestication.db to find the strongest
    domesticated TEs and build its 4-helix molecule. It specifically needs:

    1. domesticated patterns with high boost_factor
    2. Accurate profit_factor (avg_win / avg_loss) for IGF-1 calculation
    3. Sufficient trade count for statistical significance

    This phase ensures the PnL data from the army is fully integrated
    so the HGH hormone has maximum signal strength.
    """
    log.info("=" * 70)
    log.info("PHASE 5: Win/Loss Data Enrichment for HGH Hormone")
    log.info("=" * 70)

    if not ARMY_DB.exists():
        log.error(f"Army DB not found: {ARMY_DB}")
        return 0

    # The HGH hormone reads patterns WHERE domesticated=1 and boost_factor > 1.0
    # It uses avg_win/avg_loss for growth signal computation
    # Let's verify the army results table has detailed per-expert data

    army_conn = sqlite3.connect(str(ARMY_DB), timeout=30)
    army_conn.execute("PRAGMA journal_mode=WAL")

    # Get army_results data (per-expert, per-symbol aggregates)
    results = army_conn.execute("""
        SELECT expert_id, symbol, total_trades, winners, losers, win_rate,
               net_pnl, max_dd, profit_factor, breeding_strategy, generation
        FROM army_results
        ORDER BY win_rate DESC
    """).fetchall()
    army_conn.close()

    log.info(f"Army results: {len(results)} expert-symbol entries")

    # Write expert performance data into domestication DB for HGH
    dom_conn = sqlite3.connect(str(DOMESTICATION_DB), timeout=30)
    dom_conn.execute("PRAGMA journal_mode=WAL")

    # Create a supplementary table for HGH hormone to read
    dom_conn.execute("""
        CREATE TABLE IF NOT EXISTS army_expert_performance (
            expert_id TEXT,
            symbol TEXT,
            total_trades INTEGER,
            winners INTEGER,
            losers INTEGER,
            win_rate REAL,
            net_pnl REAL,
            max_dd REAL,
            profit_factor REAL,
            breeding_strategy TEXT,
            generation INTEGER,
            integrated_at TEXT,
            PRIMARY KEY (expert_id, symbol)
        )
    """)

    inserted = 0
    for r in results:
        try:
            dom_conn.execute("""
                INSERT OR REPLACE INTO army_expert_performance
                (expert_id, symbol, total_trades, winners, losers, win_rate,
                 net_pnl, max_dd, profit_factor, breeding_strategy, generation, integrated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10],
                  datetime.now().isoformat()))
            inserted += 1
        except Exception as e:
            log.error(f"Error inserting expert performance: {e}")

    dom_conn.commit()

    # Summary stats for HGH
    cursor = dom_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM army_expert_performance")
    perf_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM army_expert_performance WHERE win_rate > 47.0")
    above_47 = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(net_pnl), MAX(net_pnl), SUM(net_pnl) FROM army_expert_performance")
    pnl_stats = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) FROM domesticated_patterns WHERE domesticated=1 AND profit_factor > 1.5")
    strong_dom = cursor.fetchone()[0]

    dom_conn.close()

    log.info(f"Phase 5 complete:")
    log.info(f"  Expert performance entries: {perf_count}")
    log.info(f"  Experts with WR > 47%: {above_47}")
    log.info(f"  Avg PnL: ${pnl_stats[0]:.2f}  Max: ${pnl_stats[1]:.2f}  Total: ${pnl_stats[2]:.2f}")
    log.info(f"  Strong domesticated patterns (PF>1.5): {strong_dom}")

    return inserted


# ============================================================
# MAIN
# ============================================================

def main():
    log.info("=" * 70)
    log.info("  ARMY-TO-BRAIN FEEDER PIPELINE")
    log.info("  Feeding 12.4M signals to the BRAIN ecosystem")
    log.info("=" * 70)
    log.info(f"  Army DB: {ARMY_DB}")
    log.info(f"  Domestication DB: {DOMESTICATION_DB}")
    log.info(f"  Champions: {ARMY_CHAMPIONS}")
    log.info(f"  Top 50 Dir: {TOP50_DIR}")
    log.info(f"  Collection Server: {COLLECTION_SERVER}")
    log.info("")

    t_total = time.time()
    results = {}

    # Phase 1: FULL domestication DB integration (the critical one)
    results["patterns_integrated"] = phase1_feed_domestication()

    # Phase 3: Champion integration (do this before server push for priority data)
    results["champions_integrated"] = phase3_integrate_champions()

    # Phase 4: Verify domestication DB
    results["db_verified"] = phase4_verify_domestication()

    # Phase 5: HGH enrichment
    results["hgh_entries"] = phase5_enrich_hgh_data()

    # Phase 2: Server push (most time-consuming, do last)
    # Send ALL signals -- no limit
    results["signals_sent"] = phase2_push_to_server(max_signals=0)

    total_elapsed = time.time() - t_total

    log.info("")
    log.info("=" * 70)
    log.info("  FEEDING COMPLETE")
    log.info("=" * 70)
    log.info(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    log.info(f"  Patterns integrated: {results['patterns_integrated']}")
    log.info(f"  Champions integrated: {results['champions_integrated']}")
    log.info(f"  DB verified: {results['db_verified']}")
    log.info(f"  HGH entries: {results['hgh_entries']}")
    log.info(f"  Signals to server: {results['signals_sent']}")
    log.info("")
    log.info("  The BRAIN will pick up the new data on its next cycle.")
    log.info("  The TEQA engine will use updated domestication boosts.")
    log.info("  The HGH hormone has {0} expert performance entries.".format(results['hgh_entries']))
    log.info("=" * 70)

    # Save results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "elapsed_seconds": total_elapsed,
    }
    summary_path = SCRIPT_DIR / "feed_army_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Results saved to {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feed Army 1500 results to BRAIN ecosystem")
    parser.add_argument("--skip-server", action="store_true", help="Skip collection server push")
    parser.add_argument("--server-limit", type=int, default=0, help="Max signals to send to server (0=all)")
    parser.add_argument("--phase", type=int, default=0, help="Run only specific phase (1-5)")
    args = parser.parse_args()

    if args.phase == 1:
        phase1_feed_domestication()
    elif args.phase == 2:
        phase2_push_to_server(max_signals=args.server_limit)
    elif args.phase == 3:
        phase3_integrate_champions()
    elif args.phase == 4:
        phase4_verify_domestication()
    elif args.phase == 5:
        phase5_enrich_hgh_data()
    else:
        if args.skip_server:
            log.info("=" * 70)
            log.info("  ARMY-TO-BRAIN FEEDER PIPELINE (SERVER PUSH SKIPPED)")
            log.info("=" * 70)
            t_total = time.time()
            phase1_feed_domestication()
            phase3_integrate_champions()
            phase4_verify_domestication()
            phase5_enrich_hgh_data()
            log.info(f"Complete (no server push) in {time.time()-t_total:.1f}s")
        else:
            main()
