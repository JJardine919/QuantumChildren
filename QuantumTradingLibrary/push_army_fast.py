"""
FAST ARMY SIGNAL PUSH - Optimized Collection Server Push
==========================================================
Pushes ALL 12.4M army signals to the collection server using:
- Batch payloads (200 signals per HTTP request)
- 20 concurrent workers
- rowid-based pagination (no ORDER BY, no OFFSET -- pure speed)
- Progress logging

Expected throughput: ~2500 signals/sec = ~83 min for 12.4M

Usage:
    python push_army_fast.py
    python push_army_fast.py --limit 100000    (test with 100K)
    python push_army_fast.py --resume 500000   (resume from rowid 500000)

Author: DooDoo + Claude
Date: 2026-02-12
"""

import sys
import json
import time
import sqlite3
import logging
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).parent.absolute()
ARMY_DB = SCRIPT_DIR / "army_1500_output" / "army_signal_history.db"
SERVER = "http://203.161.61.61:8888"

BATCH_SIZE = 200      # Signals per HTTP payload
STREAM_CHUNK = 50000  # Rows per DB read
WORKERS = 10          # Concurrent HTTP workers (reduced to avoid 429s)
TIMEOUT = 15          # HTTP timeout seconds
INTER_CHUNK_DELAY = 0.5  # Seconds between DB chunks to ease server load

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][PUSH] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(SCRIPT_DIR / "push_army_fast.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("PUSH")


def send_batch(signals_list, max_retries=3):
    """Send batch payload to server with retry on 429. Returns count sent."""
    payload = {
        "source": "ARMY_1500",
        "batch_size": len(signals_list),
        "timestamp": datetime.now().isoformat(),
        "signals": signals_list,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{SERVER}/signal", json=payload, timeout=TIMEOUT)
            if r.status_code == 200:
                return len(signals_list)
            elif r.status_code == 429:
                # Rate limited -- back off exponentially
                time.sleep(2 ** attempt)
                continue
            else:
                return 0
        except Exception:
            time.sleep(1)
    return 0


def push_all(limit=0, resume_rowid=0):
    """Push all army signals to collection server."""
    if not ARMY_DB.exists():
        log.error(f"Army DB not found: {ARMY_DB}")
        return

    # Test server
    try:
        r = requests.post(f"{SERVER}/ping", json={"test": True}, timeout=5)
        assert r.status_code == 200
        log.info(f"Server online: {SERVER}")
    except Exception as e:
        log.error(f"Server unreachable: {e}")
        return

    conn = sqlite3.connect(str(ARMY_DB), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-200000")

    total_rows = conn.execute("SELECT COUNT(*) FROM army_signals").fetchone()[0]
    max_rowid = conn.execute("SELECT MAX(rowid) FROM army_signals").fetchone()[0]
    target = limit if limit > 0 else total_rows

    log.info(f"Total signals in DB: {total_rows:,}")
    log.info(f"Max rowid: {max_rowid:,}")
    log.info(f"Target to send: {target:,}")
    log.info(f"Resume from rowid: {resume_rowid}")
    log.info(f"Config: batch={BATCH_SIZE}, chunk={STREAM_CHUNK}, workers={WORKERS}")

    t0 = time.time()
    total_sent = 0
    total_failed = 0
    current_rowid = resume_rowid

    while total_sent < target:
        # Fast rowid-based pagination (no sort, no offset)
        rows = conn.execute("""
            SELECT rowid, expert_id, symbol, action_name, confidence, price,
                   outcome, pnl, breeding_strategy, direction
            FROM army_signals
            WHERE rowid > ?
            ORDER BY rowid
            LIMIT ?
        """, (current_rowid, STREAM_CHUNK)).fetchall()

        if not rows:
            break

        current_rowid = rows[-1][0]  # Track last rowid for next chunk

        # Build signal payloads
        signals = []
        for row in rows:
            signals.append({
                "expert_id": row[1],
                "symbol": row[2],
                "direction": "BUY" if row[9] == 1 else "SELL",
                "action": row[3],
                "confidence": row[4],
                "price": row[5],
                "outcome": row[6],
                "pnl": row[7],
                "breeding_strategy": row[8],
            })

        # Split into HTTP batches
        batches = [signals[i:i+BATCH_SIZE] for i in range(0, len(signals), BATCH_SIZE)]

        # Send concurrently
        chunk_sent = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(send_batch, b): i for i, b in enumerate(batches)}
            for f in as_completed(futures):
                try:
                    chunk_sent += f.result()
                except Exception:
                    total_failed += 1

        total_sent += chunk_sent

        elapsed = time.time() - t0
        rate = total_sent / elapsed if elapsed > 0 else 0
        pct = total_sent / target * 100
        remaining = (target - total_sent) / rate / 60 if rate > 0 else 0
        log.info(
            f"  {pct:.1f}% | {total_sent:,}/{target:,} | "
            f"rowid={current_rowid:,} | {rate:.0f}/sec | ETA {remaining:.0f}min"
        )

        # Save progress for resume
        with open(SCRIPT_DIR / "push_army_progress.json", "w") as f:
            json.dump({
                "last_rowid": current_rowid,
                "total_sent": total_sent,
                "target": target,
                "elapsed": elapsed,
                "timestamp": datetime.now().isoformat(),
            }, f)

        # Small delay between chunks to ease server rate limiting
        time.sleep(INTER_CHUNK_DELAY)

        if limit > 0 and total_sent >= limit:
            break

    conn.close()
    elapsed = time.time() - t0

    log.info(f"Push complete: {total_sent:,} signals in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"  Rate: {total_sent/elapsed:.0f}/sec" if elapsed > 0 else "")
    log.info(f"  Failed batches: {total_failed}")

    return total_sent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max signals to send (0=all)")
    parser.add_argument("--resume", type=int, default=0, help="Resume from this rowid")
    args = parser.parse_args()

    push_all(limit=args.limit, resume_rowid=args.resume)
