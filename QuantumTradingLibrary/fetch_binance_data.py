"""
Fetch 5 years of BTCUSDT kline data from Binance public API.
Timeframes: M1, M5, M15
Output: CSV files matching MT5 column format for walk-forward training.

Binance API: GET https://api.binance.com/api/v3/klines
  - No API key required
  - Max 1000 candles per request
  - Timestamps in milliseconds
"""
import requests
import time
import csv
import os
from datetime import datetime, timezone

# ── Configuration ──────────────────────────────────────────────────
SYMBOL = "BTCUSDT"
BASE_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 1000

# 5-year range: 2021-02-13 00:00:00 UTC to 2026-02-13 00:00:00 UTC
START_MS = int(datetime(2021, 2, 13, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2026, 2, 13, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binance_data")

TIMEFRAMES = [
    {"interval": "15m", "label": "M15", "filename": "BTCUSDT_M15_5yr.csv", "ms_per_candle": 15 * 60 * 1000},
    {"interval": "5m",  "label": "M5",  "filename": "BTCUSDT_M5_5yr.csv",  "ms_per_candle": 5 * 60 * 1000},
    {"interval": "1m",  "label": "M1",  "filename": "BTCUSDT_M1_5yr.csv",  "ms_per_candle": 1 * 60 * 1000},
]

# CSV header matching MT5 format
CSV_HEADER = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]

# ── Helpers ────────────────────────────────────────────────────────

def ms_to_date_str(ms):
    """Convert milliseconds to human-readable UTC date string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def fetch_klines(symbol, interval, start_time_ms, end_time_ms, limit=1000):
    """
    Fetch one batch of klines from Binance.
    Returns list of kline arrays, or empty list on failure.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": limit,
    }

    for attempt in range(5):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                # Rate limited -- back off
                wait = 30 * (attempt + 1)
                print(f"    [RATE LIMITED] Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            if resp.status_code == 418:
                # IP banned temporarily
                wait = 120
                print(f"    [IP BAN] Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            print(f"    [ERROR] HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(5)

        except requests.exceptions.RequestException as e:
            print(f"    [NETWORK ERROR] {e} -- retry {attempt+1}/5")
            time.sleep(10)

    return []


def fetch_all_klines(symbol, interval, start_ms, end_ms, ms_per_candle, label):
    """
    Fetch all klines for a timeframe using proper pagination.
    Paginates by using the open_time of the last candle + 1 ms_per_candle as next startTime.
    Returns list of parsed rows.
    """
    all_rows = []
    current_start = start_ms
    request_count = 0

    # Estimate total requests for progress reporting
    total_candles_est = (end_ms - start_ms) / ms_per_candle
    total_requests_est = int(total_candles_est / LIMIT) + 1

    print(f"\n{'='*70}")
    print(f"  {label} ({interval}) -- {SYMBOL}")
    print(f"  Range: {ms_to_date_str(start_ms)} -> {ms_to_date_str(end_ms)}")
    print(f"  Estimated requests: ~{total_requests_est}")
    print(f"{'='*70}")

    while current_start < end_ms:
        data = fetch_klines(symbol, interval, current_start, end_ms, LIMIT)

        if not data:
            print(f"    [WARN] Empty response at {ms_to_date_str(current_start)} -- stopping.")
            break

        request_count += 1

        for candle in data:
            # Binance kline format:
            # [0] open_time, [1] open, [2] high, [3] low, [4] close, [5] volume,
            # [6] close_time, [7] quote_volume, [8] trades, [9] taker_buy_base,
            # [10] taker_buy_quote, [11] ignore

            open_time_ms = int(candle[0])

            # Skip candles outside our range (defensive)
            if open_time_ms >= end_ms:
                continue

            row = {
                "time": open_time_ms // 1000,          # Unix seconds
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "tick_volume": float(candle[5]),        # Base volume
                "spread": 0,                             # Not available
                "real_volume": float(candle[7]),         # Quote volume
            }
            all_rows.append(row)

        # Advance: use the open_time of the last candle + one candle interval
        last_open_time = int(data[-1][0])
        current_start = last_open_time + ms_per_candle

        # Progress reporting every 20 requests
        if request_count % 20 == 0:
            pct = min(100.0, (current_start - start_ms) / (end_ms - start_ms) * 100)
            print(f"    [{label}] {request_count}/{total_requests_est} requests | "
                  f"{len(all_rows):>10,} bars | "
                  f"{pct:5.1f}% | at {ms_to_date_str(current_start)}")

        # Rate limit: ~0.08s between requests = ~750 req/min (safe under 1200 limit)
        time.sleep(0.08)

        # If we got fewer than LIMIT candles, we've reached the end
        if len(data) < LIMIT:
            break

    print(f"    [{label}] DONE -- {request_count} requests | {len(all_rows):,} bars total")
    return all_rows


def save_to_csv(rows, filepath):
    """Save rows to CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def remove_duplicates(rows):
    """Remove duplicate timestamps, keeping last occurrence."""
    seen = {}
    for row in rows:
        seen[row["time"]] = row
    deduped = sorted(seen.values(), key=lambda r: r["time"])
    removed = len(rows) - len(deduped)
    if removed > 0:
        print(f"    Removed {removed} duplicate rows")
    return deduped


# ── Main ───────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "#" * 70)
    print("#  BINANCE BTCUSDT HISTORICAL DATA FETCHER")
    print(f"#  Period: {ms_to_date_str(START_MS)} to {ms_to_date_str(END_MS)} (5 years)")
    print(f"#  Output: {OUTPUT_DIR}")
    print("#" * 70)

    results = {}
    total_start = time.time()

    for tf in TIMEFRAMES:
        tf_start = time.time()

        rows = fetch_all_klines(
            SYMBOL,
            tf["interval"],
            START_MS,
            END_MS,
            tf["ms_per_candle"],
            tf["label"],
        )

        if not rows:
            print(f"    [ERROR] No data fetched for {tf['label']}!")
            continue

        # Deduplicate
        rows = remove_duplicates(rows)

        # Save
        filepath = os.path.join(OUTPUT_DIR, tf["filename"])
        save_to_csv(rows, filepath)

        tf_elapsed = time.time() - tf_start
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

        first_dt = ms_to_date_str(rows[0]["time"] * 1000)
        last_dt = ms_to_date_str(rows[-1]["time"] * 1000)

        results[tf["label"]] = {
            "rows": len(rows),
            "file": filepath,
            "size_mb": file_size_mb,
            "elapsed": tf_elapsed,
            "first": first_dt,
            "last": last_dt,
        }

        print(f"\n    Saved: {filepath}")
        print(f"    Rows: {len(rows):,} | Size: {file_size_mb:.1f} MB | Time: {tf_elapsed:.0f}s")
        print(f"    Range: {first_dt} -> {last_dt}")

    # ── Final Summary ──
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("  DOWNLOAD COMPLETE")
    print("=" * 70)

    for label, info in results.items():
        print(f"  {label:>4}: {info['rows']:>12,} rows | {info['size_mb']:>7.1f} MB | {info['file']}")

    total_rows = sum(info["rows"] for info in results.values())
    print(f"\n  Total rows: {total_rows:,}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
