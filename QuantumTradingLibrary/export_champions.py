"""
Champion Exporter for ETARE Redux
=================================
Extracts trained LSTM champions from local SQLite database
and packages them for deployment to VPS Docker.

Usage:
    python export_champions.py [--output-dir ./champions] [--top-n 1]

Output:
    ./champions/
        champion_EURUSD.pth      # PyTorch state_dict
        champion_GBPUSD.pth
        ...
        champions_manifest.json  # Metadata (fitness, profit, etc.)
"""

import sqlite3
import torch
import json
import os
import argparse
from datetime import datetime
from pathlib import Path


def export_champions(db_path: str, output_dir: str, top_n: int = 1):
    """
    Export top N champions per symbol from the training database.

    Args:
        db_path: Path to etare_redux_v2.db
        output_dir: Directory to save champion files
        top_n: Number of top performers to export per symbol (default: 1)
    """

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Connect to database
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all trained symbols
    cursor.execute("SELECT DISTINCT symbol FROM population_state")
    symbols = [row[0] for row in cursor.fetchall()]

    if not symbols:
        print("ERROR: No trained symbols found in database")
        conn.close()
        return False

    print(f"Found {len(symbols)} trained symbols: {symbols}")
    print(f"Exporting top {top_n} champion(s) per symbol...\n")

    # Manifest to track all exports
    manifest = {
        "exported_at": datetime.now().isoformat(),
        "source_db": db_path,
        "top_n": top_n,
        "champions": {}
    }

    total_exported = 0

    for symbol in symbols:
        # Get top N champions for this symbol (ordered by fitness DESC)
        cursor.execute("""
            SELECT individual_index, weights, fitness, total_profit
            FROM population_state
            WHERE symbol = ?
            ORDER BY fitness DESC
            LIMIT ?
        """, (symbol, top_n))

        rows = cursor.fetchall()

        if not rows:
            print(f"  [{symbol}] No champions found, skipping...")
            continue

        symbol_champions = []

        for rank, (idx, weights_blob, fitness, total_profit) in enumerate(rows):
            # Determine filename
            if top_n == 1:
                filename = f"champion_{symbol}.pth"
            else:
                filename = f"champion_{symbol}_rank{rank+1}.pth"

            filepath = output_path / filename

            # Write weights blob directly to .pth file
            with open(filepath, "wb") as f:
                f.write(weights_blob)

            # Verify we can load it
            try:
                state_dict = torch.load(filepath, map_location="cpu")
                # Extract model info from state_dict
                input_size = state_dict['lstm.weight_ih_l0'].shape[1]
                hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates

                champion_info = {
                    "filename": filename,
                    "rank": rank + 1,
                    "original_index": idx,
                    "fitness": fitness,
                    "total_profit": total_profit,
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "verified": True
                }

                print(f"  [{symbol}] Rank #{rank+1}: Fitness={fitness:.4f}, Profit=${total_profit:.2f} -> {filename}")

            except Exception as e:
                champion_info = {
                    "filename": filename,
                    "rank": rank + 1,
                    "original_index": idx,
                    "fitness": fitness,
                    "total_profit": total_profit,
                    "verified": False,
                    "error": str(e)
                }
                print(f"  [{symbol}] WARNING: Could not verify {filename}: {e}")

            symbol_champions.append(champion_info)
            total_exported += 1

        manifest["champions"][symbol] = symbol_champions

    # Save manifest
    manifest_path = output_path / "champions_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Total champions exported: {total_exported}")
    print(f"  Output directory: {output_path.absolute()}")
    print(f"  Manifest: {manifest_path}")
    print(f"\nFiles ready for VPS deployment:")
    for f in sorted(output_path.glob("*.pth")):
        print(f"    {f.name}")

    conn.close()
    return True


def verify_champions(output_dir: str):
    """
    Verify exported champions can be loaded correctly.
    """
    output_path = Path(output_dir)
    manifest_path = output_path / "champions_manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Verifying {len(manifest['champions'])} symbols...\n")

    all_ok = True
    for symbol, champions in manifest["champions"].items():
        for champ in champions:
            filepath = output_path / champ["filename"]
            try:
                state_dict = torch.load(filepath, map_location="cpu")
                # Quick sanity check
                assert 'lstm.weight_ih_l0' in state_dict, "Missing LSTM weights"
                print(f"  [OK] {champ['filename']}")
            except Exception as e:
                print(f"  [FAIL] {champ['filename']}: {e}")
                all_ok = False

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ETARE Redux Champions")
    parser.add_argument("--db", default="etare_redux_v2.db", help="Path to training database")
    parser.add_argument("--output-dir", default="./champions", help="Output directory")
    parser.add_argument("--top-n", type=int, default=1, help="Export top N champions per symbol")
    parser.add_argument("--verify", action="store_true", help="Verify existing exports")

    args = parser.parse_args()

    if args.verify:
        verify_champions(args.output_dir)
    else:
        export_champions(args.db, args.output_dir, args.top_n)
