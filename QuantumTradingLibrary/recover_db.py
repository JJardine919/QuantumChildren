
import sqlite3
import json
import sys

db_path = "trading_history.db"

def inspect_and_fix():
    print(f"Opening {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check current columns
    cursor.execute("PRAGMA table_info(population)")
    columns = [info[1] for info in cursor.fetchall()]
    print(f"Current columns: {columns}")
    
    # Columns needed by the new code
    needed_cols = {
        "fitness": "REAL DEFAULT 0",
        "successful_trades": "INTEGER DEFAULT 0",
        "total_trades": "INTEGER DEFAULT 0"
    }
    
    # Add missing columns
    for col, definition in needed_cols.items():
        if col not in columns:
            print(f"Adding missing column: {col}...")
            try:
                cursor.execute(f"ALTER TABLE population ADD COLUMN {col} {definition}")
                print(f"  ✓ Added {col}")
            except Exception as e:
                print(f"  ❌ Failed to add {col}: {e}")
                
    # Recover data from 'individual' JSON blob
    print("\nAttempting to recover fitness data from JSON...")
    cursor.execute("SELECT id, individual FROM population")
    rows = cursor.fetchall()
    
    recovered_count = 0
    for row_id, individual_json in rows:
        try:
            data = json.loads(individual_json)
            # Look for fitness in the JSON structure
            fitness = data.get('fitness', 0)
            success = data.get('successful_trades', 0)
            total = data.get('total_trades', 0)
            
            # Update the separate columns
            cursor.execute("""
                UPDATE population 
                SET fitness = ?, successful_trades = ?, total_trades = ? 
                WHERE id = ?
            """, (fitness, success, total, row_id))
            recovered_count += 1
        except Exception as e:
            print(f"Error parsing row {row_id}: {e}")
            
    print(f"\n✓ Recovered data for {recovered_count} strategies.")
    
    conn.commit()
    conn.close()
    print("\nDatabase repair complete. You can now restart the main script.")

if __name__ == "__main__":
    inspect_and_fix()
