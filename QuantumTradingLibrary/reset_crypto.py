import sqlite3
db_path = 'etare_redux_v2.db'
symbols = ('BTCUSD', 'ETHUSD')

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clean training logs
    cursor.execute("DELETE FROM training_log WHERE symbol IN (?, ?)", symbols)
    log_deleted = cursor.rowcount
    
    # Clean population state
    cursor.execute("DELETE FROM population_state WHERE symbol IN (?, ?)", symbols)
    pop_deleted = cursor.rowcount
    
    conn.commit()
    print(f"Successfully reset {symbols} in {db_path}")
    print(f" - Deleted {log_deleted} training log entries.")
    print(f" - Deleted {pop_deleted} population state entries.")
    
except Exception as e:
    print(f"Error resetting symbols: {e}")
finally:
    if conn:
        conn.close()
