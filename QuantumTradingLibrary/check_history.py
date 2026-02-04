
import sqlite3
import json

db_path = "trading_history.db"

def check_history():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check history table count
    cursor.execute("SELECT count(*) FROM history")
    count = cursor.fetchone()[0]
    print(f"Total history entries: {count}")
    
    # Check max generation
    cursor.execute("SELECT MAX(generation) FROM history")
    max_gen = cursor.fetchone()[0]
    print(f"Max Generation in history: {max_gen}")
    
    conn.close()

if __name__ == "__main__":
    check_history()
