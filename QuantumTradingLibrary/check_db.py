import sqlite3
import pandas as pd

try:
    conn = sqlite3.connect('etare_redux.db')
    query = "SELECT * FROM training_log ORDER BY timestamp DESC LIMIT 5"
    df = pd.read_sql_query(query, conn)
    print(df)
    conn.close()
except Exception as e:
    print(f"Error reading DB: {e}")
