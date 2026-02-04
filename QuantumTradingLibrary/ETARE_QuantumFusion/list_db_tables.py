
import sqlite3
import os

dbs = ['etare_redux.db', 'etare_redux_v2.db', 'trading_history.db']
for db in dbs:
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = [row[0] for row in cursor.fetchall()]
        print(f"{db} tables: {tables}")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  - {table}: {count} rows")
        conn.close()
    else:
        print(f"{db} not found.")
