
import sqlite3
import torch
import json
import os

def extract():
    db_path = 'etare_redux_v2.db'
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT weights FROM population_state ORDER BY fitness DESC LIMIT 1')
        row = cursor.fetchone()
        if row:
            # Redux weights are stored as BLOB (torch state_dict)
            with open('temp_w.pth', 'wb') as f:
                f.write(row[0])
            
            state_dict = torch.load('temp_w.pth', map_location='cpu')
            
            # The Redux model is an LSTM, whereas ETARE_module is a simple MLP
            # We might need to map them or just use fresh weights if they don't match.
            # But the ETARE_module.py is what we based our 23-input on.
            
            # Let's check if trading_history.db has ANY individuals
            # If not, we'll initialize with random weights and call it "Champion Zero"
            print("Found Redux champion weights. LSTM format.")
            
            # Check for technical weights in json format if they exist
            # ...
        else:
            print("No individuals in Redux database.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    extract()
