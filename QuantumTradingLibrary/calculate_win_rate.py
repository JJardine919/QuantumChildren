
import sqlite3
import json

def calculate_win_rate():
    db_file = 'trading_history.db'
    total_trades = 0
    winning_trades = 0

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Check for the 'population' table first, as it seems to be the primary data store
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='population';")
        if cursor.fetchone():
            cursor.execute("SELECT individual FROM population")
            rows = cursor.fetchall()
            for row in rows:
                try:
                    individual_data = json.loads(row[0])
                    if 'trade_history' in individual_data and isinstance(individual_data['trade_history'], list):
                        for trade in individual_data['trade_history']:
                            if isinstance(trade, dict) and 'profit' in trade:
                                total_trades += 1
                                if trade['profit'] > 0:
                                    winning_trades += 1
                except (json.JSONDecodeError, TypeError):
                    continue # Ignore malformed data

        # If no trades found, check 'history' table as a fallback
        if total_trades == 0:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history';")
            if cursor.fetchone():
                cursor.execute("SELECT trade_history FROM history")
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        trade_history_list = json.loads(row[0])
                        if isinstance(trade_history_list, list):
                            for trade in trade_history_list:
                                if isinstance(trade, dict) and 'profit' in trade:
                                    total_trades += 1
                                    if trade['profit'] > 0:
                                        winning_trades += 1
                    except (json.JSONDecodeError, TypeError):
                        continue # Ignore malformed data

        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            print(f"Total trades found: {total_trades}")
            print(f"Winning trades: {winning_trades}")
            print(f"Calculated Win Rate: {win_rate:.2f}%")
        else:
            print("No trades found in the database ('population' or 'history' tables).")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    calculate_win_rate()
