def run_full_training_schedule():
    symbols = ["XAUUSD", "EURUSD", "ETHUSD", "BTCUSD"]  # Your desired symbols
    
    print(f"[{datetime.now()}] Starting full training schedule for {len(symbols)} symbols")
    print(f"Timeframes will be processed via existing run_cycle_logic() method")
    
    for symbol in symbols:
        print(f"\n[{datetime.now()}] Processing symbol: {symbol}")
        
        try:
            system = ETARE_System()
            
            # Check if this symbol is already fully trained (based on your DB)
            cursor = system.conn.execute("SELECT MAX(batch) FROM training_log WHERE symbol = ?", (symbol,))
            last_batch = cursor.fetchone()[0] or 0
            if last_batch >= 10:
                print(f"[{datetime.now()}] {symbol} already completed all 10 batches. Skipping.")
                continue
            
            # Temporarily override SYMBOLS to train only this one
            original_symbols = system.SYMBOLS
            system.SYMBOLS = [symbol]
            
            system.run()  # Runs your existing logic for this single symbol
            
            # Restore original SYMBOLS (safety)
            system.SYMBOLS = original_symbols
            
            print(f"[{datetime.now()}] Completed training for {symbol}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
        
        time.sleep(30)  # Pause between symbols
    
    print(f"\n[{datetime.now()}] Full schedule complete for all symbols.")