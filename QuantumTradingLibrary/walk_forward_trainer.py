import logging
from datetime import datetime
import pandas as pd
import numpy as np
import copy
import os
import json

from data_pipeline import fetch_data as fetch_mt5_data, process_data_for_model
from etare_module import HybridTrader
from backtest_engine import run_backtest
from binance_data_fetcher import fetch_binance_data

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s][%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

SYMBOLS = ["BTCUSD"]
TIMEFRAMES = ["M1", "M5", "M15"] # M1/M5 from Binance, M15 from MT5
TOTAL_MONTHS_LOOKBACK = 60
NUM_ROUNDS = 10
TRAIN_MONTHS = 4
TEST_MONTHS = 2
CHAOS_LEVEL = 0.8 
DEFAULT_VOLATILITY_FOR_MUTATION = 0.02 

RESULTS_DIR = "walk_forward_results"

def apply_evolution(trader: HybridTrader):
    """
    Performs an improved evolutionary mechanism using tournament selection and crossover.
    """
    logging.info("Applying evolutionary pressure using crossover and mutation...")
    
    if not trader.population or len(trader.population) <= 5:
        logging.warning("Population too small to apply evolution. Skipping.")
        return trader

    trader.population.sort(key=lambda x: x.fitness, reverse=True)
    
    survivors = [ind for ind in trader.population if ind.fitness >= 0]
    
    if not survivors or len(survivors) < 2: # Need at least 2 to breed
        logging.warning("Not enough successful individuals to breed from. Re-initializing population.")
        trader._initialize_population()
        return trader
        
    logging.info(f"Top {len(survivors)} experts survived. Top fitness: {survivors[0].fitness:.4f}. Breeding new generation...")
    
    num_to_replace = trader.population_size - len(survivors)
    
    new_generation = []
    for _ in range(num_to_replace):
        parent1 = trader._tournament_selection(k=3)
        parent2 = trader._tournament_selection(k=3)
        child = trader._crossover(parent1, parent2)
        child.mutate(volatility=DEFAULT_VOLATILITY_FOR_MUTATION)
        new_generation.append(child)
    
    trader.population = survivors + new_generation
    logging.info(f"Evolution complete. New population size: {len(trader.population)}")
    return trader

def generate_final_report():
    """
    Aggregates all individual cycle reports and prints a final summary.
    """
    logging.info("\n=== Generating Final Walk-Forward Report ===")
    all_results = []
    if not os.path.exists(RESULTS_DIR):
        logging.warning("Results directory not found. No final report to generate.")
        return

    for filename in sorted(os.listdir(RESULTS_DIR)):
        if filename.endswith(".json"):
            filepath = os.path.join(RESULTS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    all_results.append(json.load(f))
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {filepath}: {e}")
                continue

    if not all_results:
        logging.info("No individual cycle reports found to aggregate.")
        return

    print("\n--- Consolidated Walk-Forward Performance Summary ---")
    print(f"{'Symbol':<10} {'Timeframe':<10} {'Survived':<10} {'Avg Fitness':<15} {'Avg Profit':<15} {'Avg Win Rate':<15} {'Total Trades':<15}")
    print("-" * 100)

    for result in all_results:
        symbol = result.get('symbol', 'N/A')
        timeframe = result.get('timeframe', 'N/A')
        survived = result.get('survived_count', 0)
        avg_fitness = result.get('avg_fitness', 0.0)
        avg_profit = result.get('avg_profit', 0.0)
        avg_win_rate = result.get('avg_win_rate', 0.0)
        total_trades = result.get('total_trades', 0)
        print(f"{symbol:<10} {timeframe:<10} {survived:<10} {avg_fitness:<15.2f} {avg_profit:<15.2f} {avg_win_rate:<15.2%} {total_trades:<15}")
    print("-" * 100)
    logging.info("Final report generation complete.")

def run_walk_forward():
    """
    Main function to orchestrate the walk-forward training and testing process.
    """
    logging.info("Process Starting... Please wait for initial data fetching to complete.")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    end_of_walk_forward = datetime.now()
    start_of_walk_forward = end_of_walk_forward - pd.DateOffset(months=TOTAL_MONTHS_LOOKBACK)
    
    logging.info("=== Starting Walk-Forward Analysis ===")
    logging.info(f"Period: {start_of_walk_forward.date()} to {end_of_walk_forward.date()}")
    logging.info(f"Symbols: {SYMBOLS}, Timeframes: {TIMEFRAMES}, Chaos: {CHAOS_LEVEL}")
    logging.info("========================================")

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            logging.info(f"--- Starting Cycle: {symbol} on {timeframe} ---")
            trader = HybridTrader(symbols=[symbol])
            
            for i in range(NUM_ROUNDS):
                logging.info(f"--- Round {i+1}/{NUM_ROUNDS} for {symbol}/{timeframe} ---")
                
                round_end_date = start_of_walk_forward + pd.DateOffset(months=(i + 1) * (TRAIN_MONTHS + TEST_MONTHS))
                train_end_date = round_end_date - pd.DateOffset(months=TEST_MONTHS)
                train_start_date = train_end_date - pd.DateOffset(months=TRAIN_MONTHS)
                
                # --- DYNAMIC DATA FETCHING ---
                if timeframe in ['M1', 'M5']:
                    logging.info(f"Using Binance for {timeframe} data.")
                    binance_tf = '1m' if timeframe == 'M1' else '5m'
                    raw_train_data = fetch_binance_data(symbol, binance_tf, train_start_date, train_end_date)
                else:
                    logging.info(f"Using MetaTrader 5 for {timeframe} data.")
                    raw_train_data = fetch_mt5_data(symbol, timeframe, train_start_date, train_end_date)

                if raw_train_data is None or raw_train_data.empty:
                    logging.warning("Skipping round due to missing training data.")
                    continue
                
                processed_train_data, train_labels = process_data_for_model(raw_train_data)
                if processed_train_data is None or train_labels is None or len(processed_train_data) == 0:
                    logging.warning("Skipping round due to data processing error.")
                    continue
                
                try:
                    trader.train(processed_train_data, train_labels)
                except RuntimeError as e:
                    logging.error(f"TRAINING HALTED: {e}")
                    logging.error("Please fix your local GPU environment (PyTorch + ROCm) and try again.")
                    return # Stop the entire script if training fails

                # Fetch testing data from the same source
                if timeframe in ['M1', 'M5']:
                    binance_tf = '1m' if timeframe == 'M1' else '5m'
                    raw_test_data = fetch_binance_data(symbol, binance_tf, train_end_date, round_end_date)
                else:
                    raw_test_data = fetch_mt5_data(symbol, timeframe, train_end_date, round_end_date)
                
                if raw_test_data is None or raw_test_data.empty:
                    logging.warning("Skipping backtest due to missing testing data. Skipping round.")
                    continue

                processed_test_data, _ = process_data_for_model(raw_test_data)
                if processed_test_data is None or len(processed_test_data) == 0:
                    logging.warning("Skipping backtest due to data processing error.")
                    continue

                trader = run_backtest(trader, processed_test_data, raw_test_data, CHAOS_LEVEL)
                trader = apply_evolution(trader)
                
            logging.info(f"--- Cycle for {symbol}/{timeframe} finished. ---")

            if trader.population:
                passed_individuals = [ind for ind in trader.population if ind.fitness >= 0]
                cycle_results = {
                    "symbol": symbol, "timeframe": timeframe, "survived_count": len(passed_individuals),
                    "avg_fitness": np.mean([ind.fitness for ind in passed_individuals]) if passed_individuals else 0,
                    "avg_profit": np.mean([ind.total_profit for ind in passed_individuals]) if passed_individuals else 0,
                    "avg_win_rate": np.mean([ind.win_rate for ind in passed_individuals]) if passed_individuals else 0,
                    "total_trades": int(np.sum([ind.num_trades for ind in passed_individuals])) if passed_individuals else 0,
                    "timestamp": datetime.now().isoformat()
                }
                result_filename = os.path.join(RESULTS_DIR, f"results_{symbol}_{timeframe}.json")
                with open(result_filename, 'w') as f:
                    json.dump(cycle_results, f, indent=4)
                logging.info(f"Cycle results saved to {result_filename}")

    logging.info("=== Walk-Forward Analysis Complete ===")
    generate_final_report()

if __name__ == "__main__":
    run_walk_forward()
