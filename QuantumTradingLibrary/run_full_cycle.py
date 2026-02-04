# run_full_cycle.py - Non-interactive full training cycle
# Runs all steps without prompts

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_trader_quantum_compression import (
    load_mt5_data,
    QuantumProcessor,
    train_catboost_model,
    generate_hybrid_dataset,
    save_dataset,
    finetune_llm_with_compression,
    backtest,
    N_QUBITS, N_SHOTS, FINETUNE_SAMPLES, MODEL_NAME
)

def main():
    print("=" * 80)
    print("FULL CYCLE: QUANTUM + COMPRESSION FUSION (Non-Interactive)")
    print("=" * 80)

    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1/6: LOADING MT5 DATA")
    print("=" * 80)
    data = load_mt5_data(180)
    if not data:
        print("Failed to load data - is MT5 running?")
        return

    # Step 2-3: Train CatBoost
    print("\n" + "=" * 80)
    print("STEP 2-3/6: QUANTUM PROCESSING + CATBOOST TRAINING")
    print("=" * 80)
    quantum_processor = QuantumProcessor()
    model = train_catboost_model(data, quantum_processor)

    if model is None:
        print("CatBoost training failed")
        return

    # Step 4: Generate dataset
    print("\n" + "=" * 80)
    print("STEP 4/6: GENERATING HYBRID DATASET")
    print("=" * 80)
    dataset = generate_hybrid_dataset(data, model, quantum_processor, FINETUNE_SAMPLES)
    dataset_path = save_dataset(dataset, "dataset/quantum_compression_data.jsonl")

    # Step 5: Finetune LLM
    print("\n" + "=" * 80)
    print("STEP 5/6: FINETUNE LLM")
    print("=" * 80)
    finetune_llm_with_compression(dataset_path)

    # Step 6: Backtest
    print("\n" + "=" * 80)
    print("STEP 6/6: RUNNING BACKTEST")
    print("=" * 80)
    backtest()

    print("\n" + "=" * 80)
    print("FULL CYCLE COMPLETE!")
    print("=" * 80)
    print(f"CatBoost model: models/catboost_quantum_compression.cbm")
    print(f"LLM model: {MODEL_NAME}")
    print(f"Dataset: {dataset_path}")
    print("\nCheck the backtest results above for win rate!")

if __name__ == "__main__":
    main()
