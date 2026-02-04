
import pandas as pd
import numpy as np
import torch
import sys
import os

# Add modules directory to path
sys.path.append(os.path.abspath('ETARE_QuantumFusion/modules'))
from etare_enhanced import ETAREEnhanced, DEVICE, Action

def verify():
    # 1. Load Dataset
    data_path = "ETARE_QuantumFusion/data/fusion_training_set.csv"
    if not os.path.exists(data_path):
        return

    df = pd.read_csv(data_path)
    
    feature_cols = [
        'close', 'rsi', 'macd', 'macd_signal', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 
        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'momentum', 'atr', 'price_change', 'price_change_abs', 
        'volume_ma', 'volume_std', 'stoch_k', 'stoch_d', 'cci', 'roc', 'williams_r', 
        'ratio', 'quantum_entropy', 'fusion_score'
    ]
    
    X = df[feature_cols].values
    y_true = np.where(df['price_change'] > 0, 0, 1) # 0 for Buy, 1 for Sell
    
    # 2. Load Model
    model = ETAREEnhanced(input_size=len(feature_cols))
    model_path = "ETARE_QuantumFusion/models/fusion_champion_v1.pth"
    
    # Transfer weights manually since model is state_dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    with torch.no_grad():
        model.weights.input_weights.copy_(state_dict['input_weights'])
        model.weights.hidden_weights.copy_(state_dict['hidden_weights'])
        model.weights.output_weights.copy_(state_dict['output_weights'])
        model.weights.hidden_bias.copy_(state_dict['hidden_bias'])
        model.weights.output_bias.copy_(state_dict['output_bias'])
    
    # 3. Predict and Calculate Accuracy
    correct = 0
    total = len(df)
    
    # Turn off epsilon for verification
    model.epsilon = 0.0
    
    print("\n" + "="*40)
    print("FUSION MODEL VERIFICATION (Training Set)")
    print("="*40)
    
    for i in range(total):
        action, probs = model.predict(X[i].reshape(1, -1))
        pred_idx = action.value
        actual_idx = y_true[i]
        
        is_correct = (pred_idx == actual_idx)
        if is_correct:
            correct += 1
            
        print(f"Sample {i+1:02d}: Pred={Action(pred_idx).name}, Actual={Action(actual_idx).name} | {'✅' if is_correct else '❌'}")
        
    accuracy = (correct / total) * 100
    print("="*40)
    print(f"TOTAL ACCURACY: {accuracy:.2f}%")
    print(f"WIN RATE TARGET: 78-82%")
    print("="*40)

if __name__ == "__main__":
    verify()
