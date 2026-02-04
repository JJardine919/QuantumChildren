"""
ETARE QUANTUM FUSION - TRAINER
==============================
Trains the Quantum LSTM model on fresh market data.

Process:
1. Fetch 180 days of BTCUSD M5 data.
2. Generate Quantum Features (7 metrics).
3. Train Bidirectional LSTM.
4. Save best model to models/quantum_lstm_best.pth.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
import logging

# Add modules to path
sys.path.append(os.path.join(os.getcwd(), "ETARE_QuantumFusion"))
from modules.quantum_lstm_adapter import QuantumFeatureExtractor, QuantumLSTM

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Trainer")

# --- CONFIG ---
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
DAYS = 180
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

class MarketDataset(Dataset):
    def __init__(self, price_seq, quantum_feat, targets):
        self.price = torch.FloatTensor(price_seq)
        self.quantum = torch.FloatTensor(quantum_feat)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self): return len(self.targets)
    
    def __getitem__(self, idx):
        return self.price[idx], self.quantum[idx], self.targets[idx]

def fetch_data():
    if not mt5.initialize():
        logger.error("MT5 Init Failed")
        return None
        
    logger.info(f"Fetching {DAYS} days of {SYMBOL}...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 288 * DAYS) # 288 bars per day (M5)
    mt5.shutdown()
    
    if rates is None: return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calc Features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low'] = (df['high'] - df['low']) / df['close']
    df['close_open'] = (df['close'] - df['open']) / df['open']
    
    return df.dropna()

def prepare_dataset(df):
    logger.info("Generating Quantum Features (This takes time)...")
    extractor = QuantumFeatureExtractor()
    
    price_seqs = []
    quantum_feats = []
    targets = []
    
    # Sliding window
    # Input: 50 bars. Target: Next bar direction (1=Up, 0=Down)
    window_size = 50
    
    # Pre-calc tech features
    tech_data = df[['returns', 'log_returns', 'high_low', 'close_open', 'tick_volume']].values
    tech_data = (tech_data - np.mean(tech_data, axis=0)) / (np.std(tech_data, axis=0) + 1e-8)
    
    # Pre-calc raw prices for quantum
    raw_close = df['close'].values
    
    for i in range(window_size, len(df) - 1):
        # 1. Tech Seq
        seq = tech_data[i-window_size:i]
        
        # 2. Quantum Feat (Last window)
        q_window = raw_close[i-window_size:i]
        q_dict = extractor.extract(q_window)
        q_vec = [
            q_dict['entropy'], q_dict['dominant_state'], q_dict['superposition'],
            q_dict['coherence'], q_dict['entanglement'], 0.0, q_dict['significant_states']
        ]
        
        # 3. Target
        target = 1.0 if raw_close[i+1] > raw_close[i] else 0.0
        
        price_seqs.append(seq)
        quantum_feats.append(q_vec)
        targets.append(target)
        
        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)} bars...", end='\r')
            
    return np.array(price_seqs), np.array(quantum_feats), np.array(targets)

def train():
    df = fetch_data()
    if df is None: return
    
    X_price, X_quant, y = prepare_dataset(df)
    logger.info(f"Dataset ready: {len(y)} samples")
    
    dataset = MarketDataset(X_price, X_quant, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")
    
    model = QuantumLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    logger.info("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for price, quant, target in loader:
            price, quant, target = price.to(device), quant.to(device), target.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            output = model(price, quant)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == target).sum().item()
            total += target.size(0)
            
        acc = correct / total
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.4f}")
        
    # Save
    save_path = "ETARE_QuantumFusion/models/quantum_lstm_best.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()