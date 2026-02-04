
import json
import logging
import random
import time
from collections import deque, defaultdict
from enum import Enum

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# --- Enums and Data Classes ---

class Action(Enum):
    HOLD = 0
    OPEN_BUY = 1
    OPEN_SELL = 2

# --- LSTM Model ---

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# --- Trading Individual ---

class TradingIndividual:
    def __init__(self, input_size, hidden_size=64, output_size=3):
        self.input_size = input_size
        self.model = LSTMModel(input_size, hidden_size, output_size)
        self.fitness = -float('inf') # Start with worst possible fitness
        self.total_profit = 0.0
        self.win_rate = 0.0
        self.num_trades = 0
        self.mutation_rate = 0.2
        self.base_mutation_strength = 0.1
        self.min_mutation_strength = 0.01
        self.max_mutation_strength = 0.5
        self.average_volatility = 0.01

    def predict(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(state_tensor)
        action_idx = torch.argmax(prediction).item()
        return Action(action_idx), torch.max(torch.softmax(prediction, dim=1)).item()

    def mutate(self, volatility):
        mutation_strength = self._calculate_mutation_strength(volatility)
        if np.random.random() < self.mutation_rate:
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * mutation_strength
                        param.add_(noise)

    def _calculate_mutation_strength(self, volatility):
        volatility_factor = 1 + (volatility / self.average_volatility - 1)
        mutation_strength = self.base_mutation_strength * volatility_factor
        return np.clip(mutation_strength, self.min_mutation_strength, self.max_mutation_strength)

# --- Multi-GPU Training Worker ---

def train_worker(device_id, individuals_chunk, training_features, training_labels, results_queue):
    device = f'cuda:{device_id}'
    logging.info(f"Worker process started for {device} with {len(individuals_chunk)} individuals.")
    X_train_tensor = torch.from_numpy(training_features).float().to(device)
    y_train_tensor = torch.from_numpy(training_labels).long().to(device)
    trained_chunk = []
    for individual in individuals_chunk:
        try:
            individual.model.to(device)
            individual.model.optimizer = torch.optim.Adam(individual.model.parameters(), lr=0.001)
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
            individual.model.train()
            for epoch in range(5):
                for features, labels in train_loader:
                    individual.model.optimizer.zero_grad()
                    outputs = individual.model(features)
                    loss = individual.model.criterion(outputs, labels)
                    loss.backward()
                    individual.model.optimizer.step()
            individual.model.to('cpu')
            trained_chunk.append(individual)
        except Exception as e:
            logging.error(f"Error training individual on {device}: {e}", exc_info=True)
            individual.model.to('cpu')
            trained_chunk.append(individual)
    results_queue.put(trained_chunk)
    logging.info(f"Worker for {device} finished.")

# --- Hybrid Trader ---

class HybridTrader:
    def __init__(self, symbols, population_size=50, input_size=None):
        self.symbols = symbols
        self.population_size = population_size
        self.input_size = input_size
        self.population = []
        if self.input_size:
            self._initialize_population()

    def _initialize_population(self):
        self.population = [TradingIndividual(self.input_size) for _ in range(self.population_size)]
        logging.info(f"Initialized population with {self.population_size} individuals.")

    def train(self, training_features: np.ndarray, training_labels: np.ndarray):
        if not self.population:
            self.input_size = training_features.shape[1]
            self._initialize_population()
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logging.info(f"Found {num_gpus} GPUs. Distributing training across them.")
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
            results_queue = mp.Queue()
            processes = []
            chunk_size = self.population_size // num_gpus if self.population_size > num_gpus else 1
            chunks = [self.population[i:i + chunk_size] for i in range(0, self.population_size, chunk_size)]
            for i, chunk in enumerate(chunks):
                device_id = i % num_gpus
                process = mp.Process(target=train_worker, args=(device_id, chunk, training_features, training_labels, results_queue))
                processes.append(process)
                process.start()
            new_population = []
            for _ in processes:
                new_population.extend(results_queue.get())
            for process in processes:
                process.join()
            self.population = new_population
            logging.info("Multi-GPU training complete.")
        else:
            raise RuntimeError("No GPUs found. Training on CPU is disabled as per the new rule.")

    def _tournament_selection(self, k=5):
        tournament_contenders = random.sample(self.population, k)
        winner = max(tournament_contenders, key=lambda x: x.fitness)
        return winner

    def _crossover(self, parent1: TradingIndividual, parent2: TradingIndividual) -> TradingIndividual:
        child = TradingIndividual(self.input_size)
        p1_state_dict = parent1.model.state_dict()
        p2_state_dict = parent2.model.state_dict()
        child_state_dict = child.model.state_dict()
        for key in p1_state_dict:
            if random.random() < 0.5:
                child_state_dict[key] = p1_state_dict[key].clone()
            else:
                child_state_dict[key] = p2_state_dict[key].clone()
        child.model.load_state_dict(child_state_dict)
        return child

    # --- Feature Preparation ---
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        # Bollinger Bands
        volatility = df['close'].rolling(50).std()
        adaptive_period = int(20 * (1 + volatility.mean())) if not np.isnan(volatility.mean()) else 20
        df['bb_middle'] = df['close'].rolling(adaptive_period).mean()
        df['bb_std'] = df['close'].rolling(adaptive_period).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(10)
        df['momentum_ma'] = df['momentum'].rolling(20).mean()
        df['momentum_std'] = df['momentum'].rolling(20).std()
        # Volatility
        df['atr'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        df['price_change'] = df['close'].pct_change()
        # Volume Analysis
        df['volume_ma'] = df['tick_volume'].rolling(20).mean()
        df['volume_std'] = df['tick_volume'].rolling(20).std()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
        
        # Normalization
        numeric_cols = [col for col in df.columns if df[col].dtype in [np.int64, np.float64]]
        for col in numeric_cols:
            rolling_mean = df[col].rolling(100).mean()
            rolling_std = df[col].rolling(100).std()
            df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        df[numeric_cols] = df[numeric_cols].clip(-4, 4)
        df = df.fillna(0)
        return df
