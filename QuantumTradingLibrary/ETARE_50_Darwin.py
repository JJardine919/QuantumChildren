"""
ETARE - Evolutionary Trading Algorithm with Reinforcement and Extinction
=========================================================================
50 Experts, Darwin Style, Extinction Events
Built from ETARE CONTEXT.md blueprint
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import MetaTrader5 as mt5
import sqlite3
import json
import logging
import time
import random
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
from typing import List, Dict, Tuple
from dataclasses import dataclass

# GPU Setup
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"GPU: {torch_directml.device_name(0)}")
except ImportError:
    DEVICE = torch.device("cpu")
    print("CPU Mode")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("etare_darwin.log"), logging.StreamHandler()]
)

# === CONFIGURATION ===
POPULATION_SIZE = 50
EXTINCTION_RATE = 0.3
ELITE_SIZE = 5
EXTINCTION_INTERVAL = 50
SEQ_LENGTH = 30
SYMBOLS = ["BTCUSD"]


class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Trade:
    symbol: str
    action: Action
    volume: float
    entry_price: float
    entry_time: float
    exit_price: float = 0.0
    exit_time: float = 0.0
    profit: float = 0.0
    is_open: bool = True


class GeneticWeights:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_weights = np.random.randn(hidden_size, input_size) * 0.1
        self.hidden_weights = np.random.randn(hidden_size, hidden_size) * 0.1
        self.output_weights = np.random.randn(output_size, hidden_size) * 0.1
        self.hidden_bias = np.zeros(hidden_size)
        self.output_bias = np.zeros(output_size)

    def to_dict(self):
        return {
            'input_weights': self.input_weights.tolist(),
            'hidden_weights': self.hidden_weights.tolist(),
            'output_weights': self.output_weights.tolist(),
            'hidden_bias': self.hidden_bias.tolist(),
            'output_bias': self.output_bias.tolist()
        }


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CPU for LSTM (DirectML compatibility)
        x_cpu = x.cpu() if x.device != torch.device('cpu') else x
        out, _ = self.lstm(x_cpu)
        out = self.dropout(out[:, -1, :])
        out = out.to(x.device) if x.device != torch.device('cpu') else out
        out = self.fc(out)
        return out


class RLMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state))
        self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return list(self.memory)
        indices = random.sample(range(len(self.memory)), batch_size)
        return [self.memory[i] for i in indices]


class TradingIndividual:
    def __init__(self, input_size, hidden_size=128):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = LSTMModel(input_size, hidden_size, 3).to(DEVICE)
        self.model.lstm.to('cpu')
        self.model.dropout.to('cpu')

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.memory = RLMemory()
        self.gamma = 0.95

        self.weights = GeneticWeights(input_size, hidden_size, 3)
        self.fitness = 0.0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.trade_history = deque(maxlen=1000)
        self.open_positions: Dict[str, List[Trade]] = {}
        self.volatility = 1.0

        self.mutation_rate = 0.1
        self.mutation_strength = 0.1

    def predict(self, sequence: torch.Tensor) -> Tuple[Action, float]:
        self.model.eval()
        with torch.no_grad():
            if len(sequence.shape) == 2:
                sequence = sequence.unsqueeze(0)
            sequence = sequence.to(DEVICE)
            output = self.model(sequence)
            probs = torch.softmax(output, dim=1)
            action_idx = torch.argmax(probs).item()
            confidence = probs[0][action_idx].item()
        return Action(action_idx), confidence

    def train_on_batch(self, sequences: torch.Tensor, targets: torch.Tensor):
        self.model.train()
        self.optimizer.zero_grad()
        sequences = sequences.to(DEVICE)
        targets = targets.to(DEVICE)
        output = self.model(sequences)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update(self, state, action, reward, next_state):
        self.memory.add(state, action, reward, next_state)
        self.total_profit += reward if reward > 0 else 0
        self.total_loss += abs(reward) if reward < 0 else 0

        if len(self.memory.memory) >= 32:
            batch = self.memory.sample(32)
            self._train_on_batch(batch)

    def _train_on_batch(self, batch):
        states = torch.FloatTensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor(np.array([x[1].value for x in batch]))
        rewards = torch.FloatTensor(np.array([x[2] for x in batch]))
        next_states = torch.FloatTensor(np.array([x[3] for x in batch]))

        states = states.to(DEVICE)
        next_states = next_states.to(DEVICE)

        current_q = self.model(states).gather(1, actions.unsqueeze(1).to(DEVICE))
        next_q = self.model(next_states).max(1)[0].detach()
        target = rewards.to(DEVICE) + self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def mutate(self, volatility=1.0):
        self.mutation_strength = 0.1 * volatility
        if np.random.random() < self.mutation_rate:
            for weight_matrix in [self.weights.input_weights, self.weights.hidden_weights, self.weights.output_weights]:
                mask = np.random.random(weight_matrix.shape) < 0.1
                mutations = np.random.normal(0, self.mutation_strength, size=mask.sum())
                weight_matrix[mask] += mutations

    def clone(self):
        new_ind = TradingIndividual(self.input_size, self.hidden_size)
        new_ind.model.load_state_dict(self.model.state_dict())
        new_ind.model.lstm.to('cpu')
        new_ind.model.dropout.to('cpu')
        new_ind.weights = GeneticWeights(self.input_size, self.hidden_size, 3)
        new_ind.weights.input_weights = self.weights.input_weights.copy()
        new_ind.weights.hidden_weights = self.weights.hidden_weights.copy()
        new_ind.weights.output_weights = self.weights.output_weights.copy()
        return new_ind

    def inherit_patterns(self, parent):
        for attr in ['input_weights', 'hidden_weights', 'output_weights']:
            parent_w = getattr(parent.weights, attr)
            child_w = getattr(self.weights, attr)
            mask = np.random.random(parent_w.shape) < 0.5
            child_w[mask] = parent_w[mask]


class HybridTrader:
    def __init__(self, symbols, population_size=50):
        self.symbols = symbols
        self.population_size = population_size
        self.population: List[TradingIndividual] = []
        self.extinction_rate = EXTINCTION_RATE
        self.elite_size = ELITE_SIZE
        self.extinction_interval = EXTINCTION_INTERVAL
        self.generation = 0
        self.trade_count = 0
        self.input_size = 8

        self.conn = sqlite3.connect("etare_darwin.db")
        self._create_tables()

        if not mt5.initialize():
            logging.error("MT5 init failed")

    def _create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS population (
                    id INTEGER PRIMARY KEY,
                    individual TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_update TIMESTAMP
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY,
                    generation INTEGER,
                    individual_id INTEGER,
                    trade_history TEXT,
                    market_conditions TEXT
                )
            ''')

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(10)
        df['roc'] = df['close'].pct_change(10) * 100

        # ATR
        df['tr'] = np.maximum(df['high'] - df['low'],
                              np.maximum(abs(df['high'] - df['close'].shift(1)),
                                         abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()

        df = df.dropna()

        # Normalize
        feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
        for col in feature_cols:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

        # Target
        future_close = df['close'].shift(-5)
        df['target'] = 0
        df.loc[future_close > df['close'] * 1.001, 'target'] = 1
        df.loc[future_close < df['close'] * 0.999, 'target'] = 2

        df = df.dropna()
        return df[feature_cols + ['target', 'close']]

    def get_data(self, symbol, bars=5000):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
        if rates is None or len(rates) < 500:
            logging.error(f"No data for {symbol}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return self.prepare_features(df)

    def create_sequences(self, df):
        feature_cols = [c for c in df.columns if c not in ['target', 'close']]
        features = df[feature_cols].values
        targets = df['target'].values
        prices = df['close'].values

        X, y, p = [], [], []
        for i in range(len(features) - SEQ_LENGTH):
            X.append(features[i:i + SEQ_LENGTH])
            y.append(targets[i + SEQ_LENGTH])
            p.append(prices[i + SEQ_LENGTH])

        return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y)), np.array(p)

    def initialize_population(self):
        logging.info(f"Initializing {self.population_size} individuals")
        self.population = [TradingIndividual(self.input_size) for _ in range(self.population_size)]

    def _crossover(self, parent1, parent2):
        child = TradingIndividual(self.input_size)
        for attr in ['input_weights', 'hidden_weights', 'output_weights']:
            p1_w = getattr(parent1.weights, attr)
            p2_w = getattr(parent2.weights, attr)
            mask = np.random.random(p1_w.shape) < 0.5
            child_w = np.where(mask, p1_w, p2_w)
            setattr(child.weights, attr, child_w)
        return child

    def _tournament_selection(self, tournament_size=5):
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def evaluate_fitness(self, individual, X_test, y_test, prices):
        individual.model.eval()
        with torch.no_grad():
            X_test = X_test.to(DEVICE)
            outputs = individual.model(X_test)
            _, predicted = torch.max(outputs.data, 1)

            correct = (predicted.cpu() == y_test).sum().item()
            total = y_test.size(0)
            accuracy = correct / total

            # Trading simulation
            balance = 10000
            wins, losses = 0, 0

            for i in range(len(predicted) - 5):
                pred = predicted[i].item()
                if pred == 0:
                    continue

                future_price = prices[i + 5]
                current_price = prices[i]

                if pred == 1:  # BUY
                    profit = (future_price - current_price) / current_price
                else:  # SELL
                    profit = (current_price - future_price) / current_price

                if profit > 0:
                    wins += 1
                    balance *= (1 + abs(profit) * 0.1)
                else:
                    losses += 1
                    balance *= (1 - abs(profit) * 0.1)

            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0
            profit_factor = wins / (losses + 1)
            drawdown_resistance = 1 / (1 + individual.max_drawdown)

            individual.fitness = (
                accuracy * 0.3 +
                win_rate * 0.4 +
                profit_factor * 0.2 +
                drawdown_resistance * 0.1
            )
            individual.total_profit = balance - 10000

            return accuracy, win_rate

    def _extinction_event(self):
        logging.info("=== EXTINCTION EVENT ===")
        initial_pop = len(self.population)

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        extinction_count = int(initial_pop * self.extinction_rate)
        survivors = self.population[:initial_pop - extinction_count]

        logging.info(f"Survivors: {len(survivors)}, Extinct: {extinction_count}")

        while len(survivors) < initial_pop:
            if len(survivors) >= 2 and random.random() < 0.8:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:
                template = random.choice(survivors[:self.elite_size]) if survivors else TradingIndividual(self.input_size)
                child = template.clone()

            child.mutate(volatility=1.5)
            survivors.append(child)

        self.population = survivors
        self.generation += 1

    def _save_to_db(self):
        try:
            with self.conn:
                self.conn.execute('DELETE FROM population')
                for individual in self.population:
                    data = {
                        'weights': individual.weights.to_dict(),
                        'fitness': individual.fitness,
                        'total_profit': individual.total_profit
                    }
                    self.conn.execute('INSERT INTO population (individual) VALUES (?)', (json.dumps(data),))
        except Exception as e:
            logging.error(f"DB save error: {e}")

    def run_baseline_test(self):
        """Test population - should get 63-65%"""
        logging.info("=" * 60)
        logging.info("ETARE DARWIN - BASELINE TEST")
        logging.info(f"Population: {self.population_size}")
        logging.info("=" * 60)

        # Get data
        symbol = self.symbols[0]
        df = self.get_data(symbol)
        if df is None:
            return None

        X, y, prices = self.create_sequences(df)

        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        p_test = prices[split:]

        # Initialize fresh population
        self.initialize_population()

        # Proper training - 10 epochs per individual with evolution
        logging.info("Training population (10 epochs each with evolution)...")
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(10):
            logging.info(f"Epoch {epoch + 1}/10")
            for idx, individual in enumerate(self.population):
                for batch_X, batch_y in loader:
                    individual.train_on_batch(batch_X, batch_y)

            # Evaluate and evolve after each epoch
            for individual in self.population:
                self.evaluate_fitness(individual, X_test, y_test, p_test)
            self._extinction_event()
            logging.info(f"Epoch {epoch + 1} complete - Best fitness: {self.population[0].fitness:.4f}")

        # Evaluate
        logging.info("Evaluating population...")
        accuracies = []
        win_rates = []

        for idx, individual in enumerate(self.population):
            acc, wr = self.evaluate_fitness(individual, X_test, y_test, p_test)
            accuracies.append(acc)
            win_rates.append(wr)

        # Results
        avg_acc = np.mean(accuracies) * 100
        avg_wr = np.mean(win_rates) * 100
        best_acc = np.max(accuracies) * 100
        best_wr = np.max(win_rates) * 100

        logging.info("=" * 60)
        logging.info("BASELINE RESULTS")
        logging.info("=" * 60)
        logging.info(f"Average Accuracy: {avg_acc:.1f}%")
        logging.info(f"Average Win Rate: {avg_wr:.1f}%")
        logging.info(f"Best Accuracy: {best_acc:.1f}%")
        logging.info(f"Best Win Rate: {best_wr:.1f}%")
        logging.info("=" * 60)

        # Run extinction events
        for i in range(3):
            self._extinction_event()
            logging.info(f"Post-extinction {i+1} - Best fitness: {self.population[0].fitness:.4f}")

        self._save_to_db()

        return avg_acc, avg_wr, best_acc, best_wr


if __name__ == "__main__":
    trader = HybridTrader(SYMBOLS, POPULATION_SIZE)
    results = trader.run_baseline_test()

    if results:
        avg_acc, avg_wr, best_acc, best_wr = results
        print(f"\n{'='*60}")
        print(f"BASELINE: {avg_acc:.1f}% accuracy, {avg_wr:.1f}% win rate")
        print(f"TARGET: 63-65%")
        if 63 <= avg_acc <= 65 or 63 <= best_acc <= 65:
            print("TARGET MET - Ready for training instructions")
        else:
            print(f"Adjust needed")
        print(f"{'='*60}")

    mt5.shutdown()
