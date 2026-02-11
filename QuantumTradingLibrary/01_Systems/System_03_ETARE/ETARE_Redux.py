import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
from collections import deque
import random
from enum import Enum
import time
from copy import deepcopy
import sqlite3
import json
import os
import sys
import glob
from datetime import datetime, timedelta

# Add ETARE_QuantumFusion modules to path for FastQuantumExtractor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ETARE_QuantumFusion'))
from modules.quantum_lstm_adapter import FastQuantumExtractor

# Try to enable GPU via DirectML
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"ETARE_Redux: Using DirectML Device: {torch_directml.device_name(0)}")
except ImportError:
    DEVICE = torch.device("cpu")
    print("ETARE_Redux: Using CPU (torch_directml not found)")

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ETARE_Redux.log"),
        logging.StreamHandler()
    ]
)

# Constants
BATCHES = 10
MONTHS_PER_BATCH = 6
TOTAL_HISTORY_MONTHS = 60
SYMBOLS = [
    # Blue Guardian 366592 - Three pairs only
    "BTCUSD", "ETHUSD", "XAUUSD"
]
TIMEFRAME_M5 = mt5.TIMEFRAME_M5 # Default to M5 for speed
POPULATION_SIZE = 50
EXTINCTION_INTERVAL = 100 
SEQ_LENGTH = 30  # Sequence length for LSTM
PREDICTION_HORIZON = 5 # Predict price 5 steps ahead

# Deposit amounts for each of the 10 batches to prevent overfitting
DEPOSITS = [1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 125000, 150000]

class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3):
        super(LSTMModel, self).__init__()
        # Use LSTM (compatible with old saves). Will run on CPU to avoid DirectML crash.
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Hybrid execution: Run LSTM and Dropout on CPU to avoid DML backward issues
        x_cpu = x.cpu()
        out_cpu, _ = self.lstm(x_cpu)
        
        # Dropout on CPU
        out_drop = self.dropout(out_cpu[:, -1, :])
        
        # Move output back to device for FC layers
        out = out_drop.to(x.device)
        out = self.fc(out)
        return out

class TradingIndividual:
    def __init__(self, input_size: int, hidden_size: int = 128, quantum_seed=None):
        self.input_size = input_size
        self.model = LSTMModel(input_size, hidden_size).to(DEVICE)
        # DirectML Fix: Force LSTM and Dropout to CPU to avoid fused kernel crash and backward warnings
        self.model.lstm.to('cpu')
        self.model.dropout.to('cpu')
        
        # Quantum Weight Injection
        if quantum_seed is not None:
            self._inject_quantum_weights(quantum_seed)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.fitness = 0.0
        self.win_rate = 0.0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.trade_history = []
        self.open_positions = []
        
        # Evolution params
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1

    def _inject_quantum_weights(self, state_vector):
        """
        Maps the quantum state vector onto the Neural Network weights.
        This provides a 'cleaner' starting point than random initialization.
        """
        with torch.no_grad():
            # Flatten the state to use as a source of parameters
            flat_state = torch.tensor(state_vector.flatten(), dtype=torch.float32)
            
            # Inject into LSTM weights (first layer)
            # We iterate through parameters and fill them with quantum data cyclically
            param_idx = 0
            state_len = len(flat_state)
            
            for name, param in self.model.named_parameters():
                if 'lstm' in name: # Focus on the temporal learner
                    num_params = param.numel()
                    # Create a slice from the state vector (repeating if necessary)
                    if state_len >= num_params:
                        segment = flat_state[:num_params]
                    else:
                        # Repeat state to fill parameter
                        repeats = (num_params // state_len) + 1
                        segment = flat_state.repeat(repeats)[:num_params]
                    
                    # Reshape and add to existing random weights (Hybrid Init)
                    # We multiply by 0.1 to use it as a "bias" rather than overriding completely
                    segment = segment.reshape(param.shape)
                    param.data.add_(segment * 0.1) 

    def predict(self, sequence: torch.Tensor) -> Tuple[Action, float]:
        self.model.eval()
        with torch.no_grad():
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

    def mutate(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if np.random.random() < self.mutation_rate:
                    noise = torch.randn_like(param) * self.mutation_strength
                    param.add_(noise)

    def clone(self):
        # Note: We don't pass quantum_seed here because we are copying the state_dict anyway
        new_ind = TradingIndividual(self.input_size)
        new_ind.model.load_state_dict(self.model.state_dict())
        # CPU Fix
        new_ind.model.lstm.to('cpu')
        new_ind.model.dropout.to('cpu')
        return new_ind

class ETARE_System:
    def __init__(self):
        self.conn = sqlite3.connect("etare_redux_v3.db")
        self.create_tables()
        self.population: List[TradingIndividual] = []
        self.input_size = 0 # Will be set dynamically
        self.quantum_extractor = FastQuantumExtractor()

        # Initialize MT5 with terminal path fallback
        terminal_paths = [
            r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
            r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe",
        ]
        initialized = False
        for path in terminal_paths:
            if os.path.exists(path):
                if mt5.initialize(path=path):
                    initialized = True
                    logging.info(f"MT5 connected via {os.path.basename(os.path.dirname(path))}")
                    break
        if not initialized:
            if not mt5.initialize():
                logging.error("MT5 Initialization failed. Ensure MT5 is running.")

        # Quantum Injection: Load the cleanest available state
        self.quantum_state = self.load_quantum_state()

    def load_quantum_state(self):
        """Loads the latest Quantum Compressed State (.dqcp.npz) for weight initialization."""
        try:
            # Check user distribution first (usually has the 'cleanest' sample)
            paths = [
                "09_User_Distribution/04_Sample_Data/*.dqcp.npz",
                "04_Data/QuantumStates/*.dqcp.npz",
                "04_Data/Archive/*.dqcp.npz"
            ]
            
            files = []
            for p in paths:
                files.extend(glob.glob(p))
            
            if not files:
                logging.warning("No Quantum States found for injection. Using random init.")
                return None
                
            # Pick latest
            latest = max(files, key=os.path.getctime)
            logging.info(f"Quantum Injection: Loading state from {latest}")
            
            data = np.load(latest, allow_pickle=True)
            state = data['state']
            ratio = data['ratio']
            logging.info(f"Quantum State Loaded. Shape: {state.shape}, Ratio: {ratio:.2f}")
            return state
            
        except Exception as e:
            logging.error(f"Quantum Injection Failed: {e}")
            return None

    def create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS training_log (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    batch INTEGER,
                    cycle INTEGER,
                    train_period TEXT,
                    test_period TEXT,
                    best_fitness REAL,
                    deposit REAL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS population_state (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    individual_index INTEGER,
                    weights BLOB,
                    fitness REAL,
                    total_profit REAL
                )
            """)

    def save_population(self, symbol):
        with self.conn:
            self.conn.execute("DELETE FROM population_state WHERE symbol = ?", (symbol,))
            for i, ind in enumerate(self.population):
                # Serialize weights to a temporary file then read as bytes
                temp_file = os.path.join(os.environ.get('TEMP', '.'), f"weights_{symbol}_{i}.pth")
                torch.save(ind.model.state_dict(), temp_file)
                with open(temp_file, "rb") as f:
                    weights_blob = f.read()
                
                self.conn.execute("""
                    INSERT INTO population_state (symbol, individual_index, weights, fitness, total_profit)
                    VALUES (?, ?, ?, ?, ?)
                """, (symbol, i, weights_blob, ind.fitness, ind.total_profit))

    def load_population(self, symbol):
        cursor = self.conn.execute("SELECT weights, fitness, total_profit FROM population_state WHERE symbol = ?", (symbol,))
        rows = cursor.fetchall()
        if not rows:
            return False
        
        self.population = []
        for row in rows:
            weights_blob, fitness, total_profit = row
            temp_file = os.path.join(os.environ.get('TEMP', '.'), "temp_load.pth")
            with open(temp_file, "wb") as f:
                f.write(weights_blob)
            
            state_dict = torch.load(temp_file)
            input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            self.input_size = input_size
            
            ind = TradingIndividual(input_size)
            try:
                ind.model.load_state_dict(state_dict)
                # Re-apply CPU fix after loading state dict (just in case)
                ind.model.lstm.to('cpu')
            except Exception as e:
                logging.warning(f"Failed to load weights for {symbol} (Index {row[0]}). Starting fresh. Error: {e}")
                ind = TradingIndividual(input_size) # Reset
                ind.model.lstm.to('cpu')
                ind.model.dropout.to('cpu')
            
            ind.fitness = fitness
            ind.total_profit = total_profit
            self.population.append(ind)
        
        logging.info(f"Loaded population for {symbol} with {len(self.population)} individuals.")
        return True

    def get_data_chunk(self, symbol, timeframe, start_date, months):
        """
        Fetches 'months' worth of data starting from 'start_date'.
        """
        end_date = start_date + timedelta(days=30*months)
        
        rates = None
        # Try to get real data if connected
        if mt5.terminal_info() is not None:
             rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
             
             # Fallback 1: If range failed, try to pull latest history if the start_date is too old
             if rates is None or len(rates) < 500:
                  logging.info(f"Range failed for {symbol} ({start_date}). Trying recent history...")
                  rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 5000) # Pull last 5000 bars
        
        # Fallback 2: Simulated data if still nothing
        if rates is None or len(rates) < 500:
             logging.warning(f"No real data for {symbol}. Using SIMULATED data.")
             
             dates = pd.date_range(start_date, end_date, freq='5min')
             n = len(dates)
             start_price = 1.1000 if 'EUR' in symbol else (2000.0 if 'XAU' in symbol else 3000.0)
             returns = np.random.normal(0.0001, 0.002, n)
             price_path = start_price * np.exp(np.cumsum(returns))
             
             df = pd.DataFrame({
                 'time': dates,
                 'open': price_path,
                 'high': price_path * (1 + np.abs(np.random.normal(0, 0.001, n))),
                 'low': price_path * (1 - np.abs(np.random.normal(0, 0.001, n))),
                 'close': price_path * (1 + np.random.normal(0, 0.0005, n)),
                 'tick_volume': np.random.randint(100, 1000, size=n)
             })
             return self.prepare_features(df)
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return self.prepare_features(df)

    def prepare_features(self, df):
        df = df.copy()
        # Ensure numerical types
        cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for c in cols: df[c] = df[c].astype(float)

        # 1. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 2. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # 3. Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

        # 4. Momentum & ROC
        df['momentum'] = df['close'] / df['close'].shift(10)
        df['roc'] = df['close'].pct_change(10) * 100

        # 5. ATR
        df['tr'] = np.maximum(df['high'] - df['low'],
                              np.maximum(abs(df['high'] - df['close'].shift(1)),
                                         abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()

        # 6. Technical feature columns (original 8)
        tech_cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
        df = df.dropna()

        if df.empty: return None

        # 7. Quantum Features from close prices (7 features per bar)
        close_vals = df['close'].values
        q_cols = ['q_entropy', 'q_dominant', 'q_superposition', 'q_coherence', 'q_entanglement', 'q_phase_var', 'q_sig_states']
        window_size = 30  # Match SEQ_LENGTH for quantum window

        # Extract quantum features for each bar using a rolling window
        q_data = np.zeros((len(close_vals), 7))
        for i in range(len(close_vals)):
            start = max(0, i - window_size)
            window = close_vals[start:i+1]
            if len(window) >= 5:  # Need minimum data for meaningful quantum features
                metrics = self.quantum_extractor.extract(window)
                q_data[i] = [
                    metrics['entropy'], metrics['dominant_state'], metrics['superposition'],
                    metrics['coherence'], metrics['entanglement'], 0.0, metrics['significant_states']
                ]
            # else: stays as zeros (first few bars)

        for j, col in enumerate(q_cols):
            df[col] = q_data[:, j]

        # All 15 feature columns
        feature_cols = tech_cols + q_cols

        # Z-score normalization (all 15)
        for col in feature_cols:
             df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

        # Add targets for training (Buy if price goes up, Sell if down)
        future_close = df['close'].shift(-PREDICTION_HORIZON)
        df['target'] = 0 # Hold
        df.loc[future_close > df['close'], 'target'] = 1 # Buy
        df.loc[future_close < df['close'], 'target'] = 2 # Sell

        df = df.dropna() # Drop last rows with no target

        # Return features and targets
        return df[feature_cols + ['target', 'close']]

    def create_sequences(self, data_df):
        # Convert DF to Tensor sequences
        feature_cols = [c for c in data_df.columns if c not in ['target', 'close']]
        features = data_df[feature_cols].values
        targets = data_df['target'].values
        prices = data_df['close'].values
        
        X, y, p = [], [], []
        for i in range(len(features) - SEQ_LENGTH):
            X.append(features[i:i+SEQ_LENGTH])
            y.append(targets[i+SEQ_LENGTH]) # Target at end of sequence
            p.append(prices[i+SEQ_LENGTH])
            
        return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y)), np.array(p)

    def run_cycle_logic(self, symbol):
        # 60 months, 10 batches, 3 cycles logic
        now = datetime.now()
        start_history = now - timedelta(days=30*60)
        
        logging.info(f"Starting Training for {symbol} from {start_history}")

        for batch_idx in range(BATCHES):
            # Check if batch already done
            cursor = self.conn.execute("SELECT COUNT(*) FROM training_log WHERE symbol = ? AND batch = ?", (symbol, batch_idx+1))
            if cursor.fetchone()[0] >= 3: # 3 cycles per batch
                logging.info(f"Batch {batch_idx+1} for {symbol} already done. Skipping.")
                continue

            deposit = DEPOSITS[batch_idx]
            batch_start = start_history + timedelta(days=30 * MONTHS_PER_BATCH * batch_idx)
            logging.info(f"Processing Batch {batch_idx+1}/{BATCHES} (Start: {batch_start.date()}, Deposit: ${deposit})")
            
            # Cycle 1: M1, Train 4, Test 2
            self.execute_cycle(symbol, mt5.TIMEFRAME_M1, batch_start, batch_idx, cycle_num=1, 
                               train_months=[0,1,2,3], test_months=[4,5], deposit=deposit)
                               
            # Cycle 2: M5, Train 2, Test 2, Train 2 (Jostle Middle)
            self.execute_cycle(symbol, mt5.TIMEFRAME_M5, batch_start, batch_idx, cycle_num=2, 
                               train_months=[0,1, 4,5], test_months=[2,3], deposit=deposit)
            
            # Cycle 3: M15, Test 2, Train 4 (Jostle Start)
            self.execute_cycle(symbol, mt5.TIMEFRAME_M15, batch_start, batch_idx, cycle_num=3, 
                               train_months=[2,3,4,5], test_months=[0,1], deposit=deposit)

    def execute_cycle(self, symbol, timeframe, batch_start_date, batch_idx, cycle_num, train_months, test_months, deposit):
        logging.info(f"  Cycle {cycle_num}: TF={timeframe}, Train={train_months}, Test={test_months}")
        
        # --- Training Data Collection ---
        train_dfs = []
        for m in train_months:
            m_start = batch_start_date + timedelta(days=30*m)
            chunk = self.get_data_chunk(symbol, timeframe, m_start, 1)
            if chunk is not None:
                train_dfs.append(chunk)
        
        # --- Testing Data Collection ---
        test_dfs = []
        for m in test_months:
            m_start = batch_start_date + timedelta(days=30*m)
            chunk = self.get_data_chunk(symbol, timeframe, m_start, 1)
            if chunk is not None:
                test_dfs.append(chunk)

        if not train_dfs or not test_dfs:
            logging.warning("Insufficient data for this cycle.")
            return

        full_train_df = pd.concat(train_dfs)
        full_test_df = pd.concat(test_dfs)
        
        # Set input size based on data
        self.input_size = len(full_train_df.columns) - 2 # Exclude target and close
        
        # Initialize or load population
        if not self.population:
            if not self.load_population(symbol):
                logging.info(f"Starting fresh population for {symbol}")
                # Pass self.quantum_state to init
                self.population = [TradingIndividual(self.input_size, quantum_seed=self.quantum_state) for _ in range(POPULATION_SIZE)]

        # --- Execution ---
        self.train_population(full_train_df)
        self.evaluate_population(full_test_df, deposit)
        self.evolve_population()
        
        # Save after each cycle
        self.save_population(symbol)
        with self.conn:
            self.conn.execute("""
                INSERT INTO training_log (symbol, batch, cycle, train_period, test_period, best_fitness, deposit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, batch_idx+1, cycle_num, str(train_months), str(test_months), self.population[0].fitness, deposit))

    def train_population(self, df):
        X, y, _ = self.create_sequences(df)
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train each individual
        for individual in self.population:
            for batch_X, batch_y in loader:
                individual.train_on_batch(batch_X, batch_y)

    def evaluate_population(self, df, deposit):
        X, y, prices = self.create_sequences(df)
        # Move X to device for batch inference if possible, but watch out for memory on large sets
        # If X is too large, we might need to batch this too. For now assuming it fits.
        X_device = X.to(DEVICE)
        
        for individual in self.population:
            individual.model.eval()
            with torch.no_grad():
                outputs = individual.model(X_device)
                _, predicted = torch.max(outputs.data, 1)
                
                # Trading simulation to calculate fitness based on profit
                balance = deposit
                positions = 0
                entry_price = 0
                
                # Metrics for Sniper Logic
                wins = 0
                losses = 0
                
                for i in range(len(predicted)):
                    action = Action(predicted[i].item())
                    current_price = prices[i]
                    
                    # Check Stop Loss (0.5% drawdown)
                    if positions > 0:
                        # For BUY positions
                        drawdown = (current_price - entry_price) / entry_price
                        if drawdown <= -0.005: # 0.5% SL
                            balance = positions * current_price
                            positions = 0
                            entry_price = 0
                            losses += 1
                            continue

                    if action == Action.BUY and positions == 0:
                        positions = balance / current_price
                        entry_price = current_price
                        balance = 0
                    elif action == Action.SELL and positions > 0:
                        exit_val = positions * current_price
                        # Check if profitable trade (accounting for spread/commission approx via raw price diff)
                        # We use simple price delta here.
                        if current_price > entry_price:
                            wins += 1
                        else:
                            losses += 1
                            
                        balance = exit_val
                        positions = 0
                        entry_price = 0
                    elif action == Action.HOLD and positions > 0:
                        # Optional: check trailing stop or take profit
                        pass
                
                # Final balance including open positions
                final_value = balance + (positions * prices[-1] if positions > 0 else 0)
                roi = (final_value - deposit) / deposit
                
                # --- SNIPER LOGIC FITNESS ---
                total_trades = wins + losses
                win_rate = (wins / total_trades) if total_trades > 0 else 0.0
                
                # Hybrid Reward:
                # If profitable, scale by Win Rate squared (Sniping).
                # If losing, keep raw ROI (Punishment).
                if roi > 0:
                    individual.fitness = roi * (win_rate ** 2)
                else:
                    individual.fitness = roi
                    
                individual.total_profit = final_value - deposit
                individual.win_rate = win_rate

    def evolve_population(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Log Best
        logging.info(f"  Best ROI: {self.population[0].fitness:.4f}, Profit: ${self.population[0].total_profit:.2f}")
        
        survivors = self.population[:10] # Elite
        
        # Crossover & Mutation
        while len(survivors) < POPULATION_SIZE:
            parent1 = random.choice(survivors[:5])
            child = parent1.clone()
            child.mutate()
            survivors.append(child)
            
        self.population = survivors

    def run(self):
        # Shuffle symbols to support parallel training (multiple windows)
        training_order = list(SYMBOLS)
        random.shuffle(training_order)
        
        for symbol in training_order:
            # Check progress
            cursor = self.conn.execute("SELECT MAX(batch) FROM training_log WHERE symbol = ?", (symbol,))
            last_batch = cursor.fetchone()[0] or 0
            if last_batch >= BATCHES:
                logging.info(f"Symbol {symbol} already completed all {BATCHES} batches. Skipping.")
                continue
                
            try:
                self.run_cycle_logic(symbol)
            except Exception as e:
                logging.error(f"Error processing {symbol}: {e}")
                
if __name__ == "__main__":
    try:
        system = ETARE_System()
        system.run()
    except Exception as e:
        print(f"System Error: {e}")
        logging.error(f"System Error: {e}")
    finally:
        mt5.shutdown()