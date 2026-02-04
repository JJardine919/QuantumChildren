import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from collections import deque
import random
from enum import Enum
from dataclasses import dataclass
import time
from copy import deepcopy
import sqlite3
import json

try:
    import MetaTrader5 as mt5
except ImportError:
    # Linux Mock
    class MockMT5:
        def initialize(self): return True
        def shutdown(self): return True
        def positions_get(self, symbol=None): return []
        def orders_get(self, symbol=None): return []
        def symbol_info_tick(self, symbol): return None
        def copy_rates_from_pos(self, *args): return None
        def terminal_info(self):
            class TerminalInfo:
                data_path = "/root/.wine/drive_c/Program Files/MetaTrader 5"
            return TerminalInfo()
        TIMEFRAME_M1=1; TIMEFRAME_M5=5; TIMEFRAME_M15=15; TIMEFRAME_M30=30
        TIMEFRAME_H1=60; TIMEFRAME_H4=240; TIMEFRAME_D1=1440
        ORDER_TYPE_BUY=0; ORDER_TYPE_SELL=1
        ORDER_TYPE_BUY_LIMIT=2; ORDER_TYPE_SELL_LIMIT=3
        TRADE_ACTION_DEAL=1; TRADE_ACTION_PENDING=5; TRADE_ACTION_REMOVE=8
        TRADE_RETCODE_DONE=10009
        ORDER_TIME_GTC=0; ORDER_FILLING_FOK=0; ORDER_FILLING_IOC=1
    mt5 = MockMT5()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Action(Enum):
    OPEN_BUY = 0
    OPEN_SELL = 1
    CLOSE_BUY_PROFIT = 2
    CLOSE_BUY_LOSS = 3
    CLOSE_SELL_PROFIT = 4
    CLOSE_SELL_LOSS = 5


import torch

# Check for AMD/DirectML GPU first
try:
    import torch_directml
    # We use index 0 for the first DML device
    DEVICE = torch_directml.device()
    logging.info(f"🚀 AMD GPU DETECTED via DirectML: {torch_directml.device_name(0)}")
except ImportError:
    # Fallback to standard CUDA or CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logging.info(f"🚀 NVIDIA GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("⚠️ No AMD or NVIDIA GPU detected. Running on CPU.")

@dataclass
class GeneticWeights:
    input_weights: torch.Tensor
    hidden_weights: torch.Tensor
    output_weights: torch.Tensor
    hidden_bias: torch.Tensor
    output_bias: torch.Tensor


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
            return []
        indices = np.random.choice(len(self.memory), batch_size)
        return [self.memory[i] for i in indices]


class TradingIndividual:
    def __init__(self, input_size: int):
        self.live_trading = False # Default to safe mode
        # Initialize weights on GPU
        self.weights = GeneticWeights(
            input_weights=torch.empty(input_size, 128, device=DEVICE).uniform_(-0.5, 0.5),
            hidden_weights=torch.empty(128, 64, device=DEVICE).uniform_(-0.5, 0.5),
            output_weights=torch.empty(64, len(Action), device=DEVICE).uniform_(-0.5, 0.5),
            hidden_bias=torch.empty(128, device=DEVICE).uniform_(-0.5, 0.5),
            output_bias=torch.empty(len(Action), device=DEVICE).uniform_(-0.5, 0.5),
        )

        self.memory = RLMemory()
        self.fitness = 0
        self.total_profit = 0
        self.trade_history = deque(maxlen=1000)
        self.open_positions: Dict[str, object] = {}

        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 0.1
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1
        
        # Stats for database
        self.successful_trades = 0
        self.total_trades = 0

    def predict(self, state: np.ndarray) -> Tuple[Action, np.ndarray]:
        # Handle shape mismatch (Version 1 vs Version 2 features)
        expected_size = self.weights.input_weights.shape[0]
        
        # Determine if it's 1D or 2D and get the feature dimension
        if state.ndim == 1:
            current_size = state.shape[0]
            if current_size != expected_size:
                if current_size < expected_size:
                    new_state = np.zeros(expected_size)
                    new_state[:current_size] = state
                    state = new_state
                else:
                    state = state[:expected_size]
        elif state.ndim == 2:
            current_size = state.shape[1]
            if current_size != expected_size:
                if current_size < expected_size:
                    new_state = np.zeros((state.shape[0], expected_size))
                    new_state[:, :current_size] = state
                    state = new_state
                else:
                    state = state[:, :expected_size]

        # Convert state to tensor on GPU
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        
        # Normalize
        mean = state_tensor.mean()
        std = state_tensor.std() + 1e-8
        state_tensor = (state_tensor - mean) / std

        # Forward pass (PyTorch)
        hidden = torch.tanh(torch.matmul(state_tensor, self.weights.input_weights) + self.weights.hidden_bias)
        hidden2 = torch.tanh(torch.matmul(hidden, self.weights.hidden_weights))
        output = torch.matmul(hidden2, self.weights.output_weights) + self.weights.output_bias
        
        # Softmax
        probabilities = torch.softmax(output, dim=0) # Assuming 1D output for single prediction
        
        # Move back to CPU for numpy operations/action selection
        probs_np = probabilities.cpu().detach().numpy()

        if np.random.random() < self.epsilon:
            action = Action(np.random.randint(len(Action)))
        else:
            action = Action(np.argmax(probs_np))

        return action, probs_np

    def update(self, state, action, reward, next_state):
        self.memory.add(state, action, reward, next_state)
        self.total_profit += reward

        if len(self.memory.memory) >= 32:
            batch = self.memory.sample(32)
            self._train_on_batch(batch)

    def _train_on_batch(self, batch):
        # Unpack batch data safely
        state_list = [x[0] for x in batch]
        action_list = [x[1].value for x in batch]
        reward_list = [x[2] for x in batch]
        next_state_list = [x[3] for x in batch]

        # Convert numpy arrays to tensors efficiently
        states = torch.as_tensor(np.array(state_list), dtype=torch.float32).to(DEVICE)
        actions = torch.as_tensor(action_list, dtype=torch.long).to(DEVICE)
        rewards = torch.as_tensor(reward_list, dtype=torch.float32).to(DEVICE)
        next_states = torch.as_tensor(np.array(next_state_list), dtype=torch.float32).to(DEVICE)

        # Forward pass (Current States)
        hidden = torch.tanh(torch.matmul(states, self.weights.input_weights) + self.weights.hidden_bias)
        hidden2 = torch.tanh(torch.matmul(hidden, self.weights.hidden_weights))
        current_q_values = torch.matmul(hidden2, self.weights.output_weights) + self.weights.output_bias
        
        # Gather Q-values
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Forward pass (Next States)
        with torch.no_grad():
            next_hidden = torch.tanh(torch.matmul(next_states, self.weights.input_weights) + self.weights.hidden_bias)
            next_hidden2 = torch.tanh(torch.matmul(next_hidden, self.weights.hidden_weights))
            next_q_values = torch.matmul(next_hidden2, self.weights.output_weights) + self.weights.output_bias
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + self.gamma * max_next_q

        # Check gradients
        if not self.weights.input_weights.requires_grad:
            self.weights.input_weights.requires_grad = True
            self.weights.hidden_weights.requires_grad = True
            self.weights.output_weights.requires_grad = True
            self.weights.hidden_bias.requires_grad = True
            self.weights.output_bias.requires_grad = True

        # Compute Loss (MSE)
        loss = torch.nn.functional.mse_loss(current_q, target_q)

        # Backward using Autograd
        grad_input_w = torch.autograd.grad(loss, self.weights.input_weights, retain_graph=True)[0]
        grad_hidden_w = torch.autograd.grad(loss, self.weights.hidden_weights, retain_graph=True)[0]
        grad_output_w = torch.autograd.grad(loss, self.weights.output_weights, retain_graph=True)[0]
        grad_hidden_b = torch.autograd.grad(loss, self.weights.hidden_bias, retain_graph=True)[0]
        grad_output_b = torch.autograd.grad(loss, self.weights.output_bias, retain_graph=False)[0]

        # Update Weights (SGD)
        with torch.no_grad():
            self.weights.input_weights -= self.learning_rate * grad_input_w
            self.weights.hidden_weights -= self.learning_rate * grad_hidden_w
            self.weights.output_weights -= self.learning_rate * grad_output_w
            self.weights.hidden_bias -= self.learning_rate * grad_hidden_b
            self.weights.output_bias -= self.learning_rate * grad_output_b

    def mutate(self):
        if np.random.random() < self.mutation_rate:
            for weight_tensor in [
                self.weights.input_weights,
                self.weights.hidden_weights,
                self.weights.output_weights,
            ]:
                mask = torch.rand_like(weight_tensor) < 0.1
                noise = torch.randn_like(weight_tensor) * self.mutation_strength
                weight_tensor[mask] += noise[mask]


def initialize_mt5():
    """Initialize connection to MT5"""
    if not mt5.initialize():
        logging.error("MT5 initialization failed")
        return False
    return True


def get_mt5_data(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Get data from MT5"""
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, limit)
        if rates is None:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
    except Exception as e:
        logging.error(f"Error getting MT5 data: {str(e)}")
        return None


def prepare_features(data: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
    """Prepare features"""
    df = data.copy()

    # Technical indicators
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

    for period in [5, 10, 20, 50]:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    df["momentum"] = df["close"] / df["close"].shift(10)
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()
    df["price_change"] = df["close"].pct_change()
    df["price_change_abs"] = df["price_change"].abs()
    df["volume_ma"] = df["tick_volume"].rolling(20).mean()
    df["volume_std"] = df["tick_volume"].rolling(20).std()

    # Fill in the gaps
    df = df.ffill().bfill()

    # Normalization
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (
        df[numeric_cols].std() + 1e-8
    )

    df = df.drop(
        ["time", "open", "high", "low", "tick_volume", "spread", "real_volume"],
        axis=1,
        errors="ignore",
    )

    return df


class SniperParameters:
    def __init__(self):
        # Risk Management (Fixed % or fixed distance)
        self.stop_loss_pips = np.random.randint(10, 50)  # 10 to 50 pips SL
        self.take_profit_pips = np.random.randint(20, 100) # 20 to 100 pips TP
        self.risk_per_trade = 0.01 # 1% Risk per trade
        
        # Min Hold Time (Safety Rule: > 2 mins)
        self.min_hold_seconds = 125 # 2 mins + 5s buffer
        
        self.mutation_rate = 0.1
        self.mutation_strength = 5 # Pips mutation

    def mutate(self):
        if np.random.random() < self.mutation_rate:
            # Mutate SL/TP
            self.stop_loss_pips = max(5, self.stop_loss_pips + np.random.randint(-5, 6))
            self.take_profit_pips = max(10, self.take_profit_pips + np.random.randint(-10, 11))


class SniperTrade:
    def __init__(self, order_type, price, volume, sl, tp, ticket=None):
        self.order_type = order_type
        self.entry_price = price
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.ticket = ticket
        self.open_time = time.time()
        self.profit = 0.0
        self.is_open = True


class SniperTrader(TradingIndividual):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self.sniper_params = SniperParameters()
        self.active_trade: Dict[str, SniperTrade] = {} # Only ONE trade per symbol allowed
        
    def execute_sniper_trade(self, symbol: str, action: Action, current_price: float):
        """Execute a high-frequency sniper trade with TIGHT stops"""
        # 1. One trade per individual per symbol
        if symbol in self.active_trade:
            return None

        # 2. Calculate TIGHT Levels
        # Enforcing 'Hold it tight' - Small SL to protect account
        point = mt5.symbol_info(symbol).point
        tight_sl = 150 * point # 15 pips
        tight_tp = 300 * point # 30 pips
        
        if action == Action.OPEN_BUY:
            order_type = mt5.ORDER_TYPE_BUY
            sl = current_price - tight_sl
            tp = current_price + tight_tp
        else: # SELL
            order_type = mt5.ORDER_TYPE_SELL
            sl = current_price + tight_sl
            tp = current_price - tight_tp

        # 3. Send Market Order
        volume = 0.01 
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": current_price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 10,
            "magic": 85000, 
            "comment": "SNIPER_FULL_BLAST",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if not self.live_trading:
            # Ghost trade
            self.active_trade[symbol] = SniperTrade(order_type, current_price, volume, sl, tp)
            return self.active_trade[symbol]

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Sniper shot failed for {symbol}")
            return None
            
        self.active_trade[symbol] = SniperTrade(order_type, current_price, volume, sl, tp, result.order)
        logging.info(f"🎯 SNIPER SHOT: {symbol} {'BUY' if action == Action.OPEN_BUY else 'SELL'} @ {current_price:.5f} (SL: {sl:.5f} TP: {tp:.5f})")
        return self.active_trade[symbol]

    def manage_trade(self, symbol: str, current_price: float):
        """Manage open trades (Close if signal reverses OR time limit met)"""
        if symbol not in self.active_trade:
            return

        trade = self.active_trade[symbol]
        
        # 1. Check Minimum Hold Time (2 mins)
        duration = time.time() - trade.open_time
        if duration < self.sniper_params.min_hold_seconds:
            return # Too early to close

        # 2. Check Virtual SL/TP (for Ghost Mode)
        # (In live mode, MT5 handles SL/TP automatically, but we simulate for fitness)
        if not self.live_trading:
            if trade.order_type == mt5.ORDER_TYPE_BUY:
                if current_price <= trade.sl: self.close_trade(symbol, "SL_HIT", current_price)
                elif current_price >= trade.tp: self.close_trade(symbol, "TP_HIT", current_price)
            else: # SELL
                if current_price >= trade.sl: self.close_trade(symbol, "SL_HIT", current_price)
                elif current_price <= trade.tp: self.close_trade(symbol, "TP_HIT", current_price)

    def close_trade(self, symbol: str, reason: str, price: float):
        """Close the trade"""
        if symbol not in self.active_trade: return
        
        trade = self.active_trade[symbol]
        
        # Calculate Profit
        if trade.order_type == mt5.ORDER_TYPE_BUY:
            profit = (price - trade.entry_price) * trade.volume * 100000 # Approx
        else:
            profit = (trade.entry_price - price) * trade.volume * 100000

        self.fitness += profit
        self.total_profit += profit
        
        if profit > 0: self.successful_trades += 1
        self.total_trades += 1
        
        del self.active_trade[symbol]
        # logging.info(f"Trade Closed {symbol}: {reason} Profit: ${profit:.2f}")

    def mutate(self):
        super().mutate()
        self.sniper_params.mutate()


class HybridSniperTrader:
    def __init__(self, symbols: List[str], population_size: int = 50):
        self.symbols = symbols
        self.population_size = population_size
        self.population: List[SniperTrader] = []
        self.generation = 0
        self.live_trading = False 

        self.action_names = {
            Action.OPEN_BUY: "OPEN_BUY",
            Action.OPEN_SELL: "OPEN_SELL",
            Action.CLOSE_BUY_PROFIT: "CLOSE_BUY_PROFIT",
            Action.CLOSE_BUY_LOSS: "CLOSE_BUY_LOSS",
            Action.CLOSE_SELL_PROFIT: "CLOSE_SELL_PROFIT",
            Action.CLOSE_SELL_LOSS: "CLOSE_SELL_LOSS"
        }
        self.tournament_size = 3
        self.elite_size = 5
        self.extinction_rate = 0.3
        self.extinction_interval = 10
        self.inefficient_extinction_interval = 5
        self.deal_count = 0

        # Initialize the input data and population size
        sample_data = self._get_sample_features()
        self.input_size = len(sample_data.columns) if sample_data is not None else 100
        self._initialize_population()

        # Initialize database
        self.conn = sqlite3.connect("trading_history.db")
        self._create_tables()
        self._load_from_db()

    def _create_tables(self):
        """Create tables in the database"""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS population (
                    id INTEGER PRIMARY KEY,
                    individual TEXT
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY,
                    generation INTEGER,
                    individual_id INTEGER,
                    trade_history TEXT,
                    total_profit REAL,
                    win_rate REAL,
                    FOREIGN KEY(individual_id) REFERENCES population(id)
                )
            """
            )

    def _load_from_db(self):
        """Download state from the database"""
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    SELECT id, individual, fitness, successful_trades, total_trades 
                    FROM population 
                    ORDER BY fitness DESC 
                    LIMIT ?
                """,
                    (self.population_size,),
                )

                rows = cursor.fetchall()

                if rows:
                    self.population = []
                    for row in rows:
                        individual_data = json.loads(row[1])
                        individual = SniperTrader(self.input_size)

                        # Restore neural network weights
                        individual.weights.input_weights = torch.tensor(individual_data["input_weights"], device=DEVICE, dtype=torch.float32)
                        individual.weights.hidden_weights = torch.tensor(individual_data["hidden_weights"], device=DEVICE, dtype=torch.float32)
                        individual.weights.output_weights = torch.tensor(individual_data["output_weights"], device=DEVICE, dtype=torch.float32)
                        individual.weights.hidden_bias = torch.tensor(individual_data["hidden_bias"], device=DEVICE, dtype=torch.float32)
                        individual.weights.output_bias = torch.tensor(individual_data["output_bias"], device=DEVICE, dtype=torch.float32)

                        # Restore Sniper parameters (handle legacy Grid params gracefully)
                        if "stop_loss_pips" in individual_data:
                            individual.sniper_params.stop_loss_pips = individual_data["stop_loss_pips"]
                            individual.sniper_params.take_profit_pips = individual_data["take_profit_pips"]
                        else:
                            # Convert Grid -> Sniper (Reset params for safety)
                            individual.sniper_params = SniperParameters()

                        # Restore statistics
                        individual.fitness = row[2]
                        individual.successful_trades = row[3]
                        individual.total_trades = row[4]
                        individual.id = row[0]

                        self.population.append(individual)

                    cursor = self.conn.execute("SELECT MAX(generation) FROM history")
                    last_gen = cursor.fetchone()[0]
                    if last_gen is not None:
                        self.generation = last_gen

                    logging.info(f"Loaded {len(self.population)} individuals from database")
                else:
                    logging.info("No saved state found, starting fresh")

        except Exception as e:
            logging.error(f"Error loading from database: {str(e)}")
            self._initialize_population()

    def _get_sample_features(self) -> pd.DataFrame:
        """Get data sample to define the entry size"""
        for symbol in self.symbols:
            data = get_mt5_data(symbol, "M5", 100)
            if data is not None and not data.empty:
                return prepare_features(data, include_target=False)
        return None

    def _get_state(self, symbol: str) -> np.ndarray:
        """Get current market state for a symbol"""
        data = get_mt5_data(symbol, "M5", 100)
        if data is None or data.empty:
            return None
        features = prepare_features(data, include_target=False)
        if features is None or features.empty:
            return None
        return features.iloc[-1].values

    def _extinction_event(self):
        """Extinction event - refreshes population when performance stalls"""
        logging.info("🌋 EXTINCTION EVENT: Evoling population...")
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        survivors = self.population[: self.elite_size]

        while len(survivors) < self.population_size:
            if random.random() < 0.8:  # 80% crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:  # 20% elite mutation
                child = deepcopy(random.choice(self.population[:self.elite_size]))
                child.mutate()
            survivors.append(child)

        self.population = survivors
        logging.info(f"Population refreshed. Elite fitness: {self.population[0].fitness:.2f}")

    def _save_to_db(self):
        """Save state to the database"""
        try:
            with self.conn:
                self.conn.execute("DELETE FROM population")

                for individual in self.population:
                    individual_data = {
                        "input_weights": individual.weights.input_weights.tolist(),
                        "hidden_weights": individual.weights.hidden_weights.tolist(),
                        "output_weights": individual.weights.output_weights.tolist(),
                        "hidden_bias": individual.weights.hidden_bias.tolist(),
                        "output_bias": individual.weights.output_bias.tolist(),
                        "stop_loss_pips": individual.sniper_params.stop_loss_pips,
                        "take_profit_pips": individual.sniper_params.take_profit_pips,
                    }

                    cursor = self.conn.execute(
                        """
                        INSERT INTO population (individual, fitness, successful_trades, total_trades)
                        VALUES (?, ?, ?, ?)
                        RETURNING id
                    """,
                        (
                            json.dumps(individual_data),
                            individual.fitness,
                            individual.successful_trades,
                            individual.total_trades,
                        ),
                    )
                    individual.id = cursor.fetchone()[0]

                best_individual = max(self.population, key=lambda x: x.fitness)
                self.conn.execute(
                    """
                    INSERT INTO history (generation, individual_id, trade_history, total_profit, win_rate)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        self.generation,
                        best_individual.id,
                        json.dumps([]),
                        best_individual.fitness,
                        best_individual.successful_trades
                        / max(1, best_individual.total_trades),
                    ),
                )
                logging.info(f"Saved population state to database, generation {self.generation}")

        except Exception as e:
            logging.error(f"Error saving to database: {str(e)}")

    def _initialize_population(self):
        """Initialize population"""
        self.population = [
            SniperTrader(self.input_size) for _ in range(self.population_size)
        ]

    def _tournament_selection(self) -> SniperTrader:
        """Tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: SniperTrader, parent2: SniperTrader) -> SniperTrader:
        """Two parents' crossbreading"""
        child = SniperTrader(self.input_size)

        for attr in ["input_weights", "hidden_weights", "output_weights"]:
            parent1_weights = getattr(parent1.weights, attr)
            parent2_weights = getattr(parent2.weights, attr)
            mask = torch.rand_like(parent1_weights) < 0.5
            child_weights = torch.where(mask, parent1_weights, parent2_weights)
            setattr(child.weights, attr, child_weights)

        # Sniper Params Crossover
        if np.random.random() < 0.5:
            child.sniper_params.stop_loss_pips = parent1.sniper_params.stop_loss_pips
            child.sniper_params.take_profit_pips = parent1.sniper_params.take_profit_pips
        else:
            child.sniper_params.stop_loss_pips = parent2.sniper_params.stop_loss_pips
            child.sniper_params.take_profit_pips = parent2.sniper_params.take_profit_pips

        return child

    def _process_individual(
        self, 
        symbol: str, 
        individual: SniperTrader, 
        current_state: np.ndarray,
        cached_positions: tuple = None,
        cached_orders: tuple = None
    ):
        """Handle trading logic for Sniper"""
        try:
            # 1. Manage Active Trades
            if symbol in individual.active_trade:
                current_price = mt5.symbol_info_tick(symbol).bid # approximation
                individual.manage_trade(symbol, current_price)
                return

            # 2. Look for Entry
            # Check global limit logic here if needed (e.g. max 1 position per symbol globally)
            
            action, _ = individual.predict(current_state)
            if action in [Action.OPEN_BUY, Action.OPEN_SELL]:
                current_price = mt5.symbol_info_tick(symbol).ask if action == Action.OPEN_BUY else mt5.symbol_info_tick(symbol).bid
                individual.execute_sniper_trade(symbol, action, current_price)

        except Exception as e:
            logging.error(f"Error processing sniper individual: {str(e)}")

    def run_iteration(self):
        """FULL BLAST: Every individual trades live. No consensus. No mercy."""
        try:
            # 1. Evolve population periodically
            if self.generation % self.extinction_interval == 0:
                self._extinction_event()

            for symbol in self.symbols:
                data = get_mt5_data(symbol, "M5", 100)
                if data is None or len(data) < 100:
                    continue

                features = prepare_features(data, include_target=False)
                if features.empty:
                    continue

                current_state = features.iloc[-1].values.reshape(1, -1)
                
                # Cache MT5 data
                cached_positions = mt5.positions_get(symbol=symbol)
                cached_orders = mt5.orders_get(symbol=symbol)

                # --- STEP A: TOTAL WAR (INDIVIDUAL LIVE TRADING) ---
                # Check 100-Trade Ceiling (Increased to bypass weekend Forex blockage)
                current_positions = mt5.positions_get()
                current_count = len(current_positions) if current_positions else 0
                
                if current_count >= 100:
                    # CEILING HIT: Skip new trades, only manage existing
                    continue

                # Every strategy gets to shoot if under ceiling
                for individual in self.population:
                    # FORCE LIVE TRADING
                    individual.live_trading = True
                    
                    self._process_individual(
                        symbol,
                        individual,
                        current_state,
                        cached_positions,
                        cached_orders
                    )
                    # Tiny delay to prevent socket crash
                    time.sleep(0.5)

        except Exception as e:
            logging.error(f"Error in FULL BLAST iteration: {str(e)}")

    def run_trading_cycle(self):
        """Main trading loop (Legacy/Live)"""
        while True:
            self.run_iteration()
            time.sleep(30)  # Fast Forward default


def main():
    symbols = [
        "EURUSD.ecn",
        "GBPUSD.ecn",
        "USDCHF.ecn",
        "USDCAD.ecn",
        "AUDUSD.ecn",
        "NZDUSD.ecn",
        "EURGBP.ecn",
        "EURCHF.ecn",
        "EURCAD.ecn",
        "EURAUD.ecn",
        "EURNZD.ecn",
        "GBPCHF.ecn",
        "GBPCAD.ecn",
        "GBPAUD.ecn",
        "GBPNZD.ecn",
        "AUDNZD.ecn",
        "AUDCHF.ecn",
        "NZDCHF.ecn",
        "NZDCAD.ecn",
        "CADCHF.ecn",
        "AUDCAD.ecn",
    ]

    # Initialize MT5
    if not initialize_mt5():
        logging.error("Failed to initialize MT5")
        return

    # Launch trading loop
    trader = HybridSniperTrader(symbols)
    trader.run_trading_cycle()


if __name__ == "__main__":
    main()