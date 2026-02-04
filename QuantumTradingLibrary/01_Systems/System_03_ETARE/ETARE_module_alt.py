import numpy as np
import pandas as pd
import MetaTrader5 as mt5
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


@dataclass
class GeneticWeights:
    input_weights: np.ndarray
    hidden_weights: np.ndarray
    output_weights: np.ndarray
    hidden_bias: np.ndarray
    output_bias: np.ndarray


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
        self.weights = GeneticWeights(
            input_weights=np.random.uniform(-0.5, 0.5, (input_size, 128)),
            hidden_weights=np.random.uniform(-0.5, 0.5, (128, 64)),
            output_weights=np.random.uniform(-0.5, 0.5, (64, len(Action))),
            hidden_bias=np.random.uniform(-0.5, 0.5, (128,)),
            output_bias=np.random.uniform(-0.5, 0.5, (len(Action),)),
        )

        self.memory = RLMemory()
        self.fitness = 0
        self.total_profit = 0
        self.trade_history = deque(maxlen=1000)
        self.open_positions: Dict[str, List[GridTrade]] = {}

        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 0.1
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1

    def predict(self, state: np.ndarray) -> Tuple[Action, np.ndarray]:
        state = (state - state.mean()) / (state.std() + 1e-8)
        hidden = np.tanh(
            np.dot(state, self.weights.input_weights) + self.weights.hidden_bias
        )
        hidden2 = np.tanh(np.dot(hidden, self.weights.hidden_weights))
        output = np.dot(hidden2, self.weights.output_weights) + self.weights.output_bias
        probabilities = self._softmax(output)

        if np.random.random() < self.epsilon:
            action = Action(np.random.randint(len(Action)))
        else:
            action = Action(np.argmax(probabilities))

        return action, probabilities

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def update(self, state, action, reward, next_state):
        self.memory.add(state, action, reward, next_state)
        self.total_profit += reward

        if len(self.memory.memory) >= 32:
            batch = self.memory.sample(32)
            self._train_on_batch(batch)

    def _train_on_batch(self, batch):
        for state, action, reward, next_state in batch:
            hidden = np.tanh(
                np.dot(state, self.weights.input_weights) + self.weights.hidden_bias
            )
            hidden2 = np.tanh(np.dot(hidden, self.weights.hidden_weights))
            current_q = (
                np.dot(hidden2, self.weights.output_weights) + self.weights.output_bias
            )

            next_hidden = np.tanh(
                np.dot(next_state, self.weights.input_weights)
                + self.weights.hidden_bias
            )
            next_hidden2 = np.tanh(np.dot(next_hidden, self.weights.hidden_weights))
            next_q = (
                np.dot(next_hidden2, self.weights.output_weights)
                + self.weights.output_bias
            )

            target = current_q.copy()
            target[0, action.value] = reward + self.gamma * np.max(next_q)

            self._backprop(state, hidden, hidden2, current_q, target)

    def _backprop(self, state, hidden, hidden2, current_q, target):
        output_error = (target - current_q) * self.learning_rate
        hidden2_error = np.dot(output_error, self.weights.output_weights.T) * (
            1 - hidden2 * hidden2
        )
        hidden_error = np.dot(hidden2_error, self.weights.hidden_weights.T) * (
            1 - hidden * hidden
        )

        self.weights.output_weights += np.dot(hidden2.T, output_error)
        self.weights.hidden_weights += np.dot(hidden.T, hidden2_error)
        self.weights.input_weights += np.dot(state.T, hidden_error)

        self.weights.output_bias += output_error.sum(axis=0)
        self.weights.hidden_bias += hidden_error.sum(axis=0)

    def mutate(self):
        if np.random.random() < self.mutation_rate:
            for weight_matrix in [
                self.weights.input_weights,
                self.weights.hidden_weights,
                self.weights.output_weights,
            ]:
                mask = np.random.random(weight_matrix.shape) < 0.1
                weight_matrix[mask] += np.random.normal(
                    0, self.mutation_strength, size=mask.sum()
                )


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
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

    # EMAs
    for period in [5, 10, 20, 50]:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    # Momentum
    df["momentum"] = df["close"] / df["close"].shift(10)

    # Volatility
    df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()

    # Price changes
    df["price_change"] = df["close"].pct_change()
    df["price_change_abs"] = df["price_change"].abs()

    # Volumes
    df["volume_ma"] = df["tick_volume"].rolling(20).mean()
    df["volume_std"] = df["tick_volume"].rolling(20).std()

    # Fill in the gaps
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Normalization
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (
        df[numeric_cols].std() + 1e-8
    )

    # Remove unnecessary columns
    df = df.drop(
        ["time", "open", "high", "low", "tick_volume", "spread", "real_volume"],
        axis=1,
        errors="ignore",
    )

    return df


class GridParameters:
    def __init__(self):
        self.grid_step = np.random.uniform(0.00005, 0.0002)  # Grid step by price
        # SAFETY UPDATE: Force Micro-Lots
        self.orders_count = np.random.randint(2, 5) 
        self.base_volume = 0.01
        self.volume_step = 0.0
        
        self.mutation_rate = 0.1
        self.mutation_strength = 0.01

    def mutate(self):
        if np.random.random() < self.mutation_rate:
            # Price step mutation
            self.grid_step = max(
                0.00005, min(0.0002, self.grid_step + np.random.normal(0, 0.00005))
            )
            # Base volume mutation - KEEP SMALL
            self.base_volume = 0.01 
            
            # Volume step mutation - KEEP FLAT
            self.volume_step = 0.0
            
            # Order number mutation
            self.orders_count = max(
                2, min(5, self.orders_count + np.random.randint(-1, 2))
            )


class GridTrade:
    def __init__(self, order_type, price, volume, ticket=None):
        self.order_type = order_type
        self.price = price
        self.volume = max(
            0.01, round(volume, 2)
        )  # Minimum volume 0.01, round to 2 digits
        self.ticket = ticket
        self.profit = 0.0
        self.is_open = True


class GridTrader(TradingIndividual):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self.grid_params = GridParameters()
        self.grid_orders: Dict[str, List[GridTrade]] = {}
        self.partial_profit_taken: Dict[str, bool] = {}
        self.max_profit: Dict[str, float] = {}

    def create_grid(self, symbol: str, action: Action, current_price: float):
        """Create order grids with an increasing volume"""
        orders = []

        for i in range(self.grid_params.orders_count):
            # Volume calculation for the current order
            # Each subsequent order is increased by volume step
            current_volume = max(
                0.01,
                round(
                    self.grid_params.base_volume + (i * self.grid_params.volume_step), 2
                ),
            )

            if action == Action.OPEN_BUY:
                price = current_price - (i + 1) * self.grid_params.grid_step
                order_type = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                price = current_price + (i + 1) * self.grid_params.grid_step
                order_type = mt5.ORDER_TYPE_SELL_LIMIT

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": current_volume,  # Use calculated volume
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Grid_{i}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                orders.append(
                    GridTrade(order_type, price, current_volume, result.order)
                )
                logging.info(
                    f"Created order for {symbol}: Volume={current_volume}, Price={price}"
                )

        return orders

    def calculate_grid_profit(self, symbol: str) -> float:
        """Calculate grid total profit"""
        total_profit = 0.0
        if symbol in self.grid_orders:
            if self.live_trading:
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    for pos in positions:
                        total_profit += pos.profit + pos.swap
            else:
                # Virtual profit calculation
                tick = mt5.symbol_info_tick(symbol)
                symbol_info = mt5.symbol_info(symbol)
                if tick and symbol_info:
                    contract_size = symbol_info.trade_contract_size
                    for order in self.grid_orders[symbol]:
                        if order.order_type == mt5.ORDER_TYPE_BUY_LIMIT:
                            total_profit += (tick.bid - order.price) * order.volume * contract_size
                        else:
                            total_profit += (order.price - tick.ask) * order.volume * contract_size

        return total_profit

    def close_partial_grid(self, symbol: str):
        """Close ~50% of the grid to lock in profits"""
        if symbol in self.grid_orders:
            # Close open positions
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                # Sort by profit to close winners first
                positions = sorted(positions, key=lambda p: p.profit, reverse=True)
                
                # Close half of the positions count
                count_to_close = max(1, len(positions) // 2)
                
                for i in range(count_to_close):
                    pos = positions[i]
                    close_type = (
                        mt5.ORDER_TYPE_SELL
                        if pos.type == mt5.ORDER_TYPE_BUY
                        else mt5.ORDER_TYPE_BUY
                    )
                    symbol_info = mt5.symbol_info(symbol)
                    price = (
                        mt5.symbol_info_tick(symbol).bid
                        if close_type == mt5.ORDER_TYPE_SELL
                        else mt5.symbol_info_tick(symbol).ask
                    )
                    price = round(price, symbol_info.digits)

                    if not self.live_trading:
                        continue

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume, 
                        "type": close_type,
                        "position": pos.ticket,
                        "price": float(price),
                        "deviation": 20,
                        "magic": 123456,
                        "comment": "Partial Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    mt5.order_send(request)

    def close_grid(self, symbol: str):
        """Close all grid orders"""
        if symbol in self.grid_orders:
            # Close open positions
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for pos in positions:
                    close_type = (
                        mt5.ORDER_TYPE_SELL
                        if pos.type == mt5.ORDER_TYPE_BUY
                        else mt5.ORDER_TYPE_BUY
                    )
                    price = (
                        mt5.symbol_info_tick(symbol).bid
                        if close_type == mt5.ORDER_TYPE_SELL
                        else mt5.symbol_info_tick(symbol).ask
                    )

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": close_type,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": 123456,
                        "comment": "Close Grid",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                    }
                    mt5.order_send(request)

            # Remove pending orders
            orders = mt5.orders_get(symbol=symbol)
            if orders:
                for order in orders:
                    request = {
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order": order.ticket,
                        "magic": 123456,
                    }
                    mt5.order_send(request)

            del self.grid_orders[symbol]

    def mutate(self):
        super().mutate()
        self.grid_params.mutate()


class HybridGridTrader:
    def __init__(self, symbols: List[str], population_size: int = 50):
        self.symbols = symbols
        self.population_size = population_size
        self.population: List[GridTrader] = []
        self.generation = 0

        # Evolution parameters
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
                    FOREIGN KEY(individual_id) REFERENCES population(id)
                )
            """
            )

    def _load_from_db(self):
        """Download state from the database"""
        try:
            with self.conn:
                # Download the last population state
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
                        individual = GridTrader(self.input_size)

                        # Restore neural network weights
                        individual.weights.input_weights = np.array(
                            individual_data["input_weights"]
                        )
                        individual.weights.hidden_weights = np.array(
                            individual_data["hidden_weights"]
                        )
                        individual.weights.output_weights = np.array(
                            individual_data["output_weights"]
                        )
                        individual.weights.hidden_bias = np.array(
                            individual_data["hidden_bias"]
                        )
                        individual.weights.output_bias = np.array(
                            individual_data["output_bias"]
                        )

                        # Restore network parameters
                        individual.grid_params.grid_step = individual_data["grid_step"]
                        individual.grid_params.base_volume = individual_data[
                            "base_volume"
                        ]
                        individual.grid_params.volume_step = individual_data[
                            "volume_step"
                        ]
                        individual.grid_params.orders_count = individual_data[
                            "orders_count"
                        ]

                        # Restore statistics
                        individual.fitness = row[2]
                        individual.successful_trades = row[3]
                        individual.total_trades = row[4]
                        individual.id = row[0]

                        self.population.append(individual)

                    # Get the last generation
                    cursor = self.conn.execute(
                        """
                        SELECT MAX(generation) FROM history
                    """
                    )
                    last_gen = cursor.fetchone()[0]
                    if last_gen is not None:
                        self.generation = last_gen

                    logging.info(
                        f"Loaded {len(self.population)} individuals from database"
                    )
                else:
                    logging.info("No saved state found, starting fresh")

        except Exception as e:
            logging.error(f"Error loading from database: {str(e)}")
            self._initialize_population()

        def _get_sample_features(self):
            """Get data sample to define the entry size"""
            for symbol in self.symbols:
                data = self._get_mt5_data(symbol, "M5", 100)
                if data is not None:
                    return self._prepare_features(data)
            return None

    def _get_mt5_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
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

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features"""
        df = data.copy()

        # Technical indicators
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

        # EMAs
        for period in [5, 10, 20, 50]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # Momentum
        df["momentum"] = df["close"] / df["close"].shift(10)

        # Volatility
        df["atr"] = df["high"].rolling(14).max() - df["low"].rolling(14).min()

        # Price changes
        df["price_change"] = df["close"].pct_change()
        df["price_change_abs"] = df["price_change"].abs()

        # Volumes
        df["volume_ma"] = df["tick_volume"].rolling(20).mean()
        df["volume_std"] = df["tick_volume"].rolling(20).std()

        # Additional indicators
        # Stochastic
        low_min = df["low"].rolling(14).min()
        high_max = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # CCI
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        mean_price = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df["cci"] = (typical_price - mean_price) / (0.015 * mad)

        # ROC (Rate of Change)
        df["roc"] = df["close"].pct_change(10) * 100

        # Williams %R
        df["williams_r"] = -100 * (high_max - df["close"]) / (high_max - low_min)

        # Fill in the gaps
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / (
            df[numeric_cols].std() + 1e-8
        )

        # Remove unnecessary columns
        df = df.drop(
            ["time", "open", "high", "low", "tick_volume", "spread", "real_volume"],
            axis=1,
            errors="ignore",
        )

        return df

    def _save_to_db(self):
        """Save state to the database"""
        try:
            with self.conn:
                # Clear old entries first
                self.conn.execute("DELETE FROM population")

                # Save each individual
                for individual in self.population:
                    individual_data = {
                        "input_weights": individual.weights.input_weights.tolist(),
                        "hidden_weights": individual.weights.hidden_weights.tolist(),
                        "output_weights": individual.weights.output_weights.tolist(),
                        "hidden_bias": individual.weights.hidden_bias.tolist(),
                        "output_bias": individual.weights.output_bias.tolist(),
                        "grid_step": individual.grid_params.grid_step,
                        "base_volume": individual.grid_params.base_volume,
                        "volume_step": individual.grid_params.volume_step,
                        "orders_count": individual.grid_params.orders_count,
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

                # Save info on generation
                best_individual = max(self.population, key=lambda x: x.fitness)
                self.conn.execute(
                    """
                    INSERT INTO history (generation, individual_id, trade_history, total_profit, win_rate)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        self.generation,
                        best_individual.id,
                        json.dumps([]),  # simple trading history for simplification
                        best_individual.fitness,
                        best_individual.successful_trades
                        / max(1, best_individual.total_trades),
                    ),
                )

                logging.info(
                    f"Saved population state to database, generation {self.generation}"
                )

        except Exception as e:
            logging.error(f"Error saving to database: {str(e)}")

    def _cleanup_db(self):
        """Clear old entries from the database"""
        try:
            with self.conn:
                # Leave only the last 1000 entries in history
                self.conn.execute(
                    """
                    DELETE FROM history 
                    WHERE id NOT IN (
                        SELECT id FROM history 
                        ORDER BY generation DESC 
                        LIMIT 1000
                    )
                """
                )

                # Leave trading metric entries for the last 7 days only
                self.conn.execute(
                    """
                    DELETE FROM trades 
                    WHERE exit_time < datetime('now', '-7 days')
                """
                )

        except Exception as e:
            logging.error(f"Error cleaning database: {str(e)}")

    def _get_trade_statistics(self, individual_id: int, days: int = 7) -> dict:
        """Get trading statistics for a certain individual"""
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as profitable_trades,
                        SUM(profit) as total_profit,
                        AVG(duration) as avg_duration,
                        MIN(profit) as worst_trade,
                        MAX(profit) as best_trade
                    FROM trades 
                    WHERE individual_id = ? 
                    AND exit_time > datetime('now', ?)
                """,
                    (individual_id, f"-{days} days"),
                )

                return dict(cursor.fetchone())

        except Exception as e:
            logging.error(f"Error getting trade statistics: {str(e)}")
            return {}

    def _get_sample_features(self):
        """Get data sample to define the entry size"""
        for symbol in self.symbols:
            data = get_mt5_data(symbol, "M5", 100)
            if data is not None:
                return prepare_features(data, include_target=False)
        return None

    def _initialize_population(self):
        """Initialize population"""
        self.population = [
            GridTrader(self.input_size) for _ in range(self.population_size)
        ]

    def _tournament_selection(self) -> GridTrader:
        """Tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: GridTrader, parent2: GridTrader) -> GridTrader:
        """Two parents' crossbreading"""
        child = GridTrader(self.input_size)

        # Crossbreading neural network weights
        for attr in ["input_weights", "hidden_weights", "output_weights"]:
            parent1_weights = getattr(parent1.weights, attr)
            parent2_weights = getattr(parent2.weights, attr)
            mask = np.random.random(parent1_weights.shape) < 0.5
            child_weights = np.where(mask, parent1_weights, parent2_weights)
            setattr(child.weights, attr, child_weights)

        # Crossbreading network parameters
        if np.random.random() < 0.5:
            child.grid_params.grid_step = parent1.grid_params.grid_step
            child.grid_params.base_volume = parent1.grid_params.base_volume
            child.grid_params.volume_step = parent1.grid_params.volume_step
            child.grid_params.orders_count = parent1.grid_params.orders_count
        else:
            child.grid_params.grid_step = parent2.grid_params.grid_step
            child.grid_params.base_volume = parent2.grid_params.base_volume
            child.grid_params.volume_step = parent2.grid_params.volume_step
            child.grid_params.orders_count = parent2.grid_params.orders_count

        return child

    def _extinction_event(self):
        """Extinction event"""
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        survivors = self.population[: self.elite_size]

        while len(survivors) < self.population_size:
            if random.random() < 0.8:  # 80% crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:  # 20% elite mutation
                child = deepcopy(random.choice(survivors))
            child.mutate()
            survivors.append(child)

        self.population = survivors

    def _process_individual(
        self, symbol: str, individual: GridTrader, current_state: np.ndarray
    ):
        """Handle trading logic for the grid"""
        try:
            # CHECK GLOBAL LIMIT: prevent over-trading
            current_positions = mt5.positions_get(symbol=symbol)
            current_orders = mt5.orders_get(symbol=symbol)
            
            pos_count = len(current_positions) if current_positions else 0
            ord_count = len(current_orders) if current_orders else 0
            
            too_many_positions = (pos_count + ord_count) >= 3

            if symbol not in individual.grid_orders:
                if too_many_positions:
                    return

                action, _ = individual.predict(current_state)
                if action in [Action.OPEN_BUY, Action.OPEN_SELL]:
                    current_price = mt5.symbol_info_tick(symbol).ask
                    orders = individual.create_grid(symbol, action, current_price)
                    if orders:
                        individual.grid_orders[symbol] = orders
                        logging.info(
                            f"Created new grid for {symbol} with {len(orders)} orders"
                        )
            else:
                total_profit = individual.calculate_grid_profit(symbol)

                # Update Max Profit (High Water Mark)
                current_max = individual.max_profit.get(symbol, -float('inf'))
                if total_profit > current_max:
                    individual.max_profit[symbol] = total_profit
                    current_max = total_profit

                # Dynamic Take Profit (Partial Closure) - 50% of Target ($3.00)
                if total_profit >= 3.0 and not individual.partial_profit_taken.get(symbol, False):
                    logging.info(f"💰 PARTIAL PROFIT: Closing 50% of grid for {symbol} at ${total_profit:.2f}")
                    individual.close_partial_grid(symbol)
                    individual.partial_profit_taken[symbol] = True

                # Rolling Stop Loss (Trailing Stop)
                # Trigger: Profit > 1.5x Risk ($0.90)
                # Trail Distance: 1.5x Risk ($0.90) from Peak
                trailing_trigger = 0.90
                trailing_distance = 0.90
                
                if current_max >= trailing_trigger:
                    trailing_stop_level = current_max - trailing_distance
                    if total_profit <= trailing_stop_level:
                        logging.info(
                            f"📉 TRAILING STOP: Closing grid for {symbol} at ${total_profit:.2f} (Retraced from ${current_max:.2f})"
                        )
                        individual.close_grid(symbol)
                        individual.fitness += total_profit
                        individual.total_profit += total_profit
                        # Clear state
                        if symbol in individual.max_profit: del individual.max_profit[symbol]
                        if symbol in individual.partial_profit_taken: del individual.partial_profit_taken[symbol]
                        return # Exit early
                
                # Check Profit Target ($6.00)
                if total_profit >= 6.0:  # Adjusted to ~$6 range per user request
                    logging.info(
                        f"💰 TAKE PROFIT: Closing grid for {symbol} with profit ${total_profit:.2f}"
                    )
                    individual.close_grid(symbol)
                    individual.fitness += total_profit
                    individual.total_profit += total_profit
                    # Clear state
                    if symbol in individual.max_profit: del individual.max_profit[symbol]
                    if symbol in individual.partial_profit_taken: del individual.partial_profit_taken[symbol]
                
                # Check Stop Loss ($0.60)
                elif total_profit <= -0.60: # Adjusted to ~$0.60 risk per user request
                    logging.info(
                        f"🛑 STOP LOSS: Closing grid for {symbol} with loss ${total_profit:.2f}"
                    )
                    individual.close_grid(symbol)
                    individual.fitness += total_profit # Penalize fitness
                    individual.total_profit += total_profit
                    # Clear state
                    if symbol in individual.max_profit: del individual.max_profit[symbol]
                    if symbol in individual.partial_profit_taken: del individual.partial_profit_taken[symbol]

        except Exception as e:
            logging.error(f"Error processing grid individual: {str(e)}")

    def run_trading_cycle(self):
        """Main trading loop"""
        while True:
            try:
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

                    for individual in self.population:
                        self._process_individual(symbol, individual, current_state)

                self.generation += 1
                self.deal_count += 1
                logging.info(
                    f"Generation {self.generation}, Best fitness: {max(ind.fitness for ind in self.population)}"
                )

                # Remove inactive orders
                for symbol in self.symbols:
                    orders = mt5.orders_get(symbol=symbol)
                    if orders:
                        for order in orders:
                            if (
                                time.time() - order.time_setup
                            ) > 60:  # Older than 1 minute
                                request = {
                                    "action": mt5.TRADE_ACTION_REMOVE,
                                    "order": order.ticket,
                                    "magic": 123456,
                                }
                                mt5.order_send(request)

                # Save the state every 5 generations
                if self.generation % 50 == 0:
                    self._save_to_db()

                time.sleep(300)  # 5 minute pause between loops

            except Exception as e:
                logging.error(f"Trading cycle error: {str(e)}")
                time.sleep(60)


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
    trader = HybridGridTrader(symbols)
    trader.run_trading_cycle()


if __name__ == "__main__":
    main()
