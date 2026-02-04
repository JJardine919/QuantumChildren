# ETARE - Evolutionary Trading Algorithm with Reinforcement and Extinction

## Overview

ETARE (Evolutionary Trading Algorithm with Reinforcement and Extinction) is a hybrid trading system that combines genetic algorithms, reinforcement learning, and evolutionary principles inspired by Darwin's natural selection. The system maintains a population of trading strategies that evolve over time, with weak strategies being eliminated and strong ones breeding to create improved offspring.

## Technical Specifications

| Attribute | Value |
|-----------|-------|
| **Computing Paradigm** | Hybrid: Genetic Algorithms + Reinforcement Learning |
| **Framework** | NumPy (manual neural networks) + SQLite |
| **Algorithm** | Evolutionary + Q-Learning (manual implementation) |
| **Hardware** | CPU (no GPU required for manual NN) |
| **Training Required** | Continuous (online learning) |
| **Real-time Capable** | Yes (5-minute cycle) |
| **Input Format** | Technical indicators (RSI, MACD, BB, EMAs, etc.) |
| **Output Format** | Trading actions (Buy/Sell/Close) |

## Key Features

### 1. Evolutionary Architecture
- **Population-based:** Maintains 50 trading strategies simultaneously
- **Natural selection:** Best strategies survive and reproduce
- **Genetic crossover:** Combines successful parent strategies
- **Mutation:** Random variations create diversity
- **Mass extinction:** Periodic elimination of weak strategies

### 2. Reinforcement Learning
- **Priority experience replay:** Remembers important trades
- **Q-learning:** Manual implementation without PyTorch/TensorFlow
- **Adaptive epsilon-greedy:** Balances exploration vs exploitation
- **Continuous learning:** Updates weights after every trade

### 3. Grid Trading System
- **Adaptive grid parameters:** Grid step, order count, volumes
- **DCA (Dollar Cost Averaging):** Decreasing lot sizes
- **Dynamic profit targets:** Close when grid reaches target profit
- **Multi-symbol support:** Trades 20+ currency pairs simultaneously

### 4. Risk Management
- **Position limits:** Max positions per symbol
- **Profit-based closing:** Closes grids at specified profit levels
- **Volume reduction:** Each additional order has smaller volume
- **Time-based cleanup:** Removes stale pending orders

## System Architecture

```
┌─────────────────────────────────────────────┐
│      Population (50 Trading Strategies)     │
└─────────────┬───────────────────────────────┘
              │
              ├──────────────────┬─────────────────┐
              │                  │                 │
    ┌─────────▼─────────┐ ┌─────▼─────┐ ┌────────▼────────┐
    │  Neural Network   │ │   Grid    │ │   RL Memory     │
    │  (3 layers)       │ │ Parameters│ │  (Experience)   │
    │  128 → 64 → 6     │ │           │ │                 │
    └─────────┬─────────┘ └─────┬─────┘ └────────┬────────┘
              │                  │                 │
              │                  │                 │
              └──────────┬───────┴─────────────────┘
                         │
                  ┌──────▼──────────┐
                  │ Decision Making │
                  │  (6 Actions)    │
                  └──────┬──────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼─────┐   ┌──────▼─────┐  ┌─────▼──────┐
   │ Open Buy │   │ Open Sell  │  │ Close Grid │
   │   Grid   │   │   Grid     │  │ (Profit)   │
   └──────────┘   └────────────┘  └────────────┘
```

## Components

### TradingIndividual
Base class for a single trading strategy.

**Neural Network Architecture:**
```
Input Layer:  100+ features → 128 neurons
Hidden Layer: 128 neurons → 64 neurons
Output Layer: 64 neurons → 6 actions
```

**Actions:**
- `OPEN_BUY` - Open new buy grid
- `OPEN_SELL` - Open new sell grid
- `CLOSE_BUY_PROFIT` - Close buy grid with profit
- `CLOSE_BUY_LOSS` - Close buy grid with loss
- `CLOSE_SELL_PROFIT` - Close sell grid with profit
- `CLOSE_SELL_LOSS` - Close sell grid with loss

**Learning Parameters:**
```python
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
mutation_rate = 0.1
mutation_strength = 0.1
```

### GridTrader (extends TradingIndividual)
Enhanced strategy with grid trading capabilities.

**Grid Parameters:**
- `grid_step`: Distance between grid orders (0.00005-0.0002)
- `orders_count`: Number of orders in grid (3-10)
- `base_volume`: Initial order volume (0.01-0.1 lots)
- `volume_step`: Volume decrease per order (0.01-0.05 lots)

**Grid Logic:**
1. Opens grid of pending orders below/above current price
2. Each subsequent order has decreasing volume
3. Monitors total grid profit
4. Closes entire grid when profit target reached ($2.00+)

### RLMemory (Priority Experience Replay)
Stores trading experiences for learning.

**Capacity:** 10,000 experiences
**Structure:** `(state, action, reward, next_state)`
**Sampling:** Batch size 32 for training

### HybridGridTrader (Main Controller)
Manages population of GridTrader strategies.

**Population Management:**
- Population size: 50 strategies
- Elite size: Top 5 survivors
- Tournament size: 3 for selection
- Extinction interval: Every 10 generations
- Inefficient extinction: Every 5 generations

**Evolution Process:**
1. **Selection:** Tournament-based selection of parents
2. **Crossover:** 80% chance - combine two parents
3. **Mutation:** 20% chance - clone elite with mutations
4. **Extinction:** Periodic removal of bottom 30% performers

## Feature Engineering

### Technical Indicators (20+)

**Trend Indicators:**
- EMA (5, 10, 20, 50 periods)
- MACD (12-26-9)
- Momentum (10-period)

**Oscillators:**
- RSI (14-period)
- Stochastic (K%D)
- CCI (20-period)
- Williams %R
- ROC (Rate of Change)

**Volatility:**
- Bollinger Bands (20-period, 2σ)
- ATR (14-period)
- Price change (absolute and relative)

**Volume:**
- Volume MA (20-period)
- Volume volatility
- Volume standard deviation

**Normalization:**
- All features normalized (mean=0, std=1)
- Clipped to prevent outliers
- Forward/backward fill for missing data

## Database Schema

### Tables

#### population
Stores current generation of strategies.

```sql
CREATE TABLE population (
    id INTEGER PRIMARY KEY,
    individual TEXT,  -- JSON serialized strategy
    fitness REAL,
    successful_trades INTEGER,
    total_trades INTEGER
)
```

#### history
Tracks evolution history.

```sql
CREATE TABLE history (
    id INTEGER PRIMARY KEY,
    generation INTEGER,
    individual_id INTEGER,
    trade_history TEXT,  -- JSON
    total_profit REAL,
    win_rate REAL,
    FOREIGN KEY(individual_id) REFERENCES population(id)
)
```

## Trading Logic

### Main Loop (5-minute cycle)

1. **Check for extinction event**
   - Every 10 generations: Mass extinction
   - Every 5 generations: Remove inefficient strategies

2. **For each symbol:**
   - Fetch latest 100 M5 candles
   - Calculate all technical indicators
   - Normalize features

3. **For each individual strategy:**
   - Predict action using neural network
   - If no grid exists: Consider opening new grid
   - If grid exists: Monitor profit and close if target reached

4. **Cleanup:**
   - Remove pending orders older than 1 minute
   - Save population state every 50 generations
   - Database cleanup (keep last 1000 history entries)

### Grid Management

#### Opening Grid
```python
# For Buy grid: Place orders below current price
for i in range(orders_count):
    volume = base_volume + (i * volume_step)  # Increasing volume
    price = current_price - (i+1) * grid_step
    # Place buy limit order
```

#### Closing Grid
- Monitor total grid profit across all open positions
- **Partial Take Profit:** When profit >= $3.00 (50%), close 50% of positions to lock in gains
- **Rolling Stop Loss:** When profit > $0.90 (1.5x risk), trail stop loss by $0.90 from peak
- When profit >= $6.00: Close entire grid
- When loss >= $0.60: Close entire grid (Stop Loss)
- Close positions (market orders)
- Remove pending orders

### Position Sizing Example

| Order | Base | Step | Calculation | Volume |
|-------|------|------|-------------|--------|
| 1 | 0.10 | 0.01 | 0.10 + 0*0.01 | 0.10 |
| 2 | 0.10 | 0.01 | 0.10 + 1*0.01 | 0.11 |
| 3 | 0.10 | 0.01 | 0.10 + 2*0.01 | 0.12 |
| 4 | 0.10 | 0.01 | 0.10 + 3*0.01 | 0.13 |
| 5 | 0.10 | 0.01 | 0.10 + 4*0.01 | 0.14 |

**Total volume in grid:** 0.60 lots

## Performance Metrics

**Tracked Metrics:**
- Fitness (total profit)
- Total trades
- Successful trades
- Win rate
- Profit factor
- Average trade duration
- Worst/best trades
- Drawdown

**Evaluation:**
```python
fitness = (
    profit_factor * 0.3 +
    drawdown_resistance * 0.2 +
    consistency * 0.2 +
    (1 - abs(market_correlation)) * 0.1 +
    adaptability * 0.2
)
```

## Evolution Mechanisms

### Tournament Selection
1. Randomly select 3 individuals
2. Choose the one with highest fitness
3. Use as parent for reproduction

### Crossover (80% probability)
1. Select two parents via tournament
2. For each weight matrix:
   - Generate random mask
   - Child inherits weights: 50% from parent1, 50% from parent2
3. Grid parameters: Randomly inherit from one parent

### Mutation (20% probability)
1. Clone elite individual
2. Mutate neural network weights:
   - 10% of weights changed
   - Add Gaussian noise
3. Mutate grid parameters:
   - Grid step ± 0.00005
   - Base volume ± mutation_strength
   - Volume step ± mutation_strength
   - Orders count ± 1

### Extinction Events

#### Mass Extinction (Every 10 generations)
1. Sort population by fitness
2. Keep top 5 (elite)
3. Regenerate rest via crossover/mutation

#### Inefficient Extinction (Every 5 generations)
1. Calculate performance metrics for each strategy
2. Remove strategies with:
   - Profit factor < 1.5
   - Win rate < 0.6
3. Replace with new individuals (inherit patterns from survivors)

## Usage

### Basic Setup

```python
from ETARE_module import HybridGridTrader, initialize_mt5

# Initialize MT5
if not initialize_mt5():
    print("MT5 initialization failed")
    exit()

# Define symbols to trade
symbols = [
    'EURUSD.ecn', 'GBPUSD.ecn', 'USDJPY.ecn',
    'AUDUSD.ecn', 'USDCHF.ecn', 'USDCAD.ecn'
    # ... up to 20+ pairs
]

# Create trader
trader = HybridGridTrader(symbols, population_size=50)

# Run forever (5-min cycles)
trader.run_trading_cycle()
```

### Parameters Tuning

```python
class HybridGridTrader:
    def __init__(self, symbols, population_size=50):
        self.population_size = population_size  # Number of strategies
        self.tournament_size = 3  # Tournament selection size
        self.elite_size = 5  # Number of elites to preserve
        self.extinction_rate = 0.3  # Bottom 30% removed
        self.extinction_interval = 10  # Mass extinction every 10 gen
        self.inefficient_extinction_interval = 5  # Cleanup every 5 gen
```

### Database Recovery

On startup, ETARE automatically:
1. Loads last saved population from database
2. Restores neural network weights
3. Restores grid parameters
4. Continues from last generation number
5. If no database: Starts fresh population

```python
# Database files
trading_history.db  # SQLite database (auto-created)
```

## Strengths

1. **Self-Learning:** Continuously adapts without retraining
2. **Evolutionary:** Weak strategies automatically eliminated
3. **Diverse:** Population maintains multiple approaches
4. **Persistent:** State saved to database, recovers from crashes
5. **Scalable:** Handles 20+ symbols simultaneously
6. **No External ML:** Doesn't require PyTorch/TensorFlow
7. **Grid Trading:** Leverages market oscillations
8. **Risk-Controlled:** Volume decreases with more positions

## Weaknesses

1. **CPU-Only:** Manual neural network (not GPU-accelerated)
2. **Slow Convergence:** Evolution takes many generations
3. **Memory Usage:** Stores 10,000 experiences per strategy
4. **Database Growth:** History table grows over time
5. **No Stop Loss:** Relies on grid profit targets
6. **Fixed Timeframe:** Locked to M5 (5-minute candles)
7. **Parameter Sensitivity:** Grid parameters need tuning per symbol

## Integration Points

- **Input:** MT5 OHLC data (5-minute timeframe)
- **Output:** Trading decisions (buy/sell grids), fitness scores
- **Can Feed:** Ensemble systems, performance analyzers
- **Can Receive:** External market signals, volatility filters
- **Database:** SQLite (easily integrated with other systems)

## Best Practices

1. **Start Small:**
   - Test with 1-2 symbols first
   - Use demo account initially
   - Monitor for at least 50 generations

2. **Monitor Performance:**
   - Check fitness trends
   - Analyze win rate
   - Review grid profitability
   - Watch for database growth

3. **Tune Parameters:**
   - Adjust grid_step for symbol volatility
   - Modify profit targets based on spread
   - Increase population_size for more diversity

4. **Database Maintenance:**
   - Periodic cleanup (built-in every save)
   - Backup database before major changes
   - Monitor disk space

5. **Risk Management:**
   - Limit total symbols (start with 5-10)
   - Use appropriate account size
   - Monitor total exposure across all grids

## Evolution Stages

### Generation 1-10 (Exploration)
- Random strategies exploring market
- High diversity, low fitness
- Extinction events frequent
- Learning basic patterns

### Generation 11-50 (Convergence)
- Strategies start to specialize
- Fitness improves steadily
- Survivors breed successful patterns
- Grid parameters optimize

### Generation 51-100 (Maturity)
- Stable population with proven strategies
- Adaptive to market changes
- Consistent profitability
- Fine-tuned grid parameters

### Generation 100+ (Maintenance)
- Periodic extinctions prevent stagnation
- Continuous adaptation
- Handles regime changes
- Long-term profitability

## Troubleshooting

### Low Fitness (All Strategies Losing)
- Reduce number of symbols
- Adjust grid parameters (smaller steps)
- Lower profit targets
- Check spreads and commissions

### Database Errors
- Check disk space
- Backup and recreate database
- Verify SQLite version
- Check file permissions

### MT5 Connection Issues
- Verify MT5 terminal running
- Check symbol names (add .ecn suffix if needed)
- Ensure account has margin for grids
- Check firewall/antivirus

### No Orders Placed
- Check MT5 terminal logs
- Verify symbol liquidity
- Check minimum lot size
- Review deviation parameter

### Stale Pending Orders
- Auto-cleanup runs every cycle
- Check order time_setup
- Review MT5 order history
- Manually remove if needed

## Part of Midas Ecosystem

ETARE is one of **24 modules** in the larger Midas trading ecosystem. It operates as an independent module but can be integrated with other Midas components for:
- Signal aggregation
- Risk management
- Portfolio optimization
- Multi-strategy coordination

## Future Enhancements

- [ ] Multi-timeframe analysis
- [ ] GPU acceleration (convert to PyTorch)
- [ ] Adaptive extinction intervals
- [ ] Risk-based position sizing
- [ ] Correlation filtering
- [ ] News event detection
- [ ] Advanced feature engineering
- [ ] Transfer learning between symbols

---

**System Type:** Hybrid Evolutionary + Reinforcement Learning
**Hardware:** CPU
**Training Required:** Continuous (online)
**Real-time Capable:** Yes (5-min cycles)
**Status:** Production Ready ✓
**Part of:** Midas Ecosystem (Module 1 of 24)
