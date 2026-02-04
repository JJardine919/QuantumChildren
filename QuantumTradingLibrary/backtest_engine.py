
import logging
import random
import numpy as np
import pandas as pd

from etare_module import HybridTrader, Action, TradingIndividual

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s][%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class BacktestEngine:
    def __init__(self, population: list[TradingIndividual], features: np.ndarray, price_data: pd.DataFrame, chaos_level: float = 0.5):
        """
        Initializes the backtesting engine with prop firm evaluation rules.
        """
        self.population = population
        self.features = features
        self.prices = price_data['close'].values
        self.times = price_data['time'].values
        self.chaos_level = chaos_level
        self.num_individuals = len(population)
        self.timesteps = len(features)

        # --- Prop Firm Rules ---
        self.initial_balance = 200_000
        self.max_total_loss_pct = 0.06
        self.max_daily_loss_pct = 0.03
        self.max_total_drawdown = self.initial_balance * self.max_total_loss_pct
        self.max_daily_drawdown = self.initial_balance * self.max_daily_loss_pct
        
        # --- State for each individual ---
        self.portfolios = []
        for _ in range(self.num_individuals):
            self.portfolios.append({
                'equity': self.initial_balance,
                'daily_start_equity': self.initial_balance,
                'position': None, # 'LONG' or 'SHORT'
                'entry_price': 0,
                'trades': [],
                'is_active': True, # Becomes false if rules are broken
                'fail_reason': None
            })

    def run(self):
        """
        Runs the backtest simulation adhering to prop firm rules.
        """
        logging.info(f"Running backtest with Prop Firm rules for {self.num_individuals} individuals...")
        
        current_day = None

        for t in range(self.timesteps):
            # --- Day Change Logic ---
            timestamp_day = self.times[t].date()
            if timestamp_day != current_day:
                current_day = timestamp_day
                for portfolio in self.portfolios:
                    portfolio['daily_start_equity'] = portfolio['equity']

            current_price = self.prices[t]
            current_features = self.features[t]

            for i, individual in enumerate(self.population):
                portfolio = self.portfolios[i]

                # Skip individuals who have already failed
                if not portfolio['is_active']:
                    continue
                
                # --- Daily Loss Check (before new trades) ---
                current_daily_drawdown = portfolio['daily_start_equity'] - portfolio['equity']
                if current_daily_drawdown > self.max_daily_drawdown:
                    portfolio['is_active'] = False
                    portfolio['fail_reason'] = f"Daily Loss Limit Hit on {current_day}"
                    continue

                action, _ = individual.predict(current_features)

                # --- Trade Execution ---
                if portfolio['position'] == 'LONG' and action == Action.OPEN_SELL:
                    self._close_position(i, current_price)
                elif portfolio['position'] == 'SHORT' and action == Action.OPEN_BUY:
                    self._close_position(i, current_price)
                
                if portfolio['position'] is None:
                    if action == Action.OPEN_BUY:
                        self._open_position(i, 'LONG', current_price)
                    elif action == Action.OPEN_SELL:
                        self._open_position(i, 'SHORT', current_price)

        # Final check and performance calculation
        for i in range(self.num_individuals):
            if self.portfolios[i]['is_active'] and self.portfolios[i]['position'] is not None:
                self._close_position(i, self.prices[-1])

        self._calculate_performance()
        return self.population

    def _open_position(self, idx, position_type, price):
        self.portfolios[idx].update({'position': position_type, 'entry_price': price})

    def _close_position(self, idx, price):
        portfolio = self.portfolios[idx]
        entry_price = portfolio['entry_price']
        
        slippage_loss = 1.0 + (random.uniform(0, 0.2) * self.chaos_level)
        slippage_win = 1.0 - (random.uniform(0, 0.05) * self.chaos_level)

        pnl = (price - entry_price) if portfolio['position'] == 'LONG' else (entry_price - price)
        pnl *= slippage_win if pnl >= 0 else slippage_loss

        portfolio['trades'].append(pnl)
        portfolio['equity'] += pnl
        portfolio.update({'position': None, 'entry_price': 0})
        
        # --- Max Loss Check (after every trade) ---
        if portfolio['equity'] < (self.initial_balance - self.max_total_drawdown):
            portfolio['is_active'] = False
            portfolio['fail_reason'] = "Max Loss Limit Hit"
            
    def _calculate_performance(self):
        logging.info("Calculating final performance metrics with rule violations...")
        for i, individual in enumerate(self.population):
            portfolio = self.portfolios[i]
            
            if not portfolio['is_active']:
                # Heavy penalty for failing the challenge
                individual.fitness = -1_000_000 
                logging.info(f"Individual {i}: FAILED ({portfolio['fail_reason']}). Fitness: {individual.fitness}")
                continue

            trades = portfolio['trades']
            if not trades:
                individual.fitness = 0
                continue

            total_profit = sum(trades)
            num_trades = len(trades)
            wins = [t for t in trades if t > 0]
            win_rate = len(wins) / num_trades if num_trades > 0 else 0
            
            # A good fitness is simply the final profit if rules are respected
            fitness = total_profit
            individual.fitness = fitness
            
            individual.total_profit = total_profit
            individual.win_rate = win_rate
            individual.num_trades = num_trades
            logging.info(f"Individual {i}: PASSED. Fitness/Profit={fitness:.2f}, Win Rate={win_rate:.2%}, Trades={num_trades}")


def run_backtest(trader: HybridTrader, features: np.ndarray, price_data: pd.DataFrame, chaos_level: float) -> HybridTrader:
    """
    Wrapper function to be called from the main trainer script.
    """
    engine = BacktestEngine(trader.population, features, price_data, chaos_level)
    trader.population = engine.run()
    return trader

