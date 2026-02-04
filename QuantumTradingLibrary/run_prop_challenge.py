"""
4-WEEK ASSIMILATED PROP FIRM CHALLENGE SIMULATOR
=================================================
Tests all 50 experts through a standardized prop firm challenge.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='[%(levelname)s][%(asctime)s] %(message)s', datefmt='%H:%M:%S')

ACCOUNT_SIZE = 200_000.0
PROFIT_TARGET_PCT = 0.10
MAX_DAILY_LOSS_PCT = 0.05
MAX_TOTAL_DRAWDOWN_PCT = 0.10
CHALLENGE_DAYS = 28
PROFIT_TARGET = ACCOUNT_SIZE * PROFIT_TARGET_PCT
MAX_DAILY_LOSS = ACCOUNT_SIZE * MAX_DAILY_LOSS_PCT
MAX_TOTAL_DRAWDOWN = ACCOUNT_SIZE * MAX_TOTAL_DRAWDOWN_PCT
CHAOS_LEVEL = 0.5
RISK_PER_TRADE_PCT = 0.01

LOT_SIZE_MULTIPLIERS = {
    'AUDNZD': 100000, 'XAUUSD': 100, 'ETHUSD': 1,
    'EURCAD': 100000, 'GBPUSD': 100000, 'EURNZD': 100000, 'NZDCHF': 100000,
}

BINANCE_SYMBOL_MAP = {
    'ETHUSD': 'ETHUSDT', 'XAUUSD': None, 'AUDNZD': None,
    'EURCAD': None, 'GBPUSD': None, 'EURNZD': None, 'NZDCHF': None,
}

class Action(Enum):
    HOLD = 0
    OPEN_BUY = 1
    OPEN_SELL = 2

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

@dataclass
class ChallengeResult:
    expert_rank: int
    symbol: str
    passed: bool
    final_balance: float
    profit_loss: float
    profit_pct: float
    max_drawdown_hit: float
    max_daily_loss_hit: float
    total_trades: int
    win_rate: float
    fail_reason: Optional[str] = None
    days_survived: int = 0

@dataclass
class PortfolioState:
    equity: float = ACCOUNT_SIZE
    daily_start_equity: float = ACCOUNT_SIZE
    high_water_mark: float = ACCOUNT_SIZE
    position: Optional[str] = None
    entry_price: float = 0.0
    trades: list = field(default_factory=list)
    is_active: bool = True
    fail_reason: Optional[str] = None
    max_dd_hit: float = 0.0
    max_daily_hit: float = 0.0
    current_day: int = 0

def load_expert(expert_path, input_size, hidden_size):
    model = LSTMModel(input_size, hidden_size, output_size=3, num_layers=2)
    state_dict = torch.load(expert_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(model, features):
    state_tensor = torch.from_numpy(features).float().unsqueeze(0)
    with torch.no_grad():
        prediction = model(state_tensor)
    probs = torch.softmax(prediction, dim=1)
    action_idx = torch.argmax(prediction).item()
    confidence = probs[0, action_idx].item()
    return Action(action_idx), confidence

def prepare_features(df):
    data = df.copy()
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    data['rsi'] = 100 - (100 / (1 + rs))
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['bb_middle'] = data['close'].rolling(20).mean()
    data['bb_std'] = data['close'].rolling(20).std()
    data['bb_position'] = (data['close'] - data['bb_middle']) / (data['bb_std'] + 1e-8)
    data['momentum'] = data['close'] / data['close'].shift(10)
    data['atr'] = data['high'].rolling(14).max() - data['low'].rolling(14).min()
    data['price_change'] = data['close'].pct_change()
    feature_cols = ['rsi', 'macd', 'macd_signal', 'bb_position', 'momentum', 'atr', 'price_change', 'close']
    for col in feature_cols:
        if col in data.columns:
            mean = data[col].rolling(100, min_periods=1).mean()
            std = data[col].rolling(100, min_periods=1).std()
            data[col] = (data[col] - mean) / (std + 1e-8)
            data[col] = data[col].clip(-4, 4)
    data = data.fillna(0)
    return data[feature_cols].values

def generate_simulated_data(symbol, days):
    logging.info(f"Generating {days} days of simulated data for {symbol}...")
    params = {
        'AUDNZD': {'base_price': 1.0850, 'daily_vol': 0.004},
        'XAUUSD': {'base_price': 2650.0, 'daily_vol': 0.012},
        'EURCAD': {'base_price': 1.4800, 'daily_vol': 0.005},
        'GBPUSD': {'base_price': 1.2650, 'daily_vol': 0.006},
        'EURNZD': {'base_price': 1.8400, 'daily_vol': 0.006},
        'NZDCHF': {'base_price': 0.5150, 'daily_vol': 0.005},
        'ETHUSD': {'base_price': 3200.0, 'daily_vol': 0.035},
    }
    p = params.get(symbol, {'base_price': 100.0, 'daily_vol': 0.01})
    bars_per_day = 288
    total_bars = days * bars_per_day
    times = pd.date_range(start=datetime.now() - timedelta(days=days), periods=total_bars, freq='5min')
    np.random.seed(hash(symbol) % 2**32)
    returns = np.random.normal(0, p['daily_vol'] / np.sqrt(bars_per_day), total_bars)
    prices = p['base_price'] * np.exp(np.cumsum(returns))
    high = prices * (1 + np.abs(np.random.normal(0, 0.001, total_bars)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.001, total_bars)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = p['base_price']
    return pd.DataFrame({'time': times, 'open': open_prices, 'high': high, 'low': low, 'close': prices, 'tick_volume': np.random.randint(100, 10000, total_bars)})

def fetch_binance_data(symbol, days):
    try:
        from binance.client import Client
        client = Client()
        binance_symbol = BINANCE_SYMBOL_MAP.get(symbol)
        if not binance_symbol:
            return None
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days + 5)
        logging.info(f"Fetching Binance data for {binance_symbol}...")
        klines = client.get_historical_klines(binance_symbol, '5m', start_dt.strftime("%d %b, %Y"), end_dt.strftime("%d %b, %Y"))
        if not klines:
            return None
        columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=columns)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        df = df.rename(columns={'volume': 'tick_volume'})
        for col in ['open', 'high', 'low', 'close', 'tick_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        logging.info(f"Fetched {len(df):,} bars from Binance for {binance_symbol}")
        return df
    except Exception as e:
        logging.warning(f"Binance fetch failed for {symbol}: {e}")
        return None

def get_market_data(symbol, days):
    df = fetch_binance_data(symbol, days)
    if df is None or len(df) < 1000:
        df = generate_simulated_data(symbol, days)
    return df

def close_position(portfolio, current_price, lot_size, multiplier):
    if portfolio.position is None:
        return
    if portfolio.position == 'LONG':
        pnl = (current_price - portfolio.entry_price) * lot_size * multiplier
    else:
        pnl = (portfolio.entry_price - current_price) * lot_size * multiplier
    if pnl > 0:
        slippage = 1.0 - (random.uniform(0, 0.05) * CHAOS_LEVEL)
    else:
        slippage = 1.0 + (random.uniform(0, 0.15) * CHAOS_LEVEL)
    pnl *= slippage
    portfolio.trades.append(pnl)
    portfolio.equity += pnl
    portfolio.position = None
    portfolio.entry_price = 0

def run_challenge(model, symbol, market_data, features):
    portfolio = PortfolioState()
    lot_multiplier = LOT_SIZE_MULTIPLIERS.get(symbol, 100000)
    lot_size = (ACCOUNT_SIZE * RISK_PER_TRADE_PCT) / 100
    prices = market_data['close'].values
    times = market_data['time'].values
    current_date = None
    day_count = 0
    bars_per_day = 288
    max_bars = min(len(features), CHALLENGE_DAYS * bars_per_day)
    for t in range(100, max_bars):
        if not portfolio.is_active:
            break
        bar_date = pd.Timestamp(times[t]).date()
        if bar_date != current_date:
            current_date = bar_date
            day_count += 1
            portfolio.current_day = day_count
            portfolio.daily_start_equity = portfolio.equity
            if day_count > CHALLENGE_DAYS:
                break
        current_price = prices[t]
        current_features = features[t]
        daily_drawdown = portfolio.daily_start_equity - portfolio.equity
        if daily_drawdown > MAX_DAILY_LOSS:
            portfolio.is_active = False
            portfolio.fail_reason = f"DAILY LOSS LIMIT HIT (Day {day_count})"
            portfolio.max_daily_hit = daily_drawdown
            break
        total_drawdown = ACCOUNT_SIZE - portfolio.equity
        if total_drawdown > MAX_TOTAL_DRAWDOWN:
            portfolio.is_active = False
            portfolio.fail_reason = f"MAX DRAWDOWN HIT ({total_drawdown/ACCOUNT_SIZE*100:.1f}%)"
            portfolio.max_dd_hit = total_drawdown
            break
        action, confidence = predict(model, current_features)
        if portfolio.position == 'LONG' and action == Action.OPEN_SELL:
            close_position(portfolio, current_price, lot_size, lot_multiplier)
        elif portfolio.position == 'SHORT' and action == Action.OPEN_BUY:
            close_position(portfolio, current_price, lot_size, lot_multiplier)
        if portfolio.position is None and confidence > 0.55:
            if action == Action.OPEN_BUY:
                portfolio.position = 'LONG'
                portfolio.entry_price = current_price
            elif action == Action.OPEN_SELL:
                portfolio.position = 'SHORT'
                portfolio.entry_price = current_price
        if portfolio.equity > portfolio.high_water_mark:
            portfolio.high_water_mark = portfolio.equity
        current_daily_dd = portfolio.daily_start_equity - portfolio.equity
        current_total_dd = portfolio.high_water_mark - portfolio.equity
        portfolio.max_daily_hit = max(portfolio.max_daily_hit, current_daily_dd)
        portfolio.max_dd_hit = max(portfolio.max_dd_hit, current_total_dd)
    if portfolio.position is not None and len(prices) > 0:
        close_position(portfolio, prices[-1], lot_size, lot_multiplier)
    profit_loss = portfolio.equity - ACCOUNT_SIZE
    profit_pct = profit_loss / ACCOUNT_SIZE * 100
    passed = False
    if portfolio.is_active:
        if profit_loss >= PROFIT_TARGET:
            passed = True
        elif day_count >= CHALLENGE_DAYS:
            portfolio.fail_reason = f"TIME EXPIRED (Profit: {profit_pct:.1f}% of {PROFIT_TARGET_PCT*100}% required)"
    wins = sum(1 for t in portfolio.trades if t > 0)
    total_trades = len(portfolio.trades)
    win_rate = wins / total_trades if total_trades > 0 else 0
    return ChallengeResult(expert_rank=0, symbol=symbol, passed=passed, final_balance=portfolio.equity, profit_loss=profit_loss, profit_pct=profit_pct, max_drawdown_hit=portfolio.max_dd_hit, max_daily_loss_hit=portfolio.max_daily_hit, total_trades=total_trades, win_rate=win_rate, fail_reason=portfolio.fail_reason, days_survived=portfolio.current_day)

def run_all_challenges():
    print("=" * 70)
    print("  4-WEEK ASSIMILATED PROP FIRM CHALLENGE")
    print("=" * 70)
    print(f"\nChallenge Parameters:")
    print(f"  Account Size:     ${ACCOUNT_SIZE:,.0f}")
    print(f"  Profit Target:    {PROFIT_TARGET_PCT*100}% (${PROFIT_TARGET:,.0f})")
    print(f"  Max Daily Loss:   {MAX_DAILY_LOSS_PCT*100}% (${MAX_DAILY_LOSS:,.0f})")
    print(f"  Max Drawdown:     {MAX_TOTAL_DRAWDOWN_PCT*100}% (${MAX_TOTAL_DRAWDOWN:,.0f})")
    print(f"  Duration:         {CHALLENGE_DAYS} days (4 weeks)")
    print(f"  Chaos Level:      {CHAOS_LEVEL*100:.0f}%")
    print("=" * 70)
    manifest_path = Path("top_50_experts/top_50_manifest.json")
    if not manifest_path.exists():
        logging.error(f"Manifest not found at {manifest_path}")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    experts = manifest['experts']
    logging.info(f"Loaded manifest with {len(experts)} experts")
    market_data_cache = {}
    features_cache = {}
    results = []
    for expert in experts:
        rank = expert['rank']
        symbol = expert['symbol']
        filename = expert['filename']
        input_size = expert['input_size']
        hidden_size = expert['hidden_size']
        print(f"\n[{rank:02d}/50] Testing {symbol} expert...")
        if symbol not in market_data_cache:
            market_data = get_market_data(symbol, CHALLENGE_DAYS + 10)
            features = prepare_features(market_data)
            market_data_cache[symbol] = market_data
            features_cache[symbol] = features
        else:
            market_data = market_data_cache[symbol]
            features = features_cache[symbol]
        expert_path = Path(f"top_50_experts/{filename}")
        if not expert_path.exists():
            logging.warning(f"Expert file not found: {expert_path}")
            continue
        try:
            model = load_expert(str(expert_path), input_size, hidden_size)
        except Exception as e:
            logging.error(f"Failed to load expert {filename}: {e}")
            continue
        result = run_challenge(model, symbol, market_data, features)
        result.expert_rank = rank
        results.append(result)
        status = "PASSED" if result.passed else "FAILED"
        print(f"  Result: {status}")
        print(f"  P&L: ${result.profit_loss:+,.2f} ({result.profit_pct:+.2f}%)")
        print(f"  Win Rate: {result.win_rate*100:.1f}% ({result.total_trades} trades)")
        print(f"  Max DD: ${result.max_drawdown_hit:,.2f} | Max Daily: ${result.max_daily_loss_hit:,.2f}")
        if result.fail_reason:
            print(f"  Fail Reason: {result.fail_reason}")
    print("\n" + "=" * 70)
    print("  CHALLENGE RESULTS SUMMARY")
    print("=" * 70)
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    print(f"\nOverall Pass Rate: {len(passed)}/{len(results)} ({len(passed)/len(results)*100:.1f}%)")
    if passed:
        print(f"\n{'='*50}")
        print(f"PASSED EXPERTS ({len(passed)}):")
        print(f"{'='*50}")
        for r in sorted(passed, key=lambda x: x.profit_pct, reverse=True):
            print(f"  Rank {r.expert_rank:02d} ({r.symbol}): ${r.profit_loss:+,.0f} ({r.profit_pct:+.1f}%) | WR: {r.win_rate*100:.0f}%")
    print(f"\n{'='*50}")
    print(f"FAILED EXPERTS ({len(failed)}):")
    print(f"{'='*50}")
    fail_reasons = {}
    for r in failed:
        reason = r.fail_reason or "Unknown"
        reason_key = reason.split('(')[0].strip()
        if reason_key not in fail_reasons:
            fail_reasons[reason_key] = []
        fail_reasons[reason_key].append(r)
    for reason, experts_list in fail_reasons.items():
        print(f"\n  {reason}: {len(experts_list)} experts")
        for r in experts_list[:5]:
            print(f"    - Rank {r.expert_rank:02d} ({r.symbol}): ${r.profit_loss:+,.0f} | Day {r.days_survived}")
        if len(experts_list) > 5:
            print(f"    ... and {len(experts_list) - 5} more")
    print(f"\n{'='*50}")
    print("RESULTS BY SYMBOL:")
    print(f"{'='*50}")
    by_symbol = {}
    for r in results:
        if r.symbol not in by_symbol:
            by_symbol[r.symbol] = []
        by_symbol[r.symbol].append(r)
    for symbol, symbol_results in sorted(by_symbol.items()):
        passed_count = sum(1 for r in symbol_results if r.passed)
        total = len(symbol_results)
        avg_pnl = sum(r.profit_loss for r in symbol_results) / total
        avg_wr = sum(r.win_rate for r in symbol_results) / total * 100
        print(f"  {symbol}: {passed_count}/{total} passed | Avg P&L: ${avg_pnl:+,.0f} | Avg WR: {avg_wr:.0f}%")
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {'account_size': ACCOUNT_SIZE, 'profit_target_pct': PROFIT_TARGET_PCT, 'max_daily_loss_pct': MAX_DAILY_LOSS_PCT, 'max_total_drawdown_pct': MAX_TOTAL_DRAWDOWN_PCT, 'challenge_days': CHALLENGE_DAYS, 'chaos_level': CHAOS_LEVEL},
        'summary': {'total_experts': len(results), 'passed': len(passed), 'failed': len(failed), 'pass_rate': len(passed) / len(results) * 100 if results else 0},
        'results': [{'rank': r.expert_rank, 'symbol': r.symbol, 'passed': r.passed, 'profit_loss': r.profit_loss, 'profit_pct': r.profit_pct, 'win_rate': r.win_rate, 'total_trades': r.total_trades, 'max_drawdown': r.max_drawdown_hit, 'max_daily_loss': r.max_daily_loss_hit, 'days_survived': r.days_survived, 'fail_reason': r.fail_reason} for r in results]
    }
    with open('prop_challenge_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: prop_challenge_results.json")
    print("=" * 70)
    return results

if __name__ == '__main__':
    run_all_challenges()
