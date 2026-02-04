"""
Competition Expert Training
===========================
4 experts: 2 bullish, 2 bearish
6-month cycle: 5 months train / 1 month test
Symbols: BTCUSD, ETHUSD
Timeframes: M1, M5
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from pathlib import Path
# Removed sklearn - not needed for this training

# GPU SETUP - AMD RX 6800 XT via DirectML
DEVICE = torch_directml.device()
print(f"Using GPU: {torch_directml.device_name(0)}")

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR / '01_Systems' / 'BioNeuralTrader'))

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    'symbols': ['BTCUSD', 'ETHUSD'],
    'timeframes': {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
    },
    'train_months': 5,
    'test_months': 1,
    'output_dir': SCRIPT_DIR / 'competition_experts',
}

# ==============================================================================
# WALKFORWARD EXPERT (Feedforward NN - 20 inputs, 64 hidden, 3 outputs)
# ==============================================================================
class WalkForwardExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)  # BUY, SELL, HOLD
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_from_json(self, json_path):
        """Load weights from WalkForward JSON format"""
        with open(json_path) as f:
            data = json.load(f)

        with torch.no_grad():
            # Input weights: 20 inputs -> 64 hidden
            input_w = np.array(data['input_weights'])
            if input_w.shape == (20, 64):
                self.fc1.weight.copy_(torch.tensor(input_w.T, dtype=torch.float32))
            elif input_w.shape == (64, 20):
                self.fc1.weight.copy_(torch.tensor(input_w, dtype=torch.float32))
            else:
                # Reshape as needed
                self.fc1.weight.copy_(torch.tensor(input_w.reshape(64, 20), dtype=torch.float32))

            # Hidden weights: 64 -> 64
            hidden_w = np.array(data['hidden_weights'])
            self.fc2.weight.copy_(torch.tensor(hidden_w.reshape(64, 64), dtype=torch.float32))

            # Output weights: 64 -> 3
            output_w = np.array(data['output_weights'])
            self.fc3.weight.copy_(torch.tensor(output_w.reshape(3, 64), dtype=torch.float32))

            # Biases
            self.fc2.bias.copy_(torch.tensor(data['hidden_bias'], dtype=torch.float32))
            self.fc3.bias.copy_(torch.tensor(data['output_bias'], dtype=torch.float32))

    def predict(self, features, device=None):
        """Returns: 0=BUY, 1=SELL, 2=HOLD"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            if device:
                x = x.to(device)
            out = self.forward(x)
            probs = torch.softmax(out, dim=1)
            return torch.argmax(probs).item(), probs[0].cpu().numpy()


# ==============================================================================
# FEATURE CALCULATION (20 features for WalkForward experts)
# ==============================================================================
def calculate_features_20(df):
    """Calculate 20 features for WalkForward expert"""
    if len(df) < 30:
        return np.zeros(20)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['tick_volume'].values

    features = []

    # Price changes (4)
    features.append((close[-1] - close[-2]) / close[-2] * 100)  # 1-bar return
    features.append((close[-1] - close[-5]) / close[-5] * 100)  # 5-bar return
    features.append((close[-1] - close[-10]) / close[-10] * 100)  # 10-bar return
    features.append((close[-1] - close[-20]) / close[-20] * 100)  # 20-bar return

    # RSI (1)
    delta = np.diff(close[-15:])
    gain = np.mean(delta[delta > 0]) if len(delta[delta > 0]) > 0 else 0
    loss = -np.mean(delta[delta < 0]) if len(delta[delta < 0]) > 0 else 0.001
    rsi = 100 - (100 / (1 + gain/loss))
    features.append((rsi - 50) / 50)  # Normalize to [-1, 1]

    # MACD (2)
    ema12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
    ema26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
    macd = ema12 - ema26
    signal = pd.Series(close).ewm(span=12).mean().ewm(span=9).mean().iloc[-1]
    features.append(macd / close[-1] * 100)
    features.append((macd - signal) / close[-1] * 100)

    # Bollinger position (1)
    sma20 = np.mean(close[-20:])
    std20 = np.std(close[-20:])
    bb_pos = (close[-1] - sma20) / (std20 * 2 + 0.0001)
    features.append(np.clip(bb_pos, -1, 1))

    # ATR normalized (1)
    h14 = high[-14:]
    l14 = low[-14:]
    c14 = close[-14:]
    tr1 = h14 - l14
    tr2 = np.abs(h14[1:] - c14[:-1])
    tr3 = np.abs(l14[1:] - c14[:-1])
    tr = np.maximum(tr1[1:], np.maximum(tr2, tr3))
    atr = np.mean(tr)
    features.append(atr / close[-1] * 100)

    # Momentum (2)
    features.append((close[-1] - close[-10]) / close[-10] * 100)
    features.append((close[-1] - close[-5]) / (close[-10] - close[-15] + 0.0001))

    # Volume (2)
    vol_sma = np.mean(volume[-20:])
    features.append(volume[-1] / (vol_sma + 1) - 1)
    features.append(np.mean(volume[-5:]) / (vol_sma + 1) - 1)

    # Trend strength (2)
    sma5 = np.mean(close[-5:])
    sma20 = np.mean(close[-20:])
    features.append((sma5 - sma20) / sma20 * 100)
    features.append((close[-1] - sma5) / sma5 * 100)

    # High/Low position (2)
    high20 = np.max(high[-20:])
    low20 = np.min(low[-20:])
    range20 = high20 - low20 + 0.0001
    features.append((close[-1] - low20) / range20 * 2 - 1)
    features.append((high[-1] - low[-1]) / range20)

    # Padding to 20
    while len(features) < 20:
        features.append(0.0)

    return np.array(features[:20], dtype=np.float32)


# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_expert(expert, train_df, direction='both', epochs=10):
    """
    Train expert on data using GPU
    direction: 'bull' (only learns from up moves), 'bear' (only down), 'both'
    """
    expert = expert.to(DEVICE)
    optimizer = optim.Adam(expert.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    expert.train()
    total_loss = 0
    samples = 0

    for epoch in range(epochs):
        indices = list(range(30, len(train_df) - 1))
        np.random.shuffle(indices)

        for i in indices:
            features = calculate_features_20(train_df.iloc[:i+1])
            future_return = (train_df['close'].iloc[i+1] - train_df['close'].iloc[i]) / train_df['close'].iloc[i]

            # Determine target
            if future_return > 0.001:
                target = 0  # BUY
            elif future_return < -0.001:
                target = 1  # SELL
            else:
                target = 2  # HOLD

            # Filter by direction
            if direction == 'bull' and target == 1:
                continue  # Skip sell signals for bull expert
            if direction == 'bear' and target == 0:
                continue  # Skip buy signals for bear expert

            # Forward pass - GPU
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            y = torch.tensor([target], dtype=torch.long).to(DEVICE)

            optimizer.zero_grad()
            out = expert(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            samples += 1

    return total_loss / max(samples, 1)


def test_expert(expert, test_df, direction='both'):
    """Test expert and return win rate"""
    expert = expert.to(DEVICE)
    expert.eval()

    correct = 0
    total = 0
    trades = []

    for i in range(30, len(test_df) - 1):
        features = calculate_features_20(test_df.iloc[:i+1])
        action, probs = expert.predict(features, device=DEVICE)

        # Filter by direction
        if direction == 'bull' and action == 1:
            continue
        if direction == 'bear' and action == 0:
            continue
        if action == 2:  # HOLD
            continue

        future_return = (test_df['close'].iloc[i+1] - test_df['close'].iloc[i]) / test_df['close'].iloc[i]

        # Check if correct
        if action == 0 and future_return > 0:  # BUY and price went up
            correct += 1
            trades.append(('BUY', future_return, 'WIN'))
        elif action == 1 and future_return < 0:  # SELL and price went down
            correct += 1
            trades.append(('SELL', future_return, 'WIN'))
        else:
            trades.append(('BUY' if action == 0 else 'SELL', future_return, 'LOSS'))

        total += 1

    win_rate = correct / total * 100 if total > 0 else 0
    return win_rate, total, trades


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("COMPETITION EXPERT TRAINING")
    print("=" * 60)
    print(f"Symbols: {CONFIG['symbols']}")
    print(f"Timeframes: M1, M5")
    print(f"Train: {CONFIG['train_months']} months / Test: {CONFIG['test_months']} month")
    print()

    # Initialize MT5
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    # Create output directory
    CONFIG['output_dir'].mkdir(exist_ok=True)

    # Load base WalkForward experts
    wf_experts_path = Path(r"C:\Users\jjj10\ETARE_WalkForward\elite_experts_70plus")

    experts = {
        'bull_1': {'name': 'WF_C7_E36_Bull', 'base': 'expert_C7_E36_WR72.json', 'direction': 'bull'},
        'bull_2': {'name': 'WF_C7_E49_Bull', 'base': 'expert_C7_E49_WR73.json', 'direction': 'bull'},
        'bear_1': {'name': 'WF_C7_E3_Bear', 'base': 'expert_C7_E3_WR71.json', 'direction': 'bear'},
        'bear_2': {'name': 'WF_C7_E8_Bear', 'base': 'expert_C7_E8_WR72.json', 'direction': 'bear'},
    }

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * 6)  # 6 months
    train_end = end_date - timedelta(days=30 * CONFIG['test_months'])

    print(f"Data range: {start_date.date()} to {end_date.date()}")
    print(f"Train: {start_date.date()} to {train_end.date()}")
    print(f"Test: {train_end.date()} to {end_date.date()}")
    print()

    results = {}

    for symbol in CONFIG['symbols']:
        print(f"\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print(f"{'='*60}")

        for tf_name, tf_val in CONFIG['timeframes'].items():
            print(f"\n--- Timeframe: {tf_name} ---")

            # Get data
            rates = mt5.copy_rates_range(symbol, tf_val, start_date, end_date)
            if rates is None or len(rates) < 100:
                print(f"  No data for {symbol} {tf_name}")
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Split train/test
            train_df = df[df.index < train_end]
            test_df = df[df.index >= train_end]

            print(f"  Train samples: {len(train_df)}")
            print(f"  Test samples: {len(test_df)}")

            for exp_key, exp_config in experts.items():
                print(f"\n  Training {exp_config['name']}...")

                # Create and load expert
                expert = WalkForwardExpert()
                base_path = wf_experts_path / exp_config['base']
                if base_path.exists():
                    expert.load_from_json(base_path)
                    print(f"    Loaded base weights from {exp_config['base']}")

                # Train
                loss = train_expert(expert, train_df, direction=exp_config['direction'], epochs=5)
                print(f"    Train loss: {loss:.4f}")

                # Test
                win_rate, num_trades, trades = test_expert(expert, test_df, direction=exp_config['direction'])
                print(f"    Test WR: {win_rate:.1f}% ({num_trades} trades)")

                # Save
                save_name = f"{exp_config['name']}_{symbol}_{tf_name}.pth"
                save_path = CONFIG['output_dir'] / save_name
                torch.save(expert.state_dict(), save_path)

                # Track results
                key = f"{exp_config['name']}_{symbol}_{tf_name}"
                results[key] = {
                    'win_rate': win_rate,
                    'trades': num_trades,
                    'direction': exp_config['direction'],
                    'model_path': str(save_path)
                }

    mt5.shutdown()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - RESULTS")
    print("=" * 60)

    # Sort by win rate
    sorted_results = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)

    print(f"\n{'Expert':<40} {'Dir':<6} {'WR':>8} {'Trades':>8}")
    print("-" * 65)
    for name, data in sorted_results:
        print(f"{name:<40} {data['direction']:<6} {data['win_rate']:>7.1f}% {data['trades']:>8}")

    # Save results
    results_path = CONFIG['output_dir'] / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Best bull and bear
    bulls = [(k, v) for k, v in sorted_results if v['direction'] == 'bull']
    bears = [(k, v) for k, v in sorted_results if v['direction'] == 'bear']

    if bulls:
        print(f"\nBEST BULL: {bulls[0][0]} ({bulls[0][1]['win_rate']:.1f}%)")
    if bears:
        print(f"BEST BEAR: {bears[0][0]} ({bears[0][1]['win_rate']:.1f}%)")


if __name__ == "__main__":
    main()
