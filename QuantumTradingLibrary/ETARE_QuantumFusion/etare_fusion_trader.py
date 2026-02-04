import sys
import os
import json
import numpy as np
import MetaTrader5 as mt5
import time
from datetime import datetime
import pandas as pd

# Add the project modules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'modules'))

from signal_fusion import SignalFusionEngine

# ==============================================================================
# CONFIGURATION (Mirrors the Champion Expert)
# ==============================================================================

CONFIG = {
    'account': 1512287880,
    'password': '1a3Q@fT24@LEw',
    'server': 'FTMO-Demo',
    'symbol': 'BTCUSD',
    'timeframe': mt5.TIMEFRAME_M5,
    'magic_number': 73049,  # New magic for Fusion System
    'expert_file': 'C:/Users/jjj10/ETARE_WalkForward/elite_experts_70plus/expert_C7_E49_WR73.json',
    'forbidden_hours': [13, 16, 21, 22],
    'grid_step': 50.0,
    'orders_count': 5,
    'base_volume': 0.01,
    'volume_step': 0.01,
    'profit_target': 10.0,
    'max_grids': 2,
    'check_interval': 300,
    
    # Fusion Settings
    'fusion_config': os.path.join(script_dir, 'config/config.yaml'),
    'veto_threshold': 0.25, # If Fusion disagrees by more than this, cancel trade
}

# ==============================================================================
# REUSED CLASSES FROM CHAMPION
# ==============================================================================

class EliteExpert:
    def __init__(self, weights_file):
        with open(weights_file, 'r') as f:
            data = json.load(f)
        self.input_weights = np.array(data['input_weights'])
        self.hidden_weights = np.array(data['hidden_weights'])
        self.output_weights = np.array(data['output_weights'])

    def relu(self, x): return np.maximum(0, x)
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def predict(self, features):
        hidden = self.relu(np.dot(features, self.input_weights))
        output = np.dot(hidden, self.hidden_weights)
        output = np.dot(output, self.output_weights)
        action_probs = self.softmax(output)
        return np.argmax(action_probs), action_probs[np.argmax(action_probs)]

# Indicator logic (simplified for clarity but matches original)
def calculate_indicators(df):
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
    df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['momentum'] = df['close'] - df['close'].shift(10)
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
    df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
    df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
    df['volume_std'] = df['tick_volume'].rolling(window=20).std()
    df['price_change'] = df['close'].diff()
    return df

def extract_features(df):
    latest = df.iloc[-1]
    features = [
        (latest['close'] - latest['ema_5']) / latest['close'],
        (latest['close'] - latest['ema_10']) / latest['close'],
        (latest['close'] - latest['ema_20']) / latest['close'],
        (latest['close'] - latest['ema_50']) / latest['close'],
        latest['macd'] / latest['close'], latest['macd_signal'] / latest['close'], latest['macd_hist'] / latest['close'],
        latest['rsi'] / 100.0, latest['stoch_k'] / 100.0, latest['stoch_d'] / 100.0,
        np.clip(latest['cci'] / 100.0, -1, 1), (latest['williams_r'] + 100) / 100.0, latest['roc'] / 100.0,
        (latest['close'] - latest['bb_mid']) / latest['bb_mid'], (latest['bb_upper'] - latest['bb_lower']) / latest['bb_mid'],
        latest['atr'] / latest['close'], latest['momentum'] / latest['close'], latest['price_change'] / latest['close'],
        (latest['tick_volume'] - latest['volume_ma']) / latest['volume_ma'] if latest['volume_ma'] > 0 else 0,
        latest['volume_std'] / latest['volume_ma'] if latest['volume_ma'] > 0 else 0,
    ]
    return np.array(features, dtype=np.float32)

# ==============================================================================
# FUSION-AIDED EXECUTION
# ==============================================================================

def main():
    print("\033[95m" + "="*80)
    print("      STRIKE BOSS - ETARE QUANTUM FUSION TRADER V1.0")
    print("="*80 + "\033[0m")

    if not mt5.initialize(): return
    if not mt5.login(CONFIG['account'], password=CONFIG['password'], server=CONFIG['server']):
        print("Login Failed"); mt5.shutdown(); return

    expert = EliteExpert(CONFIG['expert_file'])
    fusion_engine = SignalFusionEngine(CONFIG['fusion_config'])

    print(f"\n[OK] Systems Synchronized. Monitoring {CONFIG['symbol']}...")

    while True:
        try:
            # 1. Get Data
            rates = mt5.copy_rates_from_pos(CONFIG['symbol'], CONFIG['timeframe'], 0, 100)
            if rates is None: time.sleep(10); continue
            df = calculate_indicators(pd.DataFrame(rates))
            
            # 2. Get Neural Expert Prediction
            action, confidence = expert.predict(extract_features(df))
            
            # 3. Get Quantum Fusion Veto
            # Note: Fusion engine uses same symbol/timeframe but broader multi-system context
            fused = fusion_engine.get_fused_signal(CONFIG['symbol'], CONFIG['timeframe'])
            composite_score = fused['composite_score']
            regime = fused['regime']['regime']
            
            # Action Mapping
            action_name = {0:"BUY", 1:"SELL"}.get(action, "CLOSE/WAIT")
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ANALYSIS REPORT:")
            print(f"  Expert: {action_name} (Conf: {confidence:.2%})")
            print(f"  Fusion: Score {composite_score:.4f} (Regime: {regime})")

            # 4. Apply Veto Logic
            veto_triggered = False
            
            # Threshold adjustment for Choppy markets
            effective_threshold = CONFIG['veto_threshold']
            if regime == "CHOPPY":
                effective_threshold *= 0.7 # Be more aggressive with vetos in choppy markets
                print(f"  [!] Regime is CHOPPY. Veto sensitivity increased.")

            if action == 0: # Neural Expert wants to BUY
                if composite_score < -effective_threshold:
                    veto_triggered = True
                    print(f"  \033[91m[VETO] Blocked BUY signal. Fusion says market is Bearish/Complex.\033[0m")
            
            elif action == 1: # Neural Expert wants to SELL
                if composite_score > effective_threshold:
                    veto_triggered = True
                    print(f"  \033[91m[VETO] Blocked SELL signal. Fusion says market is Bullish/Complex.\033[0m")

            # 5. Execute (Standard Grid Logic)
            current_price = df.iloc[-1]['close']
            
            # (Grid management code would go here - for now we log the decision)
            if not veto_triggered and action in [0, 1] and confidence > 0.3:
                print(f"  \033[92m[TRADE] Signal Validated. Ready to execute {action_name}.\033[0m")
                # actual_execute_grid(action, current_price)
            elif action in [0, 1]:
                print(f"  [WAIT] Signal rejected by Fusion or Low Confidence.")

            time.sleep(CONFIG['check_interval'])

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(10)

    mt5.shutdown()

if __name__ == "__main__":
    main()