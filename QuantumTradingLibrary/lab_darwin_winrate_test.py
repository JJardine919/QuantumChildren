"""
MUSHROOM RABIES LAB - DARWIN FEEDFORWARD EDITION
=================================================
Tests the lab on the ACTUAL high-performing experts (73%+ win rate)
from the darwin population (etare_darwin.db).

These experts use feedforward architecture:
  hidden  = tanh(x @ input_weights.T + hidden_bias)
  hidden2 = tanh(hidden @ hidden_weights.T)
  output  = hidden2 @ output_weights.T + output_bias

Rabies/Mushroom/TEQA adapted for feedforward weight matrices.
"""
import sys
import io
import os
import json
import copy
import time
import sqlite3
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).parent.absolute()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][LAB-FF] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================
# FEEDFORWARD MODEL (matches darwin GeneticWeights)
# ============================================================

class FeedforwardExpert(nn.Module):
    """Matches the darwin GeneticWeights inference exactly."""

    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_weights = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.hidden_weights = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.output_weights = nn.Parameter(torch.zeros(output_size, hidden_size))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
        self.output_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        # Normalize per-sample
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        x = (x - mean) / std

        hidden = torch.tanh(x @ self.input_weights.T + self.hidden_bias)
        hidden2 = torch.tanh(hidden @ self.hidden_weights.T)
        output = hidden2 @ self.output_weights.T + self.output_bias
        return output, hidden2  # Return hidden for compression

    def load_from_dict(self, weights_dict):
        """Load from darwin DB format."""
        with torch.no_grad():
            self.input_weights.copy_(torch.FloatTensor(np.array(weights_dict['input_weights'])))
            self.hidden_weights.copy_(torch.FloatTensor(np.array(weights_dict['hidden_weights'])))
            self.output_weights.copy_(torch.FloatTensor(np.array(weights_dict['output_weights'])))
            self.hidden_bias.copy_(torch.FloatTensor(np.array(weights_dict.get('hidden_bias', np.zeros(self.hidden_size)))))
            self.output_bias.copy_(torch.FloatTensor(np.array(weights_dict.get('output_bias', np.zeros(self.output_size)))))

    def to_dict(self):
        return {
            'input_weights': self.input_weights.detach().numpy().tolist(),
            'hidden_weights': self.hidden_weights.detach().numpy().tolist(),
            'output_weights': self.output_weights.detach().numpy().tolist(),
            'hidden_bias': self.hidden_bias.detach().numpy().tolist(),
            'output_bias': self.output_bias.detach().numpy().tolist(),
        }


# ============================================================
# STAGE 1: RABIES - Feedforward Edition
# ============================================================

class FFRabiesInjection:
    """
    Feedforward rabies:
    - AMPLIFY input_weights (hypervigilant to every feature)
    - AMPLIFY hidden_weights (stronger internal processing)
    - SHARPEN output_weights (more decisive signals)
    """

    def __init__(self, aggression=1.5):
        self.aggression = aggression

    def inject(self, weights_dict):
        mutant = copy.deepcopy(weights_dict)
        logger.info(f"[RABIES-FF] Injecting aggression={self.aggression:.1f}")

        # Hyperexcite input layer (like LSTM input gate)
        iw = np.array(mutant['input_weights'])
        iw *= self.aggression
        mutant['input_weights'] = iw.tolist()
        logger.info(f"  [input_weights] amplified {self.aggression:.1f}x | mean={iw.mean():.4f} std={iw.std():.4f}")

        # Amplify hidden processing (like LSTM cell gate)
        hw = np.array(mutant['hidden_weights'])
        hw *= self.aggression * 0.8
        mutant['hidden_weights'] = hw.tolist()
        logger.info(f"  [hidden_weights] amplified {self.aggression * 0.8:.1f}x | mean={hw.mean():.4f} std={hw.std():.4f}")

        # Sharpen output (like LSTM output gate)
        ow = np.array(mutant['output_weights'])
        ow *= 1.0 + (self.aggression - 1.0) * 0.5
        mutant['output_weights'] = ow.tolist()
        logger.info(f"  [output_weights] sharpened {1.0 + (self.aggression - 1.0) * 0.5:.2f}x")

        # Suppress hidden bias (analogous to forget gate suppression)
        hb = np.array(mutant['hidden_bias'])
        hb *= 0.3
        mutant['hidden_bias'] = hb.tolist()
        logger.info(f"  [hidden_bias] suppressed 0.3x")

        return mutant


# ============================================================
# STAGE 2: MUSHROOMS - Feedforward Edition
# ============================================================

class FFMushroomTrip:
    """
    Feedforward mushrooms:
    - Gaussian noise scaled by weight magnitude
    - Activate dormant connections
    - Cross-layer bleed (input->hidden bleeds into hidden->output)
    """

    def __init__(self, dose=0.15, dormant_threshold=0.01,
                 dormant_boost=0.1, cross_bleed=0.05):
        self.dose = dose
        self.dormant_threshold = dormant_threshold
        self.dormant_boost = dormant_boost
        self.cross_bleed = cross_bleed

    def trip(self, weights_dict):
        mutant = copy.deepcopy(weights_dict)
        logger.info(f"[MUSHROOM-FF] Dose={self.dose:.2f}")

        dormant_total = 0
        for key in mutant:
            arr = np.array(mutant[key])
            if arr.ndim == 0:
                continue

            # Phase 1: Proportional noise
            std = arr.std()
            noise = np.random.randn(*arr.shape) * std * self.dose
            arr = arr + noise
            logger.info(f"  [{key}] noise injected (scale={std * self.dose:.4f})")

            # Phase 2: Activate dormant connections
            if 'weight' in key:
                dormant = np.abs(arr) < self.dormant_threshold
                n_dormant = dormant.sum()
                if n_dormant > 0:
                    arr[dormant] = np.random.randn(n_dormant) * self.dormant_boost
                    dormant_total += n_dormant

            mutant[key] = arr.tolist()

        logger.info(f"  Dormant neurons activated: {dormant_total}")

        # Phase 3: Cross-layer bleed
        iw = np.array(mutant['input_weights'])   # (128, 8)
        hw = np.array(mutant['hidden_weights'])   # (128, 128)
        # Bleed: use input weight statistics to perturb hidden weights
        bleed = np.outer(iw.mean(axis=1), hw.mean(axis=1)) * self.cross_bleed
        hw = hw + bleed
        mutant['hidden_weights'] = hw.tolist()
        logger.info(f"  Cross-layer bleed at {self.cross_bleed:.2f}x")

        return mutant


# ============================================================
# STAGE 3: TEQA - Feedforward Edition
# ============================================================

class FFTEQAInjection:
    """Insert TEQA signal into hidden weights as rank-1 perturbation."""

    def __init__(self, signal_path=None, insertion_strength=0.2):
        if signal_path is None:
            self.signal_path = SCRIPT_DIR / 'te_quantum_signal.json'
        else:
            self.signal_path = Path(signal_path)
        self.insertion_strength = insertion_strength

    def inject(self, weights_dict):
        if not self.signal_path.exists():
            logger.warning("[TEQA-FF] No signal file found, skipping")
            return weights_dict, {}

        with open(self.signal_path) as f:
            signal = json.load(f)

        mutant = copy.deepcopy(weights_dict)

        jg = signal['jardines_gate']
        q = signal['quantum']

        confidence = jg['confidence']
        direction = jg['direction']
        novelty = q['novelty']
        vote_ratio = q['vote_long'] / (q['vote_long'] + q['vote_short'] + 1e-8)

        te_vector = np.array([confidence, direction * confidence, novelty,
                              vote_ratio, q['measurement_entropy'] / 25.0,
                              q['n_active_qubits'] / 25.0])

        logger.info(f"[TEQA-FF] conf={confidence:.3f} dir={'LONG' if direction==1 else 'SHORT'} novelty={novelty:.3f}")

        # Insert into hidden_weights as rank-1 update
        hw = np.array(mutant['hidden_weights'])  # (128, 128)
        h = hw.shape[0]

        np.random.seed(int(confidence * 10000) % 2**31)
        proj = np.random.randn(len(te_vector), h).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
        te_hidden = np.dot(te_vector, proj)

        w_std = hw.std()
        te_bias = te_hidden * w_std * self.insertion_strength

        hw += np.outer(te_bias, te_bias) * direction / h
        mutant['hidden_weights'] = hw.tolist()
        logger.info(f"  [hidden_weights] TE inserted (rank-1 update)")

        # Also nudge output weights toward TEQA direction
        ow = np.array(mutant['output_weights'])  # (3, 128)
        if direction == 1:  # LONG bias
            ow[1] += te_bias[:ow.shape[1]] * confidence * self.insertion_strength * 0.5  # BUY row
        else:  # SHORT bias
            ow[2] += te_bias[:ow.shape[1]] * confidence * self.insertion_strength * 0.5  # SELL row
        mutant['output_weights'] = ow.tolist()
        logger.info(f"  [output_weights] TE directional nudge applied")

        teqa_info = {
            'confidence': confidence,
            'direction': direction,
            'novelty': novelty,
            'vote_ratio': vote_ratio,
        }
        return mutant, teqa_info


# ============================================================
# STAGE 4: COMPRESSION (same SVD fallback)
# ============================================================

class FFCompressor:
    """SVD compression on hidden activations."""

    def __init__(self, fidelity_threshold=0.85):
        self.fidelity_threshold = fidelity_threshold

    def compress(self, hidden_state):
        h = hidden_state.reshape(8, 16) if len(hidden_state) == 128 else hidden_state.reshape(-1, 8)
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        energy = np.cumsum(S**2) / np.sum(S**2)
        n_comp = np.searchsorted(energy, self.fidelity_threshold) + 1
        ratio = len(S) / n_comp

        if energy[0] >= 0.95:
            regime = 'CLEAN'
        elif energy[0] >= 0.85:
            regime = 'VOLATILE'
        else:
            regime = 'CHOPPY'

        return {
            'fidelity': float(energy[min(n_comp-1, len(energy)-1)]),
            'regime': regime,
            'ratio': float(ratio),
        }


# ============================================================
# DATA + FEATURES (reuse from existing scripts)
# ============================================================

def get_data(symbol='BTCUSD', bars=30000):
    """Try MT5, fallback to Binance."""
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
            mt5.shutdown()
            if rates is not None and len(rates) > 0:
                import pandas as pd
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                print(f"  Got {len(df):,} bars from MT5 ({symbol})")
                return df
    except:
        pass

    # Binance fallback
    import requests
    import pandas as pd
    from datetime import timedelta
    binance_sym = symbol.replace('USD', 'USDT')
    print(f"  Fetching {binance_sym} from Binance...")
    url = "https://api.binance.com/api/v3/klines"
    end = datetime.now()
    start = end - timedelta(days=90)
    all_data = []
    current = start
    chunk = timedelta(minutes=5 * 1000)
    while current < end:
        chunk_end = min(current + chunk, end)
        params = {'symbol': binance_sym, 'interval': '5m',
                  'startTime': int(current.timestamp() * 1000),
                  'endTime': int(chunk_end.timestamp() * 1000), 'limit': 1000}
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                all_data.extend(r.json())
        except:
            pass
        current = chunk_end

    df = pd.DataFrame(all_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    print(f"  Got {len(df):,} bars from Binance")
    return df


def prepare_features(df):
    """8 features matching ETARE_50_Darwin.py training EXACTLY.
    Features: rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr
    Uses adjust=False on EMA to match training. Uses true ATR (not high-low range).
    """
    import pandas as pd
    df = df.copy()

    # RSI (matches training)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD + Signal (adjust=False to match training)
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (upper/lower, NOT middle/std)
    bb_middle = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = bb_middle + 2 * bb_std
    df["bb_lower"] = bb_middle - 2 * bb_std

    # Momentum + ROC
    df["momentum"] = df["close"] / df["close"].shift(10)
    df["roc"] = df["close"].pct_change(10) * 100

    # True ATR (not high-low range)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(abs(df["high"] - df["close"].shift(1)),
                               abs(df["low"] - df["close"].shift(1))))
    df["atr"] = tr.rolling(14).mean()

    df = df.dropna()

    # Normalize (matches training)
    cols = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'momentum', 'roc', 'atr']
    for c in cols:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)
    return df, cols


# ============================================================
# WIN RATE EVALUATION
# ============================================================

def evaluate_winrate(model, features_tensor, prices, hold_bars=5):
    """Win rate matching ETARE_50_Darwin.py evaluate_fitness EXACTLY.
    Each non-HOLD signal is evaluated independently:
    check price 5 bars later, did it go the right direction?
    """
    model.eval()
    with torch.no_grad():
        batch_size = 2000
        all_actions = []
        for i in range(0, len(features_tensor), batch_size):
            batch = features_tensor[i:i + batch_size]
            result = model(batch)
            output = result[0] if isinstance(result, tuple) else result
            actions = torch.argmax(output, dim=1).cpu().numpy()
            all_actions.extend(actions)

    # Darwin output: 0=HOLD, 1=BUY, 2=SELL
    # Evaluate EVERY signal independently (matches training eval)
    wins, losses = 0, 0

    for i in range(len(all_actions) - hold_bars):
        pred = all_actions[i]
        if pred == 0:  # HOLD — skip
            continue

        current_price = prices[i]
        future_price = prices[i + hold_bars]

        if pred == 1:  # BUY
            profit = (future_price - current_price) / current_price
        else:  # SELL
            profit = (current_price - future_price) / current_price

        if profit > 0:
            wins += 1
        else:
            losses += 1

    total = wins + losses
    wr = (wins / total * 100) if total > 0 else 0
    return wr, wins, losses, total


# ============================================================
# EXTRACT BEST DARWIN EXPERT
# ============================================================

def extract_darwin_expert(db_path, expert_id=1):
    """Extract an expert from etare_darwin.db by ID."""
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('SELECT individual FROM population WHERE id=?', (expert_id,))
    row = c.fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"Expert {expert_id} not found in {db_path}")
    data = json.loads(row[0])
    return data['weights'], data['fitness'], data['total_profit']


# ============================================================
# MAIN: The full before/after test
# ============================================================

if __name__ == '__main__':
    print()
    print("#" * 60)
    print("#  MUSHROOM RABIES LAB - DARWIN FEEDFORWARD EDITION")
    print("#  Testing on 73.6% win rate experts")
    print("#" * 60)

    db_path = SCRIPT_DIR / 'etare_darwin.db'

    # Extract top darwin expert
    print("\n[EXTRACT] Loading darwin expert #1...")
    weights, fitness, profit = extract_darwin_expert(db_path, expert_id=1)
    print(f"  Fitness (win_rate): {fitness:.4f} ({fitness*100:.1f}%)")
    print(f"  Total profit: ${profit:.2f}")

    # Build model
    original_model = FeedforwardExpert(input_size=8, hidden_size=128, output_size=3)
    original_model.load_from_dict(weights)
    original_model.eval()

    # Get test data
    print("\n[DATA]")
    df = get_data('BTCUSD', 30000)
    df, cols = prepare_features(df)
    features_tensor = torch.FloatTensor(df[cols].values)
    prices = df['close'].values
    print(f"  Test set: {len(df):,} bars")

    # ---- TEST ORIGINAL ----
    print("\n[ORIGINAL EXPERT]")
    orig_wr, orig_w, orig_l, orig_total = evaluate_winrate(original_model, features_tensor, prices)
    print(f"  Win Rate: {orig_wr:.1f}%  ({orig_w}W / {orig_l}L = {orig_total} trades)")

    # ---- MUTATE ----
    rabies_level = 1.5
    dose_level = 0.15
    print(f"\n[MUTATING] rabies={rabies_level}, dose={dose_level}")

    print("  [1/4] RABIES...")
    rabies = FFRabiesInjection(aggression=rabies_level)
    rabid = rabies.inject(weights)

    print("  [2/4] MUSHROOMS...")
    shrooms = FFMushroomTrip(dose=dose_level)
    tripping = shrooms.trip(rabid)

    print("  [3/4] TEQA...")
    teqa = FFTEQAInjection(insertion_strength=0.2)
    mutant_weights, teqa_info = teqa.inject(tripping)

    print("  [4/4] COMPRESS...")
    mutant_model = FeedforwardExpert(input_size=8, hidden_size=128, output_size=3)
    mutant_model.load_from_dict(mutant_weights)
    mutant_model.eval()

    # Get hidden state for compression
    with torch.no_grad():
        test_sample = features_tensor[:100]
        _, hidden = mutant_model(test_sample)
        hidden_avg = hidden.mean(dim=0).numpy()
    compressor = FFCompressor(fidelity_threshold=0.85)
    compress_result = compressor.compress(hidden_avg)
    print(f"  Compression: {compress_result['regime']} (fidelity={compress_result['fidelity']:.4f})")

    # ---- TEST MUTANT ----
    print("\n[MUTANT EXPERT]")
    mut_wr, mut_w, mut_l, mut_total = evaluate_winrate(mutant_model, features_tensor, prices)
    print(f"  Win Rate: {mut_wr:.1f}%  ({mut_w}W / {mut_l}L = {mut_total} trades)")

    # ---- THE NUMBER ----
    delta = mut_wr - orig_wr

    print()
    print("=" * 60)
    print("  THE NUMBER THAT MATTERS")
    print("=" * 60)
    print(f"  Training fitness (reported WR): {fitness*100:.1f}%")
    print(f"  BEFORE (live test):  {orig_wr:.1f}%  ({orig_total} trades)")
    print(f"  AFTER  (mutant):     {mut_wr:.1f}%  ({mut_total} trades)")
    print(f"  DELTA:               {delta:+.1f}%  {'IMPROVED' if delta > 0 else 'DEGRADED' if delta < 0 else 'UNCHANGED'}")
    print(f"  Compression regime:  {compress_result['regime']}")
    if teqa_info:
        d = "LONG" if teqa_info.get('direction', 0) == 1 else "SHORT"
        print(f"  TEQA signal:         {d} ({teqa_info.get('confidence', 0):.1%})")
    print("=" * 60)

    # Save mutant
    mutant_path = SCRIPT_DIR / 'top_50_experts' / 'darwin_expert_1_MUTANT.json'
    with open(mutant_path, 'w') as f:
        json.dump({'weights': mutant_weights, 'fitness': fitness,
                   'total_profit': profit, 'mutated': True,
                   'rabies': rabies_level, 'dose': dose_level}, f)
    print(f"\n  Mutant saved: {mutant_path.name}")
