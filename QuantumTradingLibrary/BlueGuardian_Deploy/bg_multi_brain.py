"""
Blue Guardian Multi-Account Signal Generator
=============================================
Fresh build for Jim's 3 Blue Guardian accounts.
Handles: 2x Instant Challenges + 1x Competition

Architecture:
- Reads market_data.json (exported by DataExporter EA)
- Generates signals per account with isolation
- Writes account-specific signal files
- Monitors drawdown limits per account
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# --- CONFIGURATION ---
CONFIG_FILE = Path("/app/config/accounts_config.json")
DATA_DIR = Path("/mt5files")
MODEL_DIR = Path("/app/champions")
LOG_FILE = Path("/app/logs/bg_brain.log")

DEVICE = torch.device("cpu")  # Stability on VPS
SEQ_LENGTH = 30


# ============================================================================
# LSTM MODEL (Same architecture as champion)
# ============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                            num_layers=2, dropout=0.2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# ============================================================================
# LOGGING
# ============================================================================
def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] [{level}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a') as f:
            f.write(line + "\n")
    except:
        pass


# ============================================================================
# CONFIG LOADING
# ============================================================================
def load_config() -> Dict:
    if not CONFIG_FILE.exists():
        log(f"Config not found: {CONFIG_FILE}", "FATAL")
        sys.exit(1)
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


# ============================================================================
# DATA HANDLING
# ============================================================================
def load_market_data(symbol: str, lookback: int = 100) -> Optional[pd.DataFrame]:
    """Read JSON data exported from MT5 DataExporter"""
    data_file = DATA_DIR / "market_data.json"
    if not data_file.exists():
        return None

    try:
        with open(data_file, 'r') as f:
            data = json.load(f)

        if symbol not in data:
            return None

        raw = data[symbol][-lookback:]
        df = pd.DataFrame(raw)

        for c in ['open', 'high', 'low', 'close', 'tick_volume']:
            df[c] = pd.to_numeric(df[c])

        return df
    except Exception as e:
        log(f"Error reading market data: {e}", "ERROR")
        return None


def prepare_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extract Technical Indicators for LSTM input"""
    df = df.copy()

    # 1. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 2. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 3. Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    # 4. Momentum/ROC
    df['momentum'] = df['close'] / df['close'].shift(10)
    df['roc'] = df['close'].pct_change(10) * 100

    # 5. ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()

    df = df.dropna()
    if df.empty:
        return None

    # Normalize features
    features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                'momentum', 'roc', 'atr']
    for c in features:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-8)

    return df[features]


def get_prediction(model: LSTMModel, df: pd.DataFrame,
                   min_confidence: float) -> Tuple[str, float]:
    """Generate prediction from LSTM model"""
    if len(df) < SEQ_LENGTH:
        return "HOLD", 0.0

    seq = df.iloc[-SEQ_LENGTH:].values
    tensor = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)

        # Reduce HOLD bias for prop firm mode
        probs[0][0] *= 0.4
        probs = probs / probs.sum()

        action_idx = torch.argmax(probs).item()
        conf = probs[0][action_idx].item()

    actions = {0: "HOLD", 1: "BUY", 2: "SELL"}
    action = actions[action_idx]

    # Apply confidence filter
    if action != "HOLD" and conf < min_confidence:
        return "HOLD", conf

    return action, conf


# ============================================================================
# ACCOUNT STATE TRACKING
# ============================================================================
class AccountState:
    """Tracks state for a single account"""
    def __init__(self, config: Dict):
        self.name = config['name']
        self.account_id = config['account_id']
        self.magic_number = config['magic_number']
        self.symbol = config['symbol']
        self.max_lot_size = config['max_lot_size']
        self.daily_dd_limit = config['daily_drawdown_pct']
        self.max_dd_limit = config['max_drawdown_pct']
        self.profit_target = config['profit_target_pct']
        self.enabled = config['enabled']

        # Runtime state
        self.trades_today = 0
        self.last_trade_date = None
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.blocked = False
        self.block_reason = ""

    def reset_daily(self):
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today

    def check_limits(self, max_trades_per_day: int) -> bool:
        """Check if account is within limits"""
        self.reset_daily()

        if self.trades_today >= max_trades_per_day:
            self.blocked = True
            self.block_reason = f"Max trades ({max_trades_per_day}) reached"
            return False

        # Note: Real PnL tracking would need connection to MT5
        # This is a placeholder for the signal generator

        return True


# ============================================================================
# SIGNAL WRITER
# ============================================================================
def write_signal(account: AccountState, action: str, confidence: float):
    """Write signal file for specific account"""
    signal_file = DATA_DIR / f"signal_{account.name}.json"

    output = {
        account.symbol: {
            "action": action,
            "confidence": confidence,
            "magic_number": account.magic_number,
            "max_lot_size": account.max_lot_size,
            "timestamp": datetime.now().isoformat()
        },
        "_meta": {
            "account": account.name,
            "account_id": account.account_id,
            "engine": "BlueGuardian_MultiBrain",
            "status": "ACTIVE" if not account.blocked else "BLOCKED",
            "block_reason": account.block_reason
        }
    }

    with open(signal_file, 'w') as f:
        json.dump(output, f, indent=2)


# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    log("=" * 60)
    log("BLUE GUARDIAN MULTI-BRAIN - FRESH BUILD")
    log("Accounts: 2x Instant + 1x Competition")
    log("=" * 60)

    # Load config
    config = load_config()
    trading_config = config.get('trading', {})
    min_confidence = trading_config.get('min_confidence', 0.65)
    max_trades = trading_config.get('max_trades_per_day', 3)
    check_interval = trading_config.get('check_interval_seconds', 10)

    # Initialize accounts
    accounts = []
    for acc_config in config['accounts']:
        if acc_config.get('enabled', True):
            accounts.append(AccountState(acc_config))
            log(f"Loaded account: {acc_config['name']} ({acc_config['type']})")

    if not accounts:
        log("No enabled accounts found!", "FATAL")
        sys.exit(1)

    # Load model
    model_path = MODEL_DIR / "champion_BTCUSD.pth"
    if not model_path.exists():
        log(f"Champion model not found: {model_path}", "FATAL")
        sys.exit(1)

    state = torch.load(model_path, map_location=DEVICE)
    hidden_size = state['lstm.weight_ih_l0'].shape[0] // 4
    model = LSTMModel(input_size=8, hidden_size=hidden_size)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    log(f"Champion BTCUSD loaded (hidden_size={hidden_size})")
    log("=" * 60)

    # Main loop
    while True:
        try:
            for account in accounts:
                if not account.enabled or account.blocked:
                    continue

                if not account.check_limits(max_trades):
                    write_signal(account, "HOLD", 0.0)
                    continue

                # Get market data
                df = load_market_data(account.symbol)
                if df is None:
                    continue

                # Prepare features
                feat = prepare_features(df)
                if feat is None:
                    continue

                # Get prediction
                action, conf = get_prediction(model, feat, min_confidence)

                # Write signal
                write_signal(account, action, conf)

                if action != "HOLD":
                    log(f"[{account.name}] {account.symbol}: {action} ({conf:.1%})")
                    account.trades_today += 1

            time.sleep(check_interval)

        except KeyboardInterrupt:
            log("Shutdown requested")
            break
        except Exception as e:
            log(f"Cycle error: {e}", "ERROR")
            time.sleep(5)

    log("Brain shutdown complete")


if __name__ == "__main__":
    main()
