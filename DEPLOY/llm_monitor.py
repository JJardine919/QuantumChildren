"""
BlueGuardian LLM Monitor
Monitors positions and writes adjustments to llm_config.json
The EA reads this file every 60 seconds
"""

import MetaTrader5 as mt5
import json
import time
import os
from datetime import datetime
import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    # MT5 Common Files folder (where EA reads from)
    'config_file': os.path.join(os.environ.get('APPDATA', ''),
                                'MetaQuotes', 'Terminal', 'Common', 'Files', 'llm_config.json'),

    # LLM Settings
    'ollama_url': 'http://localhost:11434/api/generate',
    'ollama_model': 'mistral',

    # Monitoring
    'check_interval': 30,  # seconds
    'magic_number': 365100,

    # Emergency thresholds
    'max_drawdown_pct': 8.0,  # Emergency stop if DD exceeds this
    'max_loss_per_trade': -500,  # Emergency stop if single trade loss exceeds this

    # Dynamic adjustment limits
    'max_sl_adjust': 20,  # Max SL adjustment in points
    'max_tp_adjust': 50,  # Max TP adjustment in points
}

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] LLM_MONITOR >> {msg}")

def write_config(emergency_stop=False, sl_adjust=0, tp_adjust=0, reason=""):
    """Write config file that EA reads"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "emergency_stop": emergency_stop,
        "sl_adjust": sl_adjust,
        "tp_adjust": tp_adjust,
        "reason": reason
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(CONFIG['config_file']), exist_ok=True)

    with open(CONFIG['config_file'], 'w') as f:
        json.dump(config, f, indent=2)

    if emergency_stop or sl_adjust != 0 or tp_adjust != 0:
        log(f"Config written: emergency={emergency_stop}, sl_adj={sl_adjust}, tp_adj={tp_adjust}")

def query_llm(prompt):
    """Query local Ollama LLM"""
    try:
        payload = {
            "model": CONFIG['ollama_model'],
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 200}
        }
        response = requests.post(CONFIG['ollama_url'], json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception as e:
        log(f"LLM error: {e}")
    return None

def analyze_position(position, account_info):
    """Ask LLM to analyze position and suggest adjustments"""

    profit_loss = position.profit
    entry = position.price_open
    current = position.price_current
    sl = position.sl
    tp = position.tp
    direction = "LONG" if position.type == mt5.POSITION_TYPE_BUY else "SHORT"

    # Calculate metrics
    pip_profit = (current - entry) if direction == "LONG" else (entry - current)
    risk_reward = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

    prompt = f"""You are a trading risk manager. Analyze this open position:

POSITION:
- Direction: {direction}
- Entry: {entry:.5f}
- Current: {current:.5f}
- P/L: ${profit_loss:.2f}
- Stop Loss: {sl:.5f}
- Take Profit: {tp:.5f}
- Risk/Reward: {risk_reward:.2f}

ACCOUNT:
- Balance: ${account_info.balance:.2f}
- Equity: ${account_info.equity:.2f}
- Drawdown: {((account_info.balance - account_info.equity) / account_info.balance * 100):.2f}%

Should we:
1. HOLD (no changes)
2. TIGHTEN_SL (move SL closer to protect profit)
3. EXTEND_TP (let winner run)
4. EMERGENCY_CLOSE (close immediately)

Respond with ONLY the action (HOLD, TIGHTEN_SL, EXTEND_TP, or EMERGENCY_CLOSE) and number of points if applicable.
Example: "TIGHTEN_SL 15" or "HOLD"
"""

    response = query_llm(prompt)
    return parse_llm_response(response)

def parse_llm_response(response):
    """Parse LLM response into action"""
    if not response:
        return "HOLD", 0, 0

    response = response.upper().strip()

    if "EMERGENCY" in response:
        return "EMERGENCY", 0, 0

    sl_adjust = 0
    tp_adjust = 0

    if "TIGHTEN_SL" in response:
        # Extract number if present
        parts = response.split()
        for part in parts:
            try:
                sl_adjust = int(part)
                break
            except:
                continue
        sl_adjust = min(sl_adjust, CONFIG['max_sl_adjust'])
        return "TIGHTEN_SL", sl_adjust, 0

    if "EXTEND_TP" in response:
        parts = response.split()
        for part in parts:
            try:
                tp_adjust = int(part)
                break
            except:
                continue
        tp_adjust = min(tp_adjust, CONFIG['max_tp_adjust'])
        return "EXTEND_TP", 0, tp_adjust

    return "HOLD", 0, 0

def check_emergency_conditions(account_info, positions):
    """Check for emergency stop conditions"""

    # Check account drawdown
    if account_info.balance > 0:
        dd_pct = ((account_info.balance - account_info.equity) / account_info.balance) * 100
        if dd_pct >= CONFIG['max_drawdown_pct']:
            return True, f"Drawdown {dd_pct:.2f}% exceeds {CONFIG['max_drawdown_pct']}%"

    # Check individual trade loss
    for pos in positions:
        if pos.profit <= CONFIG['max_loss_per_trade']:
            return True, f"Trade loss ${pos.profit:.2f} exceeds limit"

    return False, ""

def run_monitor():
    """Main monitoring loop"""
    log("=" * 50)
    log("BlueGuardian LLM Monitor Starting")
    log(f"Config file: {CONFIG['config_file']}")
    log("=" * 50)

    if not mt5.initialize():
        log(f"MT5 init failed: {mt5.last_error()}")
        return

    # Write initial "all clear" config
    write_config(emergency_stop=False, sl_adjust=0, tp_adjust=0)

    try:
        while True:
            account = mt5.account_info()
            if not account:
                log("Failed to get account info")
                time.sleep(CONFIG['check_interval'])
                continue

            # Get positions for our magic number
            positions = mt5.positions_get()
            our_positions = [p for p in positions if p.magic == CONFIG['magic_number']] if positions else []

            # Check emergency conditions first
            emergency, reason = check_emergency_conditions(account, our_positions)
            if emergency:
                log(f"EMERGENCY: {reason}")
                write_config(emergency_stop=True, reason=reason)
                time.sleep(CONFIG['check_interval'])
                continue

            # Analyze each position
            total_sl_adjust = 0
            total_tp_adjust = 0

            for pos in our_positions:
                action, sl_adj, tp_adj = analyze_position(pos, account)

                if action == "EMERGENCY":
                    write_config(emergency_stop=True, reason="LLM recommended emergency close")
                    break

                total_sl_adjust += sl_adj
                total_tp_adjust += tp_adj

                if action != "HOLD":
                    log(f"Position {pos.ticket}: {action} (SL adj: {sl_adj}, TP adj: {tp_adj})")

            # Write adjustments
            write_config(
                emergency_stop=False,
                sl_adjust=total_sl_adjust,
                tp_adjust=total_tp_adjust
            )

            # Status update
            if len(our_positions) > 0:
                total_profit = sum(p.profit for p in our_positions)
                log(f"Monitoring {len(our_positions)} positions | P/L: ${total_profit:.2f}")

            time.sleep(CONFIG['check_interval'])

    except KeyboardInterrupt:
        log("Monitor stopped by user")
    finally:
        # Clear config on exit
        write_config(emergency_stop=False, sl_adjust=0, tp_adjust=0)
        mt5.shutdown()

if __name__ == "__main__":
    run_monitor()
