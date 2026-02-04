"""
Quantum Children Terminal Manager
Flask-based control panel for MT5 prop firm accounts
"""

import os
import json
import subprocess
import psutil
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'BlueGuardian_Deploy', 'accounts_config.json')
MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Alternative MT5 paths to check
MT5_PATHS = [
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
    r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    r"C:\Program Files\Blue Guardian - MetaTrader 5\terminal64.exe",
    r"C:\Program Files\Atlas Funded - MetaTrader 5\terminal64.exe",
]

# Track running processes
running_terminals = {}

def find_mt5_path():
    """Find MT5 terminal executable"""
    for path in MT5_PATHS:
        if os.path.exists(path):
            return path
    return None

def load_accounts():
    """Load accounts from config file"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            return config.get('accounts', [])
    except Exception as e:
        print(f"Error loading config: {e}")
        return []

def get_terminal_status(account_name):
    """Check if a terminal is running for this account"""
    if account_name in running_terminals:
        proc = running_terminals[account_name]
        if proc.poll() is None:  # Still running
            return "RUNNING"
        else:
            del running_terminals[account_name]
    return "STOPPED"

def find_running_mt5_processes():
    """Find all running MT5 processes"""
    mt5_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'terminal64.exe' in proc.info['name'].lower():
                mt5_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return mt5_procs

@app.route('/')
def index():
    """Main dashboard"""
    accounts = load_accounts()
    mt5_path = find_mt5_path()

    # Add status to each account
    for account in accounts:
        account['status'] = get_terminal_status(account['name'])

    return render_template('index.html',
                         accounts=accounts,
                         mt5_found=mt5_path is not None,
                         mt5_path=mt5_path)

@app.route('/api/accounts')
def api_accounts():
    """API endpoint for account status"""
    accounts = load_accounts()
    for account in accounts:
        account['status'] = get_terminal_status(account['name'])
        # Don't expose passwords in API
        account['password'] = '********'
    return jsonify(accounts)

@app.route('/api/start/<account_name>', methods=['POST'])
def start_terminal(account_name):
    """Start MT5 terminal for an account"""
    accounts = load_accounts()
    account = next((a for a in accounts if a['name'] == account_name), None)

    if not account:
        return jsonify({'success': False, 'error': 'Account not found'}), 404

    if not account.get('enabled', True):
        return jsonify({'success': False, 'error': 'Account is disabled'}), 400

    if get_terminal_status(account_name) == "RUNNING":
        return jsonify({'success': False, 'error': 'Terminal already running'}), 400

    mt5_path = find_mt5_path()
    if not mt5_path:
        return jsonify({'success': False, 'error': 'MT5 not found'}), 500

    try:
        # Build command line arguments
        # MT5 supports: /login:<login> /password:<password> /server:<server>
        cmd = [
            mt5_path,
            f"/login:{account['account_id']}",
            f"/password:{account['password']}",
            f"/server:{account['server']}"
        ]

        # Start the process
        proc = subprocess.Popen(cmd,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NEW_CONSOLE)

        running_terminals[account_name] = proc

        return jsonify({
            'success': True,
            'message': f'Started terminal for {account_name}',
            'pid': proc.pid
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop/<account_name>', methods=['POST'])
def stop_terminal(account_name):
    """Stop MT5 terminal for an account"""
    if account_name not in running_terminals:
        return jsonify({'success': False, 'error': 'Terminal not running'}), 400

    try:
        proc = running_terminals[account_name]
        proc.terminate()
        proc.wait(timeout=5)
        del running_terminals[account_name]

        return jsonify({
            'success': True,
            'message': f'Stopped terminal for {account_name}'
        })

    except subprocess.TimeoutExpired:
        proc.kill()
        del running_terminals[account_name]
        return jsonify({
            'success': True,
            'message': f'Force killed terminal for {account_name}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/start-all', methods=['POST'])
def start_all():
    """Start all enabled terminals"""
    accounts = load_accounts()
    results = []

    for account in accounts:
        if account.get('enabled', True) and get_terminal_status(account['name']) == "STOPPED":
            # Small delay between launches
            time.sleep(2)
            response = start_terminal(account['name'])
            results.append({
                'account': account['name'],
                'result': response.get_json()
            })

    return jsonify({'success': True, 'results': results})

@app.route('/api/stop-all', methods=['POST'])
def stop_all():
    """Stop all running terminals"""
    results = []

    for account_name in list(running_terminals.keys()):
        response = stop_terminal(account_name)
        results.append({
            'account': account_name,
            'result': response.get_json()
        })

    return jsonify({'success': True, 'results': results})

@app.route('/api/status')
def system_status():
    """Get overall system status"""
    accounts = load_accounts()
    running_count = sum(1 for a in accounts if get_terminal_status(a['name']) == "RUNNING")
    mt5_procs = find_running_mt5_processes()

    return jsonify({
        'total_accounts': len(accounts),
        'running_terminals': running_count,
        'mt5_processes_detected': len(mt5_procs),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 50)
    print("  QUANTUM CHILDREN - Terminal Manager")
    print("=" * 50)
    print(f"  Config: {CONFIG_PATH}")
    print(f"  MT5 Path: {find_mt5_path() or 'NOT FOUND'}")
    print("=" * 50)
    print("  Starting server on http://localhost:5000")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)
