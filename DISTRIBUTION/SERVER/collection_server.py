"""
QUANTUM CHILDREN - DATA COLLECTION SERVER
==========================================
Receives entropy data from distributed trading nodes.
Provides bridge API for Base44 web dashboard.

Deploy on VPS:
    pip install flask
    python collection_server.py

Or with gunicorn:
    gunicorn -w 4 -b 0.0.0.0:8888 collection_server:app
"""

import json
import os
import sqlite3
import hashlib
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from collections import defaultdict
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================
# SECURITY & RATE LIMITING
# ============================================================

ADMIN_API_KEY = os.environ.get('QC_ADMIN_KEY', '')
_rate_limit_store = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 30  # per window for public endpoints
RATE_LIMIT_MAX_ADMIN = 10  # per window for admin endpoints


def _get_client_ip():
    return request.headers.get('X-Forwarded-For', request.remote_addr)


def rate_limit(max_requests=RATE_LIMIT_MAX_REQUESTS):
    """Simple in-memory rate limiter"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            ip = _get_client_ip()
            now = time.time()
            _rate_limit_store[ip] = [
                t for t in _rate_limit_store[ip]
                if now - t < RATE_LIMIT_WINDOW
            ]
            if len(_rate_limit_store[ip]) >= max_requests:
                return jsonify({'error': 'Rate limit exceeded'}), 429
            _rate_limit_store[ip].append(now)
            return f(*args, **kwargs)
        return wrapped
    return decorator


def require_admin_key(f):
    """Require API key for admin/bridge endpoints"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        key = request.headers.get('X-API-Key', '')
        if not ADMIN_API_KEY:
            return f(*args, **kwargs)  # No key configured = open (dev mode)
        if not key or key != ADMIN_API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return wrapped

# ============================================================
# DATABASE SETUP
# ============================================================

DB_PATH = Path("quantum_collected.db")

def init_db():
    """Initialize the collection database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Signals table
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            sig_hash TEXT UNIQUE,
            symbol TEXT,
            direction TEXT,
            confidence REAL,
            quantum_entropy REAL,
            dominant_state REAL,
            price REAL,
            features TEXT,
            timestamp TEXT,
            received_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Outcomes table
    c.execute('''
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            ticket INTEGER,
            symbol TEXT,
            outcome TEXT,
            pnl REAL,
            entry_price REAL,
            exit_price REAL,
            timestamp TEXT,
            received_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Entropy snapshots table
    c.execute('''
        CREATE TABLE IF NOT EXISTS entropy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            symbol TEXT,
            timeframe TEXT,
            quantum_entropy REAL,
            dominant_state REAL,
            significant_states INTEGER,
            quantum_variance REAL,
            regime TEXT,
            price REAL,
            timestamp TEXT,
            received_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Nodes table (track active nodes)
    c.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            first_seen TEXT,
            last_seen TEXT,
            signal_count INTEGER DEFAULT 0,
            outcome_count INTEGER DEFAULT 0,
            entropy_count INTEGER DEFAULT 0
        )
    ''')

    # Create indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_signals_node ON signals(node_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_entropy_symbol ON entropy(symbol)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_entropy_regime ON entropy(regime)')

    conn.commit()
    conn.close()

def update_node_stats(node_id: str, signal: int = 0, outcome: int = 0, entropy: int = 0):
    """Update node statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.utcnow().isoformat()

    c.execute('''
        INSERT INTO nodes (node_id, first_seen, last_seen, signal_count, outcome_count, entropy_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            last_seen = ?,
            signal_count = signal_count + ?,
            outcome_count = outcome_count + ?,
            entropy_count = entropy_count + ?
    ''', (node_id, now, now, signal, outcome, entropy, now, signal, outcome, entropy))

    conn.commit()
    conn.close()

# Initialize on startup
init_db()

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/ping', methods=['GET', 'POST'])
@rate_limit()
def ping():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'server': 'QuantumChildren', 'time': datetime.utcnow().isoformat()})

@app.route('/collect', methods=['POST'])
@app.route('/signal', methods=['POST'])
@rate_limit()
def collect_signal():
    """Receive trading signal"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data'}), 400

        node_id = data.get('node_id', 'UNKNOWN')

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute('''
            INSERT OR IGNORE INTO signals
            (node_id, sig_hash, symbol, direction, confidence, quantum_entropy,
             dominant_state, price, features, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node_id,
            data.get('sig_hash'),
            data.get('symbol'),
            data.get('direction'),
            data.get('confidence'),
            data.get('quantum_entropy'),
            data.get('dominant_state'),
            data.get('price'),
            json.dumps(data.get('features')) if data.get('features') else None,
            data.get('timestamp')
        ))

        conn.commit()
        conn.close()

        update_node_stats(node_id, signal=1)

        return jsonify({'status': 'ok', 'received': 'signal'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outcome', methods=['POST'])
@rate_limit()
def collect_outcome():
    """Receive trade outcome"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data'}), 400

        node_id = data.get('node_id', 'UNKNOWN')

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute('''
            INSERT INTO outcomes
            (node_id, ticket, symbol, outcome, pnl, entry_price, exit_price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node_id,
            data.get('ticket'),
            data.get('symbol'),
            data.get('outcome'),
            data.get('pnl'),
            data.get('entry_price'),
            data.get('exit_price'),
            data.get('timestamp')
        ))

        conn.commit()
        conn.close()

        update_node_stats(node_id, outcome=1)

        return jsonify({'status': 'ok', 'received': 'outcome'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/entropy', methods=['POST'])
@rate_limit()
def collect_entropy():
    """Receive entropy snapshot"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data'}), 400

        node_id = data.get('node_id', 'UNKNOWN')

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute('''
            INSERT INTO entropy
            (node_id, symbol, timeframe, quantum_entropy, dominant_state,
             significant_states, quantum_variance, regime, price, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node_id,
            data.get('symbol'),
            data.get('timeframe'),
            data.get('quantum_entropy'),
            data.get('dominant_state'),
            data.get('significant_states'),
            data.get('quantum_variance'),
            data.get('regime'),
            data.get('price'),
            data.get('timestamp')
        ))

        conn.commit()
        conn.close()

        update_node_stats(node_id, entropy=1)

        return jsonify({'status': 'ok', 'received': 'entropy'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
@rate_limit()
def get_stats():
    """Get collection statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Get counts
        c.execute('SELECT COUNT(*) FROM signals')
        signal_count = c.fetchone()[0]

        c.execute('SELECT COUNT(*) FROM outcomes')
        outcome_count = c.fetchone()[0]

        c.execute('SELECT COUNT(*) FROM entropy')
        entropy_count = c.fetchone()[0]

        c.execute('SELECT COUNT(*) FROM nodes')
        node_count = c.fetchone()[0]

        # Get recent activity
        c.execute('SELECT node_id, last_seen, signal_count, outcome_count FROM nodes ORDER BY last_seen DESC LIMIT 10')
        recent_nodes = [{'node_id': r[0], 'last_seen': r[1], 'signals': r[2], 'outcomes': r[3]} for r in c.fetchall()]

        conn.close()

        return jsonify({
            'total_signals': signal_count,
            'total_outcomes': outcome_count,
            'total_entropy': entropy_count,
            'active_nodes': node_count,
            'recent_nodes': recent_nodes
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# BRIDGE API (for Base44 web dashboard)
# ============================================================

@app.route('/performance', methods=['GET'])
@rate_limit()
def get_performance():
    """Live trading performance metrics for Base44 dashboard"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Overall stats
        c.execute('SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl) FROM outcomes')
        row = c.fetchone()
        total_trades = row[0] or 0
        wins = row[1] or 0
        total_pnl = row[2] or 0.0
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # Per-symbol breakdown
        c.execute('''
            SELECT symbol,
                   COUNT(*) as trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl) as pnl,
                   AVG(pnl) as avg_pnl
            FROM outcomes GROUP BY symbol ORDER BY trades DESC
        ''')
        symbols = [{'symbol': r[0], 'trades': r[1], 'wins': r[2],
                     'pnl': round(r[3] or 0, 2), 'avg_pnl': round(r[4] or 0, 4),
                     'win_rate': round((r[2] or 0) / r[1] * 100, 1) if r[1] > 0 else 0}
                    for r in c.fetchall()]

        # Recent trades (last 20)
        c.execute('''
            SELECT symbol, outcome, pnl, entry_price, exit_price, timestamp
            FROM outcomes ORDER BY received_at DESC LIMIT 20
        ''')
        recent = [{'symbol': r[0], 'outcome': r[1], 'pnl': r[2],
                    'entry': r[3], 'exit': r[4], 'time': r[5]}
                   for r in c.fetchall()]

        # Equity curve (cumulative PnL over time)
        c.execute('SELECT pnl, timestamp FROM outcomes ORDER BY received_at ASC')
        cumulative = 0
        equity_curve = []
        for r in c.fetchall():
            cumulative += (r[0] or 0)
            equity_curve.append({'pnl': round(cumulative, 2), 'time': r[1]})

        # Current regime per symbol (latest entropy readings)
        c.execute('''
            SELECT symbol, regime, quantum_entropy, dominant_state, timestamp
            FROM entropy WHERE id IN (
                SELECT MAX(id) FROM entropy GROUP BY symbol
            )
        ''')
        regimes = [{'symbol': r[0], 'regime': r[1], 'entropy': r[2],
                     'dominant_state': r[3], 'time': r[4]}
                    for r in c.fetchall()]

        conn.close()

        return jsonify({
            'total_trades': total_trades,
            'wins': wins,
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'symbols': symbols,
            'recent_trades': recent,
            'equity_curve': equity_curve,
            'regimes': regimes
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/alerts', methods=['GET'])
@rate_limit()
def get_alerts():
    """Recent significant events for notification system"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        alerts = []

        # Check for drawdown (3+ consecutive losses)
        c.execute('''
            SELECT symbol, pnl, timestamp FROM outcomes
            ORDER BY received_at DESC LIMIT 10
        ''')
        recent = c.fetchall()
        streak = 0
        for r in recent:
            if (r[1] or 0) < 0:
                streak += 1
            else:
                break
        if streak >= 3:
            alerts.append({
                'type': 'DRAWDOWN',
                'severity': 'high' if streak >= 5 else 'medium',
                'message': f'{streak} consecutive losing trades',
                'time': recent[0][2] if recent else None
            })

        # Check for large single loss
        c.execute('''
            SELECT symbol, pnl, timestamp FROM outcomes
            WHERE pnl < -0.50 ORDER BY received_at DESC LIMIT 5
        ''')
        for r in c.fetchall():
            alerts.append({
                'type': 'LARGE_LOSS',
                'severity': 'high',
                'message': f'{r[0]}: ${r[1]:.2f} loss',
                'time': r[2]
            })

        # Check win rate degradation (last 20 vs overall)
        c.execute('SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) FROM outcomes')
        overall = c.fetchone()
        overall_wr = (overall[1] or 0) / overall[0] * 100 if overall[0] > 0 else 50

        c.execute('''
            SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)
            FROM (SELECT pnl FROM outcomes ORDER BY received_at DESC LIMIT 20)
        ''')
        recent_stats = c.fetchone()
        recent_wr = (recent_stats[1] or 0) / recent_stats[0] * 100 if recent_stats[0] > 0 else 50

        if overall_wr - recent_wr > 10:
            alerts.append({
                'type': 'WIN_RATE_DROP',
                'severity': 'medium',
                'message': f'Win rate dropped: {recent_wr:.0f}% recent vs {overall_wr:.0f}% overall',
                'time': datetime.utcnow().isoformat()
            })

        # Check for new regime changes
        c.execute('''
            SELECT symbol, regime, timestamp FROM entropy
            ORDER BY received_at DESC LIMIT 5
        ''')
        for r in c.fetchall():
            if r[1] and r[1].upper() == 'CHAOTIC':
                alerts.append({
                    'type': 'REGIME_CHAOTIC',
                    'severity': 'medium',
                    'message': f'{r[0]} entered CHAOTIC regime',
                    'time': r[2]
                })

        conn.close()

        return jsonify({'alerts': alerts, 'count': len(alerts)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/backtest', methods=['POST'])
@rate_limit(max_requests=RATE_LIMIT_MAX_ADMIN)
@require_admin_key
def trigger_backtest():
    """Trigger backtest and return results from collected data"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTCUSD')
        days = min(data.get('days', 30), 90)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Pull outcomes for the requested symbol and period
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        c.execute('''
            SELECT symbol, outcome, pnl, entry_price, exit_price, timestamp
            FROM outcomes
            WHERE (symbol = ? OR ? = 'ALL')
            AND received_at >= ?
            ORDER BY received_at ASC
        ''', (symbol, symbol, cutoff))

        trades = c.fetchall()
        if not trades:
            conn.close()
            return jsonify({'error': 'No trade data for this period', 'symbol': symbol, 'days': days}), 404

        wins = sum(1 for t in trades if (t[2] or 0) > 0)
        losses = len(trades) - wins
        total_pnl = sum(t[2] or 0 for t in trades)
        max_dd = 0
        running_pnl = 0
        peak = 0
        for t in trades:
            running_pnl += (t[2] or 0)
            if running_pnl > peak:
                peak = running_pnl
            dd = peak - running_pnl
            if dd > max_dd:
                max_dd = dd

        gross_profit = sum(t[2] for t in trades if (t[2] or 0) > 0)
        gross_loss = abs(sum(t[2] for t in trades if (t[2] or 0) < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        conn.close()

        return jsonify({
            'symbol': symbol,
            'period_days': days,
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': round(wins / len(trades) * 100, 1),
            'total_pnl': round(total_pnl, 2),
            'max_drawdown': round(max_dd, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(gross_profit / wins, 4) if wins > 0 else 0,
            'avg_loss': round(gross_loss / losses, 4) if losses > 0 else 0,
            'status': 'complete'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compile', methods=['POST'])
@rate_limit(max_requests=RATE_LIMIT_MAX_ADMIN)
@require_admin_key
def trigger_compile():
    """Queue an EA compilation request"""
    try:
        data = request.get_json()
        if not data or 'ea_name' not in data:
            return jsonify({'error': 'ea_name is required'}), 400

        ea_name = data['ea_name']
        # Sanitize - only allow alphanumeric, underscore, hyphen
        if not all(c.isalnum() or c in ('_', '-') for c in ea_name):
            return jsonify({'error': 'Invalid EA name'}), 400

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Create compile requests table if needed
        c.execute('''
            CREATE TABLE IF NOT EXISTS compile_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ea_name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                result TEXT,
                requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            )
        ''')

        c.execute(
            'INSERT INTO compile_requests (ea_name) VALUES (?)',
            (ea_name,)
        )
        request_id = c.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            'status': 'queued',
            'request_id': request_id,
            'ea_name': ea_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compile/<int:request_id>', methods=['GET'])
@rate_limit()
def get_compile_status(request_id):
    """Check compilation request status"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            'SELECT ea_name, status, result, requested_at, completed_at FROM compile_requests WHERE id = ?',
            (request_id,)
        )
        row = c.fetchone()
        conn.close()

        if not row:
            return jsonify({'error': 'Request not found'}), 404

        return jsonify({
            'request_id': request_id,
            'ea_name': row[0],
            'status': row[1],
            'result': row[2],
            'requested_at': row[3],
            'completed_at': row[4]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """Landing page with neural network animation and music"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QUANTUM CHILDREN - Neural Trading Network</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #000;
            color: #0ff;
            font-family: 'Courier New', monospace;
            overflow: hidden;
            min-height: 100vh;
        }
        #neural-canvas {
            position: fixed;
            top: 0;
            left: 0;
            z-index: 0;
        }
        .container {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 20px;
            text-align: center;
        }
        h1 {
            font-size: 4em;
            text-shadow: 0 0 20px #0ff, 0 0 40px #0ff, 0 0 60px #00f;
            animation: pulse 2s ease-in-out infinite;
            margin-bottom: 10px;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; text-shadow: 0 0 20px #0ff, 0 0 40px #0ff, 0 0 60px #00f; }
            50% { opacity: 0.8; text-shadow: 0 0 30px #0ff, 0 0 60px #0ff, 0 0 90px #00f; }
        }
        h2 {
            font-size: 1.5em;
            color: #0f0;
            text-shadow: 0 0 10px #0f0;
            margin-bottom: 30px;
        }
        .stats-box {
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid #0ff;
            border-radius: 10px;
            padding: 30px;
            margin: 20px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3), inset 0 0 20px rgba(0, 255, 255, 0.1);
        }
        .stat {
            font-size: 3em;
            color: #0f0;
            text-shadow: 0 0 20px #0f0;
        }
        .stat-label {
            font-size: 1em;
            color: #0ff;
            margin-top: 5px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 30px 0;
        }
        a {
            color: #f0f;
            text-decoration: none;
            text-shadow: 0 0 10px #f0f;
            transition: all 0.3s;
        }
        a:hover {
            color: #fff;
            text-shadow: 0 0 20px #f0f, 0 0 40px #f0f;
        }
        .btn {
            display: inline-block;
            padding: 15px 40px;
            margin: 10px;
            background: transparent;
            border: 2px solid #0ff;
            color: #0ff;
            font-family: 'Courier New', monospace;
            font-size: 1.2em;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
        }
        .btn:hover {
            background: #0ff;
            color: #000;
            box-shadow: 0 0 30px #0ff;
        }
        .tagline {
            font-size: 1.2em;
            color: #888;
            margin: 20px 0;
            max-width: 600px;
        }
        #music-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 100;
            background: rgba(0,0,0,0.8);
            border: 1px solid #0ff;
            color: #0ff;
            padding: 10px 15px;
            cursor: pointer;
            font-family: monospace;
        }
        .node-list {
            text-align: left;
            margin-top: 20px;
            font-size: 0.9em;
        }
        .node-item {
            padding: 5px 0;
            border-bottom: 1px solid rgba(0,255,255,0.2);
        }
        .online { color: #0f0; }
    </style>
</head>
<body>
    <canvas id="neural-canvas"></canvas>
    <button id="music-btn" onclick="toggleMusic()">🔊 ENABLE AUDIO</button>

    <div class="container">
        <h1>⚡ QUANTUM CHILDREN</h1>
        <h2>Distributed Neural Trading Intelligence</h2>

        <p class="tagline">
            A global network of trading nodes sharing signals, entropy data, and outcomes.
            Free to use. Powered by collective intelligence.
        </p>

        <div class="stats-grid">
            <div class="stats-box">
                <div class="stat" id="signal-count">---</div>
                <div class="stat-label">SIGNALS COLLECTED</div>
            </div>
            <div class="stats-box">
                <div class="stat" id="node-count">---</div>
                <div class="stat-label">ACTIVE NODES</div>
            </div>
            <div class="stats-box">
                <div class="stat" id="outcome-count">---</div>
                <div class="stat-label">TRADE OUTCOMES</div>
            </div>
        </div>

        <div class="stats-box" style="width: 100%; max-width: 500px;">
            <h3 style="color: #0ff; margin-bottom: 15px;">CONNECTED NODES</h3>
            <div id="node-list" class="node-list">Loading...</div>
        </div>

        <div style="margin-top: 30px;">
            <a href="/stats" class="btn">📊 RAW STATS API</a>
            <a href="https://github.com/quantumchildren" class="btn">🚀 GET THE SYSTEM</a>
        </div>

        <p style="margin-top: 40px; color: #444; font-size: 0.8em;">
            The neural network sees all. The collective learns.
        </p>
    </div>

    <script>
        // Neural Network Canvas Animation
        const canvas = document.getElementById('neural-canvas');
        const ctx = canvas.getContext('2d');
        let nodes = [];
        let mouseX = 0, mouseY = 0;

        function resize() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        resize();
        window.addEventListener('resize', resize);
        document.addEventListener('mousemove', e => { mouseX = e.clientX; mouseY = e.clientY; });

        class Node {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.vx = (Math.random() - 0.5) * 0.5;
                this.vy = (Math.random() - 0.5) * 0.5;
                this.radius = Math.random() * 2 + 1;
                this.pulsePhase = Math.random() * Math.PI * 2;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;
                if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
                if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
                this.pulsePhase += 0.02;
            }
            draw() {
                const pulse = Math.sin(this.pulsePhase) * 0.5 + 1;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius * pulse, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 255, 255, ${0.5 + pulse * 0.3})`;
                ctx.fill();
            }
        }

        // Create nodes
        for (let i = 0; i < 100; i++) nodes.push(new Node());

        function animate() {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw connections
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const dx = nodes[i].x - nodes[j].x;
                    const dy = nodes[i].y - nodes[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 150) {
                        ctx.beginPath();
                        ctx.moveTo(nodes[i].x, nodes[i].y);
                        ctx.lineTo(nodes[j].x, nodes[j].y);
                        ctx.strokeStyle = `rgba(0, 255, 255, ${(150 - dist) / 150 * 0.3})`;
                        ctx.stroke();
                    }
                }
                // Connect to mouse
                const dx = nodes[i].x - mouseX;
                const dy = nodes[i].y - mouseY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 200) {
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(mouseX, mouseY);
                    ctx.strokeStyle = `rgba(0, 255, 0, ${(200 - dist) / 200 * 0.5})`;
                    ctx.stroke();
                }
            }

            nodes.forEach(n => { n.update(); n.draw(); });
            requestAnimationFrame(animate);
        }
        animate();

        // Audio - Doctor Who style synth
        let audioCtx, isPlaying = false;
        function toggleMusic() {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (isPlaying) {
                audioCtx.suspend();
                document.getElementById('music-btn').textContent = '🔊 ENABLE AUDIO';
            } else {
                audioCtx.resume();
                playTheme();
                document.getElementById('music-btn').textContent = '🔇 MUTE AUDIO';
            }
            isPlaying = !isPlaying;
        }

        function playTheme() {
            const notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25];
            let time = audioCtx.currentTime;

            function playSequence() {
                // Bass drone
                const drone = audioCtx.createOscillator();
                const droneGain = audioCtx.createGain();
                drone.type = 'sawtooth';
                drone.frequency.value = 65.41;
                droneGain.gain.value = 0.1;
                drone.connect(droneGain).connect(audioCtx.destination);
                drone.start(time);
                drone.stop(time + 8);

                // Melody
                const melody = [0, 2, 4, 5, 4, 2, 0, 2, 4, 7, 5, 4, 2, 0];
                melody.forEach((note, i) => {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'triangle';
                    osc.frequency.value = notes[note] * 2;
                    gain.gain.setValueAtTime(0.15, time + i * 0.5);
                    gain.gain.exponentialRampToValueAtTime(0.01, time + i * 0.5 + 0.4);
                    osc.connect(gain).connect(audioCtx.destination);
                    osc.start(time + i * 0.5);
                    osc.stop(time + i * 0.5 + 0.5);
                });

                // Arpeggio
                for (let i = 0; i < 16; i++) {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'sine';
                    osc.frequency.value = notes[i % 8] * 4;
                    gain.gain.setValueAtTime(0.05, time + i * 0.25);
                    gain.gain.exponentialRampToValueAtTime(0.001, time + i * 0.25 + 0.2);
                    osc.connect(gain).connect(audioCtx.destination);
                    osc.start(time + i * 0.25);
                    osc.stop(time + i * 0.25 + 0.25);
                }

                time += 8;
                if (isPlaying) setTimeout(playSequence, 7500);
            }
            playSequence();
        }

        // Fetch live stats
        async function updateStats() {
            try {
                const res = await fetch('/stats');
                const data = await res.json();
                document.getElementById('signal-count').textContent = data.total_signals.toLocaleString();
                document.getElementById('node-count').textContent = data.active_nodes;
                document.getElementById('outcome-count').textContent = data.total_outcomes.toLocaleString();

                const nodeList = document.getElementById('node-list');
                if (data.recent_nodes && data.recent_nodes.length > 0) {
                    nodeList.innerHTML = data.recent_nodes.map(n =>
                        `<div class="node-item">
                            <span class="online">●</span> ${n.node_id}
                            <span style="color:#888">| ${n.signals} signals</span>
                        </div>`
                    ).join('');
                } else {
                    nodeList.innerHTML = '<div style="color:#888">No nodes connected yet</div>';
                }
            } catch(e) {
                console.error('Stats fetch failed:', e);
            }
        }
        updateStats();
        setInterval(updateStats, 5000);
    </script>
</body>
</html>'''

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 50)
    print("  QUANTUM CHILDREN - Data Collection Server")
    print("=" * 50)
    print(f"  Database: {DB_PATH.absolute()}")
    print(f"  Listening on: 0.0.0.0:8888")
    print("=" * 50)

    app.run(host='0.0.0.0', port=8888, debug=False)
