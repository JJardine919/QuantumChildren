"""
QUANTUM CHILDREN - DATA COLLECTION SERVER
==========================================
Receives entropy data from distributed trading nodes.

Deploy on VPS:
    pip install flask
    python collection_server.py

Or with gunicorn:
    gunicorn -w 4 -b 0.0.0.0:8888 collection_server:app
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

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

@app.route('/ping', methods=['POST'])
def ping():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'server': 'QuantumChildren', 'time': datetime.utcnow().isoformat()})

@app.route('/collect', methods=['POST'])
@app.route('/signal', methods=['POST'])
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
