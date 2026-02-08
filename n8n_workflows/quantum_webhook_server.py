"""
Quantum Children Webhook Server
================================

A lightweight Flask server that:
1. Receives signals from the Quantum Children trading system
2. Broadcasts to n8n network nodes
3. Maintains a public API for signal consumers

This is the ORIGIN NODE for the cascade network.

Usage:
    python quantum_webhook_server.py --port 8889

Endpoints:
    GET  /                    - Health check / network info
    POST /signal              - Submit a new signal (from Quantum system)
    GET  /signals/latest      - Get latest signals
    POST /register            - Register as a downstream node
    GET  /network/status      - Network statistics
"""

from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import threading
import requests
import json
import os
from collections import deque

app = Flask(__name__)

# Configuration
CONFIG = {
    'node_id': os.environ.get('NODE_ID', 'QC-ORIGIN'),
    'max_signal_history': 100,
    'cascade_timeout': 5,
    'n8n_webhook': os.environ.get('N8N_WEBHOOK', 'http://localhost:5678/webhook/quantum-signal')
}

# State
signal_history = deque(maxlen=CONFIG['max_signal_history'])
downstream_nodes = []
network_stats = {
    'signals_received': 0,
    'signals_cascaded': 0,
    'active_nodes': 0,
    'started_at': datetime.now().isoformat()
}


@app.route('/')
def index():
    """Health check and network info"""
    return jsonify({
        'name': 'Quantum Children Signal Network',
        'node_id': CONFIG['node_id'],
        'version': '1.0.0',
        'status': 'online',
        'endpoints': {
            'submit_signal': 'POST /signal',
            'latest_signals': 'GET /signals/latest',
            'register_node': 'POST /register',
            'network_status': 'GET /network/status'
        },
        'how_to_join': 'Import n8n workflow from quantum-children.network/workflow',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/signal', methods=['POST'])
def receive_signal():
    """
    Receive signal from Quantum Children system.
    Validates, enriches, and cascades to network.
    """
    try:
        signal = request.get_json()

        # Validate required fields
        required = ['symbol', 'action']
        if not all(k in signal for k in required):
            return jsonify({'error': 'Missing required fields', 'required': required}), 400

        # Enrich signal
        enriched = {
            **signal,
            'source_node': CONFIG['node_id'],
            'received_at': datetime.now().isoformat(),
            'cascade_hop': 0,
            'cascade_path': [CONFIG['node_id']],
            'signal_id': f"{CONFIG['node_id']}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }

        # Add quality assessment
        entropy = signal.get('quantum_entropy', 4.0)
        confidence = signal.get('confidence', 0.5)
        enriched['quality'] = 'HIGH' if entropy < 2.5 and confidence > 0.6 else \
                              'MEDIUM' if entropy < 4.5 and confidence > 0.5 else 'LOW'

        # Store in history
        signal_history.append(enriched)
        network_stats['signals_received'] += 1

        # Cascade to n8n (async)
        threading.Thread(target=cascade_signal, args=(enriched,)).start()

        return jsonify({
            'status': 'accepted',
            'signal_id': enriched['signal_id'],
            'quality': enriched['quality'],
            'will_cascade': enriched['quality'] != 'LOW'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/signals/latest', methods=['GET'])
def latest_signals():
    """Get recent signals"""
    count = min(int(request.args.get('count', 10)), 50)
    quality = request.args.get('quality', None)

    signals = list(signal_history)[-count:]

    if quality:
        signals = [s for s in signals if s.get('quality') == quality.upper()]

    return jsonify({
        'count': len(signals),
        'signals': signals
    })


@app.route('/register', methods=['POST'])
def register_node():
    """Register a downstream node for cascade"""
    try:
        data = request.get_json()
        webhook_url = data.get('webhook_url')

        if not webhook_url:
            return jsonify({'error': 'webhook_url required'}), 400

        # Validate URL is reachable
        try:
            resp = requests.get(webhook_url.replace('/cascade-signal', '/network-info'),
                               timeout=5)
        except:
            pass  # Optional - node might not have info endpoint

        node_info = {
            'webhook_url': webhook_url,
            'registered_at': datetime.now().isoformat(),
            'node_id': data.get('node_id', f"downstream-{len(downstream_nodes)}"),
            'capabilities': data.get('capabilities', ['relay'])
        }

        downstream_nodes.append(node_info)
        network_stats['active_nodes'] = len(downstream_nodes)

        return jsonify({
            'status': 'registered',
            'your_node_info': node_info,
            'origin_node': CONFIG['node_id'],
            'message': 'You will now receive cascaded signals'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/network/status', methods=['GET'])
def network_status():
    """Network statistics"""
    return jsonify({
        'node_id': CONFIG['node_id'],
        'is_origin': True,
        'stats': network_stats,
        'downstream_count': len(downstream_nodes),
        'signal_history_size': len(signal_history),
        'uptime_since': network_stats['started_at']
    })


def cascade_signal(signal):
    """Broadcast signal to n8n and downstream nodes"""
    # Only cascade quality signals
    if signal.get('quality') == 'LOW':
        return

    # Send to local n8n
    try:
        resp = requests.post(
            CONFIG['n8n_webhook'],
            json=signal,
            timeout=CONFIG['cascade_timeout']
        )
        if resp.status_code == 200:
            network_stats['signals_cascaded'] += 1
            print(f"[CASCADE] Signal {signal['signal_id']} sent to n8n")
    except Exception as e:
        print(f"[ERROR] n8n cascade failed: {e}")

    # Send to registered downstream nodes
    for node in downstream_nodes:
        try:
            requests.post(
                node['webhook_url'],
                json=signal,
                timeout=CONFIG['cascade_timeout']
            )
        except:
            pass  # Don't block on failed downstream


def signal_generator_demo():
    """
    Demo: Generate periodic signals.
    In production, signals come from the Quantum Children system.
    """
    import random
    import time

    symbols = ['BTCUSD', 'ETHUSD', 'XAUUSD']

    while True:
        time.sleep(300)  # Every 5 minutes

        symbol = random.choice(symbols)
        entropy = random.uniform(1.5, 6.0)
        confidence = max(0.3, 0.9 - (entropy * 0.1))

        signal = {
            'symbol': symbol,
            'action': 'BUY' if random.random() > 0.5 else 'SELL',
            'confidence': round(confidence, 3),
            'quantum_entropy': round(entropy, 3),
            'dominant_state': round(random.uniform(0.02, 0.15), 4),
            'catboost_prob': round(random.uniform(0.4, 0.8), 3),
            'demo_mode': True
        }

        # Self-submit
        with app.test_client() as client:
            client.post('/signal', json=signal)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Quantum Children Webhook Server')
    parser.add_argument('--port', type=int, default=8889)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--demo', action='store_true', help='Run with demo signal generator')
    parser.add_argument('--n8n-webhook', type=str, default=CONFIG['n8n_webhook'])

    args = parser.parse_args()
    CONFIG['n8n_webhook'] = args.n8n_webhook

    print(f"""
    ============================================
    Quantum Children Signal Network - Origin Node
    ============================================
    Node ID:     {CONFIG['node_id']}
    Port:        {args.port}
    n8n Webhook: {CONFIG['n8n_webhook']}
    Demo Mode:   {args.demo}
    ============================================
    """)

    if args.demo:
        demo_thread = threading.Thread(target=signal_generator_demo, daemon=True)
        demo_thread.start()
        print("[DEMO] Signal generator started (every 5 min)")

    app.run(host=args.host, port=args.port, debug=False)
