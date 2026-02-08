"""
Quantum Children Signal Emitter for n8n Network
================================================

This script emits trading signals to the n8n workflow network.
It integrates with the existing Quantum Children trading system
and broadcasts signals for network distribution.

Usage:
    python signal_emitter.py --webhook http://your-n8n/webhook/quantum-signal

Or run as a service:
    python signal_emitter.py --service --interval 60

"""

import json
import time
import argparse
import requests
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'QuantumTradingLibrary'))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARNING] MetaTrader5 not available - using mock data")


class QuantumSignalEmitter:
    """
    Emits Quantum Children trading signals to n8n network.

    This creates the cascade effect by:
    1. Reading signals from the local Quantum system
    2. Broadcasting to n8n webhook
    3. n8n distributes to Telegram/Discord and other nodes
    """

    def __init__(self, webhook_url: str, node_id: str = None):
        self.webhook_url = webhook_url
        self.node_id = node_id or f"QC-{datetime.now().strftime('%Y%m%d')}"
        self.signal_history = []

        # Signal file paths (from Quantum Children system)
        self.signal_paths = [
            Path(__file__).parent.parent / 'QuantumTradingLibrary' / 'BlueGuardian_Deploy' / 'signal_BG_5K_INSTANT.json',
            Path(__file__).parent.parent / 'QuantumTradingLibrary' / 'BlueGuardian_Deploy' / 'signal_BG_100K_CHALLENGE.json',
            Path(__file__).parent.parent / 'QuantumTradingLibrary' / 'BlueGuardian_Deploy' / 'signal_ATLAS_300K_GRID.json',
        ]

    def read_local_signal(self) -> dict:
        """Read signal from local Quantum Children system"""
        for path in self.signal_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        signal = json.load(f)
                        if signal.get('action') not in ['HOLD', None]:
                            return signal
                except Exception as e:
                    print(f"[ERROR] Reading {path}: {e}")
        return None

    def fetch_mt5_data(self, symbol: str = 'BTCUSD') -> dict:
        """Fetch current market data from MT5"""
        if not MT5_AVAILABLE:
            return self._mock_market_data(symbol)

        if not mt5.initialize():
            print("[ERROR] MT5 initialization failed")
            return self._mock_market_data(symbol)

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return self._mock_market_data(symbol)

        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'time': datetime.fromtimestamp(tick.time).isoformat()
        }

    def _mock_market_data(self, symbol: str) -> dict:
        """Generate mock data when MT5 unavailable"""
        import random
        base_prices = {
            'BTCUSD': 95000,
            'ETHUSD': 3200,
            'XAUUSD': 2050
        }
        base = base_prices.get(symbol, 1000)
        price = base * (1 + random.uniform(-0.01, 0.01))
        return {
            'symbol': symbol,
            'bid': price,
            'ask': price * 1.0001,
            'time': datetime.now().isoformat(),
            'mock': True
        }

    def generate_signal(self, symbol: str = 'BTCUSD') -> dict:
        """
        Generate a trading signal for broadcast.

        In production, this reads from the Quantum Children system.
        The signal includes quantum entropy which is key to the value prop.
        """
        # Try to read existing signal
        local_signal = self.read_local_signal()
        if local_signal and local_signal.get('action') not in ['HOLD', 'INSUFFICIENT_DATA']:
            signal = {
                'symbol': local_signal.get('symbol', symbol),
                'action': local_signal.get('action', 'HOLD'),
                'confidence': local_signal.get('confidence', 0.5),
                'quantum_entropy': local_signal.get('quantum_entropy', 4.0),
                'dominant_state': local_signal.get('dominant_state', 0.05),
                'catboost_prob': local_signal.get('catboost_prob', 0.5),
                'llm_adjustment': local_signal.get('llm_adjustment', 0),
                'timestamp': datetime.now().isoformat(),
                'source_node': self.node_id,
                'cascade_hop': 0,
                'cascade_path': [self.node_id]
            }
        else:
            # Generate demo signal for network demonstration
            import random
            market_data = self.fetch_mt5_data(symbol)

            # Simulate quantum analysis
            entropy = random.uniform(1.5, 6.0)
            confidence = max(0.3, 0.9 - (entropy * 0.1))

            signal = {
                'symbol': symbol,
                'action': 'BUY' if random.random() > 0.5 else 'SELL',
                'confidence': round(confidence, 3),
                'quantum_entropy': round(entropy, 3),
                'dominant_state': round(random.uniform(0.02, 0.15), 4),
                'catboost_prob': round(random.uniform(0.4, 0.8), 3),
                'llm_adjustment': round(random.uniform(-0.05, 0.05), 3),
                'price': market_data.get('bid', 0),
                'timestamp': datetime.now().isoformat(),
                'source_node': self.node_id,
                'cascade_hop': 0,
                'cascade_path': [self.node_id],
                'demo_mode': True
            }

        return signal

    def emit_signal(self, signal: dict) -> bool:
        """Send signal to n8n webhook"""
        try:
            response = requests.post(
                self.webhook_url,
                json=signal,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                print(f"[OK] Signal emitted: {signal['symbol']} {signal['action']} "
                      f"(conf: {signal['confidence']:.1%}, entropy: {signal['quantum_entropy']:.2f})")
                self.signal_history.append({
                    'signal': signal,
                    'emitted_at': datetime.now().isoformat(),
                    'status': 'success'
                })
                return True
            else:
                print(f"[ERROR] Webhook returned {response.status_code}: {response.text}")
                return False

        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to webhook: {self.webhook_url}")
            return False
        except Exception as e:
            print(f"[ERROR] Emit failed: {e}")
            return False

    def register_node(self, register_url: str, config: dict) -> dict:
        """Register this node with the network"""
        registration = {
            'node_id': self.node_id,
            'capabilities': ['signal_generation', 'quantum_analysis'],
            'symbols': ['BTCUSD', 'ETHUSD', 'XAUUSD'],
            'cascade_webhook': config.get('cascade_webhook'),
            'registered_at': datetime.now().isoformat()
        }

        try:
            response = requests.post(register_url, json=registration, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"[ERROR] Registration failed: {e}")

        return None

    def run_service(self, interval: int = 60, symbols: list = None):
        """Run as a continuous service"""
        symbols = symbols or ['BTCUSD', 'ETHUSD', 'XAUUSD']
        print(f"[START] Signal Emitter Service")
        print(f"        Node ID: {self.node_id}")
        print(f"        Webhook: {self.webhook_url}")
        print(f"        Interval: {interval}s")
        print(f"        Symbols: {', '.join(symbols)}")
        print("-" * 50)

        while True:
            for symbol in symbols:
                signal = self.generate_signal(symbol)

                # Only emit high-quality signals (low entropy, decent confidence)
                if signal['quantum_entropy'] < 6.0 and signal['confidence'] > 0.5:
                    self.emit_signal(signal)
                else:
                    print(f"[SKIP] {symbol}: entropy={signal['quantum_entropy']:.2f}, "
                          f"conf={signal['confidence']:.1%} (below threshold)")

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description='Quantum Children Signal Emitter')
    parser.add_argument('--webhook', type=str,
                        default='http://localhost:5678/webhook/quantum-signal',
                        help='n8n webhook URL for signals')
    parser.add_argument('--service', action='store_true',
                        help='Run as continuous service')
    parser.add_argument('--interval', type=int, default=60,
                        help='Emission interval in seconds (service mode)')
    parser.add_argument('--symbols', nargs='+',
                        default=['BTCUSD', 'ETHUSD', 'XAUUSD'],
                        help='Symbols to monitor')
    parser.add_argument('--node-id', type=str, default=None,
                        help='Custom node ID for network')
    parser.add_argument('--demo', action='store_true',
                        help='Run single demo emission')

    args = parser.parse_args()

    emitter = QuantumSignalEmitter(
        webhook_url=args.webhook,
        node_id=args.node_id
    )

    if args.demo:
        print("[DEMO] Generating and emitting test signal...")
        for symbol in args.symbols:
            signal = emitter.generate_signal(symbol)
            print(f"\nGenerated Signal:")
            print(json.dumps(signal, indent=2))
            emitter.emit_signal(signal)
    elif args.service:
        emitter.run_service(interval=args.interval, symbols=args.symbols)
    else:
        # Single emission
        for symbol in args.symbols:
            signal = emitter.generate_signal(symbol)
            emitter.emit_signal(signal)


if __name__ == '__main__':
    main()
