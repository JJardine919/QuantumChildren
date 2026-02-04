"""
ETARE QUANTUM FUSION - MAIN TRADER
==================================
The Orchestrator. Connects Quantum Brain to MT5 Body.

Usage:
    python etare_fusion_trader.py

Logic:
    1. Loop every 5 minutes (candle close).
    2. Check Quantum Compression (The Trick).
    3. If Compressible -> Run Adapters -> Fuse -> Trade.
    4. If Noise -> Hold/Manage existing positions.
"""

import time
import logging
import sys
import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Add modules to path
sys.path.append(os.path.join(os.getcwd(), "ETARE_QuantumFusion"))

from modules.compression_layer import CompressionLayer
from modules.quantum_lstm_adapter import QuantumLSTMAdapter
# from modules.quantum_3d_adapter import Quantum3DAdapter # Removed per user request
from modules.signal_fusion import SignalFusion

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ETARE_QuantumFusion/data/logs/trader.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FusionTrader")

# --- CONFIG ---
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
MAGIC_NUMBER = 73049
RISK_PERCENT = 0.01
MAX_LOTS = 1.0

class FusionTrader:
    def __init__(self):
        logger.info("Initializing ETARE Quantum Fusion...")
        
        # 1. Connect MT5
        if not mt5.initialize():
            logger.critical(f"MT5 Init Failed: {mt5.last_error()}")
            sys.exit(1)
            
        # 2. Load Modules
        self.compression = CompressionLayer()
        self.lstm = QuantumLSTMAdapter("ETARE_QuantumFusion/models/quantum_lstm_best.pth")
        # self.q3d = Quantum3DAdapter("ETARE_QuantumFusion/models/catboost_quantum_3d.cbm")
        self.fusion = SignalFusion()
        
        logger.info("All Quantum Modules Loaded.")

    def get_market_data(self, bars=256):
        """Fetch OHLVC data."""
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, bars)
        if rates is None or len(rates) < bars:
            logger.error(f"Failed to fetch data for {SYMBOL}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def execute_trade(self, action, score):
        """Execute trade based on Fusion Action."""
        # Check existing positions
        positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
        if positions and len(positions) > 0:
            logger.info(f"Position active ({len(positions)}). Skipping entry.")
            return

        # Prepare Request
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick: return

        # Calc Lot Size (Simple)
        account = mt5.account_info()
        if not account: return
        
        # Simple lot calculation (0.01 per $1000 approx for BTC)
        # lot = max(0.01, round((account.balance / 100000) * 1, 2))
        lot = 0.05 # Fixed for safety test
        
        order_type = mt5.ORDER_TYPE_BUY if "BUY" in action else mt5.ORDER_TYPE_SELL
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        sl_pips = 50000 # BTC rough scale
        tp_pips = 100000
        point = mt5.symbol_info(SYMBOL).point
        
        sl = price - (sl_pips * point) if order_type == mt5.ORDER_TYPE_BUY else price + (sl_pips * point)
        tp = price + (tp_pips * point) if order_type == mt5.ORDER_TYPE_BUY else price - (tp_pips * point)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": MAGIC_NUMBER,
            "comment": f"QF_{score:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order Failed: {result.comment} ({result.retcode})")
        else:
            logger.info(f"ORDER FILLED: {action} @ {price} (Score: {score:.2f})")

    def run_cycle(self):
        """Single analysis cycle."""
        logger.info("--- Starting Cycle ---")
        
        # 1. Data
        df = self.get_market_data(256)
        if df is None: return
        
        prices = df['close'].values
        
        # 2. Layer 1: Compression (The Gatekeeper)
        comp_res = self.compression.process(prices)
        ratio = comp_res['ratio']
        logger.info(f"Compression Ratio: {ratio:.2f} ({comp_res['regime']})")
        
        if not comp_res['tradeable']:
            logger.info(">>> MARKET NOISE DETECTED. STANDING BY.")
            return
            
        logger.info(">>> TREND DETECTED. ACTIVATING ADAPTERS.")
        
        # 3. Layer 2: Adapters
        lstm_sig = self.lstm.predict(df)
        # q3d_sig = self.q3d.analyze(df)
        
        logger.info(f"LSTM: {lstm_sig['signal']} ({lstm_sig['confidence']:.2f})")
        # logger.info(f"3D:   {q3d_sig['signal']} ({q3d_sig['probability']:.2f})")
        
        # 4. Layer 3: Fusion
        fusion_input = {
            "compression": comp_res,
            "lstm": lstm_sig,
            # "3d": q3d_sig
        }
        
        decision = self.fusion.calculate_score(fusion_input)
        score = decision['fusion_score']
        action = decision['action']
        
        logger.info(f"FUSION DECISION: {action} (Score: {score:.4f})")
        
        # 5. Execution
        if "BUY" in action or "SELL" in action:
            self.execute_trade(action, score)

    def loop(self):
        """Main Loop."""
        logger.info(f"Fusion Trader Started on {SYMBOL}")
        try:
            while True:
                self.run_cycle()
                
                # Sleep to next candle (approx)
                # For demo test, sleep 60s
                logger.info("Sleeping 60s...")
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Stopping Trader...")
            mt5.shutdown()

if __name__ == "__main__":
    trader = FusionTrader()
    trader.loop()
