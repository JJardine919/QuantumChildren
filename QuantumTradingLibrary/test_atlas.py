import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ETARE_QuantumFusion', 'modules'))

with open('test_result.txt', 'w') as f:
    try:
        import MetaTrader5 as mt5
        import numpy as np
        import pandas as pd
        import time
        from datetime import datetime
        from signal_fusion import SignalFusionEngine

        f.write("Imports OK\n")

        path = r"C:\Program Files\Atlas Funded MT5 Terminal\terminal64.exe"
        if not mt5.initialize(path=path):
            f.write(f"MT5 init failed: {mt5.last_error()}\n")
            sys.exit(1)

        if not mt5.login(212000584, password='M6NLk79MN@', server='AtlasFunded-Server'):
            f.write(f"Login failed: {mt5.last_error()}\n")
            mt5.shutdown()
            sys.exit(1)

        acc = mt5.account_info()
        f.write(f"Connected: {acc.login} Balance ${acc.balance}\n")

        config_path = os.path.join(os.path.dirname(__file__), 'ETARE_QuantumFusion', 'config', 'config.yaml')
        engine = SignalFusionEngine(config_path)
        f.write("Fusion engine OK\n")

        # Test get_fused_signal
        fused = engine.get_fused_signal('BTCUSD', mt5.TIMEFRAME_M5)
        f.write(f"Fused signal: {fused}\n")

        mt5.shutdown()
        f.write("ALL TESTS PASSED\n")

    except Exception as e:
        f.write(f"ERROR: {e}\n")
        f.write(traceback.format_exc())
