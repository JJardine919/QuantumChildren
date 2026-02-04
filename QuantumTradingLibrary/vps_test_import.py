import sys
sys.path.insert(0, "/opt/trading/ETARE_QuantumFusion/modules")
try:
    from signal_fusion import SignalFusionEngine
    print("SUCCESS: SignalFusionEngine imported")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
