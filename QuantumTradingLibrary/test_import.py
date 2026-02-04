import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ETARE_QuantumFusion', 'modules'))
with open('test_result.txt', 'w') as f:
    try:
        from signal_fusion import SignalFusionEngine
        f.write("SUCCESS: SignalFusionEngine imported\n")
    except Exception as e:
        f.write(f"ERROR: {e}\n")
        import traceback
        f.write(traceback.format_exc())
