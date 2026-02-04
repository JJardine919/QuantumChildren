print("Starting script...", flush=True)
import sys
print("Imported sys", flush=True)
import MetaTrader5 as mt5
print("Imported MT5", flush=True)
import torch
print("Imported Torch", flush=True)

if not mt5.initialize():
    print("MT5 init failed", flush=True)
else:
    print("MT5 init success", flush=True)
    mt5.shutdown()
