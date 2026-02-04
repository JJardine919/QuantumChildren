import torch
import torch_directml
import MetaTrader5 as mt5
import pandas as pd

print("All imports OK")
print("PyTorch:", torch.__version__)
print("GPU:", torch_directml.device_name(0))

# Quick GPU test
device = torch_directml.device()
x = torch.randn(1000, 1000).to(device)
y = torch.matmul(x, x)
print("GPU compute: OK")
