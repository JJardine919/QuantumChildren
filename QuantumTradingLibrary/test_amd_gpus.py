
import torch
try:
    import torch_directml
    print(f"DirectML Available: True")
    count = torch_directml.device_count()
    print(f"Number of AMD/DirectML Devices: {count}")
    for i in range(count):
        print(f"  Device {i}: {torch_directml.device_name(i)}")
except ImportError:
    print("DirectML not installed.")
