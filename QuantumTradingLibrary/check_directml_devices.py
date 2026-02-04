import torch_directml
count = torch_directml.device_count()
print(f"DirectML Devices: {count}")
for i in range(count):
    print(f"Device {i}: {torch_directml.device_name(i)}")
