"""Quick GPU compute test"""
import time
import torch

try:
    import torch_directml
    device = torch_directml.device()
    device_name = torch_directml.device_name(0)
    print(f"Device: {device_name}")

    # Test matrix multiplication on GPU
    print("\nTesting GPU computation...")
    x = torch.randn(5000, 5000).to(device)

    start = time.time()
    for _ in range(10):
        y = torch.matmul(x, x)
    _ = y.cpu()  # Force sync by moving to CPU
    gpu_time = time.time() - start
    print(f"GPU Time (10 matmuls of 5000x5000): {gpu_time:.3f}s")

    # Compare to CPU
    x_cpu = torch.randn(5000, 5000)
    start = time.time()
    for _ in range(10):
        y = torch.matmul(x_cpu, x_cpu)
    cpu_time = time.time() - start
    print(f"CPU Time (10 matmuls of 5000x5000): {cpu_time:.3f}s")

    speedup = cpu_time / gpu_time
    print(f"\nGPU Speedup: {speedup:.1f}x faster")

    if speedup > 1.5:
        print("GPU WORKING - READY FOR TRAINING")
    else:
        print("WARNING: GPU may not be providing expected speedup")

except ImportError as e:
    print(f"DirectML not available: {e}")
except Exception as e:
    print(f"Error: {e}")
