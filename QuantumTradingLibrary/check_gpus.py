
import torch
import sys

def check_gpu_environment():
    """
    Checks the PyTorch environment for available GPUs (supporting NVIDIA CUDA and AMD ROCm).
    """
    print(f"--- GPU Environment Check ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")

    # For AMD ROCm, torch.cuda.is_available() is the correct function to call.
    # The PyTorch build for ROCm uses the same API endpoints as the CUDA build.
    is_available = torch.cuda.is_available()
    
    print(f"\nIs GPU acceleration available? {'🟢 YES' if is_available else '🔴 NO'}")

    if not is_available:
        print("\nTroubleshooting:")
        print("1. Ensure you have installed the correct PyTorch version for your ROCm driver.")
        print("   (e.g., pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocmX.X)")
        print("2. Verify your ROCm drivers are installed correctly and `rocminfo` command works.")
        print("3. Make sure you are running this in the correct Python environment.")
        return

    device_count = torch.cuda.device_count()
    print(f"Found {device_count} available GPU(s).")
    
    if device_count != 16:
        print(f"⚠️ WARNING: Expected 16 GPUs but found {device_count}. The training will use only the detected GPUs.")

    print("\n--- Detected GPU Details ---")
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        properties = torch.cuda.get_device_properties(i)
        print(f"  - GPU {i}: {gpu_name}")
        print(f"    - Compute Capability: {properties.major}.{properties.minor}")
        print(f"    - Total Memory: {properties.total_memory / (1024**3):.2f} GB")
    
    print("\nEnvironment check complete. Ready for multi-GPU processing.")

if __name__ == "__main__":
    check_gpu_environment()
