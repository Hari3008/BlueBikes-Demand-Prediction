import torch
import sys

print("="*60)
print("PyTorch GPU Diagnostics")
print("="*60)

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test GPU
    print("\nTesting GPU computation...")
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print("GPU computation successful!")
else:
    print("\n❌ CUDA not available")
    print("\nPossible reasons:")
    print("1. PyTorch CPU-only version installed")
    print("2. NVIDIA drivers not installed")
    print("3. No NVIDIA GPU in system")
    print("\nTo fix:")
    print("pip uninstall torch")
    print("pip install torch --index-url https://download.pytorch.org/whl/cu118")