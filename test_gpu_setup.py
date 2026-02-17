#!/usr/bin/env python3
"""Test GPU setup for training."""

import torch
import sys

print("="*70)
print("GPU SETUP TEST")
print("="*70)
print()

print(f"PyTorch version: {torch.__version__}")
print(f"Python: {sys.executable}")
print()

# Check CUDA/ROCm availability
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"✅ GPU is accessible!")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

    # Test a simple GPU operation
    print()
    print("Testing GPU computation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"✅ GPU computation successful! Result shape: {z.shape}")

    # Check memory
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"   GPU memory allocated: {allocated:.2f}GB")
    print(f"   GPU memory reserved: {reserved:.2f}GB")

else:
    print(f"❌ GPU is NOT accessible!")
    print(f"   This means training will run on CPU (slow + high RAM usage)")
    print()
    print("Fix this by:")
    print("  1. Use the correct Python environment:")
    print(f"     ~/.pyenv/versions/cappuccino-rocm/bin/python")
    print("  2. Set ROCm environment variable:")
    print(f"     export HSA_OVERRIDE_GFX_VERSION=11.0.0")
    sys.exit(1)

print()
print("="*70)
print("✅ GPU SETUP IS CORRECT - Ready for training!")
print("="*70)
