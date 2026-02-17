#!/bin/bash
# Install PyTorch with ROCm 7.2.0 support
# Run this AFTER rebooting from ROCm installation

set -e

echo "=========================================="
echo "PyTorch ROCm 7.2.0 Installation"
echo "=========================================="
echo ""

# Verify ROCm is working
echo "1. Verifying ROCm..."
if ! command -v rocm-smi &> /dev/null; then
    echo "✗ ROCm not found. Run 2_install_rocm.sh first"
    exit 1
fi

echo "✓ ROCm installed"
rocm-smi --showproductname

echo ""
echo "2. Backing up current Python environment..."
pip freeze > backups/pre_amd_migration/requirements_pre_pytorch_rocm.txt

echo ""
echo "3. Uninstalling CUDA PyTorch..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "4. Installing PyTorch with ROCm 6.2 support..."
echo "   Note: PyTorch officially supports ROCm 6.2 as of now"
echo "   ROCm 7.2.0 is backward compatible with ROCm 6.2 PyTorch builds"
echo ""
echo "   Installing PyTorch 2.6+ with ROCm 6.2 wheel..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

echo ""
echo "Note: PyTorch uses ROCm 6.2 wheels which are compatible with ROCm 7.2.0"
echo "      When PyTorch releases rocm7.2 wheels, you can upgrade with:"
echo "      pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2"

echo ""
echo "5. Verifying PyTorch + ROCm 7.2.0 installation..."
python3 << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"ROCm version (PyTorch wheel): {torch.version.hip}")
    print(f"GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Multi-processor count: {props.multi_processor_count}")

    # Check if this is the expected 16GB card
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem > 15:
        print(f"\n✓ Confirmed: 16GB VRAM detected ({total_mem:.1f} GB)")
    else:
        print(f"\n⚠ Warning: Expected 16GB, detected {total_mem:.1f} GB")

else:
    print("\n✗ ERROR: ROCm not detected by PyTorch!")
    print("\nTroubleshooting:")
    print("  1. Verify ROCm installed: rocm-smi --showproductname")
    print("  2. Check groups: groups (should include 'video' and 'render')")
    print("  3. Try reinstalling: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
    sys.exit(1)

print("\n✓ PyTorch + ROCm 7.2.0 verified successfully!")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ PyTorch ROCm installation successful!"
else
    echo ""
    echo "✗ PyTorch ROCm verification failed"
    exit 1
fi

echo ""
echo "6. Testing tensor operations..."
python3 << 'EOF'
import torch

# Create tensors on GPU
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')

# Matrix multiplication
z = torch.matmul(x, y)

print(f"✓ GPU tensor operations working")
print(f"  Result shape: {z.shape}")
print(f"  Device: {z.device}")
EOF

echo ""
echo "=========================================="
echo "PyTorch ROCm Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: ./infrastructure/amd_migration/4_verify_amd_setup.sh"
echo "  2. Then: ./infrastructure/amd_migration/5_update_training_config.sh"
