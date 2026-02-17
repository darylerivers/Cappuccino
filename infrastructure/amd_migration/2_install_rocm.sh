#!/bin/bash
# ROCm 7.2.0 Installation for RX 7900 GRE on Arch Linux
# Run this AFTER installing the RX 7900 GRE

set -e

echo "=========================================="
echo "ROCm 7.2.0 Installation for RX 7900 GRE"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: Do not run as root. Run as regular user."
    exit 1
fi

# Check GPU
echo "1. Detecting GPU..."
if lspci | grep -i "AMD.*7900" > /dev/null; then
    echo "✓ RX 7900 detected"
    lspci | grep -i "7900"
else
    echo "✗ RX 7900 not detected. Is it installed?"
    echo "Current GPUs:"
    lspci | grep -i "VGA\|3D"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "2. Installing ROCm 7.2.0 via pacman..."
echo "   This will install:"
echo "   - rocm-core (7.2.0)"
echo "   - rocm-hip-sdk (HIP development)"
echo "   - rocm-opencl-sdk (OpenCL support)"
echo "   - rocblas & hipblas (BLAS libraries)"
echo "   - miopen-hip (ML primitives)"
echo "   - rocrand (Random number generation)"
echo ""

# Install ROCm 7.2.0 from Arch repos
# Note: Arch may have slightly older versions - this installs latest available
# Note: rock-dkms removed - not needed on modern kernels
sudo pacman -S --needed \
    rocm-core \
    rocm-hip-sdk \
    rocm-opencl-sdk \
    rocblas \
    hipblas \
    miopen-hip \
    rocrand \
    rocm-smi-lib

echo ""
echo "Installed ROCm version:"
pacman -Qi rocm-core | grep Version

echo ""
echo "3. Adding user to video/render groups..."
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

echo ""
echo "4. Setting up environment variables for ROCm 7.2.0..."

# Check if ROCm variables already exist
if ! grep -q "ROCM_PATH" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# ROCm 7.2.0 Environment
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# ROCm GPU selection (for multi-GPU systems)
export HIP_VISIBLE_DEVICES=0

# Enable ROCm SMI
export PATH=$PATH:/opt/rocm/bin
EOF
    echo "✓ Environment variables added to ~/.bashrc"
else
    echo "✓ ROCm environment variables already configured"
fi

source ~/.bashrc

echo ""
echo "5. Verifying ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    echo "✓ rocm-smi installed"
    rocm-smi --showproductname
else
    echo "✗ rocm-smi not found"
    exit 1
fi

echo ""
echo "6. Testing HIP..."
if command -v hipconfig &> /dev/null; then
    echo "✓ HIP installed"
    hipconfig --version
else
    echo "✗ HIP not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "ROCm 7.2.0 Installation Complete!"
echo "=========================================="
echo ""
echo "Installed versions:"
rocm-smi --version 2>/dev/null || echo "  (rocm-smi will be available after reboot)"
hipconfig --version 2>/dev/null || echo "  (hipconfig will be available after reboot)"
echo ""
echo "IMPORTANT: You must REBOOT for group changes to take effect"
echo ""
echo "After reboot, verify with:"
echo "  1. rocm-smi --showproductname"
echo "  2. rocminfo | grep gfx"
echo "  3. ./infrastructure/amd_migration/3_install_pytorch_rocm.sh"
echo ""
echo "ROCm 7.2.0 Features:"
echo "  ✓ Enhanced HIP runtime performance"
echo "  ✓ Optimized graph node scaling"
echo "  ✓ Better async handler performance"
echo "  ✓ Improved memset operations"
echo "  ✓ Full RDNA3 support (RX 7900 GRE)"
echo ""
read -p "Reboot now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi
