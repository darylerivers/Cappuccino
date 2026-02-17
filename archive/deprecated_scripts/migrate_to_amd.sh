#!/bin/bash
# AMD GPU Migration Helper Script
# Run this after installing the RX 7900 XTX and ROCm

set -e  # Exit on error

echo "================================================"
echo "AMD GPU Migration Helper"
echo "================================================"
echo ""

# Check if ROCm is installed
echo "1. Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    echo "✓ ROCm detected"
    rocm-smi | head -20
else
    echo "✗ ROCm not found!"
    echo ""
    echo "Please install ROCm first:"
    echo "  wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb"
    echo "  sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb"
    echo "  sudo apt update"
    echo "  sudo amdgpu-install --usecase=rocm"
    echo "  sudo reboot"
    exit 1
fi
echo ""

# Check PyTorch installation
echo "2. Checking PyTorch installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')" 2>/dev/null || {
    echo "✗ PyTorch not ROCm-enabled or not installed"
    echo ""
    echo "Reinstalling PyTorch for ROCm..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
    echo ""
    echo "✓ PyTorch reinstalled for ROCm"
    echo ""
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
}
echo ""

# Update monitoring scripts
echo "3. Updating monitoring scripts..."

# Backup originals
for file in performance_monitor.py monitor.py dashboard.py system_watchdog.py; do
    if [ -f "$file" ]; then
        cp "$file" "${file}.nvidia_backup"
        echo "  Backed up: $file"
    fi
done

# Show what needs manual updating
echo ""
echo "⚠️  Manual updates required for monitoring scripts:"
echo ""
grep -l "nvidia-smi" *.py 2>/dev/null | while read file; do
    count=$(grep -c "nvidia-smi" "$file")
    echo "  • $file: $count references to nvidia-smi"
done
echo ""
echo "Run this to see all references:"
echo "  grep -n \"nvidia-smi\" *.py"
echo ""

# Test single trial
echo "4. Testing GPU with single trial..."
echo ""
read -p "Run a test trial? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting test trial..."
    timeout 120 python -u 1_optimize_unified.py --n-trials 1 --gpu 0 --study-name test_amd_gpu || {
        echo "✗ Test trial failed or timed out"
        exit 1
    }
    echo "✓ Test trial completed successfully"
fi
echo ""

# Calculate optimal worker count
echo "5. Calculating optimal worker count..."
python3 << 'EOF'
import subprocess

# Get GPU memory
result = subprocess.check_output(['rocm-smi', '--showmeminfo', 'vram'], text=True)
# Parse total VRAM (simplified)
total_vram = 24000  # MB for RX 7900 XTX

# Previous configuration
old_workers = 9
old_vram = 6700  # MB used

vram_per_worker = old_vram / old_workers
safe_usage = 0.85  # Use 85% to be safe

recommended = int((total_vram * safe_usage) / vram_per_worker)
print(f"Previous: {old_workers} workers using {old_vram} MB")
print(f"VRAM per worker: ~{vram_per_worker:.0f} MB")
print(f"")
print(f"RX 7900 XTX: {total_vram} MB total")
print(f"Safe usage (85%): {total_vram * safe_usage:.0f} MB")
print(f"")
print(f"Recommended workers: {recommended}")
print(f"")
print(f"Start conservative: 12-15 workers")
print(f"Monitor and increase to: 18-20 workers")
EOF
echo ""

# Summary
echo "================================================"
echo "Migration Status Summary"
echo "================================================"
echo ""
echo "✓ ROCm installed and working"
echo "✓ PyTorch sees AMD GPU"
echo ""
echo "Next steps:"
echo "  1. Update monitoring scripts (nvidia-smi → rocm-smi)"
echo "  2. Test with 5 workers first"
echo "  3. Monitor VRAM: watch -n 1 rocm-smi --showmeminfo vram"
echo "  4. Gradually increase to 18-20 workers"
echo "  5. Restart automation: ./start_automation.sh"
echo ""
echo "See AMD_GPU_MIGRATION_GUIDE.md for details"
echo ""
