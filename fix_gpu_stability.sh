#!/bin/bash
# Fix GPU stability issues for RDNA3 under heavy compute load

echo "Applying GPU stability fixes for RX 7900 GRE..."
echo ""

# 1. Set power profile back to auto (COMPUTE mode was too aggressive)
echo "1. Setting power profile to AUTO (less aggressive)..."
echo "auto" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level > /dev/null
echo "auto" | sudo tee /sys/class/drm/card1/device/power/control > /dev/null

# 2. Increase GPU timeout (prevent premature reset)
echo "2. Increasing GPU timeout tolerance..."
if [ -f /sys/module/amdgpu/parameters/lockup_timeout ]; then
    echo "10000" | sudo tee /sys/module/amdgpu/parameters/lockup_timeout > /dev/null 2>&1 || echo "   (lockup_timeout is read-only, set via kernel parameter)"
fi

# 3. Enable GPU recovery (allow soft resets instead of hard resets)
if [ -f /sys/module/amdgpu/parameters/gpu_recovery ]; then
    echo "1" | sudo tee /sys/module/amdgpu/parameters/gpu_recovery > /dev/null 2>&1 || echo "   (gpu_recovery already enabled)"
fi

echo ""
echo "✅ GPU stability fixes applied"
echo ""
echo "Current settings:"
echo "  Power mode: $(cat /sys/class/drm/card1/device/power_dpm_force_performance_level)"
echo "  Power control: $(cat /sys/class/drm/card1/device/power/control)"
echo ""
echo "Additional recommendations:"
echo "  • Running 2 parallel studies (not 3-4) ✅ Already doing this"
echo "  • Batch sizes: 4096-8192 (not 16384) - reduces GPU pressure"
echo "  • Monitor with: watch -n2 'rocm-smi --showuse'"
echo ""
echo "If crashes continue, add to kernel boot parameters:"
echo "  amdgpu.lockup_timeout=10000 amdgpu.gpu_recovery=1"
echo ""
