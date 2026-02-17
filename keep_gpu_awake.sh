#!/bin/bash
# Prevent AMD GPU from suspending (based on SWDEV-573540 workaround)

echo "=================================="
echo "AMD GPU Wake/Performance Fix"
echo "=================================="
echo ""

# 1. Disable runtime PM
echo "1. Disabling runtime power management..."
echo "on" | sudo tee /sys/class/drm/card1/device/power/control > /dev/null

# 2. Set high performance mode
echo "2. Setting high performance mode..."
echo "high" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level > /dev/null

# 3. Set power profile to compute (best for training workloads)
echo "3. Setting power profile to COMPUTE workload..."
echo "COMPUTE" | sudo tee /sys/class/drm/card1/device/pp_power_profile_mode > /dev/null 2>&1 || echo "   (Power profile not available on this GPU)"

# 4. Disable GPU reset on suspend
echo "4. Preventing GPU suspend..."
echo "0" | sudo tee /sys/class/drm/card1/device/power/runtime_auto_suspend_delay_ms > /dev/null 2>&1 || echo "   (Runtime suspend delay not available)"

# 5. Force active state
echo "5. Forcing active state..."
echo "active" | sudo tee /sys/class/drm/card1/device/power/runtime_status > /dev/null 2>&1 || echo "   (Cannot force runtime_status - read-only)"

echo ""
echo "✅ GPU configured for maximum performance"
echo ""

# Show current status
echo "Current GPU Status:"
rocm-smi --showuse
echo ""
rocm-smi --showpower

echo ""
echo "To make this persistent across reboots, create a systemd service."
echo "Run: sudo ./install_gpu_service.sh"
