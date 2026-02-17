#!/bin/bash
# Force GPU to high performance mode (requires sudo)

echo "Setting GPU to high performance mode..."
echo "high" | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level > /dev/null

echo "Disabling runtime power management..."
echo "on" | sudo tee /sys/class/drm/card1/device/power/control > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ GPU now in high performance mode (no auto-sleep)"
    echo ""
    sleep 2
    rocm-smi --showuse
else
    echo "❌ Failed - run with sudo password"
fi
