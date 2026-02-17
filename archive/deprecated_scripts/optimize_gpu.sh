#!/bin/bash
# GPU Performance Optimization Script for RTX 3070
# Maximizes training performance

echo "=== GPU Performance Optimization ==="
echo ""

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then
    echo "Note: Some optimizations require sudo. Run with: sudo ./optimize_gpu.sh"
    echo "Applying user-level optimizations only..."
    SUDO=""
else
    SUDO="sudo"
fi

echo "Current GPU Status:"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu --format=csv,noheader

echo ""
echo "=== Applying Optimizations ==="
echo ""

# 1. Enable Persistence Mode (requires sudo)
echo "[1/5] Enabling Persistence Mode..."
if [ -n "$SUDO" ]; then
    sudo nvidia-smi -pm 1
    echo "  ✓ Persistence mode enabled (keeps GPU initialized, reduces latency)"
else
    echo "  ⚠ Skipped (requires sudo)"
fi

# 2. Set Compute Mode to EXCLUSIVE_PROCESS (requires sudo)
echo ""
echo "[2/5] Setting Compute Mode to EXCLUSIVE_PROCESS..."
if [ -n "$SUDO" ]; then
    sudo nvidia-smi -c EXCLUSIVE_PROCESS
    echo "  ✓ Compute mode set (optimizes for single compute application)"
else
    echo "  ⚠ Skipped (requires sudo)"
fi

# 3. Set GPU to max performance (P0 state)
echo ""
echo "[3/5] Locking GPU to max performance state..."
if [ -n "$SUDO" ]; then
    sudo nvidia-smi -lgc 2100,2100  # Lock graphics clock to max
    sudo nvidia-smi -lmc 7001,7001  # Lock memory clock to max
    echo "  ✓ GPU locked to max clocks (Graphics: 2100 MHz, Memory: 7001 MHz)"
else
    echo "  ⚠ Skipped (requires sudo)"
fi

# 4. Increase power limit to max (already at 220W but ensuring it)
echo ""
echo "[4/5] Ensuring max power limit..."
if [ -n "$SUDO" ]; then
    sudo nvidia-smi -pl 220
    echo "  ✓ Power limit: 220W (maximum)"
else
    echo "  ⚠ Skipped (requires sudo)"
fi

# 5. Set fan to more aggressive curve for better cooling under sustained load
echo ""
echo "[5/5] Fan optimization (manual control)..."
echo "  Note: Automatic fan control is recommended for safety"
echo "  Current fan speed: $(nvidia-smi --query-gpu=fan.speed --format=csv,noheader)"
echo "  To manually set fan (USE WITH CAUTION): sudo nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=75'"
echo "  ⚠ Manual fan control disabled by default for safety"

echo ""
echo "=== Optimization Complete ==="
echo ""
echo "New GPU Status:"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu --format=csv,noheader

echo ""
echo "Performance Tips:"
echo "  • Persistence mode reduces job startup latency by ~100ms"
echo "  • Compute mode optimizes scheduler for training workloads"
echo "  • Locked clocks prevent throttling during training"
echo "  • Monitor temps: should stay under 83°C (current throttle point)"
echo ""
echo "To reset to defaults: sudo nvidia-smi -rgc && sudo nvidia-smi -rmc && sudo nvidia-smi -pm 0 && sudo nvidia-smi -c DEFAULT"
