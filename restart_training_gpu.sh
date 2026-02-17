#!/usr/bin/env bash
#
# Restart Training with GPU Support
#
# This script stops any CPU-based training and restarts with proper GPU support
#

set -e

echo "========================================================================"
echo "  RESTARTING TRAINING WITH GPU SUPPORT"
echo "========================================================================"
echo ""

# Stop any running training workers
echo "🛑 Stopping CPU-based training workers..."
pkill -f "1_optimize_unified.py" 2>/dev/null || echo "   No workers running"
sleep 2

# Verify GPU is accessible
echo ""
echo "🔍 Verifying GPU setup..."
HSA_OVERRIDE_GFX_VERSION=11.0.0 ~/.pyenv/versions/cappuccino-rocm/bin/python test_gpu_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ GPU setup failed! Fix the issues above before continuing."
    exit 1
fi

# Check current study
if [ -f ".current_study" ]; then
    STUDY=$(cat .current_study)
    echo ""
    echo "📊 Resuming study: $STUDY"
else
    STUDY="cappuccino_auto_$(date +%Y%m%d_%H%M)"
    echo "$STUDY" > .current_study
    echo ""
    echo "📊 Starting new study: $STUDY"
fi

# Launch training with GPU
echo ""
echo "🚀 Launching training workers with GPU support..."
echo ""

for i in 1 2 3; do
    LOG_FILE="logs/worker_gpu_${i}.log"

    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    ~/.pyenv/versions/cappuccino-rocm/bin/python -u \
        scripts/training/1_optimize_unified.py \
        --n-trials 500 \
        --gpu 0 \
        --study-name "$STUDY" \
        --timeframe 1h \
        --data-dir data/1h_1680 \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "✅ Worker $i started (PID: $PID, Log: $LOG_FILE)"
    sleep 2
done

echo ""
echo "========================================================================"
echo "✅ All workers launched with GPU support!"
echo "========================================================================"
echo ""
echo "Monitor with:"
echo "  • tail -f logs/worker_gpu_*.log"
echo "  • watch -n 5 'rocm-smi'"
echo "  • python scripts/automation/trial_dashboard.py"
echo ""
echo "Check GPU usage:"
echo "  rocm-smi"
echo ""
