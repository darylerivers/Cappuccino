#!/usr/bin/env bash
#
# Start GPU-Optimized Training
#
# Constrains Optuna to GPU-friendly hyperparameters:
# - Large batch sizes (1024-4096)
# - Moderate worker_num (4-8)
# - Sequential execution to avoid OOM
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create new study with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M)
STUDY="cappuccino_gpu_${TIMESTAMP}"

N_TRIALS="${1:-100}"
N_ENVS="${2:-12}"
GPU="${3:-0}"

echo "========================================================================"
echo "  GPU-OPTIMIZED TRAINING"
echo "========================================================================"
echo ""
echo "Study:       $STUDY (NEW)"
echo "Trials:      $N_TRIALS (sequential, one at a time)"
echo "GPU Envs:    $N_ENVS parallel environments"
echo "GPU ID:      $GPU"
echo ""
echo "Hyperparameter Constraints (GPU-optimized):"
echo "  • batch_size:  1024-4096  (large batches for GPU)"
echo "  • worker_num:  4-8        (reduce CPU overhead)"
echo "  • net_dimension: 512-2048 (reasonable network size)"
echo ""
echo "This will create a NEW study with GPU-optimized search space."
echo "Press Ctrl+C to stop gracefully between trials."
echo ""
read -p "Start GPU-optimized training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Kill existing workers
if pgrep -f "1_optimize_unified" > /dev/null; then
    echo ""
    echo "⚠️  Found existing workers running"
    read -p "Kill them? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "1_optimize_unified" || true
        sleep 2
    else
        exit 1
    fi
fi

# Save study name
echo "$STUDY" > .current_study

# Start training
echo ""
echo "🚀 Starting GPU-optimized sequential training..."
echo "   Log: logs/gpu_training.log"
echo ""

mkdir -p logs

nohup python -u scripts/automation/sequential_trial_scheduler.py \
    --study "$STUDY" \
    --n-trials "$N_TRIALS" \
    --gpu "$GPU" \
    --n-envs "$N_ENVS" \
    --data-dir data/1h_1680 \
    --timeframe 1h \
    > logs/gpu_training.log 2>&1 &

PID=$!
echo "✅ Training started (PID: $PID)"
echo ""
echo "Monitor:"
echo "  tail -f logs/gpu_training.log"
echo "  watch -n 1 rocm-smi  # Should see GPU at 90%+, VRAM 8-12GB"
echo ""
echo "Dashboard:"
echo "  python scripts/automation/trial_dashboard.py"
echo ""
