#!/bin/bash
# Quick script to scale training to FULL mode (10 workers)
# Use this when GPU is free and you want maximum training speed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_LOG_DIR="$SCRIPT_DIR/logs"

# Load configuration from .env.training
if [ -f "$SCRIPT_DIR/.env.training" ]; then
    source "$SCRIPT_DIR/.env.training"
    STUDY_NAME="$ACTIVE_STUDY_NAME"
    GPU_ID="${GPU_ID:-0}"
    N_TRIALS="${N_TRIALS:-500}"
else
    # Try to detect from running workers
    STUDY_NAME=$(pgrep -af "1_optimize_unified" | grep -oP '(?<=--study-name )\S+' | head -1)
    GPU_ID=0
    N_TRIALS=500
fi

if [ -z "$STUDY_NAME" ]; then
    echo "Error: No study name found in .env.training or running workers"
    exit 1
fi

echo "Scaling training to FULL mode (10 workers)..."
echo "Study: $STUDY_NAME"
echo "This will use ~7.6 GB VRAM (93% of GPU)"
echo ""

CURRENT_COUNT=$(ps aux | grep "1_optimize_unified" | grep -v grep | wc -l)
TARGET_COUNT=10

if [ $CURRENT_COUNT -ge $TARGET_COUNT ]; then
    echo "Already at or above $TARGET_COUNT workers"
    echo "Current: $CURRENT_COUNT workers"
    exit 0
fi

TO_START=$((TARGET_COUNT - CURRENT_COUNT))

echo "Will start $TO_START new workers (total will be $TARGET_COUNT)"
echo ""

read -p "Continue? (y/n): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Check if 1_optimize_unified.py exists
if [ ! -f "$SCRIPT_DIR/1_optimize_unified.py" ]; then
    echo "Error: 1_optimize_unified.py not found in $SCRIPT_DIR"
    exit 1
fi

# Create logs directory if needed
mkdir -p "$WORKER_LOG_DIR"

# Start new workers with proper arguments
for i in $(seq 1 $TO_START); do
    worker_num=$((CURRENT_COUNT + i))
    log_file="$WORKER_LOG_DIR/worker_${worker_num}_$(date +%Y%m%d_%H%M%S).log"

    nohup python -u 1_optimize_unified.py \
        --n-trials $N_TRIALS \
        --gpu $GPU_ID \
        --study-name "$STUDY_NAME" \
        > "$log_file" 2>&1 &

    new_pid=$!
    echo "  ✓ Started worker #$worker_num (PID: $new_pid)"

    # Small delay to avoid resource spikes
    sleep 0.5
done

sleep 2

echo ""
echo "New status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | \
    awk -F, '{printf "  VRAM: %s / %s (%.0f%%)\n", $1, $2, ($1/$2)*100}'
echo "  Workers: $(ps aux | grep 1_optimize_unified | grep -v grep | wc -l)"
echo ""
echo "✓ Training scaled to FULL mode"
echo "  Training speed: ~60 trials/hour (maximum)"
