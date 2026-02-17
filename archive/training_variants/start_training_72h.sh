#!/bin/bash
# Start training with 72-hour time limit instead of trial count

set -e

STUDY_NAME=${1:-"cappuccino_week_$(date +%Y%m%d)"}
NUM_WORKERS=${2:-3}
MAX_HOURS=72
GPU_ID=0

echo "========================================"
echo "Starting Time-Limited Training"
echo "========================================"
echo "Study: $STUDY_NAME"
echo "Workers: $NUM_WORKERS"
echo "Time limit: $MAX_HOURS hours"
echo ""

# Calculate timeout in seconds
TIMEOUT_SECONDS=$((MAX_HOURS * 3600))

for i in $(seq 1 $NUM_WORKERS); do
    echo "Starting worker $i (timeout: ${MAX_HOURS}h)..."
    
    # Use timeout command to limit execution time
    nohup timeout ${TIMEOUT_SECONDS}s python -u 1_optimize_unified.py \
        --n-trials 99999 \
        --gpu $GPU_ID \
        --study-name "$STUDY_NAME" \
        > logs/training_worker_${i}_72h_${STUDY_NAME}.log 2>&1 &
    
    echo "  PID: $!"
    sleep 1
done

echo ""
echo "✓ All workers started with 72-hour timeout"
echo "  They will auto-stop after 72 hours"
echo "  Monitor: ps aux | grep optimize_unified"
echo ""
