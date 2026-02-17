#!/usr/bin/env bash
#
# Train models on short timeframes (5m, 15m, 30m) for tactical trading
#
# Usage:
#   ./train_multi_timeframe.sh --timeframe 5m --workers 24 --trials 500
#   ./train_multi_timeframe.sh --timeframe 15m --workers 12 --trials 500

set -euo pipefail

# Default values
TIMEFRAME="5m"
WORKERS=24
TRIALS=500
GPU_START=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeframe)
            TIMEFRAME="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --gpu-start)
            GPU_START="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --timeframe <5m|15m|30m> --workers <N> --trials <N>"
            exit 1
            ;;
    esac
done

# Validate timeframe
if [[ ! "$TIMEFRAME" =~ ^(5m|15m|30m)$ ]]; then
    echo "Error: Invalid timeframe '$TIMEFRAME'. Use 5m, 15m, or 30m"
    exit 1
fi

# Configuration based on timeframe
case $TIMEFRAME in
    5m)
        DATA_FILE="data/crypto_5m_3mo.pkl"
        STUDY_NAME="cappuccino_5m_$(date +%Y%m%d_%H%M)"
        RECOMMENDED_WORKERS=24
        CHUNK_SIZE=512
        ;;
    15m)
        DATA_FILE="data/crypto_15m_6mo.pkl"
        STUDY_NAME="cappuccino_15m_$(date +%Y%m%d_%H%M)"
        RECOMMENDED_WORKERS=12
        CHUNK_SIZE=256
        ;;
    30m)
        DATA_FILE="data/crypto_30m_9mo.pkl"
        STUDY_NAME="cappuccino_30m_$(date +%Y%m%d_%H%M)"
        RECOMMENDED_WORKERS=6
        CHUNK_SIZE=256
        ;;
esac

echo "=============================================================================="
echo "MULTI-TIMEFRAME TRAINING"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Timeframe:    $TIMEFRAME"
echo "  Study:        $STUDY_NAME"
echo "  Data:         $DATA_FILE"
echo "  Workers:      $WORKERS"
echo "  Trials:       $TRIALS"
echo "  GPU Start:    $GPU_START"
echo "  Chunk Size:   $CHUNK_SIZE"
echo ""

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    echo ""
    echo "Run data preparation first:"
    echo "  python prepare_multi_timeframe_data.py --timeframe $TIMEFRAME"
    exit 1
fi

echo "✓ Data file found"
echo ""

# Warn if workers differs from recommended
if [ "$WORKERS" -ne "$RECOMMENDED_WORKERS" ]; then
    echo "⚠️  Note: Recommended workers for $TIMEFRAME is $RECOMMENDED_WORKERS"
    echo "   You specified $WORKERS workers."
    echo ""
fi

# Calculate GPU assignments
echo "GPU Assignment:"
GPU_ID=$GPU_START
for ((i=1; i<=WORKERS; i++)); do
    WORKER_GPU=$((GPU_ID % 1))  # Cycle through available GPUs
    echo "  Worker $i: GPU $WORKER_GPU"
    ((GPU_ID++))
done
echo ""

# Confirm before starting
echo "Press ENTER to start training, or Ctrl+C to cancel..."
read

echo ""
echo "Starting training..."
echo "=============================================================================="
echo ""

# Launch workers
PIDS=()
GPU_ID=$GPU_START

for ((i=1; i<=WORKERS; i++)); do
    WORKER_GPU=$((GPU_ID % 1))

    echo "Starting worker $i on GPU $WORKER_GPU..."

    python 1_optimize_unified.py \
        --study-name "$STUDY_NAME" \
        --trials-per-worker "$TRIALS" \
        --data-file "$DATA_FILE" \
        --timeframe "$TIMEFRAME" \
        --chunk-size "$CHUNK_SIZE" \
        --gpu-id "$WORKER_GPU" \
        > "logs/worker_${i}_${TIMEFRAME}.log" 2>&1 &

    PIDS+=($!)
    ((GPU_ID++))

    sleep 2  # Stagger starts
done

echo ""
echo "=============================================================================="
echo "All workers started!"
echo "=============================================================================="
echo ""
echo "Worker PIDs:"
for ((i=0; i<${#PIDS[@]}; i++)); do
    echo "  Worker $((i+1)): ${PIDS[i]}"
done
echo ""
echo "Logs:"
echo "  logs/worker_*_${TIMEFRAME}.log"
echo ""
echo "Monitor training:"
echo "  python dashboard_training_detailed.py --study $STUDY_NAME"
echo "  watch -n 5 'pgrep -f 1_optimize_unified.py | wc -l'"
echo ""
echo "Stop training:"
echo "  pkill -f \"study-name $STUDY_NAME\""
echo ""
echo "Study name: $STUDY_NAME"
echo ""

# Wait for completion
echo "Waiting for workers to complete..."
echo "(Press Ctrl+C to stop monitoring, workers will continue in background)"
echo ""

wait_start=$(date +%s)

while true; do
    running_count=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((running_count++))
        fi
    done

    elapsed=$(($(date +%s) - wait_start))
    hours=$((elapsed / 3600))
    minutes=$(((elapsed % 3600) / 60))

    if [ $running_count -eq 0 ]; then
        echo ""
        echo "✓ All workers completed!"
        echo "  Time: ${hours}h ${minutes}m"
        break
    fi

    echo -ne "\r  Workers running: $running_count/$WORKERS | Elapsed: ${hours}h ${minutes}m"
    sleep 10
done

echo ""
echo "=============================================================================="
echo "TRAINING COMPLETE"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Check results: python analyze_training_results.py --study $STUDY_NAME"
echo "  2. Create ensemble: python create_multi_timeframe_ensemble.py"
echo "  3. Test: python backtest_multi_timeframe.py"
echo ""
