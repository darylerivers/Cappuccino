#!/bin/bash
# Training script for Alpaca-compatible paper trading model
# 300 trials in parallel (3 processes x 100 trials each)

STUDY_NAME="cappuccino_alpaca"
N_TRIALS="${1:-100}"  # 100 trials per process
N_PARALLEL="${2:-3}"   # 3 parallel processes
TICKERS="BTC ETH LTC BCH LINK UNI AAVE"  # Alpaca-supported cryptos

echo "=========================================="
echo "Training Alpaca Paper Trading Model"
echo "=========================================="
echo "Study: $STUDY_NAME"
echo "Trials per process: $N_TRIALS"
echo "Parallel processes: $N_PARALLEL"
echo "Tickers: $TICKERS"
echo "Total trials: $((N_TRIALS * N_PARALLEL))"
echo "GPU: 0"
echo "Using Trial #13 hyperparameter ranges"
echo "=========================================="

# Kill function for cleanup
cleanup() {
    echo "Stopping all processes..."
    pkill -P $$
    exit 0
}

trap cleanup SIGINT SIGTERM

# Create logs directory
mkdir -p logs/alpaca_training

# Launch parallel processes
for i in $(seq 1 $N_PARALLEL); do
    echo "Launching training process $i..."
    python -u 1_optimize_unified.py \
        --n-trials $N_TRIALS \
        --gpu 0 \
        --use-best-ranges \
        --study-name $STUDY_NAME \
        --tickers $TICKERS \
        2>&1 | sed "s/^/[P$i] /" > logs/alpaca_training/process_$i.log &

    # Stagger launches to avoid database conflicts
    sleep 5
done

echo ""
echo "All $N_PARALLEL processes launched!"
echo "Monitor with: tail -f logs/alpaca_training/process_*.log"
echo "Or use: python monitor.py --study-name $STUDY_NAME"
echo ""
echo "Press Ctrl+C to stop all processes"

# Wait for all background jobs
wait
