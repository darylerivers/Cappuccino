#!/bin/bash
# Training script for Alpaca model with 2 parallel workers and enhanced hyperparameters
# Optimized for GPU efficiency with more fine-tuning parameters

STUDY_NAME="cappuccino_alpaca_v2"
N_TRIALS="${1:-100}"  # 100 trials per process
N_PARALLEL="${2:-2}"   # 2 parallel processes for optimal GPU utilization
TICKERS="AAVE AVAX BTC LINK ETH LTC UNI"  # Alpaca-supported cryptos

echo "=========================================="
echo "Training Alpaca Model v2"
echo "=========================================="
echo "Study: $STUDY_NAME"
echo "Trials per process: $N_TRIALS"
echo "Parallel processes: $N_PARALLEL"
echo "Tickers: $TICKERS"
echo "Total trials: $((N_TRIALS * N_PARALLEL))"
echo "GPU: 0"
echo "Using enhanced Trial #141 hyperparameter ranges"
echo "New features: PPO epochs, KL target, Adam epsilon,"
echo "              drawdown penalty, volatility penalty"
echo "=========================================="

# Kill function for cleanup
cleanup() {
    echo "Stopping all processes..."
    pkill -P $$
    exit 0
}

trap cleanup SIGINT SIGTERM

# Create logs directory
mkdir -p logs/alpaca_training_v2

# Launch parallel processes
for i in $(seq 1 $N_PARALLEL); do
    echo "Launching training process $i..."
    python -u 1_optimize_unified.py \
        --n-trials $N_TRIALS \
        --gpu 0 \
        --use-best-ranges \
        --study-name $STUDY_NAME \
        --tickers $TICKERS \
        2>&1 | sed "s/^/[P$i] /" > logs/alpaca_training_v2/process_$i.log &

    # Stagger launches to avoid database conflicts
    sleep 5
done

echo ""
echo "All $N_PARALLEL processes launched!"
echo "Monitor with: tail -f logs/alpaca_training_v2/process_*.log"
echo "Or use: python monitor.py --study-name $STUDY_NAME"
echo ""
echo "Press Ctrl+C to stop all processes"

# Wait for all background jobs
wait
