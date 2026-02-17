#!/bin/bash
# Parallel Training Script - Runs multiple trials simultaneously on same GPU

STUDY_NAME="${1:-cappuccino_exploitation}"
N_TRIALS="${2:-100}"
N_PARALLEL="${3:-3}"  # Number of parallel processes

echo "Starting $N_PARALLEL parallel optimization processes"
echo "Study: $STUDY_NAME"
echo "Trials per process: $N_TRIALS"
echo "GPU: 0"

# Kill function for cleanup
cleanup() {
    echo "Stopping all processes..."
    pkill -P $$
    exit 0
}

trap cleanup SIGINT SIGTERM

# Launch parallel processes
for i in $(seq 1 $N_PARALLEL); do
    echo "Launching process $i..."
    python -u 1_optimize_unified.py \
        --n-trials $N_TRIALS \
        --gpu 0 \
        --use-best-ranges \
        --study-name $STUDY_NAME \
        2>&1 | sed "s/^/[P$i] /" &
    
    # Stagger launches to avoid database conflicts
    sleep 5
done

echo "All $N_PARALLEL processes launched!"
echo "Press Ctrl+C to stop all processes"

# Wait for all background jobs
wait
