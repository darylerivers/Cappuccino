#!/bin/bash
# Start Training Workers - VISIBLE MODE
# Runs 3 training workers in the current terminal window

cd /opt/user-data/experiment/cappuccino

source .env.training

echo "=========================================="
echo "CAPPUCCINO TRAINING WORKERS - VISIBLE"
echo "=========================================="
echo "Study: $ACTIVE_STUDY_NAME"
echo "Workers: 3"
echo "GPU: 0"
echo ""
echo "Starting workers..."
echo "=========================================="
echo ""

# Start 3 workers in background, but logs will stream to this terminal
for i in {1..3}; do
  echo "Starting worker $i..."
  python -u 1_optimize_unified.py --n-trials 1000 --gpu 0 \
    --study-name "$ACTIVE_STUDY_NAME" > logs/worker_$i.log 2>&1 &
  WORKER_PIDS[$i]=$!
  echo "  Worker $i: PID ${WORKER_PIDS[$i]}"
  sleep 3
done

echo ""
echo "=========================================="
echo "All workers started!"
echo "=========================================="
echo ""
echo "Worker PIDs: ${WORKER_PIDS[@]}"
echo ""
echo "Monitoring worker 1 log (Ctrl+C to exit, workers continue running)..."
echo "=========================================="
echo ""

# Tail the first worker's log so user can see progress
tail -f logs/worker_1.log
