#!/bin/bash
# Launch parallel training studies WITHOUT VRAM preallocation
# This allows multiple processes to share GPU

echo "Launching parallel training studies (no VRAM preallocation)..."

# Study 3 - using env var to skip preallocation
SKIP_VRAM_PREALLOC=1 nohup python 1_optimize_unified.py --n-trials 500 --gpu 0 \
  --study-name maxvram_parallel_3 \
  --storage sqlite:///databases/optuna_maxvram_parallel_3.db \
  --data-dir data/2year_fresh_20260112 \
  > logs/training_parallel_3.log 2>&1 &
echo "Study 3 launched: PID $!"

sleep 5

# Study 4
SKIP_VRAM_PREALLOC=1 nohup python 1_optimize_unified.py --n-trials 500 --gpu 0 \
  --study-name maxvram_parallel_4 \
  --storage sqlite:///databases/optuna_maxvram_parallel_4.db \
  --data-dir data/2year_fresh_20260112 \
  > logs/training_parallel_4.log 2>&1 &
echo "Study 4 launched: PID $!"

echo ""
echo "Parallel studies launched!"
echo ""
echo "Current status:"
ps aux | grep "1_optimize_unified" | grep -v grep | wc -l
echo "training processes running"
