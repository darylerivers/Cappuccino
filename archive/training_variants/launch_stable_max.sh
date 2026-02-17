#!/bin/bash
# Launch stable maximum utilization training
# Reduced preallocation + fragmentation fixes for stability

echo "Launching STABLE MAX training..."

# Main study with 4.5GB preallocation
nohup python 1_optimize_unified.py --n-trials 2000 --gpu 0 \
  --study-name stable_max_main \
  --storage sqlite:///databases/optuna_stable_max.db \
  --data-dir data/2year_fresh_20260112 \
  > logs/training_stable_main.log 2>&1 &
MAIN_PID=$!
echo "Main study launched: PID $MAIN_PID"

sleep 10

# Parallel study 2 (no prealloc)
SKIP_VRAM_PREALLOC=1 nohup python 1_optimize_unified.py --n-trials 500 --gpu 0 \
  --study-name stable_max_parallel_2 \
  --storage sqlite:///databases/optuna_stable_max_2.db \
  --data-dir data/2year_fresh_20260112 \
  > logs/training_stable_2.log 2>&1 &
echo "Parallel 2 launched: PID $!"

sleep 10

# Parallel study 3 (no prealloc)
SKIP_VRAM_PREALLOC=1 nohup python 1_optimize_unified.py --n-trials 500 --gpu 0 \
  --study-name stable_max_parallel_3 \
  --storage sqlite:///databases/optuna_stable_max_3.db \
  --data-dir data/2year_fresh_20260112 \
  > logs/training_stable_3.log 2>&1 &
echo "Parallel 3 launched: PID $!"

echo ""
echo "STABLE MAX mode launched!"
echo "- Reduced VRAM preallocation (4.5GB instead of 6.5GB)"
echo "- Memory fragmentation fixes enabled"
echo "- Batch sizes: 98k-163k (down from 262k for stability)"
echo "- 3 parallel studies total"
echo ""
echo "Monitor: watch -n 2 './summary_status.sh'"
