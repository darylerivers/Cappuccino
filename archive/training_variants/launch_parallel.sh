#!/bin/bash
# Launch parallel training studies to max out GPU and RAM

echo "Launching parallel training studies..."

# Study 2
nohup python 1_optimize_unified.py --n-trials 500 --gpu 0 \
  --study-name maxvram_parallel_2 \
  --storage sqlite:///databases/optuna_maxvram_parallel_2.db \
  --data-dir data/2year_fresh_20260112 \
  > logs/training_parallel_2.log 2>&1 &
echo "Study 2 launched: PID $!"

sleep 5

# Study 3
nohup python 1_optimize_unified.py --n-trials 500 --gpu 0 \
  --study-name maxvram_parallel_3 \
  --storage sqlite:///databases/optuna_maxvram_parallel_3.db \
  --data-dir data/2year_fresh_20260112 \
  > logs/training_parallel_3.log 2>&1 &
echo "Study 3 launched: PID $!"

sleep 5

# Study 4
nohup python 1_optimize_unified.py --n-trials 500 --gpu 0 \
  --study-name maxvram_parallel_4 \
  --storage sqlite:///databases/optuna_maxvram_parallel_4.db \
  --data-dir data/2year_fresh_20260112 \
  > logs/training_parallel_4.log 2>&1 &
echo "Study 4 launched: PID $!"

echo ""
echo "All parallel studies launched!"
echo ""
echo "Monitor with:"
echo "  watch -n 2 './summary_status.sh'"
echo ""
echo "View logs:"
echo "  tail -f logs/training_parallel_*.log"
echo ""
echo "Kill all training:"
echo "  pkill -f 1_optimize_unified"
