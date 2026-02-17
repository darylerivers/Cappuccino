#!/bin/bash
echo "=== LUDICROUS MODE STATUS ==="
echo ""
echo "GPU STATUS:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits | awk -F, '{printf "  Compute: %s%%  |  Memory BW: %s%%  |  VRAM: %sMB / %sMB  |  Power: %sW  |  Temp: %s°C\n", $1, $2, $3, $4, $5, $6}'
echo ""
echo "RAM STATUS:"
free -h | grep Mem | awk '{printf "  Used: %s / %s  |  Available: %s\n", $3, $2, $7}'
echo ""
echo "TRAINING PROCESSES:"
ps aux | grep "1_optimize_unified" | grep -v grep | awk '{printf "  PID %s: CPU %s%%  RAM %sMB\n", $2, $3, int($6/1024)}'
echo ""
echo "CURRENT TRIAL:"
tail -100 logs/training_maxvram.log | grep -A 15 "Trial #" | tail -17
echo ""
echo "=========================="
