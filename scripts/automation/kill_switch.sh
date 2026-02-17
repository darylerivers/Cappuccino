#!/bin/bash
# Kill Switch for Training - Monitors GPU/RAM and kills if exceeded
# Usage: ./kill_switch.sh [gpu|ram|both] [threshold_percent]
#   Example: ./kill_switch.sh gpu 95      - Kill if GPU VRAM > 95%
#   Example: ./kill_switch.sh ram 90      - Kill if RAM > 90%
#   Example: ./kill_switch.sh both 95 85  - Kill if GPU>95% OR RAM>85%

MODE=${1:-both}
GPU_THRESHOLD=${2:-95}
RAM_THRESHOLD=${3:-90}

LOG_FILE="logs/kill_switch.log"
CHECK_INTERVAL=10  # Check every 10 seconds

echo "[$(date)] Kill switch started: Mode=$MODE, GPU_Threshold=${GPU_THRESHOLD}%, RAM_Threshold=${RAM_THRESHOLD}%" | tee -a "$LOG_FILE"

kill_training() {
    local reason=$1
    echo "[$(date)] KILL SWITCH ACTIVATED: $reason" | tee -a "$LOG_FILE"

    # Find all training processes
    PIDS=$(ps aux | grep "1_optimize_unified\|train_maxvram" | grep -v grep | awk '{print $2}')

    if [ -z "$PIDS" ]; then
        echo "[$(date)] No training processes found" | tee -a "$LOG_FILE"
        exit 0
    fi

    echo "[$(date)] Killing training processes: $PIDS" | tee -a "$LOG_FILE"

    # Kill gracefully first
    for pid in $PIDS; do
        kill -TERM $pid 2>/dev/null && echo "  Sent TERM to $pid" | tee -a "$LOG_FILE"
    done

    sleep 5

    # Force kill if still running
    for pid in $PIDS; do
        if ps -p $pid > /dev/null 2>&1; then
            kill -9 $pid 2>/dev/null && echo "  Force killed $pid" | tee -a "$LOG_FILE"
        fi
    done

    # Clear GPU memory
    nvidia-smi --gpu-reset 2>/dev/null || true

    echo "[$(date)] Kill switch completed" | tee -a "$LOG_FILE"
    exit 0
}

while true; do
    # Check GPU VRAM usage
    if [ "$MODE" = "gpu" ] || [ "$MODE" = "both" ]; then
        GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        GPU_PERCENT=$((GPU_USED * 100 / GPU_TOTAL))

        echo "[$(date)] GPU VRAM: ${GPU_PERCENT}% (${GPU_USED}MB / ${GPU_TOTAL}MB)" >> "$LOG_FILE"

        if [ $GPU_PERCENT -ge $GPU_THRESHOLD ]; then
            kill_training "GPU VRAM usage ${GPU_PERCENT}% >= ${GPU_THRESHOLD}%"
        fi
    fi

    # Check RAM usage
    if [ "$MODE" = "ram" ] || [ "$MODE" = "both" ]; then
        RAM_PERCENT=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100}')
        RAM_USED=$(free -h | grep Mem | awk '{print $3}')
        RAM_TOTAL=$(free -h | grep Mem | awk '{print $2}')

        echo "[$(date)] RAM: ${RAM_PERCENT}% (${RAM_USED} / ${RAM_TOTAL})" >> "$LOG_FILE"

        if [ $RAM_PERCENT -ge $RAM_THRESHOLD ]; then
            kill_training "RAM usage ${RAM_PERCENT}% >= ${RAM_THRESHOLD}%"
        fi
    fi

    sleep $CHECK_INTERVAL
done
