#!/bin/bash
# Persistent overnight training - auto-restarts if process dies
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
export HSA_ENABLE_SDMA=0
cd /opt/user-data/experiment/cappuccino

STUDY_NAME="cappuccino_ultralow_ram"
DATA_DIR="data/1h_fresh_6mo"
LOG="logs/worker_1.log"
META="logs/overnight_meta.log"

echo "[$(date)] Overnight training started" > "$META"

while true; do
    echo "[$(date)] Starting training attempt..." >> "$META"

    python -u scripts/training/1_optimize_unified.py \
        --n-trials 500 \
        --gpu 0 \
        --study-name "$STUDY_NAME" \
        --timeframe 1h \
        --data-dir "$DATA_DIR" \
        >> "$LOG" 2>&1

    EXIT_CODE=$?
    echo "[$(date)] Training exited with code $EXIT_CODE" >> "$META"

    # If exit code 0, training completed all trials
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully!" >> "$META"
        break
    fi

    echo "[$(date)] Restarting in 10 seconds..." >> "$META"
    sleep 10
done
