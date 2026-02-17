#!/bin/bash
# Launch training worker with proper process isolation
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
cd /opt/user-data/experiment/cappuccino

STUDY_NAME="${1:-cappuccino_ultralow_ram}"
DATA_DIR="${2:-data/1h_fresh_6mo}"
LOG_FILE="logs/worker_1.log"

echo "Launching training: study=$STUDY_NAME data=$DATA_DIR"
exec python -u scripts/training/1_optimize_unified.py \
    --n-trials 500 \
    --gpu 0 \
    --study-name "$STUDY_NAME" \
    --timeframe 1h \
    --data-dir "$DATA_DIR" \
    > "$LOG_FILE" 2>&1
