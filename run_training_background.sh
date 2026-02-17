#!/bin/bash
# Run training in background with signal debugging
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
cd /opt/user-data/experiment/cappuccino

# Trap signals for debugging
trap 'echo "[$(date)] Received SIGHUP" >> logs/signal_debug.log' HUP
trap 'echo "[$(date)] Received SIGTERM" >> logs/signal_debug.log; exit 1' TERM
trap 'echo "[$(date)] Received SIGINT" >> logs/signal_debug.log; exit 1' INT

echo "[$(date)] Starting training PID $$" >> logs/signal_debug.log

python -u scripts/training/1_optimize_unified.py \
    --n-trials 500 \
    --gpu 0 \
    --study-name "cappuccino_ultralow_ram" \
    --timeframe 1h \
    --data-dir data/1h_fresh_6mo \
    >> logs/worker_1.log 2>&1

EXIT_CODE=$?
echo "[$(date)] Training exited with code $EXIT_CODE" >> logs/signal_debug.log
