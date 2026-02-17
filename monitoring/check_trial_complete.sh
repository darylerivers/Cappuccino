#!/bin/bash
while true; do
    COMPLETE=$(sqlite3 /tmp/optuna_working.db "SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'" 2>/dev/null || echo "0")
    echo "[$(date '+%H:%M:%S')] Complete trials: $COMPLETE"
    if [ "$COMPLETE" -gt "0" ]; then
        echo "FIRST TRIAL COMPLETE!"
        sqlite3 /tmp/optuna_working.db "SELECT number, value, datetime_complete FROM trials WHERE state = 'COMPLETE' ORDER BY number DESC LIMIT 1" 2>/dev/null | awk -F'|' '{printf "Trial %s: score %.6f (completed %s)\n", $1, $2, $3}'
        break
    fi
    sleep 30
done
