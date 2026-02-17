#!/bin/bash
# Live monitoring of workers and trial progress

echo "Starting live monitoring... (Ctrl+C to stop)"
echo ""

INTERVAL=30  # Check every 30 seconds
BASELINE_MEM=$(ps aux | grep "1_optimize" | grep -v grep | awk '{sum+=$6} END {print sum/1024/1024}')
START_TIME=$(date +%s)

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              LIVE WORKER MONITORING                            ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    # Time elapsed
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    echo "⏱️  Monitoring Duration: ${ELAPSED}s"
    echo ""

    # Worker status
    echo "🔧 Workers:"
    ps aux | grep "1_optimize" | grep -v grep | awk '{printf "  PID %s: %.2f GB (%.0f%% CPU)\n", $2, $6/1024/1024, $3}'
    echo ""

    # Total memory
    CURRENT_MEM=$(ps aux | grep "1_optimize" | grep -v grep | awk '{sum+=$6} END {print sum/1024/1024}')
    MEM_GROWTH=$(echo "$CURRENT_MEM - $BASELINE_MEM" | bc)
    GROWTH_RATE=$(echo "scale=4; $MEM_GROWTH / ($ELAPSED / 60)" | bc)
    echo "💾 Memory:"
    echo "  Total: ${CURRENT_MEM} GB"
    echo "  Growth: ${MEM_GROWTH} GB"
    echo "  Rate: ${GROWTH_RATE} GB/min"
    echo ""

    # Recent trials
    echo "📊 Recent Trials (last 5 minutes):"
    sqlite3 databases/optuna_cappuccino.db "
    SELECT
        t.number,
        t.state,
        substr(t.datetime_complete, 12, 8) as time
    FROM trials t
    WHERE t.study_id = 37
      AND t.datetime_complete >= datetime('now', '-5 minutes')
    ORDER BY t.trial_id DESC
    LIMIT 10
    " 2>/dev/null | awk -F'|' '{printf "  Trial #%-4s %8s at %s\n", $1, $2, $3}'

    echo ""
    echo "Next update in ${INTERVAL}s... (Ctrl+C to stop)"
    sleep $INTERVAL
done
