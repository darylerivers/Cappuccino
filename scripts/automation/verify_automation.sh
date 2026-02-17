#!/bin/bash
# Quick verification of automation status

echo "=========================================="
echo "Automation Status Check"
echo "=========================================="
echo ""

# 1. Check training progress
STUDY="cappuccino_fresh_20251204_100527"
COMPLETED=$(sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials WHERE study_id = (SELECT study_id FROM studies WHERE study_name = '$STUDY') AND state = 'COMPLETE'")
RUNNING=$(sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials WHERE study_id = (SELECT study_id FROM studies WHERE study_name = '$STUDY') AND state = 'RUNNING'")

echo "Training Progress:"
echo "  Study: $STUDY"
echo "  Completed trials: $COMPLETED"
echo "  Running trials: $RUNNING"
echo ""

# 2. Check ensemble status
if [ -f "train_results/ensemble/ensemble_manifest.json" ]; then
    ENSEMBLE_COUNT=$(jq -r '.model_count' train_results/ensemble/ensemble_manifest.json)
    ENSEMBLE_STUDY=$(jq -r '.study_name' train_results/ensemble/ensemble_manifest.json)
    ENSEMBLE_UPDATED=$(jq -r '.updated' train_results/ensemble/ensemble_manifest.json)

    echo "Ensemble Status:"
    echo "  Models: $ENSEMBLE_COUNT"
    echo "  Study: $ENSEMBLE_STUDY"
    echo "  Last updated: $ENSEMBLE_UPDATED"

    # Check if ensemble matches training study
    if [ "$ENSEMBLE_STUDY" = "$STUDY" ]; then
        echo "  ✅ Ensemble synced with training study"
    else
        echo "  ⚠️  Ensemble using old study (will update once new trials complete)"
    fi
else
    echo "Ensemble Status: Not found"
fi
echo ""

# 3. Check paper trader
PAPER_PID=$(pgrep -f "paper_trader_alpaca_polling.py" | head -1)
if [ -n "$PAPER_PID" ]; then
    PAPER_AGE=$(ps -p $PAPER_PID -o etime= | xargs)
    echo "Paper Trader:"
    echo "  PID: $PAPER_PID"
    echo "  Uptime: $PAPER_AGE"

    # Check last log entry
    LATEST_LOG=$(ls -t logs/paper_trading_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        LAST_POLL=$(tail -1 "$LATEST_LOG" | grep -oP '\[\K[^\]]+' | head -1)
        echo "  Last poll: $LAST_POLL"
    fi
else
    echo "Paper Trader: NOT RUNNING"
fi
echo ""

# 4. Check watchdog recent activity
echo "Recent Watchdog Activity:"
tail -20 logs/watchdog.log | grep -E "Ensemble Updated|Restarted|Alpha" | tail -5 || echo "  No recent events"
echo ""

# 5. Check automation processes
echo "Automation Processes:"
for proc in "system_watchdog.py" "ensemble_auto_updater.py" "auto_model_deployer.py"; do
    PID=$(pgrep -f "$proc" | head -1)
    if [ -n "$PID" ]; then
        echo "  ✅ $proc (PID $PID)"
    else
        echo "  ❌ $proc NOT RUNNING"
    fi
done
echo ""

echo "=========================================="
