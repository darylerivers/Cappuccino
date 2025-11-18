#!/bin/bash
# Check status of autonomous AI advisor

echo "=========================================="
echo "Autonomous AI Advisor Status"
echo "=========================================="
echo ""

# Check PID file
PID_FILE="logs/autonomous_advisor.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        echo "✓ Advisor is RUNNING (PID: $PID)"

        # Show process info
        ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd | tail -1
        echo ""

        # Check state file
        if [ -f "analysis_reports/advisor_state.json" ]; then
            echo "State:"
            cat analysis_reports/advisor_state.json | jq -r '
                "  Study: \(.study_name)",
                "  Last trial count: \(.last_trial_count)",
                "  Analysis count: \(.analysis_count)",
                "  Tested configs: \(.tested_configs | length)",
                "  Best discovered: \(.best_discovered_value)"
            ' 2>/dev/null || cat analysis_reports/advisor_state.json
            echo ""
        fi

        # Show recent log entries
        echo "Recent activity (last 10 lines):"
        tail -10 logs/autonomous_advisor.log 2>/dev/null || echo "  No log file yet"

    else
        echo "✗ Advisor is NOT RUNNING (stale PID file)"
        rm "$PID_FILE"
    fi
else
    # Check if process is running without PID file
    PIDS=$(pgrep -f "ollama_autonomous_advisor.py")
    if [ -n "$PIDS" ]; then
        echo "⚠ Advisor is RUNNING but no PID file found"
        echo "  PIDs: $PIDS"
    else
        echo "✗ Advisor is NOT RUNNING"
    fi
fi

echo ""
echo "=========================================="
