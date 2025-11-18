#!/bin/bash
# Stop the autonomous AI advisor

PID_FILE="logs/autonomous_advisor.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping autonomous advisor (PID: $PID)..."

    if kill -0 $PID 2>/dev/null; then
        kill -TERM $PID
        echo "Sent shutdown signal. Waiting for graceful shutdown..."

        # Wait up to 30 seconds for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 $PID 2>/dev/null; then
                echo "Advisor stopped successfully"
                rm "$PID_FILE"
                exit 0
            fi
            sleep 1
        done

        # Force kill if still running
        echo "Forcing shutdown..."
        kill -9 $PID 2>/dev/null
        rm "$PID_FILE"
        echo "Advisor stopped (forced)"
    else
        echo "Process not running"
        rm "$PID_FILE"
    fi
else
    echo "No PID file found. Checking for running processes..."
    PIDS=$(pgrep -f "ollama_autonomous_advisor.py")
    if [ -n "$PIDS" ]; then
        echo "Found running advisor processes: $PIDS"
        kill -TERM $PIDS
        echo "Sent shutdown signals"
    else
        echo "No running advisor processes found"
    fi
fi
