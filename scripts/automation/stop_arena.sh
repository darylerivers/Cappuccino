#!/bin/bash
# Stop the Model Arena runner

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="arena_state/arena_runner.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping arena runner (PID: $PID)..."
        kill $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            echo "Force killing..."
            kill -9 $PID
        fi
        echo "Arena runner stopped"
    else
        echo "Arena runner not running (stale PID file)"
    fi
    rm -f "$PID_FILE"
else
    # Try to find by process name
    PIDS=$(pgrep -f "arena_runner.py")
    if [ -n "$PIDS" ]; then
        echo "Stopping arena runner processes: $PIDS"
        kill $PIDS
        sleep 2
        echo "Done"
    else
        echo "Arena runner not running"
    fi
fi
