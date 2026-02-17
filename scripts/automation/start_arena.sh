#!/bin/bash
# Start the Model Arena runner as a background service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment (only simple key=value pairs, skip multi-line keys)
if [ -f .env ]; then
    set -a
    source .env 2>/dev/null || true
    set +a
fi

LOG_FILE="logs/arena_runner_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs arena_state

echo "Starting Model Arena Runner..."
echo "Log file: $LOG_FILE"

# Check if already running
if pgrep -f "arena_runner.py" > /dev/null; then
    echo "Arena runner is already running!"
    exit 1
fi

# Start arena runner
nohup python -u arena_runner.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "Arena runner started with PID: $PID"
echo $PID > arena_state/arena_runner.pid

# Wait a moment and check if it's still running
sleep 2
if kill -0 $PID 2>/dev/null; then
    echo "Arena runner is running successfully"
    echo ""
    echo "To view logs: tail -f $LOG_FILE"
    echo "To stop: ./stop_arena.sh"
else
    echo "ERROR: Arena runner failed to start!"
    echo "Check logs: cat $LOG_FILE"
    exit 1
fi
