#!/bin/bash
# Show Model Arena status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "MODEL ARENA STATUS"
echo "=========================================="

# Check if running
if pgrep -f "arena_runner.py" > /dev/null; then
    PID=$(pgrep -f "arena_runner.py")
    echo "Status: RUNNING (PID: $PID)"
else
    echo "Status: STOPPED"
fi
echo ""

# Show heartbeat
HEARTBEAT="arena_state/.heartbeat"
if [ -f "$HEARTBEAT" ]; then
    echo "Last heartbeat:"
    cat "$HEARTBEAT" | python3 -m json.tool 2>/dev/null || cat "$HEARTBEAT"
    echo ""
fi

# Show leaderboard
LEADERBOARD="arena_state/leaderboard.txt"
if [ -f "$LEADERBOARD" ]; then
    echo ""
    cat "$LEADERBOARD"
else
    echo "No leaderboard available yet"
fi

# Show promotion candidate
PROMOTION="arena_state/promotion_candidate.json"
if [ -f "$PROMOTION" ]; then
    echo ""
    echo "=========================================="
    echo "PROMOTION CANDIDATE"
    echo "=========================================="
    cat "$PROMOTION" | python3 -m json.tool 2>/dev/null || cat "$PROMOTION"
fi
