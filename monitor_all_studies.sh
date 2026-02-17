#!/usr/bin/env bash
#
# Monitor all active training studies
#

echo "========================================================================"
echo "  CAPPUCCINO TRAINING MONITOR"
echo "========================================================================"
echo ""

# Get active studies
ACTIVE_STUDIES=$(ps aux | grep "[1]_optimize_unified" | grep -o "study-name [^ ]*" | awk '{print $2}' | sort -u)

if [ -z "$ACTIVE_STUDIES" ]; then
    echo "❌ No active training workers found"
    exit 1
fi

echo "Active studies:"
echo "$ACTIVE_STUDIES" | nl
echo ""

# Get active log files
ACTIVE_LOGS=$(ls logs/worker_safe_*.log 2>/dev/null)

echo "Monitoring logs:"
echo "$ACTIVE_LOGS" | nl
echo ""

echo "Press Ctrl+C to exit"
echo "========================================================================"
echo ""

# Tail all active logs
tail -f $ACTIVE_LOGS
