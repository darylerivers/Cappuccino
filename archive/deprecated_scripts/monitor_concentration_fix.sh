#!/bin/bash
# Monitor logs for concentration limit enforcement messages

echo "=========================================="
echo "Monitoring Concentration Limit Fix"
echo "=========================================="
echo ""
echo "Watching both traders for concentration limit messages..."
echo "Press Ctrl+C to stop"
echo ""
echo "✅ = Fix working (capping trades)"
echo "⚠️  = No messages yet (waiting for buy attempts)"
echo ""

# Use tail -f to follow both logs and filter for concentration messages
tail -f logs/ensemble_fixed.log logs/single_fixed.log 2>/dev/null | grep --line-buffered -E "Concentration limit|🛡️|New bars detected|cash=|total=" | while read line; do
    # Add timestamp
    timestamp=$(date "+%H:%M:%S")

    # Color code the output
    if [[ $line == *"Concentration limit"* ]]; then
        echo "[$timestamp] ✅ $line"
    elif [[ $line == *"🛡️"* ]]; then
        echo "[$timestamp] ✅ $line"
    else
        echo "[$timestamp] $line"
    fi
done
