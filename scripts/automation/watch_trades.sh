#!/bin/bash
# Watch for actual trading activity

echo "=== WAITING FOR 21:00 UTC BAR TO COMPLETE ==="
date -u
echo ""

while true; do
    clear
    echo "=========================================="
    echo "TRADE ACTIVITY MONITOR"
    echo "Current time: $(date -u +%H:%M:%S)"
    echo "=========================================="
    echo ""

    # Check for portfolio changes (trades executed)
    for trial in 0 1 2 3; do
        echo "Trial $trial:"

        # Get last 3 trading decisions
        grep "^\[2026.*cash=" logs/paper_trading_trial${trial}.log 2>/dev/null | tail -3 | while read line; do
            timestamp=$(echo "$line" | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}')
            cash=$(echo "$line" | grep -oP 'cash=\K[0-9.]+')
            total=$(echo "$line" | grep -oP 'total=\K[0-9.]+')

            if [ "$cash" != "1000.00" ] || [ "$total" != "1000.00" ]; then
                echo "  ✅ $timestamp - TRADE! cash=\$$cash total=\$$total"
            else
                echo "  ⏳ $timestamp - No change (cash=\$$cash total=\$$total)"
            fi
        done
        echo ""
    done

    sleep 10
done
