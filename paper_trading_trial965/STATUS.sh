#!/bin/bash
# Quick status check

cd "$(dirname "$0")"

echo "======================================================================"
echo "PAPER TRADING STATUS - TRIAL #965"
echo "======================================================================"
echo ""

# Check if running
if [ -f "trading.pid" ]; then
    PID=$(cat trading.pid)
    if ps -p $PID > /dev/null 2>&1; then
        ELAPSED=$(ps -p $PID -o etime= | tr -d ' ')
        echo "✓ Trading is RUNNING"
        echo "  PID: $PID"
        echo "  Runtime: $ELAPSED"
    else
        echo "✗ Trading NOT running (stale PID)"
        rm trading.pid
    fi
else
    echo "✗ Trading NOT running (no PID file)"
fi

echo ""

# Check latest log
if ls logs/fast_*.log 1> /dev/null 2>&1; then
    LATEST_LOG=$(ls -t logs/fast_*.log | head -1)
    echo "📋 Latest Log: $LATEST_LOG"
    echo "   Last 3 lines:"
    tail -3 "$LATEST_LOG" | sed 's/^/   /'
fi

echo ""

# Check CSV
if ls results/performance_*.csv 1> /dev/null 2>&1; then
    LATEST_CSV=$(ls -t results/performance_*.csv | head -1)
    LINES=$(wc -l < "$LATEST_CSV")
    ITERATIONS=$((LINES - 1))

    echo "📊 Performance Data: $LATEST_CSV"
    echo "   Iterations: $ITERATIONS"

    if [ $ITERATIONS -gt 1 ]; then
        echo ""
        echo "   Latest values:"
        tail -1 "$LATEST_CSV" | awk -F',' '{
            printf "   Portfolio: $%'\''d\n", $3
            printf "   Return: %.2f%%\n", $5
        }'
    fi
else
    echo "⚠️  No performance data yet"
    echo "   (Data appears after ~30 seconds)"
fi

echo ""
echo "======================================================================"
echo "Commands:"
echo "  Monitor:  ./MONITOR.sh"
echo "  Stop:     ./STOP.sh"
echo "  Logs:     tail -f logs/fast_*.log"
echo "======================================================================"
