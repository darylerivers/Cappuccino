#!/bin/bash
# Stop paper trading

cd "$(dirname "$0")"

echo "======================================================================"
echo "STOPPING PAPER TRADING"
echo "======================================================================"
echo ""

if [ ! -f "trading.pid" ]; then
    echo "⚠️  No trading.pid file found"
    echo "   Trading may not be running"
    exit 1
fi

PID=$(cat trading.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping trading bot (PID: $PID)..."
    kill $PID
    sleep 2

    if ps -p $PID > /dev/null 2>&1; then
        echo "Process still running, forcing stop..."
        kill -9 $PID
    fi

    rm trading.pid
    echo "✓ Trading stopped successfully"
else
    echo "⚠️  Process $PID is not running"
    rm trading.pid
fi

echo ""
echo "View results:"
echo "  ls -lt results/"
echo "  ls -lt logs/"
echo ""
