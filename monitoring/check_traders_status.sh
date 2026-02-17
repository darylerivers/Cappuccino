#!/bin/bash
# Quick status check for paper traders

echo "=========================================="
echo "Paper Trading Status Check"
echo "=========================================="
echo ""

# Check ensemble trader process
echo "1. Ensemble Trader Process:"
ENSEMBLE_PID=$(ps aux | grep "paper_trader_alpaca_polling.py" | grep "ensemble" | grep -v grep | awk '{print $2}')
if [ -n "$ENSEMBLE_PID" ]; then
    echo "   ✓ Running (PID: $ENSEMBLE_PID)"
    ps aux | grep "$ENSEMBLE_PID" | grep -v grep | awk '{print "   Started:", $9, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "   ✗ Not running"
fi
echo ""

# Check single model trader process
echo "2. Single Model Trader Process:"
SINGLE_PID=$(ps aux | grep "trial_861" | grep -v grep | awk '{print $2}')
if [ -n "$SINGLE_PID" ]; then
    echo "   ✓ Running (PID: $SINGLE_PID)"
    ps aux | grep "$SINGLE_PID" | grep -v grep | awk '{print "   Started:", $9, "CPU:", $3"%", "MEM:", $4"%"}'
else
    echo "   ✗ Not running"
fi
echo ""

# Check ensemble CSV
echo "3. Ensemble Data File:"
ENSEMBLE_CSV=$(ls -t paper_trades/watchdog_session_*.csv 2>/dev/null | head -1)
if [ -n "$ENSEMBLE_CSV" ]; then
    echo "   ✓ Found: $ENSEMBLE_CSV"
    LINES=$(wc -l < "$ENSEMBLE_CSV")
    SIZE=$(ls -lh "$ENSEMBLE_CSV" | awk '{print $5}')
    MODIFIED=$(stat -c %y "$ENSEMBLE_CSV" | cut -d'.' -f1)
    echo "   Lines: $LINES | Size: $SIZE"
    echo "   Last modified: $MODIFIED"
else
    echo "   ✗ No ensemble CSV found"
fi
echo ""

# Check single model CSV
echo "4. Single Model Data File:"
SINGLE_CSV="paper_trades/single_model_trial861.csv"
if [ -f "$SINGLE_CSV" ]; then
    echo "   ✓ Found: $SINGLE_CSV"
    LINES=$(wc -l < "$SINGLE_CSV")
    SIZE=$(ls -lh "$SINGLE_CSV" | awk '{print $5}')
    MODIFIED=$(stat -c %y "$SINGLE_CSV" | cut -d'.' -f1)
    echo "   Lines: $LINES | Size: $SIZE"
    echo "   Last modified: $MODIFIED"
else
    echo "   ✗ File not found: $SINGLE_CSV"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary:"
echo "=========================================="

ISSUES=0

if [ -z "$ENSEMBLE_PID" ]; then
    echo "⚠️  Ensemble trader not running"
    ISSUES=$((ISSUES + 1))
fi

if [ -z "$SINGLE_PID" ]; then
    echo "⚠️  Single model trader not running"
    ISSUES=$((ISSUES + 1))
fi

if [ -z "$ENSEMBLE_CSV" ]; then
    echo "⚠️  No ensemble data file"
    ISSUES=$((ISSUES + 1))
fi

if [ ! -f "$SINGLE_CSV" ]; then
    echo "⚠️  No single model data file"
    ISSUES=$((ISSUES + 1))
fi

if [ $ISSUES -eq 0 ]; then
    echo "✅ All systems operational!"
    echo ""
    echo "Ready to launch dashboard:"
    echo "  ./start_dashboard.sh"
else
    echo ""
    echo "❌ Found $ISSUES issue(s)"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check logs: tail -50 logs/single_model_trial861.log"
    echo "  - Check processes: ps aux | grep paper_trader"
    echo "  - Restart traders if needed"
fi
echo ""
