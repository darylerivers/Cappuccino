#!/bin/bash
# Quick Validation Script
# Runs critical tests and baseline comparison

set -e  # Exit on error

echo "=========================================="
echo "Cappuccino Validation Suite"
echo "=========================================="
echo ""

# Check dependencies
echo "Checking dependencies..."
command -v pytest >/dev/null 2>&1 || { echo "❌ pytest not installed. Run: pip install pytest"; exit 1; }
echo "✓ pytest found"

echo ""
echo "=========================================="
echo "Step 1: Running Critical Tests"
echo "=========================================="
pytest tests/test_critical.py -v --tb=short

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "❌ Some tests failed. Review output above."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Checking for Data Files"
echo "=========================================="

if [ -f "data/price_array_val.npy" ]; then
    echo "✓ Validation data found"

    echo ""
    echo "=========================================="
    echo "Step 3: Running Buy-and-Hold Baseline"
    echo "=========================================="
    python baselines/buy_and_hold.py \
        --data data/price_array_val.npy \
        --initial-capital 1000 \
        --output baselines/results_buy_and_hold.json

    echo ""
    echo "=========================================="
    echo "Baseline Results Saved"
    echo "=========================================="
    cat baselines/results_buy_and_hold.json | python -m json.tool

else
    echo "⚠️  Validation data not found: data/price_array_val.npy"
    echo ""
    echo "To generate data, run:"
    echo "  python 0_dl_trainval_data.py"
    echo ""
    echo "Skipping baseline comparison..."
fi

echo ""
echo "=========================================="
echo "Step 4: Checking System Status"
echo "=========================================="

# Check if paper trader is running
if pgrep -f "paper_trader_alpaca_polling" > /dev/null; then
    echo "✓ Paper trader is running"
    TRADER_PID=$(pgrep -f "paper_trader_alpaca_polling")
    echo "  PID: $TRADER_PID"

    # Check recent activity
    if [ -f "paper_trades/alpaca_session.csv" ]; then
        LINES=$(wc -l < paper_trades/alpaca_session.csv)
        echo "  Trading log: $LINES lines"

        # Show last trade
        echo ""
        echo "Last trade:"
        tail -1 paper_trades/alpaca_session.csv
    fi
else
    echo "⚠️  Paper trader not running"
fi

# Check automation
echo ""
if pgrep -f "system_watchdog" > /dev/null; then
    echo "✓ Watchdog running"
else
    echo "⚠️  Watchdog not running"
fi

if pgrep -f "auto_model_deployer" > /dev/null; then
    echo "✓ Auto-deployer running"
else
    echo "⚠️  Auto-deployer not running"
fi

echo ""
echo "=========================================="
echo "Step 5: Check Profit Protection Log"
echo "=========================================="

if [ -f "paper_trades/profit_protection.log" ]; then
    EVENTS=$(wc -l < paper_trades/profit_protection.log)
    echo "✓ Profit protection log found: $EVENTS events"

    if [ $EVENTS -gt 0 ]; then
        echo ""
        echo "Recent profit protection events:"
        tail -5 paper_trades/profit_protection.log
    else
        echo "  (No events yet - profit protection not triggered)"
    fi
else
    echo "⚠️  No profit protection log yet"
    echo "  (Will be created when paper trader starts)"
fi

echo ""
echo "=========================================="
echo "Validation Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Critical tests passed"
if [ -f "baselines/results_buy_and_hold.json" ]; then
    echo "  ✓ Baseline comparison complete"
else
    echo "  ⚠️  Baseline comparison skipped (no data)"
fi
echo ""
echo "Next steps:"
echo "  1. Review baseline results in baselines/results_buy_and_hold.json"
echo "  2. Compare DRL performance to baseline Sharpe ratio"
echo "  3. If baseline > DRL: investigate reward function"
echo "  4. Monitor paper_trades/profit_protection.log for new events"
echo ""
echo "Quick commands:"
echo "  tail -f paper_trades/profit_protection.log  # Monitor protection"
echo "  ./status_automation.sh                       # Check system"
echo "  pytest tests/test_critical.py -v             # Re-run tests"
echo ""
