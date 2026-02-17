#!/usr/bin/env bash
#
# Start Paper Trader from Archived Trial
#
# Automatically starts paper trading with the best archived trial
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "  STARTING PAPER TRADER FROM BEST ARCHIVED TRIAL"
echo "========================================================================"
echo ""

# Get best trial from archive
BEST_TRIAL=$(python -c "
import json
from pathlib import Path

registry = Path('trial_archive/trial_registry.json')
if not registry.exists():
    print('ERROR: No archived trials found')
    exit(1)

with open(registry) as f:
    data = json.load(f)

trials = data.get('trials', [])
if not trials:
    print('ERROR: No trials in registry')
    exit(1)

# Sort by grade then sharpe
trials.sort(key=lambda x: (x['grade_numeric'], x['sharpe']), reverse=True)
best = trials[0]

print(f\"{best['vin']}\")
print(f\"Grade: {best['grade']}, Sharpe: {best['sharpe']:.4f}\", file=__import__('sys').stderr)
")

if [ $? -ne 0 ]; then
    echo "❌ $BEST_TRIAL"
    exit 1
fi

echo "🏆 Best Trial: $BEST_TRIAL"
echo ""

# Check if paper trader is already running
if pgrep -f "paper_trader.*alpaca" > /dev/null; then
    echo "⚠️  Paper trader already running!"
    echo "   Stop it first: pkill -f paper_trader"
    read -p "Stop existing trader and start new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "paper_trader.*alpaca"
        sleep 2
    else
        exit 0
    fi
fi

# Find the paper trader script
PAPER_TRADER_SCRIPT=""
if [ -f "scripts/deployment/paper_trader_alpaca_polling.py" ]; then
    PAPER_TRADER_SCRIPT="scripts/deployment/paper_trader_alpaca_polling.py"
elif [ -f "paper_trader_alpaca_polling.py" ]; then
    PAPER_TRADER_SCRIPT="paper_trader_alpaca_polling.py"
else
    echo "❌ Paper trader script not found!"
    exit 1
fi

# Start paper trader
echo "🚀 Starting paper trader..."
echo "   Script: $PAPER_TRADER_SCRIPT"
echo "   Trial: $BEST_TRIAL"
echo "   Log: logs/paper_trader_live.log"
echo ""

nohup python -u "$PAPER_TRADER_SCRIPT" \
    --trial-name "$BEST_TRIAL" \
    > logs/paper_trader_live.log 2>&1 &

PID=$!
echo "✅ Paper trader started (PID: $PID)"
echo ""
echo "Monitor with:"
echo "  tail -f logs/paper_trader_live.log"
echo "  python scripts/automation/trial_dashboard.py"
echo ""
