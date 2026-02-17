#!/bin/bash
# Start FAST paper trading - results in 30 seconds!

cd "$(dirname "$0")"

echo "======================================================================"
echo "FAST PAPER TRADER - TRIAL #965"
echo "======================================================================"
echo ""
echo "This runs FAST (30-second updates) so you see results immediately!"
echo ""
echo "Performance (from stress test):"
echo "  • Sharpe: 11.52"
echo "  • Win Rate: 84.8%"
echo "  • Max Drawdown: -3.6%"
echo ""

# Stop old trader if running
if [ -f "trading.pid" ]; then
    kill $(cat trading.pid) 2>/dev/null
    rm trading.pid
    sleep 1
fi

# Ask for initial capital
echo "Enter your initial capital (default: \$1,000):"
read -p "Capital (\$): " CAPITAL

if [ -z "$CAPITAL" ]; then
    CAPITAL=1000
fi

# Validate it's a number
if ! [[ "$CAPITAL" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "Invalid amount. Using \$1,000"
    CAPITAL=1000
fi

echo ""
echo "Starting with capital: \$$CAPITAL"

# Update config file
python3 << EOF
import json
with open('config/trading_config.json', 'r') as f:
    config = json.load(f)
config['trading_settings']['initial_capital'] = float($CAPITAL)
with open('config/trading_config.json', 'w') as f:
    json.dump(config, f, indent=2)
EOF

echo ""
echo "Starting fast trader..."
echo "  - Updates every 30 seconds"
echo "  - Saves CSV immediately"
echo "  - Monitor will work right away"
echo ""

nohup python3 -u paper_trader.py --minutes 60 > logs/fast_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > trading.pid

sleep 2

if ps -p $(cat trading.pid) > /dev/null 2>&1; then
    echo "✓ Trading started!"
    echo "  PID: $(cat trading.pid)"
    echo "  Capital: \$$CAPITAL"
    echo ""
    echo "WAIT 30 SECONDS then run:"
    echo "  ./MONITOR.sh"
    echo ""
    echo "Or watch live:"
    echo "  tail -f logs/fast_*.log"
    echo ""
    echo "Stop anytime:"
    echo "  ./STOP.sh"
    echo ""
    echo "======================================================================"
else
    echo "✗ Failed to start"
    rm trading.pid
fi
