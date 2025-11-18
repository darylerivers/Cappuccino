#!/bin/bash
#
# Paper Trading with Arbitrage Scanner
#
# Runs both the main RL paper trading agent and the arbitrage scanner in parallel.
# The RL agent handles primary trading, while the arbitrage scanner monitors for
# cross-currency opportunities.
#

set -euo pipefail

# Configuration
MODEL_DIR="${1:-train_results/cwd_tests/trial_3358_1h}"
ARBITRAGE_INTERVAL="${2:-300}"  # 5 minutes default
ARBITRAGE_MIN_PROFIT="${3:-0.005}"  # 0.5% default (more realistic than 1%)

echo "================================================"
echo "Paper Trading with Arbitrage Scanner"
echo "================================================"
echo "Model: $MODEL_DIR"
echo "Arbitrage scan interval: ${ARBITRAGE_INTERVAL}s"
echo "Arbitrage min profit: $(echo "$ARBITRAGE_MIN_PROFIT * 100" | bc)%"
echo "================================================"
echo ""

# Check if paper trading is already running
if pgrep -f "paper_trader_alpaca_polling.py" > /dev/null; then
    echo "⚠️  Paper trading already running!"
    echo "Stop it first with: pkill -f paper_trader_alpaca_polling"
    exit 1
fi

# Check if arbitrage scanner is already running
if pgrep -f "arbitrage_scanner.py" > /dev/null; then
    echo "⚠️  Arbitrage scanner already running!"
    echo "Stop it first with: pkill -f arbitrage_scanner.py"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start arbitrage scanner in background
echo "Starting arbitrage scanner..."
python arbitrage_scanner.py \
    --interval "$ARBITRAGE_INTERVAL" \
    --min-profit "$ARBITRAGE_MIN_PROFIT" \
    > logs/arbitrage_scanner_stdout.log 2>&1 &

ARBITRAGE_PID=$!
echo "✓ Arbitrage scanner started (PID: $ARBITRAGE_PID)"

# Give it a moment to initialize
sleep 2

# Check if scanner is still running
if ! ps -p $ARBITRAGE_PID > /dev/null 2>&1; then
    echo "✗ Arbitrage scanner failed to start!"
    tail -20 logs/arbitrage_scanner.log
    exit 1
fi

# Start paper trading (using the failsafe wrapper)
echo ""
echo "Starting paper trading with failsafe..."
./paper_trading_failsafe.sh "$MODEL_DIR" &

PAPER_TRADING_PID=$!
echo "✓ Paper trading started (PID: $PAPER_TRADING_PID)"

echo ""
echo "================================================"
echo "Both systems running!"
echo "================================================"
echo "Paper trading PID:    $PAPER_TRADING_PID"
echo "Arbitrage scanner PID: $ARBITRAGE_PID"
echo ""
echo "Logs:"
echo "  Paper trading:  tail -f logs/paper_trading_live.log"
echo "  Arbitrage:      tail -f logs/arbitrage_scanner.log"
echo "  Opportunities:  tail -f logs/arbitrage_opportunities.json"
echo ""
echo "To stop:"
echo "  pkill -f paper_trading_failsafe.sh"
echo "  pkill -f arbitrage_scanner.py"
echo "================================================"

# Wait for both processes (this script will stay running)
wait
