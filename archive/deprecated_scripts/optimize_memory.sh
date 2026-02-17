#!/bin/bash
# Memory Optimization Helper Script

echo "========================================"
echo "Memory Optimization Helper"
echo "========================================"
echo ""

# Check current resource usage
echo "Current Resource Usage:"
echo "------------------------"
echo ""
echo "GPU VRAM:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | \
  awk -F, '{printf "  Used: %s MB / %s MB (%.0f%%)\n  GPU Util: %s\n", $1, $2, ($1/$2)*100, $3}'
echo ""

echo "Training Workers:"
WORKER_COUNT=$(ps aux | grep "optimize_unified" | grep -v grep | wc -l)
WORKER_RAM=$(ps aux | grep "optimize_unified" | grep -v grep | awk '{sum+=$6} END {print int(sum/1024)}')
echo "  Count: $WORKER_COUNT workers"
echo "  Total RAM: ${WORKER_RAM} MB"
if [ $WORKER_COUNT -gt 0 ]; then
  echo "  Per worker: $((WORKER_RAM / WORKER_COUNT)) MB average"
fi
echo ""

echo "Paper Traders:"
TRADER_COUNT=$(ps aux | grep "paper_trader" | grep -v grep | wc -l)
TRADER_RAM=$(ps aux | grep "paper_trader" | grep -v grep | awk '{sum+=$6} END {print int(sum/1024)}')
echo "  Count: $TRADER_COUNT traders"
echo "  Total RAM: ${TRADER_RAM} MB"
echo ""

echo "Total Python RAM: $(ps aux | grep python | grep -v grep | awk '{sum+=$6} END {print int(sum/1024)}') MB"
echo ""

# Optimization options
echo "========================================"
echo "Optimization Options:"
echo "========================================"
echo ""

echo "1. STOP 2 TRAINING WORKERS (Recommended)"
echo "   Savings: ~2 GB VRAM, ~2.5 GB RAM"
echo "   Impact: -22% training speed (still good at 1,351 trials)"
echo "   Risk: LOW"
echo ""

echo "2. STOP 3 TRAINING WORKERS"
echo "   Savings: ~3 GB VRAM, ~4 GB RAM"
echo "   Impact: -33% training speed"
echo "   Risk: LOW-MEDIUM"
echo ""

echo "3. CREATE LEAN ENSEMBLE (5 models instead of 10)"
echo "   Savings: ~1 GB RAM (paper traders only)"
echo "   Impact: Less model diversity"
echo "   Risk: MEDIUM (requires testing)"
echo ""

echo "4. SHOW DETAILED PROCESS LIST"
echo "   Just view current processes"
echo ""

echo "5. EXIT"
echo ""

read -p "Choose option (1-5): " choice

case $choice in
  1)
    echo ""
    echo "Stopping 2 training workers..."
    echo ""

    # Get last 2 worker PIDs
    PIDS=$(ps aux | grep "optimize_unified" | grep -v grep | tail -2 | awk '{print $2}')

    if [ -z "$PIDS" ]; then
      echo "No workers found to stop!"
      exit 1
    fi

    echo "Will stop these workers:"
    ps aux | grep "optimize_unified" | grep -v grep | tail -2 | \
      awk '{printf "  PID: %s, RAM: %d MB, CPU: %s%%\n", $2, $6/1024, $3}'
    echo ""

    read -p "Confirm? (y/n): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
      for pid in $PIDS; do
        kill $pid && echo "  Stopped PID: $pid"
      done

      echo ""
      echo "Waiting 3 seconds for cleanup..."
      sleep 3

      echo ""
      echo "New resource usage:"
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | \
        awk -F, '{printf "  VRAM: %s MB / %s MB (%.0f%%)\n", $1, $2, ($1/$2)*100}'
      echo "  Remaining workers: $(ps aux | grep optimize_unified | grep -v grep | wc -l)"
      echo ""
      echo "✅ Optimization complete!"
    else
      echo "Cancelled."
    fi
    ;;

  2)
    echo ""
    echo "Stopping 3 training workers..."
    echo ""

    PIDS=$(ps aux | grep "optimize_unified" | grep -v grep | tail -3 | awk '{print $2}')

    if [ -z "$PIDS" ]; then
      echo "No workers found to stop!"
      exit 1
    fi

    echo "Will stop these workers:"
    ps aux | grep "optimize_unified" | grep -v grep | tail -3 | \
      awk '{printf "  PID: %s, RAM: %d MB, CPU: %s%%\n", $2, $6/1024, $3}'
    echo ""

    read -p "Confirm? (y/n): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
      for pid in $PIDS; do
        kill $pid && echo "  Stopped PID: $pid"
      done

      echo ""
      echo "Waiting 3 seconds for cleanup..."
      sleep 3

      echo ""
      echo "New resource usage:"
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | \
        awk -F, '{printf "  VRAM: %s MB / %s MB (%.0f%%)\n", $1, $2, ($1/$2)*100}'
      echo "  Remaining workers: $(ps aux | grep optimize_unified | grep -v grep | wc -l)"
      echo ""
      echo "✅ Optimization complete!"
    else
      echo "Cancelled."
    fi
    ;;

  3)
    echo ""
    echo "Creating lean ensemble (5 models)..."
    echo ""

    if [ ! -d "train_results/ensemble" ]; then
      echo "❌ Ensemble directory not found!"
      exit 1
    fi

    # Backup original
    if [ ! -d "train_results/ensemble_original" ]; then
      echo "  Backing up original ensemble..."
      cp -r train_results/ensemble train_results/ensemble_original
    fi

    # Create lean version
    if [ ! -d "train_results/ensemble_lean" ]; then
      echo "  Creating lean ensemble..."
      cp -r train_results/ensemble train_results/ensemble_lean
    fi

    # Update top_n parameter
    echo "  Updating ensemble_params.json..."
    python -c "
import json
params_file = 'train_results/ensemble_lean/ensemble_params.json'
with open(params_file, 'r') as f:
    params = json.load(f)
params['top_n'] = 5
with open(params_file, 'w') as f:
    json.dump(params, f, indent=2)
print('  Updated top_n to 5')
"

    echo ""
    echo "✅ Lean ensemble created at train_results/ensemble_lean"
    echo ""
    echo "To use it:"
    echo "  1. Stop current ensemble trader"
    echo "  2. Restart with --model-dir train_results/ensemble_lean"
    echo "  3. Monitor for 24h to compare performance"
    echo ""
    echo "Example:"
    echo "  kill <ensemble_trader_PID>"
    echo "  nohup python -u paper_trader_alpaca_polling.py \\"
    echo "    --model-dir train_results/ensemble_lean \\"
    echo "    --tickers BTC/USD ETH/USD LTC/USD BCH/USD LINK/USD UNI/USD AAVE/USD \\"
    echo "    --timeframe 1h --poll-interval 60 --gpu -1 \\"
    echo "    --log-file paper_trades/ensemble_lean.csv \\"
    echo "    --max-position-pct 0.30 --stop-loss-pct 0.10 \\"
    echo "    > logs/ensemble_lean.log 2>&1 &"
    ;;

  4)
    echo ""
    echo "Detailed Process List:"
    echo "======================"
    echo ""

    echo "Training Workers:"
    ps aux | grep "optimize_unified" | grep -v grep | \
      awk '{printf "  PID: %-8s CPU: %4s%% RAM: %6d MB\n", $2, $3, $6/1024}'
    echo ""

    echo "Paper Traders:"
    ps aux | grep "paper_trader" | grep -v grep | \
      awk '{printf "  PID: %-8s CPU: %4s%% RAM: %6d MB  Model: %s\n", $2, $3, $6/1024, $13}'
    echo ""

    echo "Monitoring:"
    ps aux | grep -E "alert_system|watchdog|monitor|dashboard" | grep -v grep | \
      awk '{printf "  PID: %-8s CPU: %4s%% RAM: %6d MB  %s\n", $2, $3, $6/1024, $11}'
    ;;

  5)
    echo "Exiting..."
    exit 0
    ;;

  *)
    echo "Invalid option!"
    exit 1
    ;;
esac
