#!/bin/bash
# Stop Entire Cappuccino System
# Kills all training, pipeline, automation, and paper trading processes

cd /opt/user-data/experiment/cappuccino

echo "=========================================="
echo "STOPPING CAPPUCCINO COMPLETE SYSTEM"
echo "=========================================="
echo ""

# 1. Stop Training Workers
echo "[1/6] Stopping training workers..."
pkill -f "1_optimize_unified.py" && echo "  ✓ Training workers stopped" || echo "  ℹ No training workers running"
sleep 1

# 2. Stop Pipeline
echo "[2/6] Stopping pipeline..."
pkill -f "pipeline_v2.py" && echo "  ✓ Pipeline stopped" || echo "  ℹ No pipeline running"
sleep 1

# 3. Stop Paper Traders
echo "[3/6] Stopping paper traders..."
pkill -f "paper_trader_alpaca_polling.py" && echo "  ✓ Paper traders stopped" || echo "  ℹ No paper traders running"
sleep 1

# 4. Stop Auto-Model Deployer
echo "[4/6] Stopping auto-model deployer..."
pkill -f "auto_model_deployer.py" && echo "  ✓ Auto deployer stopped" || echo "  ℹ No auto deployer running"
sleep 1

# 5. Stop System Watchdog
echo "[5/6] Stopping system watchdog..."
pkill -f "system_watchdog.py" && echo "  ✓ Watchdog stopped" || echo "  ℹ No watchdog running"
sleep 1

# 6. Stop Performance Monitor and Ensemble Updater
echo "[6/6] Stopping monitors and updaters..."
pkill -f "performance_monitor.py" && echo "  ✓ Performance monitor stopped" || echo "  ℹ No performance monitor running"
pkill -f "ensemble_auto_updater.py" && echo "  ✓ Ensemble updater stopped" || echo "  ℹ No ensemble updater running"
sleep 1

echo ""
echo "=========================================="
echo "VERIFYING SHUTDOWN"
echo "=========================================="
echo ""

# Check for any remaining processes
REMAINING=$(ps aux | grep -E "1_optimize|pipeline_v2|paper_trader|auto_model|watchdog|performance_monitor|ensemble_auto" | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "✓ All systems stopped successfully"
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk '{printf "  VRAM: %d / %d MB (%.1f%%)\n  GPU Util: %d%%\n", $1, $2, ($1/$2)*100, $3}'
else
    echo "⚠ Warning: $REMAINING processes still running"
    echo ""
    echo "Remaining processes:"
    ps aux | grep -E "1_optimize|pipeline_v2|paper_trader|auto_model|watchdog|performance_monitor|ensemble_auto" | grep -v grep
    echo ""
    echo "Force kill all remaining? (y/n)"
    read -r FORCE
    if [ "$FORCE" = "y" ]; then
        pkill -9 -f "1_optimize|pipeline_v2|paper_trader|auto_model|watchdog|performance_monitor|ensemble_auto"
        echo "✓ Force killed all remaining processes"
    fi
fi

echo ""
echo "=========================================="
echo "SHUTDOWN COMPLETE"
echo "=========================================="
echo ""
echo "To restart: ./start_everything.sh"
echo ""
