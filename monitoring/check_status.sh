#!/bin/bash
# Quick status check for 14-indicator training

echo "════════════════════════════════════════════════════════════════"
echo "  CAPPUCCINO TRAINING STATUS - 14 Indicators"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Training process
echo "📊 Training Process:"
if ps -p 200185 > /dev/null 2>&1; then
    echo "  ✅ Active (PID: 200185)"
    ELAPSED=$(ps -p 200185 -o etime= | tr -d ' ')
    echo "  ⏱️  Elapsed: $ELAPSED"
else
    echo "  ❌ Not running"
fi
echo ""

# Latest log entry
echo "📝 Latest Training Log:"
tail -5 logs/training/training_14indicators_20260205_155835.log 2>/dev/null || echo "  No log yet"
echo ""

# GPU status
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"
echo ""

# Paper trader
echo "💹 Paper Trader (Old Model):"
if ps -p 13287 > /dev/null 2>&1; then
    echo "  ✅ Running (PID: 13287)"
else
    echo "  ❌ Stopped"
fi
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "Run './monitor_training.sh' for detailed monitoring"
echo "════════════════════════════════════════════════════════════════"
