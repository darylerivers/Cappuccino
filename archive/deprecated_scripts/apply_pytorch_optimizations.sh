#!/bin/bash
# Apply PyTorch optimizations and restart training

echo "=== Applying PyTorch Performance Optimizations ==="
echo ""
echo "Changes applied to 1_optimize_unified.py:"
echo "  ✓ TF32 enabled (8x faster matmuls on RTX 3070)"
echo "  ✓ cuDNN benchmark enabled (auto-tune for best performance)"
echo "  ✓ Batch sizes increased: 49k-98k (was 32k-65k)"
echo "  ✓ Debugging overhead disabled"
echo ""
echo "To apply optimizations, restart training:"
echo "  1. Stop current training: pkill -f '1_optimize_unified.py'"
echo "  2. Start new training with optimizations"
echo ""
read -p "Restart training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Stopping current training..."
    pkill -f '1_optimize_unified.py'
    sleep 3
    
    echo "Starting optimized training..."
    nohup python 1_optimize_unified.py \
        --n-trials 100 \
        --gpu 0 \
        --study-name maxgpu_optimized \
        --storage sqlite:////tmp/optuna_working.db \
        --data-dir data/2year_fresh_20260112 \
        > logs/training_working.log 2>&1 &
    
    echo "Training PID: $!"
    sleep 5
    
    echo ""
    echo "=== Performance Check ==="
    nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv
    
    echo ""
    echo "Monitor with: python watch_training.py"
else
    echo "Skipped. Run manually when ready."
fi
