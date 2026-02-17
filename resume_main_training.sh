#!/bin/bash
# Start NEW FT-Transformer study with GPU-optimized ranges
# (Old study had small batch sizes, this one uses 65K-131K batches and huge models)

cd /opt/user-data/experiment/cappuccino
source activate_rocm_env.sh

echo "=========================================="
echo "Starting NEW FT-Transformer study"
echo "GPU-Optimized: 65K-131K batch, 2048-4096 net_dim"
echo "Expected VRAM: 8-16GB (vs old 1-2GB)"
echo "=========================================="
echo ""

# Start new study with GPU-optimized hyperparameter ranges
# This will use the new large batch sizes and model dimensions
python scripts/training/1_optimize_unified.py \
    --study-name cappuccino_ft_16gb_optimized \
    --n-trials 1000 \
    --force-ft \
    --timeframe 1h \
    --data-dir data/1h_1680

echo ""
echo "Training completed!"
