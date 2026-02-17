#!/bin/bash
# Launch 5min training on CPU (runs in parallel with 1hr GPU training)
# Slower but doesn't compete for GPU resources

STUDY_DB="databases/5min_campaign.db"
TIMEFRAME="5m"
N_TRIALS=100
GPU=-1  # -1 = CPU mode

echo "================================================================================"
echo "5-MINUTE TIMEFRAME TRAINING CAMPAIGN (CPU MODE)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Timeframe: $TIMEFRAME"
echo "  Device: CPU (doesn't conflict with GPU training)"
echo "  Trials per study: $N_TRIALS"
echo "  Total trials: 500 (5 studies × 100)"
echo "  Database: $STUDY_DB"
echo ""
echo "⚠️  CPU training is ~10x slower than GPU, but runs in parallel!"
echo "    Expected completion: 8-10 days (vs 2 days on GPU)"
echo ""
echo "Studies:"
echo "  1. ensemble_5m_conservative (baseline MLP, conservative risk)"
echo "  2. ensemble_5m_balanced     (baseline MLP, balanced risk)"
echo "  3. ensemble_5m_aggressive   (baseline MLP, aggressive risk)"
echo "  4. ft_5m_small              (FT-Transformer, smaller)"
echo "  5. ft_5m_large              (FT-Transformer, larger)"
echo ""
echo "================================================================================"
echo ""

# Wait for data download to finish
echo "Checking if 5min data is ready..."
DATA_FILE="data/crypto_5m_6mo.pkl"

while [ ! -f "$DATA_FILE" ]; do
    echo "  Waiting for data download to complete..."
    echo "  Checking for: $DATA_FILE"
    sleep 30
done

echo "✓ Data file found: $DATA_FILE"
echo ""

# Create logs directory
mkdir -p logs/training

# 1. Ensemble Conservative
echo "Launching ensemble_5m_conservative (CPU)..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_5m_conservative_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-baseline \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ensemble_5m_conservative_cpu.log 2>&1 &
PID1=$!
echo "  PID: $PID1"

sleep 2

# 2. Ensemble Balanced
echo "Launching ensemble_5m_balanced (CPU)..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_5m_balanced_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-baseline \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ensemble_5m_balanced_cpu.log 2>&1 &
PID2=$!
echo "  PID: $PID2"

sleep 2

# 3. Ensemble Aggressive
echo "Launching ensemble_5m_aggressive (CPU)..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ensemble_5m_aggressive_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-baseline \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ensemble_5m_aggressive_cpu.log 2>&1 &
PID3=$!
echo "  PID: $PID3"

sleep 2

# 4. FT-Transformer Small
echo "Launching ft_5m_small (CPU)..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ft_5m_small_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-ft \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ft_5m_small_cpu.log 2>&1 &
PID4=$!
echo "  PID: $PID4"

sleep 2

# 5. FT-Transformer Large
echo "Launching ft_5m_large (CPU)..."
nohup python scripts/training/1_optimize_unified.py \
    --study-name ft_5m_large_$(date +%Y%m%d) \
    --n-trials $N_TRIALS --gpu $GPU --timeframe $TIMEFRAME \
    --force-ft \
    --storage sqlite:///$STUDY_DB \
    > logs/training/ft_5m_large_cpu.log 2>&1 &
PID5=$!
echo "  PID: $PID5"

echo ""
echo "================================================================================"
echo "5-MINUTE TRAINING CAMPAIGN LAUNCHED (CPU MODE)!"
echo "================================================================================"
echo ""
echo "Process IDs:"
echo "  ensemble_5m_conservative: $PID1"
echo "  ensemble_5m_balanced:     $PID2"
echo "  ensemble_5m_aggressive:   $PID3"
echo "  ft_5m_small:              $PID4"
echo "  ft_5m_large:              $PID5"
echo ""
echo "Running on: CPU (all cores)"
echo "Parallel with: 1hr GPU training (no conflict)"
echo ""
echo "Monitor progress:"
echo "  python monitor_training.py --db databases/5min_campaign.db"
echo "  tail -f logs/training/ensemble_5m_conservative_cpu.log"
echo "  htop  # Check CPU usage"
echo ""
echo "Discord notifications enabled: Training updates every 10 trials"
echo ""
echo "Estimated completion: 8-10 days (CPU is slower but runs in parallel)"
echo ""
echo "⚡ TIP: When 1hr GPU training finishes, you can migrate some 5min"
echo "   studies to GPU for faster completion if desired."
echo "================================================================================"
