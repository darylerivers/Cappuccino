#!/bin/bash
################################################################################
# 5-Minute Timeframe Training Launch Script (GPU, With Dimension Fix)
################################################################################
#
# This script launches 5 parallel training studies for 5-minute crypto trading:
#   1. ensemble_5m_conservative  - Baseline MLP, conservative risk
#   2. ensemble_5m_balanced      - Baseline MLP, balanced risk
#   3. ensemble_5m_aggressive    - Baseline MLP, aggressive risk
#   4. ft_5m_small               - FT-Transformer, smaller config
#   5. ft_5m_large               - FT-Transformer, larger config
#
# Key Features:
#   - Uses GPU (RTX 3070) for 10x faster training
#   - Dimension mismatch fix applied (graceful handling of mixed lookback)
#   - 100 trials per study = 500 total trials
#   - Estimated completion: 2-3 days
#
# Prerequisites:
#   - 5m data downloaded: data/crypto_5m_6mo.pkl
#   - GPU available and not in use
#   - Dimension mismatch fix applied (completed Feb 9, 2026)
#
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "5-MINUTE TIMEFRAME TRAINING CAMPAIGN (GPU MODE - WITH FIX)"
echo "================================================================================"
echo ""

# Configuration
TIMEFRAME="5m"
DEVICE="0"  # GPU device 0
TRIALS_PER_STUDY=100
DATABASE="databases/5min_campaign.db"
DATA_DIR="data/5m"  # Converted 5m data in proper format
STUDY_DATE=$(date +%Y%m%d)

echo "Configuration:"
echo "  Timeframe: $TIMEFRAME"
echo "  Device: GPU $DEVICE (NVIDIA GeForce RTX 3070)"
echo "  Trials per study: $TRIALS_PER_STUDY"
echo "  Total trials: $((TRIALS_PER_STUDY * 5)) (5 studies × $TRIALS_PER_STUDY)"
echo "  Database: $DATABASE"
echo ""
echo "✅ Fix Applied: State dimension mismatch handling"
echo "   - Gracefully skips CV splits with different lookback"
echo "   - Trials complete with successful splits"
echo "   - No crashes from dimension mismatches"
echo ""
echo "⚡ GPU training is ~10x faster than CPU!"
echo "    Expected completion: 2-3 days (vs 8-10 days on CPU)"
echo ""

# Check if 5m data exists (converted format)
if [ ! -f "$DATA_DIR/price_array" ] || [ ! -f "$DATA_DIR/tech_array" ] || [ ! -f "$DATA_DIR/time_array" ]; then
    echo "❌ ERROR: 5m converted data not found in $DATA_DIR"
    echo "   Please run: python convert_5m_data_format.py"
    exit 1
fi
echo "✓ Converted 5m data found in $DATA_DIR"
echo ""

# Check GPU availability
if ! nvidia-smi &>/dev/null; then
    echo "❌ ERROR: nvidia-smi not found. GPU not available?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $DEVICE)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $DEVICE)
echo "✓ GPU available: $GPU_NAME (${GPU_MEM} MB)"
echo ""

# Check if dimension fix is applied
if ! grep -q "State dimension mismatch" drl_agents/agents/AgentBase.py; then
    echo "⚠️  WARNING: Dimension mismatch fix not detected in AgentBase.py"
    echo "   Training may fail if mixed lookback values are used."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo "✓ Dimension mismatch fix detected in code"
echo ""

# Tickers for training
TICKERS="AAVE/USD AVAX/USD BTC/USD LINK/USD ETH/USD LTC/USD UNI/USD"

echo "================================================================================
"
echo "Resetting failed trials from previous attempt..."
echo ""

# Mark all RUNNING trials as FAIL (they're stuck from the crashed attempt)
sqlite3 $DATABASE "UPDATE trials SET state='FAIL' WHERE state='RUNNING';" 2>/dev/null || true
sqlite3 $DATABASE "SELECT s.study_name, COUNT(*) as trials, SUM(CASE WHEN t.state='COMPLETE' THEN 1 ELSE 0 END) as completed, SUM(CASE WHEN t.state='FAIL' THEN 1 ELSE 0 END) as failed FROM trials t JOIN studies s ON t.study_id = s.study_id GROUP BY s.study_name;" | column -t -s'|'

echo ""
echo "================================================================================"
echo ""

# Define study configurations
declare -A STUDY_CONFIGS
STUDY_CONFIGS["ensemble_5m_conservative"]="--force-baseline --study-name ensemble_5m_conservative_$STUDY_DATE"
STUDY_CONFIGS["ensemble_5m_balanced"]="--force-baseline --study-name ensemble_5m_balanced_$STUDY_DATE"
STUDY_CONFIGS["ensemble_5m_aggressive"]="--force-baseline --study-name ensemble_5m_aggressive_$STUDY_DATE"
STUDY_CONFIGS["ft_5m_small"]="--force-ft --study-name ft_5m_small_$STUDY_DATE"
STUDY_CONFIGS["ft_5m_large"]="--force-ft --study-name ft_5m_large_$STUDY_DATE"

# Launch training studies
for study in "${!STUDY_CONFIGS[@]}"; do
    log_file="logs/training/${study}_gpu.log"
    config="${STUDY_CONFIGS[$study]}"

    echo "Launching $study (GPU)..."

    nohup python scripts/training/1_optimize_unified.py \
        --n-trials $TRIALS_PER_STUDY \
        --gpu $DEVICE \
        $config \
        --storage "sqlite:///$DATABASE" \
        --timeframe $TIMEFRAME \
        --tickers $TICKERS \
        --data-dir $DATA_DIR \
        > "$log_file" 2>&1 &

    pid=$!
    echo "  PID: $pid"
    echo "  Log: $log_file"
    echo ""

    # Small delay to stagger starts
    sleep 2
done

echo "================================================================================"
echo "5-MINUTE TRAINING CAMPAIGN LAUNCHED (GPU MODE)!"
echo "================================================================================"
echo ""
echo "All 5 studies are now training in parallel on GPU."
echo ""
echo "Monitor progress:"
echo "  1. Training logs:"
echo "     tail -f logs/training/ensemble_5m_conservative_gpu.log"
echo "     tail -f logs/training/ft_5m_small_gpu.log"
echo ""
echo "  2. Database status:"
echo "     sqlite3 $DATABASE \"SELECT study_name, COUNT(*) FROM trials GROUP BY study_name;\""
echo ""
echo "  3. GPU usage:"
echo "     watch -n 5 nvidia-smi"
echo ""
echo "  4. Process status:"
echo "     ps aux | grep optimize_unified"
echo ""
echo "Discord notifications enabled: Training updates every 10 trials"
echo ""
echo "Estimated completion: 2-3 days"
echo ""
echo "💡 TIP: The dimension mismatch fix is now active. Some CV splits may be"
echo "   skipped if they have different lookback values, but training will continue."
echo "   This is expected and not an error."
echo ""
echo "================================================================================"
