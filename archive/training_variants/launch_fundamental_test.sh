#!/bin/bash
# Launch Fundamental Fixes Test - 50 Trial Validation
#
# This script launches the validation test for the three fundamental fixes:
# 1. Enhanced concentration penalty
# 2. Revised reward function
# 3. Cash reserve enforcement
#
# Expected duration: 8-10 hours (overnight run)

set -e

echo "=============================================================================="
echo "FUNDAMENTAL FIXES - 50 TRIAL VALIDATION TEST"
echo "=============================================================================="
echo ""
echo "This will test the three fundamental fixes applied to environment_Alpaca.py:"
echo "  1. Enhanced concentration penalty (require 3+ positions, cap at 40%)"
echo "  2. Revised reward function (50% alpha + 30% absolute + 20% cash)"
echo "  3. Cash reserve enforcement (penalty for violations)"
echo ""
echo "Configuration:"
echo "  Study: cappuccino_fundamentals_test_20251215"
echo "  Trials: 50"
echo "  Expected duration: 8-10 hours"
echo "  GPU: 0 (default)"
echo ""
echo "=============================================================================="
echo ""

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Check if data directory exists
if [ ! -d "data/1h_1680" ]; then
    echo "ERROR: Data directory 'data/1h_1680' not found!"
    echo "Please ensure training data is available before running test."
    exit 1
fi

echo "Starting test in 5 seconds... (Press Ctrl+C to cancel)"
sleep 5

# Run test in background with nohup
nohup python test_fundamental_fixes.py \
    --n-trials 50 \
    --gpu 0 \
    --study-name cappuccino_fundamentals_test_20251215 \
    --data-dir data/1h_1680 \
    --num-paths 3 \
    --k-test-groups 2 \
    > logs/fundamental_test_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TEST_PID=$!

echo ""
echo "=============================================================================="
echo "TEST LAUNCHED SUCCESSFULLY"
echo "=============================================================================="
echo "  Process ID: $TEST_PID"
echo "  Log file: logs/fundamental_test_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "To monitor progress:"
echo "  tail -f logs/fundamental_test_*.log"
echo ""
echo "To check status:"
echo "  ps aux | grep test_fundamental_fixes"
echo ""
echo "To view results (after completion):"
echo "  python analyze_test_results.py --study cappuccino_fundamentals_test_20251215"
echo ""
echo "=============================================================================="
echo ""
