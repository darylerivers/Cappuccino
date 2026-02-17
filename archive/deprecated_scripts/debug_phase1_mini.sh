#!/bin/bash
# Debug script to run Phase 1 mini-test with visible output
# Shows real-time progress of training

echo "========================================"
echo "Phase 1 Mini-Test Debug Runner"
echo "========================================"
echo ""
echo "This will run 2 combinations × 5 trials = 10 total trials"
echo "Estimated time: 5-10 minutes"
echo ""
echo "Starting..."
echo ""

# Run with unbuffered Python output
python -u phase1_timeframe_optimizer.py --mini-test

echo ""
echo "========================================"
echo "Phase 1 Mini-Test Complete!"
echo "========================================"
echo ""
echo "Results:"
ls -lh phase1_*.json 2>/dev/null || echo "No result files yet"
echo ""
echo "Training directories created:"
ls -1 train_results/phase1/ | wc -l
echo "trials"
