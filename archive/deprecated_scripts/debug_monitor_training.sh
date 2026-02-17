#!/bin/bash
# Monitor active training progress in real-time
# Runs in a loop showing current status

echo "========================================"
echo "Training Progress Monitor"
echo "========================================"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Training Status - $(date '+%H:%M:%S')"
    echo "=========================================="
    echo ""

    # Check for running training processes
    PHASE1_PID=$(pgrep -f "phase1_timeframe_optimizer" | head -1)
    PHASE2_PID=$(pgrep -f "phase2_feature_maximizer" | head -1)

    if [ -n "$PHASE1_PID" ]; then
        echo "✓ Phase 1 Training ACTIVE (PID: $PHASE1_PID)"
        CPU=$(ps -p $PHASE1_PID -o %cpu= 2>/dev/null || echo "0")
        MEM=$(ps -p $PHASE1_PID -o %mem= 2>/dev/null || echo "0")
        TIME=$(ps -p $PHASE1_PID -o etime= 2>/dev/null || echo "0")
        echo "  CPU: ${CPU}% | Memory: ${MEM}% | Runtime: ${TIME}"
    else
        echo "○ Phase 1 Training: Not running"
    fi

    if [ -n "$PHASE2_PID" ]; then
        echo "✓ Phase 2 Training ACTIVE (PID: $PHASE2_PID)"
        CPU=$(ps -p $PHASE2_PID -o %cpu= 2>/dev/null || echo "0")
        MEM=$(ps -p $PHASE2_PID -o %mem= 2>/dev/null || echo "0")
        TIME=$(ps -p $PHASE2_PID -o etime= 2>/dev/null || echo "0")
        echo "  CPU: ${CPU}% | Memory: ${MEM}% | Runtime: ${TIME}"
    else
        echo "○ Phase 2 Training: Not running"
    fi

    echo ""
    echo "=========================================="
    echo "Trials Completed"
    echo "=========================================="

    # Count Phase 1 trials
    if [ -d "train_results/phase1" ]; then
        PHASE1_COUNT=$(ls -1 train_results/phase1/ 2>/dev/null | wc -l)
        echo "Phase 1: $PHASE1_COUNT trials"
    fi

    # Count Phase 2 trials
    if [ -d "train_results/phase2_ppo" ]; then
        PHASE2_PPO_COUNT=$(ls -1 train_results/phase2_ppo/ 2>/dev/null | wc -l)
        echo "Phase 2 PPO: $PHASE2_PPO_COUNT trials"
    fi

    if [ -d "train_results/phase2_ddqn" ]; then
        PHASE2_DDQN_COUNT=$(ls -1 train_results/phase2_ddqn/ 2>/dev/null | wc -l)
        echo "Phase 2 DDQN: $PHASE2_DDQN_COUNT trials"
    fi

    echo ""
    echo "=========================================="
    echo "Latest Trial Directories"
    echo "=========================================="

    if [ -d "train_results/phase1" ]; then
        echo "Phase 1:"
        ls -1t train_results/phase1/ 2>/dev/null | head -3 | sed 's/^/  /'
    fi

    if [ -d "train_results/phase2_ppo" ]; then
        echo "Phase 2 PPO:"
        ls -1t train_results/phase2_ppo/ 2>/dev/null | head -3 | sed 's/^/  /'
    fi

    echo ""
    echo "=========================================="
    echo "Results Files"
    echo "=========================================="

    if [ -f "phase1_winner.json" ]; then
        echo "✓ phase1_winner.json"
        WINNER_TF=$(python3 -c "import json; print(json.load(open('phase1_winner.json'))['timeframe'])" 2>/dev/null || echo "?")
        WINNER_INT=$(python3 -c "import json; print(json.load(open('phase1_winner.json'))['interval'])" 2>/dev/null || echo "?")
        echo "  Winner: $WINNER_TF @ $WINNER_INT"
    else
        echo "○ phase1_winner.json: Not yet"
    fi

    if [ -f "phase2_comparison.json" ]; then
        echo "✓ phase2_comparison.json"
        WINNER_ALG=$(python3 -c "import json; print(json.load(open('phase2_comparison.json'))['winner'])" 2>/dev/null || echo "?")
        echo "  Winner: $WINNER_ALG"
    else
        echo "○ phase2_comparison.json: Not yet"
    fi

    if [ -f "two_phase_training_report.json" ]; then
        echo "✓ two_phase_training_report.json"
    else
        echo "○ two_phase_training_report.json: Not yet"
    fi

    echo ""
    echo "Refreshing in 5 seconds... (Ctrl+C to stop)"

    sleep 5
done
