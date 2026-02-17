#!/bin/bash
# Show current training progress and latest trial logs

echo "========================================"
echo "Two-Phase Training Progress"
echo "========================================"
echo ""

# Function to show trial count and latest
show_phase_status() {
    local phase_name=$1
    local phase_dir=$2

    if [ -d "$phase_dir" ]; then
        local count=$(ls -1 "$phase_dir" 2>/dev/null | wc -l)
        echo "$phase_name: $count trials completed"

        # Show latest trial directory
        local latest=$(ls -1t "$phase_dir" 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "  Latest: $latest"

            # Check if it has a log file
            if [ -f "$phase_dir/$latest/log.txt" ]; then
                echo "  Log (last 10 lines):"
                tail -n 10 "$phase_dir/$latest/log.txt" 2>/dev/null | sed 's/^/    /'
            fi
        fi
    else
        echo "$phase_name: No trials yet"
    fi
    echo ""
}

# Check Phase 1
show_phase_status "Phase 1" "train_results/phase1"

# Check Phase 2 PPO
show_phase_status "Phase 2 PPO" "train_results/phase2_ppo"

# Check Phase 2 DDQN
show_phase_status "Phase 2 DDQN" "train_results/phase2_ddqn"

echo "========================================"
echo "Optuna Study Databases"
echo "========================================"
echo ""

if [ -f "databases/phase1_optuna.db" ]; then
    SIZE=$(du -h databases/phase1_optuna.db | cut -f1)
    echo "✓ Phase 1 DB: $SIZE"
else
    echo "○ Phase 1 DB: Not created yet"
fi

if [ -f "databases/phase2_ppo_optuna.db" ]; then
    SIZE=$(du -h databases/phase2_ppo_optuna.db | cut -f1)
    echo "✓ Phase 2 PPO DB: $SIZE"
else
    echo "○ Phase 2 PPO DB: Not created yet"
fi

if [ -f "databases/phase2_ddqn_optuna.db" ]; then
    SIZE=$(du -h databases/phase2_ddqn_optuna.db | cut -f1)
    echo "✓ Phase 2 DDQN DB: $SIZE"
else
    echo "○ Phase 2 DDQN DB: Not created yet"
fi

echo ""
echo "========================================"
echo "Running Processes"
echo "========================================"
echo ""

ps aux | grep -E "phase1_timeframe|phase2_feature|run_two_phase" | grep -v grep | while read line; do
    echo "$line"
done || echo "No training processes running"

echo ""
echo "========================================"
echo "Quick Commands"
echo "========================================"
echo ""
echo "Monitor in real-time:"
echo "  ./debug_monitor_training.sh"
echo ""
echo "View Optuna study:"
echo "  python -c \"import optuna; study = optuna.load_study(study_name='phase1_5d_1h', storage='sqlite:///databases/phase1_optuna.db'); print(f'Best value: {study.best_value:.4f}')\""
echo ""
echo "Kill training:"
echo "  pkill -f phase1_timeframe_optimizer"
echo ""
