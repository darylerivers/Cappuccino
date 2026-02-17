#!/bin/bash
# Start Entire Cappuccino System - VISIBLE MODE
# Launches training, pipeline, and automation in separate visible windows

cd /opt/user-data/experiment/cappuccino

# Load configuration
source .env.training

# Detect terminal emulator
if command -v kitty &> /dev/null; then
    TERMINAL="kitty"
elif command -v gnome-terminal &> /dev/null; then
    TERMINAL="gnome-terminal --"
elif command -v konsole &> /dev/null; then
    TERMINAL="konsole -e"
elif command -v alacritty &> /dev/null; then
    TERMINAL="alacritty -e"
else
    TERMINAL="xterm -e"
fi

echo "=========================================="
echo "CAPPUCCINO COMPLETE SYSTEM LAUNCHER"
echo "=========================================="
echo "Study: $ACTIVE_STUDY_NAME"
echo ""
echo "Launching all components in separate windows..."
echo ""

# 1. PIPELINE WINDOW
cat > /tmp/run_pipeline.sh << 'EOF'
#!/bin/bash
cd /opt/user-data/experiment/cappuccino
echo "=========================================="
echo "CAPPUCCINO PIPELINE V2"
echo "=========================================="
echo "Monitoring for completed trials..."
echo "Auto-deploying to paper trading..."
echo ""
python -u scripts/training/pipeline_v2.py --daemon 2>&1 | tee logs/pipeline_v2_visible.log
read -p "Press Enter to close..."
EOF
chmod +x /tmp/run_pipeline.sh

if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "Pipeline V2" bash /tmp/run_pipeline.sh &
else
    $TERMINAL bash /tmp/run_pipeline.sh &
fi
echo "✓ Pipeline window opened"
sleep 1

# 2. AUTO-MODEL DEPLOYER WINDOW
cat > /tmp/run_deployer.sh << EOF
#!/bin/bash
cd /opt/user-data/experiment/cappuccino
source .env.training
echo "=========================================="
echo "AUTO-MODEL DEPLOYER"
echo "=========================================="
echo "Study: \$ACTIVE_STUDY_NAME"
echo ""
python -u scripts/deployment/auto_model_deployer.py --study "\$ACTIVE_STUDY_NAME" --daemon 2>&1 | tee logs/auto_deployer_visible.log
read -p "Press Enter to close..."
EOF
chmod +x /tmp/run_deployer.sh

if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "Auto Deployer" bash /tmp/run_deployer.sh &
else
    $TERMINAL bash /tmp/run_deployer.sh &
fi
echo "✓ Auto Deployer window opened"
sleep 1

# 3. SYSTEM WATCHDOG WINDOW
cat > /tmp/run_watchdog.sh << 'EOF'
#!/bin/bash
cd /opt/user-data/experiment/cappuccino
echo "=========================================="
echo "SYSTEM WATCHDOG"
echo "=========================================="
echo "Monitoring system health..."
echo ""
python -u monitoring/system_watchdog.py --check-interval 60 --max-restarts 3 2>&1 | tee logs/watchdog_visible.log
read -p "Press Enter to close..."
EOF
chmod +x /tmp/run_watchdog.sh

if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "System Watchdog" bash /tmp/run_watchdog.sh &
else
    $TERMINAL bash /tmp/run_watchdog.sh &
fi
echo "✓ Watchdog window opened"
sleep 1

# 4. PERFORMANCE MONITOR WINDOW
cat > /tmp/run_monitor.sh << EOF
#!/bin/bash
cd /opt/user-data/experiment/cappuccino
source .env.training
echo "=========================================="
echo "PERFORMANCE MONITOR"
echo "=========================================="
echo "Study: \$ACTIVE_STUDY_NAME"
echo ""
python -u monitoring/performance_monitor.py --study "\$ACTIVE_STUDY_NAME" --check-interval 300 2>&1 | tee logs/performance_monitor_visible.log
read -p "Press Enter to close..."
EOF
chmod +x /tmp/run_monitor.sh

if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "Performance Monitor" bash /tmp/run_monitor.sh &
else
    $TERMINAL bash /tmp/run_monitor.sh &
fi
echo "✓ Performance Monitor window opened"
sleep 1

# 5. ENSEMBLE AUTO-UPDATER WINDOW
cat > /tmp/run_ensemble.sh << EOF
#!/bin/bash
cd /opt/user-data/experiment/cappuccino
source .env.training
echo "=========================================="
echo "ENSEMBLE AUTO-UPDATER"
echo "=========================================="
echo "Study: \$ACTIVE_STUDY_NAME"
echo ""
python -u models/ensemble_auto_updater.py --study "\$ACTIVE_STUDY_NAME" --ensemble-dir train_results/ensemble --top-n 20 --interval 600 2>&1 | tee logs/ensemble_updater_visible.log
read -p "Press Enter to close..."
EOF
chmod +x /tmp/run_ensemble.sh

if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "Ensemble Updater" bash /tmp/run_ensemble.sh &
else
    $TERMINAL bash /tmp/run_ensemble.sh &
fi
echo "✓ Ensemble Updater window opened"
sleep 2

echo ""
echo "=========================================="
echo "All automation windows launched!"
echo "=========================================="
echo ""
echo "Now starting training in THIS window..."
echo "Study: $ACTIVE_STUDY_NAME"
echo "Workers: 3"
echo ""

# START TRAINING WORKERS IN CURRENT WINDOW
for i in {1..3}; do
  echo "Starting worker $i..."
  python -u scripts/training/1_optimize_unified.py --n-trials 1000 --gpu 0 \
    --study-name "$ACTIVE_STUDY_NAME" > logs/worker_$i.log 2>&1 &
  WORKER_PIDS[$i]=$!
  echo "  PID: ${WORKER_PIDS[$i]}"
  sleep 3
done

echo ""
echo "=========================================="
echo "COMPLETE SYSTEM RUNNING"
echo "=========================================="
echo ""
echo "Training Workers: ${WORKER_PIDS[@]}"
echo ""
echo "Windows Opened:"
echo "  • Pipeline V2 (auto-deployment)"
echo "  • Auto-Model Deployer"
echo "  • System Watchdog"
echo "  • Performance Monitor"
echo "  • Ensemble Auto-Updater"
echo ""
echo "Monitoring worker 1 log..."
echo "(Ctrl+C to exit view, all processes continue running)"
echo "=========================================="
echo ""

# Stream first worker's log in current window
tail -f logs/worker_1.log
