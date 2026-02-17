#!/bin/bash
# Start Automation Systems - VISIBLE MODE in NEW WINDOWS
# Opens separate terminal windows for each automation component

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
echo "STARTING AUTOMATION SYSTEMS - VISIBLE MODE"
echo "=========================================="
echo "Study: $ACTIVE_STUDY_NAME"
echo "Opening terminal windows for each component..."
echo ""

# 1. Auto-Model Deployer Window
cat > /tmp/run_deployer.sh << EOF
#!/bin/bash
cd /opt/user-data/experiment/cappuccino
source .env.training
echo "=========================================="
echo "AUTO-MODEL DEPLOYER"
echo "=========================================="
echo "Study: \$ACTIVE_STUDY_NAME"
echo "Monitoring for new best models..."
echo ""
python -u auto_model_deployer.py --study "\$ACTIVE_STUDY_NAME" --daemon 2>&1 | tee logs/auto_deployer_visible.log
read -p "Press Enter to close..."
EOF
chmod +x /tmp/run_deployer.sh

if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "Auto Deployer" bash /tmp/run_deployer.sh &
elif [ "$TERMINAL" = "gnome-terminal --" ]; then
    gnome-terminal --title="Auto Deployer" -- bash /tmp/run_deployer.sh &
else
    $TERMINAL bash /tmp/run_deployer.sh &
fi
echo "✓ Auto-Model Deployer window opened"
sleep 1

# 2. System Watchdog Window
cat > /tmp/run_watchdog.sh << 'EOF'
#!/bin/bash
cd /opt/user-data/experiment/cappuccino
echo "=========================================="
echo "SYSTEM WATCHDOG"
echo "=========================================="
echo "Monitoring system health..."
echo "Auto-restarting failed services..."
echo ""
python -u system_watchdog.py --check-interval 60 --max-restarts 3 2>&1 | tee logs/watchdog_visible.log
read -p "Press Enter to close..."
EOF
chmod +x /tmp/run_watchdog.sh

if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "System Watchdog" bash /tmp/run_watchdog.sh &
elif [ "$TERMINAL" = "gnome-terminal --" ]; then
    gnome-terminal --title="System Watchdog" -- bash /tmp/run_watchdog.sh &
else
    $TERMINAL bash /tmp/run_watchdog.sh &
fi
echo "✓ System Watchdog window opened"
sleep 1

# 3. Performance Monitor Window
cat > /tmp/run_monitor.sh << EOF
#!/bin/bash
cd /opt/user-data/experiment/cappuccino
source .env.training
echo "=========================================="
echo "PERFORMANCE MONITOR"
echo "=========================================="
echo "Study: \$ACTIVE_STUDY_NAME"
echo "Tracking trial performance..."
echo ""
python -u performance_monitor.py --study "\$ACTIVE_STUDY_NAME" --check-interval 300 2>&1 | tee logs/performance_monitor_visible.log
read -p "Press Enter to close..."
EOF
chmod +x /tmp/run_monitor.sh

if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "Performance Monitor" bash /tmp/run_monitor.sh &
elif [ "$TERMINAL" = "gnome-terminal --" ]; then
    gnome-terminal --title="Performance Monitor" -- bash /tmp/run_monitor.sh &
else
    $TERMINAL bash /tmp/run_monitor.sh &
fi
echo "✓ Performance Monitor window opened"
sleep 1

# 4. Ensemble Auto-Updater Window
cat > /tmp/run_ensemble.sh << EOF
#!/bin/bash
cd /opt/user-data/experiment/cappuccino
source .env.training
echo "=========================================="
echo "ENSEMBLE AUTO-UPDATER"
echo "=========================================="
echo "Study: \$ACTIVE_STUDY_NAME"
echo "Updating ensemble with top models..."
echo ""
python -u ensemble_auto_updater.py --study "\$ACTIVE_STUDY_NAME" --ensemble-dir train_results/ensemble --top-n 20 --interval 600 2>&1 | tee logs/ensemble_updater_visible.log
read -p "Press Enter to close..."
EOF
chmod +x /tmp/run_ensemble.sh

if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "Ensemble Updater" bash /tmp/run_ensemble.sh &
elif [ "$TERMINAL" = "gnome-terminal --" ]; then
    gnome-terminal --title="Ensemble Updater" -- bash /tmp/run_ensemble.sh &
else
    $TERMINAL bash /tmp/run_ensemble.sh &
fi
echo "✓ Ensemble Auto-Updater window opened"

echo ""
echo "=========================================="
echo "All automation windows opened!"
echo "=========================================="
echo ""
echo "Windows:"
echo "  1. Auto-Model Deployer"
echo "  2. System Watchdog"
echo "  3. Performance Monitor"
echo "  4. Ensemble Auto-Updater"
echo ""
