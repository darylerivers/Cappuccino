#!/bin/bash
# Start Pipeline - VISIBLE MODE in NEW WINDOW
# Opens a new terminal window to monitor pipeline

cd /opt/user-data/experiment/cappuccino

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
echo "STARTING PIPELINE V2 - VISIBLE MODE"
echo "=========================================="
echo "Opening new terminal window..."
echo ""

# Create a script to run in the new window
cat > /tmp/run_pipeline.sh << 'EOF'
#!/bin/bash
cd /opt/user-data/experiment/cappuccino

echo "=========================================="
echo "CAPPUCCINO PIPELINE V2"
echo "=========================================="
echo "Monitoring for completed trials..."
echo "Auto-deploying to paper trading..."
echo ""
echo "Press Ctrl+C to stop pipeline"
echo "=========================================="
echo ""

# Run pipeline in foreground (not daemon mode) so output is visible
python -u pipeline_v2.py --daemon 2>&1 | tee logs/pipeline_v2_visible.log

echo ""
echo "Pipeline stopped."
read -p "Press Enter to close this window..."
EOF

chmod +x /tmp/run_pipeline.sh

# Launch in new terminal window
if [ "$TERMINAL" = "kitty" ]; then
    kitty --title "Cappuccino Pipeline" bash /tmp/run_pipeline.sh &
elif [ "$TERMINAL" = "gnome-terminal --" ]; then
    gnome-terminal -- bash /tmp/run_pipeline.sh &
else
    $TERMINAL bash /tmp/run_pipeline.sh &
fi

echo "✓ Pipeline window opened"
echo ""
