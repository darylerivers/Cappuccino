#!/bin/bash
# Monitor test training and auto-resume main training when done

cd /opt/user-data/experiment/cappuccino

echo "🔍 Monitoring test_run_huge training..."
echo "   Will auto-start cappuccino_ft_transformer when test completes"
echo ""

# Wait for test training to complete
while pgrep -f "study-name test_run_huge" > /dev/null; do
    # Get progress
    progress=$(python -c "
import sqlite3
conn = sqlite3.connect('databases/optuna_cappuccino.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT COUNT(*) FROM trials t
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = 'test_run_huge' AND t.state = 'COMPLETE'
''')
print(cursor.fetchone()[0])
conn.close()
" 2>/dev/null || echo "?")

    echo "[$(date +%H:%M:%S)] Test training: $progress/10 trials complete..."
    sleep 60
done

echo ""
echo "✓ Test training complete!"
echo "🚀 Starting main training: cappuccino_ft_transformer"
echo ""

# Start main training
nohup ./resume_main_training.sh > training_main.log 2>&1 &

echo "✓ Main training started!"
echo "  Log: training_main.log"
echo "  Monitor with: tail -f training_main.log"
