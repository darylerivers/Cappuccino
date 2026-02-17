#!/usr/bin/env bash
#
# Quick Start - Automated Training Pipeline
#
# This script demonstrates the key commands for the automated training system.
# Choose one of the options below based on what you want to do.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Cappuccino Automated Training System${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to display menu
show_menu() {
    echo -e "${GREEN}Select an option:${NC}"
    echo ""
    echo "  1) Run Full Automated Pipeline (clean, train, archive, deploy)"
    echo "  2) Start Training Only (background mode)"
    echo "  3) Archive Existing Trials"
    echo "  4) Deploy Best Model to Paper Trading"
    echo "  5) Show Trial Dashboard"
    echo "  6) List Archived Trials"
    echo "  7) Clean Old Data (logs and trials)"
    echo "  8) Show Training Worker Status"
    echo "  9) Stop All Training Workers"
    echo "  0) Exit"
    echo ""
}

# Function to run full pipeline
run_full_pipeline() {
    echo -e "${YELLOW}Running full automated pipeline...${NC}"
    echo ""
    echo "This will:"
    echo "  • Clean old logs and trials"
    echo "  • Start 3 training workers"
    echo "  • Archive top 10% of trials"
    echo "  • Deploy best model"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/automation/automated_training_pipeline.py --mode full
    fi
}

# Function to start training
start_training() {
    echo -e "${YELLOW}Starting training in background mode...${NC}"
    echo ""
    python scripts/automation/automated_training_pipeline.py \
        --mode training \
        --background \
        --workers 3 \
        --trials 500
    echo ""
    echo -e "${GREEN}✅ Training started!${NC}"
    echo ""
    echo "Monitor progress with:"
    echo "  tail -f logs/worker_auto_*.log"
    echo "  OR"
    echo "  python scripts/automation/trial_dashboard.py"
}

# Function to archive trials
archive_trials() {
    echo -e "${YELLOW}Archiving top trials...${NC}"
    echo ""
    python scripts/automation/automated_training_pipeline.py --mode archive
}

# Function to deploy best model
deploy_model() {
    echo -e "${YELLOW}Deploying best model...${NC}"
    echo ""
    python scripts/automation/automated_training_pipeline.py --mode deploy --deployment-slot 0
}

# Function to show dashboard
show_dashboard() {
    echo -e "${YELLOW}Trial Dashboard Options:${NC}"
    echo ""
    echo "  1) Live Dashboard (auto-refresh)"
    echo "  2) Show Current Statistics"
    echo "  3) Show Top 10 Trials"
    echo "  0) Back to main menu"
    echo ""
    read -p "Select option: " dash_choice

    case $dash_choice in
        1)
            echo ""
            echo -e "${YELLOW}Launching live dashboard...${NC}"
            echo "Press Ctrl+C to exit dashboard"
            sleep 2
            python scripts/automation/trial_dashboard.py
            ;;
        2)
            echo ""
            echo -e "${YELLOW}Current Training Statistics:${NC}"
            echo ""
            python -c "
import sqlite3
from pathlib import Path

db_path = Path('databases/optuna_cappuccino.db')
if not db_path.exists():
    print('⚠️  No database found')
    exit(0)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get current study
try:
    with open('.current_study', 'r') as f:
        study_name = f.read().strip()
    cursor.execute('SELECT study_id FROM studies WHERE study_name = ?', (study_name,))
    study_row = cursor.fetchone()
    if not study_row:
        print(f'⚠️  Study not found: {study_name}')
        exit(0)
    study_id = study_row[0]
except:
    cursor.execute('SELECT study_id, study_name FROM studies ORDER BY study_id DESC LIMIT 1')
    study_row = cursor.fetchone()
    if not study_row:
        print('⚠️  No studies found')
        exit(0)
    study_id, study_name = study_row

print(f'Study: {study_name}')
print('=' * 70)

# Get statistics (including running trials)
cursor.execute('''
    SELECT
        SUM(CASE WHEN t.state = \"COMPLETE\" THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN t.state = \"RUNNING\" THEN 1 ELSE 0 END) as running,
        AVG(CASE WHEN t.state = \"COMPLETE\" THEN tv.value END) as avg_val,
        MIN(CASE WHEN t.state = \"COMPLETE\" THEN tv.value END) as min_val,
        MAX(CASE WHEN t.state = \"COMPLETE\" THEN tv.value END) as max_val
    FROM trials t
    LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
    WHERE t.study_id = ?
''', (study_id,))

completed, running, avg, min_val, max_val = cursor.fetchone()

if completed or running:
    if running:
        print(f'🔄 Running Trials:       {running}')
    if completed:
        print(f'✅ Completed Trials:     {completed}')
        print(f'📊 Average Sharpe:       {avg:.4f}' if avg else '📊 Average Sharpe:       N/A')
        print(f'📉 Min Sharpe:           {min_val:.4f}' if min_val else '📉 Min Sharpe:           N/A')
        print(f'📈 Max Sharpe:           {max_val:.4f}' if max_val else '📈 Max Sharpe:           N/A')
    else:
        print(f'⏳ Waiting for trials to complete...')
else:
    print(f'⏳ No trials started yet')

    # Grade distribution (only for completed trials)
    if completed and completed > 0:
        print()
        print('Grade Distribution (Completed Trials):')
        cursor.execute('''
            SELECT tv.value
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = \"COMPLETE\"
        ''', (study_id,))

        sharpes = [row[0] for row in cursor.fetchall() if row[0] is not None]
        grades = {'S': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}

        for s in sharpes:
            if s >= 0.30: grades['S'] += 1
            elif s >= 0.20: grades['A'] += 1
            elif s >= 0.15: grades['B'] += 1
            elif s >= 0.10: grades['C'] += 1
            elif s >= 0.05: grades['D'] += 1
            else: grades['F'] += 1

        for grade, count in grades.items():
            if count > 0:
                pct = (count / len(sharpes)) * 100
                bar = '█' * int(pct / 2)
                icon = {'S': '🏆', 'A': '⭐', 'B': '✅', 'C': '🔵', 'D': '⚠️', 'F': '❌'}
                print(f'  {icon[grade]} {grade}: {count:3d} ({pct:5.1f}%) {bar}')

conn.close()
"
            ;;
        3)
            echo ""
            echo -e "${YELLOW}Top 10 Trials:${NC}"
            echo ""
            python -c "
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from utils.trial_naming import generate_trial_vin

db_path = Path('databases/optuna_cappuccino.db')
if not db_path.exists():
    print('⚠️  No database found')
    exit(0)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get current study
try:
    with open('.current_study', 'r') as f:
        study_name = f.read().strip()
    cursor.execute('SELECT study_id FROM studies WHERE study_name = ?', (study_name,))
    study_row = cursor.fetchone()
    if not study_row:
        print(f'⚠️  Study not found: {study_name}')
        exit(0)
    study_id = study_row[0]
except:
    cursor.execute('SELECT study_id, study_name FROM studies ORDER BY study_id DESC LIMIT 1')
    study_row = cursor.fetchone()
    if not study_row:
        print('⚠️  No studies found')
        exit(0)
    study_id, study_name = study_row

print(f'Study: {study_name}')
print('=' * 80)

# Get top 10 trials
cursor.execute('''
    SELECT t.trial_id, t.number, tv.value, t.datetime_complete
    FROM trials t
    LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
    WHERE t.study_id = ? AND t.state = \"COMPLETE\" AND tv.value IS NOT NULL
    ORDER BY tv.value DESC
    LIMIT 10
''', (study_id,))

print(f'{\"Rank\":<6} {\"Trial\":<8} {\"VIN Code\":<50} {\"Grade\":<7} {\"Sharpe\":<10}')
print(f'{\"-\"*6} {\"-\"*8} {\"-\"*50} {\"-\"*7} {\"-\"*10}')

from datetime import datetime
for rank, row in enumerate(cursor.fetchall(), 1):
    trial_id, number, sharpe, complete_time = row

    # Get params for VIN
    cursor.execute('SELECT param_name, param_value FROM trial_params WHERE trial_id = ?', (trial_id,))
    params = {}
    for param_name, param_value in cursor.fetchall():
        if isinstance(param_value, (int, float)):
            params[param_name] = param_value
        else:
            try:
                params[param_name] = float(param_value) if '.' in str(param_value) else int(param_value)
            except:
                params[param_name] = param_value

    # Generate VIN
    vin, grade, _ = generate_trial_vin(
        'ppo',
        sharpe,
        params,
        datetime.fromisoformat(complete_time) if complete_time else datetime.now()
    )

    grade_icon = {'S': '🏆', 'A': '⭐', 'B': '✅', 'C': '🔵', 'D': '⚠️', 'F': '❌'}.get(grade, '?')

    print(f'{rank:<6} #{number:<7} {vin:<50} {grade_icon} {grade:<5} {sharpe:>8.4f}')

conn.close()
"
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
}

# Function to list archived trials
list_trials() {
    echo -e "${YELLOW}Archived trials:${NC}"
    echo ""
    python utils/trial_manager.py --list
}

# Function to clean old data
clean_data() {
    echo -e "${YELLOW}Cleaning old data...${NC}"
    echo ""
    echo "This will delete logs and trials older than 7 days."
    echo ""

    # Show what would be deleted
    echo -e "${BLUE}Preview (dry run):${NC}"
    python utils/trial_manager.py --clean-trials --clean-logs --dry-run
    echo ""

    read -p "Proceed with deletion? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python utils/trial_manager.py --clean-trials --clean-logs
        echo -e "${GREEN}✅ Cleanup complete!${NC}"
    fi
}

# Function to show worker status
show_workers() {
    echo -e "${YELLOW}Training worker status:${NC}"
    echo ""
    ps aux | grep "1_optimize_unified.py" | grep -v grep || echo "No workers running"
    echo ""

    if [ -f ".current_study" ]; then
        echo -e "${BLUE}Current study:${NC} $(cat .current_study)"
    fi
    echo ""

    echo "Recent logs:"
    ls -lt logs/worker_auto_*.log 2>/dev/null | head -3 || echo "No worker logs found"
}

# Function to stop workers
stop_workers() {
    echo -e "${RED}Stopping all training workers...${NC}"
    echo ""

    PIDS=$(ps aux | grep "1_optimize_unified.py" | grep -v grep | awk '{print $2}')

    if [ -z "$PIDS" ]; then
        echo "No workers running"
    else
        echo "Found workers with PIDs: $PIDS"
        read -p "Stop all workers? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$PIDS" | xargs kill
            echo -e "${GREEN}✅ Workers stopped${NC}"
        fi
    fi
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [0-9]: " choice
    echo ""

    case $choice in
        1) run_full_pipeline ;;
        2) start_training ;;
        3) archive_trials ;;
        4) deploy_model ;;
        5) show_dashboard ;;
        6) list_trials ;;
        7) clean_data ;;
        8) show_workers ;;
        9) stop_workers ;;
        0)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
    echo ""
done
