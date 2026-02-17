#!/bin/bash
# Sync Ensemble and Automation to Training Study
# This script ensures ensemble and automation use the same study as active training

set -e

STUDY_NAME="$1"

if [ -z "$STUDY_NAME" ]; then
    echo "Usage: ./sync_training_study.sh <study_name>"
    echo ""
    echo "Example: ./sync_training_study.sh cappuccino_1year_20251121"
    echo ""
    echo "This will:"
    echo "  1. Update ensemble with top 10 trials from the study"
    echo "  2. Update automation scripts to use this study"
    echo "  3. Restart automation systems"
    exit 1
fi

echo "=========================================="
echo "Syncing to Study: $STUDY_NAME"
echo "=========================================="
echo ""

# Step 1: Update ensemble
echo "[1/4] Updating ensemble with top 10 trials from $STUDY_NAME..."
python ensemble_auto_updater.py \
    --study "$STUDY_NAME" \
    --ensemble-dir train_results/ensemble \
    --top-n 10 \
    --once

if [ $? -ne 0 ]; then
    echo "Error: Failed to update ensemble"
    exit 1
fi
echo ""

# Step 2: Fix ensemble manifest (add required fields)
echo "[2/4] Fixing ensemble manifest..."
python -c "
import json
from pathlib import Path
import sqlite3

# Load current manifest
manifest_path = Path('train_results/ensemble/ensemble_manifest.json')
with open(manifest_path) as f:
    manifest = json.load(f)

# Add missing fields
trial_paths = []
actor_paths = []
for trial_num in manifest['trial_numbers']:
    trial_path = f'train_results/cwd_tests/trial_{trial_num}_1h'
    actor_path = f'train_results/cwd_tests/trial_{trial_num}_1h/actor.pth'
    trial_paths.append(trial_path)
    actor_paths.append(actor_path)

# Get trial values from DB
from_db = {}
conn = sqlite3.connect('databases/optuna_cappuccino.db')
cursor = conn.cursor()
for trial_num in manifest['trial_numbers']:
    cursor.execute('''
        SELECT tv.value FROM trial_values tv
        JOIN trials t ON tv.trial_id = t.trial_id
        WHERE t.number = ? AND t.study_id = (
            SELECT study_id FROM studies WHERE study_name = ?
        )
    ''', (trial_num, manifest['study_name']))
    result = cursor.fetchone()
    if result:
        from_db[trial_num] = result[0]
conn.close()

manifest['trial_paths'] = trial_paths
manifest['actor_paths'] = actor_paths
manifest['trial_values'] = [from_db.get(t, 0.0) for t in manifest['trial_numbers']]
manifest['worst_value'] = min(manifest['trial_values']) if manifest['trial_values'] else 0

# Save updated manifest
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'✓ Manifest updated: {len(manifest[\"trial_numbers\"])} models')
print(f'  Best value: {max(manifest[\"trial_values\"]):.6f}')
print(f'  Mean value: {manifest[\"mean_value\"]:.6f}')
"

if [ $? -ne 0 ]; then
    echo "Error: Failed to fix manifest"
    exit 1
fi
echo ""

# Step 3: Update automation scripts
echo "[3/4] Updating automation scripts to use $STUDY_NAME..."

# Update start_automation.sh
sed -i "s/--study [a-zA-Z0-9_]*/--study $STUDY_NAME/g" start_automation.sh

echo "✓ Updated start_automation.sh"
echo ""

# Step 4: Restart automation (if running)
echo "[4/4] Restarting automation systems..."

# Check if automation is running
if pgrep -f "auto_model_deployer.py" > /dev/null; then
    echo "Stopping existing automation..."
    ./stop_automation.sh
    sleep 3
    echo ""
    echo "Starting automation with new study..."
    ./start_automation.sh
    echo ""
else
    echo "Automation not running. Start it with: ./start_automation.sh"
    echo ""
fi

echo "=========================================="
echo "✓ Sync Complete!"
echo "=========================================="
echo ""
echo "Study: $STUDY_NAME"
echo "Ensemble: train_results/ensemble"
echo ""
echo "Next steps:"
echo "  - Paper trading will use this ensemble"
echo "  - Ensemble auto-updater will keep it in sync (every 10 min)"
echo "  - Auto-deployer monitors for improvements (every hour)"
echo ""
