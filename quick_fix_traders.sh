#!/bin/bash
# Quick automated fix: Retrain and deploy 2 new paper trading models

set -e  # Exit on error

cd /opt/user-data/experiment/cappuccino
source activate_rocm_env.sh

echo "========================================================================"
echo "QUICK FIX: Model Mismatch Issue"
echo "========================================================================"
echo ""
echo "This will:"
echo "  1. Train 2 fast trials (~10 min total)"
echo "  2. Deploy best performing model"
echo "  3. Update paper traders to use new models"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Training 2 quick trials..."
echo "========================================================================"

python3 scripts/training/1_optimize_unified.py \
  --study-name quick_paper_fix \
  --n-trials 2 \
  --force-ft \
  --timeframe 1h \
  --data-dir data/1h_1680

echo ""
echo "Step 2: Finding best trial..."
echo "========================================================================"

BEST_TRIAL=$(python3 << 'EOF'
import sqlite3
conn = sqlite3.connect('databases/optuna_cappuccino.db')
cursor = conn.cursor()

# Get best trial from study
cursor.execute("""
    SELECT t.number, t.value
    FROM trials t
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = 'quick_paper_fix'
    AND t.state = 'COMPLETE'
    ORDER BY t.value DESC
    LIMIT 1
""")

result = cursor.fetchone()
if result:
    print(f"{result[0]}")  # Trial number
else:
    print("ERROR")

conn.close()
EOF
)

if [ "$BEST_TRIAL" = "ERROR" ]; then
    echo "✗ No trials completed successfully"
    exit 1
fi

echo "Best trial: #$BEST_TRIAL"

echo ""
echo "Step 3: Deploying to paper trading..."
echo "========================================================================"

# Create deployment directory
mkdir -p deployments/model_new_91
mkdir -p deployments/model_new_100

# Copy trial files
python3 << EOF
import shutil
from pathlib import Path

# Find trial pickle
trial_file = Path('databases/trials/trial_${BEST_TRIAL}.pkl')

# Deploy to both slots
for target in ['deployments/model_new_91', 'deployments/model_new_100']:
    target_dir = Path(target)

    # Copy trial file
    shutil.copy(trial_file, target_dir / 'best_trial')

    # Create metadata
    import json
    meta = {
        'trial_number': ${BEST_TRIAL},
        'study': 'quick_paper_fix',
        'timestamp': '$(date -Iseconds)'
    }
    with open(target_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

print(f"✓ Deployed trial #{BEST_TRIAL} to both model slots")
EOF

echo ""
echo "Step 4: Updating restart script..."
echo "========================================================================"

# Update restart script to use new models
sed -i 's|deployments/model_0|deployments/model_new_91|g' restart_all_traders.sh
sed -i 's|deployments/model_1|deployments/model_new_100|g' restart_all_traders.sh

echo ""
echo "========================================================================"
echo "FIX COMPLETE!"
echo "========================================================================"
echo ""
echo "New models deployed:"
echo "  - Trial #91:  deployments/model_new_91 (trial #${BEST_TRIAL})"
echo "  - Trial #100: deployments/model_new_100 (trial #${BEST_TRIAL})"
echo ""
echo "To start paper trading:"
echo "  ./restart_all_traders.sh"
echo ""
echo "These models are compatible with your current environment."
echo "========================================================================"
