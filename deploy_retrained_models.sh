#!/bin/bash
# Deploy retrained models to replace broken trials 91 and 100

set -e
cd /opt/user-data/experiment/cappuccino
source activate_rocm_env.sh

echo "========================================================================"
echo "DEPLOYING RETRAINED MODELS"
echo "========================================================================"
echo ""

# Get best 2 trials from retraining study
echo "Finding best 2 trials from paper_traders_retrain study..."

python3 << 'EOF'
import sqlite3
import json
from pathlib import Path

conn = sqlite3.connect('databases/optuna_cappuccino.db')
cursor = conn.cursor()

# Get best 2 completed trials
cursor.execute("""
    SELECT t.number, t.value
    FROM trials t
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name = 'paper_traders_retrain'
    AND t.state = 'COMPLETE'
    ORDER BY t.value DESC
    LIMIT 2
""")

results = cursor.fetchall()

if len(results) < 2:
    print(f"ERROR: Only {len(results)} trials completed. Need at least 2.")
    exit(1)

trial1, sharpe1 = results[0]
trial2, sharpe2 = results[1]

print(f"\nBest trials:")
print(f"  Trial #{trial1}: Sharpe {sharpe1:.4f}")
print(f"  Trial #{trial2}: Sharpe {sharpe2:.4f}")

# Save for bash script
with open('/tmp/retrain_trials.json', 'w') as f:
    json.dump({
        'trial1': trial1,
        'trial2': trial2,
        'sharpe1': sharpe1,
        'sharpe2': sharpe2
    }, f)

conn.close()
EOF

if [ $? -ne 0 ]; then
    echo "✗ Failed to find completed trials"
    exit 1
fi

# Load trial numbers
TRIAL1=$(python3 -c "import json; print(json.load(open('/tmp/retrain_trials.json'))['trial1'])")
TRIAL2=$(python3 -c "import json; print(json.load(open('/tmp/retrain_trials.json'))['trial2'])")

echo ""
echo "Deploying:"
echo "  Trial #91  ← Training trial #$TRIAL1"
echo "  Trial #100 ← Training trial #$TRIAL2"
echo ""

# Create deployment directories
mkdir -p deployments/model_91_new
mkdir -p deployments/model_100_new

# Deploy trial 1 to model 91
echo "Deploying trial #$TRIAL1 to model_91_new..."
python3 << EOF
import shutil
import json
from pathlib import Path

trial_file = Path('databases/trials/trial_${TRIAL1}.pkl')
target_dir = Path('deployments/model_91_new')

# Copy trial file as best_trial
if trial_file.exists():
    shutil.copy(trial_file, target_dir / 'best_trial')

    # Create metadata
    meta = {
        'trial_number': ${TRIAL1},
        'study': 'paper_traders_retrain',
        'deployed_for': 'trial_91_replacement',
        'compatible': True
    }
    with open(target_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("✓ Deployed trial #${TRIAL1} to model_91_new")
else:
    print(f"✗ Trial file not found: {trial_file}")
    exit(1)
EOF

# Deploy trial 2 to model 100
echo "Deploying trial #$TRIAL2 to model_100_new..."
python3 << EOF
import shutil
import json
from pathlib import Path

trial_file = Path('databases/trials/trial_${TRIAL2}.pkl')
target_dir = Path('deployments/model_100_new')

# Copy trial file as best_trial
if trial_file.exists():
    shutil.copy(trial_file, target_dir / 'best_trial')

    # Create metadata
    meta = {
        'trial_number': ${TRIAL2},
        'study': 'paper_traders_retrain',
        'deployed_for': 'trial_100_replacement',
        'compatible': True
    }
    with open(target_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("✓ Deployed trial #${TRIAL2} to model_100_new")
else:
    print(f"✗ Trial file not found: {trial_file}")
    exit(1)
EOF

echo ""
echo "========================================================================"
echo "DEPLOYMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "New models ready:"
echo "  deployments/model_91_new  (trial #$TRIAL1)"
echo "  deployments/model_100_new (trial #$TRIAL2)"
echo ""
echo "To activate them:"
echo "  1. Stop current broken traders: pkill -f 'paper_trader.*model_[01]'"
echo "  2. Update restart script to use new directories"
echo "  3. Start new traders: ./restart_all_traders.sh"
echo ""
echo "Or run: ./activate_new_models.sh (I'll create this next)"
echo "========================================================================"
