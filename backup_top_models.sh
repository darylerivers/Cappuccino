#!/bin/bash
# Backup Top Model Files
# Prevents loss of best performing models

set -e

# Load configuration
if [ -f ".env.training" ]; then
    source .env.training
else
    echo "ERROR: .env.training not found"
    exit 1
fi

DB="databases/optuna_cappuccino.db"
BACKUP_DIR="model_backups/${ACTIVE_STUDY_NAME}"
BACKUP_TOP_N=50
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Model Backup System"
echo "=========================================="
echo "Study:      $ACTIVE_STUDY_NAME"
echo "Backup Top: $BACKUP_TOP_N models"
echo "Backup Dir: $BACKUP_DIR"
echo "Time:       $TIMESTAMP"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Get top N trial numbers and their values
echo "Querying database for top $BACKUP_TOP_N trials..."
TRIALS=$(sqlite3 "$DB" "
SELECT t.number, tv.value
FROM trials t
JOIN studies s ON t.study_id = s.study_id
JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE s.study_name = '$ACTIVE_STUDY_NAME'
  AND t.state = 'COMPLETE'
ORDER BY tv.value DESC
LIMIT $BACKUP_TOP_N
")

if [ -z "$TRIALS" ]; then
    echo "No trials found for study: $ACTIVE_STUDY_NAME"
    exit 0
fi

# Count trials found
TRIAL_COUNT=$(echo "$TRIALS" | wc -l)
echo "Found $TRIAL_COUNT trials to backup"
echo ""

# Backup each trial
BACKED_UP=0
MISSING=0

echo "$TRIALS" | while IFS='|' read -r trial_num value; do
    # Try different possible locations
    for trial_dir in "train_results/cwd_tests/trial_${trial_num}_1h" "train_results/trial_${trial_num}_1h"; do
        if [ -d "$trial_dir" ]; then
            # Backup actor.pth (main model)
            if [ -f "$trial_dir/actor.pth" ]; then
                DEST="$BACKUP_DIR/trial_${trial_num}_value_${value}_actor.pth"
                cp "$trial_dir/actor.pth" "$DEST"
                echo "✓ Backed up trial $trial_num (value: $value)"
                BACKED_UP=$((BACKED_UP + 1))

                # Also backup best_trial metadata if it exists
                if [ -f "$trial_dir/best_trial" ]; then
                    cp "$trial_dir/best_trial" "$BACKUP_DIR/trial_${trial_num}_best_trial"
                fi

                break
            fi
        fi
    done

    # Check if we found the model
    if [ ! -f "$BACKUP_DIR/trial_${trial_num}_value_${value}_actor.pth" ]; then
        echo "✗ Trial $trial_num - model files not found"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "=========================================="
echo "Backup Summary"
echo "=========================================="
echo "Backed up: $BACKED_UP models"
echo "Missing:   $MISSING models"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""

# Create backup manifest
cat > "$BACKUP_DIR/backup_manifest_${TIMESTAMP}.txt" << EOF
Backup Manifest
Created: $TIMESTAMP
Study: $ACTIVE_STUDY_NAME
Top N: $BACKUP_TOP_N
Backed Up: $BACKED_UP
Missing: $MISSING

Backed up trials:
$TRIALS
EOF

echo "✓ Backup complete!"
echo "Manifest: $BACKUP_DIR/backup_manifest_${TIMESTAMP}.txt"
