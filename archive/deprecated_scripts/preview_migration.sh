#!/bin/bash
# Preview what will be migrated WITHOUT making any changes

CURRENT_STUDY="cappuccino_fresh_20251204_100527"
TRIALS_DIR="train_results/cwd_tests"

echo "=========================================="
echo "Migration Preview (No Changes Made)"
echo "=========================================="
echo ""

# Current disk usage
echo "Current Disk Usage:"
df -h /home | tail -1 | awk '{print "  /home (nvme1n1): " $3 " used / " $2 " total (" $5 " full)"}'
df -h / | tail -1 | awk '{print "  /     (nvme0n1): " $3 " used / " $2 " total (" $5 " full)"}'
echo ""

# Show trial distribution by study
echo "Trials by Study:"
sqlite3 databases/optuna_cappuccino.db <<EOF
SELECT
    s.study_name,
    COUNT(*) as total_trials,
    SUM(CASE WHEN t.state = 'COMPLETE' THEN 1 ELSE 0 END) as completed,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM trials), 1) as pct
FROM studies s
LEFT JOIN trials t ON s.study_id = t.study_id
GROUP BY s.study_name
ORDER BY total_trials DESC
LIMIT 10;
EOF
echo ""

# Estimate space by study
echo "Estimated Space by Study:"
for study in "cappuccino_1year_20251121" "cappuccino_trailing_20251125" "cappuccino_3workers_20251102_2325" "$CURRENT_STUDY"; do
    # Get trial numbers for this study
    TRIAL_NUMS=$(sqlite3 databases/optuna_cappuccino.db \
      "SELECT t.number FROM trials t
       JOIN studies s ON t.study_id = s.study_id
       WHERE s.study_name = '$study' AND t.state = 'COMPLETE'" 2>/dev/null)

    if [ -z "$TRIAL_NUMS" ]; then
        continue
    fi

    TRIAL_COUNT=$(echo "$TRIAL_NUMS" | wc -l)

    # Calculate total size for this study's trials
    TOTAL_SIZE=0
    for trial_num in $TRIAL_NUMS; do
        TRIAL_DIR="$TRIALS_DIR/trial_${trial_num}_1h"
        if [ -d "$TRIAL_DIR" ]; then
            SIZE=$(du -sb "$TRIAL_DIR" 2>/dev/null | awk '{print $1}')
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
        fi
    done

    # Convert to human readable
    SIZE_GB=$(echo "scale=2; $TOTAL_SIZE / 1024 / 1024 / 1024" | bc)
    echo "  $study: $TRIAL_COUNT trials, ~${SIZE_GB}GB"
done
echo ""

# Show what will be kept vs archived
echo "Migration Plan:"
echo "  KEEP on /home (nvme1n1):"
echo "    - $CURRENT_STUDY (current training)"
echo "    - Ensemble models (train_results/ensemble/)"
echo ""
echo "  ARCHIVE to / (nvme0n1):"
echo "    - cappuccino_1year_20251121 (old study, 1,344 trials)"
echo "    - cappuccino_trailing_20251125 (old study, 894 trials)"
echo "    - cappuccino_3workers_20251102_2325 (very old, 5,558 trials)"
echo ""

# Calculate potential savings
OLD_STUDIES=("cappuccino_1year_20251121" "cappuccino_trailing_20251125" "cappuccino_3workers_20251102_2325")
TOTAL_TO_ARCHIVE=0

for study in "${OLD_STUDIES[@]}"; do
    TRIAL_NUMS=$(sqlite3 databases/optuna_cappuccino.db \
      "SELECT t.number FROM trials t
       JOIN studies s ON t.study_id = s.study_id
       WHERE s.study_name = '$study'" 2>/dev/null)

    for trial_num in $TRIAL_NUMS; do
        TRIAL_DIR="$TRIALS_DIR/trial_${trial_num}_1h"
        if [ -d "$TRIAL_DIR" ]; then
            SIZE=$(du -sb "$TRIAL_DIR" 2>/dev/null | awk '{print $1}')
            TOTAL_TO_ARCHIVE=$((TOTAL_TO_ARCHIVE + SIZE))
        fi
    done
done

ARCHIVE_GB=$(echo "scale=1; $TOTAL_TO_ARCHIVE / 1024 / 1024 / 1024" | bc)

echo "Expected Space Freed on /home: ~${ARCHIVE_GB}GB"
echo ""
echo "=========================================="
echo ""
echo "To proceed with migration:"
echo "  ./migrate_old_trials.sh"
echo ""
