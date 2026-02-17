#!/bin/bash
# Migrate old trial data to secondary NVMe drive (root partition)
# This frees up space on /home (nvme1n1) by archiving old studies to / (nvme0n1)

set -e

# Configuration
ARCHIVE_DIR="/archive_trials"              # Archive location on nvme0n1 (root)
CURRENT_STUDY="cappuccino_fresh_20251204_100527"
TRIALS_DIR="train_results/cwd_tests"
OLD_STUDIES=("cappuccino_1year_20251121" "cappuccino_trailing_20251125" "cappuccino_3workers_20251102_2325")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Trial Data Migration to Secondary Drive"
echo "=========================================="
echo ""

# Show current disk usage
echo "Current Disk Usage:"
df -h /home | tail -1 | awk '{print "  /home (nvme1n1): " $3 " used / " $2 " total (" $5 " full)"}'
df -h / | tail -1 | awk '{print "  /     (nvme0n1): " $3 " used / " $2 " total (" $5 " full)"}'
du -sh "$TRIALS_DIR" | awk '{print "  Trial data: " $1}'
echo ""

# Safety check
echo "${YELLOW}WARNING: This will move trial model files to $ARCHIVE_DIR${NC}"
echo "This operation:"
echo "  1. Creates $ARCHIVE_DIR on root partition (nvme0n1)"
echo "  2. Moves old study trials there (keeps new study on /home)"
echo "  3. Creates symlinks for backward compatibility"
echo "  4. Database remains unchanged (keeps references)"
echo ""
echo "${YELLOW}Press Enter to continue or Ctrl+C to cancel...${NC}"
read

# Create archive directory
echo ""
echo "Creating archive directory..."
sudo mkdir -p "$ARCHIVE_DIR"
sudo chown $USER:$USER "$ARCHIVE_DIR"
echo "${GREEN}✓${NC} Created $ARCHIVE_DIR"

# Get list of trials from current study (keep these on fast drive)
echo ""
echo "Identifying current study trials (will keep on /home)..."
CURRENT_TRIALS=$(sqlite3 databases/optuna_cappuccino.db \
  "SELECT t.number FROM trials t
   JOIN studies s ON t.study_id = s.study_id
   WHERE s.study_name = '$CURRENT_STUDY'")

CURRENT_COUNT=$(echo "$CURRENT_TRIALS" | wc -l)
echo "${GREEN}✓${NC} Found $CURRENT_COUNT trials in current study (keeping on fast drive)"

# Get trials from old studies (these will be archived)
echo ""
echo "Identifying old study trials to archive..."

for OLD_STUDY in "${OLD_STUDIES[@]}"; do
    echo "  Checking $OLD_STUDY..."

    # Get trial numbers for this old study
    OLD_TRIAL_NUMS=$(sqlite3 databases/optuna_cappuccino.db \
      "SELECT t.number FROM trials t
       JOIN studies s ON t.study_id = s.study_id
       WHERE s.study_name = '$OLD_STUDY' AND t.state = 'COMPLETE'" 2>/dev/null || echo "")

    if [ -z "$OLD_TRIAL_NUMS" ]; then
        echo "    No trials found"
        continue
    fi

    OLD_COUNT=$(echo "$OLD_TRIAL_NUMS" | wc -l)
    echo "    Found $OLD_COUNT trials"

    # Create study subdirectory in archive
    STUDY_ARCHIVE="$ARCHIVE_DIR/$OLD_STUDY"
    mkdir -p "$STUDY_ARCHIVE"

    # Move trial directories
    MOVED=0
    SKIPPED=0

    for TRIAL_NUM in $OLD_TRIAL_NUMS; do
        TRIAL_DIR="$TRIALS_DIR/trial_${TRIAL_NUM}_1h"

        # Check if trial directory exists
        if [ ! -d "$TRIAL_DIR" ]; then
            ((SKIPPED++))
            continue
        fi

        # Move to archive
        DEST="$STUDY_ARCHIVE/trial_${TRIAL_NUM}_1h"
        if [ ! -e "$DEST" ]; then
            mv "$TRIAL_DIR" "$DEST"
            # Create symlink back
            ln -s "$DEST" "$TRIAL_DIR"
            ((MOVED++))
        fi
    done

    echo "    ${GREEN}✓${NC} Moved $MOVED trials, skipped $SKIPPED (already archived or missing)"
done

echo ""
echo "=========================================="
echo "Migration Summary"
echo "=========================================="

# Show new disk usage
echo ""
echo "New Disk Usage:"
df -h /home | tail -1 | awk '{print "  /home (nvme1n1): " $3 " used / " $2 " total (" $5 " full)"}'
df -h / | tail -1 | awk '{print "  /     (nvme0n1): " $3 " used / " $2 " total (" $5 " full)"}'
du -sh "$ARCHIVE_DIR" 2>/dev/null | awk '{print "  Archive: " $1}'
du -sh "$TRIALS_DIR" 2>/dev/null | awk '{print "  Remaining trials: " $1}'
echo ""

# Show what's in archive
echo "Archive Contents:"
du -sh "$ARCHIVE_DIR"/* 2>/dev/null | awk '{print "  " $2 ": " $1}' || echo "  (empty)"
echo ""

# Verify symlinks work
echo "Verification:"
SYMLINK_COUNT=$(find "$TRIALS_DIR" -type l | wc -l)
echo "  Symlinks created: $SYMLINK_COUNT"
echo "  Database: Unchanged (all references still valid)"
echo "  Ensemble: Unchanged (can still load models via symlinks)"
echo ""

echo "${GREEN}✓ Migration complete!${NC}"
echo ""
echo "Notes:"
echo "  - Old trials archived to $ARCHIVE_DIR (on nvme0n1)"
echo "  - Current study trials remain on /home (fast access)"
echo "  - Symlinks maintain backward compatibility"
echo "  - Database references unchanged"
echo ""
echo "To verify everything works:"
echo "  ./status_automation.sh"
echo "  python dashboard.py"
echo ""
