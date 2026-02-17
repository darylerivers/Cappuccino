#!/bin/bash
# Pre-Migration Checklist for RX 7900 GRE
# Run this BEFORE swapping GPUs

set -e

echo "=========================================="
echo "RX 7900 GRE Pre-Migration Checklist"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_passed=0
check_failed=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((check_passed++))
    else
        echo -e "${RED}✗${NC} $1"
        ((check_failed++))
    fi
}

# 1. Backup current models
echo "1. Checking model backups..."
if [ -d "train_results/cwd_tests/trial_250_1h" ]; then
    mkdir -p backups/pre_amd_migration
    tar -czf backups/pre_amd_migration/trial_250_$(date +%Y%m%d).tar.gz \
        train_results/cwd_tests/trial_250_1h/
    check "Trial 250 backed up"
else
    echo -e "${YELLOW}⚠${NC} No Trial 250 found (OK if not trained yet)"
fi

# 2. Export current training state
echo ""
echo "2. Checking Optuna database..."
if [ -f "databases/optuna_cappuccino.db" ]; then
    cp databases/optuna_cappuccino.db backups/pre_amd_migration/
    check "Optuna DB backed up"
else
    echo -e "${RED}✗${NC} Optuna DB not found"
    ((check_failed++))
fi

# 3. Check current CUDA setup
echo ""
echo "3. Current CUDA/GPU setup..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > backups/pre_amd_migration/old_gpu_info.txt
check "Current GPU info saved"

# 4. Export Python packages
echo ""
echo "4. Exporting Python environment..."
pip freeze > backups/pre_amd_migration/requirements_cuda.txt
check "CUDA requirements exported"

# 5. Check disk space
echo ""
echo "5. Checking disk space..."
AVAILABLE=$(df -BG /opt/user-data | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE" -gt 50 ]; then
    check "Disk space: ${AVAILABLE}GB available (>50GB required)"
else
    echo -e "${RED}✗${NC} Disk space: ${AVAILABLE}GB available (<50GB - may need cleanup)"
    ((check_failed++))
fi

# 6. Stop all training
echo ""
echo "6. Checking for running training..."
TRAINING_PROCS=$(pgrep -f "1_optimize_unified.py" | wc -l)
if [ "$TRAINING_PROCS" -eq 0 ]; then
    check "No training processes running"
else
    echo -e "${YELLOW}⚠${NC} $TRAINING_PROCS training processes still running"
    echo "    Run: pkill -9 -f 1_optimize_unified.py"
fi

# 7. Stop paper trading
echo ""
echo "7. Checking paper trader..."
TRADER_PROCS=$(pgrep -f "paper_trader" | wc -l)
if [ "$TRADER_PROCS" -eq 0 ]; then
    check "No paper trader running"
else
    echo -e "${YELLOW}⚠${NC} Paper trader still running (PID: $(pgrep -f paper_trader))"
    echo "    This is OK - will restart after migration"
fi

# 8. Create migration guide
echo ""
echo "8. Creating migration guide..."
cat > backups/pre_amd_migration/MIGRATION_STEPS.txt << 'EOF'
RX 7900 GRE Migration Steps
===========================

Day 0 (Before GPU Arrives):
✓ Run 1_pre_migration_checklist.sh
✓ Download ROCm installer
✓ Prepare ROCm installation script

Day 1 (GPU Arrival):
1. Shutdown system
2. Swap RTX 3070 → RX 7900 GRE
3. Boot system
4. Run 2_install_rocm.sh
5. Run 3_install_pytorch_rocm.sh
6. Run 4_verify_amd_setup.sh
7. Run 5_update_training_config.sh
8. Start paper trader
9. Start training (10 workers!)

Estimated time: 2-3 hours
EOF
check "Migration guide created"

# Summary
echo ""
echo "=========================================="
echo "Pre-Migration Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}${check_passed}${NC}"
echo -e "Failed: ${RED}${check_failed}${NC}"
echo ""

if [ $check_failed -eq 0 ]; then
    echo -e "${GREEN}✓ System ready for GPU migration${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Download ROCm: ./infrastructure/amd_migration/download_rocm.sh"
    echo "  2. When GPU arrives: ./infrastructure/amd_migration/2_install_rocm.sh"
else
    echo -e "${RED}✗ Fix issues above before migrating${NC}"
fi

echo ""
echo "Backup location: backups/pre_amd_migration/"
