#!/bin/bash
# System Verification Script
# Checks that all components are using the correct active study

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "System Configuration Verification"
echo "=========================================="
echo ""

# Load config
if [ -f ".env.training" ]; then
    source .env.training
    echo -e "${GREEN}✓${NC} Configuration file: .env.training"
    echo "  Active Study: $ACTIVE_STUDY_NAME"
else
    echo -e "${RED}✗${NC} Configuration file NOT FOUND"
    exit 1
fi

echo ""
echo "=========================================="
echo "Component Status"
echo "=========================================="
echo ""

ERRORS=0

# 1. Check training workers
echo "[1/5] Training Workers"
RUNNING_STUDIES=$(ps aux | grep "1_optimize" | grep -v grep | grep -oP -- '--study-name \K[^ ]+' | sort -u)

if [ -z "$RUNNING_STUDIES" ]; then
    echo -e "  ${YELLOW}⚠${NC}  No training workers running"
else
    while IFS= read -r study; do
        COUNT=$(ps aux | grep "1_optimize" | grep -v grep | grep -c "$study" || true)
        if [ "$study" = "$ACTIVE_STUDY_NAME" ]; then
            echo -e "  ${GREEN}✓${NC}  $COUNT workers on $study (CORRECT)"
        else
            echo -e "  ${RED}✗${NC}  $COUNT workers on $study (WRONG - should be $ACTIVE_STUDY_NAME)"
            ERRORS=$((ERRORS + 1))
        fi
    done <<< "$RUNNING_STUDIES"
fi

# 2. Check auto-deployer
echo ""
echo "[2/5] Auto-Model Deployer"
if pgrep -f "auto_model_deployer.py" > /dev/null; then
    DEPLOYER_STUDY=$(ps aux | grep "auto_model_deployer.py" | grep -v grep | grep -oP -- '--study \K[^ ]+' | head -1)
    if [ "$DEPLOYER_STUDY" = "$ACTIVE_STUDY_NAME" ]; then
        echo -e "  ${GREEN}✓${NC}  Running with $DEPLOYER_STUDY"
    else
        echo -e "  ${RED}✗${NC}  Running with $DEPLOYER_STUDY (should be $ACTIVE_STUDY_NAME)"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "  ${YELLOW}⚠${NC}  Not running"
fi

# 3. Check performance monitor
echo ""
echo "[3/5] Performance Monitor"
if pgrep -f "performance_monitor.py" > /dev/null; then
    MONITOR_STUDY=$(ps aux | grep "performance_monitor.py" | grep -v grep | grep -oP -- '--study \K[^ ]+' | head -1)
    if [ "$MONITOR_STUDY" = "$ACTIVE_STUDY_NAME" ]; then
        echo -e "  ${GREEN}✓${NC}  Running with $MONITOR_STUDY"
    else
        echo -e "  ${RED}✗${NC}  Running with $MONITOR_STUDY (should be $ACTIVE_STUDY_NAME)"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "  ${YELLOW}⚠${NC}  Not running"
fi

# 4. Check ensemble updater
echo ""
echo "[4/5] Ensemble Auto-Updater"
if pgrep -f "ensemble_auto_updater.py" > /dev/null; then
    ENSEMBLE_STUDY=$(ps aux | grep "ensemble_auto_updater.py" | grep -v grep | grep -oP -- '--study \K[^ ]+' | head -1)
    if [ "$ENSEMBLE_STUDY" = "$ACTIVE_STUDY_NAME" ]; then
        echo -e "  ${GREEN}✓${NC}  Running with $ENSEMBLE_STUDY"
    else
        echo -e "  ${RED}✗${NC}  Running with $ENSEMBLE_STUDY (should be $ACTIVE_STUDY_NAME)"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "  ${YELLOW}⚠${NC}  Not running"
fi

# 5. Check paper trader
echo ""
echo "[5/5] Paper Trader"
if pgrep -f "paper_trader_alpaca_polling.py" > /dev/null; then
    echo -e "  ${GREEN}✓${NC}  Running"
    # Paper trader uses ensemble, so it's OK as long as ensemble is synced
else
    echo -e "  ${YELLOW}⚠${NC}  Not running"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All components configured correctly!${NC}"
    echo ""
    echo "Active Study: $ACTIVE_STUDY_NAME"
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS configuration errors${NC}"
    echo ""
    echo "To fix:"
    echo "  1. ./stop_automation.sh"
    echo "  2. pkill -f 1_optimize_unified.py (for wrong studies)"
    echo "  3. ./start_automation.sh"
    echo "  4. ./start_training.sh (if needed)"
    exit 1
fi
