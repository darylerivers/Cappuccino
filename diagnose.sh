#!/bin/bash
# Quick System Diagnostic Script
# Runs automated checks to identify common issues

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Cappuccino System Diagnostics"
echo "=========================================="
echo ""

WARNINGS=0
ERRORS=0
CRITICAL=0

# Helper functions
log_ok() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

log_error() {
    echo -e "${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

log_critical() {
    echo -e "${RED}✗✗${NC} $1"
    CRITICAL=$((CRITICAL + 1))
}

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# 1. Configuration Check
echo "[1/10] Configuration Files"
if [ -f ".env.training" ]; then
    source .env.training
    log_ok ".env.training found"
    log_info "Active Study: $ACTIVE_STUDY_NAME"
else
    log_critical ".env.training missing - REQUIRED FOR OPERATION"
fi

if [ -f ".env" ]; then
    log_ok ".env found (Alpaca API keys)"
else
    log_error ".env missing - required for paper trading"
fi
echo ""

# 2. Database Check
echo "[2/10] Database"
if [ -f "databases/optuna_cappuccino.db" ]; then
    log_ok "Database file exists"

    # Test database access
    if sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM studies LIMIT 1;" >/dev/null 2>&1; then
        log_ok "Database is accessible"

        # Get study count
        STUDY_COUNT=$(sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM studies;")
        log_info "Studies in database: $STUDY_COUNT"

        # Check if active study exists
        if [ -n "$ACTIVE_STUDY_NAME" ]; then
            ACTIVE_EXISTS=$(sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM studies WHERE study_name='$ACTIVE_STUDY_NAME';")
            if [ "$ACTIVE_EXISTS" -eq "1" ]; then
                log_ok "Active study exists in database"

                # Get trial count
                TRIAL_COUNT=$(sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials WHERE study_id=(SELECT study_id FROM studies WHERE study_name='$ACTIVE_STUDY_NAME');")
                log_info "Trials in active study: $TRIAL_COUNT"
            else
                log_error "Active study '$ACTIVE_STUDY_NAME' not found in database"
            fi
        fi
    else
        log_error "Database exists but cannot be accessed (locked or corrupted?)"
    fi
else
    log_error "Database file not found"
fi
echo ""

# 3. GPU Check
echo "[3/10] GPU"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        log_ok "GPU accessible"

        # Get GPU utilization
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

        log_info "GPU Utilization: ${GPU_UTIL}%"
        log_info "GPU Memory: ${GPU_MEM}MB / ${GPU_MEM_TOTAL}MB"

        if [ "$GPU_UTIL" -lt "10" ]; then
            log_warn "GPU utilization low - training may not be running"
        fi
    else
        log_error "nvidia-smi failed - GPU may be unavailable"
    fi
else
    log_warn "nvidia-smi not found - cannot check GPU"
fi
echo ""

# 4. Disk Space Check
echo "[4/10] Disk Space"
DISK_USAGE=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')

if [ "$DISK_USAGE" -lt "80" ]; then
    log_ok "Disk space OK (${DISK_USAGE}% used, ${DISK_AVAIL} available)"
elif [ "$DISK_USAGE" -lt "90" ]; then
    log_warn "Disk space getting low (${DISK_USAGE}% used, ${DISK_AVAIL} available)"
else
    log_error "Disk space critical (${DISK_USAGE}% used, ${DISK_AVAIL} available)"
fi
echo ""

# 5. Memory Check
echo "[5/10] System Memory"
MEM_TOTAL=$(free -g | awk 'NR==2 {print $2}')
MEM_USED=$(free -g | awk 'NR==2 {print $3}')
MEM_AVAIL=$(free -g | awk 'NR==2 {print $7}')
MEM_PERCENT=$(awk "BEGIN {printf \"%.0f\", ($MEM_USED/$MEM_TOTAL)*100}")

if [ "$MEM_PERCENT" -lt "80" ]; then
    log_ok "Memory OK (${MEM_USED}GB/${MEM_TOTAL}GB used, ${MEM_AVAIL}GB available)"
elif [ "$MEM_PERCENT" -lt "90" ]; then
    log_warn "Memory usage high (${MEM_USED}GB/${MEM_TOTAL}GB used, ${MEM_AVAIL}GB available)"
else
    log_error "Memory critical (${MEM_USED}GB/${MEM_TOTAL}GB used, ${MEM_AVAIL}GB available)"
fi
echo ""

# 6. Training Workers Check
echo "[6/10] Training Workers"
WORKER_COUNT=$(pgrep -f "1_optimize_unified.py" | wc -l)

if [ "$WORKER_COUNT" -gt "0" ]; then
    log_ok "$WORKER_COUNT training worker(s) running"

    # Check which study they're using
    RUNNING_STUDIES=$(ps aux | grep "1_optimize" | grep -v grep | grep -oP -- '--study-name \K[^ ]+' | sort -u)

    if [ -n "$ACTIVE_STUDY_NAME" ]; then
        while IFS= read -r study; do
            COUNT=$(ps aux | grep "1_optimize" | grep -v grep | grep -c "$study" || true)
            if [ "$study" = "$ACTIVE_STUDY_NAME" ]; then
                log_ok "$COUNT workers on correct study: $study"
            else
                log_error "$COUNT workers on WRONG study: $study (should be $ACTIVE_STUDY_NAME)"
            fi
        done <<< "$RUNNING_STUDIES"
    else
        log_warn "Cannot verify study (ACTIVE_STUDY_NAME not set)"
    fi
else
    log_warn "No training workers running"
fi
echo ""

# 7. Automation Components Check
echo "[7/10] Automation Components"

check_component() {
    local name=$1
    local pid_file=$2
    local pattern=$3

    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            log_ok "$name running (PID: $PID)"
        else
            log_error "$name NOT running (stale PID: $PID)"
            log_info "  Fix: rm $pid_file && ./start_automation.sh"
        fi
    else
        PID=$(pgrep -f "$pattern" | head -1)
        if [ -n "$PID" ]; then
            log_warn "$name running but no PID file (PID: $PID)"
        else
            log_warn "$name not running"
        fi
    fi
}

check_component "Auto-Model Deployer" "deployments/auto_deployer.pid" "auto_model_deployer.py"
check_component "System Watchdog" "deployments/watchdog.pid" "system_watchdog.py"
check_component "Performance Monitor" "deployments/performance_monitor.pid" "performance_monitor.py"
check_component "Ensemble Updater" "deployments/ensemble_updater.pid" "ensemble_auto_updater.py"
echo ""

# 8. Paper Trader Check
echo "[8/10] Paper Trader"
if pgrep -f "paper_trader_alpaca_polling.py" > /dev/null; then
    PID=$(pgrep -f "paper_trader_alpaca_polling.py" | head -1)
    log_ok "Paper trader running (PID: $PID)"

    # Check for recent activity in log
    if [ -f "logs/paper_trading_live.log" ]; then
        LAST_LOG=$(tail -1 logs/paper_trading_live.log 2>/dev/null)
        if [ -n "$LAST_LOG" ]; then
            log_info "Recent activity detected"
        fi
    fi
else
    log_warn "Paper trader not running"
fi
echo ""

# 9. Alpaca API Check
echo "[9/10] Alpaca API Connection"
if [ -f ".env" ]; then
    source .env

    if python3 -c "
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
try:
    client = TradingClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'), paper=True)
    account = client.get_account()
    exit(0)
except Exception as e:
    exit(1)
" 2>/dev/null; then
        log_ok "Alpaca API connected"

        BUYING_POWER=$(python3 -c "
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
load_dotenv()
client = TradingClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'), paper=True)
account = client.get_account()
print(f'\${float(account.buying_power):,.2f}')
" 2>/dev/null)

        log_info "Buying Power: $BUYING_POWER"
    else
        log_error "Cannot connect to Alpaca API - check credentials in .env"
    fi
else
    log_warn "Cannot test API (.env file missing)"
fi
echo ""

# 10. Recent Errors Check
echo "[10/10] Recent Errors in Logs"
ERROR_COUNT=0

if [ -d "logs" ]; then
    # Check for recent errors (last 100 lines of each log)
    RECENT_ERRORS=$(find logs/ -name "*.log" -type f -exec tail -100 {} \; 2>/dev/null | grep -i "error\|exception\|traceback\|failed" | wc -l)

    if [ "$RECENT_ERRORS" -eq "0" ]; then
        log_ok "No recent errors in logs"
    elif [ "$RECENT_ERRORS" -lt "10" ]; then
        log_warn "$RECENT_ERRORS recent errors/warnings found"
        log_info "  Check: grep -i error logs/*.log | tail -20"
    else
        log_error "$RECENT_ERRORS recent errors/warnings found"
        log_info "  Check: grep -i error logs/*.log | tail -20"
    fi
else
    log_warn "logs/ directory not found"
fi
echo ""

# Summary
echo "=========================================="
echo "Diagnostic Summary"
echo "=========================================="

if [ $CRITICAL -gt 0 ]; then
    echo -e "${RED}✗✗ CRITICAL ISSUES: $CRITICAL${NC}"
fi

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}✗ Errors: $ERRORS${NC}"
fi

if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠ Warnings: $WARNINGS${NC}"
fi

if [ $CRITICAL -eq 0 ] && [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "System is healthy and operational."
    exit 0
elif [ $CRITICAL -gt 0 ]; then
    echo ""
    echo "CRITICAL issues detected - system may not function properly."
    echo "See DIAGNOSTIC_GUIDE.md for troubleshooting steps."
    exit 2
elif [ $ERRORS -gt 0 ]; then
    echo ""
    echo "Errors detected - some components may not be working."
    echo "See DIAGNOSTIC_GUIDE.md for troubleshooting steps."
    exit 1
else
    echo ""
    echo "System operational with minor warnings."
    exit 0
fi
