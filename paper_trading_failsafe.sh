#!/bin/bash
# Paper Trading Fail-Safe Wrapper
# Auto-restarts paper trading if it crashes, with exponential backoff

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODEL_DIR="${1:-train_results/cwd_tests/trial_3358_1h}"
TICKERS="${2:-AAVE/USD AVAX/USD BTC/USD LINK/USD ETH/USD LTC/USD UNI/USD}"
POLL_INTERVAL="${3:-60}"
MAX_RESTART_ATTEMPTS=999999  # Essentially unlimited
BACKOFF_BASE=5               # Start with 5 seconds
BACKOFF_MAX=300              # Max 5 minutes between restarts
LOG_DIR="logs"
PID_FILE="deployments/paper_trading.pid"
STATE_FILE="deployments/paper_trading_state.json"

mkdir -p "$LOG_DIR" deployments

# Initialize state
if [ ! -f "$STATE_FILE" ]; then
    echo '{"restart_count": 0, "last_restart": null, "consecutive_failures": 0}' > "$STATE_FILE"
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/paper_trading_failsafe.log"
}

# Update state
update_state() {
    local restart_count=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('restart_count', 0))")
    local consecutive=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('consecutive_failures', 0))")

    python3 -c "
import json
from datetime import datetime
state = {
    'restart_count': $restart_count + 1,
    'last_restart': datetime.now().isoformat(),
    'consecutive_failures': $consecutive + 1,
    'last_model': '$MODEL_DIR'
}
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"
}

# Reset consecutive failures on successful run
reset_consecutive() {
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['consecutive_failures'] = 0
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"
}

# Calculate backoff
get_backoff() {
    local consecutive=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('consecutive_failures', 0))")
    local backoff=$((BACKOFF_BASE * (2 ** consecutive)))
    if [ $backoff -gt $BACKOFF_MAX ]; then
        backoff=$BACKOFF_MAX
    fi
    echo $backoff
}

# Cleanup on exit
cleanup() {
    log "Fail-safe wrapper shutting down..."
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping paper trading process (PID: $pid)..."
            kill "$pid" 2>/dev/null
            sleep 2
            kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$PID_FILE"
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

log "=========================================="
log "Paper Trading Fail-Safe Started"
log "=========================================="
log "Model: $MODEL_DIR"
log "Tickers: $TICKERS"
log "Poll Interval: ${POLL_INTERVAL}s"
log "Max Restarts: Unlimited (with backoff)"
log "=========================================="

attempt=0
while true; do
    attempt=$((attempt + 1))

    # Check if model directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        log "ERROR: Model directory not found: $MODEL_DIR"
        log "Waiting for model to be created..."
        sleep 60
        continue
    fi

    # Check if required files exist
    if [ ! -f "$MODEL_DIR/actor.pth" ] && [ ! -f "$MODEL_DIR/best_trial.pkl" ]; then
        log "ERROR: Model files not found in $MODEL_DIR"
        log "Waiting for model files to be created..."
        sleep 60
        continue
    fi

    log "Starting paper trading (attempt #$attempt)..."

    # Start paper trading
    python -u paper_trader_alpaca_polling.py \
        --model-dir "$MODEL_DIR" \
        --tickers $TICKERS \
        --poll-interval $POLL_INTERVAL \
        --history-hours 24 \
        > "$LOG_DIR/paper_trading_live.log" 2>&1 &

    TRADING_PID=$!
    echo $TRADING_PID > "$PID_FILE"

    log "Paper trading started (PID: $TRADING_PID)"

    # Monitor the process
    start_time=$(date +%s)
    while kill -0 $TRADING_PID 2>/dev/null; do
        sleep 10

        # Check if it's been running for more than 60 seconds (successful start)
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -gt 60 ]; then
            # Reset consecutive failures if running for more than 1 minute
            reset_consecutive
        fi
    done

    # Process died
    wait $TRADING_PID
    exit_code=$?

    log "Paper trading stopped (exit code: $exit_code)"

    # Check if we should restart
    if [ $exit_code -eq 0 ]; then
        log "Clean exit detected. Not restarting."
        rm -f "$PID_FILE"
        break
    fi

    # Update state and calculate backoff
    update_state
    backoff=$(get_backoff)

    log "Process crashed. Restarting in ${backoff}s..."
    log "Check logs/paper_trading_live.log for error details"

    sleep $backoff
done

log "Fail-safe wrapper stopped"
