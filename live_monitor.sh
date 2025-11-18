#!/bin/bash
# Live Training Monitor - Comprehensive View
# Usage: ./live_monitor.sh

INTERVAL=3  # Update every 3 seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

while true; do
    clear

    echo -e "${MAGENTA}${BOLD}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}${BOLD}║                    CAPPUCCINO LIVE TRAINING MONITOR                            ║${NC}"
    echo -e "${MAGENTA}${BOLD}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # GPU Status
    echo -e "${CYAN}${BOLD}━━━ GPU STATUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed --format=csv,noheader,nounits 2>/dev/null)

    if [ $? -eq 0 ]; then
        IFS=',' read -r gpu_util mem_used mem_total temp power fan <<< "$GPU_INFO"

        # Color code GPU utilization
        if [ "$gpu_util" -gt 70 ]; then
            util_color="${GREEN}"
        elif [ "$gpu_util" -gt 40 ]; then
            util_color="${YELLOW}"
        else
            util_color="${RED}"
        fi

        # Color code temperature
        if [ "$temp" -gt 80 ]; then
            temp_color="${RED}"
        elif [ "$temp" -gt 70 ]; then
            temp_color="${YELLOW}"
        else
            temp_color="${GREEN}"
        fi

        mem_pct=$((mem_used * 100 / mem_total))

        echo -e "  GPU Utilization:  ${util_color}${gpu_util}%${NC}"
        echo -e "  Memory Usage:     ${mem_used}/${mem_total} MiB (${mem_pct}%)"
        echo -e "  Temperature:      ${temp_color}${temp}°C${NC}"
        echo -e "  Power Draw:       ${power}W"
        echo -e "  Fan Speed:        ${fan}%"
    else
        echo -e "  ${RED}GPU not available${NC}"
    fi

    echo ""

    # Training Processes
    echo -e "${CYAN}${BOLD}━━━ TRAINING PROCESSES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    PROC_INFO=$(ps aux | grep "python.*1_optimize_unified" | grep -v grep)

    if [ -n "$PROC_INFO" ]; then
        echo -e "${BOLD}  PID      CPU%   MEM%   TIME     STATUS${NC}"
        echo "$PROC_INFO" | while read -r line; do
            pid=$(echo "$line" | awk '{print $2}')
            cpu=$(echo "$line" | awk '{print $3}')
            mem=$(echo "$line" | awk '{print $4}')
            time=$(echo "$line" | awk '{print $10}')

            # Color code based on CPU usage
            if (( $(echo "$cpu > 100" | bc -l) )); then
                cpu_color="${GREEN}"
            elif (( $(echo "$cpu > 50" | bc -l) )); then
                cpu_color="${YELLOW}"
            else
                cpu_color="${RED}"
            fi

            echo -e "  ${pid}  ${cpu_color}${cpu}%${NC}    ${mem}%    ${time}    ${GREEN}RUNNING${NC}"
        done

        # Count processes
        proc_count=$(echo "$PROC_INFO" | wc -l)
        echo ""
        echo -e "  Total active processes: ${GREEN}${proc_count}${NC}"
    else
        echo -e "  ${RED}No training processes found${NC}"
    fi

    echo ""

    # Database Status
    echo -e "${CYAN}${BOLD}━━━ TRAINING PROGRESS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    DB_PATH="databases/optuna_cappuccino.db"

    if [ -f "$DB_PATH" ]; then
        # Count trials
        total=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trials" 2>/dev/null)
        complete=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trials WHERE state='COMPLETE'" 2>/dev/null)
        running=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trials WHERE state='RUNNING'" 2>/dev/null)
        failed=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trials WHERE state='FAIL'" 2>/dev/null)

        echo -e "  Total Trials:     ${total}"
        echo -e "  ${GREEN}Complete:${NC}         ${complete}"
        echo -e "  ${YELLOW}Running:${NC}          ${running}"
        if [ "$failed" -gt 0 ]; then
            echo -e "  ${RED}Failed:${NC}           ${failed}"
        fi

        # Calculate progress
        if [ "$total" -gt 0 ]; then
            progress=$((complete * 100 / 100))  # Out of 100 trials
            echo ""
            echo -e "  Progress: [${progress}%] "

            # Progress bar
            filled=$((progress / 2))
            echo -n "  ["
            for ((i=0; i<50; i++)); do
                if [ $i -lt $filled ]; then
                    echo -n "="
                else
                    echo -n " "
                fi
            done
            echo "]"
        fi
    else
        echo -e "  ${RED}Database not found${NC}"
    fi

    echo ""

    # Latest Output Preview
    echo -e "${CYAN}${BOLD}━━━ LATEST OUTPUT (last 10 lines) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Try to find latest log file
    LATEST_LOG=$(find train_results/cwd_tests/ -name "recorder.log" -type f -printf '%T+ %p\n' 2>/dev/null | sort -r | head -1 | awk '{print $2}')

    if [ -n "$LATEST_LOG" ]; then
        tail -10 "$LATEST_LOG" 2>/dev/null | sed 's/^/  /'
    else
        echo -e "  ${YELLOW}No log files found yet${NC}"
    fi

    echo ""
    echo -e "${MAGENTA}${BOLD}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}Press Ctrl+C to exit | Updating every ${INTERVAL}s${NC}"

    sleep $INTERVAL
done
