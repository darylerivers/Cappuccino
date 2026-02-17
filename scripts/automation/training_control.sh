#!/bin/bash
# Training Worker Control Script
# Dynamically scale training workers up/down without stopping

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_LOG_DIR="$SCRIPT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Training Worker Control${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

get_worker_count() {
    ps aux | grep "scripts/training/1_optimize_unified" | grep -v grep | wc -l
}

get_vram_usage() {
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | \
        awk -F, '{printf "%d %d %.0f", $1, $2, ($1/$2)*100}'
}

show_status() {
    echo -e "${BLUE}Current Status:${NC}"
    echo "------------------------"

    WORKER_COUNT=$(get_worker_count)
    read VRAM_USED VRAM_TOTAL VRAM_PCT <<< $(get_vram_usage)

    echo -e "Training Workers: ${GREEN}$WORKER_COUNT${NC}"
    echo -e "GPU VRAM: ${VRAM_USED} MB / ${VRAM_TOTAL} MB (${VRAM_PCT}%)"

    if [ $VRAM_PCT -gt 85 ]; then
        echo -e "  Status: ${RED}HIGH${NC} (consider scaling down)"
    elif [ $VRAM_PCT -gt 70 ]; then
        echo -e "  Status: ${YELLOW}MODERATE${NC}"
    else
        echo -e "  Status: ${GREEN}GOOD${NC}"
    fi

    echo ""
    echo -e "Per-worker estimate: ~850 MB VRAM, ~1.2 GB RAM"
    echo ""
}

stop_workers() {
    local target_count=$1
    local current_count=$(get_worker_count)

    if [ $current_count -le $target_count ]; then
        echo -e "${YELLOW}Already at or below $target_count workers${NC}"
        return
    fi

    local to_stop=$((current_count - target_count))

    echo -e "${YELLOW}Stopping $to_stop workers (keeping $target_count running)...${NC}"
    echo ""

    # Get PIDs to stop (newest workers = highest PIDs)
    PIDS=$(ps aux | grep "optimize_unified" | grep -v grep | \
           sort -k2 -n -r | head -n $to_stop | awk '{print $2}')

    if [ -z "$PIDS" ]; then
        echo -e "${RED}No workers found to stop!${NC}"
        return 1
    fi

    echo "Will stop these workers:"
    ps aux | grep "optimize_unified" | grep -v grep | \
        sort -k2 -n -r | head -n $to_stop | \
        awk '{printf "  PID: %s, RAM: %d MB, CPU: %s%%\n", $2, $6/1024, $3}'
    echo ""

    read -p "Confirm? (y/n): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        return
    fi

    for pid in $PIDS; do
        kill $pid && echo -e "  ${GREEN}✓${NC} Stopped PID: $pid"
    done

    echo ""
    echo "Waiting 3 seconds for cleanup..."
    sleep 3

    show_status
    echo -e "${GREEN}✓ Scaled down to $target_count workers${NC}"
}

start_workers() {
    local target_count=$1
    local current_count=$(get_worker_count)

    if [ $current_count -ge $target_count ]; then
        echo -e "${YELLOW}Already at or above $target_count workers${NC}"
        return
    fi

    local to_start=$((target_count - current_count))

    echo -e "${YELLOW}Starting $to_start new workers (total will be $target_count)...${NC}"
    echo ""

    # Check if scripts/training/1_optimize_unified.py exists
    if [ ! -f "$SCRIPT_DIR/../../scripts/training/1_optimize_unified.py" ]; then
        echo -e "${RED}Error: scripts/training/1_optimize_unified.py not found${NC}"
        return 1
    fi

    # Start new workers
    for i in $(seq 1 $to_start); do
        local worker_num=$((current_count + i))
        local log_file="$WORKER_LOG_DIR/worker_${worker_num}_$(date +%Y%m%d_%H%M%S).log"

        nohup python -u scripts/training/1_optimize_unified.py \
            > "$log_file" 2>&1 &

        local new_pid=$!
        echo -e "  ${GREEN}✓${NC} Started worker #$worker_num (PID: $new_pid)"

        # Small delay to avoid resource spikes
        sleep 0.5
    done

    echo ""
    echo "Waiting 3 seconds for workers to initialize..."
    sleep 3

    show_status
    echo -e "${GREEN}✓ Scaled up to $target_count workers${NC}"
}

scale_to() {
    local target_count=$1
    local current_count=$(get_worker_count)

    if [ $current_count -eq $target_count ]; then
        echo -e "${GREEN}Already at $target_count workers${NC}"
        return
    elif [ $current_count -gt $target_count ]; then
        stop_workers $target_count
    else
        start_workers $target_count
    fi
}

show_menu() {
    print_header
    show_status

    echo -e "${BLUE}Quick Presets:${NC}"
    echo "------------------------"
    echo ""
    echo "1. MINIMAL (2 workers)"
    echo "   Frees: ~7 GB VRAM for other GPU tasks"
    echo "   Training: ~12 trials/hour (slow but continuous)"
    echo ""
    echo "2. LIGHT (4 workers)"
    echo "   Frees: ~5 GB VRAM"
    echo "   Training: ~24 trials/hour (moderate)"
    echo ""
    echo "3. HALF (5 workers)"
    echo "   Frees: ~4 GB VRAM"
    echo "   Training: ~30 trials/hour (balanced)"
    echo ""
    echo "4. RECOMMENDED (7 workers)"
    echo "   Uses: ~6 GB VRAM (75% of GPU)"
    echo "   Training: ~42 trials/hour (good speed)"
    echo ""
    echo "5. FULL (10 workers)"
    echo "   Uses: ~7.6 GB VRAM (93% of GPU)"
    echo "   Training: ~60 trials/hour (maximum speed)"
    echo ""
    echo "6. CUSTOM (specify worker count)"
    echo ""
    echo "7. STOP ALL WORKERS"
    echo ""
    echo "8. SHOW DETAILED STATUS"
    echo ""
    echo "9. EXIT"
    echo ""
}

detailed_status() {
    print_header
    show_status

    echo -e "${BLUE}Detailed Process List:${NC}"
    echo "========================"
    echo ""

    echo "Training Workers:"
    if [ $(get_worker_count) -eq 0 ]; then
        echo -e "  ${YELLOW}No workers running${NC}"
    else
        ps aux | grep "scripts/training/1_optimize_unified" | grep -v grep | \
            sort -k2 -n | \
            awk '{printf "  PID: %-8s CPU: %4s%% RAM: %6d MB  Started: %s\n", $2, $3, $6/1024, $9}'
    fi
    echo ""

    echo "Paper Traders:"
    if [ $(ps aux | grep "paper_trader" | grep -v grep | wc -l) -eq 0 ]; then
        echo -e "  ${YELLOW}No traders running${NC}"
    else
        ps aux | grep "paper_trader" | grep -v grep | \
            awk '{printf "  PID: %-8s CPU: %4s%% RAM: %6d MB\n", $2, $3, $6/1024}'
    fi
    echo ""

    echo "GPU Details:"
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader | \
        awk -F, '{printf "  VRAM Used: %s\n  VRAM Free: %s\n  GPU Util: %s\n  Temp: %s\n", $1, $2, $3, $4}'
    echo ""

    read -p "Press Enter to continue..."
}

stop_all() {
    local current_count=$(get_worker_count)

    if [ $current_count -eq 0 ]; then
        echo -e "${YELLOW}No workers running${NC}"
        return
    fi

    echo -e "${RED}WARNING: This will stop ALL training workers${NC}"
    echo "Current workers: $current_count"
    echo ""
    read -p "Are you sure? (type 'yes' to confirm): " confirm

    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        return
    fi

    echo ""
    echo "Stopping all workers..."

    PIDS=$(ps aux | grep "optimize_unified" | grep -v grep | awk '{print $2}')

    for pid in $PIDS; do
        kill $pid && echo -e "  ${GREEN}✓${NC} Stopped PID: $pid"
    done

    echo ""
    echo "Waiting 3 seconds for cleanup..."
    sleep 3

    show_status
    echo -e "${GREEN}✓ All workers stopped${NC}"
}

# Main menu loop
while true; do
    show_menu
    read -p "Choose option (1-9): " choice

    case $choice in
        1)
            echo ""
            scale_to 2
            echo ""
            read -p "Press Enter to continue..."
            ;;
        2)
            echo ""
            scale_to 4
            echo ""
            read -p "Press Enter to continue..."
            ;;
        3)
            echo ""
            scale_to 5
            echo ""
            read -p "Press Enter to continue..."
            ;;
        4)
            echo ""
            scale_to 7
            echo ""
            read -p "Press Enter to continue..."
            ;;
        5)
            echo ""
            scale_to 10
            echo ""
            read -p "Press Enter to continue..."
            ;;
        6)
            echo ""
            read -p "Enter target worker count (1-15): " custom_count
            if [[ $custom_count =~ ^[0-9]+$ ]] && [ $custom_count -ge 1 ] && [ $custom_count -le 15 ]; then
                echo ""
                scale_to $custom_count
            else
                echo -e "${RED}Invalid count. Must be 1-15.${NC}"
            fi
            echo ""
            read -p "Press Enter to continue..."
            ;;
        7)
            echo ""
            stop_all
            echo ""
            read -p "Press Enter to continue..."
            ;;
        8)
            detailed_status
            ;;
        9)
            echo ""
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option!${NC}"
            sleep 1
            ;;
    esac
done
