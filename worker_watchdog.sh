#!/bin/bash
# Worker Watchdog - Auto-restart workers every 60 min to prevent memory leaks
# Also monitors memory usage and triggers emergency restarts if needed
# Gracefully waits for trial completion before restart

RESTART_INTERVAL=3600  # 60 minutes in seconds
CHECK_INTERVAL=60      # Check every minute
WORKER_SCRIPT="./start_safe_workers.sh"
PID_FILE="logs/worker_pids.txt"
LOG_FILE="logs/watchdog.log"

# Memory safety thresholds
MEMORY_WARNING_GB=4    # Warn when system has less than 4GB free
MEMORY_CRITICAL_GB=2   # Emergency restart when less than 2GB free
WORKER_MAX_MEM_GB=8    # Max memory per worker before emergency restart

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

get_worker_age() {
    local pid=$1
    local start_time=$2
    local now=$(date +%s)
    echo $((now - start_time))
}

get_system_free_mem_gb() {
    # Get available memory in GB (using 'available' which is more accurate than 'free')
    free -g | awk '/^Mem:/ {print $7}'
}

get_process_mem_gb() {
    local pid=$1
    # Get RSS memory in GB for specific process
    ps -p $pid -o rss= 2>/dev/null | awk '{printf "%.2f", $1/1024/1024}'
}

check_memory_health() {
    local pid=$1
    local worker_num=$2

    # Get system memory
    local free_mem=$(get_system_free_mem_gb)

    # Get worker memory
    local worker_mem=$(get_process_mem_gb $pid)

    # Check critical system memory
    if [ "$free_mem" -le "$MEMORY_CRITICAL_GB" ]; then
        log "🚨 CRITICAL: System memory at ${free_mem}GB (threshold: ${MEMORY_CRITICAL_GB}GB)"
        log "   Emergency restart triggered!"
        return 2  # Critical - needs immediate restart
    fi

    # Check worker memory limit
    if (( $(echo "$worker_mem > $WORKER_MAX_MEM_GB" | bc -l) )); then
        log "⚠️  Worker $worker_num using ${worker_mem}GB (limit: ${WORKER_MAX_MEM_GB}GB)"
        log "   Worker memory leak detected - scheduling restart"
        return 1  # Warning - should restart soon
    fi

    # Check system memory warning
    if [ "$free_mem" -le "$MEMORY_WARNING_GB" ]; then
        log "⚠️  System memory at ${free_mem}GB (warning: ${MEMORY_WARNING_GB}GB), Worker $worker_num: ${worker_mem}GB"
    fi

    return 0  # All good
}

restart_worker() {
    local pid=$1
    local worker_num=$2
    
    log "⚠️  Worker $worker_num (PID $pid) reached restart interval"
    log "   Sending graceful shutdown signal..."
    
    # Send SIGTERM for graceful shutdown
    kill -TERM $pid 2>/dev/null
    
    # Wait up to 2 minutes for graceful shutdown
    for i in {1..120}; do
        if ! kill -0 $pid 2>/dev/null; then
            log "   ✅ Worker $worker_num shut down gracefully"
            return 0
        fi
        sleep 1
    done
    
    # Force kill if still running
    log "   ⚠️  Force killing worker $worker_num"
    kill -9 $pid 2>/dev/null
}

log "========================================="
log "Worker Watchdog Started"
log "Restart interval: ${RESTART_INTERVAL}s ($(($RESTART_INTERVAL/60))min)"
log "========================================="

while true; do
    # Check if PID file exists
    if [ ! -f "$PID_FILE" ]; then
        log "⚠️  No worker PID file found. Workers may not be running."
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # Read worker PIDs and start times
    worker_num=0
    while read pid start_time; do
        worker_num=$((worker_num + 1))

        # Skip if PID is not a number
        if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
            continue
        fi

        # Check if process is still running
        if ! kill -0 $pid 2>/dev/null; then
            log "⚠️  Worker PID $pid is dead. Starting new workers..."
            rm -f "$PID_FILE"
            $WORKER_SCRIPT
            sleep 10
            continue 2
        fi

        # Check memory health
        check_memory_health $pid $worker_num
        mem_status=$?

        if [ $mem_status -eq 2 ]; then
            # Critical memory - emergency restart
            log "🚨 EMERGENCY: Critical memory condition - restarting workers NOW"
            restart_worker $pid $worker_num
            rm -f "$PID_FILE"
            $WORKER_SCRIPT
            sleep 10
            break
        elif [ $mem_status -eq 1 ]; then
            # Worker memory leak detected - restart this worker
            log "🔄 Worker $worker_num memory leak detected - restarting"
            restart_worker $pid $worker_num
            rm -f "$PID_FILE"
            $WORKER_SCRIPT
            sleep 10
            break
        fi

        # Check worker age
        age=$(get_worker_age $pid $start_time)

        if [ $age -ge $RESTART_INTERVAL ]; then
            # Time to restart
            restart_worker $pid $worker_num

            # Remove old PID file and restart all workers
            rm -f "$PID_FILE"
            log "🔄 Restarting all workers..."
            $WORKER_SCRIPT
            sleep 10
            break
        else
            remaining=$((RESTART_INTERVAL - age))
            worker_mem=$(get_process_mem_gb $pid)
            log "✅ Worker $worker_num PID $pid healthy (age: ${age}s/${RESTART_INTERVAL}s, mem: ${worker_mem}GB, restart in ${remaining}s)"
        fi
    done < "$PID_FILE"
    
    # Wait before next check
    sleep $CHECK_INTERVAL
done
