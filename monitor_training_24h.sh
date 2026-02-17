#!/bin/bash
# 24-hour training monitor - tracks GPU, RAM, and training progress

LOG_FILE="monitoring/training_monitor_$(date +%Y%m%d_%H%M%S).log"
mkdir -p monitoring

echo "=== 24-Hour Training Monitor Started: $(date) ===" | tee -a "$LOG_FILE"
echo "Logging to: $LOG_FILE"
echo ""

# Monitor for 24 hours (check every 60 seconds)
for i in $(seq 1 1440); do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

    # Get GPU usage
    GPU_USE=$(rocm-smi --showuse 2>/dev/null | grep "GPU\[0\]" | grep -oP '\d+(?=%)')

    # Get VRAM usage (in GB)
    VRAM_USED=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Used" | grep -oP '\d+' | head -1)
    VRAM_GB=$(echo "scale=1; $VRAM_USED / 1024 / 1024 / 1024" | bc)

    # Get RAM usage
    RAM_USED=$(free -g | awk '/^Mem:/ {print $3}')
    RAM_TOTAL=$(free -g | awk '/^Mem:/ {print $2}')

    # Check training processes
    PROCS=$(ps aux | grep "1_optimize_unified.py" | grep -v grep | wc -l)

    # Get trial counts from database
    TRIALS_INFO=$(/home/mrc/.pyenv/versions/cappuccino-rocm/bin/python << 'EOF'
import sqlite3
conn = sqlite3.connect('databases/optuna_cappuccino.db')
c = conn.cursor()
c.execute("""
    SELECT
        SUM(CASE WHEN t.state='COMPLETE' THEN 1 ELSE 0 END),
        SUM(CASE WHEN t.state='RUNNING' THEN 1 ELSE 0 END),
        SUM(CASE WHEN t.state='FAIL' THEN 1 ELSE 0 END)
    FROM trials t
    JOIN studies s ON t.study_id = s.study_id
    WHERE s.study_name LIKE 'cappuccino_5m_aggressive_%'
""")
row = c.fetchone()
print(f"{row[0] or 0},{row[1] or 0},{row[2] or 0}")
conn.close()
EOF
)

    IFS=',' read -r COMPLETE RUNNING FAILED <<< "$TRIALS_INFO"

    # Log the data
    echo "$TIMESTAMP | GPU:${GPU_USE}% | VRAM:${VRAM_GB}GB | RAM:${RAM_USED}/${RAM_TOTAL}GB | Procs:$PROCS | Trials: ✅$COMPLETE 🔄$RUNNING ❌$FAILED" | tee -a "$LOG_FILE"

    # Alert if GPU drops below 70%
    if [ "$GPU_USE" -lt 70 ]; then
        echo "  ⚠️  WARNING: GPU usage dropped to ${GPU_USE}%" | tee -a "$LOG_FILE"
    fi

    # Alert if processes died
    if [ "$PROCS" -lt 3 ]; then
        echo "  ❌ ERROR: Only $PROCS training processes running (expected 3)" | tee -a "$LOG_FILE"
    fi

    # Sleep 60 seconds
    sleep 60
done

echo "=== 24-Hour Monitor Completed: $(date) ===" | tee -a "$LOG_FILE"
