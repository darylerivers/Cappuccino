#!/usr/bin/env bash
#
# Quick Status - All Studies
#

clear
echo "========================================================================"
echo "  CAPPUCCINO MULTI-STUDY STATUS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""

# GPU Status
echo "🖥️  GPU STATUS:"
rocm-smi 2>/dev/null | grep -A 1 "GPU\[0\]" | tail -1
echo ""

# RAM Status
echo "💾 RAM STATUS:"
free -h | grep "Mem:" | awk '{print "   Used: "$3" / "$2" | Free: "$4" | Available: "$7}'
echo ""

# Workers by Study
echo "👥 WORKERS BY STUDY:"
ps aux | grep "[1]_optimize_unified" | grep -o "study-name [^ ]*" | awk '{print $2}' | sort | uniq -c | \
    awk '{printf "   %-40s %2d workers\n", $2, $1}'
echo ""

# Trial Status (Active Studies)
echo "📊 TRIAL STATUS:"
sqlite3 databases/optuna_cappuccino.db <<EOF
.mode column
.width 40 6 8 6 6
SELECT
    SUBSTR(s.study_name, 1, 40) as study,
    COUNT(CASE WHEN t.state = 'COMPLETE' THEN 1 END) as done,
    COUNT(CASE WHEN t.state = 'RUNNING' THEN 1 END) as run,
    printf('%.3f', AVG(CASE WHEN t.state = 'COMPLETE' THEN tv.value END)) as avg,
    printf('%.3f', MAX(tv.value)) as max
FROM studies s
JOIN trials t ON s.study_id = t.study_id
LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE s.study_name LIKE '%20260214%'
GROUP BY s.study_name
HAVING COUNT(t.trial_id) > 0
ORDER BY MAX(tv.value) DESC
LIMIT 10;
EOF
echo ""

# Paper Trading
echo "📈 PAPER TRADING:"
if pgrep -f "paper_trader_alpaca_polling" > /dev/null; then
    echo "   ✅ Running"
    tail -3 logs/paper_trader_live.log 2>/dev/null | grep -E "Portfolio|Next poll" | head -2 | sed 's/^/   /'
else
    echo "   ❌ Not running"
fi
echo ""

echo "========================================================================"
echo "Commands:"
echo "  ./monitor_all_studies.sh  - Live tail of all study logs"
echo "  ./quick_status.sh         - This status (refresh)"
echo "  python scripts/automation/trial_dashboard.py  - Interactive dashboard"
echo "========================================================================"
