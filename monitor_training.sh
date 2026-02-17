#!/bin/bash
clear
echo "========================================================================"
echo "  CAPPUCCINO TRAINING MONITOR - $(date '+%H:%M:%S')"
echo "========================================================================"

WORKERS=$(ps aux | grep '[1]_optimize' | wc -l)
echo "👥 Workers: $WORKERS"

free -h | grep Mem | awk '{print "💾 RAM: "$3" / "$2}'

rocm-smi 2>/dev/null | grep "GPU\[0\]" -A 1 | tail -1 | awk '{print "🖥️  GPU: "$14" | VRAM: "$13}'

echo ""
sqlite3 databases/optuna_cappuccino.db "
SELECT '✅ Completed: '||COUNT(CASE WHEN state='COMPLETE' THEN 1 END)||
       ' | 🔄 Running: '||COUNT(CASE WHEN state='RUNNING' THEN 1 END)||
       ' | Avg: '||printf('%.4f',AVG(CASE WHEN state='COMPLETE' THEN tv.value END))
FROM trials t JOIN studies s ON t.study_id=s.study_id 
LEFT JOIN trial_values tv ON t.trial_id=tv.trial_id
WHERE s.study_name='cappuccino_auto_20260214_2059'
"

echo ""
echo "Latest 3 trials:"
sqlite3 databases/optuna_cappuccino.db "
SELECT '  #'||t.number||': '||printf('%.4f',tv.value)||' at '||substr(DATETIME(t.datetime_complete),12,5)
FROM trials t JOIN studies s ON t.study_id=s.study_id
LEFT JOIN trial_values tv ON t.trial_id=tv.trial_id
WHERE s.study_name='cappuccino_auto_20260214_2059' AND t.state='COMPLETE'
ORDER BY t.datetime_complete DESC LIMIT 3
"
echo "========================================================================"
