# Weekly Training Cycle Guide

**Purpose:** Train models on FRESH data every week for maximum performance

---

## 📅 Weekly Schedule

**Recommended:** Start fresh training **Friday evening** or **Saturday morning**

By end of week (Sunday), you'll have **100+ models** trained on current market data.

---

## 🚀 Quick Start (End of Week)

### Option 1: Automated Script (Recommended)
```bash
./start_fresh_weekly_training.sh
```

This script will:
1. Download 1 year of fresh data from Alpaca (5-10 min)
2. Create new study: `cappuccino_week_YYYYMMDD`
3. Update `.env.training` to new study
4. Stop old training workers
5. Start 3 new training workers on fresh data

### Option 2: Manual Steps

```bash
# 1. Download fresh data
python3 prepare_1year_training_data.py \
    --months 12 \
    --output-dir data/1h_fresh_$(date +%Y%m%d) \
    --train-pct 0.8

# 2. Stop old training
pkill -f 1_optimize_unified.py

# 3. Edit .env.training
nano .env.training
# Change: ACTIVE_STUDY_NAME="cappuccino_week_20251206"

# 4. Start new training
for i in {1..3}; do
    nohup python -u 1_optimize_unified.py \
        --n-trials 500 \
        --gpu 0 \
        --study-name cappuccino_week_$(date +%Y%m%d) \
        --data-dir data/1h_fresh_$(date +%Y%m%d) \
        > logs/training_worker_${i}.log 2>&1 &
done

# 5. Start automation (after 100+ trials)
./start_automation.sh
```

---

## ⏱️ Timeline

### Friday Evening / Saturday Morning
- **Action:** Run `./start_fresh_weekly_training.sh`
- **Time:** 5-10 minutes to download data + start workers
- **Result:** 3 workers training on current data

### Saturday - Sunday
- **What's Happening:** Training accumulates 100-150 trials
- **Monitoring:** Check dashboard periodically
- **Action:** Once 100+ trials complete, run `./start_automation.sh`

### Monday - Friday
- **What's Happening:**
  - Paper trading uses ensemble of best models
  - Auto-deployer updates as better models complete
  - Ensemble auto-syncs top 20 models every 10 min
- **Monitoring:** Watch dashboard, check paper trading performance

### Next Friday
- **Action:** Start fresh cycle again with current data!

---

## 📊 Why Weekly Fresh Data Matters

### Problem with Stale Data
```
Old Study (Nov 21):
  ❌ Training on 2-week-old market conditions
  ❌ Models don't adapt to current volatility
  ❌ Poor live performance due to regime change
```

### Fresh Data Advantage
```
Fresh Study (Dec 6):
  ✅ Training on current market conditions
  ✅ Models capture recent patterns
  ✅ Better live trading performance
```

**Example:**
- Old data: Nov 21 (BTC at $89k, bull run)
- Current: Dec 6 (BTC at $89k, consolidation)
- **Market regime changed** → need fresh models!

---

## 🎯 Current Status (2025-12-06 20:13)

### Fresh Data Downloaded
- **Location:** `data/1h_fresh_20251206/`
- **Downloaded:** December 6, 2025
- **Range:** 1 year of hourly data
- **Status:** ✅ Ready for training

### Old System (Still Running)
- **Study:** `cappuccino_1year_20251121`
- **Data Age:** 16 days old
- **Trials:** 1,369 (many model files deleted)
- **Action:** Will be replaced by fresh weekly study

### Recommended Next Steps
```bash
# 1. Start fresh training NOW
./start_fresh_weekly_training.sh

# 2. Monitor in dashboard
# Check "TRAINING" section - should show new study name

# 3. Wait for 100+ trials (24-48 hours)

# 4. Start automation
./start_automation.sh

# 5. Verify ensemble loads
ls train_results/ensemble/model_* | wc -l
# Should show 20 models

# 6. Check paper trading
tail -f logs/paper_trading_ensemble.log
```

---

## 🔍 Monitoring Progress

### Check Training Status
```bash
# View study progress
./status_automation.sh

# Check worker logs
tail -f logs/training_worker_1_cappuccino_week_*.log

# Count completed trials
sqlite3 databases/optuna_cappuccino.db "SELECT COUNT(*) FROM trials WHERE study_name = 'cappuccino_week_20251206' AND state = 'COMPLETE'"
```

### Check Dashboard
```bash
python3 dashboard.py
```

Look for:
- **Training section:** Shows new study name
- **Trial count:** Increasing over time
- **Best value:** Improving as better models found

### Check Ensemble
```bash
cat train_results/ensemble/ensemble_manifest.json | python3 -m json.tool | head -30
```

Should show:
- `"model_count": 20`
- `"study_name": "cappuccino_week_20251206"`
- Recent `"updated"` timestamp

---

## 🛠️ Troubleshooting

### Ensemble shows 0 models
**Cause:** Not enough trials with saved model files yet

**Solution:**
```bash
# Check how many trials have model files
ls train_results/cwd_tests/trial_*_1h/*.pth | wc -l

# If < 20, wait for more trials to complete
# If > 20, manually trigger ensemble sync
python3 ensemble_auto_updater.py --study cappuccino_week_20251206 --top-n 20 --once
```

### Training workers crashed
**Solution:**
```bash
# Check GPU
nvidia-smi

# Check logs
tail -100 logs/training_worker_1*.log

# Restart workers
pkill -f 1_optimize_unified.py
./start_fresh_weekly_training.sh
```

### Paper trader not using new models
**Solution:**
```bash
# Verify automation is running
./status_automation.sh

# Restart automation
./stop_automation.sh
./start_automation.sh

# Check ensemble was updated
cat train_results/ensemble/ensemble_manifest.json | grep study_name
```

---

## 📝 Automation After Fresh Start

**Wait for 100+ trials before starting automation!**

Why?
- Ensemble needs 20+ trials with model files
- Early trials may be poor quality
- Better to have selection of good models

Once you have 100+ trials:
```bash
./start_automation.sh
```

This starts:
- **Auto-deployer:** Finds and deploys best models
- **Ensemble updater:** Keeps top 20 models synced
- **Performance monitor:** Tracks training progress
- **System watchdog:** Restarts crashed processes

---

## 🎓 Best Practices

1. **Start Fresh Weekly**
   - Markets change rapidly in crypto
   - Fresh data = better models

2. **Monitor First 24 Hours**
   - Check workers are running
   - Verify trials completing
   - Watch for GPU issues

3. **Start Automation After 100 Trials**
   - Gives ensemble good selection
   - Avoids deploying poor early trials

4. **Check Paper Trading Daily**
   - Monitor performance
   - Look for concerning patterns
   - Verify ensemble is updating

5. **Archive Old Studies**
   - Old studies consume disk space
   - Keep database records but delete old `train_results/`

---

## 📂 File Management

### Weekly Cleanup (Optional)
```bash
# Archive old trial directories (keeps database records)
OLD_STUDY="cappuccino_1year_20251121"

# Create archive
tar -czf archive/trials_${OLD_STUDY}_$(date +%Y%m%d).tar.gz train_results/cwd_tests/trial_*

# Delete old trials (optional - frees 50GB+)
# rm -rf train_results/cwd_tests/trial_*_1h
```

### Keep Database
**Never delete:** `databases/optuna_cappuccino.db`

This contains:
- All trial hyperparameters
- Performance metrics
- Optimization history
- Required for analysis

---

## 🔄 Weekly Workflow Summary

```
Friday Evening:
  └─> ./start_fresh_weekly_training.sh
       └─> Downloads fresh data (5-10 min)
       └─> Starts 3 training workers
       └─> Creates new study: cappuccino_week_YYYYMMDD

Saturday - Sunday:
  └─> Workers accumulate 100-150 trials
  └─> Monitor via dashboard
  └─> Once 100+ trials: ./start_automation.sh

Monday - Friday:
  └─> Paper trading with ensemble
  └─> Auto-deployer updates best models
  └─> Ensemble syncs top 20 every 10 min
  └─> Monitor performance daily

Next Friday:
  └─> Repeat! Start fresh with new data
```

---

## ✅ Checklist

Before starting fresh training:
- [ ] GPU is available (check `nvidia-smi`)
- [ ] Disk space available (need ~50GB for new data + models)
- [ ] Alpaca API keys configured (in `.env`)
- [ ] No critical paper trades open (will restart trader)

After starting fresh training:
- [ ] 3 workers running (`pgrep -f 1_optimize_unified`)
- [ ] Trials completing (check dashboard)
- [ ] GPU utilized (~100% during training)
- [ ] New data directory created

After 100+ trials:
- [ ] Automation started (`./status_automation.sh`)
- [ ] Ensemble has 20 models (`ls train_results/ensemble/model_*`)
- [ ] Paper trader running with new ensemble
- [ ] System verified (`./verify_system.sh`)

---

**Ready to start fresh training?**

```bash
./start_fresh_weekly_training.sh
```

Then monitor progress in the dashboard! 📊
