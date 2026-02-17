# Full Automation - Training + Pipeline ✅

## Current Status: FULLY OPERATIONAL

Both training and pipeline are now running automatically!

### Running Processes

**Training Daemon:**
- PID: 14405
- Running: `continuous_training.py`
- Study: `cappuccino_fixed_norm_20260122`
- Mode: 1 trial per cycle, 5min cooldown
- GPU: 0
- Log: `logs/continuous_training.log`

**Pipeline Daemon:**
- PID: 11807
- Running: `pipeline_orchestrator.py --daemon`
- Checking every 30 minutes
- Log: `logs/pipeline_orchestrator.log`

## Complete Workflow (Now Automated)

```
[AUTOMATED] Continuous Training
    ↓ Runs 1 trial
    ↓ Saves to Optuna database
    ↓ Preserves model weights
    ↓ Triggers pipeline check
    ↓
[AUTOMATED] Pipeline Orchestrator
    ↓ Detects new trial
    ↓ Gate 1: Backtest validation
    ↓ Gate 2: CGE stress test (200 scenarios)
    ↓ Gate 3: Deploy to paper trading
    ↓ 7 days Model Arena evaluation
    ↓ Grade A/B → Auto-promote to live
    ↓
[AUTOMATED] Live Trading
```

## Management Commands

### Check Status
```bash
./status_automation.sh
```

### Monitor Logs

**Training:**
```bash
tail -f logs/continuous_training.log
```

**Pipeline:**
```bash
tail -f logs/pipeline_orchestrator.log
```

**Both:**
```bash
tail -f logs/continuous_training.log logs/pipeline_orchestrator.log
```

### Stop Everything
```bash
./stop_automation.sh
```

### Restart Everything
```bash
./stop_automation.sh
./start_full_automation.sh
```

## What Happens Automatically

### Training Cycle (Every ~30-60 minutes)

1. **Continuous trainer** runs 1 Optuna trial
2. Trial completes, saves to database
3. Model weights preserved in `train_results/cwd_tests/trial_XXX_1h/`
4. **Triggers pipeline check**

### Pipeline Check (Every 30 minutes + after training)

1. **Scans Optuna database** for new completed trials
2. **Finds trials with saved weights**
3. **Runs backtest** on validation data
4. **Validates** against adaptive thresholds
5. If passed: **Runs CGE stress test** (200 scenarios)
6. If passed: **Deploys to paper trading** (Model Arena)
7. After 7 days: **Checks grade** (A/B required)
8. If grade A/B: **Auto-promotes to live trading**

## Configuration

### Training Settings

Edit `.env.training`:
```bash
ACTIVE_STUDY_NAME="cappuccino_fixed_norm_20260122"
TRAINING_WORKERS=10
GPU_ID=0
N_TRIALS=2000
```

### Pipeline Settings

Edit `config/pipeline_config.json`:
- Gate thresholds (backtest, CGE, paper trading)
- Auto-promotion rules
- Check intervals
- Notification preferences

### Continuous Training Settings

Modify `continuous_training.py` or pass arguments:
```bash
python continuous_training.py \
    --trials-per-cycle 3 \
    --cooldown 600 \
    --gpu 0
```

## Monitoring

### Real-time Status
```bash
watch -n 5 ./status_automation.sh
```

### Resource Usage
```bash
# CPU/Memory
top -p 14405,11807

# GPU
nvidia-smi
```

### Database Growth
```bash
sqlite3 databases/optuna_cappuccino.db \
  "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';"
```

### Pipeline Progress
```bash
cat deployments/pipeline_state.json | jq '.trials | length'
```

## Troubleshooting

### Training Not Starting

```bash
# Check logs
tail -20 logs/training_daemon.log

# Check process
ps aux | grep continuous_training

# Restart
./stop_automation.sh
./start_full_automation.sh
```

### Pipeline Not Processing Trials

**Most common: Old trials have weights cleaned up**

Solution: Wait for new trials from continuous training (they preserve weights automatically)

Or manually specify trials:
```bash
# Edit config/pipeline_config.json
{
  "pipeline": {
    "use_manual_trials": true,
    "manual_trials": [965]
  }
}
```

### Check Both Systems
```bash
./status_automation.sh
```

## Expected Behavior

### First Hour
- Training: 1-2 trials completed
- Pipeline: Checking every 30 min, may skip old trials
- No errors in logs

### First Day
- Training: 12-24 trials completed
- Pipeline: Processing new trials through backtesting
- Some trials pass, some fail gates (normal)

### First Week
- Training: 150-200 trials completed
- Pipeline: 10-20 trials through backtesting
- 3-5 trials through CGE stress test
- 1-2 trials deployed to paper trading

### First Month
- Training: 800-1000 trials completed
- Pipeline: Fully operational
- Multiple trials in Model Arena
- First auto-promotion to live trading

## Files Created

### Core Scripts
- `continuous_training.py` - Training daemon ✅
- `start_full_automation.sh` - Start both systems ✅
- `stop_automation.sh` - Stop both systems ✅
- `status_automation.sh` - Check status ✅

### Existing Pipeline
- `pipeline_orchestrator.py` - Pipeline daemon ✅
- `pipeline/*.py` - Pipeline modules ✅
- `config/pipeline_config.json` - Configuration ✅

### Documentation
- `AUTOMATION_COMPLETE.md` - This file ✅
- `PIPELINE_README.md` - Pipeline guide ✅
- `PIPELINE_TROUBLESHOOTING.md` - Problem solving ✅

## Success Indicators

✅ Both processes running (check with `./status_automation.sh`)
✅ Training logs show "Starting training cycle"
✅ Pipeline logs show "Pipeline check starting"
✅ New trials appearing in database
✅ Pipeline detecting and processing trials
✅ No critical errors in logs

## Next Steps

### Immediate (Done!)
1. ✅ Training running
2. ✅ Pipeline running
3. ✅ Both integrated

### Today
- Monitor logs for first few cycles
- Verify trials are completing successfully
- Check pipeline is detecting new trials

### This Week
- Watch first trials go through gates
- Tune thresholds if needed
- Monitor paper trading deployments

### This Month
- Review first live promotions
- Adjust automation parameters
- Optimize for your needs

---

## Summary

**You now have FULLY AUTOMATED trading system deployment:**

Training runs continuously → Pipeline validates automatically → 
Paper trading tests rigorously → Live trading deploys automatically

**Zero manual intervention required!** 🚀

Just monitor the logs and let the system work. When you have good models, they'll automatically flow from training all the way to live trading.

---

**Status: OPERATIONAL ✅**
**Training: RUNNING ✅**
**Pipeline: RUNNING ✅**
**Integration: COMPLETE ✅**
