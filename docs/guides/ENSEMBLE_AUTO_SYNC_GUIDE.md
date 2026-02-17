# Ensemble Auto-Sync Guide

## Overview

Your paper trading ensemble now **automatically stays in sync** with your active training study. When you start new training runs, the ensemble and automation will automatically use the same study.

## What Was Fixed

### Before (Mismatched)
- **Training**: `cappuccino_1year_20251121` (872 trials, best: 0.01523)
- **Paper Trading Ensemble**: `cappuccino_trailing_20251125` (894 trials, best: 0.01152)
- **Problem**: 32% performance gap, ensemble using outdated study

### After (Aligned)
- **Training**: `cappuccino_1year_20251121` ✓
- **Paper Trading Ensemble**: `cappuccino_1year_20251121` ✓
- **Auto-Deployer**: `cappuccino_1year_20251121` ✓
- **Ensemble Updater**: Running (syncs top 10 every 10 min) ✓
- **Result**: All systems aligned and auto-syncing!

## Current Ensemble

**Study**: `cappuccino_1year_20251121`
**Models**: 10 (top performers)
**Best Trial**: #795 (value: 0.01523)
**Mean Value**: 0.01248

### Ensemble Models

| Trial # | Value    | Trailing Stop |
|---------|----------|---------------|
| 795     | 0.01523  | No            |
| 875     | 0.01305  | **12%**       |
| 756     | 0.01263  | No            |
| 737     | 0.01255  | No            |
| 649     | 0.01225  | No            |
| 845     | 0.01224  | **11%**       |
| 612     | 0.01185  | No            |
| 680     | 0.01182  | No            |
| 682     | 0.01172  | No            |
| 686     | 0.01149  | No            |

**Note**: Only 2/10 models were trained with trailing stop parameters. The ensemble uses a 10% trailing stop at the paper trading level.

## Automation Components

### 1. **Ensemble Auto-Updater** (NEW!)
- **What**: Keeps ensemble synced with top 10 trials
- **Frequency**: Every 10 minutes
- **Process**:
  - Queries database for top 10 trials
  - Adds new top performers
  - Removes underperformers
  - Signals paper trader to reload (hot reload)
- **PID**: Check with `./status_automation.sh`

### 2. **Auto-Model Deployer**
- **What**: Monitors for new best single models
- **Frequency**: Every hour
- **Note**: Currently has strict validation (disabled for now)

### 3. **System Watchdog**
- **What**: Restarts crashed processes
- **Monitors**: Training workers, paper trading, advisor

### 4. **Performance Monitor**
- **What**: Tracks training progress and alerts

## Starting New Training (Automated!)

Use the new automated training script:

```bash
./start_training.sh --study cappuccino_new_20251129 --n-trials 1000 --workers 3
```

This will **automatically**:
1. ✅ Update ensemble with top 10 trials from the study
2. ✅ Update automation scripts to use the study
3. ✅ Start training workers
4. ✅ Deploy ensemble to paper trading
5. ✅ Start ensemble auto-updater

### Manual Sync (if needed)

If you need to manually sync to a different study:

```bash
./sync_training_study.sh <study_name>
```

Example:
```bash
./sync_training_study.sh cappuccino_1year_20251121
```

## Monitoring

### Check Status
```bash
./status_automation.sh
```

### View Logs
```bash
# Training workers
tail -f logs/training_worker_*.log

# Paper trading (ensemble)
tail -f logs/paper_trading_ensemble.log

# Ensemble auto-updater
tail -f logs/ensemble_updater_console.log

# Auto-deployer
tail -f logs/auto_deployer.log
```

### Check Ensemble Composition
```bash
cat train_results/ensemble/ensemble_manifest.json
```

## How Auto-Sync Works

### When Training is Running

1. **Every 10 minutes**: Ensemble auto-updater checks for new top trials
   - If new trial enters top 10 → Added to ensemble
   - If trial drops out of top 10 → Removed from ensemble
   - Paper trader hot-reloads models (no restart needed)

2. **Every hour**: Auto-deployer checks for significant improvements
   - If a single model is much better → Can deploy it individually
   - (Currently disabled due to strict validation)

3. **Continuous**: Watchdog monitors all processes
   - If paper trader crashes → Auto-restart
   - If training worker crashes → Auto-restart (up to 3 times)

### When Starting New Training

Using `./start_training.sh`:
- Automatically syncs ensemble to the new study
- Updates all automation scripts
- Starts workers and paper trading
- Everything stays aligned from the start

## File Structure

### New Scripts
- `start_training.sh` - Start training with auto-sync
- `sync_training_study.sh` - Manual sync to a study
- `start_automation.sh` - Start automation (updated to include ensemble updater)
- `stop_automation.sh` - Stop automation (updated)
- `status_automation.sh` - Check status (updated)

### Ensemble Directory
```
train_results/ensemble/
├── ensemble_manifest.json    # Ensemble metadata
├── model_0/                   # Top trial (795)
│   ├── actor.pth
│   └── best_trial
├── model_1/                   # 2nd best (875)
│   ├── actor.pth
│   └── best_trial
├── ...
└── model_9/                   # 10th best (686)
```

## Stopping Everything

```bash
# Stop automation
./stop_automation.sh

# Stop training workers
pkill -f 1_optimize_unified.py

# Stop paper trading
pkill -f paper_trader_alpaca_polling.py
```

## Troubleshooting

### Ensemble not updating?
```bash
# Check ensemble updater is running
./status_automation.sh

# Check logs
tail -f logs/ensemble_updater_console.log

# Manually sync
./sync_training_study.sh cappuccino_1year_20251121
```

### Paper trader using wrong study?
```bash
# Check which study ensemble is using
cat train_results/ensemble/ensemble_manifest.json | grep study_name

# Resync
./sync_training_study.sh <correct_study_name>
```

### Training and ensemble mismatched?
```bash
# Find active training study
ps aux | grep 1_optimize_unified.py | grep study-name

# Sync ensemble to it
./sync_training_study.sh <study_from_above>
```

## Benefits

✅ **No More Mismatches**: Training and paper trading always use the same study
✅ **Always Up-to-Date**: Ensemble refreshes with top performers every 10 minutes
✅ **Automatic**: Set it and forget it - just start training
✅ **Hot Reload**: Paper trader reloads models without restart
✅ **Easy to Start**: One command starts everything aligned

## Next Steps

1. **Current training** continues on `cappuccino_1year_20251121`
   - Ensemble auto-updates every 10 min
   - Paper trading uses latest top 10 models

2. **When starting new training**:
   ```bash
   ./start_training.sh --study cappuccino_new_study --n-trials 1000
   ```
   Everything auto-syncs!

3. **Monitor progress**:
   ```bash
   ./status_automation.sh
   watch -n 60 cat train_results/ensemble/ensemble_manifest.json
   ```

---

**Generated**: 2025-11-29
**Study**: cappuccino_1year_20251121
**Ensemble**: 10 models (mean: 0.01248, best: 0.01523)
