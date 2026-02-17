# Automated Training System Guide

Complete guide for the VIN-based automated training pipeline that manages trials, archival, and deployment without manual intervention.

## 🎯 Overview

The automated training system provides:

1. **VIN-Like Trial Naming**: Human-readable codes that encode hyperparameters and performance
2. **Automatic Archival**: Top 10% of trials saved with models and metadata
3. **Automatic Cleanup**: Old logs and trials removed to save disk space
4. **Seamless Deployment**: Best models automatically deployed to paper trading
5. **No Manual Study Management**: Dynamic study naming and lifecycle management

## 📋 VIN Code Format

Each trial gets a unique VIN (Vehicle Identification Number-style) code:

```
Format: [TYPE]-[GRADE]-[ARCH]-[OPT]-[ENV]-[TIMESTAMP]

Example: PPO-A-N1024B4K-L2E6G98-LB5TD20-20260214
```

### Breakdown:

- **TYPE**: `PPO` - Model type (PPO, DDPG, etc.)
- **GRADE**: `A` - Performance grade based on Sharpe ratio
  - `S` = Elite (Sharpe ≥ 0.30) - Top 1%
  - `A` = Excellent (Sharpe ≥ 0.20) - Top 10%
  - `B` = Good (Sharpe ≥ 0.15) - Top 25%
  - `C` = Fair (Sharpe ≥ 0.10) - Top 50%
  - `D` = Poor (Sharpe ≥ 0.05) - Bottom 50%
  - `F` = Failed (Sharpe < 0.05)
- **ARCH**: `N1024B4K` - Architecture (Net dimension 1024, Batch 4K)
- **OPT**: `L2E6G98` - Optimizer (Learning rate 2e-6, Gamma 0.98)
- **ENV**: `LB5TD20` - Environment (Lookback 5, Time decay 0.20)
- **TIMESTAMP**: `20260214` - Training date (YYYYMMDD)

## 🚀 Quick Start

### Full Automated Pipeline

Run the complete pipeline (clean, train, archive, deploy):

```bash
python scripts/automation/automated_training_pipeline.py --mode full
```

This will:
1. Clean logs older than 7 days
2. Remove old trial directories (keeps top 20)
3. Start 3 training workers
4. Wait for completion
5. Archive top 10% of trials
6. Deploy best model to paper trading

### Training Only (Background)

Start training without waiting for completion:

```bash
python scripts/automation/automated_training_pipeline.py \
    --mode training \
    --background \
    --workers 3 \
    --trials 500
```

### Archive Existing Trials

Archive top performers from completed training:

```bash
python scripts/automation/automated_training_pipeline.py --mode archive
```

### Deploy Best Model

Deploy the best archived model to paper trading:

```bash
python scripts/automation/automated_training_pipeline.py --mode deploy
```

### Monitor Progress

Real-time dashboard showing trials, workers, and paper trading:

```bash
python scripts/automation/trial_dashboard.py
```

Display once (no auto-refresh):

```bash
python scripts/automation/trial_dashboard.py --once
```

## 📊 Manual Trial Management

### List Archived Trials

```bash
python utils/trial_manager.py --list
```

### Archive Specific Study

```bash
python utils/trial_manager.py \
    --archive \
    --study-name cappuccino_auto_20260214_1530 \
    --top-percent 10
```

### Clean Old Data

```bash
# Dry run (see what would be deleted)
python utils/trial_manager.py --clean-trials --clean-logs --dry-run

# Actually clean
python utils/trial_manager.py --clean-trials --clean-logs
```

## 📁 Directory Structure

```
cappuccino/
├── trial_archive/              # Archived trials
│   ├── PPO-A-N1024B4K-L2E6G98-LB5TD20-20260214/
│   │   ├── metadata.json       # Full trial metadata
│   │   ├── hyperparams.json    # Hyperparameters
│   │   ├── actor.pth          # Trained model
│   │   └── ...
│   └── trial_registry.json     # Global trial registry
├── deployments/                # Deployed models
│   ├── model_0/               # Deployment slot 0
│   │   ├── actor.pth
│   │   ├── trial_metadata.json
│   │   └── deployment_info.json
│   └── ...
├── logs/                       # Training logs
│   ├── worker_auto_1.log
│   ├── worker_auto_2.log
│   └── ...
└── .current_study              # Current study name
```

## 🔧 Configuration

### Pipeline Options

```bash
python scripts/automation/automated_training_pipeline.py \
    --mode full \
    --workers 3 \              # Number of parallel workers
    --trials 500 \             # Trials per worker
    --gpu 0 \                  # GPU device ID
    --data-dir data/1h_1680 \  # Training data
    --timeframe 1h \           # Timeframe
    --top-percent 10 \         # Archive top 10%
    --deployment-slot 0        # Deploy to slot 0
```

### Trial Manager Options

```bash
python utils/trial_manager.py \
    --archive \                # Archive trials
    --clean-trials \           # Clean old trials
    --clean-logs \             # Clean old logs
    --study-name my_study \    # Specific study
    --top-percent 10 \         # Top percentage
    --keep-days 7 \            # Days to keep
    --dry-run                  # Preview only
```

## 🎨 Example Workflows

### Daily Training Cycle

```bash
# Morning: Start fresh training
python scripts/automation/automated_training_pipeline.py \
    --mode training \
    --background \
    --workers 3 \
    --trials 300

# Evening: Archive and deploy
python scripts/automation/automated_training_pipeline.py --mode archive
python scripts/automation/automated_training_pipeline.py --mode deploy
```

### Weekly Full Cycle

```bash
# Sunday night: Full automated run
python scripts/automation/automated_training_pipeline.py \
    --mode full \
    --workers 3 \
    --trials 500
```

### Monitor Training Progress

```bash
# Terminal 1: Dashboard
python scripts/automation/trial_dashboard.py

# Terminal 2: Worker logs
tail -f logs/worker_auto_*.log

# Terminal 3: Best trials
watch -n 30 'python utils/trial_manager.py --list | head -20'
```

## 🏆 Understanding Grades

Grades are assigned based on Sharpe ratio performance:

| Grade | Sharpe Range | Percentile | Meaning |
|-------|-------------|------------|---------|
| S     | ≥ 0.30      | Top 1%     | Elite performer - exceptional results |
| A     | ≥ 0.20      | Top 10%    | Excellent - production ready |
| B     | ≥ 0.15      | Top 25%    | Good - viable for ensemble |
| C     | ≥ 0.10      | Top 50%    | Fair - needs improvement |
| D     | ≥ 0.05      | Bottom 50% | Poor - not recommended |
| F     | < 0.05      | Failed     | Failed - discard |

**Default Archival**: Only trials with grade A or better (top 10%) are archived by default.

## 📈 Integration with Paper Trading

The automated pipeline seamlessly integrates with paper trading:

1. **Training** → Best trials archived with VIN codes
2. **Archival** → Top 10% models saved with metadata
3. **Deployment** → Best model (grade S or A) deployed to slot 0
4. **Dashboard** → Monitor both training and live trading

### Deploy to Multiple Slots

```bash
# Deploy best model to slot 0 (primary)
python scripts/automation/automated_training_pipeline.py --mode deploy --deployment-slot 0

# Deploy 2nd best to slot 1 (backup)
# (requires manual trial selection)
```

## 🔍 Troubleshooting

### No Trials Being Archived

**Issue**: Archive step finds no trials meeting criteria.

**Solution**:
```bash
# Check recent trials
python utils/trial_manager.py --list

# Lower minimum Sharpe threshold
# Edit trial_manager.py: min_sharpe = 0.05 (default is 0.10)

# Or increase top percentage
python scripts/automation/automated_training_pipeline.py --mode archive --top-percent 20
```

### Workers Not Starting

**Issue**: Training workers fail to start.

**Solution**:
```bash
# Check logs
tail -100 logs/worker_auto_1.log

# Verify data directory exists
ls -la data/1h_1680/

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Disk Space Issues

**Issue**: Running out of disk space from trials/logs.

**Solution**:
```bash
# Aggressive cleanup (keep only 3 days)
python utils/trial_manager.py --clean-trials --clean-logs --keep-days 3

# Remove old experiments directory
rm -rf experiments/trial_*
```

## 🎯 Best Practices

1. **Run Full Pipeline Weekly**: Complete training cycle every Sunday
2. **Archive Daily**: Archive new trials every evening
3. **Monitor Dashboard**: Keep dashboard running to spot issues
4. **Backup Archives**: Periodically backup `trial_archive/` directory
5. **Deploy Conservatively**: Only deploy grade A or S models to production
6. **Clean Regularly**: Run cleanup weekly to manage disk space

## 📝 Advanced Usage

### Custom Grading Thresholds

Edit `utils/trial_naming.py`:

```python
def sharpe_to_grade(sharpe: float) -> str:
    """Customize these thresholds."""
    if sharpe >= 0.35:  # Increase S grade threshold
        return 'S'
    elif sharpe >= 0.25:  # Increase A grade threshold
        return 'A'
    # ... etc
```

### Integration with Automation Scripts

The automated pipeline can be called from existing automation:

```bash
# In start_automation.sh
python scripts/automation/automated_training_pipeline.py --mode full &

# In your cron job
0 2 * * 0 cd /opt/user-data/experiment/cappuccino && \
    python scripts/automation/automated_training_pipeline.py --mode full >> logs/auto_training.log 2>&1
```

## 🚦 Status Indicators

The dashboard uses these indicators:

- 🏆 **S Grade**: Elite trial (top 1%)
- ⭐ **A Grade**: Excellent trial (top 10%)
- ✅ **B Grade**: Good trial
- 🔵 **C Grade**: Fair trial
- ⚠️ **D Grade**: Poor trial
- ❌ **F Grade**: Failed trial

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review trial registry: `trial_archive/trial_registry.json`
3. Run diagnostic: `python utils/trial_manager.py --list`
4. Check GitHub issues (if applicable)
