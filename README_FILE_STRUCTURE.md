# Cappuccino File Structure (Reorganized Feb 2026)

## Quick Start Scripts (Root Directory)

```bash
./start_everything.sh    # Launch all automation + training
./stop_everything.sh     # Stop all processes
```

## Organized Directory Structure

### 📁 `scripts/`
Main executable scripts organized by function

#### `scripts/training/`
- **1_optimize_unified.py** - Main DRL training with Optuna
- **pipeline_v2.py** - Training pipeline orchestrator
- **continuous_training.py** - Continuous training scheduler
- **rerun_best_trial_v2.py** - Rerun specific trials

#### `scripts/deployment/`
- **auto_model_deployer.py** - Automatically deploy best models
- **paper_trader_alpaca_polling.py** - Live paper trading
- **export_trial_for_trading.py** - Export trial for deployment

#### `scripts/automation/`
- **start_everything.sh** - Launch complete system
- **stop_everything.sh** - Stop all processes
- **start_automation.sh** - Start automation only
- **status_automation.sh** - Check automation status
- **training_control.sh** - Training management
- **watch_training.py** - Monitor training progress

### 📁 `monitoring/`
System monitoring and dashboards

- **system_watchdog.py** - Process health monitoring
- **performance_monitor.py** - Trading performance tracking
- **dashboard.py** - Main monitoring dashboard
- **check_status.sh** - Quick status check

### 📁 `models/`
Agent implementations and ensemble systems

- **ensemble_auto_updater.py** - Auto-update ensemble models
- **simple_ensemble_agent.py** - Basic ensemble
- **multi_timeframe_ensemble_agent.py** - Multi-timeframe ensemble
- **model_arena.py** - Model competition arena

### 📁 `config/`
Configuration files

- **discord.py** - Discord bot configuration (created by you)
- **pipeline_v2_config.json** - Pipeline settings

### 📁 `integrations/`
External service integrations

- **discord_notifier.py** - Discord notification helper

### 📁 `pipeline/`
Pipeline components

- **backtest_runner.py** - Backtesting executor
- **state_manager.py** - Pipeline state management
- **gates.py** - Quality gates
- **notifications.py** - Notification system

### 📁 `drl_agents/`
Deep RL agent implementations (unchanged)

- PPO, DDPG, A2C implementations
- Network architectures
- Training utilities

## File Locations Quick Reference

### Training
```
scripts/training/1_optimize_unified.py     # Main training script
scripts/training/pipeline_v2.py            # Pipeline orchestrator
```

### Deployment
```
scripts/deployment/auto_model_deployer.py         # Auto deployer
scripts/deployment/paper_trader_alpaca_polling.py # Paper trader
```

### Monitoring
```
monitoring/system_watchdog.py      # Process monitor
monitoring/performance_monitor.py  # Performance tracker
monitoring/dashboard.py            # Main dashboard
```

### Automation
```
scripts/automation/start_everything.sh  # Launch system
scripts/automation/stop_everything.sh   # Stop system
scripts/automation/training_control.sh  # Training manager
```

## Deprecated Files (Archived)

Moved to `archive/` during cleanup:
- Old training variants
- Test scripts
- Deprecated dashboards
- Duplicate analysis tools

## Common Tasks

### Start Full System
```bash
./start_everything.sh
```

This launches in separate windows:
1. Pipeline V2 (auto-deployment)
2. Auto-Model Deployer
3. System Watchdog
4. Performance Monitor
5. Ensemble Auto-Updater
6. Training workers (3x)

### Start Only Training
```bash
source .env.training
python -u scripts/training/1_optimize_unified.py \
  --n-trials 1000 --gpu 0 \
  --study-name "$ACTIVE_STUDY_NAME" \
  > logs/worker_1.log 2>&1 &
```

### Start Only Automation
```bash
scripts/automation/start_automation.sh
```

### Monitor System
```bash
python monitoring/dashboard.py  # Interactive dashboard
monitoring/check_status.sh      # Quick CLI status
```

### Paper Trading
```bash
python scripts/deployment/paper_trader_alpaca_polling.py \
  --deployment-dir deployments/trial_XXX_live
```

## Environment Files

- **.env** - Discord credentials (DO NOT COMMIT)
- **.env.training** - Training configuration
- **.env.discord.template** - Discord template

## Logs Directory

All logs go to `logs/`:
- `worker_*.log` - Training workers
- `auto_deployer.log` - Model deployer
- `watchdog.log` - System watchdog
- `performance_monitor.log` - Performance tracker
- `pipeline_v2.log` - Pipeline

## Database Files

- `databases/optuna_cappuccino.db` - Main Optuna study
- `pipeline_v2.db` - Pipeline state

## Training Results

- `train_results/` - Model checkpoints
- `deployments/` - Deployed models
- `paper_trades/` - Paper trading state

## Why Reorganized?

**Before:** 300+ files in root directory (chaos)
**After:** Organized into logical folders (clean)

Benefits:
- Easy to find files
- Clear separation of concerns
- Better for git management
- Easier onboarding
