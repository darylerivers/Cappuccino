# Cappuccino Trading System - Centralized Configuration

**Last Updated:** 2025-12-06
**Status:** ✅ Fully Operational

---

## 🎯 Quick Start

### Check System Status
```bash
./verify_system.sh
```

### Start Everything
```bash
# Start training (if not already running)
./start_training.sh

# Start automation systems
./start_automation.sh
```

### Stop Everything
```bash
./stop_automation.sh
pkill -f 1_optimize_unified.py
```

---

## 📋 Centralized Configuration

**All system components now read from:** `.env.training`

This is the **single source of truth** for which study is active.

### Current Configuration
```bash
ACTIVE_STUDY_NAME="cappuccino_1year_20251121"
```

### To Switch Studies
1. Edit `.env.training` and change `ACTIVE_STUDY_NAME`
2. Stop all systems: `./stop_automation.sh && pkill -f 1_optimize_unified.py`
3. Restart: `./start_training.sh` and `./start_automation.sh`
4. Verify: `./verify_system.sh`

---

## 🔧 System Components

### 1. Training Workers (3 workers)
- **Study:** cappuccino_1year_20251121
- **Status:** ✅ Running
- **Trials:** 1,369 complete, 38 running
- **Command:** Reads from `.env.training`
- **Logs:** `logs/training_worker_*.log`

### 2. Auto-Model Deployer
- **Study:** cappuccino_1year_20251121
- **Status:** ✅ Running (PID: 11328)
- **Function:** Monitors for new best trials, validates, deploys to paper trading
- **Check Interval:** Every 3600s (1 hour)
- **Min Improvement:** 1.0% to deploy
- **Logs:** `logs/auto_deployer.log`

### 3. System Watchdog
- **Status:** ✅ Running (PID: 11351)
- **Function:** Monitors and restarts crashed processes
- **Check Interval:** Every 60s
- **Logs:** `logs/watchdog.log`

### 4. Performance Monitor
- **Study:** cappuccino_1year_20251121
- **Status:** ✅ Running (PID: 11361)
- **Function:** Tracks training metrics and performance
- **Check Interval:** Every 300s (5 min)
- **Logs:** `logs/performance_monitor.log`

### 5. Ensemble Auto-Updater
- **Study:** cappuccino_1year_20251121
- **Status:** ✅ Running (PID: 11398)
- **Function:** Keeps ensemble synced with top 20 models
- **Update Interval:** Every 600s (10 min)
- **Logs:** `logs/ensemble_updater_console.log`

### 6. Paper Trader
- **Model:** Ensemble (train_results/ensemble)
- **Status:** ✅ Running
- **Tickers:** BTC/USD, ETH/USD, LTC/USD, BCH/USD, LINK/USD, UNI/USD, AAVE/USD
- **Timeframe:** 1h
- **Risk Management:**
  - Max position: 30% per asset
  - Stop-loss: 10% from entry
- **Logs:** `logs/paper_trading_ensemble.log`

---

## 🔍 Verification & Troubleshooting

### Run System Verification
```bash
./verify_system.sh
```

Expected output:
```
✓ All components configured correctly!
Active Study: cappuccino_1year_20251121
```

### Common Issues

#### ❌ Components using wrong study
**Fix:**
```bash
./stop_automation.sh
./start_automation.sh
./verify_system.sh
```

#### ❌ Training workers on multiple studies
**Fix:**
```bash
# Kill all training workers
pkill -f 1_optimize_unified.py

# Restart with correct study
./start_training.sh
```

#### ❌ Paper trading not deploying new models
**Check:**
```bash
tail -50 logs/auto_deployer.log
```

Common reasons:
- Model files not found (trial directory doesn't exist yet)
- Improvement < 1.0% threshold
- Validation failed

---

## 📊 Current Status (2025-12-06 20:07)

### Active Study: cappuccino_1year_20251121
- **Total Trials:** 1,369 complete, 38 running
- **Trials with Model Files:** 284 (out of 1,369)
- **Best Trial with Files:** Trial 1400 (value: 0.009298)
- **Ensemble:** ✅ 20 models loaded successfully

### Ensemble Status
- **Models Loaded:** 20/20 ✅
- **Top Trial:** Trial 1400 (0.009298)
- **Trial Range:** 216-1400
- **Last Updated:** 2025-12-06 20:07

### Issue Resolved (2025-12-06)
**Problem:** Ensemble showed 0 models despite 1,369 completed trials
**Root Cause:**
1. Top trials by value (795, 994, etc.) had their model files deleted
2. Ensemble updater only fetched top 40 trials, none had files
3. Required fetching top 1000 trials to find 201 with model files

**Fix Applied:**
- Modified `ensemble_auto_updater.py` to fetch `top_n * 50` trials
- Added filtering to only use trials with existing model files
- Successfully loaded 20 best available models

---

## 📝 File Structure

### Configuration Files
- `.env.training` - Centralized configuration (SINGLE SOURCE OF TRUTH)
- `start_automation.sh` - Start all automation systems (uses .env.training)
- `start_training.sh` - Start training workers (uses .env.training)
- `stop_automation.sh` - Stop automation
- `verify_system.sh` - Verify all components are synchronized

### Deployment State
- `deployments/deployment_state.json` - Current deployment info
- `deployments/archive/` - Historical deployment backups

### Logs
- `logs/auto_deployer.log` - Auto-deployment events
- `logs/watchdog.log` - System watchdog events
- `logs/performance_monitor.log` - Performance metrics
- `logs/ensemble_updater_console.log` - Ensemble sync events
- `logs/training_worker_*.log` - Training worker output
- `logs/paper_trading_ensemble.log` - Paper trading activity

---

## 🚀 Best Practices

### 1. Always verify before making changes
```bash
./verify_system.sh
```

### 2. Use centralized config
- **Never** hardcode study names in scripts
- **Always** update `.env.training` first
- Restart systems after config changes

### 3. Monitor deployment logs
```bash
tail -f logs/auto_deployer.log
```

### 4. Archive old studies
When starting a new study:
1. Update `.env.training`
2. Stop all systems
3. Archive old deployment state (automatically done)
4. Restart with new study

### 5. Check ensemble sync
```bash
tail -f logs/ensemble_updater_console.log
```

Ensemble should sync top 20 models every 10 minutes.

---

## 🔄 Migration from Old System

### What Changed (2025-12-06)

#### Before:
- Study names hardcoded in multiple scripts
- Training: `cappuccino_fresh_20251204_100527`
- Paper trading: Using wrong study's models
- Multiple orphan training workers

#### After:
- **Single config file:** `.env.training`
- **All components synchronized:** `cappuccino_1year_20251121`
- **Orphan workers killed:** Only 3 workers on correct study
- **Verification script:** `./verify_system.sh`

### Archived Data
- Old deployment state: `deployments/archive/deployment_state_backup_*.json`
- Old study still in database (no data lost)

---

## 📞 Quick Reference

```bash
# Check everything
./verify_system.sh

# Start/stop automation
./start_automation.sh
./stop_automation.sh

# Start/stop training
./start_training.sh
pkill -f 1_optimize_unified.py

# Monitor logs
tail -f logs/auto_deployer.log
tail -f logs/paper_trading_ensemble.log
tail -f logs/training_worker_1.log

# Check current study
cat .env.training | grep ACTIVE_STUDY_NAME

# View deployment history
cat deployments/deployment_state.json | jq '.deployment_history[-5:]'
```

---

## ✅ System Health Checklist

- [ ] `./verify_system.sh` passes
- [ ] All automation PIDs valid
- [ ] Training workers on correct study
- [ ] Auto-deployer monitoring correct study
- [ ] Ensemble updater syncing correct study
- [ ] Paper trader running
- [ ] Recent logs show activity

**Current Status:** ✅ All systems operational and synchronized
