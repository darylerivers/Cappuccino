# Repository Cleanup Summary

## Problem
- **342 total files** (Python, Shell, Markdown)
- **142 markdown files** with massive duplication
- Multiple deprecated versions of same scripts
- Test files from completed work
- Unrelated utilities (job search, hardware guides)

## Solution
Safe cleanup that **archives instead of deletes** - you can restore anything if needed.

## Quick Commands

### See what will be cleaned (dry run):
```bash
./cleanup_repo.sh --dry-run
```

### Actually clean up:
```bash
./cleanup_repo.sh
```

## What Gets Archived

### 1. Deprecated Training Scripts (~30 files)
- `train_maxvram.py`, `train_ensemble.py`, `launch_max_training.py`
- `phase1_timeframe_optimizer.py`, `phase2_feature_maximizer.py`
- All old `train_*.sh`, `launch_*.sh`, `start_fresh_*.sh` variants
- **Keep:** `1_optimize_unified.py`, `start_training.sh`

### 2. Test Scripts (~15 files)
- All `test_*.py` files (tests completed)
- All `stress_test_*.py` files
- All `debug_*.sh` files
- **Keep:** Nothing (tests served their purpose)

### 3. Dashboard Variants (6 files)
- `dashboard_backup_20251125_104511.py`
- `dashboard_optimized.py`, `dashboard_unified.py`
- `dashboard_ensemble_votes.py`, etc.
- **Keep:** `dashboard.py` (main production dashboard)

### 4. Old Pipeline Versions (3 files)
- `pipeline_orchestrator.py`, `pipeline_viewer.py`
- `deploy_v2.py`
- **Keep:** `pipeline_v2.py`, `deploy_to_arena.py`, `auto_model_deployer.py`

### 5. Duplicate Analysis Scripts (4 files)
- `analyze_training.py`, `analyze_test_results.py`
- `analyze_arena_trades.py`, `compare_trading_performance.py`
- **Keep:** `analyze_training_results.py`, `analyze_hyperparameters.py`

### 6. Unrelated Utilities (3 files)
- `job_hunter.py`, `job_apply_helper.py`, `job_search_quickstart.sh`
- **Keep:** Nothing (unrelated to trading)

### 7. Markdown Documentation (110+ files → ~20)

**Alert System** (3 → 1)
- Archive: ALERT_SYSTEM_DEPLOYED.md, ALERT_SYSTEM_README.md
- **Keep:** ALERT_SYSTEM_GUIDE.md

**Automation** (3 → 1)
- Archive: AUTOMATION_COMPLETE.md, AUTONOMOUS_OPERATION_FIXES_*.md
- **Keep:** Best variant or create consolidated AUTOMATION_GUIDE.md

**Dashboard** (5 → 1)
- Archive: DASHBOARD_USAGE.md, DASHBOARD_NAVIGATION.md, etc.
- **Keep:** DASHBOARD_README.md

**Ensemble** (3 → 1)
- Archive: ENSEMBLE_VOTING_DASHBOARD.md
- **Keep:** ENSEMBLE_AUTO_SYNC_GUIDE.md

**Pipeline** (6 → 1)
- Archive: PIPELINE_README.md, PIPELINE_STATUS.md, etc.
- **Keep:** PIPELINE_V2_DESIGN.md

**Training** (5+ → 1)
- Archive: TRAINING_MONITOR.md, TRAINING_STATUS_REPORT.md, etc.
- **Keep:** TRAINING_CONTROL_README.md

**System** (5 → 1)
- Archive: SYSTEM_CONFIGURATION.md, SYSTEM_REPORT_*.md, etc.
- **Keep:** SYSTEM_ARCHITECTURE.md

**Bug Reports/Fixes** (All archived)
- BUG_REPORT_*.md, CRITICAL_ISSUES_FOUND.md, ROOT_CAUSE_IDENTIFIED.md
- OVERFITTING_PROBLEM.md, ALPHA_DECAY_SOLUTION.md
- All dated status reports and fix summaries

**Unrelated Docs** (All archived)
- STORAGE_BUYING_GUIDE.md, EBAY_HDD_BUYING_GUIDE.md
- Hardware purchasing guides

## Archive Structure

After cleanup, files will be in:
```
archive/
├── deprecated_scripts/      # Old Python/Shell scripts
├── old_documentation/       # Consolidated markdown
├── test_scripts/           # Test files
├── dashboard_variants/     # Old dashboard versions
├── training_variants/      # Old training launchers
└── analysis_scripts/       # Duplicate analysis tools
```

## What Stays (Core ~60 files)

### Training
- `1_optimize_unified.py` - Main training
- `0_dl_trainval_data.py` - Data download
- `start_training.sh` - Training starter

### Trading
- `paper_trader_alpaca_polling.py` - Paper trading
- `model_arena.py` - Model evaluation
- `simple_ensemble_agent.py` - Ensemble trading
- `ensemble_agent.py` - Full ensemble
- `multi_timeframe_ensemble_agent.py` - Strategic ensemble

### Automation
- `pipeline_v2.py` - Pipeline orchestrator
- `auto_model_deployer.py` - Auto deployment
- `system_watchdog.py` - System monitor
- `performance_monitor.py` - Performance tracker
- `ensemble_auto_updater.py` - Ensemble updater

### Monitoring
- `dashboard.py` - Main dashboard
- `show_ensemble_votes.py` - Vote display
- `start_automation.sh` - Automation starter
- `status_automation.sh` - Status check

### Core Infrastructure
- `constants.py`, `environment_Alpaca.py`
- `function_CPCV.py`, `function_train_test.py`
- `drl_agents/` directory

## Expected Results

**Before Cleanup:**
- 342 files total
- 142 markdown files
- Difficult to find what you need

**After Cleanup:**
- ~150-180 files total
- ~20 essential markdown files
- Clear, organized structure

## Safety

- **Nothing is deleted** - everything goes to `archive/`
- You can restore any file: `mv archive/<category>/<file> .`
- Original files preserved in archive with same structure
- Dry run available to preview changes

## Restore Example

If you need something back:
```bash
# Restore a specific file
mv archive/deprecated_scripts/train_maxvram.py .

# Restore all test scripts
mv archive/test_scripts/* .

# Restore all documentation
mv archive/old_documentation/* .
```

## Estimated Impact

- **~160 files archived**
- **Repository 50% smaller**
- **100% functionality maintained**
- **Much easier to navigate**

## Run the Cleanup

1. **Preview first (dry run):**
   ```bash
   ./cleanup_repo.sh --dry-run
   ```

2. **Review what will move**

3. **Execute cleanup:**
   ```bash
   ./cleanup_repo.sh
   ```

4. **Done!** Repository is clean, everything archived safely.
