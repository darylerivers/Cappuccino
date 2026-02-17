# Cappuccino Quick Reference

## 🚀 Start Here (First Time)

```bash
# 1. Run validation suite
./run_validation.sh

# 2. Check current status
./status_automation.sh

# 3. Read essential context
cat CRITICAL_CONTEXT.md
```

---

## 📁 Key Files (What & Where)

### Configuration
- `constants.py` - ALL magic numbers (change thresholds here)
- `config_main.py` - Global settings (tickers, timeframes)

### Trading
- `paper_trader_alpaca_polling.py` - Main trading loop
- `environment_Alpaca.py` - RL environment & reward
- `paper_trading_failsafe.sh` - Auto-restart wrapper

### Models
- `ultra_simple_ensemble.py` - 10-model voting
- `adaptive_ensemble_agent.py` - Game theory voting
- `train_results/ensemble/` - Top 10 model checkpoints

### Training
- `1_optimize_unified.py` - Hyperparameter search
- `2_validate.py` - Out-of-sample eval
- `4_backtest.py` - Historical replay

### Testing & Baselines
- `tests/test_critical.py` - 25 critical tests
- `baselines/buy_and_hold.py` - Passive baseline
- `run_validation.sh` - Run all checks

### Monitoring
- `system_watchdog.py` - Auto-restart crashed processes
- `auto_model_deployer.py` - Deploy top models
- `performance_monitor.py` - Track metrics

### Documentation
- `CRITICAL_CONTEXT.md` - Daily essential (3KB)
- `PROJECT_OVERVIEW_FOR_OPUS.md` - Full review (33KB)
- `contexts/` - Focused topic docs

---

## 🎯 Common Tasks

### Check System Status
```bash
./status_automation.sh                   # Basic status
python dashboard_optimized.py            # Full dashboard with optimizations
python dashboard_optimized.py --once     # Single view (no refresh)
```

### View Recent Trades
```bash
tail -50 paper_trades/alpaca_session.csv
```

### Monitor Profit Protection
```bash
tail -f paper_trades/profit_protection.log
```

### Check for Crashes
```bash
tail -100 logs/paper_trading_failsafe.log
```

### Run Tests
```bash
pytest tests/test_critical.py -v
```

### Run Baseline Comparison
```bash
python baselines/buy_and_hold.py --data data/price_array_val.npy
```

### Start/Stop Trading
```bash
./start_automation.sh   # Start all systems
./stop_automation.sh    # Stop all systems
```

---

## 🔧 Configuration (Quick Tweaks)

### Adjust Risk Thresholds
**File**: `constants.py`
```python
# Edit these lines:
STOP_LOSS_PCT: float = 0.10              # Currently 10%
PORTFOLIO_TRAILING_STOP_PCT: float = 0.015  # Currently 1.5%
PROFIT_TAKE_THRESHOLD_PCT: float = 0.03     # Currently 3%
```

### Change Trading Tickers
**File**: `config_main.py`
```python
TICKER_LIST = ['BTC/USD', 'ETH/USD', 'LTC/USD', ...]
```

### Adjust Trading Frequency
**File**: `paper_trading_failsafe.sh`
```bash
POLL_INTERVAL="${3:-60}"  # Change 60 to different seconds
```

---

## 📊 Performance Metrics

### Check Optuna Database
```python
import optuna
study = optuna.load_study(
    "cappuccino_1year_20251121",
    "sqlite:///databases/optuna_cappuccino.db"
)
print(f"Best trial: {study.best_value}")
print(f"Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
```

### View Paper Trading Performance
```python
import pandas as pd
df = pd.read_csv("paper_trades/alpaca_session.csv")
print(f"Total return: {(df['total_asset'].iloc[-1] / df['total_asset'].iloc[0] - 1) * 100:.2f}%")
```

---

## 🐛 Debugging

### Check for Errors
```bash
grep -i "error\|exception\|failed" logs/paper_trading_live.log | tail -20
```

### Verify Data Quality
```bash
python -c "
import numpy as np
prices = np.load('data/price_array_val.npy')
print(f'Shape: {prices.shape}')
print(f'Has NaN: {np.isnan(prices).any()}')
print(f'Has zeros: {(prices == 0).any()}')
print(f'Min: {prices.min()}, Max: {prices.max()}')
"
```

### Check Process Memory
```bash
ps aux | grep -E "paper_trader|watchdog|deployer" | awk '{print $6/1024 " MB", $11}'
```

---

## ⚡ Quick Fixes

### Restart Paper Trader
```bash
pkill -f paper_trader_alpaca_polling
# Failsafe will auto-restart, or manually:
./paper_trading_failsafe.sh
```

### Clear Old Logs (Free Disk Space)
```bash
# Archive logs older than 7 days
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;
```

### Reset Profit Protection State
```bash
# Delete log to start fresh
rm paper_trades/profit_protection.log
```

---

## 🔍 Finding Functions (Code Navigation)

**Use**: `code_maps/function_index.md`

```bash
# Quick lookup
grep -n "_apply_risk_management" code_maps/function_index.md
# Output: paper_trader_alpaca_polling.py - line 699
```

---

## 📚 Learning Resources

### Understand Risk Management
```bash
cat contexts/risk_management.md
```

### See Prioritized Tasks
```bash
cat contexts/quick_wins.md
```

### Full System Overview
```bash
cat PROJECT_OVERVIEW_FOR_OPUS.md
```

---

## ⚠️ Known Issues (Check Before Debugging)

1. **Data Quality** - Alpaca occasionally returns incomplete bars
   - Check: Data quality checks in `paper_trader_alpaca_polling.py:600-613`

2. **Complex Reward** - 5+ components, may be hard to optimize
   - File: `environment_Alpaca.py:224-280`

3. **Hyperparameter Search** - 20+ params, 150 trials may not be enough
   - Check importance: `optuna.visualization.plot_param_importances(study)`

4. **Transaction Costs** - 0.5% per round-trip eats profits
   - Test action dampening: `--action-dampening 0.5`

5. **Profit Protection Untested** - Added Nov 24, needs validation
   - Run: `pytest tests/test_critical.py::TestProfitProtection -v`

---

## 🆘 Emergency Commands

### Stop Everything
```bash
./stop_automation.sh
pkill -f "paper_trader\|watchdog\|deployer\|monitor"
```

### Check if System is Healthy
```bash
# All should show PIDs
pgrep -f paper_trader
pgrep -f watchdog
pgrep -f auto_model_deployer
```

### Rollback to Previous Model
```bash
# Find old ensemble backup
ls -lt train_results/ensemble.backup.*
# Restore
mv train_results/ensemble train_results/ensemble.broken
mv train_results/ensemble.backup.YYYYMMDD train_results/ensemble
```

---

## 📞 Getting Help

1. **Check logs first**: `logs/paper_trading_live.log`
2. **Read critical context**: `CRITICAL_CONTEXT.md`
3. **Run tests**: `pytest tests/test_critical.py -v`
4. **Search for similar issues**: `grep -r "error message" logs/`

---

## 🎓 Best Practices

### Before Making Changes
1. Read relevant context file (`contexts/*.md`)
2. Check function location (`code_maps/function_index.md`)
3. Run tests before & after (`pytest tests/test_critical.py`)

### When Tuning Parameters
1. Change in `constants.py` (not scattered in code)
2. Document reason for change
3. Test on validation data first

### When Debugging
1. Check logs chronologically
2. Verify data quality
3. Isolate issue with tests
4. Fix and add regression test

---

**Updated**: 2025-11-24
**Status**: ✅ All optimizations applied
**Next**: Run `./run_validation.sh`
