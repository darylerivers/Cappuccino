# Quick Wins - High-Impact, Low-Effort Tasks

## 1-Hour Tasks

### ✅ COMPLETED: Baseline Strategy
**File**: `baselines/buy_and_hold.py`
**Status**: Created, not run yet
**Action**:
```bash
# Requires price data file
python baselines/buy_and_hold.py --data data/price_array_val.npy
```
**Impact**: Validates if DRL adds value over passive investing

### ✅ COMPLETED: Test Suite
**File**: `tests/test_critical.py`
**Status**: Created, not run yet
**Action**:
```bash
pytest tests/test_critical.py -v
```
**Impact**: Catches bugs in profit protection (added today) and risk management

### ✅ COMPLETED: Constants Extraction
**File**: `constants.py`
**Status**: Created, not integrated yet
**Action**: Update imports in main files:
```python
# Replace scattered magic numbers with:
from constants import RISK, NORMALIZATION, TRADING
stop_loss = RISK.STOP_LOSS_PCT
```
**Impact**: Easier tuning, better readability

## 2-Hour Tasks

### Hyperparameter Importance Analysis
**Goal**: Find which hyperparameters actually matter
**Action**:
```python
import optuna
study = optuna.load_study(
    "cappuccino_1year_20251121",
    "sqlite:///databases/optuna_cappuccino.db"
)

# Plot importance
import optuna.visualization as vis
vis.plot_param_importances(study).show()

# Check if some params always near default
vis.plot_parallel_coordinate(study).show()
```
**Impact**: Narrow search space, faster convergence

### Validate Profit Protection on Historical Logs
**Goal**: Verify new profit protection would have worked
**Action**:
```python
# Read existing paper_trades/alpaca_session.csv
# Simulate profit protection logic
# Check if it would have locked in gains
```
**Impact**: Confidence before deploying live

### Check Transaction Cost Impact
**Goal**: Quantify how much fees eat profits
**Action**:
```python
# Parse paper trading logs
trades_per_day = count_trades()
costs_per_day = trades_per_day * 0.005  # 0.25% × 2
profit_lost_to_fees = costs_per_day / daily_returns

print(f"Fees eat {profit_lost_to_fees*100:.1f}% of profits")
```
**Impact**: Understand if action dampening needed

## 4-Hour Tasks

### Reward Function Ablation Study
**Goal**: Test which reward components help/hurt
**Action**:
```python
# Train 5 trials each with:
1. Returns only (no penalties)
2. Returns + transaction costs
3. Returns + costs + concentration penalty
4. Returns + costs + time decay
5. All components (current)

# Compare validation Sharpe ratios
```
**Impact**: Simplify reward, improve learning

### Implement Moving Average Baseline
**Goal**: Compare to simple momentum strategy
**Action**:
```python
# baselines/moving_average.py
# Buy when 10h MA > 50h MA, sell otherwise
```
**Impact**: Another benchmark

### Add Attention Weights Logging
**Goal**: See which indicators model uses
**Action**:
```python
# Modify actor network to output attention weights
# Log top 3 indicators per decision
```
**Impact**: Interpretability, debugging

## 1-Day Tasks

### Backtest on 2024 Out-of-Sample Data
**Goal**: Validate models generalize
**Action**:
```bash
python 4_backtest.py \
    --model train_results/ensemble \
    --data data/price_array_2024.npy \
    --output plots_and_metrics/backtest_2024
```
**Impact**: Detect overfitting early

### Implement Baseline Dashboard
**Goal**: Compare DRL vs. baselines visually
**Action**:
```python
# baselines/dashboard.py
# Plot cumulative returns: DRL vs Buy-Hold vs MA
# Show Sharpe, DD, win rate side-by-side
```
**Impact**: Easy performance tracking

### Add Memory Profiling
**Goal**: Find memory leaks in paper trader
**Action**:
```python
# Use memory_profiler
@profile
def run():
    ...

# Run for 24h, check growth
```
**Impact**: Reliability for long-running processes

## Bug Fixes (Under 30 Minutes Each)

### Add Input Validation
**File**: `environment_Alpaca.py:127`
```python
def step(self, actions):
    assert actions.shape == (self.n_assets,), f"Expected {self.n_assets}"
    actions = np.clip(actions, -100, 100)  # Sanity check
    ...
```

### Fix Error Handling in API Calls
**File**: `paper_trader_alpaca_polling.py:544`
```python
def _fetch_latest_bars(self):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return self.api.get_crypto_bars(...)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Add Data Quality Checks
**File**: `paper_trader_alpaca_polling.py:583`
```python
def _process_new_bar(self, df):
    # Check for NaN
    if df.isnull().any().any():
        print("⚠️ NaN detected in data, skipping bar")
        return

    # Check for zero prices
    if (df['close'] <= 0).any():
        print("⚠️ Invalid prices detected, skipping bar")
        return
    ...
```

## Priority Order

**Week 1** (Critical):
1. Run tests (find bugs)
2. Run baseline (validate DRL)
3. Validate profit protection (confidence)

**Week 2** (Performance):
4. Hyperparameter importance (optimize search)
5. Transaction cost analysis (reduce trading)
6. Reward ablation (simplify learning)

**Week 3** (Reliability):
7. Out-of-sample backtest (detect overfitting)
8. Memory profiling (fix leaks)
9. Error handling (robustness)

**Week 4** (Production):
10. Baseline dashboard (monitoring)
11. Attention weights (interpretability)
12. Live trading pilot ($100)
