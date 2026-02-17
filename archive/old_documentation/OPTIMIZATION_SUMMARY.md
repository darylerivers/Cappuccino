# Cappuccino Optimization Summary

## What Was Accomplished (Nov 24, 2025)

### 1. ✅ Project Documentation Restructured

**Created Token-Efficient Context System**:
- `CRITICAL_CONTEXT.md` (3KB) - Essential info, replaces 33KB full doc for daily use
- `contexts/` directory - Focused topic files
  - `risk_management.md` - Risk systems detailed
  - `quick_wins.md` - High-impact tasks prioritized
- `code_maps/` directory
  - `function_index.md` - Quick function lookup by file

**Impact**: ~70-80% token reduction for Claude-Code interactions

### 2. ✅ Buy-and-Hold Baseline Implemented

**File**: `baselines/buy_and_hold.py` (220 lines)

**Features**:
- Equal-weight portfolio
- Optional rebalancing
- All standard metrics (Sharpe, returns, max DD, win rate, Sortino)
- JSON output for comparison

**Usage**:
```bash
python baselines/buy_and_hold.py --data data/price_array_val.npy
```

**Next Step**: Run to validate if DRL beats passive investing (CRITICAL)

### 3. ✅ Critical Test Suite Created

**File**: `tests/test_critical.py` (350+ lines, 25 tests)

**Test Coverage**:
- ✅ Stop-loss triggers (3 tests)
- ✅ Position limits enforcement (3 tests)
- ✅ Ensemble voting correctness (3 tests)
- ✅ NEW profit protection logic (7 tests) ⚠️
- ✅ State normalization consistency (5 tests)
- ✅ Data quality checks (4 tests)

**Usage**:
```bash
pytest tests/test_critical.py -v
```

**Next Step**: Run to catch bugs in new profit protection code

### 4. ✅ Constants Centralized

**File**: `constants.py` (280 lines)

**Extracted**:
- Risk parameters (all thresholds)
- Normalization scales (2^-N values)
- Trading config (fees, capital, intervals)
- Training config (trials, windows, hyperparameter ranges)
- Paths (all directories and files)

**Usage**:
```python
from constants import RISK, NORMALIZATION, TRADING
stop_loss = RISK.STOP_LOSS_PCT  # 0.10
initial_capital = TRADING.INITIAL_CAPITAL  # 1000
```

**Next Step**: Refactor imports in main files to use constants

### 5. ✅ Comprehensive Project Review

**File**: `PROJECT_OVERVIEW_FOR_OPUS.md` (33KB, 864 lines)

**Contents**:
- Complete architecture documentation
- 18 known issues with investigation paths
- Testing gaps identified
- Security considerations
- Scalability bottlenecks
- Quick wins prioritized

**Usage**: Reference for deep dives, give to Claude Opus for feedback

---

## Token Usage Optimization Achieved

### Before:
- Full project context: 33KB (~10,000 tokens)
- Reading full files for simple tasks
- Regenerating code from scratch

### After:
- Essential context: 3KB (~1,000 tokens) - **70% reduction**
- Focused context files: 5-10KB each
- Function index for quick lookups
- Constants prevent magic number questions

### Example Savings:
**Task**: "Fix stop-loss bug"

**Before**:
1. Read full paper_trader_alpaca_polling.py (1000 lines, 3000 tokens)
2. Read environment_Alpaca.py for context (500 lines, 1500 tokens)
3. Generate fix
**Total**: ~5000 tokens

**After**:
1. Read CRITICAL_CONTEXT.md (3KB, 1000 tokens)
2. Check function_index.md for line number
3. Read only lines 699-773 of paper_trader_alpaca_polling.py (200 tokens)
4. Generate fix
**Total**: ~1200 tokens (**76% reduction**)

---

## Immediate Next Steps (Priority Order)

### Week 1: Validation & Bug Fixes

1. **Run Tests** (10 minutes)
   ```bash
   pytest tests/test_critical.py -v
   ```
   **Why**: Catch bugs in NEW profit protection code before it loses money

2. **Run Baseline** (1 hour)
   ```bash
   # Need to generate data file first
   python 0_dl_trainval_data.py
   python baselines/buy_and_hold.py --data data/price_array_val.npy
   ```
   **Why**: Validate DRL actually adds value

3. **Validate Profit Protection** (4 hours)
   - Parse existing `paper_trades/alpaca_session.csv`
   - Simulate profit protection logic
   - Check if it would have locked in gains
   **Why**: Confidence before deploying

4. **Refactor Imports** (2 hours)
   ```python
   # Update these files to use constants.py:
   - paper_trader_alpaca_polling.py
   - environment_Alpaca.py
   - 1_optimize_unified.py
   ```
   **Why**: Eliminate magic numbers, easier tuning

### Week 2: Performance Optimization

5. **Hyperparameter Importance** (2 hours)
   - Plot Optuna importance scores
   - Identify which params don't matter
   - Narrow search space

6. **Transaction Cost Analysis** (2 hours)
   - Calculate fees as % of profits
   - Test action dampening (0.5x, 0.25x)

7. **Reward Ablation Study** (1 day)
   - Train 5 trials with different reward components
   - Compare validation Sharpe
   - Simplify reward function

### Week 3: Reliability & Testing

8. **Out-of-Sample Backtest** (1 day)
   - Test on 2024 data (if training on 2023)
   - Detect overfitting

9. **Memory Profiling** (4 hours)
   - Run paper trader for 24h with profiler
   - Find leaks

10. **Integration Tests** (1 day)
    - End-to-end training pipeline
    - Mock Alpaca API for paper trading

---

## Files Created Today

```
baselines/
  buy_and_hold.py          220 lines - Equal-weight baseline

tests/
  test_critical.py         350 lines - 25 critical tests

constants.py               280 lines - All magic numbers centralized

CRITICAL_CONTEXT.md        3KB - Essential context for daily use
OPTIMIZATION_SUMMARY.md    (this file)
PROJECT_OVERVIEW_FOR_OPUS.md  33KB - Comprehensive review

contexts/
  risk_management.md       Detailed risk system docs
  quick_wins.md            Prioritized tasks

code_maps/
  function_index.md        Quick function lookup
```

---

## Metrics

**Lines of Code Added**: ~850 lines
- Tests: 350 lines
- Baseline: 220 lines
- Constants: 280 lines

**Documentation Added**: ~40KB
- Context files: 7KB
- Project overview: 33KB

**Token Efficiency**: 70-80% reduction per interaction

**Test Coverage**: 0% → 25+ critical tests (not run yet)

**Baselines**: 0 → 1 (buy-and-hold)

---

## How to Use This Structure with Claude-Code

### For Quick Tasks:
```bash
# Always start with critical context
claude-code "Read CRITICAL_CONTEXT.md and fix stop-loss bug"

# Use function index for navigation
claude-code "Check function_index.md then optimize _select_action()"
```

### For Deep Analysis:
```bash
# Point to specific context
claude-code "Read contexts/risk_management.md and review profit protection logic"

# Full context when needed
claude-code "Read PROJECT_OVERVIEW_FOR_OPUS.md section 8 and investigate issue #3"
```

### For Implementation:
```bash
# Use constants
claude-code "Update paper_trader to use constants.py for risk thresholds"

# Use templates
claude-code "Implement momentum baseline using baselines/buy_and_hold.py as template"
```

---

## Success Criteria

**Short-term** (1 week):
- [ ] All tests pass
- [ ] Baseline comparison done
- [ ] Profit protection validated
- [ ] Constants integrated

**Medium-term** (1 month):
- [ ] DRL Sharpe > Buy-and-Hold Sharpe by 20%+
- [ ] Profit protection proven effective
- [ ] Transaction costs reduced 30%
- [ ] Memory leaks fixed

**Long-term** (3 months):
- [ ] Live trading with $100 (if validated)
- [ ] Automated baseline comparisons
- [ ] Full test coverage (80%+)
- [ ] Model interpretability added

---

## Questions for Claude Opus 4.1

1. **Critical Issues**: Any logical errors in new profit protection code?
2. **Performance**: Best approach to validate reward function?
3. **Testing**: What tests are missing from critical suite?
4. **Architecture**: Should reward function be decomposed into separate components?
5. **Security**: Any vulnerabilities before live trading?

---

**Status**: ✅ Optimization foundation complete, ready for validation phase
**Next Session**: Run tests and baseline, validate profit protection
**Risk**: New profit protection code untested - high priority to validate
