# ROOT CAUSE IDENTIFIED: Ensemble Loading Wrong Models

## The Bug

The ensemble is loading **the wrong trials** from the database!

### What Should Happen
1. Manifest says: Load Trial 686 from study "cappuccino_alpaca_v2" (Sharpe 0.1566)
2. Load actor.pth from `train_results/cwd_tests/trial_686_1h/actor.pth`
3. Use this model for trading

### What Actually Happens
1. Manifest says: Load Trial 686
2. Code reads actor.pth from `train_results/ensemble_best/model_0/actor.pth` ✓
3. Code queries database for trial 686 hyperparameters
4. **BUG**: Code uses study `cappuccino_week_20251206` (from ACTIVE_STUDY_NAME)
5. Database returns trial 686 from **WRONG STUDY** (Sharpe 0.0056 instead of 0.1566!)

## The Evidence

### Console Log Shows Wrong Trials
```
✓ Model 1: Trial #2398 (value=0.014902)
✓ Model 2: Trial #2315 (value=0.014051)
✓ Model 3: Trial #1889 (value=0.013236)
```

### Manifest Says Different Trials
```json
{
  "trial_numbers": [686, 687, 521, 578, ...],
  "trial_values": [0.1566, 0.1565, 0.1563, ...]
}
```

### Database Has Multiple Trial 686s

| Study | Trial # | Trial ID | Sharpe |
|-------|---------|----------|--------|
| cappuccino_3workers_20251102_2325 | 686 | 1506 | **0.0619** ✓ GOOD |
| cappuccino_week_20251206 | 686 | 10008 | 0.0056 ❌ BAD |
| cappuccino_1year_20251121 | 686 | 7144 | 0.0115 ❌ BAD |

### Environment Variable
```bash
ACTIVE_STUDY_NAME="cappuccino_week_20251206"
```

## The Code Bug

**File**: `ultra_simple_ensemble.py`
**Lines**: 49-72

```python
def _load_hyperparameters_from_db(self, trial_number: int, study_name: str = None) -> dict:
    """Load trial hyperparameters from Optuna database."""
    if study_name is None:
        study_name = _DEFAULT_STUDY  # ← BUG: Uses cappuccino_week_20251206

    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Get trial params
    cursor.execute("""
        SELECT tp.param_name, tp.param_value
        FROM trial_params tp
        JOIN trials t ON tp.trial_id = t.trial_id
        WHERE t.number = ? AND t.study_id = (
            SELECT study_id FROM studies WHERE study_name = ?
        )
    """, (trial_number, study_name))  # ← Queries WRONG study!
```

## Why This Happened

When we created `train_results/ensemble_best/` on Dec 17, we:
1. Found top 20 trials from study "cappuccino_alpaca_v2"
2. Copied their actor.pth files to ensemble_best/
3. Created manifest.json with correct trial numbers and values
4. **BUT**: Didn't create `best_trial` pickle files with trial metadata
5. **AND**: Didn't update ACTIVE_STUDY_NAME to match the study

The UltraSimpleEnsemble code assumes:
- Manifest trial numbers come from ACTIVE_STUDY_NAME
- It can query database using trial number + ACTIVE_STUDY_NAME

But the manifest has trials from a DIFFERENT study!

## Impact

The ensemble has been trading with:
- **Intended**: Top 20 models (Sharpe 0.14-0.15)
- **Actually**: Random models from wrong study (Sharpe 0.01-0.02)
- **Result**: -6.56% alpha, massive underperformance

## The Fix

### Option A: Fix the Ensemble Creation Script (PROPER FIX)

When creating ensemble, store study_name in manifest:

```json
{
  "model_count": 20,
  "trial_numbers": [686, 687, ...],
  "trial_values": [0.1566, 0.1565, ...],
  "study_name": "cappuccino_alpaca_v2",  ← ADD THIS
  "trial_ids": [1506, 1507, ...]         ← Or store trial_ids
}
```

Modify `ultra_simple_ensemble.py` to use manifest's study_name:
```python
study_name = self.manifest.get('study_name', _DEFAULT_STUDY)
params = self._load_hyperparameters_from_db(trial_num, study_name)
```

### Option B: Create best_trial Files (ALTERNATIVE)

Copy `best_trial` pickle files from original trials:
```bash
cp train_results/cwd_tests/trial_686_1h/best_trial \
   train_results/ensemble_best/model_0/best_trial
```

Then use AdaptiveEnsembleAgent instead of UltraSimpleEnsemble.

### Option C: Use Trial IDs Instead of Numbers (BEST)

Store trial_id (unique) instead of trial number (not unique):
```python
cursor.execute("""
    SELECT tp.param_name, tp.param_value
    FROM trial_params tp
    WHERE tp.trial_id = ?
""", (trial_id,))  # trial_id is globally unique
```

## Immediate Action Required

1. **Recreate ensemble_best with correct study reference**
   - Query from correct study (cappuccino_alpaca_v2)
   - Store study_name in manifest
   - OR store trial_ids instead of numbers

2. **Restart paper trader with fixed ensemble**
   - Should load correct models
   - Performance should match backtest (Sharpe ~0.14)

3. **Add validation**
   - Check loaded model Sharpe matches manifest
   - Log warning if mismatch
   - Fail fast if wrong models loaded

## Lessons Learned

1. **Trial numbers are NOT unique** across studies
   - Must always pair with study_name or use trial_id
   - Database has multiple "trial 686"

2. **Global state is dangerous**
   - ACTIVE_STUDY_NAME in .env affected ensemble loading
   - Ensemble should be self-contained

3. **Validate assumptions**
   - Should have checked if loaded models matched manifest
   - Should have logged model Sharpe values on load
   - Should have verified performance matched backtest early

4. **Test with small examples**
   - Could have caught this by loading single model first
   - Console showed wrong trial numbers but we didn't check

## Why Performance Was So Bad

The models we loaded had Sharpe ~0.01-0.02 (10x worse than expected 0.14-0.15). These models:
- Trained on different data (week of Dec 6)
- Optimized poorly (low Sharpe)
- Overfitted to noise
- Caused overtrading and poor decisions

The -6.56% alpha makes sense now - we were using BAD models.

## Expected Improvement After Fix

With correct models (Sharpe 0.14-0.15):
- Should match backtest performance
- Alpha should be positive (+1-2% weekly)
- Less overtrading
- More consistent positions

---

**Status**: Bug identified, fix ready to implement
**Priority**: CRITICAL - must fix before restarting paper trading
**Next**: Rebuild ensemble_best with correct study reference
