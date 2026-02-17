# Grading Criteria Update

**Date:** 2026-01-30
**Change:** Lowered win rate threshold from 80% to 60%
**Reason:** More realistic for crypto trading volatility

---

## Updated Promotion Criteria

Paper trading models must meet ALL of the following criteria before promotion to live trading:

### ✅ Required Metrics

| Metric | Old Threshold | **New Threshold** | Rationale |
|--------|---------------|-------------------|-----------|
| Trading Duration | 7+ days | 7+ days | Unchanged - sufficient data needed |
| **Win Rate** | **80%** | **60%** | More realistic for crypto markets |
| Alpha vs Benchmark | Positive | Positive | Unchanged - must beat buy-and-hold |
| Maximum Drawdown | 15% | 15% | Unchanged - risk control |

### 📊 Why 60% is More Appropriate

**80% Win Rate Issues:**
- Extremely difficult to achieve in volatile crypto markets
- May encourage over-conservative strategies
- Could prevent good models from being promoted
- Unrealistic for real-world trading conditions

**60% Win Rate Benefits:**
- Still indicates solid performance (better than 50/50)
- Allows for normal market losses while remaining profitable
- More achievable target encourages model deployment
- Focuses on overall profitability (alpha) rather than perfection

### 🎯 Profitability Math

Even with 60% win rate, model can be highly profitable:

```
Example Trade Distribution:
- 60 winning trades @ +2% each = +120%
- 40 losing trades @ -1% each = -40%
- Net Profit: +80% over 100 trades

With good risk/reward ratio (2:1 or better), 60% win rate 
is more than sufficient for strong returns.
```

---

## Files Updated

1. **performance_grader.py** (4 locations)
   - Line 8: Documentation comment
   - Line 37: Default parameter
   - Line 48: Criteria dictionary
   - Line 526: CLI argument default

2. **coinbase_live_trader.py** (1 location)
   - Line 148: Error message for promotion verification

3. **TECHNICAL_ANALYSIS_REPORT.md**
   - All references to 80% win rate threshold

---

## Verification

To verify the change:

```bash
# Check performance grader
grep "min_win_rate" performance_grader.py
# Should show: "min_win_rate": 0.60

# Test grading (dry run)
python performance_grader.py --check

# View current criteria
python performance_grader.py --help
```

---

## Impact on Current System

- **No immediate impact** - paper traders still need 7+ days of data
- **Future promotions** - Will use new 60% threshold
- **Existing grades** - Will be re-evaluated with new criteria on next check
- **Live trader** - Promotion verification now requires 60%, not 80%

---

**Change Status:** ✅ Complete
**Restart Required:** No - changes take effect immediately
