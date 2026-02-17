# Performance Metrics Bug Fix

## Problem
The Arena Models Performance section in the dashboard was showing impossible returns like -77%, -74%, -72% even though crypto markets aren't that volatile.

## Root Cause
**Line 381 in dashboard.py had a critical bug:**

```python
# WRONG - only looks at cash!
return_pct = ((cash - initial) / initial) * 100
```

This calculated returns based **only on cash**, completely ignoring crypto holdings.

### Why This Caused Negative Returns

When a model buys crypto:
- Initial portfolio: $1,000 (all cash)
- After buying crypto: $200 cash + $800 in BTC/ETH holdings
- **Wrong calculation**: (200 - 1000) / 1000 = **-80%** ❌
- **Correct calculation**: ((200 + 800) - 1000) / 1000 = **0%** ✓

Since all models were actively trading (buying crypto), their cash balances were low, making it appear as massive losses when in reality they had valuable crypto positions.

## Solution

Fixed the calculation to use **total portfolio value** (cash + holdings):

```python
# CORRECT - uses total portfolio value from history
value_history = portfolio.get('value_history', [])
if len(value_history) > 0:
    current_value = value_history[-1][1]  # Latest total value
else:
    current_value = cash  # Fallback

return_pct = ((current_value - initial) / initial) * 100
```

## Additional Fixes

### 1. Sharpe Ratio Calculation
Previously always showed 0.00. Now calculates actual Sharpe ratio from value history:

```python
if len(value_history) >= 10:
    values = [v for _, v in value_history]
    returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
    if len(returns) > 0 and np.std(returns) > 0:
        # Annualized Sharpe (assuming hourly data)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(24 * 365)
```

### 2. Added NumPy Import
Added `import numpy as np` to support the Sharpe ratio calculation.

## Expected Results After Fix

You should now see:
- **Realistic return percentages** (-5% to +15% typical for crypto)
- **Actual Sharpe ratios** (0.5 to 3.0 typical for good strategies)
- **Max drawdowns** that make sense relative to returns

## Testing

Restart your dashboard and check Page 3 (Model Arena):

```bash
python dashboard.py
# Press '3' to go to Model Arena page
```

The return percentages should now reflect actual portfolio performance, not just cash balances.

## Files Modified
- `dashboard.py` (lines 381-420)

## Date
2025-12-10
