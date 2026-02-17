# Critical Bug: Concentration Limit Not Working

**Date:** January 17, 2026
**Severity:** 🚨 **CRITICAL** - Risk management failure
**Status:** ❌ **CONFIRMED**
**Affects:** All paper trading and live trading sessions

---

## TL;DR

**The 30% per-asset concentration limit is NOT being enforced.** Traders are taking positions up to 48.6% in a single asset, despite being configured with a 30% limit.

**Root Cause:** Concentration limit check operates on un-scaled model actions, but actual trades execute with scaled actions (scaled by ~1000-2000x).

**Impact:** Extreme concentration risk - a single asset crash could cause portfolio losses >30% instead of the expected max 10-15%.

---

## Evidence

### Single Model Trader (Trial #861)

**Configuration:**
```
✓ Per-position risk management:
  • Max position: 30% per asset
  • Stop-loss: 10% from entry
```

**Actual Behavior:**
```
2026-01-16 18:00: LINK/USD = 49.5% of portfolio ($426 / $861)
2026-01-16 20:00: LINK/USD = 48.3% of portfolio ($411 / $852)
2026-01-16 22:00: LINK/USD = 48.6% of portfolio ($416 / $856)
2026-01-17 00:00: LINK/USD = 48.6% of portfolio ($415 / $855)
```

**Expected:** Maximum 30% per asset = $258 max position
**Actual:** Consistently 48-49% per asset = $415-426 positions
**Violation:** 60% over the limit!

---

## Root Cause Analysis

### The Bug

The concentration limit check is performed on **un-scaled actions**, but trading executes with **scaled actions**.

**paper_trader_alpaca_polling.py** (lines 922-940):
```python
# 2. POSITION LIMIT CHECK (for buy orders)
if modified_action[i] > 0 and self.risk_mgmt.max_position_pct > 0:
    # Calculate what position would be after this buy
    buy_value = modified_action[i] * current_price  # ← BUG: uses un-scaled action
    new_position_value = position_value + buy_value
    new_total = total_asset + buy_value  # ← Also wrong calculation
    new_position_pct = new_position_value / new_total

    if new_position_pct > self.risk_mgmt.max_position_pct:
        # Cap the buy...
```

**environment_Alpaca.py** (lines 298-300):
```python
for i in range(self.action_dim):
    norm_vector_i = self.action_norm_vector[i]
    actions[i] = actions[i] * norm_vector_i  # ← Scaling happens AFTER risk check
```

### The Flow

1. **Model output:** action[LINK] = 0.0142 (small value)
2. **Risk management checks:**
   - buy_value = 0.0142 * $13.65 = **$0.19** ✓ Looks safe
   - Concentration check: $0.19 / $860 = **0.02%** ✓ Well under 30%
   - ✅ Action allowed to proceed
3. **Environment scales action:**
   - norm_vector[LINK] ≈ 2,200 (calculated from price magnitude)
   - scaled_action = 0.0142 * 2,200 = **31.2 shares**
4. **Actual trade executes:**
   - Buy value = 31.2 * $13.65 = **$426**
   - Concentration = $426 / $860 = **49.5%** 🚨 WAY OVER LIMIT!

### Secondary Bug

Line 926 has an additional bug:
```python
new_total = total_asset + buy_value  # WRONG!
```

This incorrectly calculates the total portfolio value. When you buy an asset:
- Cash decreases by `buy_value`
- Holdings increase by `buy_value`
- **Total remains the same** (minus fees)

Should be:
```python
new_total = total_asset  # Total doesn't change on buy
```

This secondary bug actually makes the primary bug worse, because it inflates the denominator and makes the concentration check even more permissive.

---

## Impact Assessment

### Risk Impact

**Without fix:**
- Single asset can reach 50% of portfolio
- If that asset crashes -30%, portfolio loses 15% in one asset
- Multiple concentrated positions = cascading risk
- Stop-loss triggers too late (only after 10% asset loss, but 50% concentration = 5% portfolio loss)

**Expected behavior with fix:**
- Max 30% per asset
- Asset crash -30% = max 9% portfolio loss
- Diversification across 3-4 assets minimum
- Stop-loss contains damage to <3% portfolio loss per position

### Trading Impact

Looking at the single model trader data:
- **6 trades executed** in 7 hours
- **All trades were LINK/USD** (100% concentration in trading activity)
- **Position repeatedly opened/closed** (bought at 18:00, 20:00, 22:00, 00:00)
- **Each time at ~49% concentration**

This suggests:
1. Model strongly prefers LINK/USD
2. Without concentration limit, it puts half the portfolio in one asset
3. News system reduces positions by 50%, triggering sells
4. Model immediately re-buys at same high concentration
5. Creates overtrading + extreme concentration

---

## Verification

### Test Case 1: Ensemble Trader

Check if ensemble trader has same bug:
```bash
python dashboard_snapshot.py
# Look for max concentration > 30%
```

### Test Case 2: Log Inspection

The bug is silent - no warning messages because the check passes on un-scaled actions:
```bash
grep "Position limit" logs/single_model_trial861.log
# Returns nothing - check never triggered!
```

Expected: Should see messages like:
```
📊 Position limit: LINK/USD buy capped from 31.2 to 18.9 (at 30% limit)
```

---

## The Fix

### Option 1: Apply Check After Scaling (Recommended)

Move the concentration limit check to **inside the environment's step function**, after action scaling.

**Pros:**
- Checks actual trade quantities
- Catches all edge cases
- Clean separation of concerns

**Cons:**
- Requires modifying environment_Alpaca.py
- Affects all agents using this environment

### Option 2: Account for Scaling in Risk Management

Multiply the action by the norm_vector before checking concentration.

**Pros:**
- Keeps risk management in paper trader
- Doesn't affect environment

**Cons:**
- Requires passing norm_vector to risk management
- More complex to maintain
- Could miss edge cases

### Option 3: Apply Limit in Both Places

Check in risk management AND in environment for defense in depth.

**Pros:**
- Maximum safety
- Catches bugs in either location

**Cons:**
- Code duplication
- More complex

---

## Recommended Fix (Option 1)

**Step 1:** Add concentration check in `environment_Alpaca.py` after line 300:

```python
# After scaling actions (line 300)
for i in range(self.action_dim):
    norm_vector_i = self.action_norm_vector[i]
    actions[i] = actions[i] * norm_vector_i

# NEW: Enforce concentration limit on scaled actions
if hasattr(self, 'max_position_pct') and self.max_position_pct > 0:
    price = self.price_array[self.time]
    total_asset = self.cash + np.sum(self.stocks * price)

    for i in range(self.action_dim):
        if actions[i] > 0:  # Buy orders only
            current_position_value = self.stocks[i] * price[i]
            buy_value = actions[i] * price[i]
            new_position_value = current_position_value + buy_value
            new_position_pct = new_position_value / total_asset if total_asset > 0 else 0

            if new_position_pct > self.max_position_pct:
                # Cap the buy to stay within limit
                max_position_value = self.max_position_pct * total_asset
                max_additional_value = max_position_value - current_position_value

                if max_additional_value > 0:
                    max_additional_shares = max_additional_value / price[i]
                    actions[i] = min(actions[i], max_additional_shares)
                else:
                    actions[i] = 0  # Already at or over limit
```

**Step 2:** Pass max_position_pct to environment:

In `paper_trader_alpaca_polling.py`, when creating the environment:
```python
self.env = CryptoEnvAlpaca(...)
self.env.max_position_pct = self.risk_mgmt.max_position_pct  # Add this
```

**Step 3:** Remove or keep the existing check as redundant safety:

The check in `_apply_risk_management` can stay as a first-pass filter (even though it operates on un-scaled actions, it might catch some extreme cases). Or remove it to avoid confusion.

---

## Testing the Fix

### Test 1: Single Asset Test
```python
# Start fresh paper trader with fix
# Model should want to buy LINK at 49% concentration
# Fix should cap it to 30%
# Verify in CSV: holding_LINK/USD * price_LINK/USD / total_asset ≤ 0.30
```

### Test 2: Log Messages
```bash
grep "concentration\|Position limit" logs/test_trader.log
# Should see messages showing action was capped
```

### Test 3: Multi-Asset Test
```python
# Model wants to buy multiple assets
# Each should be capped at 30%
# Total should allow 3+ assets (portfolio can't exceed 100%)
```

---

## Urgent Actions

1. **STOP live trading immediately** if any is running - concentration risk is uncontrolled
2. **Fix the bug** using recommended approach
3. **Restart paper trading** with fix applied
4. **Verify fix** with test cases above
5. **Review past trades** to assess if concentration violations occurred

---

## Related Issues

### Ensemble Trader Status

Need to verify if ensemble trader has same bug. Check:
```bash
python dashboard_snapshot.py
# Look at "Max Concentration" for ensemble
```

If ensemble also shows >30%, same bug affects it.

### Historical Risk Assessment

Review all paper trading logs to check for concentration violations:
```bash
# Check all historical sessions
for csv in paper_trades/*.csv; do
    echo "=== $csv ==="
    python -c "
import pandas as pd
import sys
df = pd.read_csv('$csv')
# Calculate max concentration for each row
# Flag any >30%
"
done
```

---

**Status:** Bug confirmed and documented
**Next:** Implement fix and retest
**Priority:** 🚨 **CRITICAL** - Must fix before any live trading
