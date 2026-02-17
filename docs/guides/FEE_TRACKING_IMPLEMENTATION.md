# Fee Tracking Implementation

## Summary

Implemented explicit fee tracking in the trading environment to provide full visibility into trading costs and their impact on returns.

## Problem Identified

The environment was paying fees on every trade (0.25% buy + 0.25% sell = 0.5% round-trip), but:
- **No explicit tracking** of total fees paid
- **No visibility** into how much performance was lost to trading costs
- **No separation** between gross return (before fees) and net return (after fees)
- Made it impossible to analyze if poor performance was due to bad strategy or excessive trading costs

## Implementation Details

### Changes to `environment_Alpaca.py`

#### 1. Added Fee Tracking Variables (Lines 103-108)
```python
# Fee tracking
self.total_fees_paid = 0.0
self.buy_fees_paid = 0.0
self.sell_fees_paid = 0.0
self.num_buy_trades = 0
self.num_sell_trades = 0
```

#### 2. Track Fees on ALL Trading Operations

**Trailing Stop Loss Sells** (Lines 196-204):
- Calculates fee: `fee = sell_value * self.sell_cost_pct`
- Tracks: `self.sell_fees_paid += fee`

**Regular Sells** (Lines 229-236):
- Same fee calculation and tracking
- Increments trade counter

**Forced Cooldown Sells** (Lines 248-255):
- Tracks fees even on automatic 5% liquidations
- Every half-day forced sell now tracked

**Buy Operations** (Lines 287-292):
- Calculates buy fee: `fee = buy_value * self.buy_cost_pct`
- Tracks: `self.buy_fees_paid += fee`

#### 3. Return Info Dict with Fee Data (Lines 332-339)

Every `step()` now returns info dict containing:
```python
{
    'total_fees_paid': float,
    'buy_fees_paid': float,
    'sell_fees_paid': float,
    'num_buy_trades': int,
    'num_sell_trades': int,
}
```

#### 4. Episode-End Reporting (Lines 341-371)

When episode completes (`done=True`):
- Calculates **gross return**: `(total_asset + total_fees_paid) / initial_cash`
- Calculates **net return**: `total_asset / initial_cash`
- Calculates **fee impact**: `gross_return - net_return`
- Prints detailed fee report (if logging enabled)
- Adds to info dict:
  - `episode_return_net`
  - `episode_return_gross`
  - `fee_impact_pct`

### Example Output

```
======================================================================
EPISODE COMPLETE - Fee Report
======================================================================
  Initial Capital:      $10,000.00
  Final Portfolio:      $10,491.72
  Net Return:           +4.92%
  Total Fees Paid:      $13.98 (0.14% of initial)
  Gross Return:         +5.06% (before fees)
  Fee Impact:           0.14%
  Buy Fees:             $11.19 (15 trades)
  Sell Fees:            $2.78 (7 trades)
  Total Trades:         22
  Avg Fee per Trade:    $0.64
======================================================================
```

## Benefits

1. **Full Transparency**: See exactly how much you're paying in fees
2. **Performance Attribution**: Separate trading skill from fee drag
3. **Strategy Optimization**: Identify if agent is overtrading
4. **Cost Analysis**: Compare different fee structures
5. **Debug Support**: Verify buy/sell operations are balanced

## Testing

Created `test_fee_tracking.py` which:
- ✅ Verifies fees tracked on buys
- ✅ Verifies fees tracked on sells
- ✅ Confirms gross vs net return calculations
- ✅ Validates info dict contains fee data
- ✅ Tests with realistic trading scenario

Run with:
```bash
python test_fee_tracking.py
```

## Remaining Issues

### Issue #1: Equal-Weight Benchmark Doesn't Pay Fees ⚠️

**Location**: `environment_Alpaca.py:56-59, 264-270`

The equal-weight portfolio (benchmark) is buy-and-hold with **zero fees**, while the DRL agent pays 0.5% round-trip fees. The reward function compares them directly:

```python
reward = (delta_bot - delta_eqw) * self.norm_reward * decay_factor
```

This is **fundamentally unfair** - like comparing an active trader to someone who bought once and held forever.

**Proposed Fixes**:
1. Add realistic fees to equal-weight benchmark (simulate rebalancing costs)
2. Remove equal-weight from reward calculation entirely
3. Use fee-adjusted returns for comparison

### Issue #3: Trade P&L Doesn't Include Fees ⚠️

**Location**: `trade_history_analyzer.py:179`

```python
pnl = (price - position_entry_price) * sell_qty_actual
```

This calculates trade P&L **without subtracting fees**. A "winning" trade with +0.4% gain would actually lose money after 0.5% fees, but still count as a win!

**Impact**: Win-rate metric may be inflated because it counts fee-losing trades as wins.

**Proposed Fix**:
```python
entry_value = sell_qty_actual * position_entry_price
exit_value = sell_qty_actual * price
entry_fee = entry_value * 0.0025
exit_fee = exit_value * 0.0025
pnl = (exit_value - exit_fee) - (entry_value + entry_fee)
```

## Next Steps

1. Address Issue #1: Fix equal-weight benchmark or remove from reward
2. Address Issue #3: Fix trade P&L to include fees for accurate win-rate
3. Add fee tracking to dashboard displays
4. Consider adding fee-adjusted Sharpe ratio calculation
