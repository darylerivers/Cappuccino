# Stop-Loss Dashboard Integration - Complete

## What Was Added

### 1. Position State Logging (Paper Trader)
**File**: `paper_trader_alpaca_polling.py`
- **Method**: `_log_position_state()` (lines 972-1029)
- **Called**: Every time a trade is logged (line 965)
- **Output**: `paper_trades/positions_state.json`

**Data Logged**:
```json
{
  "timestamp": "2025-11-24T18:34:43",
  "portfolio_value": 1033.92,
  "cash": 300.03,
  "positions": [
    {
      "ticker": "BTC/USD",
      "holdings": 0.005,
      "current_price": 87903.72,
      "entry_price": 84950.70,
      "position_value": 436.62,
      "pnl_pct": 3.48,
      "stop_loss_price": 76455.63,
      "trailing_stop_price": 83817.19,
      "high_water_mark": 85017.12,
      "distance_to_stop_pct": 13.02
    }
  ]
}
```

### 2. Dashboard Display Integration
**File**: `dashboard.py`
- **Helper Methods Added** (lines 595-619):
  - `_load_position_state()` - Loads JSON file
  - `_get_stop_loss_info(ticker, position_state)` - Extracts stop-loss data

- **Display Integration** (lines 940-970):
  - Shows stop-loss price alongside each position
  - Color-coded distance warnings:
    - 🔴 Red: < 2% from stop-loss (imminent)
    - 🟡 Yellow: 2-5% from stop-loss (close)
    - 🟢 Green: > 5% from stop-loss (safe)

## Expected Display

**Before** (current):
```
BTC/USD: 0.0050 × $84950.70 → $87903.72 (+3.48%) = $436.62
ETH/USD: 0.1012 × $2772.15 → $2936.74 (+5.94%) = $297.27
```

**After** (once positions_state.json is created):
```
BTC/USD: 0.0050 × $84950.70 → $87903.72 (+3.48%) = $436.62 | Stop: $76455.63 (13.0%)
ETH/USD: 0.1012 × $2772.15 → $2936.74 (+5.94%) = $297.27 | Stop: $2495.04 (15.0%)
```

## When Will It Appear?

The stop-loss information will appear **automatically** on the next trade cycle when:
1. Paper trader polls for new bars
2. Model makes a trading decision
3. Trade is logged to CSV (triggers `_log_position_state()`)
4. `positions_state.json` is created/updated
5. Dashboard reads it on next refresh

**Current Status**:
- ✅ Paper trader running (PID 3037631)
- ✅ Last poll: 2025-11-25T00:34:26
- ⏳ Waiting for next trade to create positions_state.json
- ⏳ Dashboard will show stop-loss on next refresh after that

## Manual Testing

If you want to see it immediately, you can create a dummy file:

```bash
cat > paper_trades/positions_state.json << 'EOF'
{
  "timestamp": "2025-11-24T18:34:43",
  "portfolio_value": 1033.92,
  "cash": 300.03,
  "positions": [
    {
      "ticker": "BTC/USD",
      "holdings": 0.005,
      "current_price": 87903.72,
      "entry_price": 84950.70,
      "position_value": 436.62,
      "pnl_pct": 3.48,
      "stop_loss_price": 76455.63,
      "distance_to_stop_pct": 13.02
    },
    {
      "ticker": "ETH/USD",
      "holdings": 0.1012,
      "current_price": 2936.74,
      "entry_price": 2772.15,
      "position_value": 297.27,
      "pnl_pct": 5.94,
      "stop_loss_price": 2495.04,
      "distance_to_stop_pct": 15.04
    }
  ]
}
EOF

python dashboard.py --once
```

Then delete it to let the paper trader create it properly:
```bash
rm paper_trades/positions_state.json
```

## Technical Details

### Stop-Loss Calculation
- **Initial Stop-Loss**: `entry_price × (1 - RISK.STOP_LOSS_PCT)`
  - Default: 10% below entry (from `constants.py`)
- **Trailing Stop**: `high_water_mark × (1 - RISK.TRAILING_STOP_PCT)`
  - Default: 1.5% below peak (if enabled)
- **Distance**: `((current_price - stop_loss_price) / current_price) × 100`

### File Location
- **State File**: `paper_trades/positions_state.json`
- **Updated**: Every trade cycle
- **Read By**: Dashboard on every refresh (3 seconds)

## Verification

Check if file exists:
```bash
ls -lh paper_trades/positions_state.json
```

View contents:
```bash
cat paper_trades/positions_state.json | python -m json.tool
```

Monitor creation:
```bash
watch -n 1 'ls -lh paper_trades/positions_state.json 2>/dev/null || echo "Not created yet"'
```

## Summary

✅ **Integration Complete**:
- Paper trader logs position state
- Dashboard reads and displays it
- Color-coded warnings for proximity to stop-loss
- Automatic updates every trade cycle

⏳ **Waiting For**:
- Next trade cycle to create positions_state.json
- Then stop-loss info will appear automatically

🎯 **Result**:
You'll now see exactly where your stop-loss levels are for each position, helping you monitor risk in real-time!
