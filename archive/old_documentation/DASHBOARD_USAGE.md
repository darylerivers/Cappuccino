# Paper Trading Dashboard

**Status:** ✅ Ready to use
**Created:** January 17, 2026

---

## Quick Start

```bash
cd /opt/user-data/experiment/cappuccino

# Simple launch (auto-detects CSV files)
./start_dashboard.sh

# Or run directly
python paper_trading_dashboard.py
```

---

## What It Shows

### Real-Time Metrics (Side-by-Side Comparison)

**Performance:**
- Total Return (% gain/loss from starting capital)
- Sharpe Ratio (risk-adjusted returns)
- Total Asset Value
- Available Cash

**Trading Activity:**
- Trades in last 24 hours
- Trades in last 7 days
- Total trades (all time)
- Data points collected

**Risk Management:**
- Number of active positions
- Max concentration per asset (⚠️ alerts if >30%)
- Current position breakdown with percentages

**Status:**
- ✓ Active - Updated within last 2 hours
- ⚠️ Idle - No updates for 2-24 hours
- ✗ Stale - No updates for >24 hours

---

## Features

### 1. Auto-Refresh
- Updates every 60 seconds by default
- Customize with `--refresh N` (seconds)

### 2. Concentration Alerts
- **Green:** <25% per asset
- **Yellow:** 25-30% per asset (approaching limit)
- **Red:** >30% per asset (⚠️ OVER LIMIT)

### 3. Position Tracking
Shows all current positions sorted by value:
- Asset name
- Percentage of portfolio
- Dollar value

### 4. Performance Comparison
See at a glance which trader is performing better:
- Higher returns
- Better Sharpe ratio
- More efficient trading (fewer trades with better results)

---

## Command Line Options

```bash
python paper_trading_dashboard.py [OPTIONS]

Options:
  --ensemble-csv PATH   Path to ensemble trader CSV
                        (default: paper_trades/watchdog_session_20260116_183945.csv)

  --single-csv PATH     Path to single model trader CSV
                        (default: paper_trades/single_model_trial861.csv)

  --refresh N           Refresh interval in seconds
                        (default: 60)

  -h, --help            Show help message
```

### Examples

```bash
# Use custom refresh rate (every 30 seconds)
python paper_trading_dashboard.py --refresh 30

# Monitor specific CSV files
python paper_trading_dashboard.py \
    --ensemble-csv paper_trades/ensemble_custom.csv \
    --single-csv paper_trades/single_custom.csv

# Fast refresh for active monitoring (every 10 seconds)
python paper_trading_dashboard.py --refresh 10
```

---

## Enhanced Display (Optional)

For a prettier terminal UI with colors and formatting, install the `rich` library:

```bash
pip install rich
```

**With rich:**
- ✅ Colored status indicators
- ✅ Formatted tables with borders
- ✅ Panel layouts
- ✅ Better visual hierarchy

**Without rich:**
- ✅ Still fully functional
- ✅ Simple text-based output
- ✅ All metrics shown

---

## Interpreting Results

### What to Look For

**1. Trade Frequency**
- **Too high:** Overtrading, eating into profits with fees
- **Too low:** Not capitalizing on opportunities
- **Target:** ~1-2 trades per day per asset (7-14 per week for 7 assets)

**2. Concentration Limits**
- Should always be **<30%** per asset
- If red alert shows, risk management is failing
- Ensemble should naturally have better diversification

**3. Performance Comparison**
After 3-5 days of data:
- Compare total returns (ensemble vs single)
- Compare Sharpe ratios (which is more consistent?)
- Look at trade counts (which is more efficient?)

**4. Position Behavior**
- Are positions being held long enough? (min 4 hours)
- Is the trader closing losing positions? (stop-loss at -10%)
- Is it diversifying across assets?

### Expected Behavior

**Ensemble Trader:**
- More conservative (top-20 model voting)
- Better diversification
- More stable returns
- Potentially fewer trades

**Single Model Trader (Trial #861):**
- More aggressive (single best model)
- Potentially higher returns
- Higher variance
- May trade more frequently

---

## Monitoring Tips

### 1. Run in tmux/screen
Keep dashboard running even when disconnected:

```bash
# Start tmux session
tmux new -s dashboard

# Run dashboard
./start_dashboard.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t dashboard
```

### 2. Check Daily
- Morning: Check overnight activity
- Evening: Review day's performance
- Compare traders every 24h

### 3. What to Act On

**⚠️ Immediate Action Required:**
- Concentration >30% (risk management failing)
- Trader status "Stale" for >24h (process crashed)
- Negative returns <-5% (strategy underperforming)

**📊 Monitor Closely:**
- Trade frequency >3 per day per asset (overtrading)
- Sharpe ratio <0.5 (poor risk-adjusted returns)
- Position count always 0 or always 7 (not adapting)

**✅ Good Signs:**
- Concentration 15-25% per asset
- Sharpe ratio >0.8
- 1-2 trades per day per asset
- Positive returns accumulating

---

## Troubleshooting

### Dashboard Won't Start

**Check CSV files exist:**
```bash
ls -lh paper_trades/watchdog_session_*.csv
ls -lh paper_trades/single_model_trial861.csv
```

**Check paper traders are running:**
```bash
ps aux | grep paper_trader_alpaca_polling
```

**Check for errors:**
```bash
python paper_trading_dashboard.py --refresh 60
# Look for error messages
```

### No Data Showing

**Check CSV has data:**
```bash
wc -l paper_trades/watchdog_session_20260116_183945.csv
wc -l paper_trades/single_model_trial861.csv
```

Should have >1 line (header + data rows)

**Check timestamps are recent:**
```bash
tail -1 paper_trades/single_model_trial861.csv
```

### Dashboard Crashes

**Python version:**
```bash
python --version  # Should be 3.8+
```

**Check dependencies:**
```bash
python -c "import pandas, numpy; print('OK')"
```

**Install missing packages:**
```bash
pip install pandas numpy rich
```

---

## Files Created

```
paper_trading_dashboard.py    # Main dashboard script
start_dashboard.sh             # Quick launcher
DASHBOARD_USAGE.md            # This file
```

---

## Integration with Other Tools

### With Alert System (Future)
Dashboard can be extended to:
- Send notifications when concentration >30%
- Alert on trader crashes
- Email daily performance summary

### With Trade Analysis Tool (Future)
Dashboard shows real-time, analysis tool shows historical:
- Detailed trade-by-trade breakdown
- Performance attribution
- Statistical significance tests

### With Automated Backup (Future)
Dashboard reads from CSVs that are being backed up hourly.

---

## Example Output

```
================================================================================
PAPER TRADING DASHBOARD - 2026-01-17 00:40:15
================================================================================

ENSEMBLE TRADER                          SINGLE MODEL TRADER
---------------------------------------- ----------------------------------------
Status               ✓ Active            Status               ✓ Active
Total Return         +2.34%              Total Return         -0.76%
Sharpe Ratio         0.0142              Sharpe Ratio         0.0098
Total Asset          $861.42             Total Asset          $856.19
Cash                 $440.32             Cash                 $440.76

Trades (24h)         12                  Trades (24h)         3
Trades (7d)          45                  Trades (7d)          9
Trades (All)         78                  Trades (All)         12
Data Points          48                  Data Points          8

Active Positions     3                   Active Positions     1
Max Concentration    28.5%               Max Concentration    48.6% ⚠️

ENSEMBLE POSITIONS                       SINGLE MODEL POSITIONS
---------------------------------------- ----------------------------------------
BTC/USD   24.2% ($208.45)                LINK/USD  48.6% ($415.46)
ETH/USD   22.8% ($196.32)
LINK/USD  28.5% ($245.51)
```

---

**Dashboard Status:** ✅ Active and monitoring
**Last Updated:** January 17, 2026, 00:40 UTC
