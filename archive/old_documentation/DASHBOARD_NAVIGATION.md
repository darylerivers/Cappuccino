# Dashboard Navigation Guide

The Cappuccino dashboard now has **multi-page support** with keyboard navigation!

## How to Use

### Starting the Dashboard

```bash
python dashboard.py
```

### Navigation Controls

- **→ Right Arrow**: Go to next page
- **← Left Arrow**: Go to previous page
- **Ctrl+C**: Exit dashboard
- **q**: Quick exit (also works)

### Available Pages

#### Page 1/3: Main Dashboard
Shows the complete trading system overview:
- **Training Status**: Current study, workers, trials, best values
- **Paper Trading**: Model info, positions, P/L
- **Market Overview**: Live crypto prices and 24h changes
- **Autonomous AI Advisor**: Analysis count, tested configs
- **GPU Status**: Utilization, temperature, memory
- **System Health**: CPU, RAM, disk usage
- **Alerts**: Any system warnings or errors

#### Page 2/3: Ensemble Voting
Shows detailed model voting breakdown:
- **Model Performance Metrics**: Each model's training performance, rank, percentile
- **Individual Model Predictions**: What each of the 10 models voted for each asset
- **Ensemble Average**: The final averaged decision that gets executed
- **Consensus Analysis**: Agreement scores, vote counts (BUY/HOLD/SELL), standard deviation

#### Page 3/3: Portfolio History & Forecast
Shows historical performance and future projections:
- **Portfolio Summary**: Starting value, current value, total change, last hour performance
- **ASCII Line Chart**: Visual representation of portfolio value over time
  - █ Solid blocks: Historical data
  - · Dots: Forecasted values (4 hours ahead)
- **Recent Intervals**: Last 8 data points in 15-minute chunks (interpolated from hourly)
- **Linear Regression Forecast**: Projects portfolio value 4 hours into the future
- **Performance Metrics**: Time range, data points, value changes with percentages

## Features

### Auto-Refresh
Both pages refresh every 3 seconds automatically to show the latest data.

### Page Wrapping
- Pressing → on Page 3 wraps back to Page 1
- Pressing ← on Page 1 wraps to Page 3

### Non-Blocking Input
The dashboard uses non-blocking keyboard input, so you can switch pages at any time without waiting for the next refresh.

## Example Workflow

1. **Start on Main Dashboard (Page 1)**
   - See overall system status
   - Check if training and paper trading are running
   - Monitor GPU utilization

2. **Press → to view Ensemble Voting (Page 2)**
   - See what each model voted for the last decision
   - Check if models have high consensus
   - Identify which models are performing best

3. **Press → again to view Portfolio History (Page 3)**
   - See how the portfolio has performed over time
   - View the ASCII line chart showing trends
   - Check the 4-hour forecast projection
   - Analyze recent 15-minute intervals

4. **Press ← to navigate back through pages**
   - Continue monitoring system health
   - Watch for new trades

## Color Coding

### Main Dashboard
- **Green**: Good status, positive changes
- **Red**: Errors, negative changes, warnings
- **Yellow**: Warnings, moderate status
- **Cyan**: Information, neutral values
- **Bold**: Headers and important labels

### Ensemble Voting
- **Green**: BUY signals, high agreement, top performers
- **Red**: SELL signals, low values
- **Yellow**: Lower performing models, moderate agreement
- **Cyan**: Neutral information

## Tips

1. **Check consensus before trusting signals**: High agreement (>0.995) means all models agree
2. **Watch for divergence**: If models disagree (agreement <0.99), be cautious
3. **Compare model rankings**: Models 1-3 are the best performers, their votes may be more reliable
4. **Monitor in real-time**: Both pages update automatically, no need to manually refresh
5. **Quick navigation**: Arrow keys work instantly, no need to wait for refresh

## Troubleshooting

### Portfolio History shows "No portfolio history available yet"
- This is normal on first startup
- The system needs to execute at least one trade to have data
- Check back after the paper trader has been running for an hour

### Ensemble Voting shows "Waiting for first trading decision"
- This is normal on startup
- The paper trader needs to make at least one decision first
- With 1h timeframe, this happens once per hour
- Check back after the next hour mark (e.g., 17:00, 18:00)

### Arrow keys not working
- Make sure you're running in a proper terminal (not a restricted environment)
- Terminal must support ANSI escape codes
- Try pressing the arrow keys firmly
- If still not working, use Ctrl+C to exit and restart

### Dashboard not refreshing
- Check your internet connection (for market data)
- Verify processes are running: `./status_automation.sh`
- Make sure database is accessible: `ls -la databases/optuna_cappuccino.db`

## Command Line Options

```bash
# Normal mode (all information)
python dashboard.py

# Compact mode (essential info only - Page 1 only)
python dashboard.py --compact

# Single refresh (no loop, no navigation)
python dashboard.py --once

# Custom refresh interval
python dashboard.py --interval 5  # Refresh every 5 seconds
```

## What's Next?

You can extend the dashboard by:
1. Adding more pages (increase `self.num_pages` in `__init__`)
2. Creating new `render_*` methods for custom views
3. Adding more keyboard shortcuts (modify `get_keypress()`)
4. Customizing the display for specific needs
