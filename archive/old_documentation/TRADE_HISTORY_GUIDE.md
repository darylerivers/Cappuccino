# Trade History Dashboard Page

## Overview
A new page (Page 5) has been added to your dashboard that shows all completed trades from your paper trading sessions with detailed profit/loss analysis.

## How to Access
1. Start the dashboard: `python dashboard.py`
2. Press the **Right Arrow** key repeatedly to cycle through pages until you reach Page 5
3. Or press the **Left Arrow** to go backwards through pages

## What You'll See

### Summary Section
- **Total Completed Trades**: Total number of round-trip trades (buy + sell)
- **Win Rate**: Percentage of profitable trades
- **Total Realized P&L**: Sum of all profits and losses
- **Best Trade**: Highest profit single trade
- **Worst Trade**: Biggest loss single trade

### Recent Trades Table (Last 25)
Shows the most recent 25 completed trades with:
- **Ticker**: Asset symbol (e.g., BTC/USD, ETH/USD)
- **Entry Time**: When the position was opened
- **Exit Time**: When the position was closed
- **Entry Price**: Average entry price
- **Exit Price**: Exit price
- **Quantity**: Amount traded
- **P&L $**: Profit or loss in dollars (green = profit, red = loss)
- **P&L %**: Profit or loss percentage
- **Hours**: How long the position was held

### Per-Ticker Statistics
Aggregated stats for each asset:
- **Trades**: Total number of completed trades
- **Wins**: Number of profitable trades
- **Losses**: Number of losing trades
- **Win Rate**: Percentage of winning trades for this asset
- **Total P&L**: Cumulative profit/loss for this asset

## Your Current Performance

Based on the latest data:

- **563 completed trades** across all assets
- **40.1% win rate** (226 wins, 337 losses)
- **Best performer**: ETH/USD (98.8% win rate, +$35,653 total P&L)
- **Good performer**: AAVE/USD (70.0% win rate, +$7,256 total P&L)

### Notable Trades
- **Best trade**: AAVE/USD +$2,527 (+2,245% gain)
- **Recent ETH/USD trades**: Multiple 40%+ gains over ~15 day holding periods

## Navigation
- **← Left Arrow**: Go to previous page (Training Stats)
- **→ Right Arrow**: Go to next page (Main Dashboard)
- **Ctrl+C**: Exit dashboard

## Trade Analysis Tool
You can also generate a detailed trade report from the command line:
```bash
python trade_history_analyzer.py
```

This will show:
- All trade actions (every buy/sell)
- Completed round-trip trades with P&L
- Summary statistics

## Portfolio Forecasting
To see future projections based on your 13% portfolio uplift:
```bash
python simple_portfolio_forecast.py
```

This shows:
- Current portfolio state
- Projected values for 1 week, 1 month, 3 months, 6 months, 1 year
- Conservative/Moderate/Aggressive scenarios
- Key milestones (e.g., when you'll reach $2,000, $5,000, etc.)

## Notes
- Only completed round-trip trades (buy → sell) are shown
- Open positions are not included in the statistics
- P&L is calculated from average entry price to exit price
- Historical data from all paper trading sessions is included
