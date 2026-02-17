# Portfolio History & Forecast Guide

## Overview

Page 3 of the dashboard shows historical portfolio performance and forecasts future values using linear regression.

## What You'll See

### 1. Portfolio Summary
```
Time Range: 2025-11-03 06:00 to 2025-11-25 17:00 (539.0h)
Data Points: 217 (hourly)

Starting Value: $1000.00
Current Value:  $1010.03
Total Change:   +$10.03 (+1.00%)
Last Hour:      +$14.25 (+1.43%)
```

- **Time Range**: When data collection started and ended
- **Data Points**: Number of actual data points (hourly trading decisions)
- **Starting Value**: Initial portfolio value ($1000 for paper trading)
- **Current Value**: Most recent portfolio value
- **Total Change**: Overall performance since start
- **Last Hour**: Most recent hourly change

### 2. Portfolio Value Chart

```
$1012.40 │                                 █················
         │                                █
         │                               █
         │                              █
$ 986.33 │                            █
         │                           █
         │                          █
         │ █████████████████████████
$ 960.27 │█
         └──────────────────────────────────────────────────
         08:45                                        21:00

Current: $1010.03 │ Forecasted (4h): $1012.40 │ Projected: +$2.37 (+0.24%)
```

**Chart Elements:**
- **█ (Solid blocks)**: Historical actual data
- **· (Dots)**: Forecasted values for next 4 hours
- **Y-axis**: Portfolio value in dollars
- **X-axis**: Time (start time to end time)

### 3. Recent Intervals (15-minute chunks)

```
Time                           Value               Change
─────────────────────────────────────────────────────────
2025-11-25 15:15     $       971.03                 -
2025-11-25 15:30     $       979.28     +$8.25 (+0.85%)
2025-11-25 15:45     $       987.53     +$8.25 (+0.84%)
2025-11-25 16:00     $       995.78     +$8.25 (+0.84%)
```

Shows the last 8 intervals with:
- **Time**: Timestamp (15-minute intervals, interpolated from hourly)
- **Value**: Portfolio value at that time
- **Change**: Difference from previous interval

## How the Forecast Works

### Linear Regression

The forecast uses simple linear regression:
1. Takes the last 12 hours of actual data points
2. Fits a straight line through the data
3. Extrapolates the trend forward 4 hours

**Formula:**
```
future_value = current_value + (slope × hours_ahead)
```

Where:
- `slope` = rate of change per hour ($/hour)
- `hours_ahead` = how many hours into the future

### Example Calculation

If the portfolio gained $14.25 in the last hour:
- Slope ≈ +$14.25/hour
- Forecast for 4 hours ahead: $1010.03 + ($14.25 × 4) = $1067.03

But since we use regression over 12 hours, it averages out short-term volatility.

### Accuracy

**Important Notes:**
- Forecasts are **extrapolations**, not predictions
- Only use for short-term trend visualization (4 hours)
- Crypto markets are volatile - actual results will vary
- The forecast assumes the current trend continues
- Does not account for:
  - Market events
  - Trading strategy changes
  - Price volatility
  - Model decisions

## Interpreting the Data

### Good Signs
- **Positive slope**: Portfolio trending upward
- **Consistent gains**: Multiple green intervals in a row
- **Forecast above current**: Positive momentum

### Warning Signs
- **Negative slope**: Portfolio trending downward
- **High volatility**: Large swings between intervals
- **Forecast significantly different from current**: Unstable trend

### Color Coding
- **🟢 Green**: Positive changes
- **🔴 Red**: Negative changes
- **White**: No change or baseline

## Data Interpolation

Since trading happens on **1-hour intervals** but the display shows **15-minute intervals**:

1. **Actual hourly data** is collected from paper trading
2. **Linear interpolation** creates smooth transitions between hours
3. **Display** shows interpolated 15-min intervals for visualization
4. **Forecast** uses only actual hourly data (not interpolated)

Example:
```
16:00 → $995.78 (actual)
16:15 → $999.34 (interpolated)
16:30 → $1002.90 (interpolated)
16:45 → $1006.47 (interpolated)
17:00 → $1010.03 (actual)
```

## Use Cases

### 1. Performance Tracking
Monitor how the portfolio is performing over hours/days.

### 2. Trend Analysis
Identify if the strategy is gaining or losing momentum.

### 3. Quick Checks
See at a glance if the portfolio is up or down without detailed calculations.

### 4. Comparing Sessions
Look at different time ranges to see performance patterns.

## Limitations

1. **Limited Data**: New sessions have few data points
2. **Hourly Resolution**: Only updates once per hour
3. **Linear Assumption**: Assumes constant rate of change
4. **No External Factors**: Doesn't consider market conditions
5. **Short Forecast Window**: Only 4 hours ahead

## Tips

1. **Wait for more data**: The more data points, the better the trend
2. **Compare multiple intervals**: Look at last hour vs total change
3. **Don't over-rely on forecast**: Use as a guide, not a guarantee
4. **Watch for trend changes**: Sharp changes in slope indicate strategy shifts
5. **Cross-reference with Page 1**: Check actual positions and trades

## Advanced: Understanding the Math

### Slope Calculation
```python
# For each data point (t, v) where t=time, v=value:
x_mean = average(all_times)
y_mean = average(all_values)

numerator = sum((time - x_mean) * (value - y_mean))
denominator = sum((time - x_mean)²)

slope = numerator / denominator
```

### Forecast Generation
```python
for each 15-minute interval in next 4 hours:
    hours_ahead = interval * 0.25
    forecast_value = current_value + (slope * hours_ahead)
```

This gives a straight-line projection based on recent trend.

## Example Scenarios

### Scenario 1: Steady Growth
```
Start:  $1000.00
Current: $1015.50
Change: +$15.50 (+1.55%)
Slope:  +$3.10/hour

Forecast (4h): $1027.90
Projection: +$12.40 (+0.80%)
```
**Interpretation**: Consistent upward trend, forecast looks promising.

### Scenario 2: Recent Decline
```
Start:  $1000.00
Current: $985.20
Change: -$14.80 (-1.48%)
Slope:  -$7.40/hour

Forecast (4h): $955.60
Projection: -$29.60 (-3.00%)
```
**Interpretation**: Downward trend, may want to review strategy.

### Scenario 3: Volatile
```
Start:  $1000.00
Current: $1002.40
Change: +$2.40 (+0.24%)
Recent Hour: +$25.80 (+2.64%)

Forecast (4h): $1105.60
Projection: +$103.20 (+10.30%)
```
**Interpretation**: Recent spike, but forecast may be overly optimistic based on short-term gain.

---

**Remember**: The forecast is a tool for visualization, not a guarantee. Always consider market conditions and monitor actual trading decisions.
