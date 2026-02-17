#!/usr/bin/env python3
"""
Simple Portfolio Forecast - Clean projections based on current performance

Shows realistic future portfolio values based on:
- Current 13% return trajectory
- Conservative/moderate/aggressive scenarios
- Time-based projections (1 week, 1 month, 3 months, 6 months, 1 year)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path


def load_current_state():
    """Load current portfolio state."""
    positions_file = Path("paper_trades/positions_state.json")
    with open(positions_file) as f:
        return json.load(f)


def calculate_daily_rate(current_value, initial_value, days_elapsed):
    """Calculate compound daily growth rate."""
    total_return = current_value / initial_value
    daily_rate = total_return ** (1 / days_elapsed) - 1
    return daily_rate


def forecast_value(current_value, daily_rate, days_forward):
    """Project portfolio value forward."""
    return current_value * (1 + daily_rate) ** days_forward


def generate_forecast():
    """Generate clean, readable forecast."""
    state = load_current_state()

    current_value = state['portfolio_value']
    initial_value = state['portfolio_protection']['initial_value']
    high_water_mark = state['portfolio_protection']['high_water_mark']

    current_return = (current_value / initial_value - 1) * 100

    # Estimate time elapsed (from positions or assume recent)
    # For now, assume 7 days based on typical paper trading duration
    assumed_days = 7
    daily_rate = calculate_daily_rate(current_value, initial_value, assumed_days)

    print("=" * 90)
    print(" " * 25 + "PORTFOLIO FORECAST - Future Valuations")
    print("=" * 90)
    print()

    # Current snapshot
    print("CURRENT PORTFOLIO STATE")
    print("-" * 90)
    print(f"  Current Value:        ${current_value:,.2f}")
    print(f"  Initial Investment:   ${initial_value:,.2f}")
    print(f"  Total Return:         {current_return:+.2f}%")
    print(f"  Peak Value (HWM):     ${high_water_mark:,.2f}")
    print(f"  Daily Return Rate:    {daily_rate*100:+.3f}%")
    print()

    # Current positions
    print("ACTIVE POSITIONS")
    print("-" * 90)
    total_position_value = 0
    for pos in state['positions']:
        print(f"  {pos['ticker']:10s}  "
              f"{pos['holdings']:8.4f} @ ${pos['current_price']:9.2f}  =  "
              f"${pos['position_value']:9.2f}  ({pos['pnl_pct']:+6.2f}%)")
        total_position_value += pos['position_value']
    print(f"  {'Cash':10s}  ${state['cash']:,.2f}")
    print(f"  {'TOTAL':10s}  ${current_value:,.2f}")
    print()

    # Scenario definitions
    scenarios = {
        'Conservative (50% rate)': daily_rate * 0.5,
        'Moderate (current rate)': daily_rate,
        'Aggressive (150% rate)': daily_rate * 1.5,
    }

    # Time periods
    periods = [
        ('1 Week', 7),
        ('2 Weeks', 14),
        ('1 Month', 30),
        ('2 Months', 60),
        ('3 Months', 90),
        ('6 Months', 180),
        ('1 Year', 365),
    ]

    print("PROJECTED PORTFOLIO VALUES")
    print("-" * 90)

    # Header
    print(f"{'Time Period':<15s} {'Conservative':>15s} {'Moderate':>15s} {'Aggressive':>15s}")
    print("-" * 90)

    for period_name, days in periods:
        values = []
        for scenario_name, rate in scenarios.items():
            projected_value = forecast_value(current_value, rate, days)
            values.append(projected_value)

        cons_val, mod_val, agg_val = values
        print(f"{period_name:<15s} "
              f"${cons_val:>13,.2f} "
              f"${mod_val:>13,.2f} "
              f"${agg_val:>13,.2f}")

    print()

    # Return percentages
    print("PROJECTED RETURNS (from current value)")
    print("-" * 90)
    print(f"{'Time Period':<15s} {'Conservative':>15s} {'Moderate':>15s} {'Aggressive':>15s}")
    print("-" * 90)

    for period_name, days in periods:
        returns = []
        for scenario_name, rate in scenarios.items():
            projected_value = forecast_value(current_value, rate, days)
            return_pct = (projected_value / current_value - 1) * 100
            returns.append(return_pct)

        cons_ret, mod_ret, agg_ret = returns
        print(f"{period_name:<15s} "
              f"{cons_ret:>+14.2f}% "
              f"{mod_ret:>+14.2f}% "
              f"{agg_ret:>+14.2f}%")

    print()

    # Key milestones
    print("KEY MILESTONES")
    print("-" * 90)

    milestones = [
        ('Double initial investment', initial_value * 2),
        ('Triple initial investment', initial_value * 3),
        ('Reach $2,000', 2000),
        ('Reach $5,000', 5000),
        ('Reach $10,000', 10000),
    ]

    print(f"{'Milestone':<30s} {'Days (Moderate)':>20s} {'Date (Moderate)':>25s}")
    print("-" * 90)

    for milestone_name, target_value in milestones:
        # Calculate days needed at moderate rate
        if target_value > current_value and daily_rate > 0:
            days_needed = np.log(target_value / current_value) / np.log(1 + daily_rate)
            target_date = datetime.now() + timedelta(days=int(days_needed))
            print(f"{milestone_name:<30s} "
                  f"{int(days_needed):>18d} days "
                  f"{target_date.strftime('%Y-%m-%d'):>25s}")
        elif target_value <= current_value:
            print(f"{milestone_name:<30s} {'Already achieved!':>45s}")
        else:
            print(f"{milestone_name:<30s} {'N/A (negative rate)':>45s}")

    print()

    # Risk notes
    print("IMPORTANT NOTES")
    print("-" * 90)
    print("  • Projections based on current performance trajectory")
    print(f"  • Current daily return rate: {daily_rate*100:+.3f}%")
    print("  • Conservative scenario: 50% of current rate (realistic if market conditions worsen)")
    print("  • Moderate scenario: Current rate continues (assumes stable conditions)")
    print("  • Aggressive scenario: 150% of current rate (optimistic, requires improvement)")
    print("  • Past performance does not guarantee future results")
    print("  • Crypto markets are highly volatile - actual results may vary significantly")
    print()

    print("=" * 90)
    print(f"Forecast generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)


if __name__ == "__main__":
    import numpy as np  # For log calculation
    generate_forecast()
