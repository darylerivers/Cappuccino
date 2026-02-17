#!/usr/bin/env python3
"""
12-Month Return Projection for $500 Starting Capital

Scenarios based on Sharpe ratios and compounding with retraining.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def project_returns(
    initial_capital=500,
    sharpe_ratio=0.22,
    annual_volatility=0.50,
    months=12,
    trading_days_per_year=365,  # Crypto trades 24/7
    retraining_boost=1.05,  # 5% improvement from retraining
    market_regime_factor=0.85,  # Account for changing market conditions
    num_simulations=10000
):
    """
    Monte Carlo simulation of 12-month returns.

    Args:
        initial_capital: Starting account balance
        sharpe_ratio: Expected Sharpe ratio (return/volatility)
        annual_volatility: Expected annual volatility
        months: Number of months to project
        trading_days_per_year: Trading frequency
        retraining_boost: Improvement from retraining with new data
        market_regime_factor: Degradation from market regime changes
        num_simulations: Number of Monte Carlo paths
    """

    # Calculate expected return from Sharpe ratio
    # Sharpe = (Return - RiskFreeRate) / Volatility
    # Assuming risk-free rate ~5% annually
    risk_free_rate = 0.05
    expected_annual_return = sharpe_ratio * annual_volatility + risk_free_rate

    # Adjust for market conditions and model degradation
    adjusted_return = expected_annual_return * market_regime_factor

    # Daily parameters
    days = int(months * 30.44)  # Average days per month
    daily_return = adjusted_return / trading_days_per_year
    daily_vol = annual_volatility / np.sqrt(trading_days_per_year)

    # Run simulations
    results = []

    for sim in range(num_simulations):
        capital = initial_capital
        monthly_capitals = [capital]

        for month in range(months):
            # Apply retraining boost every 3 months
            if month > 0 and month % 3 == 0:
                daily_return *= retraining_boost
                # But also apply some degradation from market changes
                daily_return *= 0.98

            # Simulate daily returns for this month
            days_in_month = 30
            for day in range(days_in_month):
                # Random return from normal distribution
                daily_ret = np.random.normal(daily_return, daily_vol)
                capital *= (1 + daily_ret)

                # Risk of ruin check (account below $100 = stop trading)
                if capital < 100:
                    capital = max(capital, 0)
                    break

            monthly_capitals.append(capital)

        results.append(monthly_capitals)

    return np.array(results)


def analyze_scenarios():
    """Analyze different scenarios for 12-month returns."""

    scenarios = {
        'Conservative (Ensemble)': {
            'sharpe': 0.20,
            'volatility': 0.45,
            'description': 'Ensemble model, conservative risk management'
        },
        'Expected (Ensemble)': {
            'sharpe': 0.25,
            'volatility': 0.50,
            'description': 'Ensemble model with average performance'
        },
        'Optimistic (Good Market)': {
            'sharpe': 0.35,
            'volatility': 0.55,
            'description': 'Ensemble + favorable market conditions'
        },
        'Pessimistic (Bad Market)': {
            'sharpe': 0.10,
            'volatility': 0.60,
            'description': 'Model struggles in difficult market'
        },
        'Reality Check': {
            'sharpe': 0.15,
            'volatility': 0.55,
            'description': 'Realistic with model degradation and surprises'
        }
    }

    print("="*80)
    print("12-MONTH RETURN PROJECTION: $500 INITIAL CAPITAL")
    print("="*80)
    print()
    print("Assumptions:")
    print("  • Starting capital: $500")
    print("  • Trading: 24/7 (crypto markets)")
    print("  • Retraining: Every 3 months with new data")
    print("  • Compounding: All profits reinvested")
    print("  • Risk management: Stop if account < $100")
    print("  • Simulations: 10,000 Monte Carlo paths per scenario")
    print()
    print("="*80)
    print()

    for scenario_name, params in scenarios.items():
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*80}")
        print(f"Description: {params['description']}")
        print(f"Sharpe Ratio: {params['sharpe']:.2f}")
        print(f"Annual Volatility: {params['volatility']*100:.0f}%")
        print()

        # Run simulation
        results = project_returns(
            initial_capital=500,
            sharpe_ratio=params['sharpe'],
            annual_volatility=params['volatility']
        )

        # Analyze results
        final_capitals = results[:, -1]

        # Statistics
        median = np.median(final_capitals)
        mean = np.mean(final_capitals)
        p10 = np.percentile(final_capitals, 10)
        p25 = np.percentile(final_capitals, 25)
        p75 = np.percentile(final_capitals, 75)
        p90 = np.percentile(final_capitals, 90)
        best = np.max(final_capitals)
        worst = np.min(final_capitals)

        # Probability of profit
        profitable = (final_capitals > 500).sum() / len(final_capitals) * 100
        doubled = (final_capitals > 1000).sum() / len(final_capitals) * 100
        wiped = (final_capitals < 100).sum() / len(final_capitals) * 100

        # Returns
        median_return = (median - 500) / 500 * 100
        mean_return = (mean - 500) / 500 * 100

        print(f"12-Month Projections:")
        print(f"  Median Final Capital:    ${median:,.2f}   ({median_return:+.1f}%)")
        print(f"  Mean Final Capital:      ${mean:,.2f}   ({mean_return:+.1f}%)")
        print()
        print(f"  Best Case (Top 10%):     ${p90:,.2f}")
        print(f"  75th Percentile:         ${p75:,.2f}")
        print(f"  25th Percentile:         ${p25:,.2f}")
        print(f"  Worst Case (Bottom 10%): ${p10:,.2f}")
        print()
        print(f"  Best Outcome:            ${best:,.2f}   ({(best-500)/500*100:+.1f}%)")
        print(f"  Worst Outcome:           ${worst:,.2f}   ({(worst-500)/500*100:+.1f}%)")
        print()
        print(f"Probabilities:")
        print(f"  Profitable (> $500):     {profitable:.1f}%")
        print(f"  Double Money (> $1000):  {doubled:.1f}%")
        print(f"  Significant Loss (< $100): {wiped:.1f}%")
        print()

        # Monthly progression (median path)
        monthly_medians = np.median(results, axis=0)
        print(f"Median Monthly Progression:")
        for month, capital in enumerate(monthly_medians):
            if month == 0:
                print(f"  Month {month:2d} (Start):  ${capital:,.2f}")
            else:
                monthly_return = (capital - monthly_medians[month-1]) / monthly_medians[month-1] * 100
                print(f"  Month {month:2d}:          ${capital:,.2f}  ({monthly_return:+.1f}%)")


if __name__ == '__main__':
    np.random.seed(42)  # Reproducible results
    analyze_scenarios()

    print("\n" + "="*80)
    print("IMPORTANT DISCLAIMERS")
    print("="*80)
    print()
    print("⚠️  These are PROJECTIONS, not guarantees!")
    print("⚠️  Past performance (backtests) ≠ future results")
    print("⚠️  Crypto is highly volatile and unpredictable")
    print("⚠️  Model performance can degrade over time")
    print("⚠️  Black swan events can wipe out accounts")
    print("⚠️  Always trade with money you can afford to lose")
    print()
    print("📊 These scenarios assume:")
    print("   • Model continues to work as expected")
    print("   • No major bugs or system failures")
    print("   • Markets remain somewhat similar to training data")
    print("   • Risk management works as designed")
    print("   • You rebalance/retrain every 3 months")
    print()
    print("="*80)
