#!/usr/bin/env python3
"""
Analyze market regime characteristics between old and new data
"""
import numpy as np
import os
from pathlib import Path

def load_data(data_dir):
    """Load price and technical indicator data"""
    price_array = np.load(f"{data_dir}/price_array.npy")
    tech_array = np.load(f"{data_dir}/tech_array.npy")
    time_array = np.load(f"{data_dir}/time_array.npy")
    return price_array, tech_array, time_array

def analyze_volatility(price_data):
    """Calculate volatility metrics"""
    # Price data shape: (n_samples, n_assets)
    # Calculate returns
    returns = np.diff(price_data, axis=0) / price_data[:-1]

    # Volatility metrics
    volatility = np.std(returns, axis=0)
    mean_return = np.mean(returns, axis=0)

    return {
        'mean_volatility': np.mean(volatility),
        'max_volatility': np.max(volatility),
        'min_volatility': np.min(volatility),
        'mean_return': np.mean(mean_return),
        'volatility_by_asset': volatility,
        'returns_by_asset': mean_return
    }

def analyze_trends(price_data, window=168):
    """Analyze trend characteristics (168 hours = 1 week)"""
    # Calculate moving average trends
    n_samples = price_data.shape[0]
    n_assets = price_data.shape[1]

    trends = []
    for i in range(n_assets):
        asset_prices = price_data[:, i]

        # Calculate % above/below moving average
        if n_samples > window:
            ma = np.convolve(asset_prices, np.ones(window)/window, mode='valid')
            # Compare recent prices to MA
            recent_vs_ma = (asset_prices[window-1:] - ma) / ma
            trends.append({
                'mean_deviation': np.mean(recent_vs_ma),
                'positive_pct': np.mean(recent_vs_ma > 0) * 100
            })

    mean_deviation = np.mean([t['mean_deviation'] for t in trends])
    mean_positive_pct = np.mean([t['positive_pct'] for t in trends])

    return {
        'mean_ma_deviation': mean_deviation,
        'mean_above_ma_pct': mean_positive_pct,
        'trends_by_asset': trends
    }

def analyze_correlations(price_data):
    """Analyze asset correlations"""
    returns = np.diff(price_data, axis=0) / price_data[:-1]
    corr_matrix = np.corrcoef(returns.T)

    # Get upper triangle (excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

    return {
        'mean_correlation': np.mean(upper_tri),
        'max_correlation': np.max(upper_tri),
        'min_correlation': np.min(upper_tri),
        'correlation_matrix': corr_matrix
    }

def analyze_drawdowns(price_data):
    """Analyze drawdown characteristics"""
    n_assets = price_data.shape[1]
    max_drawdowns = []

    for i in range(n_assets):
        prices = price_data[:, i]
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        max_drawdowns.append(np.min(drawdown))

    return {
        'mean_max_drawdown': np.mean(max_drawdowns),
        'worst_drawdown': np.min(max_drawdowns),
        'drawdowns_by_asset': max_drawdowns
    }

def main():
    # Data directories
    old_data_dir = "data/2year_fresh_20251218"
    new_data_dir = "data/2year_fresh_20260112"

    print("=" * 100)
    print("MARKET REGIME ANALYSIS: Dec 18, 2025 vs Jan 12, 2026")
    print("=" * 100)

    # Check if old data exists
    if not os.path.exists(old_data_dir):
        print(f"\n⚠️  Old data not found at {old_data_dir}")
        print("Comparing only the fresh data characteristics...")
        old_data_dir = None

    # Load data
    print("\nLoading data...")
    new_price, new_tech, new_time = load_data(new_data_dir)
    print(f"✓ Loaded fresh data: {new_price.shape[0]} samples, {new_price.shape[1]} assets")

    if old_data_dir:
        old_price, old_tech, old_time = load_data(old_data_dir)
        print(f"✓ Loaded old data: {old_price.shape[0]} samples, {old_price.shape[1]} assets")

    # Analyze fresh data
    print("\n" + "=" * 100)
    print("FRESH DATA CHARACTERISTICS (Jan 12, 2026)")
    print("=" * 100)

    new_vol = analyze_volatility(new_price)
    new_trend = analyze_trends(new_price)
    new_corr = analyze_correlations(new_price)
    new_dd = analyze_drawdowns(new_price)

    print(f"\nVolatility:")
    print(f"  Mean volatility:     {new_vol['mean_volatility']:.6f}")
    print(f"  Min volatility:      {new_vol['min_volatility']:.6f}")
    print(f"  Max volatility:      {new_vol['max_volatility']:.6f}")
    print(f"  Mean return:         {new_vol['mean_return']:.6f}")

    print(f"\nTrend Characteristics:")
    print(f"  Mean MA deviation:   {new_trend['mean_ma_deviation']:.4f}")
    print(f"  % above MA:          {new_trend['mean_above_ma_pct']:.1f}%")

    print(f"\nCorrelations:")
    print(f"  Mean correlation:    {new_corr['mean_correlation']:.4f}")
    print(f"  Min correlation:     {new_corr['min_correlation']:.4f}")
    print(f"  Max correlation:     {new_corr['max_correlation']:.4f}")

    print(f"\nDrawdowns:")
    print(f"  Mean max drawdown:   {new_dd['mean_max_drawdown']:.4f} ({new_dd['mean_max_drawdown']*100:.1f}%)")
    print(f"  Worst drawdown:      {new_dd['worst_drawdown']:.4f} ({new_dd['worst_drawdown']*100:.1f}%)")

    if old_data_dir:
        # Analyze old data
        print("\n" + "=" * 100)
        print("OLD DATA CHARACTERISTICS (Dec 18, 2025)")
        print("=" * 100)

        old_vol = analyze_volatility(old_price)
        old_trend = analyze_trends(old_price)
        old_corr = analyze_correlations(old_price)
        old_dd = analyze_drawdowns(old_price)

        print(f"\nVolatility:")
        print(f"  Mean volatility:     {old_vol['mean_volatility']:.6f}")
        print(f"  Min volatility:      {old_vol['min_volatility']:.6f}")
        print(f"  Max volatility:      {old_vol['max_volatility']:.6f}")
        print(f"  Mean return:         {old_vol['mean_return']:.6f}")

        print(f"\nTrend Characteristics:")
        print(f"  Mean MA deviation:   {old_trend['mean_ma_deviation']:.4f}")
        print(f"  % above MA:          {old_trend['mean_above_ma_pct']:.1f}%")

        print(f"\nCorrelations:")
        print(f"  Mean correlation:    {old_corr['mean_correlation']:.4f}")
        print(f"  Min correlation:     {old_corr['min_correlation']:.4f}")
        print(f"  Max correlation:     {old_corr['max_correlation']:.4f}")

        print(f"\nDrawdowns:")
        print(f"  Mean max drawdown:   {old_dd['mean_max_drawdown']:.4f} ({old_dd['mean_max_drawdown']*100:.1f}%)")
        print(f"  Worst drawdown:      {old_dd['worst_drawdown']:.4f} ({old_dd['worst_drawdown']*100:.1f}%)")

        # Compare
        print("\n" + "=" * 100)
        print("REGIME CHANGE ANALYSIS (Old → New)")
        print("=" * 100)

        vol_change = (new_vol['mean_volatility'] - old_vol['mean_volatility']) / old_vol['mean_volatility'] * 100
        ret_change = (new_vol['mean_return'] - old_vol['mean_return']) / abs(old_vol['mean_return']) * 100 if old_vol['mean_return'] != 0 else 0
        corr_change = (new_corr['mean_correlation'] - old_corr['mean_correlation']) / old_corr['mean_correlation'] * 100
        dd_change = (new_dd['mean_max_drawdown'] - old_dd['mean_max_drawdown']) / abs(old_dd['mean_max_drawdown']) * 100

        print(f"\nVolatility Change:        {vol_change:+.1f}% {'🔴 Higher' if vol_change > 0 else '🟢 Lower'}")
        print(f"Return Change:            {ret_change:+.1f}%")
        print(f"Correlation Change:       {corr_change:+.1f}% {'🔴 More correlated' if corr_change > 0 else '🟢 Less correlated'}")
        print(f"Drawdown Change:          {dd_change:+.1f}% {'🔴 Deeper' if dd_change < 0 else '🟢 Shallower'}")

        print("\n" + "=" * 100)
        print("KEY INSIGHTS")
        print("=" * 100)

        insights = []

        if abs(vol_change) > 10:
            direction = "increased" if vol_change > 0 else "decreased"
            insights.append(f"1. Volatility has {direction} significantly ({abs(vol_change):.1f}%)")
            insights.append(f"   → Models may need different risk management parameters")

        if abs(corr_change) > 10:
            direction = "increased" if corr_change > 0 else "decreased"
            insights.append(f"2. Asset correlations have {direction} ({abs(corr_change):.1f}%)")
            if corr_change > 0:
                insights.append(f"   → Diversification benefits reduced, harder to find uncorrelated opportunities")
            else:
                insights.append(f"   → More diversification opportunities available")

        if abs(dd_change) > 10:
            direction = "deeper" if dd_change < 0 else "shallower"
            insights.append(f"3. Drawdowns are {direction} ({abs(dd_change):.1f}%)")

        if insights:
            print("\n" + "\n".join(insights))
        else:
            print("\nNo major regime changes detected (< 10% change in key metrics)")

        print("\n" + "=" * 100)

    # Analyze recent period (last 30 days)
    print("\nRECENT MARKET ACTIVITY (Last ~30 days / 720 hours)")
    print("=" * 100)

    recent_samples = min(720, new_price.shape[0])
    recent_price = new_price[-recent_samples:]

    recent_vol = analyze_volatility(recent_price)
    recent_dd = analyze_drawdowns(recent_price)

    print(f"\nRecent volatility:        {recent_vol['mean_volatility']:.6f}")
    print(f"Recent mean return:       {recent_vol['mean_return']:.6f}")
    print(f"Recent max drawdown:      {recent_dd['mean_max_drawdown']:.4f} ({recent_dd['mean_max_drawdown']*100:.1f}%)")

    if old_data_dir:
        vol_vs_full = (recent_vol['mean_volatility'] - new_vol['mean_volatility']) / new_vol['mean_volatility'] * 100
        print(f"\nRecent vs full dataset:")
        print(f"  Volatility:             {vol_vs_full:+.1f}% {'🔴 More volatile' if vol_vs_full > 0 else '🟢 Less volatile'}")

    print("\n" + "=" * 100)
    print("\n✓ Market regime analysis complete!")
    print("\n")

if __name__ == "__main__":
    main()
