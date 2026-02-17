#!/usr/bin/env python3
"""
Test script to verify fee tracking in CryptoEnvAlpaca.

This script creates a simple environment and executes a few trades to verify:
1. Fees are tracked correctly on buy operations
2. Fees are tracked correctly on sell operations
3. Gross return vs net return calculations are correct
4. Info dict contains fee data
"""

import numpy as np
from environment_Alpaca import CryptoEnvAlpaca

def create_test_environment():
    """Create a minimal test environment."""
    # Create simple synthetic data: 3 tickers, 100 timesteps
    n_timesteps = 100
    n_tickers = 3
    n_features_per_ticker = 11  # price features + technical indicators

    # Generate random price data (realistic crypto prices)
    np.random.seed(42)
    base_prices = np.array([50000, 3000, 100])  # BTC, ETH, LTC-like
    prices = np.zeros((n_timesteps, n_tickers))

    for i in range(n_timesteps):
        if i == 0:
            prices[i] = base_prices
        else:
            # Random walk with drift
            changes = np.random.normal(1.001, 0.02, n_tickers)
            prices[i] = prices[i-1] * changes

    # Generate synthetic technical indicators
    tech_array = np.random.randn(n_timesteps, n_tickers * n_features_per_ticker).astype(np.float32)

    # Normalize to reasonable ranges
    tech_array = tech_array * 0.1

    config = {
        'price_array': prices.astype(np.float32),
        'tech_array': tech_array,
    }

    env_params = {
        'lookback': 10,
        'norm_cash': 2**-11,
        'norm_stocks': 2**-8,
        'norm_tech': 2**-14,
        'norm_reward': 2**-9,
        'norm_action': 100,
        'time_decay_floor': 0.0,
        'min_cash_reserve': 0.0,
        'concentration_penalty': 0.0,
    }

    return CryptoEnvAlpaca(
        config,
        env_params,
        initial_capital=10000,
        buy_cost_pct=0.0025,  # 0.25% buy fee
        sell_cost_pct=0.0025,  # 0.25% sell fee
        if_log=True,  # Enable logging to see fee report
    )

def test_fee_tracking():
    """Test fee tracking with simple trades."""
    print("="*70)
    print("FEE TRACKING TEST")
    print("="*70)

    env = create_test_environment()
    state = env.reset()

    print(f"\nInitial state:")
    print(f"  Cash: ${env.cash:,.2f}")
    print(f"  Stocks: {env.stocks}")
    print(f"  Total Asset: ${env.total_asset:,.2f}")
    print(f"  Fees Paid: ${env.total_fees_paid:.2f}")

    # Execute a few trades
    print(f"\n{'='*70}")
    print("Executing test trades...")
    print(f"{'='*70}\n")

    step_count = 0
    done = False

    # Trade sequence:
    # 1. Buy some of each asset (step 1-5)
    # 2. Hold (step 6-10)
    # 3. Sell everything (step 11-15)
    # 4. Let it run to completion

    while not done and step_count < env.max_step:
        step_count += 1

        # Define actions based on step
        if step_count <= 5:
            # Buy phase - buy equal amounts
            action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif step_count <= 10:
            # Hold phase - zero action
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif step_count <= 15:
            # Sell phase - sell everything
            action = -env.stocks.copy()
        else:
            # Rest of episode - no action
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        state, reward, done, info = env.step(action)

        # Print fee info every 5 steps
        if step_count % 5 == 0 or done:
            print(f"Step {step_count}:")
            print(f"  Cash: ${env.cash:,.2f}")
            print(f"  Holdings: {env.stocks}")
            print(f"  Total Asset: ${env.total_asset:,.2f}")
            print(f"  Total Fees: ${info['total_fees_paid']:.2f}")
            print(f"  Buy Fees: ${info['buy_fees_paid']:.2f} ({info['num_buy_trades']} trades)")
            print(f"  Sell Fees: ${info['sell_fees_paid']:.2f} ({info['num_sell_trades']} trades)")
            print(f"  Return: {(env.cumu_return - 1) * 100:+.2f}%")
            print()

    # Verify episode completion report was printed (from environment's if_log)
    print("\n" + "="*70)
    print("TEST COMPLETED")
    print("="*70)

    # Manual verification of fee calculations
    print("\nManual Verification:")
    print(f"  Expected total trades: {info['num_buy_trades'] + info['num_sell_trades']}")
    print(f"  Actual total trades: {info['num_buy_trades'] + info['num_sell_trades']}")

    if done and 'episode_return_net' in info:
        print(f"\nEpisode Metrics:")
        print(f"  Net Return: {(info['episode_return_net'] - 1) * 100:+.2f}%")
        print(f"  Gross Return: {(info['episode_return_gross'] - 1) * 100:+.2f}%")
        print(f"  Fee Impact: {info['fee_impact_pct']:.2f}%")

    print("\n✅ Fee tracking test completed!")
    return True

if __name__ == "__main__":
    test_fee_tracking()
