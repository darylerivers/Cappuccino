#!/usr/bin/env python3
"""
Test script to verify concentration limit fix works correctly.

This simulates the bug scenario where a model wants to buy
49% of portfolio in a single asset, and verifies the fix
caps it at 30%.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from environment_Alpaca import CryptoEnvAlpaca


def test_concentration_limit():
    """Test that concentration limit is enforced on scaled actions."""

    print("=" * 80)
    print("CONCENTRATION LIMIT FIX - TEST")
    print("=" * 80)
    print()

    # Create minimal environment config
    np.random.seed(42)

    # Create fake price data (7 assets, 100 timesteps)
    num_assets = 7
    num_steps = 100
    lookback = 20

    # Prices around realistic crypto values
    base_prices = np.array([95000, 3300, 75, 600, 14, 5, 175])  # BTC, ETH, LTC, BCH, LINK, UNI, AAVE

    price_array = np.zeros((num_steps, num_assets))
    for i in range(num_steps):
        # Add some random walk
        price_array[i] = base_prices * (1 + np.random.randn(num_assets) * 0.01)

    # Create fake technical indicators (7 assets * 11 indicators = 77 features)
    tech_array = np.random.randn(num_steps, 77).astype(np.float32)

    config = {
        'price_array': price_array,
        'tech_array': tech_array,
    }

    env_params = {
        'lookback': lookback,
        'norm_cash': 1000,
        'norm_stocks': 1,
        'norm_tech': 1,
        'norm_reward': 1,
        'norm_action': 10000,  # This is the key - action scaling factor
    }

    tickers = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD', 'LINK/USD', 'UNI/USD', 'AAVE/USD']

    # Create environment
    env = CryptoEnvAlpaca(
        config=config,
        env_params=env_params,
        initial_capital=1000,
        if_log=False,
        tickers=tickers
    )

    # Set concentration limit
    env.max_position_pct = 0.30  # 30% max per asset

    print(f"Initial capital: ${env.cash:.2f}")
    print(f"Concentration limit: {env.max_position_pct * 100:.0f}%")
    print()

    # Reset environment
    state = env.reset()

    # Warm up to a reasonable timestep
    for _ in range(lookback + 10):
        zero_action = np.zeros(num_assets)
        state, _, done, _ = env.step(zero_action)
        if done:
            break

    print(f"Starting cash: ${env.cash:.2f}")
    print(f"Current step: {env.time}")
    print()

    # Test Case 1: Model wants to buy 49% of portfolio in LINK (asset index 4)
    print("=" * 80)
    print("TEST CASE 1: Model wants 49% concentration in LINK/USD")
    print("=" * 80)
    print()

    # Calculate what un-scaled action would result in 49% concentration
    current_price = env.price_array[env.time]
    link_price = current_price[4]  # LINK/USD

    # Want 49% of $1000 = $490 in LINK
    target_value = 0.49 * env.cash
    target_shares = target_value / link_price

    # This is the SCALED action (what we want after scaling)
    # To get the UN-SCALED action, divide by norm_vector
    norm_vector = env.action_norm_vector[4]

    # Create action that would result in 49% concentration WITHOUT the fix
    # Un-scaled action that becomes target_shares after scaling
    unscaled_action = target_shares / norm_vector

    print(f"LINK price: ${link_price:.2f}")
    print(f"Target: 49% concentration = ${target_value:.2f}")
    print(f"Target shares: {target_shares:.2f}")
    print(f"Action norm vector[LINK]: {norm_vector:.2f}")
    print(f"Un-scaled action: {unscaled_action:.6f}")
    print(f"Scaled action (before fix): {unscaled_action * norm_vector:.2f} shares")
    print()

    # Create action array (only buy LINK)
    action = np.zeros(num_assets)
    action[4] = unscaled_action  # LINK

    print("Executing step with concentration limit fix enabled...")
    print()

    # Execute step - the environment should cap this to 30%
    state, reward, done, info = env.step(action)

    # Check result
    link_holdings = env.stocks[4]
    link_value = link_holdings * link_price
    total_value = env.cash + np.sum(env.stocks * current_price)
    actual_concentration = (link_value / total_value) * 100 if total_value > 0 else 0

    print("RESULTS:")
    print(f"  LINK holdings: {link_holdings:.2f} shares")
    print(f"  LINK value: ${link_value:.2f}")
    print(f"  Total portfolio: ${total_value:.2f}")
    print(f"  Actual concentration: {actual_concentration:.1f}%")
    print()

    # Verify fix worked
    if actual_concentration <= 30.1:  # Allow tiny rounding error
        print("✅ TEST PASSED: Concentration capped at 30%")
        test1_pass = True
    else:
        print(f"❌ TEST FAILED: Concentration {actual_concentration:.1f}% exceeds 30% limit!")
        test1_pass = False

    print()
    print()

    # Test Case 2: Model wants to buy multiple assets, all under limit
    print("=" * 80)
    print("TEST CASE 2: Model wants to buy multiple assets under limit")
    print("=" * 80)
    print()

    # Reset environment
    env.cash = 1000
    env.stocks = np.zeros(num_assets)

    # Create action to buy 20% in 3 different assets
    action2 = np.zeros(num_assets)
    for idx in [0, 1, 4]:  # BTC, ETH, LINK
        price_i = current_price[idx]
        target_value_i = 0.20 * 1000  # 20% of portfolio
        target_shares_i = target_value_i / price_i
        norm_vec_i = env.action_norm_vector[idx]
        action2[idx] = target_shares_i / norm_vec_i

    print("Trying to buy 20% in BTC, ETH, and LINK...")
    print()

    state2, reward2, done2, info2 = env.step(action2)

    # Check concentrations
    total_value2 = env.cash + np.sum(env.stocks * current_price)

    print("RESULTS:")
    for idx, ticker in enumerate(tickers):
        if env.stocks[idx] > 0:
            value = env.stocks[idx] * current_price[idx]
            pct = (value / total_value2) * 100
            print(f"  {ticker}: {pct:.1f}% (${value:.2f})")

    print(f"  Total portfolio: ${total_value2:.2f}")
    print()

    # All should be under 30%
    concentrations = []
    for idx in range(num_assets):
        if env.stocks[idx] > 0:
            value = env.stocks[idx] * current_price[idx]
            pct = (value / total_value2) * 100
            concentrations.append(pct)

    if all(c <= 30.1 for c in concentrations):
        print("✅ TEST PASSED: All concentrations under 30%")
        test2_pass = True
    else:
        print("❌ TEST FAILED: Some concentrations exceed 30%")
        test2_pass = False

    print()
    print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"Test 1 (Block 49% concentration): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (Allow <30% concentration): {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print()

    if test1_pass and test2_pass:
        print("🎉 ALL TESTS PASSED - Fix is working correctly!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Fix needs adjustment")
        return 1


if __name__ == "__main__":
    sys.exit(test_concentration_limit())
