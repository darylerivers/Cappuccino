#!/usr/bin/env python3
"""
Test script to verify trailing stop loss functionality in CryptoEnvAlpaca.

This script creates a simple test case with simulated price movements to verify
that the trailing stop loss mechanism works correctly.
"""

import numpy as np
from environment_Alpaca import CryptoEnvAlpaca


def test_trailing_stop_loss():
    """Test trailing stop loss with a simple price scenario."""
    print("="*80)
    print("Testing Trailing Stop Loss Functionality")
    print("="*80)

    # Create simple price data: 1 asset, price goes 100 -> 110 -> 105 -> 95 (should trigger at 99)
    # With 10% trailing stop: highest is 110, stop should trigger at 110 * 0.90 = 99
    lookback = 2
    n_timesteps = 10
    n_assets = 1

    # Price trajectory: start at 100, go to 110 (peak), then decline
    prices = np.array([
        [100.0],  # t=0 (lookback)
        [100.0],  # t=1 (lookback)
        [105.0],  # t=2 - buy here
        [110.0],  # t=3 - new high
        [109.0],  # t=4 - small dip
        [105.0],  # t=5 - larger dip but still above stop (110 * 0.9 = 99)
        [100.0],  # t=6 - getting close
        [98.0],   # t=7 - TRIGGER STOP LOSS (below 99)
        [95.0],   # t=8 - after stop
        [90.0],   # t=9 - after stop
    ])

    # Simple tech indicators (just copy prices for simplicity)
    tech_array = np.repeat(prices, 3, axis=1)  # 3 indicators per asset

    # Environment parameters with 10% trailing stop
    env_params = {
        'lookback': lookback,
        'norm_cash': 2**-15,
        'norm_stocks': 2**-10,
        'norm_tech': 2**-18,
        'norm_reward': 2**-13,
        'norm_action': 15000,
        'time_decay_floor': 0.2,
        'min_cash_reserve': 0.0,
        'concentration_penalty': 0.0,
        'trailing_stop_pct': 0.10,  # 10% trailing stop
    }

    config = {
        'price_array': prices,
        'tech_array': tech_array,
    }

    # Create environment
    env = CryptoEnvAlpaca(
        config=config,
        env_params=env_params,
        initial_capital=1000.0,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        if_log=False
    )

    print(f"\nEnvironment created:")
    print(f"  Initial capital: $1000")
    print(f"  Trailing stop: {env.trailing_stop_pct:.1%}")
    print(f"  Price trajectory: {prices.flatten().tolist()}")

    # Reset environment
    state = env.reset()

    # Step 1: Buy at t=2 (price = 105)
    print(f"\n{'='*80}")
    print("Step 1: Buy 9 shares at price $105")
    buy_action = np.array([9.0])  # Buy 9 shares
    state, reward, done, _ = env.step(buy_action)
    print(f"  Position: {env.stocks[0]:.4f} shares")
    print(f"  Cash: ${env.cash:.2f}")
    print(f"  Highest price since buy: ${env.highest_price_since_buy[0]:.2f}")
    assert env.stocks[0] > 0, "Should have bought shares"
    assert env.highest_price_since_buy[0] == 105.0, "Highest price should be initialized to buy price"

    # Step 2: Price goes to 110 (new high)
    print(f"\n{'='*80}")
    print("Step 2: Hold as price rises to $110")
    hold_action = np.array([0.0])
    state, reward, done, _ = env.step(hold_action)
    print(f"  Current price: ${env.price_array[env.time][0]:.2f}")
    print(f"  Position: {env.stocks[0]:.4f} shares")
    print(f"  Highest price since buy: ${env.highest_price_since_buy[0]:.2f}")
    print(f"  Stop loss trigger at: ${env.highest_price_since_buy[0] * (1 - env.trailing_stop_pct):.2f}")
    assert env.highest_price_since_buy[0] == 110.0, "Highest price should update to 110"

    # Step 3-5: Price declines to 109, 105, 100 (still above stop of 99)
    for step in range(3, 6):
        print(f"\n{'='*80}")
        print(f"Step {step}: Price declines to ${env.price_array[env.time+1][0]:.2f}")
        state, reward, done, _ = env.step(hold_action)
        print(f"  Position: {env.stocks[0]:.4f} shares")
        print(f"  Highest price: ${env.highest_price_since_buy[0]:.2f}")
        print(f"  Stop loss trigger at: ${env.highest_price_since_buy[0] * (1 - env.trailing_stop_pct):.2f}")
        print(f"  Stop triggered: {env.trailing_stop_triggered[0]}")
        assert env.stocks[0] > 0, f"Should still hold position at step {step}"
        assert not env.trailing_stop_triggered[0], f"Stop should not trigger yet at step {step}"

    # Step 6: Price drops to 98 - SHOULD TRIGGER STOP LOSS
    print(f"\n{'='*80}")
    print(f"Step 6: Price drops to ${env.price_array[env.time+1][0]:.2f} - SHOULD TRIGGER STOP LOSS")
    print(f"  Stop loss trigger price: ${env.highest_price_since_buy[0] * (1 - env.trailing_stop_pct):.2f}")
    state, reward, done, _ = env.step(hold_action)
    print(f"  Position after step: {env.stocks[0]:.4f} shares")
    print(f"  Cash after step: ${env.cash:.2f}")
    print(f"  Stop triggered: {env.trailing_stop_triggered[0]}")
    print(f"  Highest price reset: ${env.highest_price_since_buy[0]:.2f}")

    # Verify stop loss was triggered
    assert env.stocks[0] == 0, "Position should be closed by trailing stop loss"
    assert env.trailing_stop_triggered[0], "Trailing stop should be triggered"
    assert env.highest_price_since_buy[0] == 0, "Highest price should be reset"

    # Note: bought at 105, sold at 98 is a loss (as expected with trailing stop)
    # Bought ~9.4 shares at 105 = ~989 spent + 0.1% fee
    # Sold ~9.4 shares at 98 = ~923 received - 0.1% fee
    # Total: started with 1000, ended with ~932 (a controlled loss, which is the point!)
    print(f"\n{'='*80}")
    print("✓ Trailing stop loss test PASSED!")
    print(f"  Final position: {env.stocks[0]} shares (should be 0)")
    print(f"  Final cash: ${env.cash:.2f}")
    print(f"  Stop triggered correctly at ${98:.2f} (stop price was ${110 * 0.9:.2f})")
    print("="*80)

    return True


def test_trailing_stop_disabled():
    """Test that trailing stop can be disabled."""
    print("\n" + "="*80)
    print("Testing Trailing Stop Loss DISABLED")
    print("="*80)

    lookback = 2
    n_timesteps = 8

    # Same price scenario but stop disabled
    prices = np.array([
        [100.0],  # t=0 (lookback)
        [100.0],  # t=1 (lookback)
        [105.0],  # t=2
        [110.0],  # t=3
        [98.0],   # t=4 - price drops but stop disabled
        [95.0],   # t=5
        [90.0],   # t=6
        [85.0],   # t=7
    ])

    tech_array = np.repeat(prices, 3, axis=1)

    env_params = {
        'lookback': lookback,
        'norm_cash': 2**-15,
        'norm_stocks': 2**-10,
        'norm_tech': 2**-18,
        'norm_reward': 2**-13,
        'norm_action': 15000,
        'time_decay_floor': 0.2,
        'min_cash_reserve': 0.0,
        'concentration_penalty': 0.0,
        'trailing_stop_pct': 0.0,  # DISABLED
    }

    config = {'price_array': prices, 'tech_array': tech_array}

    env = CryptoEnvAlpaca(config=config, env_params=env_params, if_log=False)

    print(f"\nTrailing stop disabled: {not env.use_trailing_stop}")

    state = env.reset()

    # Buy
    buy_action = np.array([9.0])
    state, reward, done, _ = env.step(buy_action)
    initial_position = env.stocks[0]

    # Hold through price drop
    hold_action = np.array([0.0])
    for i in range(5):
        state, reward, done, _ = env.step(hold_action)

    # Position should still be held (no automatic stop)
    assert env.stocks[0] == initial_position, "Position should not be closed when stop is disabled"
    assert not env.trailing_stop_triggered[0], "Stop should never trigger when disabled"

    print("✓ Trailing stop disabled test PASSED!")
    print(f"  Position maintained: {env.stocks[0]:.4f} shares")
    print("="*80)

    return True


if __name__ == "__main__":
    try:
        test_trailing_stop_loss()
        test_trailing_stop_disabled()
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
