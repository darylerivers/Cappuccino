#!/usr/bin/env python3
"""
Comprehensive test for fee calculation fixes.

Tests:
1. Equal-weight benchmark accounts for initial purchase fees
2. Trade P&L calculations include fees
3. Win-rate is calculated correctly with fee-adjusted P&L
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import csv

from environment_Alpaca import CryptoEnvAlpaca
from trade_history_analyzer import TradeHistoryAnalyzer


def test_equal_weight_benchmark_fees():
    """Test that equal-weight benchmark accounts for initial purchase fees."""
    print("="*80)
    print("TEST 1: Equal-Weight Benchmark Fee Calculation")
    print("="*80)

    # Create simple test environment
    n_timesteps = 50
    n_tickers = 2
    n_features_per_ticker = 11

    # Fixed prices for predictable results
    prices = np.ones((n_timesteps, n_tickers), dtype=np.float32) * 100.0  # All prices = $100
    tech_array = np.random.randn(n_timesteps, n_tickers * n_features_per_ticker).astype(np.float32) * 0.1

    config = {
        'price_array': prices,
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

    initial_capital = 10000.0
    buy_fee = 0.0025  # 0.25%

    env = CryptoEnvAlpaca(
        config,
        env_params,
        initial_capital=initial_capital,
        buy_cost_pct=buy_fee,
        sell_cost_pct=buy_fee,
        if_log=False,
    )

    # Calculate expected equal-weight holdings
    # For each asset: shares = (initial_capital / n_tickers) / (price * (1 + fee))
    per_asset_investment = initial_capital / n_tickers
    expected_shares_per_asset = per_asset_investment / (100.0 * (1 + buy_fee))
    expected_total_value = expected_shares_per_asset * 100.0 * n_tickers

    # Check actual equal-weight allocation
    actual_shares_0 = env.equal_weight_stock[0]
    actual_shares_1 = env.equal_weight_stock[1]
    actual_total_value = env.total_asset_eqw

    print(f"\nInitial Capital: ${initial_capital:,.2f}")
    print(f"Buy Fee: {buy_fee*100:.2f}%")
    print(f"Number of Assets: {n_tickers}")
    print(f"Price per Asset: $100.00")
    print(f"\nExpected equal-weight allocation:")
    print(f"  Shares per asset: {expected_shares_per_asset:.6f}")
    print(f"  Total portfolio value: ${expected_total_value:,.2f}")
    print(f"\nActual equal-weight allocation:")
    print(f"  Shares of asset 0: {actual_shares_0:.6f}")
    print(f"  Shares of asset 1: {actual_shares_1:.6f}")
    print(f"  Total portfolio value: ${actual_total_value:,.2f}")

    # Calculate fee paid by equal-weight
    print(f"\nEqual-weight initial fee: ${env.eqw_initial_fee:,.2f}")
    expected_fee = expected_total_value * buy_fee
    print(f"Expected fee: ${expected_fee:,.2f}")

    # Verify
    tolerance = 0.01
    assert abs(actual_shares_0 - expected_shares_per_asset) < tolerance, \
        f"Asset 0 shares mismatch: {actual_shares_0} vs {expected_shares_per_asset}"
    assert abs(actual_shares_1 - expected_shares_per_asset) < tolerance, \
        f"Asset 1 shares mismatch: {actual_shares_1} vs {expected_shares_per_asset}"
    assert abs(env.eqw_initial_fee - expected_fee) < tolerance, \
        f"Fee mismatch: {env.eqw_initial_fee} vs {expected_fee}"

    # The equal-weight should have slightly less value than initial capital due to fees
    value_loss_to_fees = initial_capital - actual_total_value
    print(f"\nValue lost to fees: ${value_loss_to_fees:,.2f}")
    print(f"Fee percentage of initial: {(value_loss_to_fees / initial_capital) * 100:.3f}%")

    # Since we're buying with fees, the actual total value should equal the cost after fees
    # Cost = shares * price * (1 + fee)
    # Value = shares * price
    # So: Value = Cost / (1 + fee)
    expected_value_loss = initial_capital * buy_fee / (1 + buy_fee)
    print(f"Expected value loss: ${expected_value_loss:,.2f}")

    print("\n✅ Equal-weight benchmark fee calculation PASSED!")
    return True


def test_trade_pnl_with_fees():
    """Test that trade P&L calculations include fees."""
    print("\n" + "="*80)
    print("TEST 2: Trade P&L with Fees")
    print("="*80)

    # Create a simple trading scenario
    # Buy at $100, sell at $105 (5% gross gain)
    # With 0.25% buy + 0.25% sell fees = 0.5% round-trip
    # Expected net gain: ~4.5%

    quantity = 10.0
    entry_price = 100.0
    exit_price = 105.0
    buy_fee = 0.0025
    sell_fee = 0.0025

    # Manual calculation
    entry_value = quantity * entry_price  # $1000
    exit_value = quantity * exit_price    # $1050
    entry_fee = entry_value * buy_fee     # $2.50
    exit_fee = exit_value * sell_fee      # $2.625

    pnl_gross = exit_value - entry_value  # $50
    pnl_net = (exit_value - exit_fee) - (entry_value + entry_fee)  # $50 - $5.125 = $44.875
    cost_basis = entry_value + entry_fee  # $1002.50
    pnl_pct = (pnl_net / cost_basis) * 100  # 4.476%

    print(f"\nTrade Details:")
    print(f"  Quantity: {quantity}")
    print(f"  Entry Price: ${entry_price:.2f}")
    print(f"  Exit Price: ${exit_price:.2f}")
    print(f"  Buy Fee: {buy_fee*100:.2f}%")
    print(f"  Sell Fee: {sell_fee*100:.2f}%")
    print(f"\nCalculated P&L:")
    print(f"  Entry Value: ${entry_value:.2f}")
    print(f"  Exit Value: ${exit_value:.2f}")
    print(f"  Entry Fee: ${entry_fee:.2f}")
    print(f"  Exit Fee: ${exit_fee:.2f}")
    print(f"  Gross P&L (before fees): ${pnl_gross:.2f}")
    print(f"  Net P&L (after fees): ${pnl_net:.2f}")
    print(f"  Cost Basis: ${cost_basis:.2f}")
    print(f"  Return %: {pnl_pct:.3f}%")

    # Create a mock CSV to test TradeHistoryAnalyzer
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_session.csv"

        # Create a CSV with a simple trade
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cash', 'total_asset', 'reward',
                           'holding_BTC', 'price_BTC', 'action_BTC'])

            # Entry: buy 10 BTC at $100
            ts1 = datetime.now()
            writer.writerow([ts1.isoformat(), 9000, 10000, 0.0,
                           10.0, 100.0, 10.0])

            # Hold for a bit
            ts2 = ts1 + timedelta(hours=1)
            writer.writerow([ts2.isoformat(), 9000, 10050, 0.0,
                           10.0, 105.0, 0.0])

            # Exit: sell 10 BTC at $105
            ts3 = ts1 + timedelta(hours=2)
            writer.writerow([ts3.isoformat(), 10045, 10045, 0.0,
                           0.0, 105.0, -10.0])

        # Analyze trades
        analyzer = TradeHistoryAnalyzer([csv_path], buy_fee_pct=buy_fee, sell_fee_pct=sell_fee)
        trades = analyzer.extract_completed_trades()

        print(f"\n✅ Found {len(trades)} completed trade(s)")

        if len(trades) > 0:
            trade = trades[0]
            print(f"\nAnalyzer Results:")
            print(f"  Entry Value: ${trade.entry_value:.2f}")
            print(f"  Exit Value: ${trade.exit_value:.2f}")
            print(f"  Entry Fee: ${trade.entry_fee:.2f}")
            print(f"  Exit Fee: ${trade.exit_fee:.2f}")
            print(f"  Gross P&L: ${trade.pnl_gross:.2f}")
            print(f"  Net P&L: ${trade.pnl:.2f}")
            print(f"  Return %: {trade.pnl_pct:.3f}%")

            # Verify
            tolerance = 0.01
            assert abs(trade.pnl - pnl_net) < tolerance, \
                f"Net P&L mismatch: {trade.pnl} vs {pnl_net}"
            assert abs(trade.pnl_pct - pnl_pct) < tolerance, \
                f"Return % mismatch: {trade.pnl_pct} vs {pnl_pct}"

            print("\n✅ Trade P&L with fees calculation PASSED!")
        else:
            print("\n❌ No trades extracted!")
            return False

    return True


def test_win_rate_accuracy():
    """Test that win-rate correctly identifies fee-losing trades."""
    print("\n" + "="*80)
    print("TEST 3: Win-Rate Accuracy with Fees")
    print("="*80)

    # Create scenarios:
    # Trade 1: Buy at $100, sell at $105 → 5% gross, ~4.5% net → WIN
    # Trade 2: Buy at $100, sell at $100.4 → 0.4% gross, ~-0.1% net → LOSS (fee exceeds gain!)
    # Trade 3: Buy at $100, sell at $95 → -5% gross, ~-5.5% net → LOSS

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_session.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cash', 'total_asset', 'reward',
                           'holding_BTC', 'price_BTC', 'action_BTC'])

            base_time = datetime.now()

            # Trade 1: WIN (5% gain >> 0.5% fees)
            writer.writerow([base_time.isoformat(), 9000, 10000, 0, 10.0, 100.0, 10.0])
            writer.writerow([(base_time + timedelta(hours=1)).isoformat(), 10045, 10045, 0, 0.0, 105.0, -10.0])

            # Trade 2: LOSS (0.4% gain < 0.5% fees)
            writer.writerow([(base_time + timedelta(hours=2)).isoformat(), 9000, 10040, 0, 10.0, 100.0, 10.0])
            writer.writerow([(base_time + timedelta(hours=3)).isoformat(), 9995, 9995, 0, 0.0, 100.4, -10.0])

            # Trade 3: LOSS (obvious loss)
            writer.writerow([(base_time + timedelta(hours=4)).isoformat(), 9000, 10000, 0, 10.0, 100.0, 10.0])
            writer.writerow([(base_time + timedelta(hours=5)).isoformat(), 9445, 9445, 0, 0.0, 95.0, -10.0])

        analyzer = TradeHistoryAnalyzer([csv_path], buy_fee_pct=0.0025, sell_fee_pct=0.0025)
        trades = analyzer.extract_completed_trades()

        print(f"\n✅ Extracted {len(trades)} trades")
        print(f"\nTrade Details:")

        winning_trades = []
        losing_trades = []

        for i, trade in enumerate(trades, 1):
            is_win = trade.pnl > 0
            if is_win:
                winning_trades.append(trade)
            else:
                losing_trades.append(trade)

            print(f"\nTrade {i}:")
            print(f"  Entry: ${trade.entry_price:.2f} → Exit: ${trade.exit_price:.2f}")
            print(f"  Gross P&L: ${trade.pnl_gross:.2f}")
            print(f"  Fees: ${trade.entry_fee + trade.exit_fee:.2f}")
            print(f"  Net P&L: ${trade.pnl:.2f}")
            print(f"  Result: {'✅ WIN' if is_win else '❌ LOSS'}")

        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        print(f"\n📊 Win Rate: {win_rate:.1f}% ({len(winning_trades)}/{len(trades)} wins)")

        # Verify expectations
        assert len(trades) == 3, f"Expected 3 trades, got {len(trades)}"
        assert len(winning_trades) == 1, f"Expected 1 winning trade, got {len(winning_trades)}"
        assert len(losing_trades) == 2, f"Expected 2 losing trades, got {len(losing_trades)}"
        assert abs(win_rate - 33.3) < 1.0, f"Expected ~33% win rate, got {win_rate:.1f}%"

        print("\n✅ Win-rate accuracy test PASSED!")
        print("   → Trade 1: Big win (5% gain)")
        print("   → Trade 2: Fee-killed (0.4% gain eaten by 0.5% fees)")
        print("   → Trade 3: Clear loss (-5%)")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE FEE CALCULATION TESTS")
    print("="*80)

    results = []

    try:
        results.append(("Equal-Weight Benchmark Fees", test_equal_weight_benchmark_fees()))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Equal-Weight Benchmark Fees", False))

    try:
        results.append(("Trade P&L with Fees", test_trade_pnl_with_fees()))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Trade P&L with Fees", False))

    try:
        results.append(("Win-Rate Accuracy", test_win_rate_accuracy()))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Win-Rate Accuracy", False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:<40s} {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
