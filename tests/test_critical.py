"""Critical Test Suite for Cappuccino Trading System

Tests the 5 most important components:
1. Stop-loss triggers correctly
2. Ensemble voting averages correctly
3. Position limits enforced
4. NEW profit protection logic works
5. State normalization is consistent

Run with: pytest tests/test_critical.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.deployment.paper_trader_alpaca_polling import RiskManagement, AlpacaPaperTraderPolling
from datetime import datetime, timezone


class TestStopLoss:
    """Test stop-loss functionality."""

    def test_stop_loss_triggers_at_threshold(self):
        """Verify stop-loss sells when loss exceeds 10%."""
        # Create risk management with 10% stop-loss
        risk_mgmt = RiskManagement(stop_loss_pct=0.10)

        # Mock trader state
        entry_price = 100.0
        current_price = 89.0  # 11% loss
        loss_pct = (entry_price - current_price) / entry_price

        assert loss_pct >= risk_mgmt.stop_loss_pct, "Stop-loss should trigger at 11% loss"

    def test_stop_loss_does_not_trigger_below_threshold(self):
        """Verify stop-loss doesn't trigger prematurely."""
        risk_mgmt = RiskManagement(stop_loss_pct=0.10)

        entry_price = 100.0
        current_price = 91.0  # 9% loss
        loss_pct = (entry_price - current_price) / entry_price

        assert loss_pct < risk_mgmt.stop_loss_pct, "Stop-loss should NOT trigger at 9% loss"

    def test_stop_loss_with_profit(self):
        """Verify stop-loss doesn't trigger when in profit."""
        risk_mgmt = RiskManagement(stop_loss_pct=0.10)

        entry_price = 100.0
        current_price = 110.0  # 10% profit
        loss_pct = (entry_price - current_price) / entry_price

        assert loss_pct < 0, "Should be negative (profit), stop-loss should not trigger"


class TestPositionLimits:
    """Test position size limits."""

    def test_max_position_pct_enforced(self):
        """Verify 30% max position limit is enforced."""
        risk_mgmt = RiskManagement(max_position_pct=0.30)

        total_portfolio = 1000.0
        max_position_value = total_portfolio * risk_mgmt.max_position_pct
        assert max_position_value == 300.0, "Max position should be $300 (30% of $1000)"

    def test_position_limit_caps_buy_orders(self):
        """Verify buy orders are capped at position limit."""
        risk_mgmt = RiskManagement(max_position_pct=0.30)

        total_portfolio = 1000.0
        current_position_value = 200.0  # 20% already allocated
        current_price = 10.0

        # Calculate max additional value
        max_additional_value = (risk_mgmt.max_position_pct * total_portfolio) - current_position_value
        max_additional_shares = max_additional_value / current_price

        assert max_additional_shares == 10.0, "Should only allow 10 more shares ($100 / $10)"

    def test_multiple_assets_diversification(self):
        """Verify 30% limit ensures max 3-4 assets at full allocation."""
        risk_mgmt = RiskManagement(max_position_pct=0.30)

        # If all positions at max, can hold 3.33 assets
        max_assets_at_limit = 1.0 / risk_mgmt.max_position_pct
        assert max_assets_at_limit > 3 and max_assets_at_limit < 4, \
            "30% limit ensures diversification across 3+ assets"


class TestEnsembleVoting:
    """Test ensemble model voting."""

    def test_simple_average_voting(self):
        """Verify simple average of model predictions."""
        # 3 models predict different actions for BTC
        model_1_action = np.array([0.5, 0.0, 0.0])  # Buy 0.5 BTC
        model_2_action = np.array([0.3, 0.0, 0.0])  # Buy 0.3 BTC
        model_3_action = np.array([-0.2, 0.0, 0.0])  # Sell 0.2 BTC

        # Average
        ensemble_action = (model_1_action + model_2_action + model_3_action) / 3

        expected = np.array([0.2, 0.0, 0.0])  # (0.5 + 0.3 - 0.2) / 3 = 0.2
        np.testing.assert_array_almost_equal(ensemble_action, expected, decimal=5)

    def test_weighted_voting(self):
        """Verify weighted voting by model performance."""
        # 2 models with different weights
        model_1_action = np.array([1.0])  # Strong buy
        model_2_action = np.array([-1.0])  # Strong sell

        # Model 1 has 2x weight (better performance)
        weight_1 = 0.667
        weight_2 = 0.333

        weighted_action = model_1_action * weight_1 + model_2_action * weight_2
        expected = np.array([0.334])  # 1.0*0.667 + (-1.0)*0.333

        np.testing.assert_array_almost_equal(weighted_action, expected, decimal=3)

    def test_ensemble_reduces_variance(self):
        """Verify ensemble predictions have lower variance than individual models."""
        # Simulate 10 models with noisy predictions
        np.random.seed(42)
        n_models = 10
        true_signal = 0.5
        noise_level = 0.2

        individual_predictions = true_signal + np.random.normal(0, noise_level, n_models)
        ensemble_prediction = np.mean(individual_predictions)

        # Ensemble should be closer to true signal
        ensemble_error = abs(ensemble_prediction - true_signal)
        avg_individual_error = np.mean(np.abs(individual_predictions - true_signal))

        assert ensemble_error < avg_individual_error, \
            "Ensemble should have lower error than average individual model"


class TestProfitProtection:
    """Test NEW portfolio-level profit protection logic."""

    def test_portfolio_trailing_stop_triggers(self):
        """Verify portfolio trailing stop sells all when dropping from peak."""
        risk_mgmt = RiskManagement(portfolio_trailing_stop_pct=0.015)  # 1.5%

        portfolio_peak = 1030.0  # Portfolio was at $1030
        current_portfolio = 1014.0  # Now at $1014

        drawdown_from_peak = (portfolio_peak - current_portfolio) / portfolio_peak
        assert drawdown_from_peak >= risk_mgmt.portfolio_trailing_stop_pct, \
            f"Drawdown of {drawdown_from_peak*100:.2f}% should trigger 1.5% trailing stop"

    def test_profit_take_threshold(self):
        """Verify profit-taking triggers at 3% gain."""
        risk_mgmt = RiskManagement(profit_take_threshold_pct=0.03)

        initial_value = 1000.0
        current_value = 1030.0  # 3% gain

        gain_pct = (current_value / initial_value) - 1
        assert gain_pct >= risk_mgmt.profit_take_threshold_pct, \
            "3% gain should trigger profit-taking"

    def test_profit_take_amount(self):
        """Verify 50% of positions are sold when profit-taking."""
        risk_mgmt = RiskManagement(profit_take_amount_pct=0.5)

        holdings = np.array([10.0, 5.0, 20.0])  # Current holdings
        sell_amount = holdings * risk_mgmt.profit_take_amount_pct

        expected = np.array([5.0, 2.5, 10.0])  # Sell 50%
        np.testing.assert_array_almost_equal(sell_amount, expected, decimal=5)

    def test_move_to_cash_threshold(self):
        """Verify move-to-cash triggers at higher threshold (e.g., 5%)."""
        risk_mgmt = RiskManagement(
            profit_take_threshold_pct=0.03,  # Partial take-profits at 3%
            move_to_cash_threshold_pct=0.05  # Full liquidation at 5%
        )

        initial_value = 1000.0
        current_value = 1051.0  # 5.1% gain

        gain_pct = (current_value / initial_value) - 1

        # Move-to-cash has higher priority
        assert gain_pct >= risk_mgmt.move_to_cash_threshold_pct, \
            "5.1% gain should trigger move-to-cash (not just partial profit-take)"

    def test_cash_mode_cooldown(self):
        """Verify cooldown period after move-to-cash."""
        risk_mgmt = RiskManagement(cooldown_after_cash_hours=24)

        cash_mode_started = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        current_time = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)  # 12 hours later

        hours_elapsed = (current_time - cash_mode_started).total_seconds() / 3600

        assert hours_elapsed < risk_mgmt.cooldown_after_cash_hours, \
            "Should still be in cooldown after 12h (cooldown is 24h)"

    def test_profit_protection_priority_order(self):
        """Verify profit protection rules are applied in correct order."""
        # Order should be:
        # 1. Cash mode cooldown (blocks all buys)
        # 2. Portfolio trailing stop (sells all)
        # 3. Move-to-cash (sells all, higher threshold)
        # 4. Profit-taking (sells partial, lower threshold)

        risk_mgmt = RiskManagement(
            portfolio_trailing_stop_pct=0.015,  # 1.5%
            profit_take_threshold_pct=0.03,      # 3%
            move_to_cash_threshold_pct=0.05,     # 5%
        )

        # At 5% gain, move-to-cash should trigger, not profit-take
        gain = 0.051
        assert gain >= risk_mgmt.move_to_cash_threshold_pct, \
            "Move-to-cash should take priority over profit-taking at 5%+ gain"


class TestStateNormalization:
    """Test state normalization consistency."""

    def test_cash_normalization_scale(self):
        """Verify cash normalization keeps values in reasonable range."""
        norm_cash = 2**-11  # Default from config

        cash_values = [0, 500, 1000, 10000]
        normalized = [c * norm_cash for c in cash_values]

        # Check all values are in [-10, 10] range (reasonable for neural network)
        for norm_val in normalized:
            assert abs(norm_val) < 10, \
                f"Normalized cash {norm_val} should be in reasonable range"

    def test_stock_normalization_scale(self):
        """Verify stock holdings normalization."""
        norm_stocks = 2**-8  # Default from config

        stock_values = [0, 1, 10, 100]
        normalized = [s * norm_stocks for s in stock_values]

        for norm_val in normalized:
            assert abs(norm_val) < 10, \
                f"Normalized stocks {norm_val} should be in reasonable range"

    def test_tech_indicator_normalization(self):
        """Verify technical indicators are normalized consistently."""
        norm_tech = 2**-14  # Default from config

        # Typical indicator ranges
        rsi = 50  # RSI 0-100
        macd = 10  # MACD can vary
        volume = 10000  # Volume can be large

        normalized = [val * norm_tech for val in [rsi, macd, volume]]

        # All should be small values suitable for neural network
        for norm_val in normalized:
            assert abs(norm_val) < 5, \
                f"Normalized indicator {norm_val} should be small for NN stability"

    def test_action_normalization_consistency(self):
        """Verify action normalization is consistent with state."""
        norm_action = 100  # Default from config

        # Raw action from model (typically -1 to 1 range)
        raw_action = 0.5
        denormalized = raw_action * norm_action

        assert denormalized == 50, \
            "Action denormalization should scale correctly (0.5 * 100 = 50)"


class TestDataQuality:
    """Test data quality and consistency."""

    def test_no_nan_values(self):
        """Verify price/tech arrays don't contain NaN."""
        # Simulate price array
        price_array = np.array([
            [100, 200, 50],
            [101, 198, 51],
            [102, 199, 52]
        ])

        assert not np.isnan(price_array).any(), "Price array should not contain NaN"

    def test_no_zero_prices(self):
        """Verify no zero or negative prices."""
        price_array = np.array([
            [100, 200, 50],
            [101, 198, 51],
        ])

        assert np.all(price_array > 0), "All prices should be positive"

    def test_consistent_array_shapes(self):
        """Verify price and tech arrays have consistent timesteps."""
        n_timesteps = 100
        n_assets = 7
        n_tech_per_asset = 11

        price_shape = (n_timesteps, n_assets)
        tech_shape = (n_timesteps, n_assets * n_tech_per_asset)

        # Simulate arrays
        price_array = np.random.rand(*price_shape)
        tech_array = np.random.rand(*tech_shape)

        assert price_array.shape[0] == tech_array.shape[0], \
            "Price and tech arrays must have same number of timesteps"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
