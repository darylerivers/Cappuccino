"""Time-Frame Constraint for Trading Environment

This module implements time-frame constraints that force models to complete
trades within a specified time horizon (e.g., 3 days, 5 days, 7 days).

Key behavior:
- Models must complete all trades within the time-frame
- If deadline is reached, positions are conceptually liquidated at market price
- Rewards are capped at the time-frame boundary
- No credit for holding positions longer than the time-frame

Example:
    >>> constraint = TimeFrameConstraint('5d', '1h')
    >>> print(f"Deadline: {constraint.max_candles} candles")
    Deadline: 120 candles

    >>> # In trading loop
    >>> if constraint.is_deadline_reached(current_step):
    ...     # Force liquidation
    ...     portfolio_value = cash + sum(stocks * prices)
    ...     reward = constraint.calculate_final_reward(portfolio_value, initial_value)
    ...     done = True
"""

import numpy as np
from typing import Optional
from config_two_phase import TIMEFRAME


class TimeFrameConstraint:
    """
    Enforces time-frame constraints on trading episodes.

    Ensures models don't hold positions indefinitely by imposing a deadline.
    When the deadline is reached, all positions are conceptually liquidated
    and the episode ends.
    """

    def __init__(self, timeframe: str, interval: str, lookback: int = 5):
        """
        Initialize time-frame constraint.

        Args:
            timeframe: Trading horizon (e.g., '3d', '5d', '7d', '10d', '14d')
            interval: Data interval (e.g., '5m', '15m', '30m', '1h', '4h')
            lookback: Number of candles needed for observation (default: 5)

        Raises:
            ValueError: If timeframe or interval is invalid
        """
        if timeframe not in TIMEFRAME.TIMEFRAME_HOURS:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. "
                f"Valid options: {list(TIMEFRAME.TIMEFRAME_HOURS.keys())}"
            )

        if interval not in TIMEFRAME.CANDLES_PER_DAY:
            raise ValueError(
                f"Invalid interval '{interval}'. "
                f"Valid options: {list(TIMEFRAME.CANDLES_PER_DAY.keys())}"
            )

        self.timeframe = timeframe
        self.interval = interval
        self.lookback = lookback

        # Calculate total candles in time-frame
        self.max_candles = TIMEFRAME.get_candles_in_timeframe(timeframe, interval)

        # Timeline offset (needed for lookback window)
        self._timeline_offset = lookback - 1

        # Actual deadline step (accounting for lookback)
        self.deadline_step = self._timeline_offset + self.max_candles

    def is_deadline_reached(self, current_step: int) -> bool:
        """
        Check if current step has reached or exceeded the deadline.

        Args:
            current_step: Current timestep in environment

        Returns:
            True if deadline reached, False otherwise
        """
        return current_step >= self.deadline_step

    def is_within_timeframe(self, current_step: int) -> bool:
        """
        Check if current step is within the allowed time-frame.

        Args:
            current_step: Current timestep in environment

        Returns:
            True if within time-frame, False if deadline passed
        """
        return not self.is_deadline_reached(current_step)

    def get_steps_remaining(self, current_step: int) -> int:
        """
        Get number of steps remaining until deadline.

        Args:
            current_step: Current timestep in environment

        Returns:
            Steps remaining (0 if deadline reached)
        """
        remaining = self.deadline_step - current_step
        return max(0, remaining)

    def get_progress(self, current_step: int) -> float:
        """
        Get progress through time-frame as percentage.

        Args:
            current_step: Current timestep in environment

        Returns:
            Progress from 0.0 (start) to 1.0 (deadline)
        """
        # Adjust for timeline offset
        effective_step = current_step - self._timeline_offset
        progress = effective_step / self.max_candles
        return np.clip(progress, 0.0, 1.0)

    def calculate_forced_liquidation_value(
        self,
        cash: float,
        stocks: np.ndarray,
        current_prices: np.ndarray
    ) -> float:
        """
        Calculate portfolio value if forced to liquidate at current prices.

        Args:
            cash: Current cash balance
            stocks: Holdings in each asset (numpy array)
            current_prices: Current market prices (numpy array)

        Returns:
            Total portfolio value (cash + liquidated positions)
        """
        liquidation_value = cash + np.sum(stocks * current_prices)
        return float(liquidation_value)

    def calculate_final_reward(
        self,
        final_portfolio_value: float,
        initial_portfolio_value: float,
        benchmark_value: float,
        norm_reward: float = 1.0
    ) -> float:
        """
        Calculate final reward at time-frame boundary.

        Reward is based on portfolio gain vs benchmark (HODL), capped at deadline.

        Args:
            final_portfolio_value: Portfolio value at deadline
            initial_portfolio_value: Starting portfolio value
            benchmark_value: Benchmark (equal-weight HODL) value at deadline
            norm_reward: Reward normalization factor

        Returns:
            Final reward value
        """
        # Portfolio return
        portfolio_gain = final_portfolio_value - initial_portfolio_value

        # Benchmark return
        benchmark_gain = benchmark_value - initial_portfolio_value

        # Reward = outperformance vs benchmark
        reward = (portfolio_gain - benchmark_gain) * norm_reward

        return float(reward)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TimeFrameConstraint(timeframe='{self.timeframe}', "
            f"interval='{self.interval}', "
            f"max_candles={self.max_candles}, "
            f"deadline_step={self.deadline_step})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        return (
            f"TimeFrame: {self.timeframe} @ {self.interval} "
            f"({self.max_candles} candles, deadline at step {self.deadline_step})"
        )

    def get_info_dict(self, current_step: int) -> dict:
        """
        Get constraint information as dictionary (for environment info).

        Args:
            current_step: Current timestep

        Returns:
            Dictionary with constraint status
        """
        return {
            'timeframe': self.timeframe,
            'interval': self.interval,
            'max_candles': self.max_candles,
            'deadline_step': self.deadline_step,
            'current_step': current_step,
            'steps_remaining': self.get_steps_remaining(current_step),
            'progress': self.get_progress(current_step),
            'deadline_reached': self.is_deadline_reached(current_step),
        }


# =============================================================================
# Constraint Factory
# =============================================================================

def create_constraint(
    timeframe: str,
    interval: str,
    lookback: int = 5
) -> TimeFrameConstraint:
    """
    Factory function to create time-frame constraints.

    Args:
        timeframe: Trading horizon (e.g., '3d', '5d', '7d')
        interval: Data interval (e.g., '5m', '1h', '4h')
        lookback: Lookback window size

    Returns:
        TimeFrameConstraint instance

    Example:
        >>> constraint = create_constraint('5d', '1h')
        >>> print(constraint)
        TimeFrame: 5d @ 1h (120 candles, deadline at step 124)
    """
    return TimeFrameConstraint(timeframe, interval, lookback)


# =============================================================================
# Testing/Debugging
# =============================================================================

if __name__ == '__main__':
    print("Time-Frame Constraint Testing")
    print("=" * 60)

    # Test various combinations
    test_cases = [
        ('3d', '1h'),
        ('5d', '1h'),
        ('7d', '4h'),
        ('10d', '15m'),
        ('14d', '5m'),
    ]

    for timeframe, interval in test_cases:
        constraint = create_constraint(timeframe, interval, lookback=5)
        print(f"\n{constraint}")
        print(f"  Max candles: {constraint.max_candles}")
        print(f"  Deadline step: {constraint.deadline_step}")

        # Test progress at various points
        test_steps = [
            constraint._timeline_offset,  # Start
            constraint._timeline_offset + constraint.max_candles // 2,  # Midpoint
            constraint.deadline_step - 10,  # Near deadline
            constraint.deadline_step,  # At deadline
            constraint.deadline_step + 10,  # Past deadline
        ]

        print(f"  Progress:")
        for step in test_steps:
            progress = constraint.get_progress(step)
            remaining = constraint.get_steps_remaining(step)
            deadline_status = "✓ DEADLINE" if constraint.is_deadline_reached(step) else "  ongoing"
            print(f"    Step {step:4d}: {progress:5.1%} progress, {remaining:3d} remaining {deadline_status}")

    # Test forced liquidation calculation
    print("\n" + "=" * 60)
    print("Forced Liquidation Test")
    print("=" * 60)

    constraint = create_constraint('5d', '1h')
    cash = 500.0
    stocks = np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.08, 0.12])  # 7 assets
    prices = np.array([45000, 2500, 180, 25, 3200, 150, 12])

    liquidation_value = constraint.calculate_forced_liquidation_value(cash, stocks, prices)
    print(f"Cash: ${cash:.2f}")
    print(f"Stocks value: ${np.sum(stocks * prices):.2f}")
    print(f"Total liquidation value: ${liquidation_value:.2f}")

    # Test reward calculation
    initial_value = 1000.0
    benchmark_value = 1050.0
    reward = constraint.calculate_final_reward(
        liquidation_value, initial_value, benchmark_value, norm_reward=0.001
    )
    print(f"\nReward calculation:")
    print(f"  Initial: ${initial_value:.2f}")
    print(f"  Final: ${liquidation_value:.2f}")
    print(f"  Benchmark: ${benchmark_value:.2f}")
    print(f"  Reward: {reward:.6f}")

    # Test info dict
    print("\n" + "=" * 60)
    print("Info Dictionary Test")
    print("=" * 60)
    info = constraint.get_info_dict(current_step=100)
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n✓ All tests completed successfully")
