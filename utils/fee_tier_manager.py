"""Fee Tier Manager for Coinbase Fee Tiers

This module implements dynamic fee tier management based on 30-day trading volume.
Simulates realistic Coinbase fee progression as trading volume accumulates.

Key features:
- Tracks 30-day rolling window of trading volume
- Progressive fee interpolation (smooth transition between tiers)
- Static fee jumps (discrete tier progression)
- Realistic simulation of new trader fee experience (0.6% → 0.25%)

Example:
    >>> manager = FeeTierManager(interval='1h', mode='progressive')
    >>> # Simulate trades
    >>> for trade_volume in [5000, 10000, 15000]:
    ...     cumulative_volume = manager.update_volume(trade_volume, current_step)
    ...     maker_fee, taker_fee = manager.get_current_fees()
    ...     print(f"Volume: ${cumulative_volume:,} → Maker: {maker_fee:.3%}")
    Volume: $5,000 → Maker: 0.560%
    Volume: $15,000 → Maker: 0.480%
    Volume: $30,000 → Maker: 0.250%
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Optional
from config_two_phase import FEE_TIER


class FeeTierManager:
    """
    Manages dynamic fee tiers based on 30-day trading volume.

    Tracks cumulative trading volume over a rolling 30-day window and
    calculates applicable fees based on Coinbase's tier structure.
    """

    def __init__(
        self,
        interval: str = '1h',
        mode: str = 'progressive',
        initial_volume: float = 0.0
    ):
        """
        Initialize fee tier manager.

        Args:
            interval: Data interval (e.g., '5m', '1h', '4h') for calculating window size
            mode: Fee calculation mode
                  - 'progressive': Interpolate smoothly between tiers
                  - 'static': Jump discretely to new tier
            initial_volume: Starting 30-day volume (default: 0 for new trader)

        Raises:
            ValueError: If mode is invalid
        """
        if mode not in [FEE_TIER.FEE_MODE_PROGRESSIVE, FEE_TIER.FEE_MODE_STATIC]:
            raise ValueError(
                f"Invalid mode '{mode}'. "
                f"Valid options: ['{FEE_TIER.FEE_MODE_PROGRESSIVE}', '{FEE_TIER.FEE_MODE_STATIC}']"
            )

        self.interval = interval
        self.mode = mode

        # Calculate 30-day window in candles
        self.window_candles = FEE_TIER.get_window_candles(interval)

        # Volume history: list of (timestep, volume) tuples
        self.volume_history: deque = deque()

        # Cumulative volume tracker
        self.cumulative_volume = initial_volume

        # Fee tiers (from config)
        self.fee_tiers = list(FEE_TIER.FEE_TIERS)

        # Current tier index
        self.current_tier_index = 0

        # Statistics
        self.total_trades = 0
        self.tier_progression_count = 0

    def update_volume(self, trade_volume: float, current_timestep: int) -> float:
        """
        Update 30-day rolling volume with new trade.

        Args:
            trade_volume: Dollar value of trade
            current_timestep: Current timestep in environment

        Returns:
            Updated cumulative 30-day volume
        """
        # Add new trade to history
        self.volume_history.append((current_timestep, trade_volume))
        self.total_trades += 1

        # Remove trades older than 30 days
        cutoff_timestep = current_timestep - self.window_candles

        while self.volume_history and self.volume_history[0][0] <= cutoff_timestep:
            self.volume_history.popleft()

        # Calculate cumulative volume
        self.cumulative_volume = sum(vol for _, vol in self.volume_history)

        # Update current tier
        old_tier = self.current_tier_index
        self.current_tier_index = self._get_tier_index(self.cumulative_volume)

        if self.current_tier_index > old_tier:
            self.tier_progression_count += 1

        return self.cumulative_volume

    def get_current_fees(self, volume: Optional[float] = None) -> Tuple[float, float]:
        """
        Get current applicable fees based on volume.

        Args:
            volume: 30-day volume (if None, uses internal cumulative_volume)

        Returns:
            Tuple of (maker_fee, taker_fee) as decimals (e.g., 0.006 for 0.6%)
        """
        if volume is None:
            volume = self.cumulative_volume

        if self.mode == FEE_TIER.FEE_MODE_PROGRESSIVE:
            return self._get_progressive_fees(volume)
        else:  # static mode
            return self._get_static_fees(volume)

    def _get_tier_index(self, volume: float) -> int:
        """
        Get tier index for given volume.

        Args:
            volume: 30-day trading volume

        Returns:
            Tier index (0 = lowest tier)
        """
        for i in range(len(self.fee_tiers) - 1, -1, -1):
            if volume >= self.fee_tiers[i]['volume_threshold']:
                return i
        return 0

    def _get_static_fees(self, volume: float) -> Tuple[float, float]:
        """
        Get fees using static tier jumps (discrete transitions).

        Args:
            volume: 30-day trading volume

        Returns:
            (maker_fee, taker_fee)
        """
        tier_index = self._get_tier_index(volume)
        tier = self.fee_tiers[tier_index]
        return tier['maker'], tier['taker']

    def _get_progressive_fees(self, volume: float) -> Tuple[float, float]:
        """
        Get fees using progressive interpolation (smooth transitions).

        Interpolates linearly between tiers based on progress toward next tier.

        Args:
            volume: 30-day trading volume

        Returns:
            (maker_fee, taker_fee)
        """
        # Find current and next tier
        current_tier_index = self._get_tier_index(volume)
        current_tier = self.fee_tiers[current_tier_index]

        # If at highest tier, return those fees
        if current_tier_index >= len(self.fee_tiers) - 1:
            return current_tier['maker'], current_tier['taker']

        # Calculate progress to next tier
        next_tier = self.fee_tiers[current_tier_index + 1]
        volume_range = next_tier['volume_threshold'] - current_tier['volume_threshold']
        volume_progress = volume - current_tier['volume_threshold']

        if volume_range <= 0:
            # Edge case: identical thresholds
            return current_tier['maker'], current_tier['taker']

        # Progress percentage (0.0 to 1.0)
        progress_pct = min(1.0, volume_progress / volume_range)

        # Linear interpolation
        maker_fee = current_tier['maker'] + progress_pct * (next_tier['maker'] - current_tier['maker'])
        taker_fee = current_tier['taker'] + progress_pct * (next_tier['taker'] - current_tier['taker'])

        return maker_fee, taker_fee

    def get_next_tier_info(self) -> Optional[Dict]:
        """
        Get information about the next fee tier.

        Returns:
            Dictionary with next tier info, or None if at highest tier
        """
        if self.current_tier_index >= len(self.fee_tiers) - 1:
            return None

        next_tier = self.fee_tiers[self.current_tier_index + 1]
        volume_needed = next_tier['volume_threshold'] - self.cumulative_volume

        return {
            'next_tier_index': self.current_tier_index + 1,
            'volume_threshold': next_tier['volume_threshold'],
            'volume_needed': max(0, volume_needed),
            'maker_fee': next_tier['maker'],
            'taker_fee': next_tier['taker'],
        }

    def get_tier_info(self) -> Dict:
        """
        Get current tier information.

        Returns:
            Dictionary with current tier details
        """
        current_tier = self.fee_tiers[self.current_tier_index]
        maker_fee, taker_fee = self.get_current_fees()

        return {
            'tier_index': self.current_tier_index,
            'volume_threshold': current_tier['volume_threshold'],
            'cumulative_volume': self.cumulative_volume,
            'base_maker_fee': current_tier['maker'],
            'base_taker_fee': current_tier['taker'],
            'current_maker_fee': maker_fee,
            'current_taker_fee': taker_fee,
            'mode': self.mode,
            'total_trades': self.total_trades,
            'tier_progression_count': self.tier_progression_count,
        }

    def reset(self):
        """Reset the fee tier manager to initial state."""
        self.volume_history.clear()
        self.cumulative_volume = 0.0
        self.current_tier_index = 0
        self.total_trades = 0
        self.tier_progression_count = 0

    def __repr__(self) -> str:
        """String representation."""
        maker, taker = self.get_current_fees()
        return (
            f"FeeTierManager(interval='{self.interval}', mode='{self.mode}', "
            f"volume=${self.cumulative_volume:,.0f}, "
            f"tier={self.current_tier_index}, "
            f"maker={maker:.3%}, taker={taker:.3%})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        maker, taker = self.get_current_fees()
        tier = self.fee_tiers[self.current_tier_index]
        return (
            f"Fee Tier {self.current_tier_index}: "
            f"${tier['volume_threshold']:,}+ volume → "
            f"Maker: {maker:.3%}, Taker: {taker:.3%} "
            f"(Current volume: ${self.cumulative_volume:,.0f})"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_fee_manager(
    interval: str = '1h',
    mode: str = 'progressive',
    initial_volume: float = 0.0
) -> FeeTierManager:
    """
    Factory function to create fee tier managers.

    Args:
        interval: Data interval
        mode: 'progressive' or 'static'
        initial_volume: Starting 30-day volume

    Returns:
        FeeTierManager instance
    """
    return FeeTierManager(interval, mode, initial_volume)


# =============================================================================
# Testing/Debugging
# =============================================================================

if __name__ == '__main__':
    print("Fee Tier Manager Testing")
    print("=" * 60)

    # Test progressive mode
    print("\n1. Progressive Mode Test")
    print("-" * 60)

    manager_prog = create_fee_manager(interval='1h', mode='progressive')
    print(f"Initial: {manager_prog}")

    # Simulate trading sequence
    trades = [
        (10, 2000),   # Step 10: $2K trade
        (20, 3000),   # Step 20: $3K trade
        (30, 5000),   # Step 30: $5K trade (total $10K → tier 2)
        (40, 8000),   # Step 40: $8K trade
        (50, 10000),  # Step 50: $10K trade (total $28K → approaching tier 3)
        (60, 5000),   # Step 60: $5K trade (total $33K → past tier 3)
    ]

    print("\nTrade sequence:")
    for step, volume in trades:
        cumulative = manager_prog.update_volume(volume, step)
        maker, taker = manager_prog.get_current_fees()
        print(f"  Step {step:3d}: +${volume:>6,} → Total ${cumulative:>7,.0f} "
              f"(Tier {manager_prog.current_tier_index}, Maker: {maker:.3%})")

    # Test static mode
    print("\n2. Static Mode Test")
    print("-" * 60)

    manager_static = create_fee_manager(interval='1h', mode='static')
    print(f"Initial: {manager_static}")

    print("\nTrade sequence:")
    for step, volume in trades:
        cumulative = manager_static.update_volume(volume, step)
        maker, taker = manager_static.get_current_fees()
        print(f"  Step {step:3d}: +${volume:>6,} → Total ${cumulative:>7,.0f} "
              f"(Tier {manager_static.current_tier_index}, Maker: {maker:.3%})")

    # Test 30-day rolling window
    print("\n3. Rolling Window Test (30-day expiration)")
    print("-" * 60)

    manager_window = create_fee_manager(interval='1h', mode='progressive')

    # Add trades at different times
    print("\nAdding trades over time:")
    manager_window.update_volume(10000, 0)
    print(f"  Step 0: ${manager_window.cumulative_volume:,.0f}")

    manager_window.update_volume(10000, 100)
    print(f"  Step 100: ${manager_window.cumulative_volume:,.0f}")

    manager_window.update_volume(10000, 500)
    print(f"  Step 500: ${manager_window.cumulative_volume:,.0f} (all 3 trades within 30 days)")

    # Advance past 30-day window (720 candles for 1h interval)
    manager_window.update_volume(5000, 750)
    print(f"  Step 750: ${manager_window.cumulative_volume:,.0f} (first trade expired)")

    # Test next tier info
    print("\n4. Next Tier Information")
    print("-" * 60)

    manager_next = create_fee_manager(interval='1h', mode='progressive')
    manager_next.update_volume(5000, 10)

    next_tier = manager_next.get_next_tier_info()
    if next_tier:
        print(f"Current volume: ${manager_next.cumulative_volume:,.0f}")
        print(f"Next tier at: ${next_tier['volume_threshold']:,.0f}")
        print(f"Volume needed: ${next_tier['volume_needed']:,.0f}")
        print(f"Next tier fees: Maker {next_tier['maker_fee']:.3%}, Taker {next_tier['taker_fee']:.3%}")

    # Test tier info
    print("\n5. Detailed Tier Information")
    print("-" * 60)

    info = manager_prog.get_tier_info()
    for key, value in info.items():
        if isinstance(value, float) and value < 1.0:
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, (int, float)):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")

    # Test comparison: Progressive vs Static
    print("\n6. Progressive vs Static Comparison")
    print("-" * 60)

    volumes = [0, 5000, 10000, 15000, 20000, 25000, 30000, 50000, 75000, 100000]
    print(f"{'Volume':<12} {'Progressive Maker':<18} {'Static Maker':<15}")
    print("-" * 60)

    for vol in volumes:
        prog_maker, _ = create_fee_manager(mode='progressive').get_current_fees(vol)
        static_maker, _ = create_fee_manager(mode='static').get_current_fees(vol)
        print(f"${vol:<10,}  {prog_maker:>6.3%}            {static_maker:>6.3%}")

    print("\n✓ All tests completed successfully")
