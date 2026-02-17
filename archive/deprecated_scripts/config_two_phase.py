"""Configuration for Two-Phase Training System

This module contains all configuration for the two-phase training system:
- Phase 1: Time-frame optimization
- Phase 2: Feature-enhanced training with fee tiers

Centralized configuration makes it easy to tune parameters and maintain consistency.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict


# =============================================================================
# Phase 1: Time-Frame Optimization Configuration
# =============================================================================

@dataclass(frozen=True)
class Phase1Config:
    """Phase 1: Time-Frame Optimization configuration."""

    # Search space
    TIME_FRAMES: Tuple[str, ...] = ('3d', '5d', '7d', '10d', '14d')
    INTERVALS: Tuple[str, ...] = ('5m', '15m', '30m', '1h', '4h')
    TRIALS_PER_COMBINATION: int = 20

    # Total combinations
    @property
    def TOTAL_COMBINATIONS(self) -> int:
        return len(self.TIME_FRAMES) * len(self.INTERVALS)

    @property
    def TOTAL_TRIALS(self) -> int:
        return self.TOTAL_COMBINATIONS * self.TRIALS_PER_COMBINATION

    # Fixed hyperparameters for Phase 1
    FIXED_TARGET_STEP: int = 256
    FIXED_BREAK_STEP: int = 50000
    FIXED_WORKER_NUM: int = 12
    FIXED_THREAD_NUM: int = 8

    # Fees (worst-case Coinbase maker)
    BUY_COST_PCT: float = 0.006  # 0.6%
    SELL_COST_PCT: float = 0.006  # 0.6%

    # Simplified hyperparameter ranges
    LEARNING_RATE_MIN: float = 1e-5
    LEARNING_RATE_MAX: float = 1e-4
    BATCH_SIZES: Tuple[int, ...] = (2048, 3072)
    GAMMA_MIN: float = 0.95
    GAMMA_MAX: float = 0.99
    GAMMA_STEP: float = 0.01
    NET_DIM_MIN: int = 1024
    NET_DIM_MAX: int = 1536
    NET_DIM_STEP: int = 128
    LOOKBACK_MIN: int = 3
    LOOKBACK_MAX: int = 5
    TRAILING_STOP_MIN: float = 0.05
    TRAILING_STOP_MAX: float = 0.15
    TRAILING_STOP_STEP: float = 0.05

    # Optuna database
    STUDY_NAME_PREFIX: str = 'phase1'
    DB_PATH: str = 'databases/optuna_phase1.db'

    # Output
    WINNER_FILE: str = 'phase1_winner.json'


# =============================================================================
# Phase 2: Feature-Enhanced Training Configuration
# =============================================================================

@dataclass(frozen=True)
class Phase2Config:
    """Phase 2: Feature-Enhanced Training configuration."""

    # Trials
    TRIALS_PPO: int = 200
    TRIALS_DDQN: int = 200

    @property
    def TOTAL_TRIALS(self) -> int:
        return self.TRIALS_PPO + self.TRIALS_DDQN

    # Rolling mean windows (in days)
    ROLLING_WINDOW_SHORT: int = 7   # 7-day rolling means
    ROLLING_WINDOW_LONG: int = 30   # 30-day rolling means

    # State dimension calculation
    # Base: 1 (cash) + 7 (stocks) + 11 (tech) * 5 (lookback) = 63
    # Phase 2: + 7 (cryptos) * 4 (7d_val, 7d_vol, 30d_val, 30d_vol) = +28
    # Total: 91
    BASE_STATE_DIM: int = 63
    ROLLING_FEATURES_PER_CRYPTO: int = 4
    N_CRYPTOS: int = 7

    @property
    def ENHANCED_STATE_DIM(self) -> int:
        return self.BASE_STATE_DIM + (self.N_CRYPTOS * self.ROLLING_FEATURES_PER_CRYPTO)

    # Network size adjustment (44% larger state → larger networks)
    NET_DIM_MIN: int = 1024
    NET_DIM_MAX: int = 2560
    NET_DIM_STEP: int = 128

    # Optuna databases
    STUDY_NAME_PPO: str = 'phase2_ppo'
    STUDY_NAME_DDQN: str = 'phase2_ddqn'
    DB_PATH_PPO: str = 'databases/optuna_phase2_ppo.db'
    DB_PATH_DDQN: str = 'databases/optuna_phase2_ddqn.db'

    # Output
    RESULTS_FILE: str = 'phase2_results.json'

    # DDQN-specific
    DDQN_ACTION_BINS: int = 10  # Discretize actions: 0%, 10%, ..., 100%
    DDQN_REPLAY_BUFFER_SIZE: int = 100000
    DDQN_MIN_REPLAY_SIZE: int = 1000
    DDQN_TARGET_UPDATE_FREQ: int = 1000


# =============================================================================
# Time-Frame Constraint Configuration
# =============================================================================

@dataclass(frozen=True)
class TimeFrameConfig:
    """Time-frame constraint configuration."""

    # Timeframe duration in hours
    TIMEFRAME_HOURS: Dict[str, int] = None

    # Candles per day for each interval
    CANDLES_PER_DAY: Dict[str, int] = None

    def __post_init__(self):
        """Initialize computed values."""
        object.__setattr__(self, 'TIMEFRAME_HOURS', {
            '3d': 3 * 24,    # 72 hours
            '5d': 5 * 24,    # 120 hours
            '7d': 7 * 24,    # 168 hours
            '10d': 10 * 24,  # 240 hours
            '14d': 14 * 24,  # 336 hours
        })

        object.__setattr__(self, 'CANDLES_PER_DAY', {
            '5m': 288,   # 12 per hour * 24
            '15m': 96,   # 4 per hour * 24
            '30m': 48,   # 2 per hour * 24
            '1h': 24,    # 1 per hour * 24
            '4h': 6,     # 0.25 per hour * 24
        })

    def get_candles_in_timeframe(self, timeframe: str, interval: str) -> int:
        """
        Calculate number of candles in a given timeframe.

        Args:
            timeframe: e.g., '3d', '5d', '7d'
            interval: e.g., '5m', '15m', '1h'

        Returns:
            Number of candles

        Example:
            >>> config = TimeFrameConfig()
            >>> config.get_candles_in_timeframe('5d', '1h')
            120  # 5 days * 24 hours/day
        """
        if self.TIMEFRAME_HOURS is None or self.CANDLES_PER_DAY is None:
            self.__post_init__()

        hours = self.TIMEFRAME_HOURS[timeframe]
        candles_per_hour = self.CANDLES_PER_DAY[interval] / 24.0
        return int(hours * candles_per_hour)


# =============================================================================
# Fee Tier Configuration
# =============================================================================

@dataclass(frozen=True)
class FeeTierConfig:
    """Coinbase fee tier configuration."""

    # Coinbase maker fee tiers (sorted by volume threshold)
    FEE_TIERS: Tuple[Dict[str, float], ...] = (
        {'volume_threshold': 0,       'maker': 0.0060, 'taker': 0.0120},  # $0
        {'volume_threshold': 10000,   'maker': 0.0040, 'taker': 0.0080},  # $10K
        {'volume_threshold': 25000,   'maker': 0.0025, 'taker': 0.0050},  # $25K
        {'volume_threshold': 75000,   'maker': 0.00125, 'taker': 0.0025}, # $75K
        {'volume_threshold': 250000,  'maker': 0.00075, 'taker': 0.0015}, # $250K
        {'volume_threshold': 500000,  'maker': 0.00060, 'taker': 0.00125},# $500K VIP 1
        {'volume_threshold': 1000000, 'maker': 0.00050, 'taker': 0.00100},# $1M VIP 2
        {'volume_threshold': 5000000, 'maker': 0.00040, 'taker': 0.00085},# $5M VIP 3
    )

    # 30-day rolling window (in candles)
    # Will be computed based on interval
    WINDOW_DAYS: int = 30

    # Fee mode
    FEE_MODE_PROGRESSIVE: str = 'progressive'  # Interpolate between tiers
    FEE_MODE_STATIC: str = 'static'            # Jump to tier
    DEFAULT_FEE_MODE: str = FEE_MODE_PROGRESSIVE

    def get_window_candles(self, interval: str) -> int:
        """
        Get number of candles in 30-day window for given interval.

        Args:
            interval: e.g., '5m', '1h', '4h'

        Returns:
            Number of candles in 30 days
        """
        candles_per_day = {
            '5m': 288, '15m': 96, '30m': 48, '1h': 24, '4h': 6
        }
        return self.WINDOW_DAYS * candles_per_day[interval]


# =============================================================================
# Data Preparation Configuration
# =============================================================================

@dataclass(frozen=True)
class DataConfig:
    """Data preparation configuration."""

    # Data directories
    DATA_DIR: str = 'data'
    PHASE1_DATA_DIR: str = 'data/phase1'
    PHASE2_DATA_DIR: str = 'data/phase2'

    # Data files pattern
    PRICE_ARRAY_FILE: str = 'price_array_{interval}_{months}mo.npy'
    TECH_ARRAY_FILE: str = 'tech_array_{interval}_{months}mo.npy'
    TIME_ARRAY_FILE: str = 'time_array_{interval}_{months}mo.npy'

    # Phase 2 enhanced data
    ENHANCED_TECH_ARRAY_FILE: str = 'tech_array_enhanced_{interval}_{months}mo.npy'

    # Lookback for data preparation
    DEFAULT_MONTHS: int = 12

    # Technical indicators (base)
    TECH_INDICATORS: Tuple[str, ...] = (
        'open', 'high', 'low', 'close', 'volume',
        'macd', 'macd_signal', 'macd_hist',
        'rsi', 'cci', 'dx'
    )

    # Rolling mean features (Phase 2)
    ROLLING_FEATURES: Tuple[str, ...] = (
        'close_ma7d', 'volume_ma7d',    # 7-day rolling means
        'close_ma30d', 'volume_ma30d',  # 30-day rolling means
    )


# =============================================================================
# Create Singleton Instances
# =============================================================================

PHASE1 = Phase1Config()
PHASE2 = Phase2Config()
TIMEFRAME = TimeFrameConfig()
FEE_TIER = FeeTierConfig()
DATA = DataConfig()


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    'PHASE1',
    'PHASE2',
    'TIMEFRAME',
    'FEE_TIER',
    'DATA',
    'Phase1Config',
    'Phase2Config',
    'TimeFrameConfig',
    'FeeTierConfig',
    'DataConfig',
]


# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """Validate configuration values."""
    # Phase 1 validation
    assert PHASE1.TOTAL_COMBINATIONS == 25, "Expected 25 combinations (5 timeframes × 5 intervals)"
    assert PHASE1.TOTAL_TRIALS == 500, "Expected 500 total Phase 1 trials"
    assert PHASE1.BUY_COST_PCT == 0.006, "Phase 1 should use 0.6% maker fees"

    # Phase 2 validation
    assert PHASE2.ENHANCED_STATE_DIM == 91, "Expected state dim 91 (63 base + 28 rolling)"
    assert PHASE2.TOTAL_TRIALS == 400, "Expected 400 total Phase 2 trials (200 PPO + 200 DDQN)"

    # Timeframe validation
    timeframe_cfg = TimeFrameConfig()
    assert timeframe_cfg.get_candles_in_timeframe('5d', '1h') == 120
    assert timeframe_cfg.get_candles_in_timeframe('7d', '4h') == 42

    # Fee tier validation
    assert len(FEE_TIER.FEE_TIERS) == 8, "Expected 8 fee tiers"
    assert FEE_TIER.FEE_TIERS[0]['maker'] == 0.006, "First tier should be 0.6% maker"
    assert FEE_TIER.FEE_TIERS[2]['volume_threshold'] == 25000, "Third tier should be $25K"

    print("✓ Configuration validation passed")


# Run validation on import
if __name__ != '__main__':
    validate_config()


# =============================================================================
# Testing/Debugging
# =============================================================================

if __name__ == '__main__':
    print("Two-Phase Training Configuration")
    print("=" * 60)

    print("\nPhase 1: Time-Frame Optimization")
    print(f"  Time-frames: {PHASE1.TIME_FRAMES}")
    print(f"  Intervals: {PHASE1.INTERVALS}")
    print(f"  Trials per combo: {PHASE1.TRIALS_PER_COMBINATION}")
    print(f"  Total combinations: {PHASE1.TOTAL_COMBINATIONS}")
    print(f"  Total trials: {PHASE1.TOTAL_TRIALS}")
    print(f"  Fees: {PHASE1.BUY_COST_PCT:.3%} maker")

    print("\nPhase 2: Feature-Enhanced Training")
    print(f"  PPO trials: {PHASE2.TRIALS_PPO}")
    print(f"  DDQN trials: {PHASE2.TRIALS_DDQN}")
    print(f"  Total trials: {PHASE2.TOTAL_TRIALS}")
    print(f"  State dimension: {PHASE2.BASE_STATE_DIM} → {PHASE2.ENHANCED_STATE_DIM}")
    print(f"  Rolling windows: {PHASE2.ROLLING_WINDOW_SHORT}d, {PHASE2.ROLLING_WINDOW_LONG}d")

    print("\nTime-Frame Examples")
    tf_config = TimeFrameConfig()
    for timeframe in ['3d', '5d', '7d']:
        for interval in ['1h', '4h']:
            candles = tf_config.get_candles_in_timeframe(timeframe, interval)
            print(f"  {timeframe} @ {interval}: {candles} candles")

    print("\nFee Tiers")
    for i, tier in enumerate(FEE_TIER.FEE_TIERS[:5]):
        vol = tier['volume_threshold']
        maker = tier['maker']
        print(f"  Tier {i+1}: ${vol:>7,} → {maker:.3%} maker")

    print("\n" + "=" * 60)
    validate_config()
