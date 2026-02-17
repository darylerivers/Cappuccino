"""Centralized Constants for Cappuccino Trading System

All magic numbers, default values, and configuration constants in one place.
Makes tuning easier and improves code readability.

Usage:
    from constants import RISK, NORMALIZATION, TRADING, DISCORD
    stop_loss = RISK.STOP_LOSS_PCT
"""

from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env before initializing constants
except ImportError:
    pass  # dotenv not required if env vars are set manually


@dataclass(frozen=True)
class RiskConstants:
    """Risk management default values."""

    # Per-position limits
    MAX_POSITION_PCT: float = 0.30  # 30% max per asset
    STOP_LOSS_PCT: float = 0.10  # 10% stop-loss from entry
    TRAILING_STOP_PCT: float = 0.0  # Disabled by default
    ACTION_DAMPENING: float = 1.0  # No dampening

    # Portfolio-level profit protection
    PORTFOLIO_TRAILING_STOP_PCT: float = 0.015  # 1.5% from peak
    PROFIT_TAKE_THRESHOLD_PCT: float = 0.03  # Take profits at 3%
    PROFIT_TAKE_AMOUNT_PCT: float = 0.50  # Sell 50% when taking profits
    MOVE_TO_CASH_THRESHOLD_PCT: float = 0.0  # Disabled by default
    COOLDOWN_AFTER_CASH_HOURS: int = 24  # 24h cooldown

    # Safety factors
    ALPACA_QTY_SAFETY_FACTOR: float = 1.1  # 10% buffer on min qty
    STOCK_BUY_SAFETY_FACTOR: float = 0.99  # 1% safety on cash


@dataclass(frozen=True)
class NormalizationConstants:
    """State/action normalization scales."""

    # State normalization (powers of 2 for efficiency)
    NORM_CASH_EXP: int = -11  # 2^-11 ≈ 0.00048828
    NORM_STOCKS_EXP: int = -8  # 2^-8 = 0.00390625
    NORM_TECH_EXP: int = -14  # 2^-14 ≈ 0.000061
    NORM_REWARD_EXP: int = -9  # 2^-9 ≈ 0.001953

    # Computed values
    NORM_CASH: float = 2**NORM_CASH_EXP
    NORM_STOCKS: float = 2**NORM_STOCKS_EXP
    NORM_TECH: float = 2**NORM_TECH_EXP
    NORM_REWARD: float = 2**NORM_REWARD_EXP

    # Action scaling
    NORM_ACTION: float = 100.0  # Scale model outputs


@dataclass(frozen=True)
class TradingConstants:
    """Trading execution parameters."""

    # Transaction costs
    BUY_COST_PCT: float = 0.0025  # 0.25% Alpaca fee
    SELL_COST_PCT: float = 0.0025  # 0.25% Alpaca fee

    # Paper trading
    INITIAL_CAPITAL: float = 500.0  # Reduced for February 2026 testing
    POLL_INTERVAL_SECONDS: int = 60
    HISTORY_HOURS_DEFAULT: int = 120  # 5 days
    HISTORY_HOURS_MIN: int = 24
    HISTORY_HOURS_MAX: int = 120

    # Environment
    DISCOUNT_FACTOR: float = 0.99  # Gamma for RL
    MIN_CASH_RESERVE_DEFAULT: float = 0.10  # 10% min cash
    CONCENTRATION_PENALTY_DEFAULT: float = 0.05
    TIME_DECAY_FLOOR_DEFAULT: float = 0.0


@dataclass(frozen=True)
class TrainingConstants:
    """Training/optimization parameters."""

    # Optuna
    N_TRIALS: int = 150
    KCV_GROUPS: int = 5
    K_TEST_GROUPS: int = 2
    NUM_PATHS: int = 3

    # Time windows (hours)
    TRAIN_WINDOW_HOURS: int = 1440  # 60 days
    VAL_WINDOW_HOURS: int = 240  # 10 days

    # Model architecture ranges
    NET_DIM_MIN: int = 64
    NET_DIM_MAX: int = 512
    LOOKBACK_MIN: int = 20
    LOOKBACK_MAX: int = 180

    # Learning rates
    LR_MIN: float = 1e-5
    LR_MAX: float = 1e-3

    # Random seed
    SEED: int = 2390408


@dataclass(frozen=True)
class DataConstants:
    """Data processing parameters."""

    import numpy as np

    # Timeframes
    TIMEFRAME_DEFAULT: str = '1h'

    # Technical indicators (14 per asset - includes new regime detection features)
    TECH_INDICATORS: tuple = (
        'open', 'high', 'low', 'close', 'volume',
        'macd', 'macd_signal', 'macd_hist',
        'rsi', 'cci', 'dx',
        'atr_regime_shift',        # ATR regime shift detector
        'range_breakout_volume',   # Range breakout with volume confirmation
        'trend_reacceleration'     # Trend strength re-acceleration (ADX 2nd derivative)
    )

    # Tickers (default 7 assets)
    DEFAULT_TICKERS: tuple = (
        'AAVE/USD', 'AVAX/USD', 'BTC/USD', 'LINK/USD',
        'ETH/USD', 'LTC/USD', 'UNI/USD'
    )

    # Minimum buy limits (adjusted for Alpaca/Coinbase $1 minimum order)
    # Must match DEFAULT_TICKERS exactly (7 tickers)
    ALPACA_LIMITS: tuple = (
        0.005,    # AAVE (~$1.00 at $200/AAVE)
        0.03,     # AVAX (~$1.00 at $35/AVAX)
        0.00001,  # BTC (~$1.10 at $110k/BTC)
        0.1,      # LINK (~$1.00 at $10/LINK)
        0.0003,   # ETH (~$1.20 at $4k/ETH)
        0.01,     # LTC (~$0.95 at $95/LTC)
        0.15,     # UNI (~$1.00 at $7/UNI)
    )

    # Data points per year by timeframe
    TIMEFRAME_PERIODS_PER_YEAR: dict = None  # Set in __post_init__

    def __post_init__(self):
        """Calculate periods per year."""
        object.__setattr__(self, 'TIMEFRAME_PERIODS_PER_YEAR', {
            '1m': 60 * 24 * 365,
            '5m': 12 * 24 * 365,
            '15m': 4 * 24 * 365,
            '30m': 2 * 24 * 365,
            '1h': 24 * 365,
            '4h': 6 * 365,
            '12h': 2 * 365,
            '1d': 365,
        })


@dataclass(frozen=True)
class MonitoringConstants:
    """System monitoring parameters."""

    # Watchdog
    WATCHDOG_CHECK_INTERVAL_SECONDS: int = 60
    MAX_RESTARTS: int = 3
    RESTART_COOLDOWN_SECONDS: int = 300  # 5 minutes

    # Performance monitor
    MONITOR_CHECK_INTERVAL_SECONDS: int = 300  # 5 minutes

    # Auto-deployer
    DEPLOYER_CHECK_INTERVAL_SECONDS: int = 3600  # 1 hour
    MIN_IMPROVEMENT_PCT: float = 1.0  # 1% better than current

    # Ollama advisor
    ADVISOR_INTERVAL_HOURS: int = 2


@dataclass(frozen=True)
class DiscordConstants:
    """Discord bot and notification settings."""

    import os

    # Bot configuration
    BOT_TOKEN: str = os.getenv('DISCORD_BOT_TOKEN', '')
    WEBHOOK_URL: str = os.getenv('DISCORD_WEBHOOK_URL', '')

    # Channel IDs (as integers)
    TRADE_CHANNEL_ID: int = int(os.getenv('DISCORD_TRADE_CHANNEL_ID', '0'))
    ALERT_CHANNEL_ID: int = int(os.getenv('DISCORD_ALERT_CHANNEL_ID', '0'))

    # Feature flags
    ENABLED: bool = os.getenv('DISCORD_ENABLED', 'false').lower() == 'true'
    NOTIFY_TRADES: bool = True
    NOTIFY_ALERTS: bool = True
    NOTIFY_TRAINING: bool = True
    NOTIFY_DEPLOYMENTS: bool = True


@dataclass(frozen=True)
class PathConstants:
    """File paths and directories."""

    # Directories
    DATA_DIR: str = 'data'
    LOGS_DIR: str = 'logs'
    DATABASES_DIR: str = 'databases'
    TRAIN_RESULTS_DIR: str = 'train_results'
    PAPER_TRADES_DIR: str = 'paper_trades'
    DEPLOYMENTS_DIR: str = 'deployments'
    BASELINES_DIR: str = 'baselines'
    TESTS_DIR: str = 'tests'
    CONTEXTS_DIR: str = 'contexts'

    # Database files
    OPTUNA_DB: str = 'databases/optuna_cappuccino.db'

    # Log files
    PAPER_TRADING_LOG: str = 'logs/paper_trading_live.log'
    FAILSAFE_LOG: str = 'logs/paper_trading_failsafe.log'
    WATCHDOG_LOG: str = 'logs/watchdog.log'
    MONITOR_LOG: str = 'logs/performance_monitor.log'

    # Trading logs
    PAPER_TRADES_CSV: str = 'paper_trades/alpaca_session.csv'
    PROFIT_PROTECTION_LOG: str = 'paper_trades/profit_protection.log'
    ENSEMBLE_VOTES_JSON: str = 'paper_trades/ensemble_votes.json'

    # Model directories
    ENSEMBLE_DIR: str = 'train_results/ensemble'
    ADAPTIVE_ENSEMBLE_DIR: str = 'train_results/adaptive_ensemble'


# Create singleton instances
RISK = RiskConstants()
NORMALIZATION = NormalizationConstants()
TRADING = TradingConstants()
TRAINING = TrainingConstants()
DATA = DataConstants()
MONITORING = MonitoringConstants()
DISCORD = DiscordConstants()
PATHS = PathConstants()


# Convenience exports
__all__ = [
    'RISK',
    'NORMALIZATION',
    'TRADING',
    'TRAINING',
    'DATA',
    'MONITORING',
    'DISCORD',
    'PATHS',
]


# Validation on import
def validate_constants():
    """Validate that constants are reasonable."""
    # Risk checks
    assert 0 < RISK.MAX_POSITION_PCT <= 1.0, "Max position must be 0-100%"
    assert 0 < RISK.STOP_LOSS_PCT <= 1.0, "Stop-loss must be 0-100%"
    assert 0 <= RISK.PROFIT_TAKE_AMOUNT_PCT <= 1.0, "Profit take amount must be 0-100%"

    # Normalization checks
    assert NORMALIZATION.NORM_CASH > 0, "Cash normalization must be positive"
    assert NORMALIZATION.NORM_ACTION > 0, "Action normalization must be positive"

    # Trading checks
    assert TRADING.BUY_COST_PCT >= 0, "Transaction costs must be non-negative"
    assert TRADING.INITIAL_CAPITAL > 0, "Initial capital must be positive"
    assert 0 < TRADING.DISCOUNT_FACTOR <= 1.0, "Discount factor must be 0-1"

    # Training checks
    assert TRAINING.N_TRIALS > 0, "Must have positive number of trials"
    assert TRAINING.LOOKBACK_MIN < TRAINING.LOOKBACK_MAX, "Lookback range invalid"


# Run validation on import
validate_constants()
