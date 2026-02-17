"""Paper trading with Alpaca Markets API using REST API polling.

This version uses REST API polling instead of websockets since crypto websocket
access may not be available for all accounts.

Usage example::

    python paper_trader_alpaca_polling.py \
        --model-dir train_results/validation_60d_7d_2025-10-15__17_54_22 \
        --tickers BTC/USD ETH/USD LTC/USD \
        --timeframe 5m \
        --history-hours 120 \
        --poll-interval 60 \
        --log-file paper_trades/alpaca_session.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch

# Import centralized constants
from constants import RISK, NORMALIZATION, TRADING, DISCORD

# Discord notifications
try:
    from integrations.discord_notifier import DiscordNotifier
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    print("⚠️  Discord integration not available (integrations.discord_notifier not found)")

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
except ImportError:
    pass  # dotenv not required if env vars are set manually

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
except ImportError:
    print("ERROR: alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
    sys.exit(1)

from config_main import TECHNICAL_INDICATORS_LIST
from drl_agents.elegantrl_models import MODELS
from environment_Alpaca import CryptoEnvAlpaca
from train.config import Arguments
from train.run import init_agent


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    holdings: Dict[str, float]
    prices: Dict[str, float]
    total_asset: float
    reward: float
    action: List[float]


@dataclass
class RiskManagement:
    """Risk management configuration for paper trading.

    Defaults imported from constants.RISK for consistency.
    """
    # Per-position limits
    max_position_pct: float = RISK.MAX_POSITION_PCT
    stop_loss_pct: float = RISK.STOP_LOSS_PCT
    trailing_stop_pct: float = RISK.TRAILING_STOP_PCT
    min_trade_interval_hours: int = 4  # Minimum time to hold position before allowing sells
    action_dampening: float = RISK.ACTION_DAMPENING

    # Portfolio-level profit protection
    portfolio_trailing_stop_pct: float = RISK.PORTFOLIO_TRAILING_STOP_PCT
    profit_take_threshold_pct: float = RISK.PROFIT_TAKE_THRESHOLD_PCT
    profit_take_amount_pct: float = RISK.PROFIT_TAKE_AMOUNT_PCT
    move_to_cash_threshold_pct: float = RISK.MOVE_TO_CASH_THRESHOLD_PCT
    cooldown_after_cash_hours: int = RISK.COOLDOWN_AFTER_CASH_HOURS


class AlpacaPaperTraderPolling:
    def __init__(
        self,
        *,
        tickers: List[str],
        timeframe: str,
        model_dir: Path,
        history_hours: int,
        poll_interval: int,
        gpu_id: int,
        log_file: Path,
        paper: bool = True,
        enable_sentiment: bool = False,
        risk_management: Optional[RiskManagement] = None,
    ) -> None:
        self.model_dir = model_dir
        self.timeframe = timeframe
        self.history_hours = history_hours
        self.poll_interval = poll_interval
        self.gpu_id = gpu_id
        self.log_file = log_file
        self.paper = paper
        self.enable_sentiment = enable_sentiment

        # Risk management
        self.risk_mgmt = risk_management or RiskManagement()
        self.entry_prices: Dict[str, float] = {}  # Track entry price per ticker
        self.high_water_mark: Dict[str, float] = {}  # Track highest price since entry (for trailing stops)
        self.last_trade_time: Dict[str, datetime] = {}  # Track last trade time per ticker
        self.stop_loss_triggered: Dict[str, bool] = {}  # Track if stop-loss was triggered

        # Portfolio-level profit protection tracking
        self.initial_portfolio_value: Optional[float] = None  # Set on first trade
        self.portfolio_high_water_mark: float = 0.0  # Highest portfolio value seen
        self.profit_taken: bool = False  # Has partial profit been taken?
        self.in_cash_mode: bool = False  # Are we waiting in cash after hitting target?
        self.cash_mode_started: Optional[datetime] = None  # When did we enter cash mode?
        self.portfolio_protection_log_path = Path("paper_trades/profit_protection.log")

        # Heartbeat for staleness detection (watchdog monitors this)
        # Make heartbeat trial-specific so dashboard can track each trader
        heartbeat_name = self.log_file.stem.replace('_session', '_heartbeat.json')
        self.heartbeat_path = Path("paper_trades") / heartbeat_name
        self.poll_count = 0
        self.last_poll_completed = None  # Track when poll actually completes

        # Extract trial number from model_dir path (e.g., "deployment_trial250_...")
        self.trial_number = "unknown"
        if model_dir and "trial" in str(model_dir).lower():
            import re
            match = re.search(r'trial[_\s]*(\d+)', str(model_dir), re.IGNORECASE)
            if match:
                self.trial_number = match.group(1)

        # Tiburtina integration (macro-aware position sizing)
        self.tiburtina_bridge = None
        self.macro_regime = "unknown"
        self.macro_multiplier = 1.0
        self.macro_reason = "not_initialized"
        try:
            from integrations.tiburtina_helper import get_tiburtina_bridge
            self.tiburtina_bridge = get_tiburtina_bridge()
            if self.tiburtina_bridge.is_available():
                print("✅ Tiburtina integration active - macro-aware position sizing enabled")
            else:
                print("⚠️  Tiburtina unavailable - using standard position sizing")
        except Exception as e:
            print(f"⚠️  Tiburtina integration failed: {e}")
            print("   Continuing with standard position sizing...")

        # Discord notifications
        self.discord = None
        if DISCORD_AVAILABLE and DISCORD.ENABLED:
            try:
                self.discord = DiscordNotifier()
                if self.discord.enabled:
                    print("✅ Discord notifications enabled")
                else:
                    print("⚠️  Discord webhook not configured - notifications disabled")
                    self.discord = None
            except Exception as e:
                print(f"⚠️  Discord integration failed: {e}")
                self.discord = None
        else:
            if not DISCORD_AVAILABLE:
                print("ℹ️  Discord integration not available")
            elif not DISCORD.ENABLED:
                print("ℹ️  Discord notifications disabled in config")

        # Auto-configure tickers from model checkpoint
        self._auto_configure_from_model(tickers)

        # Normalize ticker format
        self.tickers = [tic if "/" in tic else f"{tic}/USD" for tic in self.tickers]

        # Get Alpaca credentials
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.api_secret = os.getenv("ALPACA_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET environment variables required")

        # Initialize Alpaca API
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.api = tradeapi.REST(self.api_key, self.api_secret, base_url, api_version='v2')

        # Data storage
        self.raw_bars: Dict[str, pd.DataFrame] = {tic: pd.DataFrame() for tic in self.tickers}
        self.price_array: np.ndarray = np.zeros((0, len(self.tickers)), dtype=np.float32)
        self.tech_array: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.time_array: deque[datetime] = deque(maxlen=10000)
        self.last_timestamp: Optional[datetime] = None
        self.last_bar_signature: Optional[str] = None  # Track last processed bar to avoid duplicates

        self.env: Optional[CryptoEnvAlpaca] = None
        self.agent = None
        self.act = None
        self.device = torch.device("cpu")
        self.state: Optional[np.ndarray] = None

        self._stop = False

        # Setup
        # Check if ensemble mode (skip trial loading)
        self.is_ensemble = "ensemble" in str(self.model_dir).lower()

        if not self.is_ensemble:
            self._load_trial_and_agent()
        else:
            # Ensemble mode - will load hyperparameters from ensemble
            # These are temporary placeholders
            self.lookback = 60
            self.norm_cash = 2**-11
            self.norm_stocks = 2**-8
            self.norm_tech = 2**-14
            self.norm_reward = 2**-9
            self.norm_action = 100
            self.time_decay_floor = 0.0
            self.min_cash_reserve = 0.1
            self.concentration_penalty = 0.05

        self._bootstrap_history()
        self._prepare_environment()
        self._warmup_environment()
        self._load_saved_state()  # Restore positions from previous session
        self._prepare_logger()

    def _auto_configure_from_model(self, requested_tickers: List[str]) -> None:
        """Auto-configure paper trader from model checkpoint."""
        import torch

        # Find checkpoint
        actor_path = self.model_dir / "stored_agent" / "actor.pth"
        if not actor_path.exists():
            actor_path = self.model_dir / "actor.pth"

        if not actor_path.exists():
            print(f"⚠️  Warning: Could not find checkpoint, using requested tickers")
            self.tickers = requested_tickers
            return

        # Load checkpoint to inspect
        checkpoint = torch.load(actor_path, map_location='cpu', weights_only=True)

        # Extract number of assets from action dimension
        if 'net.4.weight' in checkpoint:
            n_actions = checkpoint['net.4.weight'].shape[0]
            n_tickers_model = n_actions

            print(f"🔍 Model expects {n_tickers_model} tickers")

            # Try to use requested tickers if count matches
            if len(requested_tickers) == n_tickers_model:
                print(f"✓ Using requested tickers (count matches)")
                self.tickers = requested_tickers
            else:
                # Use default training tickers
                from config_main import TICKER_LIST
                if len(TICKER_LIST) >= n_tickers_model:
                    self.tickers = TICKER_LIST[:n_tickers_model]
                    print(f"⚠️  Requested {len(requested_tickers)} tickers, model needs {n_tickers_model}")
                    print(f"✓ Using training tickers: {self.tickers}")
                else:
                    print(f"❌ Error: Model needs {n_tickers_model} tickers, only {len(TICKER_LIST)} available")
                    self.tickers = requested_tickers  # Fallback
        else:
            print(f"⚠️  Could not determine model ticker count, using requested")
            self.tickers = requested_tickers

    def _load_trial_and_agent(self) -> None:
        """Load trained model and hyperparameters."""
        trial_path = self.model_dir / "best_trial"
        if not trial_path.exists():
            raise FileNotFoundError(f"Missing best_trial in {self.model_dir}")

        with trial_path.open("rb") as handle:
            self.trial = pickle.load(handle)

        self.model_name = self.trial.user_attrs.get("model_name", "ppo")
        if self.model_name not in MODELS:
            raise ValueError(f"Unsupported model '{self.model_name}'.")

        params = self.trial.params
        self.lookback = int(params["lookback"])
        self.norm_cash = float(2 ** params.get("norm_cash_exp", params.get("norm_cash", -11)))
        self.norm_stocks = float(2 ** params.get("norm_stocks_exp", params.get("norm_stocks", -8)))
        self.norm_tech = float(2 ** params.get("norm_tech_exp", params.get("norm_tech", -14)))
        self.norm_reward = float(2 ** params.get("norm_reward_exp", params.get("norm_reward", -9)))
        # Override norm_action for paper trading with $1000 capital
        # Training uses large norm_action (10k-20k) for backtesting, but this makes
        # trades too small for Alpaca's minimum quantities. Use 100 for $1000 capital.
        self.norm_action = 100.0  # float(params["norm_action"])
        self.net_dimension = int(params["net_dimension"])
        self.time_decay_floor = float(params.get("time_decay_floor", 0.0))
        self.min_cash_reserve = float(params.get("min_cash_reserve", 0.1))
        self.concentration_penalty = float(params.get("concentration_penalty", 0.05))

        # Check for FT-Transformer configuration
        self.use_ft_encoder = params.get("use_ft_encoder", False)
        if self.use_ft_encoder:
            print(f"\n{'='*70}")
            print(f"🔍 Detected FT-Transformer Model")
            print(f"{'='*70}")

            # Load FT configuration
            self.ft_config = {
                'd_token': int(params.get('ft_d_token', 32)),
                'n_blocks': int(params.get('ft_n_blocks', 2)),
                'n_heads': int(params.get('ft_n_heads', 4)),
                'dropout': float(params.get('ft_dropout', 0.1)),
            }
            self.pretrained_encoder_path = params.get('pretrained_encoder_path', None)
            self.ft_freeze_encoder = params.get('ft_freeze_encoder', False)

            print(f"  FT Config: {self.ft_config}")
            print(f"  Pre-trained: {self.pretrained_encoder_path is not None}")
            print(f"  Freeze encoder: {self.ft_freeze_encoder}")
            print(f"{'='*70}\n")
        else:
            self.ft_config = None
            self.pretrained_encoder_path = None
            self.ft_freeze_encoder = False

        # Use model_dir directly as the weights directory
        # (model files are in model_dir, not in a stored_agent subdirectory)
        self.cwd_path = self.model_dir
        if not self.cwd_path.exists():
            raise FileNotFoundError(f"Weights directory not found: {self.cwd_path}")

        # Check actual net_dimension from checkpoint (may differ from trial params)
        actor_checkpoint = self.cwd_path / "actor.pth"
        if actor_checkpoint.exists():
            import torch
            checkpoint_state = torch.load(actor_checkpoint, map_location='cpu', weights_only=True)

            # For FT models, check for encoder weights
            if self.use_ft_encoder:
                has_encoder = 'encoder.token_embedding.weight' in checkpoint_state
                print(f"  ✓ Checkpoint has FT encoder weights: {has_encoder}")

            if 'net.0.weight' in checkpoint_state:
                actual_net_dim = checkpoint_state['net.0.weight'].shape[0]
                if actual_net_dim != self.net_dimension:
                    print(f"⚠️  Warning: Trial params say net_dimension={self.net_dimension}, "
                          f"but checkpoint has {actual_net_dim}. Using actual: {actual_net_dim}")
                    self.net_dimension = actual_net_dim

                # Auto-detect correct lookback from model input dimension
                # State = 1 (cash) + num_tickers (stocks) + (num_tickers * tech_indicators) * lookback
                # For 7 tickers with 14 tech indicators: state_dim = 8 + 98 * lookback
                model_input_dim = checkpoint_state['net.0.weight'].shape[1]
                num_tickers = len(self.tickers)
                tech_per_ticker = 14  # OHLCV + 9 technical indicators
                base_features = 1 + num_tickers  # cash + stocks
                tech_features_per_step = num_tickers * tech_per_ticker

                # Calculate implied lookback
                implied_lookback = (model_input_dim - base_features) / tech_features_per_step

                if abs(implied_lookback - round(implied_lookback)) < 0.01:  # Close to integer
                    detected_lookback = int(round(implied_lookback))
                    if detected_lookback != self.lookback:
                        print(f"⚠️  Warning: Trial params say lookback={self.lookback}, "
                              f"but model expects lookback={detected_lookback}. Using detected: {detected_lookback}")
                        print(f"    Model input_dim={model_input_dim}, base={base_features}, "
                              f"tech_per_step={tech_features_per_step}")
                        self.lookback = detected_lookback
                    else:
                        print(f"✓ Verified lookback={self.lookback} matches model input dimension")
                else:
                    print(f"⚠️  Warning: Could not auto-detect lookback (implied={implied_lookback:.2f}). "
                          f"Using trial param: {self.lookback}")

    def _timeframe_to_alpaca(self, tf: str) -> TimeFrame:
        """Convert timeframe string to Alpaca TimeFrame."""
        mapping = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day,
        }
        if tf in mapping:
            return mapping[tf]
        raise ValueError(f"Unsupported timeframe '{tf}' for Alpaca")

    def _bootstrap_history(self) -> None:
        """Download historical data to initialize technical indicators."""
        print(f"Downloading {self.history_hours}h of historical data...")

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=self.history_hours)

        alpaca_tf = self._timeframe_to_alpaca(self.timeframe)

        all_data = []
        for symbol in self.tickers:
            bars = self.api.get_crypto_bars(
                symbol,
                alpaca_tf,
                start.isoformat(),
                end.isoformat(),
            ).df

            if bars.empty:
                raise RuntimeError(f"No historical data returned for {symbol}")

            bars = bars.reset_index()
            bars['tic'] = symbol
            bars = bars.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'timestamp': 'timestamp'
            })
            self.raw_bars[symbol] = bars
            all_data.append(bars)

        # Combine and process
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values(['timestamp', 'tic']).reset_index(drop=True)

        # Add technical indicators
        df = self._add_technical_indicators(df)

        # Check for duplicate columns (can happen if indicators already exist)
        if df.columns.duplicated().any():
            print(f"⚠️  Warning: Duplicate columns detected!")
            print(f"All columns: {list(df.columns)}")
            dupes = df.columns[df.columns.duplicated()].unique()
            print(f"Duplicates: {list(dupes)}")
            # Keep first occurrence
            df = df.loc[:, ~df.columns.duplicated()]
            print(f"After dedup: {list(df.columns)}")

        df = df.dropna()

        # Convert to arrays
        self.price_array, self.tech_array, self.time_array = self._df_to_arrays(df)
        self.last_timestamp = max(self.time_array)
        print(f"Loaded {len(self.time_array)} bars for {len(self.tickers)} tickers")
        print(f"Latest timestamp: {self.last_timestamp}")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        import talib
        import pandas as pd

        df_list = []
        for tic in self.tickers:
            tic_df = df[df['tic'] == tic].copy()

            # Calculate indicators using TA-Lib
            close = tic_df['close'].values
            high = tic_df['high'].values
            low = tic_df['low'].values
            volume = tic_df['volume'].values

            # Standard indicators
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            tic_df['macd'] = macd
            tic_df['macd_signal'] = macd_signal
            tic_df['macd_hist'] = macd_hist
            tic_df['rsi'] = talib.RSI(close, timeperiod=14)
            tic_df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            tic_df['dx'] = talib.DX(high, low, close, timeperiod=14)

            # NEW FEATURE 1: ATR regime shift
            # Detects volatility regime changes by comparing current ATR to historical average
            atr_14 = talib.ATR(high, low, close, timeperiod=14)
            atr_50_ma = pd.Series(atr_14).rolling(window=50, min_periods=1).mean().values
            tic_df['atr_regime_shift'] = (atr_14 - atr_50_ma) / (atr_50_ma + 1e-8)

            # NEW FEATURE 2: Range breakout + volume
            # Detects price breakouts from recent range with volume confirmation
            range_period = 20
            recent_high = pd.Series(high).rolling(window=range_period, min_periods=1).max().values
            recent_low = pd.Series(low).rolling(window=range_period, min_periods=1).min().values

            # Breakout signal: 1 if breaking above high, -1 if breaking below low, 0 otherwise
            breakout_signal = np.where(close > recent_high, 1.0,
                                     np.where(close < recent_low, -1.0, 0.0))

            # Volume confirmation: compare to 20-period average volume
            avg_volume = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
            volume_ratio = volume / (avg_volume + 1e-8)

            # Combined signal: breakout * volume ratio
            tic_df['range_breakout_volume'] = breakout_signal * volume_ratio

            # NEW FEATURE 3: Trend strength re-acceleration
            # Detects when trend strength (ADX) is accelerating (second derivative)
            adx_14 = talib.ADX(high, low, close, timeperiod=14)
            adx_change = pd.Series(adx_14).diff(periods=1).values  # First derivative
            adx_acceleration = pd.Series(adx_change).diff(periods=1).values  # Second derivative
            tic_df['trend_reacceleration'] = adx_acceleration

            df_list.append(tic_df)

        return pd.concat(df_list, ignore_index=True)

    def _df_to_arrays(self, df: pd.DataFrame) -> tuple:
        """Convert dataframe to price_array, tech_array, time_array."""
        # Match training data: 14 indicators per ticker (includes open, close, and 3 new features)
        tech_cols = ['open', 'high', 'low', 'close', 'volume',
                     'macd', 'macd_signal', 'macd_hist',
                     'rsi', 'cci', 'dx',
                     'atr_regime_shift', 'range_breakout_volume', 'trend_reacceleration']

        # Get unique timestamps
        timestamps = sorted(df['timestamp'].unique())
        n_times = len(timestamps)
        n_tickers = len(self.tickers)

        price_array = np.zeros((n_times, n_tickers), dtype=np.float32)
        tech_array = np.zeros((n_times, n_tickers * len(tech_cols)), dtype=np.float32)

        # Store last known values for forward filling
        last_known_price = {}
        last_known_tech = {}

        for i, ts in enumerate(timestamps):
            ts_data = df[df['timestamp'] == ts].sort_values('tic')

            for j, tic in enumerate(self.tickers):
                tic_data = ts_data[ts_data['tic'] == tic]
                if not tic_data.empty:
                    # Data available - use it and store
                    price_array[i, j] = tic_data['close'].iloc[0]

                    tech_values = tic_data[tech_cols].values.flatten()

                    last_known_price[tic] = price_array[i, j]
                    last_known_tech[tic] = tech_values
                else:
                    # Data missing - forward fill from last known value
                    if tic in last_known_price:
                        price_array[i, j] = last_known_price[tic]
                        tech_values = last_known_tech[tic]
                    else:
                        # First occurrence - use a safe default
                        # For crypto, this should rarely happen if we have enough history
                        raise ValueError(f"No initial data for {tic} at timestamp {ts}. Ensure sufficient history is downloaded.")

                # Write tech values to array
                start_idx = j * len(tech_cols)
                end_idx = start_idx + len(tech_cols)
                tech_array[i, start_idx:end_idx] = tech_values

        return price_array, tech_array, deque(timestamps)

    def _prepare_environment(self) -> None:
        """Initialize trading environment and load agent."""
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "if_train": False,
        }
        # Enable sentiment (trial was trained with sentiment)
        # Use real sentiment service if enabled, otherwise use zeros
        if self.enable_sentiment:
            print("✓ Initializing REAL sentiment analysis service...")
            try:
                from sentiment_analysis.service import SentimentService
                from pathlib import Path
                sentiment_service = SentimentService(
                    tickers=[tic.replace('/USD', '') for tic in self.tickers],  # Convert BTC/USD -> BTC
                    cache_duration_minutes=60,  # Cache for 1 hour
                    device='cpu',  # Use CPU for sentiment analysis
                    cache_dir=Path('sentiment_analysis/cache'),
                )
                sentiment_service.start_background_refresh(interval_minutes=60)  # Refresh every hour
                print(f"  ✓ Sentiment service started for {sentiment_service.tickers}")
                self.sentiment_service = sentiment_service
            except Exception as e:
                print(f"  ⚠ Failed to initialize sentiment service: {e}")
                print(f"  → Error details: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("  → Falling back to zeros")
                self.enable_sentiment = False
                sentiment_service = None

        if not self.enable_sentiment:
            # Create dummy sentiment service that returns zeros
            print("✓ Using dummy sentiment (zeros)")
            class DummySentimentService:
                def get_all_sentiment_arrays(self):
                    return {}  # Empty dict - environment will use zeros
            sentiment_service = DummySentimentService()
            self.sentiment_service = None

        # Check if using ensemble - load hyperparameters BEFORE creating environment
        if str(self.model_dir) == "train_results/ensemble" or "ensemble" in str(self.model_dir).lower():
            # Check for adaptive ensemble
            is_adaptive = "adaptive" in str(self.model_dir).lower()

            if is_adaptive:
                print("🧠 Loading ADAPTIVE ENSEMBLE mode (game theory voting)...")
                from models.adaptive_ensemble_agent import AdaptiveEnsembleAgent
                ensemble = AdaptiveEnsembleAgent(
                    ensemble_dir=str(self.model_dir),
                    device="cpu" if self.gpu_id < 0 else f"cuda:{self.gpu_id}",
                    elimination_interval=24,  # Eliminate worst every 24 steps
                    min_models=5,
                )
                self.adaptive_ensemble = ensemble
            else:
                print("🧠 Loading ENSEMBLE mode (10 models)...")
                from archive.deprecated_scripts.ultra_simple_ensemble import UltraSimpleEnsemble
                ensemble = UltraSimpleEnsemble(gpu_id=self.gpu_id)
                self.adaptive_ensemble = None

            # Get hyperparameters from ensemble
            ensemble_params = ensemble.get_required_hyperparameters()
            self.lookback = ensemble_params['lookback']
            self.norm_cash = ensemble_params['norm_cash']
            self.norm_stocks = ensemble_params['norm_stocks']
            self.norm_tech = ensemble_params['norm_tech']
            self.norm_reward = ensemble_params['norm_reward']
            self.norm_action = ensemble_params['norm_action']
            self.time_decay_floor = ensemble_params.get('time_decay_floor', 0.0)
            self.min_cash_reserve = ensemble_params.get('min_cash_reserve', 0.1)
            self.concentration_penalty = ensemble_params.get('concentration_penalty', 0.05)

            print(f"✓ Using ensemble hyperparameters: lookback={self.lookback}")

        # NOW create env_params with the correct hyperparameters
        env_params = {
            "lookback": self.lookback,
            "norm_cash": self.norm_cash,
            "norm_stocks": self.norm_stocks,
            "norm_tech": self.norm_tech,
            "norm_reward": self.norm_reward,
            "norm_action": self.norm_action,
            "time_decay_floor": self.time_decay_floor,
            "min_cash_reserve": self.min_cash_reserve,
            "concentration_penalty": self.concentration_penalty,
        }

        # Create environment with correct hyperparameters
        self.env = CryptoEnvAlpaca(
            env_config, env_params, if_log=True,
            use_sentiment=self.enable_sentiment, sentiment_service=sentiment_service, tickers=self.tickers
        )

        # Pass concentration limit to environment for enforcement on scaled actions
        self.env.max_position_pct = self.risk_mgmt.max_position_pct

        # Finish ensemble or single model setup
        if str(self.model_dir) == "train_results/ensemble" or "ensemble" in str(self.model_dir).lower():
            self.agent = ensemble
            self.act = ensemble.act
            self.device = ensemble.device
            self.ensemble = ensemble  # Store reference for voting logs
            print("✓ Ensemble loaded - using averaged predictions from 10 top models")
        else:
            self.ensemble = None
            # Single model mode

            # Check if using FT-Transformer
            if self.use_ft_encoder and self.model_name == 'ppo':
                print(f"\n{'='*70}")
                print(f"Loading FT-Transformer Enhanced Agent")
                print(f"{'='*70}\n")

                # Import FT agent
                from drl_agents.agents import AgentPPO_FT

                # Create args with FT configuration
                args = Arguments(agent=AgentPPO_FT, env=self.env)
                args.cwd = str(self.cwd_path)
                args.if_remove = False
                args.net_dim = self.net_dimension

                # Add FT-specific args
                args.use_ft_encoder = True
                args.ft_config = self.ft_config
                args.pretrained_encoder_path = self.pretrained_encoder_path
                args.freeze_encoder = self.ft_freeze_encoder

                print(f"  ✓ Using AgentPPO_FT")
                print(f"  ✓ FT Config: {self.ft_config}")
                if self.pretrained_encoder_path:
                    print(f"  ✓ Pre-trained encoder: {self.pretrained_encoder_path}")
                print(f"{'='*70}\n")

                agent = init_agent(args, gpu_id=self.gpu_id, env=self.env)
            else:
                # Standard agent (baseline MLP)
                args = Arguments(agent=MODELS[self.model_name], env=self.env)
                args.cwd = str(self.cwd_path)
                args.if_remove = False
                args.net_dim = self.net_dimension
                agent = init_agent(args, gpu_id=self.gpu_id, env=self.env)

            self.agent = agent
            self.act = agent.act
            self.device = agent.device

    def _warmup_environment(self) -> None:
        """Step through historical data with zero actions."""
        state = self.env.reset().astype(np.float32)
        zeros = np.zeros(self.env.action_dim, dtype=np.float32)
        while self.env.time < self.env.max_step:
            state, _, done, _ = self.env.step(zeros)
            if done:
                break
        self.state = state.astype(np.float32)
        print(f"Environment warmed up at step {self.env.time}/{self.env.max_step}")

    def _load_saved_state(self) -> bool:
        """Load saved position state from previous session.

        This allows the paper trader to resume with existing positions after restart,
        so the ensemble can decide whether to keep or sell them.

        Returns:
            True if state was loaded, False if starting fresh.
        """
        import json

        state_file = Path("paper_trades/positions_state.json")
        if not state_file.exists():
            print("📋 No saved state found - starting fresh")
            return False

        try:
            with state_file.open("r") as f:
                saved_state = json.load(f)

            # Check staleness - if state is too old (e.g., >24h), start fresh
            saved_time = datetime.fromisoformat(saved_state["timestamp"].replace('Z', '+00:00'))
            age_hours = (datetime.now(saved_time.tzinfo) - saved_time).total_seconds() / 3600

            if age_hours > 24:
                print(f"📋 Saved state is {age_hours:.1f}h old (>24h) - starting fresh")
                return False

            # Build ticker index map
            ticker_to_idx = {tic: i for i, tic in enumerate(self.tickers)}

            # Restore positions (holdings) into env.stocks
            restored_positions = 0
            for pos in saved_state.get("positions", []):
                ticker = pos.get("ticker")
                if ticker not in ticker_to_idx:
                    print(f"  ⚠️  Skipping unknown ticker: {ticker}")
                    continue

                idx = ticker_to_idx[ticker]
                holdings = pos.get("holdings", 0)
                entry_price = pos.get("entry_price", 0)
                high_water = pos.get("high_water_mark", entry_price)

                if holdings > 0:
                    # Restore holdings
                    self.env.stocks[idx] = holdings

                    # Restore tracking dicts
                    self.entry_prices[ticker] = entry_price
                    self.high_water_mark[ticker] = high_water
                    self.stop_loss_triggered[ticker] = pos.get("stop_loss_triggered", False)

                    # Restore last trade time (default to saved timestamp if not present)
                    if "last_trade_time" in pos:
                        self.last_trade_time[ticker] = datetime.fromisoformat(pos["last_trade_time"])
                    else:
                        # Use saved state timestamp as fallback
                        self.last_trade_time[ticker] = datetime.fromisoformat(saved_state["timestamp"])

                    restored_positions += 1
                    print(f"  ✅ Restored {ticker}: {holdings:.4f} @ entry ${entry_price:.2f}")

            # Restore cash
            saved_cash = saved_state.get("cash", self.env.initial_cash)
            self.env.cash = saved_cash

            # Restore portfolio protection state
            prot = saved_state.get("portfolio_protection", {})
            if prot.get("initial_value"):
                self.initial_portfolio_value = prot["initial_value"]
            self.portfolio_high_water_mark = prot.get("high_water_mark", 0.0)
            self.in_cash_mode = prot.get("in_cash_mode", False)
            self.profit_taken = prot.get("profit_taken", False)

            # Calculate and display restored portfolio value
            prices = self.price_array[-1] if len(self.price_array) > 0 else np.zeros(len(self.tickers))
            total_value = self.env.cash + np.dot(self.env.stocks, prices)

            print(f"\n📋 State Restored (from {age_hours:.1f}h ago):")
            print(f"   Cash: ${self.env.cash:.2f}")
            print(f"   Positions: {restored_positions}")
            print(f"   Portfolio Value: ${total_value:.2f}")

            if self.in_cash_mode:
                print("   ⚠️  In cash protection mode")

            # CRITICAL FIX: Update last_timestamp from CSV, not from historical fetch
            # This allows processing of bars that happened between CSV's last entry and now
            if self.log_file.exists():
                try:
                    csv_df = pd.read_csv(self.log_file)
                    if len(csv_df) > 0:
                        csv_last_ts = pd.to_datetime(csv_df['timestamp'].iloc[-1])
                        historical_ts = self.last_timestamp
                        self.last_timestamp = csv_last_ts
                        print(f"   ✓ CSV last bar: {csv_last_ts}")
                        print(f"   ✓ Historical fetch included: {historical_ts}")
                        hours_gap = (historical_ts - csv_last_ts).total_seconds() / 3600
                        if hours_gap > 0.5:
                            print(f"   → Will process {hours_gap:.1f}h of missed bars")
                except Exception as e:
                    print(f"   ⚠️  Could not read CSV last timestamp: {e}")

            return True

        except Exception as e:
            print(f"⚠️  Failed to load saved state: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _prepare_logger(self) -> None:
        """Create log file with headers."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with self.log_file.open("w", newline="") as handle:
                writer = csv.writer(handle)
                header = ["timestamp", "cash", "total_asset", "reward"]
                header += [f"holding_{tic}" for tic in self.tickers]
                header += [f"price_{tic}" for tic in self.tickers]
                header += [f"action_{tic}" for tic in self.tickers]
                writer.writerow(header)

    def _fetch_latest_bars(self) -> Optional[pd.DataFrame]:
        """Fetch the latest bars from Alpaca API."""
        end = datetime.now(timezone.utc)
        # Fetch enough history to capture at least 2-3 bars of our timeframe
        # For 1h bars, fetch last 4 hours to be safe
        hours_to_fetch = max(4, int(self.timeframe.replace('h', '').replace('m', '')) if 'h' in self.timeframe else 1)
        start = end - timedelta(hours=hours_to_fetch)

        alpaca_tf = self._timeframe_to_alpaca(self.timeframe)

        all_data = []
        for symbol in self.tickers:
            try:
                bars = self.api.get_crypto_bars(
                    symbol,
                    alpaca_tf,
                    start.isoformat(),
                    end.isoformat(),
                ).df

                if bars.empty:
                    print(f"    {symbol}: No bars returned")
                    continue

                print(f"    {symbol}: {len(bars)} bars")
                bars = bars.reset_index()
                bars['tic'] = symbol
                all_data.append(bars)
            except Exception as e:
                print(f"    Error fetching {symbol}: {e}")
                continue

        if not all_data:
            print(f"  No data from any ticker (fetched from {start.isoformat()} to {end.isoformat()})")
            return None

        df = pd.concat(all_data, ignore_index=True)
        return df

    def _compute_bar_signature(self, df: pd.DataFrame) -> str:
        """Compute a unique signature for a bar based on timestamp and data.

        This helps detect when we're re-fetching the same incomplete bar vs a completed bar.
        """
        import hashlib

        # Get latest timestamp in this batch
        if df.empty:
            return ""

        latest_ts = df['timestamp'].max()
        ts_data = df[df['timestamp'] == latest_ts].sort_values('tic')

        # Create signature from timestamp + close prices + volumes
        sig_parts = [str(latest_ts)]
        for _, row in ts_data.iterrows():
            sig_parts.append(f"{row['tic']}:{row.get('close', 0):.6f}:{row.get('volume', 0):.2f}")

        sig_string = "|".join(sig_parts)
        return hashlib.md5(sig_string.encode()).hexdigest()

    def _process_new_bar(self, df: pd.DataFrame) -> None:
        """Process new bar data - only process COMPLETE bars (strict > comparison)."""
        print(f"  Processing bars... Last known timestamp: {self.last_timestamp}")

        # Only process bars AFTER last_timestamp (strict inequality)
        # This ensures we only trade on complete bars, never incomplete ones
        df = df[df['timestamp'] > self.last_timestamp].copy() if self.last_timestamp else df.copy()

        if df.empty:
            print(f"  No new bars (waiting for next complete hourly bar)")
            return

        # Data quality checks
        if df.isnull().any().any():
            print(f"  ⚠️ Data quality check FAILED: NaN values detected, skipping bar")
            print(f"     Columns with NaN: {df.columns[df.isnull().any()].tolist()}")
            return

        if 'close' in df.columns and (df['close'] <= 0).any():
            print(f"  ⚠️ Data quality check FAILED: Invalid prices (<=0) detected, skipping bar")
            print(f"     Invalid tickers: {df[df['close'] <= 0]['tic'].tolist()}")
            return

        if 'volume' in df.columns and (df['volume'] < 0).any():
            print(f"  ⚠️ Data quality check FAILED: Negative volume detected, skipping bar")
            return

        print(f"  Found {len(df)} new bars to process")

        # Get unique timestamps
        timestamps = sorted(df['timestamp'].unique())

        for ts in timestamps:
            # Skip timestamps we've already processed (strict inequality from filter above ensures this)
            ts_data = df[df['timestamp'] == ts]

            # Check if we have all tickers
            if len(ts_data) != len(self.tickers):
                continue

            # Add to historical data
            for symbol in self.tickers:
                symbol_data = ts_data[ts_data['tic'] == symbol]
                self.raw_bars[symbol] = pd.concat([self.raw_bars[symbol], symbol_data], ignore_index=True).tail(1000)

            # Recalculate technical indicators
            all_df = pd.concat(list(self.raw_bars.values()), ignore_index=True)
            all_df = self._add_technical_indicators(all_df)
            all_df = all_df.dropna()

            # Get latest row per ticker - MAINTAIN ORDER from self.tickers!
            latest = all_df.groupby('tic').tail(1)

            # CRITICAL: Sort by ticker ORDER from self.tickers, NOT alphabetically
            # Create a mapping of ticker to its index in self.tickers
            ticker_order = {ticker: i for i, ticker in enumerate(self.tickers)}
            latest = latest.copy()  # Avoid SettingWithCopyWarning
            latest['ticker_order'] = latest['tic'].map(ticker_order)
            latest = latest.sort_values('ticker_order').drop(columns=['ticker_order'])

            if len(latest) != len(self.tickers):
                continue

            # Extract arrays
            price_row = latest['close'].to_numpy(dtype=np.float32)
            tech_cols = ['open', 'high', 'low', 'close', 'volume',
                         'macd', 'macd_signal', 'macd_hist',
                         'rsi', 'cci', 'dx',
                         'atr_regime_shift', 'range_breakout_volume', 'trend_reacceleration']
            tech_row = latest[tech_cols].values.flatten().astype(np.float32)

            # Update environment - append new bar
            self.price_array = np.vstack([self.price_array, price_row])
            self.tech_array = np.vstack([self.tech_array, tech_row])
            self.time_array.append(ts)
            self.last_timestamp = ts

            self.env.price_array = self.price_array
            self.env.tech_array = self.tech_array
            self.env.max_step = self.price_array.shape[0] - self.env.lookback - 1

            # Get agent action
            raw_action = self._select_action(self.state)

            # Apply portfolio-level profit protection FIRST (higher priority)
            action_after_portfolio = self._apply_portfolio_profit_protection(raw_action, price_row, ts)

            # Apply per-position risk management (stop-loss, position limits)
            old_holdings = self.env.stocks.copy()
            action = self._apply_risk_management(action_after_portfolio, price_row, ts)

            # Execute action
            next_state, reward, done, _ = self.env.step(action)
            self.state = next_state.astype(np.float32)

            # Track positions for stop-loss calculation and minimum hold time
            self._update_position_tracking(price_row, old_holdings, self.env.stocks, ts)

            # Update adaptive ensemble scores (if using adaptive mode)
            if hasattr(self, 'adaptive_ensemble') and self.adaptive_ensemble is not None:
                self.adaptive_ensemble.update_scores(price_row, reward)

            # Create snapshot
            snapshot = PortfolioSnapshot(
                timestamp=ts,
                cash=float(self.env.cash),
                holdings={tic: float(h) for tic, h in zip(self.tickers, self.env.stocks)},
                prices={tic: float(p) for tic, p in zip(self.tickers, price_row)},
                total_asset=float(self.env.cash + np.dot(self.env.stocks, price_row)),
                reward=float(reward),
                action=action.tolist(),  # Log the modified action
            )

            self._log_snapshot(snapshot)

            # Send Discord notifications
            if self.discord:
                # Individual trade notifications for significant trades (>$5)
                self._send_trade_notifications(old_holdings, self.env.stocks, price_row, snapshot.total_asset)

                # Hourly summary notification (always sent)
                self._send_hourly_summary(ts, snapshot, old_holdings, self.env.stocks, price_row)

            if done:
                self.env.time = max(self.env.time - 1, self.env.lookback)

    def _select_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from agent."""
        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_tensor = self.act(state_tensor)
        action = action_tensor.cpu().numpy()[0]

        # Log ensemble votes if using ensemble
        if hasattr(self, 'ensemble') and self.ensemble is not None:
            try:
                import json
                from datetime import datetime, timezone
                votes = self.ensemble.get_voting_breakdown(ticker_names=self.tickers)
                votes['timestamp'] = datetime.now(timezone.utc).isoformat()
                votes_path = Path("paper_trades/ensemble_votes.json")
                with votes_path.open("w") as f:
                    json.dump(votes, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
            except Exception as e:
                pass  # Don't crash if logging fails

        return action

    def _send_trade_notifications(self, old_holdings: np.ndarray, new_holdings: np.ndarray,
                                   prices: np.ndarray, portfolio_value: float) -> None:
        """Send Discord notifications for significant trades."""
        if not self.discord or not DISCORD.NOTIFY_TRADES:
            return

        # Detect significant position changes (>1% change in quantity)
        for i, ticker in enumerate(self.tickers):
            old_qty = old_holdings[i]
            new_qty = new_holdings[i]
            price = prices[i]

            qty_change = new_qty - old_qty

            # Only notify on significant changes (> $5 or >1% of position)
            trade_value = abs(qty_change * price)
            if trade_value < 5.0:
                continue

            # Determine action type
            if qty_change > 0:
                action = "BUY"
            elif qty_change < 0:
                action = "SELL"
            else:
                continue

            # Send notification
            try:
                self.discord.send_trade_notification(
                    symbol=ticker,
                    action=action,
                    quantity=abs(qty_change),
                    price=price,
                    portfolio_value=portfolio_value
                )
            except Exception as e:
                # Don't crash on notification failures
                print(f"⚠️  Discord notification failed: {e}")

    def _send_hourly_summary(self, timestamp: datetime, snapshot, old_holdings: np.ndarray,
                             new_holdings: np.ndarray, prices: np.ndarray) -> None:
        """Send hourly summary notification to Discord."""
        if not self.discord:
            return

        try:
            # Calculate total position changes
            trades_executed = []
            total_trade_value = 0.0

            for i, ticker in enumerate(self.tickers):
                old_qty = old_holdings[i]
                new_qty = new_holdings[i]
                price = prices[i]
                qty_change = new_qty - old_qty

                if abs(qty_change) > 0.0001:
                    action = "BUY" if qty_change > 0 else "SELL"
                    trade_value = abs(qty_change * price)
                    total_trade_value += trade_value
                    trades_executed.append({
                        'ticker': ticker,
                        'action': action,
                        'qty': abs(qty_change),
                        'price': price,
                        'value': trade_value
                    })

            # Count active positions
            active_positions = sum(1 for h in new_holdings if h > 0.0001)

            # Format time
            time_str = timestamp.strftime('%Y-%m-%d %H:%M UTC')

            # Determine color based on performance
            initial_capital = 500.0
            total_return_pct = (snapshot.total_asset - initial_capital) / initial_capital * 100
            color = 0x00ff00 if total_return_pct >= 0 else 0xff0000

            # Build fields
            fields = [
                {"name": "💰 Portfolio Value", "value": f"${snapshot.total_asset:.2f}", "inline": True},
                {"name": "💵 Cash", "value": f"${snapshot.cash:.2f}", "inline": True},
                {"name": "📊 Return", "value": f"{total_return_pct:+.2f}%", "inline": True},
                {"name": "📦 Active Positions", "value": str(active_positions), "inline": True},
                {"name": "💎 Reward", "value": f"{snapshot.reward:.2f}", "inline": True},
            ]

            # Add trade summary
            if trades_executed:
                # Enhanced trade summary showing both delta and new position
                trade_lines = []
                for i, t in enumerate(trades_executed):
                    # Find new position for this ticker
                    ticker_idx = self.tickers.index(t['ticker'])
                    new_position = new_holdings[ticker_idx]
                    trade_lines.append(
                        f"{t['action']} {t['qty']:.4f} {t['ticker']} @ ${t['price']:.2f} → {new_position:.4f} held"
                    )
                trade_summary = "\n".join(trade_lines)
                fields.append({
                    "name": "🔄 Trades This Hour",
                    "value": trade_summary,
                    "inline": False
                })
            else:
                fields.append({
                    "name": "🔄 Trades This Hour",
                    "value": "No trades executed (HODL)",
                    "inline": False
                })

            # Send embed
            self.discord.send_message(
                content="",
                embed={
                    "title": f"📈 Hourly Trading Summary",
                    "description": time_str,
                    "color": color,
                    "fields": fields,
                    "timestamp": timestamp.isoformat(),
                    "footer": {"text": f"Trial #{getattr(self, 'trial_number', 'unknown')}"}
                }
            )

        except Exception as e:
            print(f"⚠️  Discord hourly summary failed: {e}")
            import traceback
            traceback.print_exc()

    def _apply_risk_management(self, action: np.ndarray, prices: np.ndarray, ts: datetime) -> np.ndarray:
        """Apply risk management rules to modify actions.

        Rules applied:
        1. Stop-loss: Force sell if position down X% from entry
        2. Position limits: Cap buy actions if position would exceed max %
        3. Action dampening: Scale all actions by dampening factor
        """
        modified_action = action.copy()
        total_asset = self.env.cash + np.dot(self.env.stocks, prices)

        for i, tic in enumerate(self.tickers):
            current_price = prices[i]
            current_holdings = self.env.stocks[i]
            position_value = current_holdings * current_price
            position_pct = position_value / total_asset if total_asset > 0 else 0

            # Update high water mark for trailing stops
            if tic in self.entry_prices and current_holdings > 0:
                if tic not in self.high_water_mark or current_price > self.high_water_mark[tic]:
                    self.high_water_mark[tic] = current_price

            # 0. MINIMUM HOLD TIME CHECK (for sells)
            if modified_action[i] < 0 and current_holdings > 0 and self.risk_mgmt.min_trade_interval_hours > 0:
                if tic in self.last_trade_time:
                    time_held = (ts - self.last_trade_time[tic]).total_seconds() / 3600  # hours
                    if time_held < self.risk_mgmt.min_trade_interval_hours:
                        # Not held long enough - prevent sell (unless stop-loss)
                        # We'll allow stop-loss to override this later
                        modified_action[i] = 0
                        if i == 0 or np.abs(action[i]) > 0.01:  # Only log first asset or significant actions
                            print(f"  ⏱️  Min hold time: {tic} held for {time_held:.1f}h < {self.risk_mgmt.min_trade_interval_hours}h, blocking sell")

            # 1. STOP-LOSS CHECK
            if current_holdings > 0 and tic in self.entry_prices and self.risk_mgmt.stop_loss_pct > 0:
                entry_price = self.entry_prices[tic]
                loss_pct = (entry_price - current_price) / entry_price

                # Check fixed stop-loss
                if loss_pct >= self.risk_mgmt.stop_loss_pct:
                    # Force sell everything
                    modified_action[i] = -current_holdings * 1.1  # Sell all + buffer
                    self.stop_loss_triggered[tic] = True
                    print(f"  ⚠️  STOP-LOSS triggered for {tic}: down {loss_pct*100:.1f}% from entry ${entry_price:.2f}")
                    continue

                # Check trailing stop (if enabled)
                if self.risk_mgmt.trailing_stop_pct > 0 and tic in self.high_water_mark:
                    high = self.high_water_mark[tic]
                    trail_loss = (high - current_price) / high
                    if trail_loss >= self.risk_mgmt.trailing_stop_pct:
                        modified_action[i] = -current_holdings * 1.1
                        self.stop_loss_triggered[tic] = True
                        print(f"  ⚠️  TRAILING STOP triggered for {tic}: down {trail_loss*100:.1f}% from high ${high:.2f}")
                        continue

            # 2. POSITION LIMIT CHECK (for buy orders)
            # NOTE: This operates on un-scaled actions and serves as a first-pass filter.
            # The authoritative concentration limit check happens in the environment after
            # action scaling (environment_Alpaca.py:302-337) to ensure actual trade quantities
            # respect the limit. This pre-check may catch extreme cases early.
            if modified_action[i] > 0 and self.risk_mgmt.max_position_pct > 0:
                # Calculate what position would be after this buy
                buy_value = modified_action[i] * current_price
                new_position_value = position_value + buy_value
                new_total = total_asset  # Total doesn't change on buy (cash → holdings)
                new_position_pct = new_position_value / new_total if new_total > 0 else 0

                if new_position_pct > self.risk_mgmt.max_position_pct:
                    # Cap the buy to stay within limit
                    max_additional_value = (self.risk_mgmt.max_position_pct * total_asset) - position_value
                    if max_additional_value > 0:
                        max_additional_shares = max_additional_value / current_price
                        old_action = modified_action[i]
                        modified_action[i] = min(modified_action[i], max_additional_shares)
                        if modified_action[i] < old_action * 0.5:  # Significantly reduced
                            print(f"  📊 Position limit (pre-check): {tic} buy capped from {old_action:.4f} to {modified_action[i]:.4f} (at {position_pct*100:.0f}% limit)")
                    else:
                        modified_action[i] = 0  # Already at or over limit

            # 3. Clear stop-loss flag on new buy
            if modified_action[i] > 0 and tic in self.stop_loss_triggered and self.stop_loss_triggered[tic]:
                print(f"  ℹ️  Clearing stop-loss flag for {tic} on new position")
                self.stop_loss_triggered[tic] = False

        # 4. APPLY ACTION DAMPENING (scales all actions)
        if self.risk_mgmt.action_dampening != 1.0:
            modified_action = modified_action * self.risk_mgmt.action_dampening

        # 5. APPLY MACRO-AWARE POSITION SIZING (Tiburtina integration)
        if self.tiburtina_bridge and self.tiburtina_bridge.is_available():
            try:
                # Get macro regime and position size multiplier
                status = self.tiburtina_bridge.get_status()
                self.macro_regime = status.get('regime', 'unknown')

                # Use total_asset as reference size for multiplier calculation
                _, macro_reason = self.tiburtina_bridge.get_position_size_multiplier(total_asset)

                # Get the multiplier by calculating the ratio
                test_size = 1000.0
                adjusted_size, _ = self.tiburtina_bridge.get_position_size_multiplier(test_size)
                macro_multiplier = adjusted_size / test_size

                # Update instance variables for logging
                self.macro_multiplier = macro_multiplier
                self.macro_reason = macro_reason

                # Apply macro scaling to all actions
                if macro_multiplier != 1.0:
                    old_action = modified_action.copy()
                    modified_action = modified_action * macro_multiplier

                    # Log macro adjustment
                    has_significant_action = np.any(np.abs(modified_action) > 0.01)
                    if has_significant_action:
                        print(f"  🌍 Macro Regime: {self.macro_regime.upper()} → Position sizing: {macro_multiplier:.1%} ({macro_reason})")

            except Exception as e:
                # Fail gracefully - don't break trading if Tiburtina has issues
                print(f"  ⚠️  Macro adjustment failed: {e}")

        # 6. NEWS-BASED ADJUSTMENTS (Tiburtina integration)
        if self.tiburtina_bridge and (self.tiburtina_bridge.is_available() or self.tiburtina_bridge.alpaca_api):
            try:
                # Check news for buy actions only (be cautious with new positions)
                for i, tic in enumerate(self.tickers):
                    if modified_action[i] > 0:  # Only check buy orders
                        # Check pre-trade news
                        news_check = self.tiburtina_bridge.check_pre_trade_news(tic)

                        if news_check['recommendation'] == 'skip':
                            # Cancel trade due to bearish news
                            print(f"  📰 NEWS ALERT: Skipping {tic} trade")
                            print(f"     Reason: {news_check['reason']}")
                            if news_check['bearish_signals']:
                                print(f"     Signals: {news_check['bearish_signals'][0]}")
                            modified_action[i] = 0  # Cancel the buy

                        elif news_check['recommendation'] == 'reduce':
                            # Reduce position size by 50% due to mixed news
                            old_val = modified_action[i]
                            modified_action[i] = modified_action[i] * 0.5
                            print(f"  📰 NEWS CAUTION: Reducing {tic} position by 50%")
                            print(f"     Reason: {news_check['reason']}")
                            if news_check['bearish_signals']:
                                print(f"     Signal: {news_check['bearish_signals'][0]}")

                        elif news_check['has_news'] and news_check['bullish_signals']:
                            # Optionally log bullish news (don't increase size, just inform)
                            print(f"  📈 NEWS: Bullish signals for {tic} - {news_check['reason']}")

            except Exception as e:
                # Fail gracefully - don't break trading if news check fails
                print(f"  ⚠️  News check failed: {e}")

        return modified_action

    def _update_position_tracking(self, prices: np.ndarray, old_holdings: np.ndarray, new_holdings: np.ndarray, ts: datetime) -> None:
        """Update entry price and trade time tracking after trades."""
        for i, tic in enumerate(self.tickers):
            old_qty = old_holdings[i]
            new_qty = new_holdings[i]
            current_price = prices[i]

            # Position increased (buy)
            if new_qty > old_qty:
                if old_qty <= 0:
                    # New position - set entry price and trade time
                    self.entry_prices[tic] = current_price
                    self.high_water_mark[tic] = current_price
                    self.last_trade_time[tic] = ts
                else:
                    # Adding to position - average entry price, update trade time
                    old_value = old_qty * self.entry_prices.get(tic, current_price)
                    add_value = (new_qty - old_qty) * current_price
                    self.entry_prices[tic] = (old_value + add_value) / new_qty
                    self.last_trade_time[tic] = ts

            # Position closed
            elif new_qty <= 0 and old_qty > 0:
                if tic in self.entry_prices:
                    exit_price = current_price
                    entry = self.entry_prices[tic]
                    pnl_pct = (exit_price - entry) / entry * 100
                    print(f"  💰 Closed {tic}: entry ${entry:.2f} → exit ${exit_price:.2f} ({pnl_pct:+.1f}%)")
                    del self.entry_prices[tic]
                if tic in self.high_water_mark:
                    del self.high_water_mark[tic]
                if tic in self.last_trade_time:
                    del self.last_trade_time[tic]

    def _log_profit_protection(self, message: str, ts: datetime) -> None:
        """Log profit protection events to dedicated file."""
        self.portfolio_protection_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.portfolio_protection_log_path.open("a") as f:
            f.write(f"[{ts.isoformat()}] {message}\n")
        print(f"  🛡️  {message}")

    def _apply_portfolio_profit_protection(
        self, action: np.ndarray, prices: np.ndarray, ts: datetime
    ) -> np.ndarray:
        """Apply portfolio-level profit protection rules.

        Rules applied in order:
        1. Check if in cash mode cooldown - block all buys
        2. Portfolio trailing stop - sell all if portfolio drops X% from peak
        3. Move-to-cash - sell all if portfolio hits high target
        4. Profit-taking - sell partial if portfolio hits target

        Returns modified action array.
        """
        modified_action = action.copy()
        current_portfolio = self.env.cash + np.dot(self.env.stocks, prices)

        # Initialize initial portfolio value on first call
        if self.initial_portfolio_value is None:
            self.initial_portfolio_value = current_portfolio
            self.portfolio_high_water_mark = current_portfolio
            self._log_profit_protection(
                f"Initial portfolio value: ${current_portfolio:.2f}", ts
            )

        # Update portfolio high water mark
        if current_portfolio > self.portfolio_high_water_mark:
            old_hwm = self.portfolio_high_water_mark
            self.portfolio_high_water_mark = current_portfolio
            gain_pct = (current_portfolio / self.initial_portfolio_value - 1) * 100
            if gain_pct >= 1.0:  # Only log significant milestones
                self._log_profit_protection(
                    f"New portfolio high: ${current_portfolio:.2f} (+{gain_pct:.1f}% from start)", ts
                )

        # Calculate current metrics
        gain_from_start = (current_portfolio / self.initial_portfolio_value - 1) if self.initial_portfolio_value > 0 else 0
        drawdown_from_peak = (self.portfolio_high_water_mark - current_portfolio) / self.portfolio_high_water_mark if self.portfolio_high_water_mark > 0 else 0

        # 1. CHECK CASH MODE COOLDOWN
        if self.in_cash_mode:
            if self.cash_mode_started is not None:
                hours_in_cash = (ts - self.cash_mode_started).total_seconds() / 3600
                if hours_in_cash < self.risk_mgmt.cooldown_after_cash_hours:
                    # Still in cooldown - block all buys
                    for i in range(len(modified_action)):
                        if modified_action[i] > 0:
                            modified_action[i] = 0
                    remaining_hours = self.risk_mgmt.cooldown_after_cash_hours - hours_in_cash
                    if int(hours_in_cash) % 4 == 0:  # Log every 4 hours
                        print(f"  ⏸️  Cash mode: {remaining_hours:.1f}h remaining before re-entry allowed")
                    return modified_action
                else:
                    # Cooldown expired - exit cash mode
                    self._log_profit_protection(
                        f"Cash mode cooldown expired after {hours_in_cash:.1f}h - trading resumed", ts
                    )
                    self.in_cash_mode = False
                    self.cash_mode_started = None
                    # Reset profit-taken flag to allow new profit protection cycle
                    self.profit_taken = False
                    # Reset initial value to current (new cycle)
                    self.initial_portfolio_value = current_portfolio
                    self.portfolio_high_water_mark = current_portfolio

        # 2. PORTFOLIO TRAILING STOP
        if self.risk_mgmt.portfolio_trailing_stop_pct > 0:
            if drawdown_from_peak >= self.risk_mgmt.portfolio_trailing_stop_pct:
                # Trigger portfolio-wide sell
                self._log_profit_protection(
                    f"PORTFOLIO TRAILING STOP: Down {drawdown_from_peak*100:.1f}% from peak "
                    f"(${self.portfolio_high_water_mark:.2f} → ${current_portfolio:.2f})", ts
                )
                # Force sell all positions
                for i, tic in enumerate(self.tickers):
                    if self.env.stocks[i] > 0:
                        modified_action[i] = -self.env.stocks[i] * 1.1  # Sell all + buffer
                return modified_action

        # 3. MOVE-TO-CASH (highest priority target)
        if self.risk_mgmt.move_to_cash_threshold_pct > 0:
            if gain_from_start >= self.risk_mgmt.move_to_cash_threshold_pct and not self.in_cash_mode:
                self._log_profit_protection(
                    f"MOVE TO CASH: Portfolio up {gain_from_start*100:.1f}% - liquidating all positions "
                    f"(target was {self.risk_mgmt.move_to_cash_threshold_pct*100:.1f}%)", ts
                )
                # Sell everything
                for i, tic in enumerate(self.tickers):
                    if self.env.stocks[i] > 0:
                        modified_action[i] = -self.env.stocks[i] * 1.1
                # Enter cash mode
                self.in_cash_mode = True
                self.cash_mode_started = ts
                return modified_action

        # 4. PARTIAL PROFIT-TAKING
        if self.risk_mgmt.profit_take_threshold_pct > 0:
            if gain_from_start >= self.risk_mgmt.profit_take_threshold_pct and not self.profit_taken:
                sell_pct = self.risk_mgmt.profit_take_amount_pct
                self._log_profit_protection(
                    f"PROFIT TAKING: Portfolio up {gain_from_start*100:.1f}% - selling {sell_pct*100:.0f}% of positions "
                    f"(target was {self.risk_mgmt.profit_take_threshold_pct*100:.1f}%)", ts
                )
                # Sell portion of each position
                for i, tic in enumerate(self.tickers):
                    if self.env.stocks[i] > 0:
                        sell_qty = self.env.stocks[i] * sell_pct
                        modified_action[i] = -sell_qty
                self.profit_taken = True
                return modified_action

        return modified_action

    def _log_snapshot(self, snap: PortfolioSnapshot) -> None:
        """Log portfolio snapshot to CSV."""
        row = [
            snap.timestamp.isoformat(),
            f"{snap.cash:.6f}",
            f"{snap.total_asset:.6f}",
            f"{snap.reward:.6f}",
        ]
        row += [f"{snap.holdings[tic]:.6f}" for tic in self.tickers]
        row += [f"{snap.prices[tic]:.6f}" for tic in self.tickers]
        row += [f"{a:.6f}" for a in snap.action]

        with self.log_file.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)

        # Log position state (entry prices, stop-loss levels) to JSON
        self._log_position_state(snap)

        print(
            f"[{snap.timestamp.isoformat()}] cash={snap.cash:.2f} total={snap.total_asset:.2f} "
            f"reward={snap.reward:.6f} actions={[round(a, 4) for a in snap.action]}"
        )

    def _log_position_state(self, snap: PortfolioSnapshot) -> None:
        """Log detailed position state including entry prices and stop-loss levels."""
        import json

        positions = []
        for tic in self.tickers:
            holdings = snap.holdings.get(tic, 0)
            current_price = snap.prices.get(tic, 0)

            if holdings > 0:  # Only log open positions
                entry_price = self.entry_prices.get(tic, current_price)
                high_water = self.high_water_mark.get(tic, current_price)

                # Calculate stop-loss trigger price
                stop_loss_price = entry_price * (1 - self.risk_mgmt.stop_loss_pct) if self.risk_mgmt.stop_loss_pct > 0 else 0

                # Calculate trailing stop trigger price (if enabled)
                trailing_stop_price = high_water * (1 - self.risk_mgmt.trailing_stop_pct) if self.risk_mgmt.trailing_stop_pct > 0 else 0

                # Calculate P&L
                pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                position_value = holdings * current_price

                # Distance to stop-loss
                distance_to_stop = ((current_price - stop_loss_price) / current_price * 100) if stop_loss_price > 0 else 0

                positions.append({
                    "ticker": tic,
                    "holdings": float(holdings),
                    "current_price": float(current_price),
                    "entry_price": float(entry_price),
                    "position_value": float(position_value),
                    "pnl_pct": float(pnl_pct),
                    "stop_loss_price": float(stop_loss_price),
                    "trailing_stop_price": float(trailing_stop_price),
                    "high_water_mark": float(high_water),
                    "distance_to_stop_pct": float(distance_to_stop),
                    "stop_loss_triggered": self.stop_loss_triggered.get(tic, False),
                    "last_trade_time": self.last_trade_time[tic].isoformat() if tic in self.last_trade_time else None,
                })

        state = {
            "timestamp": snap.timestamp.isoformat(),
            "portfolio_value": float(snap.total_asset),
            "cash": float(snap.cash),
            "positions": positions,
            "portfolio_protection": {
                "initial_value": float(self.initial_portfolio_value) if self.initial_portfolio_value else None,
                "high_water_mark": float(self.portfolio_high_water_mark),
                "in_cash_mode": self.in_cash_mode,
                "profit_taken": self.profit_taken,
            }
        }

        # Write to JSON file
        positions_file = Path("paper_trades/positions_state.json")
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        with positions_file.open("w") as f:
            json.dump(state, f, indent=2)

    def _write_heartbeat(self, next_poll_time: Optional[datetime] = None) -> None:
        """Write heartbeat file for watchdog staleness detection.

        The watchdog monitors this file to detect if the paper trader is
        actually running and making decisions, not just existing as a process.

        Args:
            next_poll_time: When the next poll is scheduled (for dashboard countdown)
        """
        import json
        self.poll_count += 1
        now = datetime.now(timezone.utc)
        self.last_poll_completed = now

        heartbeat = {
            "timestamp": now.isoformat(),
            "last_poll_completed": now.isoformat(),
            "next_poll_scheduled": next_poll_time.isoformat() if next_poll_time else None,
            "poll_count": self.poll_count,
            "poll_interval": self.poll_interval,
            "timeframe": self.timeframe,
            "tickers": self.tickers,
            "pid": os.getpid(),
            "status": "running",
        }
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        with self.heartbeat_path.open("w") as f:
            json.dump(heartbeat, f, indent=2)

    def run(self):
        """Start polling for new data."""
        print(f"\nStarting Alpaca {'paper' if self.paper else 'live'} trading (REST API polling)...")
        print(f"Tickers: {self.tickers}")
        print(f"Poll interval: {self.poll_interval}s")

        # Show per-position risk management status
        if self.risk_mgmt.stop_loss_pct > 0 or self.risk_mgmt.max_position_pct > 0:
            print(f"\n📊 Per-Position Risk Management:")
            if self.risk_mgmt.max_position_pct > 0:
                print(f"   Max position: {self.risk_mgmt.max_position_pct*100:.0f}% per asset")
            if self.risk_mgmt.stop_loss_pct > 0:
                print(f"   Stop-loss: {self.risk_mgmt.stop_loss_pct*100:.0f}% from entry")
            if self.risk_mgmt.trailing_stop_pct > 0:
                print(f"   Trailing stop: {self.risk_mgmt.trailing_stop_pct*100:.0f}% from high")
        else:
            print(f"\n⚠️  No per-position risk management active")

        # Show portfolio profit protection status
        has_profit_protection = (
            self.risk_mgmt.portfolio_trailing_stop_pct > 0 or
            self.risk_mgmt.profit_take_threshold_pct > 0 or
            self.risk_mgmt.move_to_cash_threshold_pct > 0
        )
        if has_profit_protection:
            print(f"\n🛡️  Portfolio Profit Protection:")
            if self.risk_mgmt.portfolio_trailing_stop_pct > 0:
                print(f"   Portfolio trailing stop: {self.risk_mgmt.portfolio_trailing_stop_pct*100:.1f}% from peak")
            if self.risk_mgmt.profit_take_threshold_pct > 0:
                print(f"   Profit-taking: sell {self.risk_mgmt.profit_take_amount_pct*100:.0f}% when up {self.risk_mgmt.profit_take_threshold_pct*100:.1f}%")
            if self.risk_mgmt.move_to_cash_threshold_pct > 0:
                print(f"   Move-to-cash: liquidate when up {self.risk_mgmt.move_to_cash_threshold_pct*100:.1f}%")
                print(f"   Cooldown after cash: {self.risk_mgmt.cooldown_after_cash_hours}h")
        else:
            print(f"\n⚠️  No portfolio profit protection active")

        print("\nPress Ctrl+C to stop.\n")

        # Send startup notification to Discord
        if self.discord:
            trial_num = getattr(self, 'trial_number', 'unknown')
            risk_desc = []
            if self.risk_mgmt.max_position_pct > 0:
                risk_desc.append(f"Max position: {self.risk_mgmt.max_position_pct*100:.0f}%")
            if self.risk_mgmt.stop_loss_pct > 0:
                risk_desc.append(f"Stop-loss: {self.risk_mgmt.stop_loss_pct*100:.0f}%")

            self.discord.send_message(
                content="🚀 **Paper Trader Started**",
                embed={
                    "title": "Paper Trading Session Initialized",
                    "color": 0x00ff00,
                    "fields": [
                        {"name": "Trial", "value": f"#{trial_num}", "inline": True},
                        {"name": "Tickers", "value": ", ".join(self.tickers), "inline": False},
                        {"name": "Timeframe", "value": self.timeframe, "inline": True},
                        {"name": "Mode", "value": "Paper" if self.paper else "Live", "inline": True},
                        {"name": "Risk Management", "value": "\n".join(risk_desc) if risk_desc else "None", "inline": False}
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        def stop_handler(signum, frame):
            print("\nReceived interrupt, stopping...")
            self._stop = True

        signal.signal(signal.SIGINT, stop_handler)

        try:
            while not self._stop:
                try:
                    print(f"[{datetime.now(timezone.utc).isoformat()}] Polling for new bars...")
                    df = self._fetch_latest_bars()
                    if df is not None:
                        print(f"  Fetched {len(df)} raw bars")
                        self._process_new_bar(df)
                    else:
                        print("  No data returned from API")
                except Exception as e:
                    print(f"Error processing bars: {e}")
                    import traceback
                    traceback.print_exc()

                    # Send error alert to Discord
                    if self.discord and DISCORD.NOTIFY_ALERTS:
                        try:
                            error_details = f"{type(e).__name__}: {str(e)}"
                            self.discord.send_alert(
                                message="Paper trader encountered an error",
                                level="error",
                                details=error_details
                            )
                        except:
                            pass  # Don't crash on notification failures

                # Check for hot-reload of ensemble models
                if hasattr(self, 'adaptive_ensemble') and self.adaptive_ensemble is not None:
                    self.adaptive_ensemble.check_for_reload()

                # Smart scheduling: Align polls to top of hour + 2 minutes
                # This ensures we process bars within 2 minutes of completion
                now = datetime.now(timezone.utc)

                # For 1h timeframe, poll at :02 past the hour
                # For other timeframes, use simple interval
                if self.timeframe == "1h":
                    # Calculate next hour boundary
                    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                    # Add 2-minute offset to ensure bar is complete
                    next_poll = next_hour + timedelta(minutes=2)
                    sleep_seconds = (next_poll - now).total_seconds()

                    # If next poll is very soon (<60s), skip to the following hour
                    if sleep_seconds < 60:
                        next_poll = next_poll + timedelta(hours=1)
                        sleep_seconds = (next_poll - now).total_seconds()

                    print(f"  Next poll at {next_poll.strftime('%H:%M:%S UTC')} (in {sleep_seconds/60:.1f} minutes)")
                else:
                    # Simple interval for non-hourly timeframes
                    next_poll = now + timedelta(seconds=self.poll_interval)
                    sleep_seconds = self.poll_interval
                    print(f"  Sleeping for {sleep_seconds}s until next poll...")

                # Write heartbeat with next poll time for dashboard
                self._write_heartbeat(next_poll_time=next_poll)

                time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            # Sentiment service uses daemon thread, will auto-cleanup
            if hasattr(self, 'sentiment_service') and self.sentiment_service is not None:
                print("\n✓ Sentiment service will auto-cleanup (daemon thread)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpaca paper trading with DRL agent (polling mode)")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to trained model directory")
    parser.add_argument("--tickers", nargs="+", default=["BTC/USD", "ETH/USD", "LTC/USD"], help="Crypto tickers")
    parser.add_argument("--timeframe", type=str, default="5m", help="Bar timeframe (1m, 5m, 15m, 1h, 1d)")
    parser.add_argument("--history-hours", type=int, default=120, help="Hours of historical data to bootstrap")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between API polls")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id (-1 for CPU)")
    parser.add_argument("--log-file", type=Path, default=Path("paper_trades/alpaca_session.csv"))
    parser.add_argument("--live", action="store_true", help="Use live trading (default: paper)")
    parser.add_argument("--enable-sentiment", action="store_true", help="Enable real sentiment analysis (default: zeros)")

    # Per-position risk management arguments
    parser.add_argument("--max-position-pct", type=float, default=0.30,
                        help="Max %% of portfolio in single asset (default: 0.30 = 30%%)")
    parser.add_argument("--stop-loss-pct", type=float, default=0.10,
                        help="Stop-loss threshold (default: 0.10 = 10%% loss from entry)")
    parser.add_argument("--trailing-stop-pct", type=float, default=0.0,
                        help="Trailing stop threshold (default: 0 = disabled)")
    parser.add_argument("--action-dampening", type=float, default=1.0,
                        help="Multiply all actions by this factor (default: 1.0)")
    parser.add_argument("--no-risk-management", action="store_true",
                        help="Disable all risk management (use raw model actions)")

    # Portfolio-level profit protection arguments
    parser.add_argument("--portfolio-trailing-stop-pct", type=float, default=0.015,
                        help="Sell ALL positions if portfolio drops X%% from peak (default: 0.015 = 1.5%%)")
    parser.add_argument("--profit-take-threshold-pct", type=float, default=0.03,
                        help="Take partial profits when portfolio up X%% (default: 0.03 = 3%%)")
    parser.add_argument("--profit-take-amount-pct", type=float, default=0.5,
                        help="Sell X%% of positions when taking profits (default: 0.5 = 50%%)")
    parser.add_argument("--move-to-cash-threshold-pct", type=float, default=0.0,
                        help="Go 100%% cash when up X%% (default: 0 = disabled)")
    parser.add_argument("--cooldown-after-cash-hours", type=int, default=24,
                        help="Hours to wait before re-entering after move-to-cash (default: 24)")
    parser.add_argument("--no-profit-protection", action="store_true",
                        help="Disable all portfolio-level profit protection")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure risk management
    if args.no_risk_management and args.no_profit_protection:
        risk_mgmt = RiskManagement(
            max_position_pct=0.0,
            stop_loss_pct=0.0,
            trailing_stop_pct=0.0,
            action_dampening=1.0,
            portfolio_trailing_stop_pct=0.0,
            profit_take_threshold_pct=0.0,
            move_to_cash_threshold_pct=0.0,
        )
        print("⚠️  ALL risk management DISABLED - using raw model actions")
    else:
        # Per-position settings
        max_pos = 0.0 if args.no_risk_management else args.max_position_pct
        stop_loss = 0.0 if args.no_risk_management else args.stop_loss_pct
        trailing = 0.0 if args.no_risk_management else args.trailing_stop_pct
        dampening = 1.0 if args.no_risk_management else args.action_dampening

        # Portfolio-level settings
        portfolio_trailing = 0.0 if args.no_profit_protection else args.portfolio_trailing_stop_pct
        profit_take = 0.0 if args.no_profit_protection else args.profit_take_threshold_pct
        profit_take_amt = args.profit_take_amount_pct
        move_to_cash = 0.0 if args.no_profit_protection else args.move_to_cash_threshold_pct
        cooldown = args.cooldown_after_cash_hours

        risk_mgmt = RiskManagement(
            max_position_pct=max_pos,
            stop_loss_pct=stop_loss,
            trailing_stop_pct=trailing,
            action_dampening=dampening,
            portfolio_trailing_stop_pct=portfolio_trailing,
            profit_take_threshold_pct=profit_take,
            profit_take_amount_pct=profit_take_amt,
            move_to_cash_threshold_pct=move_to_cash,
            cooldown_after_cash_hours=cooldown,
        )

        # Display per-position risk management
        if not args.no_risk_management:
            print(f"✓ Per-position risk management:")
            print(f"  • Max position: {max_pos*100:.0f}% per asset")
            print(f"  • Stop-loss: {stop_loss*100:.0f}% from entry")
            if trailing > 0:
                print(f"  • Trailing stop: {trailing*100:.0f}% from high")
            if dampening != 1.0:
                print(f"  • Action dampening: {dampening:.2f}x")
        else:
            print("⚠️  Per-position risk management DISABLED")

        # Display portfolio profit protection
        if not args.no_profit_protection:
            print(f"\n🛡️  Portfolio profit protection:")
            if portfolio_trailing > 0:
                print(f"  • Portfolio trailing stop: {portfolio_trailing*100:.1f}% from peak")
            if profit_take > 0:
                print(f"  • Profit-taking: sell {profit_take_amt*100:.0f}% when up {profit_take*100:.1f}%")
            if move_to_cash > 0:
                print(f"  • Move-to-cash: 100% liquidation when up {move_to_cash*100:.1f}%")
                print(f"  • Cooldown after cash: {cooldown}h before re-entry")
        else:
            print("\n⚠️  Portfolio profit protection DISABLED")

    trader = AlpacaPaperTraderPolling(
        tickers=args.tickers,
        timeframe=args.timeframe,
        model_dir=args.model_dir,
        history_hours=args.history_hours,
        poll_interval=args.poll_interval,
        gpu_id=args.gpu,
        log_file=args.log_file,
        paper=not args.live,
        enable_sentiment=args.enable_sentiment,
        risk_management=risk_mgmt,
    )

    trader.run()


if __name__ == "__main__":
    main()
