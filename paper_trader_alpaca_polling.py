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

import numpy as np
import pandas as pd
import torch

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
    ) -> None:
        self.model_dir = model_dir
        self.timeframe = timeframe
        self.history_hours = history_hours
        self.poll_interval = poll_interval
        self.gpu_id = gpu_id
        self.log_file = log_file
        self.paper = paper
        self.enable_sentiment = enable_sentiment

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

        self.env: Optional[CryptoEnvAlpaca] = None
        self.agent = None
        self.act = None
        self.device = torch.device("cpu")
        self.state: Optional[np.ndarray] = None

        self._stop = False

        # Setup
        self._load_trial_and_agent()
        self._bootstrap_history()
        self._prepare_environment()
        self._warmup_environment()
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
        self.norm_action = float(params["norm_action"])
        self.net_dimension = int(params["net_dimension"])
        self.time_decay_floor = float(params.get("time_decay_floor", 0.0))
        self.min_cash_reserve = float(params.get("min_cash_reserve", 0.1))
        self.concentration_penalty = float(params.get("concentration_penalty", 0.05))

        name_folder = self.trial.user_attrs.get("name_folder")
        self.cwd_path = Path("train_results") / name_folder / "stored_agent"
        if not self.cwd_path.exists():
            raise FileNotFoundError(f"Weights directory not found: {self.cwd_path}")

        # Check actual net_dimension from checkpoint (may differ from trial params)
        actor_checkpoint = self.cwd_path / "actor.pth"
        if actor_checkpoint.exists():
            import torch
            checkpoint_state = torch.load(actor_checkpoint, map_location='cpu', weights_only=True)
            if 'net.0.weight' in checkpoint_state:
                actual_net_dim = checkpoint_state['net.0.weight'].shape[0]
                if actual_net_dim != self.net_dimension:
                    print(f"⚠️  Warning: Trial params say net_dimension={self.net_dimension}, "
                          f"but checkpoint has {actual_net_dim}. Using actual: {actual_net_dim}")
                    self.net_dimension = actual_net_dim

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

        df_list = []
        for tic in self.tickers:
            tic_df = df[df['tic'] == tic].copy()

            # Calculate indicators using TA-Lib
            close = tic_df['close'].values
            high = tic_df['high'].values
            low = tic_df['low'].values

            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            tic_df['macd'] = macd
            tic_df['macd_signal'] = macd_signal
            tic_df['macd_hist'] = macd_hist
            tic_df['rsi'] = talib.RSI(close, timeperiod=14)
            tic_df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            tic_df['dx'] = talib.DX(high, low, close, timeperiod=14)

            df_list.append(tic_df)

        return pd.concat(df_list, ignore_index=True)

    def _df_to_arrays(self, df: pd.DataFrame) -> tuple:
        """Convert dataframe to price_array, tech_array, time_array."""
        # Match training data: 11 indicators per ticker (includes open, close)
        tech_cols = ['open', 'high', 'low', 'close', 'volume', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx']

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

        self.env = CryptoEnvAlpaca(
            env_config, env_params, if_log=True,
            use_sentiment=self.enable_sentiment, sentiment_service=sentiment_service, tickers=self.tickers
        )

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

    def _process_new_bar(self, df: pd.DataFrame) -> None:
        """Process new bar data."""
        print(f"  Processing bars... Last known timestamp: {self.last_timestamp}")

        # Filter to only new timestamps
        df = df[df['timestamp'] > self.last_timestamp].copy()

        if df.empty:
            print(f"  No new bars (all bars <= {self.last_timestamp})")
            return

        print(f"  Found {len(df)} new bars to process")

        # Get unique timestamps
        timestamps = sorted(df['timestamp'].unique())

        for ts in timestamps:
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

            # Get latest row per ticker
            latest = all_df.groupby('tic').tail(1).sort_values('tic')

            if len(latest) != len(self.tickers):
                continue

            # Extract arrays
            price_row = latest['close'].to_numpy(dtype=np.float32)
            tech_cols = ['open', 'high', 'low', 'close', 'volume', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx']
            tech_row = latest[tech_cols].values.flatten().astype(np.float32)

            # Update environment
            self.price_array = np.vstack([self.price_array, price_row])
            self.tech_array = np.vstack([self.tech_array, tech_row])
            self.time_array.append(ts)
            self.last_timestamp = ts

            self.env.price_array = self.price_array
            self.env.tech_array = self.tech_array
            self.env.max_step = self.price_array.shape[0] - self.env.lookback - 1

            # Get agent action
            action = self._select_action(self.state)
            next_state, reward, done, _ = self.env.step(action)
            self.state = next_state.astype(np.float32)

            # Create snapshot
            snapshot = PortfolioSnapshot(
                timestamp=ts,
                cash=float(self.env.cash),
                holdings={tic: float(h) for tic, h in zip(self.tickers, self.env.stocks)},
                prices={tic: float(p) for tic, p in zip(self.tickers, price_row)},
                total_asset=float(self.env.cash + np.dot(self.env.stocks, price_row)),
                reward=float(reward),
                action=action.tolist(),
            )

            self._log_snapshot(snapshot)

            if done:
                self.env.time = max(self.env.time - 1, self.env.lookback)

    def _select_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from agent."""
        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_tensor = self.act(state_tensor)
        return action_tensor.cpu().numpy()[0]

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

        print(
            f"[{snap.timestamp.isoformat()}] cash={snap.cash:.2f} total={snap.total_asset:.2f} "
            f"reward={snap.reward:.6f} actions={[round(a, 4) for a in snap.action]}"
        )

    def run(self):
        """Start polling for new data."""
        print(f"\nStarting Alpaca {'paper' if self.paper else 'live'} trading (REST API polling)...")
        print(f"Tickers: {self.tickers}")
        print(f"Poll interval: {self.poll_interval}s")
        print("Press Ctrl+C to stop.\n")

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

                # Wait for next poll
                print(f"  Sleeping for {self.poll_interval}s until next poll...")
                time.sleep(self.poll_interval)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    )

    trader.run()


if __name__ == "__main__":
    main()
