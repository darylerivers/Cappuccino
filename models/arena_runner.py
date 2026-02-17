#!/usr/bin/env python3
"""
Arena Runner - Continuous evaluation service for Model Arena.

Runs hourly to:
1. Fetch latest price data from Alpaca
2. Step all models in the arena
3. Update performance metrics
4. Optionally promote best performer to paper trading
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import alpaca_trade_api as tradeapi
import optuna
import pandas as pd
import stockstats

from model_arena import ModelArena


class ArenaRunner:
    """Service to run the Model Arena on a schedule."""

    def __init__(
        self,
        poll_interval: int = 3600,  # 1 hour default
        auto_promote: bool = False,
        auto_add_models: bool = True,
        top_n_models: int = 20,  # Increased tournament size!
    ):
        self.poll_interval = poll_interval
        self.auto_promote = auto_promote
        self.auto_add_models = auto_add_models
        self.top_n_models = top_n_models

        self.tickers = [
            "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD",
            "LINK/USD", "UNI/USD", "AAVE/USD"
        ]

        # Initialize arena
        self.arena = ModelArena(
            tickers=self.tickers,
            max_models=top_n_models,
            min_evaluation_hours=48,  # 2 days (48 hours)
            promotion_threshold=0.02,  # 2% return
        )

        # Alpaca API
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.api_secret = os.getenv("ALPACA_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET required")

        self.api = tradeapi.REST(
            self.api_key, self.api_secret,
            "https://paper-api.alpaca.markets",
            api_version='v2'
        )

        # Optuna database for getting top models
        self.optuna_db = "sqlite:///databases/optuna_cappuccino.db"
        self.study_name = "cappuccino_3workers_20251102_2325"

        # Setup logging
        self.logger = logging.getLogger("ArenaRunner")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [ArenaRunner] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self.logger.addHandler(handler)

            # Also log to file
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "arena_runner.log")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [ArenaRunner] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self.logger.addHandler(file_handler)

        self._stop = False

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received")
        self._stop = True

    def _fetch_prices(self) -> Optional[Dict[str, float]]:
        """Fetch current prices from Alpaca."""
        prices = {}
        try:
            for ticker in self.tickers:
                bars = self.api.get_crypto_bars(
                    ticker,
                    tradeapi.TimeFrame.Hour,
                    (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ).df

                if not bars.empty:
                    prices[ticker] = float(bars['close'].iloc[-1])

            if len(prices) == len(self.tickers):
                return prices

        except Exception as e:
            self.logger.error(f"Failed to fetch prices: {e}")

        return None

    def _fetch_market_data(self) -> Optional[tuple]:
        """Fetch market data with full technical indicators for model inference."""
        try:
            # Fetch enough history for technical indicators + lookback
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=150)  # Extra buffer for indicator calculation

            # Collect data for all tickers
            dfs = []
            for ticker in self.tickers:
                bars = self.api.get_crypto_bars(
                    ticker,
                    tradeapi.TimeFrame.Hour,
                    start.isoformat(),
                    end.isoformat(),
                ).df

                if bars.empty:
                    self.logger.warning(f"No data for {ticker}")
                    continue

                # Reset index to get timestamp as column
                bars = bars.reset_index()
                bars['tic'] = ticker.replace('/', '')  # Remove / for consistency
                dfs.append(bars[['timestamp', 'tic', 'open', 'high', 'low', 'close', 'volume']])

            if len(dfs) < len(self.tickers):
                self.logger.error(f"Only got data for {len(dfs)}/{len(self.tickers)} tickers")
                return None

            # Combine all ticker data
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values(['timestamp', 'tic']).reset_index(drop=True)

            # Calculate technical indicators using stockstats
            # stockstats column names: macd -> creates 'macd', 'macds' (signal), 'macdh' (histogram)
            tech_indicators_stockstats = ['macd', 'rsi', 'cci', 'dx']

            # Process each ticker separately (stockstats requirement)
            processed_dfs = []
            for ticker in self.tickers:
                ticker_clean = ticker.replace('/', '')
                ticker_df = df[df['tic'] == ticker_clean].copy()

                # Use stockstats to calculate indicators
                stock = stockstats.StockDataFrame.retype(ticker_df)
                for indicator in tech_indicators_stockstats:
                    # Access indicator to trigger calculation (stockstats lazy evaluation)
                    _ = stock[indicator]

                # Rename stockstats columns to match our expected names
                if 'macds' in stock.columns:
                    stock['macd_signal'] = stock['macds']
                if 'macdh' in stock.columns:
                    stock['macd_hist'] = stock['macdh']

                processed_dfs.append(stock)

            # Combine processed data
            df_with_tech = pd.concat(processed_dfs, ignore_index=True)
            df_with_tech = df_with_tech.sort_values(['timestamp', 'tic']).reset_index(drop=True)

            # Fill NaN values (from indicator calculation warmup)
            df_with_tech = df_with_tech.fillna(method='bfill').fillna(method='ffill')

            # Get number of timesteps and tickers
            timestamps = df_with_tech['timestamp'].unique()
            n_tickers = len(self.tickers)

            # Build price_array [timesteps, tickers]
            price_array = np.zeros((len(timestamps), n_tickers), dtype=np.float32)
            for i, ts in enumerate(timestamps):
                ts_data = df_with_tech[df_with_tech['timestamp'] == ts]
                for j, ticker in enumerate(self.tickers):
                    ticker_clean = ticker.replace('/', '')
                    ticker_row = ts_data[ts_data['tic'] == ticker_clean]
                    if not ticker_row.empty:
                        price_array[i, j] = float(ticker_row['close'].iloc[0])

            # Build tech_array [timesteps, features * tickers]
            # Features: open, high, low, close, volume, macd, macd_signal, macd_hist, rsi, cci, dx
            feature_cols = ['open', 'high', 'low', 'close', 'volume',
                           'macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx']
            n_features = len(feature_cols)

            tech_array = np.zeros((len(timestamps), n_features * n_tickers), dtype=np.float32)
            for i, ts in enumerate(timestamps):
                ts_data = df_with_tech[df_with_tech['timestamp'] == ts]
                for j, ticker in enumerate(self.tickers):
                    ticker_clean = ticker.replace('/', '')
                    ticker_row = ts_data[ts_data['tic'] == ticker_clean]
                    if not ticker_row.empty:
                        for k, feat in enumerate(feature_cols):
                            tech_array[i, j * n_features + k] = float(ticker_row[feat].iloc[0])

            self.logger.info(f"Fetched market data: price_array={price_array.shape}, tech_array={tech_array.shape}")
            return price_array, tech_array

        except Exception as e:
            self.logger.error(f"Failed to fetch market data: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _get_top_trials(self) -> List[Dict]:
        """Get top trials from Optuna database."""
        try:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.optuna_db,
            )

            # Get completed trials sorted by value (descending)
            trials = [
                {
                    "trial_id": t.number,
                    "value": t.value,
                    "params": t.params,
                }
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
            ]

            trials.sort(key=lambda x: x["value"], reverse=True)
            return trials[:self.top_n_models * 2]  # Get extra in case some don't have files

        except Exception as e:
            self.logger.error(f"Failed to get trials from Optuna: {e}")
            return []

    def _sync_arena_models(self) -> None:
        """Sync arena with top models from training."""
        if not self.auto_add_models:
            return

        top_trials = self._get_top_trials()
        current_models = set(self.arena.portfolios.keys())

        for trial in top_trials:
            trial_id = trial["trial_id"]
            model_id = f"trial_{trial_id}"

            if model_id in current_models:
                continue

            # Check if model files exist
            trial_dir = Path(f"train_results/cwd_tests/trial_{trial_id}_1h")
            if not (trial_dir / "actor.pth").exists():
                continue

            # Add to arena
            if self.arena.add_model(trial_dir, trial_id, trial["value"]):
                self.logger.info(f"Added {model_id} to arena (value={trial['value']:.6f})")

            if len(self.arena.portfolios) >= self.top_n_models:
                break

    def _check_promotion(self) -> None:
        """Check if any model should be promoted to paper trading."""
        if not self.auto_promote:
            return

        candidate = self.arena.get_promotion_candidate()
        if candidate is None:
            return

        self.logger.info(
            f"Promotion candidate: {candidate['model_id']} "
            f"(return={candidate['return_pct']:.2f}%, sharpe={candidate['sharpe_ratio']:.2f})"
        )

        # Write promotion recommendation to file for auto_model_deployer
        promotion_file = Path("arena_state/promotion_candidate.json")
        promotion_file.parent.mkdir(exist_ok=True)
        with promotion_file.open("w") as f:
            json.dump({
                "model_id": candidate["model_id"],
                "trial_number": candidate["trial_number"],
                "metrics": candidate,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    def _write_heartbeat(self) -> None:
        """Write heartbeat file for monitoring."""
        heartbeat_file = Path("arena_state/.heartbeat")
        heartbeat_file.parent.mkdir(exist_ok=True)
        with heartbeat_file.open("w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "models": len(self.arena.portfolios),
                "status": "running",
            }, f, indent=2)

    def run_once(self) -> bool:
        """Run one iteration of the arena."""
        self.logger.info("Starting arena iteration...")

        # Sync models from training
        self._sync_arena_models()

        if len(self.arena.portfolios) == 0:
            self.logger.warning("No models in arena")
            return False

        # Fetch market data
        data = self._fetch_market_data()
        if data is None:
            self.logger.error("Failed to fetch market data")
            return False

        price_array, tech_array = data

        # Step all models
        results = self.arena.step(price_array, tech_array)

        # Log results
        self.logger.info(f"Stepped {len(results)} models")
        for model_id, result in results.items():
            if "error" in result:
                self.logger.warning(f"  {model_id}: ERROR - {result['error']}")
            else:
                self.logger.info(
                    f"  {model_id}: ${result['value']:.2f} ({result['return_pct']:+.2f}%)"
                )

        # Check for promotion
        self._check_promotion()

        # Write heartbeat
        self._write_heartbeat()

        # Export leaderboard
        leaderboard = self.arena.export_leaderboard(Path("arena_state/leaderboard.txt"))
        self.logger.info("\n" + leaderboard)

        return True

    def run(self) -> None:
        """Run continuous arena evaluation."""
        self.logger.info("=" * 60)
        self.logger.info("MODEL ARENA RUNNER STARTING")
        self.logger.info("=" * 60)
        self.logger.info(f"Poll interval: {self.poll_interval}s")
        self.logger.info(f"Auto-promote: {self.auto_promote}")
        self.logger.info(f"Auto-add models: {self.auto_add_models}")
        self.logger.info(f"Max models: {self.top_n_models}")
        self.logger.info("=" * 60)

        while not self._stop:
            try:
                self.run_once()
            except Exception as e:
                self.logger.error(f"Error in arena iteration: {e}")
                import traceback
                traceback.print_exc()

            if self._stop:
                break

            # Wait until next hour
            now = datetime.now(timezone.utc)
            next_hour = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
            wait_seconds = (next_hour - now).total_seconds()

            self.logger.info(f"Sleeping {wait_seconds/60:.1f} minutes until next iteration...")

            # Sleep in small increments to allow graceful shutdown
            sleep_end = time.time() + wait_seconds
            while time.time() < sleep_end and not self._stop:
                time.sleep(min(60, sleep_end - time.time()))

        self.logger.info("Arena runner stopped")


def main():
    """Run arena runner as a service."""
    import argparse

    parser = argparse.ArgumentParser(description="Arena Runner Service")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--auto-promote", action="store_true", help="Enable auto-promotion")
    parser.add_argument("--interval", type=int, default=3600, help="Poll interval in seconds")
    args = parser.parse_args()

    runner = ArenaRunner(
        poll_interval=args.interval,
        auto_promote=args.auto_promote,
    )

    if args.once:
        runner.run_once()
    else:
        runner.run()


if __name__ == "__main__":
    main()
