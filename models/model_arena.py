#!/usr/bin/env python3
"""
Model Arena - Independent model performance tracking system.

Runs multiple DRL models in simulated trading environments to evaluate
their real-world performance before promotion to paper/live trading.

Key features:
- Each model has its own virtual portfolio ($1000 starting)
- No actual trades - just simulations based on price data
- Tracks P&L, Sharpe ratio, max drawdown, win rate
- Auto-promotes best performer to paper trading
- Persists state across restarts
"""

import json
import logging
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from environment_Alpaca import CryptoEnvAlpaca


@dataclass
class MarketBenchmark:
    """Buy-and-hold benchmark portfolio for comparison."""
    name: str
    initial_value: float = 1000.0
    holdings: Dict[str, float] = field(default_factory=dict)
    value_history: List[Tuple[str, float]] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.holdings:
            self.holdings = {}
        if not self.value_history:
            self.value_history = []

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        return sum(
            qty * prices.get(ticker, 0)
            for ticker, qty in self.holdings.items()
        )

    def get_return_pct(self, prices: Dict[str, float]) -> float:
        """Calculate return percentage."""
        current_value = self.get_total_value(prices)
        return ((current_value - self.initial_value) / self.initial_value) * 100

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from value history."""
        if len(self.value_history) < 2:
            return 0.0

        values = [v for _, v in self.value_history]
        peak = values[0]
        max_dd = 0.0

        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        return max_dd * 100

    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from value history."""
        if len(self.value_history) < 10:
            return 0.0

        values = [v for _, v in self.value_history]
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        annual_factor = np.sqrt(24 * 365)
        excess_return = np.mean(returns) - risk_free_rate / (24 * 365)

        return (excess_return / np.std(returns)) * annual_factor

    def get_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio - only penalizes downside volatility."""
        if len(self.value_history) < 10:
            return 0.0

        values = [v for _, v in self.value_history]
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0:
            return 0.0

        target_return = risk_free_rate / (24 * 365)
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            return 999.0

        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))

        if downside_deviation == 0:
            return 0.0

        annual_factor = np.sqrt(24 * 365)
        excess_return = np.mean(returns) - target_return

        return (excess_return / downside_deviation) * annual_factor

    def get_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(self.value_history) < 10:
            return 0.0

        max_dd = self.get_max_drawdown()
        if max_dd == 0:
            return 0.0

        values = [v for _, v in self.value_history]
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0:
            return 0.0

        annual_return = np.mean(returns) * 24 * 365 * 100
        return annual_return / max_dd

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "initial_value": float(self.initial_value),
            "holdings": {k: float(v) for k, v in self.holdings.items()},
            "value_history": [(ts, float(v)) for ts, v in self.value_history[-1000:]],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MarketBenchmark":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            initial_value=data["initial_value"],
            holdings=data.get("holdings", {}),
            value_history=data.get("value_history", []),
            created_at=data["created_at"],
        )


@dataclass
class VirtualPortfolio:
    """Virtual portfolio for a single model."""
    model_id: str
    trial_number: int
    training_value: float  # Optuna objective value from training

    # Portfolio state
    cash: float = 1000.0
    holdings: Dict[str, float] = field(default_factory=dict)
    entry_prices: Dict[str, float] = field(default_factory=dict)

    # Performance tracking
    initial_value: float = 1000.0
    peak_value: float = 1000.0
    total_trades: int = 0
    winning_trades: int = 0

    # Time series for metrics
    value_history: List[Tuple[str, float]] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)

    # Timestamps
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.holdings:
            self.holdings = {}
        if not self.entry_prices:
            self.entry_prices = {}
        if not self.value_history:
            self.value_history = []
        if not self.trade_history:
            self.trade_history = []

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        holdings_value = sum(
            qty * prices.get(ticker, 0)
            for ticker, qty in self.holdings.items()
        )
        return self.cash + holdings_value

    def get_return_pct(self, prices: Dict[str, float]) -> float:
        """Calculate return percentage."""
        current_value = self.get_total_value(prices)
        return ((current_value - self.initial_value) / self.initial_value) * 100

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from value history."""
        if len(self.value_history) < 2:
            return 0.0

        values = [v for _, v in self.value_history]
        peak = values[0]
        max_dd = 0.0

        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        return max_dd * 100  # Return as percentage

    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from value history."""
        if len(self.value_history) < 10:
            return 0.0

        values = [v for _, v in self.value_history]
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Annualize (assuming hourly data)
        annual_factor = np.sqrt(24 * 365)
        excess_return = np.mean(returns) - risk_free_rate / (24 * 365)

        return (excess_return / np.std(returns)) * annual_factor

    def get_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio - better for crypto than Sharpe.
        Only penalizes downside volatility, not upside gains.
        """
        if len(self.value_history) < 10:
            return 0.0

        values = [v for _, v in self.value_history]
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0:
            return 0.0

        # Calculate downside deviation (only negative returns)
        target_return = risk_free_rate / (24 * 365)
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            # No downside - perfect score
            return 999.0  # Cap at high value

        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))

        if downside_deviation == 0:
            return 0.0

        # Annualize
        annual_factor = np.sqrt(24 * 365)
        excess_return = np.mean(returns) - target_return

        return (excess_return / downside_deviation) * annual_factor

    def get_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        Popular in crypto - measures return per unit of worst-case loss.
        """
        if len(self.value_history) < 10:
            return 0.0

        max_dd = self.get_max_drawdown()
        if max_dd == 0:
            return 0.0  # Avoid division by zero

        values = [v for _, v in self.value_history]
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0:
            return 0.0

        # Annualize return (hourly data)
        annual_return = np.mean(returns) * 24 * 365 * 100  # As percentage

        # Calmar = annual_return / max_drawdown
        return annual_return / max_dd

    def get_win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def get_recent_return(self, hours: int = 24, prices: Dict[str, float] = None) -> Optional[float]:
        """
        Calculate return over the last N hours.

        Args:
            hours: Number of hours to look back
            prices: Current prices (for latest value)

        Returns:
            Return percentage over the period, or None if insufficient data
        """
        if len(self.value_history) < 2:
            return None

        # Get current value
        if prices:
            current_value = self.get_total_value(prices)
        else:
            current_value = self.value_history[-1][1]

        # Find value from N hours ago
        target_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Find closest historical value
        past_value = None
        for timestamp_str, value in reversed(self.value_history[:-1]):
            try:
                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if ts <= target_time:
                    past_value = value
                    break
            except:
                continue

        if past_value is None or past_value == 0:
            return None

        return ((current_value - past_value) / past_value) * 100

    def get_recent_volatility(self, hours: int = 168) -> float:
        """
        Calculate volatility (standard deviation of returns) over recent period.

        Args:
            hours: Number of hours to look back (default 7 days)

        Returns:
            Annualized volatility as percentage
        """
        if len(self.value_history) < 10:
            return 0.0

        # Filter to recent history
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_values = []

        for timestamp_str, value in self.value_history:
            try:
                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if ts >= cutoff_time:
                    recent_values.append(value)
            except:
                continue

        if len(recent_values) < 2:
            # Fall back to all history
            recent_values = [v for _, v in self.value_history]

        if len(recent_values) < 2:
            return 0.0

        # Calculate returns
        returns = np.diff(recent_values) / recent_values[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Annualize volatility (assuming hourly data)
        annual_factor = np.sqrt(24 * 365)
        return np.std(returns) * annual_factor * 100

    def get_drawdown_velocity(self) -> float:
        """
        Calculate how fast the portfolio is declining (drawdown per hour).
        Measures recent drawdown rate over last 24 hours.

        Returns:
            Percentage decline per hour (negative = declining)
        """
        if len(self.value_history) < 24:
            return 0.0

        # Get last 24 hours of data
        recent_24h = self.value_history[-24:]

        if len(recent_24h) < 2:
            return 0.0

        # Calculate peak and current
        values = [v for _, v in recent_24h]
        peak = max(values)
        current = values[-1]

        if peak == 0:
            return 0.0

        # Calculate drawdown
        drawdown_pct = ((peak - current) / peak) * 100

        # Velocity = drawdown per hour
        hours_elapsed = len(recent_24h)
        if hours_elapsed == 0:
            return 0.0

        return -drawdown_pct / hours_elapsed if drawdown_pct > 0 else 0.0

    def detect_performance_issues(self, prices: Dict[str, float] = None) -> Dict[str, any]:
        """
        Detect performance issues and outliers.

        Returns:
            Dict with warning flags and severity
        """
        warnings = {
            'severe_decline': False,
            'moderate_decline': False,
            'high_volatility': False,
            'rapid_drawdown': False,
            'severity': 'normal',  # normal, warning, critical
            'messages': []
        }

        # Check 24h return
        return_24h = self.get_recent_return(24, prices)
        if return_24h is not None:
            if return_24h < -5.0:
                warnings['severe_decline'] = True
                warnings['severity'] = 'critical'
                warnings['messages'].append(f"Severe 24h decline: {return_24h:.1f}%")
            elif return_24h < -2.0:
                warnings['moderate_decline'] = True
                if warnings['severity'] == 'normal':
                    warnings['severity'] = 'warning'
                warnings['messages'].append(f"Moderate 24h decline: {return_24h:.1f}%")

        # Check volatility
        volatility = self.get_recent_volatility(168)  # 7 days
        if volatility > 100.0:  # >100% annualized volatility
            warnings['high_volatility'] = True
            if warnings['severity'] == 'normal':
                warnings['severity'] = 'warning'
            warnings['messages'].append(f"High volatility: {volatility:.0f}%")

        # Check drawdown velocity
        dd_velocity = self.get_drawdown_velocity()
        if dd_velocity < -0.5:  # Losing >0.5% per hour
            warnings['rapid_drawdown'] = True
            warnings['severity'] = 'critical'
            warnings['messages'].append(f"Rapid drawdown: {dd_velocity:.2f}%/hr")

        return warnings

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "trial_number": self.trial_number,
            "training_value": float(self.training_value),
            "cash": float(self.cash),
            "holdings": {k: float(v) for k, v in self.holdings.items()},
            "entry_prices": {k: float(v) for k, v in self.entry_prices.items()},
            "initial_value": float(self.initial_value),
            "peak_value": float(self.peak_value),
            "total_trades": int(self.total_trades),
            "winning_trades": int(self.winning_trades),
            "value_history": [(ts, float(v)) for ts, v in self.value_history[-1000:]],  # Keep last 1000 points
            "trade_history": self.trade_history[-100:],  # Keep last 100 trades
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VirtualPortfolio":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            trial_number=data["trial_number"],
            training_value=data["training_value"],
            cash=data["cash"],
            holdings=data.get("holdings", {}),
            entry_prices=data.get("entry_prices", {}),
            initial_value=data["initial_value"],
            peak_value=data["peak_value"],
            total_trades=data["total_trades"],
            winning_trades=data["winning_trades"],
            value_history=data.get("value_history", []),
            trade_history=data.get("trade_history", []),
            created_at=data["created_at"],
            last_updated=data.get("last_updated", ""),
        )


class ModelArena:
    """
    Arena for evaluating multiple DRL models in simulated trading.
    """

    def __init__(
        self,
        tickers: List[str] = None,
        max_models: int = 10,
        state_dir: Path = None,
        min_evaluation_hours: int = 168,  # 7 days minimum before promotion
        promotion_threshold: float = 0.02,  # 2% return threshold
        prune_interval_hours: int = 24,  # How often to prune underperformers
        below_average_threshold: float = 0.25,  # Remove bottom 25% after evaluation
    ):
        self.tickers = tickers or [
            "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD",
            "LINK/USD", "UNI/USD", "AAVE/USD"
        ]
        self.max_models = max_models
        self.state_dir = state_dir or Path("arena_state")
        self.min_evaluation_hours = min_evaluation_hours
        self.promotion_threshold = promotion_threshold
        self.prune_interval_hours = prune_interval_hours
        self.below_average_threshold = below_average_threshold

        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Track last prune time
        self.last_prune_time = datetime.now(timezone.utc)

        # Model storage
        self.portfolios: Dict[str, VirtualPortfolio] = {}
        self.loaded_models: Dict[str, Any] = {}  # Cached model actors

        # Market benchmarks for comparison
        self.benchmarks: Dict[str, MarketBenchmark] = {}

        # Current market data
        self.current_prices: Dict[str, float] = {}
        self.price_history: deque = deque(maxlen=1000)

        # Setup logging
        self.logger = logging.getLogger("ModelArena")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [Arena] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self.logger.addHandler(handler)

        # Load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load arena state from disk."""
        state_file = self.state_dir / "arena_state.json"
        if state_file.exists():
            try:
                with state_file.open() as f:
                    data = json.load(f)

                for model_id, portfolio_data in data.get("portfolios", {}).items():
                    self.portfolios[model_id] = VirtualPortfolio.from_dict(portfolio_data)

                for bench_name, bench_data in data.get("benchmarks", {}).items():
                    self.benchmarks[bench_name] = MarketBenchmark.from_dict(bench_data)

                # Load last prune time
                if "last_prune_time" in data:
                    self.last_prune_time = datetime.fromisoformat(data["last_prune_time"])

                self.logger.info(f"Loaded {len(self.portfolios)} portfolios and {len(self.benchmarks)} benchmarks from state")
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")

        # Initialize benchmarks if not loaded
        self._initialize_benchmarks()

    def _save_state(self) -> None:
        """Save arena state to disk."""
        state_file = self.state_dir / "arena_state.json"
        try:
            data = {
                "portfolios": {
                    model_id: portfolio.to_dict()
                    for model_id, portfolio in self.portfolios.items()
                },
                "benchmarks": {
                    bench_name: benchmark.to_dict()
                    for bench_name, benchmark in self.benchmarks.items()
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "last_prune_time": self.last_prune_time.isoformat(),
            }
            with state_file.open("w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def _initialize_benchmarks(self) -> None:
        """Initialize market benchmark portfolios if they don't exist."""
        if not self.current_prices:
            return  # Need prices to initialize

        # Equal-weight portfolio across all tickers
        if "equal_weight" not in self.benchmarks:
            bench = MarketBenchmark(name="Equal Weight Portfolio")
            value_per_ticker = bench.initial_value / len(self.tickers)
            for ticker in self.tickers:
                price = self.current_prices.get(ticker, 0)
                if price > 0:
                    bench.holdings[ticker] = value_per_ticker / price
            self.benchmarks["equal_weight"] = bench
            self.logger.info("Initialized equal-weight benchmark")

        # BTC-only portfolio
        if "btc_only" not in self.benchmarks and "BTC/USD" in self.tickers:
            bench = MarketBenchmark(name="BTC Only")
            price = self.current_prices.get("BTC/USD", 0)
            if price > 0:
                bench.holdings["BTC/USD"] = bench.initial_value / price
            self.benchmarks["btc_only"] = bench
            self.logger.info("Initialized BTC-only benchmark")

        # 60/40 BTC/ETH portfolio
        if "btc_eth_6040" not in self.benchmarks:
            if "BTC/USD" in self.tickers and "ETH/USD" in self.tickers:
                bench = MarketBenchmark(name="60/40 BTC/ETH")
                btc_price = self.current_prices.get("BTC/USD", 0)
                eth_price = self.current_prices.get("ETH/USD", 0)
                if btc_price > 0 and eth_price > 0:
                    bench.holdings["BTC/USD"] = (bench.initial_value * 0.6) / btc_price
                    bench.holdings["ETH/USD"] = (bench.initial_value * 0.4) / eth_price
                self.benchmarks["btc_eth_6040"] = bench
                self.logger.info("Initialized 60/40 BTC/ETH benchmark")

    def add_model(self, trial_dir: Path, trial_number: int, training_value: float) -> bool:
        """Add a new model to the arena."""
        model_id = f"trial_{trial_number}"

        if model_id in self.portfolios:
            self.logger.info(f"Model {model_id} already in arena")
            return False

        # Check if we need to remove worst performer
        if len(self.portfolios) >= self.max_models:
            self._remove_worst_model()

        # Verify model files exist
        actor_path = trial_dir / "actor.pth"
        if not actor_path.exists():
            self.logger.warning(f"No actor.pth found in {trial_dir}")
            return False

        # Create virtual portfolio
        portfolio = VirtualPortfolio(
            model_id=model_id,
            trial_number=trial_number,
            training_value=training_value,
        )

        self.portfolios[model_id] = portfolio
        self.logger.info(f"Added {model_id} to arena (training_value={training_value:.6f})")

        self._save_state()
        return True

    def _remove_worst_model(self) -> None:
        """Remove the worst performing model from arena."""
        if not self.portfolios:
            return

        # Calculate current returns for all models
        rankings = self.get_rankings()
        if rankings:
            worst = rankings[-1]
            model_id = worst["model_id"]
            del self.portfolios[model_id]
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
            self.logger.info(f"Removed worst performer: {model_id}")

    def _prune_underperformers(self) -> None:
        """
        Remove models performing significantly below average after minimum evaluation period.

        This prevents the arena from getting clogged with poor performers that will never
        get promoted. Models must have at least min_evaluation_hours to be considered for pruning.
        """
        if len(self.portfolios) <= 3:
            # Keep at least 3 models for comparison
            return

        # Get all models that have been evaluated long enough
        rankings = self.get_rankings()
        evaluated_models = [
            m for m in rankings
            if m["eligible_for_promotion"]  # Has enough hours
        ]

        if len(evaluated_models) < 4:
            # Need at least 4 evaluated models to calculate meaningful average
            return

        # Calculate average performance metrics
        avg_return = np.mean([m["return_pct"] for m in evaluated_models])
        avg_sortino = np.mean([m["sortino_ratio"] for m in evaluated_models if m["sortino_ratio"] > -50])

        # Calculate how many to remove (bottom X%)
        num_to_remove = max(1, int(len(evaluated_models) * self.below_average_threshold))

        # Don't remove too many at once - cap at 5
        num_to_remove = min(num_to_remove, 5)

        # Sort by combined performance score (Sortino is primary, return is secondary)
        evaluated_models_sorted = sorted(
            evaluated_models,
            key=lambda m: (m["sortino_ratio"], m["return_pct"])
        )

        # Remove bottom performers
        removed_count = 0
        for model in evaluated_models_sorted[:num_to_remove]:
            model_id = model["model_id"]

            # Only remove if significantly below average
            is_below_avg = (
                model["return_pct"] < avg_return - 1.0 or  # More than 1% below avg return
                model["sortino_ratio"] < avg_sortino - 0.5  # Or 0.5 below avg Sortino
            )

            if is_below_avg:
                del self.portfolios[model_id]
                if model_id in self.loaded_models:
                    del self.loaded_models[model_id]

                self.logger.info(
                    f"Pruned underperformer: {model_id} "
                    f"(return={model['return_pct']:.2f}% vs avg={avg_return:.2f}%, "
                    f"sortino={model['sortino_ratio']:.2f} vs avg={avg_sortino:.2f})"
                )
                removed_count += 1

        if removed_count > 0:
            self.logger.info(f"Pruned {removed_count} underperforming models from arena")
            self._save_state()

    def _check_and_prune(self) -> None:
        """Check if it's time to prune underperformers and do so if needed."""
        now = datetime.now(timezone.utc)
        hours_since_prune = (now - self.last_prune_time).total_seconds() / 3600

        if hours_since_prune >= self.prune_interval_hours:
            self.logger.info(f"Running periodic underperformer pruning ({hours_since_prune:.1f}h since last prune)")
            self._prune_underperformers()
            self.last_prune_time = now

    def force_prune_underperformers(self) -> None:
        """Manually trigger pruning of underperformers (bypasses interval check)."""
        self.logger.info("Manual pruning triggered")
        self._prune_underperformers()
        self.last_prune_time = datetime.now(timezone.utc)

    def _load_model(self, model_id: str) -> Optional[Any]:
        """Load model actor for inference."""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        portfolio = self.portfolios.get(model_id)
        if not portfolio:
            return None

        trial_dir = Path(f"train_results/cwd_tests/trial_{portfolio.trial_number}_1h")
        actor_path = trial_dir / "actor.pth"

        if not actor_path.exists():
            # Try stored_agent subdirectory
            actor_path = trial_dir / "stored_agent" / "actor.pth"
            if not actor_path.exists():
                self.logger.warning(f"Actor not found: {actor_path}")
                return None

        try:
            device = torch.device("cpu")
            # Load state dict
            state_dict = torch.load(actor_path, map_location=device, weights_only=True)

            # Determine dimensions from state dict
            # Actor architecture: input -> net_dim -> net_dim -> action_dim
            # net.0.weight has shape [net_dim, state_dim]
            # net.4.weight has shape [action_dim, net_dim]
            if 'net.0.weight' not in state_dict or 'net.4.weight' not in state_dict:
                self.logger.error(f"Invalid state dict for {model_id}")
                return None

            state_dim = state_dict['net.0.weight'].shape[1]
            net_dim = state_dict['net.0.weight'].shape[0]
            action_dim = state_dict['net.4.weight'].shape[0]

            # Build actor network (matches ElegantRL PPO actor)
            actor = torch.nn.Sequential(
                torch.nn.Linear(state_dim, net_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(net_dim, net_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(net_dim, action_dim),
            )

            # Strip "net." prefix from keys and filter out non-net keys (like a_std_log)
            net_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('net.'):
                    new_key = key[4:]  # Remove "net." prefix
                    net_state_dict[new_key] = value

            # Load weights and set to eval mode
            actor.load_state_dict(net_state_dict)
            actor.to(device)
            actor.eval()

            self.loaded_models[model_id] = actor
            return actor
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return None

    def update_prices(self, prices: Dict[str, float], timestamp: datetime = None) -> None:
        """Update current market prices."""
        self.current_prices = prices
        ts = timestamp or datetime.now(timezone.utc)
        self.price_history.append((ts.isoformat(), prices.copy()))

        # Initialize benchmarks if this is the first time we have prices
        if not self.benchmarks and prices:
            self._initialize_benchmarks()

    def step(self, price_array: np.ndarray, tech_array: np.ndarray = None) -> Dict[str, Dict]:
        """
        Run one step of simulation for all models.

        Args:
            price_array: Historical prices [timesteps, n_tickers]
            tech_array: Historical technical indicators [timesteps, n_features * n_tickers]

        Returns:
            Dict of model_id -> portfolio status
        """
        results = {}
        timestamp = datetime.now(timezone.utc).isoformat()

        # Use latest prices for portfolio valuation
        if price_array.ndim == 1:
            # Old format - just current prices
            current_prices = price_array
        else:
            # New format - historical data
            current_prices = price_array[-1, :]  # Latest timestep

        # Update prices dict
        prices = {ticker: float(current_prices[i]) for i, ticker in enumerate(self.tickers)}
        self.update_prices(prices)

        for model_id, portfolio in self.portfolios.items():
            try:
                # Calculate hours in arena
                created = datetime.fromisoformat(portfolio.created_at.replace('Z', '+00:00'))
                hours_in_arena = (datetime.now(timezone.utc) - created).total_seconds() / 3600

                # Try to run model inference if we have historical data
                if price_array.ndim == 2 and tech_array is not None and tech_array.ndim == 2:
                    # Load model
                    actor = self._load_model(model_id)
                    if actor is not None:
                        try:
                            # Infer lookback from model's state_dim
                            # state_dim = 1 (cash) + n_tickers + tech_features_per_step * lookback
                            n_tickers = len(self.tickers)
                            tech_features_per_step = tech_array.shape[1]  # features * tickers

                            # Get model's expected input dimension
                            state_dim = actor[0].in_features  # First layer input size

                            # Calculate lookback
                            lookback = (state_dim - 1 - n_tickers) // tech_features_per_step
                            lookback = max(1, min(lookback, 5))  # Clamp to reasonable range (most models use 2)

                            # Build state with correct lookback
                            state = self._build_state_with_lookback(
                                portfolio, price_array, tech_array, lookback=lookback
                            )

                            # Run model inference
                            with torch.no_grad():
                                action = actor(torch.FloatTensor(state).unsqueeze(0))
                                action = action.cpu().numpy().flatten()

                            # Execute virtual trade
                            trade_info = self._execute_virtual_trade(
                                portfolio, action, prices, timestamp
                            )

                        except Exception as e:
                            self.logger.error(f"Error running inference for {model_id}: {e}")
                            # Fall through to simple tracking mode

                # Update portfolio value history
                current_value = portfolio.get_total_value(prices)
                portfolio.value_history.append((timestamp, current_value))
                portfolio.peak_value = max(portfolio.peak_value, current_value)
                portfolio.last_updated = timestamp

                results[model_id] = {
                    "value": current_value,
                    "return_pct": portfolio.get_return_pct(prices),
                    "hours": hours_in_arena,
                    "training_value": portfolio.training_value,
                }

            except Exception as e:
                self.logger.error(f"Error stepping {model_id}: {e}")
                results[model_id] = {"error": str(e)}

        # Update benchmarks
        for bench_name, benchmark in self.benchmarks.items():
            current_value = benchmark.get_total_value(prices)
            benchmark.value_history.append((timestamp, current_value))
            results[f"benchmark_{bench_name}"] = {
                "value": current_value,
                "return_pct": benchmark.get_return_pct(prices),
                "is_benchmark": True,
            }

        # Check if it's time to prune underperformers
        self._check_and_prune()

        self._save_state()
        return results

    def _build_state_with_lookback(
        self,
        portfolio: VirtualPortfolio,
        price_array: np.ndarray,
        tech_array: np.ndarray,
        lookback: int = 20
    ) -> np.ndarray:
        """
        Build state vector with lookback matching training environment.

        State structure (matches environment_Alpaca.py get_state):
        [cash, stocks, tech[t], tech[t-1], ..., tech[t-lookback+1]]

        Args:
            portfolio: Virtual portfolio with current cash/holdings
            price_array: Historical prices [timesteps, n_tickers]
            tech_array: Historical tech indicators [timesteps, n_features * n_tickers]
            lookback: Number of historical timesteps to include

        Returns:
            State vector [state_dim]
        """
        n_timesteps = price_array.shape[0]
        n_tickers = len(self.tickers)
        current_prices = price_array[-1, :]  # Latest prices

        # Calculate total portfolio value
        total_value = portfolio.get_total_value(
            {t: current_prices[i] for i, t in enumerate(self.tickers)}
        )

        # Normalize cash (using same norm as training - typically 1/10000)
        norm_cash = 1.0 / 10000.0
        cash_normalized = portfolio.cash * norm_cash

        # Normalize stock holdings (using same norm as training - typically 1/100)
        norm_stocks = 1.0 / 100.0
        stocks_normalized = np.zeros(n_tickers, dtype=np.float32)
        for i, ticker in enumerate(self.tickers):
            qty = portfolio.holdings.get(ticker, 0)
            stocks_normalized[i] = qty * norm_stocks

        # Start building state with cash and stocks
        state = np.concatenate([[cash_normalized], stocks_normalized])

        # Add technical indicators with lookback
        # Use same normalization as training (typically 1/10000 for tech)
        norm_tech = 1.0 / 10000.0

        # Get last 'lookback' timesteps of tech indicators
        if n_timesteps >= lookback:
            for i in range(lookback):
                # Get tech at time t-i (most recent first)
                tech_t = tech_array[-(i+1), :] * norm_tech
                state = np.concatenate([state, tech_t])
        else:
            # Not enough history - pad with zeros or repeat earliest
            for i in range(lookback):
                if i < n_timesteps:
                    tech_t = tech_array[-(i+1), :] * norm_tech
                else:
                    # Pad with earliest available data
                    tech_t = tech_array[0, :] * norm_tech
                state = np.concatenate([state, tech_t])

        return state.astype(np.float32)

    def _build_state(
        self,
        portfolio: VirtualPortfolio,
        price_array: np.ndarray,
        tech_array: np.ndarray
    ) -> np.ndarray:
        """Build state vector for model inference (legacy - use _build_state_with_lookback)."""
        n_tickers = len(self.tickers)

        # Normalize cash and holdings
        total_value = portfolio.get_total_value(
            {t: price_array[i] for i, t in enumerate(self.tickers)}
        )

        cash_norm = portfolio.cash / total_value if total_value > 0 else 1.0

        holdings_norm = np.zeros(n_tickers)
        for i, ticker in enumerate(self.tickers):
            qty = portfolio.holdings.get(ticker, 0)
            holdings_norm[i] = (qty * price_array[i]) / total_value if total_value > 0 else 0

        # Build state: [cash, holdings, prices, tech_indicators]
        # Simplified - actual implementation should match training environment
        state = np.concatenate([
            [cash_norm],
            holdings_norm,
            price_array / price_array.mean() if price_array.mean() > 0 else price_array,
            tech_array.flatten()[:50] if len(tech_array.flatten()) > 50 else tech_array.flatten(),
        ])

        return state.astype(np.float32)

    def _execute_virtual_trade(
        self,
        portfolio: VirtualPortfolio,
        action: np.ndarray,
        prices: Dict[str, float],
        timestamp: str,
    ) -> Optional[Dict]:
        """Execute a virtual trade based on model action."""
        trade_info = None

        for i, ticker in enumerate(self.tickers):
            if i >= len(action):
                break

            act = action[i]
            price = prices.get(ticker, 0)
            if price <= 0:
                continue

            current_qty = portfolio.holdings.get(ticker, 0)

            # Calculate target position change
            # action > 0 = buy, action < 0 = sell
            # Scale action to reasonable trade size (max 20% of portfolio per trade)
            max_trade_value = portfolio.get_total_value(prices) * 0.2

            if act > 0.1:  # Buy signal
                # Calculate buy quantity
                buy_value = min(portfolio.cash * abs(act), max_trade_value)
                buy_qty = buy_value / price

                if buy_qty > 0.0001 and portfolio.cash >= buy_value:
                    portfolio.cash -= buy_value
                    portfolio.holdings[ticker] = current_qty + buy_qty

                    # Update entry price (weighted average)
                    old_value = current_qty * portfolio.entry_prices.get(ticker, price)
                    new_value = buy_qty * price
                    portfolio.entry_prices[ticker] = (old_value + new_value) / (current_qty + buy_qty)

                    portfolio.total_trades += 1
                    trade_info = {
                        "ticker": ticker,
                        "side": "buy",
                        "qty": float(buy_qty),
                        "price": float(price),
                        "timestamp": timestamp,
                    }
                    portfolio.trade_history.append(trade_info)

            elif act < -0.1 and current_qty > 0:  # Sell signal
                # Calculate sell quantity
                sell_qty = min(current_qty * abs(act), current_qty)
                sell_value = sell_qty * price

                if sell_qty > 0.0001:
                    portfolio.cash += sell_value
                    portfolio.holdings[ticker] = current_qty - sell_qty

                    # Check if profitable
                    entry_price = portfolio.entry_prices.get(ticker, price)
                    if price > entry_price:
                        portfolio.winning_trades += 1

                    portfolio.total_trades += 1
                    trade_info = {
                        "ticker": ticker,
                        "side": "sell",
                        "qty": float(sell_qty),
                        "price": float(price),
                        "pnl": float((price - entry_price) * sell_qty),
                        "timestamp": timestamp,
                    }
                    portfolio.trade_history.append(trade_info)

                    # Clean up if position closed
                    if portfolio.holdings[ticker] < 0.0001:
                        portfolio.holdings[ticker] = 0
                        if ticker in portfolio.entry_prices:
                            del portfolio.entry_prices[ticker]

        return trade_info

    def get_rankings(self) -> List[Dict]:
        """Get models ranked by performance."""
        rankings = []

        for model_id, portfolio in self.portfolios.items():
            current_value = portfolio.get_total_value(self.current_prices)
            return_pct = portfolio.get_return_pct(self.current_prices)
            sharpe = portfolio.get_sharpe_ratio()
            sortino = portfolio.get_sortino_ratio()
            calmar = portfolio.get_calmar_ratio()
            max_dd = portfolio.get_max_drawdown()
            win_rate = portfolio.get_win_rate()

            # Calculate hours in arena
            try:
                created = datetime.fromisoformat(portfolio.created_at.replace('Z', '+00:00'))
                hours_active = (datetime.now(timezone.utc) - created).total_seconds() / 3600
            except:
                hours_active = 0

            # Get recent performance metrics
            return_24h = portfolio.get_recent_return(24, self.current_prices)
            return_7d = portfolio.get_recent_return(168, self.current_prices)  # 7 days
            volatility = portfolio.get_recent_volatility()
            dd_velocity = portfolio.get_drawdown_velocity()

            # Detect performance issues
            warnings = portfolio.detect_performance_issues(self.current_prices)

            # Composite score: Use Sortino (better for crypto) + return - drawdown penalty
            # Sortino weighted higher because it's more relevant for asymmetric crypto returns
            score = return_pct + (sortino * 3) + (sharpe * 1) - (max_dd * 0.5)

            rankings.append({
                "model_id": model_id,
                "trial_number": portfolio.trial_number,
                "training_value": portfolio.training_value,
                "current_value": current_value,
                "return_pct": return_pct,
                "return_24h": return_24h,
                "return_7d": return_7d,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "calmar_ratio": calmar,
                "max_drawdown": max_dd,
                "volatility": volatility,
                "drawdown_velocity": dd_velocity,
                "win_rate": win_rate,
                "total_trades": portfolio.total_trades,
                "hours_active": hours_active,
                "score": score,
                "eligible_for_promotion": hours_active >= self.min_evaluation_hours,
                "warnings": warnings,
            })

        # Sort by score (highest first)
        rankings.sort(key=lambda x: x["score"], reverse=True)
        return rankings

    def get_promotion_candidate(self) -> Optional[Dict]:
        """Get the best model eligible for promotion to paper trading."""
        rankings = self.get_rankings()

        for model in rankings:
            if not model["eligible_for_promotion"]:
                continue
            if model["return_pct"] < self.promotion_threshold * 100:
                continue
            # Use Sortino instead of Sharpe for crypto (better metric)
            if model["sortino_ratio"] < 0:
                continue

            return model

        return None

    def get_status(self) -> Dict:
        """Get arena status for dashboard."""
        rankings = self.get_rankings()
        promotion_candidate = self.get_promotion_candidate()

        # Get benchmark stats
        benchmark_stats = {}
        for bench_name, benchmark in self.benchmarks.items():
            benchmark_stats[bench_name] = {
                "name": benchmark.name,
                "return_pct": benchmark.get_return_pct(self.current_prices),
                "sharpe_ratio": benchmark.get_sharpe_ratio(),
                "sortino_ratio": benchmark.get_sortino_ratio(),
                "calmar_ratio": benchmark.get_calmar_ratio(),
                "max_drawdown": benchmark.get_max_drawdown(),
                "current_value": benchmark.get_total_value(self.current_prices),
            }

        return {
            "total_models": len(self.portfolios),
            "max_models": self.max_models,
            "rankings": rankings,
            "benchmarks": benchmark_stats,
            "promotion_candidate": promotion_candidate,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "min_evaluation_hours": self.min_evaluation_hours,
            "promotion_threshold_pct": self.promotion_threshold * 100,
        }

    def export_leaderboard(self, output_path: Path = None) -> str:
        """Export leaderboard as formatted string."""
        rankings = self.get_rankings()

        lines = [
            "=" * 100,
            "MODEL ARENA LEADERBOARD (Crypto-Optimized Metrics)",
            "=" * 100,
            f"{'Rank':<5} {'Model':<13} {'Return %':<10} {'Sortino':<9} {'Sharpe':<8} {'Calmar':<8} {'MaxDD %':<8} {'Trades':<7} {'Hours':<7} {'Status':<8}",
            "-" * 100,
        ]

        for i, model in enumerate(rankings, 1):
            status = "READY" if model["eligible_for_promotion"] and model["return_pct"] > 0 else "EVAL"
            # Cap Sortino/Calmar display at reasonable values for readability
            sortino_display = min(model['sortino_ratio'], 99.9)
            calmar_display = min(model['calmar_ratio'], 99.9) if model['calmar_ratio'] != 0 else 0.0

            lines.append(
                f"{i:<5} {model['model_id']:<13} {model['return_pct']:>8.2f}% "
                f"{sortino_display:>8.2f} {model['sharpe_ratio']:>7.2f} "
                f"{calmar_display:>7.2f} {model['max_drawdown']:>7.2f}% "
                f"{model['total_trades']:>6} {model['hours_active']:>6.0f}h {status:<8}"
            )

        # Add benchmark comparison section
        if self.benchmarks:
            lines.append("=" * 100)
            lines.append("MARKET BENCHMARKS (Buy & Hold)")
            lines.append("=" * 100)
            lines.append(f"{'Benchmark':<30} {'Return %':<12} {'Sortino':<10} {'Sharpe':<10} {'Calmar':<10} {'MaxDD %':<10}")
            lines.append("-" * 100)

            for bench_name, benchmark in self.benchmarks.items():
                return_pct = benchmark.get_return_pct(self.current_prices)
                sharpe = benchmark.get_sharpe_ratio()
                sortino = benchmark.get_sortino_ratio()
                calmar = benchmark.get_calmar_ratio()
                max_dd = benchmark.get_max_drawdown()

                # Cap display values
                sortino_display = min(sortino, 99.9)
                calmar_display = min(calmar, 99.9) if calmar != 0 else 0.0

                lines.append(
                    f"{benchmark.name:<30} {return_pct:>10.2f}% "
                    f"{sortino_display:>9.2f} {sharpe:>9.2f} "
                    f"{calmar_display:>9.2f} {max_dd:>9.2f}%"
                )

        lines.append("=" * 100)

        result = "\n".join(lines)

        if output_path:
            output_path.write_text(result)

        return result


def main():
    """Run arena as standalone service."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Arena - Independent model evaluation")
    parser.add_argument("--show-status", action="store_true", help="Show current arena status")
    parser.add_argument("--add-model", type=int, help="Add a trial to the arena")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard")
    args = parser.parse_args()

    arena = ModelArena()

    if args.show_status:
        status = arena.get_status()
        print(json.dumps(status, indent=2))

    elif args.add_model:
        trial_num = args.add_model
        trial_dir = Path(f"train_results/cwd_tests/trial_{trial_num}_1h")
        # Try to get training value from optuna db
        training_value = 0.005  # Default
        arena.add_model(trial_dir, trial_num, training_value)
        print(f"Added trial {trial_num} to arena")

    elif args.leaderboard:
        print(arena.export_leaderboard())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
