"""
Pipeline Validation Gates
Each gate validates model readiness for the next stage.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple


class BaseGate:
    """Base class for validation gates."""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate(self, trial_number: int, metrics: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate metrics against gate criteria.

        Returns:
            (passed, error_message)
        """
        raise NotImplementedError


class BacktestGate(BaseGate):
    """
    Gate 1: Backtest Validation
    Adaptive thresholds based on total trial count.
    """

    def __init__(self, config: Dict, db_path: str = "databases/optuna_cappuccino.db"):
        super().__init__(config)
        self.db_path = db_path

    def _get_trial_count(self) -> int:
        """Get total number of completed trials."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM trials
                WHERE state = 'COMPLETE'
            """)
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            self.logger.error(f"Failed to get trial count: {e}")
            return 0

    def _get_adaptive_thresholds(self) -> Dict:
        """Get thresholds based on training progress."""
        trial_count = self._get_trial_count()

        thresholds = self.config.get("thresholds", {})

        if trial_count < 50:
            return thresholds.get("early", {
                "max_loss": -0.5,
                "min_sharpe": None,
                "max_drawdown": 0.80
            })
        elif trial_count < 200:
            return thresholds.get("mid", {
                "max_loss": -0.2,
                "min_sharpe": -0.5,
                "max_drawdown": 0.50
            })
        elif trial_count < 500:
            return thresholds.get("late", {
                "max_loss": -0.1,
                "min_sharpe": 0.0,
                "max_drawdown": 0.35
            })
        else:
            return thresholds.get("mature", {
                "max_loss": 0.0,
                "min_sharpe": 0.3,
                "max_drawdown": 0.15
            })

    def validate(self, trial_number: int, metrics: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate backtest metrics.

        Expected metrics:
        - total_return: float
        - sharpe: float
        - max_drawdown: float
        - volatility: float
        """
        if not self.config.get("enabled", True):
            self.logger.info("Backtest gate disabled")
            return True, None

        thresholds = self._get_adaptive_thresholds()
        trial_count = self._get_trial_count()

        self.logger.info(f"Validating trial {trial_number} (total trials: {trial_count})")
        self.logger.info(f"Thresholds: {thresholds}")
        self.logger.info(f"Metrics: {metrics}")

        # Extract metrics
        total_return = metrics.get("total_return", 0)
        sharpe = metrics.get("sharpe", 0)
        max_drawdown = metrics.get("max_drawdown", 1)

        # Check return threshold
        max_loss = thresholds.get("max_loss")
        if max_loss is not None and total_return < max_loss:
            error = f"Total return {total_return*100:.2f}% below threshold {max_loss*100:.1f}%"
            self.logger.warning(error)
            return False, error

        # Check Sharpe threshold
        min_sharpe = thresholds.get("min_sharpe")
        if min_sharpe is not None and sharpe < min_sharpe:
            error = f"Sharpe {sharpe:.3f} below threshold {min_sharpe:.2f}"
            self.logger.warning(error)
            return False, error

        # Check drawdown threshold
        max_dd = thresholds.get("max_drawdown")
        if max_dd is not None and max_drawdown > max_dd:
            error = f"Max drawdown {max_drawdown*100:.2f}% exceeds threshold {max_dd*100:.1f}%"
            self.logger.warning(error)
            return False, error

        self.logger.info(f"Trial {trial_number} PASSED backtest gate")
        return True, None


class CGEStressGate(BaseGate):
    """
    Gate 2: CGE Stress Test Validation
    Validates model robustness across diverse economic scenarios.
    """

    def validate(self, trial_number: int, metrics: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate CGE stress test metrics.

        Expected metrics:
        - median_sharpe: float
        - profitable_pct: float (0-1)
        - max_drawdown: float
        - catastrophic_failures: int
        - total_scenarios: int
        """
        if not self.config.get("enabled", True):
            self.logger.info("CGE stress gate disabled")
            return True, None

        self.logger.info(f"Validating CGE stress test for trial {trial_number}")
        self.logger.info(f"Metrics: {metrics}")

        # Extract metrics
        median_sharpe = metrics.get("median_sharpe", 0)
        profitable_pct = metrics.get("profitable_pct", 0)
        max_drawdown = metrics.get("max_drawdown", 1)
        catastrophic_failures = metrics.get("catastrophic_failures", 0)

        # Check profitable percentage
        min_profitable = self.config.get("min_profitable_pct", 0.40)
        if profitable_pct < min_profitable:
            error = f"Only {profitable_pct*100:.1f}% profitable (need {min_profitable*100:.0f}%)"
            self.logger.warning(error)
            return False, error

        # Check median Sharpe
        min_sharpe = self.config.get("min_median_sharpe", 0.0)
        if median_sharpe < min_sharpe:
            error = f"Median Sharpe {median_sharpe:.3f} below threshold {min_sharpe:.2f}"
            self.logger.warning(error)
            return False, error

        # Check max drawdown
        max_dd = self.config.get("max_drawdown", 0.25)
        if max_drawdown > max_dd:
            error = f"Max drawdown {max_drawdown*100:.2f}% exceeds threshold {max_dd*100:.1f}%"
            self.logger.warning(error)
            return False, error

        # Check catastrophic failures
        max_catastrophic = self.config.get("max_catastrophic_loss", -0.90)
        if catastrophic_failures > 0:
            error = f"{catastrophic_failures} catastrophic failures (loss > {max_catastrophic*100:.0f}%)"
            self.logger.warning(error)
            return False, error

        self.logger.info(f"Trial {trial_number} PASSED CGE stress gate")
        return True, None


class PaperTradingGate(BaseGate):
    """
    Gate 3: Paper Trading Validation
    Validates performance in Model Arena (7+ days).
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.arena_state_path = Path("arena_state/arena_state.json")

    def validate(self, trial_number: int, metrics: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate paper trading metrics from Model Arena.

        Expected metrics:
        - days_evaluated: int
        - win_rate: float
        - sharpe: float
        - max_drawdown: float
        - alpha: float (vs market)
        - emergency_stops: int
        """
        if not self.config.get("enabled", True):
            self.logger.info("Paper trading gate disabled")
            return True, None

        self.logger.info(f"Validating paper trading for trial {trial_number}")
        self.logger.info(f"Metrics: {metrics}")

        # Check evaluation period
        min_days = self.config.get("min_days", 7)
        days_evaluated = metrics.get("days_evaluated", 0)
        if days_evaluated < min_days:
            error = f"Only {days_evaluated} days evaluated (need {min_days})"
            self.logger.warning(error)
            return False, error

        # Check win rate
        min_win_rate = self.config.get("min_win_rate", 0.80)
        win_rate = metrics.get("win_rate", 0)
        if win_rate < min_win_rate:
            error = f"Win rate {win_rate*100:.1f}% below threshold {min_win_rate*100:.0f}%"
            self.logger.warning(error)
            return False, error

        # Check Sharpe
        min_sharpe = self.config.get("min_sharpe", 0.5)
        sharpe = metrics.get("sharpe", 0)
        if sharpe < min_sharpe:
            error = f"Sharpe {sharpe:.3f} below threshold {min_sharpe:.2f}"
            self.logger.warning(error)
            return False, error

        # Check max drawdown
        max_dd = self.config.get("max_drawdown", 0.15)
        max_drawdown = metrics.get("max_drawdown", 1)
        if max_drawdown > max_dd:
            error = f"Max drawdown {max_drawdown*100:.2f}% exceeds threshold {max_dd*100:.1f}%"
            self.logger.warning(error)
            return False, error

        # Check alpha (must be positive)
        alpha = metrics.get("alpha", 0)
        if alpha <= 0:
            error = f"Alpha {alpha*100:.2f}% not positive"
            self.logger.warning(error)
            return False, error

        # Check emergency stops
        emergency_stops = metrics.get("emergency_stops", 0)
        if emergency_stops > 0:
            error = f"{emergency_stops} emergency stops triggered"
            self.logger.warning(error)
            return False, error

        self.logger.info(f"Trial {trial_number} PASSED paper trading gate")
        return True, None
