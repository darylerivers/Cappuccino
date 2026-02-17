#!/usr/bin/env python3
"""
Conviction Scoring System

Filters trades by multi-factor conviction scoring to reduce 300+ trades/month to 75-100 high-quality trades.

Key Features:
- Position size scoring (prefer larger positions = higher confidence)
- Sharpe ratio momentum (recent performance trend)
- Ensemble agreement (if using multiple models)
- Volatility regime filtering (avoid high-volatility periods)
- Portfolio context (diversification needs, cash levels)

Usage:
    scorer = ConvictionScorer(min_score=0.6)
    score = scorer.score_trade(action, state, model_output)
    if score >= scorer.min_score:
        execute_trade(action)
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from datetime import datetime


@dataclass
class TradeSignal:
    """Trade signal with conviction score."""
    ticker: str
    action: float  # -1 to +1 (sell to buy)
    conviction: float  # 0 to 1
    factors: Dict[str, float]  # Individual factor scores
    timestamp: str


class ConvictionScorer:
    """
    Multi-factor conviction scoring system.

    Filters trades to only execute high-conviction signals.
    Target: 75-100 trades/month (2-3 trades/day) instead of 300+ trades/month (10+ trades/day).
    """

    def __init__(
        self,
        min_score: float = 0.6,  # Minimum conviction to trade
        position_size_weight: float = 0.3,
        sharpe_momentum_weight: float = 0.2,
        ensemble_agreement_weight: float = 0.2,
        volatility_filter_weight: float = 0.15,
        portfolio_context_weight: float = 0.15,
        recent_sharpe_window: int = 50,  # Bars for Sharpe momentum
        high_volatility_threshold: float = 0.03,  # 3% hourly vol = avoid
    ):
        self.min_score = min_score

        # Factor weights (should sum to 1.0)
        self.weights = {
            'position_size': position_size_weight,
            'sharpe_momentum': sharpe_momentum_weight,
            'ensemble_agreement': ensemble_agreement_weight,
            'volatility_filter': volatility_filter_weight,
            'portfolio_context': portfolio_context_weight,
        }

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        self.recent_sharpe_window = recent_sharpe_window
        self.high_volatility_threshold = high_volatility_threshold

        # State tracking
        self.recent_returns = deque(maxlen=recent_sharpe_window)
        self.recent_trades = deque(maxlen=100)  # Last 100 trades
        self.trade_count_by_hour = {}  # Hour -> count

        self.logger = logging.getLogger(__name__)

    def score_trade(
        self,
        ticker: str,
        action: float,  # Model output: -1 to +1
        state: Dict,  # Environment state
        model_outputs: Optional[Dict] = None,  # For ensemble models
    ) -> TradeSignal:
        """
        Score a potential trade across multiple factors.

        Returns TradeSignal with conviction score and factor breakdown.
        """

        factors = {}

        # Factor 1: Position Size Score
        # Larger actions = higher model confidence
        factors['position_size'] = self._score_position_size(action)

        # Factor 2: Sharpe Momentum Score
        # Recent performance trend
        factors['sharpe_momentum'] = self._score_sharpe_momentum()

        # Factor 3: Ensemble Agreement Score
        # If multiple models agree, higher conviction
        factors['ensemble_agreement'] = self._score_ensemble_agreement(
            ticker, action, model_outputs
        )

        # Factor 4: Volatility Filter Score
        # Avoid trading in extreme volatility
        factors['volatility_filter'] = self._score_volatility_filter(state)

        # Factor 5: Portfolio Context Score
        # Consider portfolio needs (diversification, cash levels)
        factors['portfolio_context'] = self._score_portfolio_context(
            ticker, action, state
        )

        # Weighted sum
        conviction = sum(
            factors[name] * self.weights[name]
            for name in factors
        )

        # Create signal
        signal = TradeSignal(
            ticker=ticker,
            action=action,
            conviction=conviction,
            factors=factors,
            timestamp=datetime.now().isoformat()
        )

        # Log if high conviction
        if conviction >= self.min_score:
            self.logger.info(
                f"HIGH CONVICTION: {ticker} action={action:.3f} "
                f"conviction={conviction:.3f} factors={factors}"
            )

        return signal

    def _score_position_size(self, action: float) -> float:
        """
        Score based on action magnitude.

        Larger absolute values indicate higher model confidence.
        """
        # Map action magnitude to 0-1 score
        # |action| = 0.0 → score = 0.0
        # |action| = 0.5 → score = 0.5
        # |action| = 1.0 → score = 1.0

        magnitude = abs(action)

        # Apply sigmoid-like curve to emphasize strong signals
        # score = 1 / (1 + exp(-k * (magnitude - threshold)))
        threshold = 0.3  # Actions below this get low scores
        k = 10  # Steepness

        score = 1.0 / (1.0 + np.exp(-k * (magnitude - threshold)))

        return score

    def _score_sharpe_momentum(self) -> float:
        """
        Score based on recent Sharpe ratio trend.

        If model is performing well recently, higher conviction.
        If model is struggling, lower conviction.
        """

        if len(self.recent_returns) < 20:
            # Not enough data, return neutral
            return 0.5

        # Calculate rolling Sharpe
        returns = np.array(self.recent_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            sharpe = 0.0
        else:
            sharpe = mean_return / std_return * np.sqrt(252)  # Annualized

        # Map Sharpe to 0-1 score
        # Sharpe < -1.0 → score = 0.0 (performing poorly, avoid trading)
        # Sharpe = 0.0 → score = 0.5 (neutral)
        # Sharpe > +1.0 → score = 1.0 (performing well, trade confidently)

        if sharpe <= -1.0:
            score = 0.0
        elif sharpe >= 1.0:
            score = 1.0
        else:
            # Linear interpolation between -1 and +1
            score = (sharpe + 1.0) / 2.0

        return score

    def _score_ensemble_agreement(
        self,
        ticker: str,
        action: float,
        model_outputs: Optional[Dict]
    ) -> float:
        """
        Score based on ensemble agreement.

        If multiple models agree on direction and magnitude, higher conviction.
        """

        if model_outputs is None or len(model_outputs) <= 1:
            # Single model or no ensemble data, return neutral
            return 0.5

        # Get actions for this ticker from all models
        actions = []
        for model_name, outputs in model_outputs.items():
            if ticker in outputs:
                actions.append(outputs[ticker])

        if len(actions) <= 1:
            return 0.5

        actions = np.array(actions)

        # Calculate agreement metrics
        mean_action = np.mean(actions)
        std_action = np.std(actions)

        # Agreement score based on:
        # 1. Low standard deviation (models agree on magnitude)
        # 2. All same sign (models agree on direction)

        # Direction agreement
        all_positive = np.all(actions > 0)
        all_negative = np.all(actions < 0)
        direction_agreement = 1.0 if (all_positive or all_negative) else 0.0

        # Magnitude agreement (low std = high agreement)
        # std=0.0 → score=1.0, std=0.5 → score=0.0
        magnitude_agreement = max(0.0, 1.0 - std_action * 2.0)

        # Combined
        score = 0.6 * direction_agreement + 0.4 * magnitude_agreement

        return score

    def _score_volatility_filter(self, state: Dict) -> float:
        """
        Score based on current volatility regime.

        Avoid trading in extreme volatility (likely to get whipsawed).
        """

        # Extract recent price volatility from state
        # Assuming state contains price history

        if 'price_history' not in state:
            return 0.5  # Neutral if no data

        prices = state['price_history']

        if len(prices) < 20:
            return 0.5

        # Calculate recent volatility (std of returns)
        returns = np.diff(np.log(prices[-20:]))
        volatility = np.std(returns)

        # Map volatility to score
        # vol < 1% → score = 1.0 (low vol, safe to trade)
        # vol = 2% → score = 0.5 (moderate vol)
        # vol > 3% → score = 0.0 (high vol, avoid)

        if volatility < 0.01:
            score = 1.0
        elif volatility > self.high_volatility_threshold:
            score = 0.0
        else:
            # Linear interpolation
            score = 1.0 - (volatility - 0.01) / (self.high_volatility_threshold - 0.01)

        return score

    def _score_portfolio_context(
        self,
        ticker: str,
        action: float,
        state: Dict
    ) -> float:
        """
        Score based on portfolio needs.

        Higher score if trade improves:
        - Diversification (adding to underweight positions)
        - Cash management (buying when high cash, selling when low cash)
        - Position sizing (avoiding overconcentration)
        """

        # Extract portfolio state
        positions = state.get('positions', {})
        cash = state.get('cash', 0)
        portfolio_value = state.get('portfolio_value', 1.0)

        # Current position weight
        current_position = positions.get(ticker, 0)
        current_weight = (current_position * state.get('prices', {}).get(ticker, 1.0)) / portfolio_value

        # Cash ratio
        cash_ratio = cash / portfolio_value

        # Factor 1: Diversification benefit
        # Higher score if adding to underweight positions or reducing overweight
        num_positions = len([p for p in positions.values() if p > 0])
        target_weight = 1.0 / 7  # 7 tickers, equal weight target

        if action > 0:  # Buy
            # Buying underweight positions is good
            diversification_score = max(0.0, 1.0 - current_weight / target_weight)
        else:  # Sell
            # Selling overweight positions is good
            diversification_score = max(0.0, current_weight / target_weight - 1.0)

        # Factor 2: Cash management
        # Buy when cash high, sell when cash low
        if action > 0:  # Buy
            # Buying with high cash is good
            cash_score = min(1.0, cash_ratio / 0.5)  # 50%+ cash = score 1.0
        else:  # Sell
            # Selling with low cash is good
            cash_score = min(1.0, (1.0 - cash_ratio) / 0.5)

        # Factor 3: Position sizing
        # Avoid trades that would create overconcentration
        if action > 0:  # Buy
            # Would this make position too large?
            estimated_new_weight = current_weight + 0.1  # Assume 10% addition
            if estimated_new_weight > 0.4:  # Would exceed 40% limit
                sizing_score = 0.0
            else:
                sizing_score = 1.0
        else:
            sizing_score = 1.0  # Selling never hurts sizing

        # Combined
        score = (
            0.4 * diversification_score +
            0.3 * cash_score +
            0.3 * sizing_score
        )

        return score

    def update_performance(self, portfolio_return: float):
        """Update recent performance tracking."""
        self.recent_returns.append(portfolio_return)

    def record_trade(self, signal: TradeSignal):
        """Record executed trade."""
        self.recent_trades.append(signal)

        # Update hourly trade count
        hour = datetime.now().strftime("%Y-%m-%d %H")
        self.trade_count_by_hour[hour] = self.trade_count_by_hour.get(hour, 0) + 1

    def get_stats(self) -> Dict:
        """Get conviction scoring statistics."""

        if not self.recent_trades:
            return {
                'total_trades': 0,
                'avg_conviction': 0.0,
                'min_conviction': 0.0,
                'max_conviction': 0.0,
            }

        convictions = [t.conviction for t in self.recent_trades]

        # Calculate trades per day
        hours_with_trades = len(self.trade_count_by_hour)
        days = max(1, hours_with_trades / 24)
        trades_per_day = len(self.recent_trades) / days

        return {
            'total_trades': len(self.recent_trades),
            'avg_conviction': np.mean(convictions),
            'min_conviction': np.min(convictions),
            'max_conviction': np.max(convictions),
            'trades_per_day': trades_per_day,
            'projected_monthly_trades': trades_per_day * 30,
        }


class ConvictionFilter:
    """
    Simple wrapper that filters actions through conviction scoring.

    Usage in trader:
        filter = ConvictionFilter(min_score=0.6)

        raw_actions = model.predict(state)
        filtered_actions = filter.filter_actions(raw_actions, state)

        # filtered_actions only contains high-conviction trades
    """

    def __init__(self, min_score: float = 0.6, **scorer_kwargs):
        self.scorer = ConvictionScorer(min_score=min_score, **scorer_kwargs)
        self.logger = logging.getLogger(__name__)

    def filter_actions(
        self,
        raw_actions: Dict[str, float],  # ticker -> action
        state: Dict,
        model_outputs: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Filter actions through conviction scoring.

        Returns only high-conviction trades.
        """

        filtered = {}
        rejected = []

        for ticker, action in raw_actions.items():
            # Skip very small actions (noise)
            if abs(action) < 0.01:
                continue

            # Score the trade
            signal = self.scorer.score_trade(ticker, action, state, model_outputs)

            if signal.conviction >= self.scorer.min_score:
                # High conviction, keep it
                filtered[ticker] = action
                self.scorer.record_trade(signal)

                self.logger.info(
                    f"✓ PASSED: {ticker} action={action:.3f} "
                    f"conviction={signal.conviction:.3f}"
                )
            else:
                # Low conviction, filter it out
                rejected.append((ticker, action, signal.conviction))

                self.logger.debug(
                    f"✗ FILTERED: {ticker} action={action:.3f} "
                    f"conviction={signal.conviction:.3f} (below {self.scorer.min_score:.2f})"
                )

        # Log summary
        if raw_actions:
            pass_rate = len(filtered) / len(raw_actions) * 100
            self.logger.info(
                f"Conviction Filter: {len(filtered)}/{len(raw_actions)} trades passed "
                f"({pass_rate:.1f}%)"
            )

        return filtered

    def update_performance(self, portfolio_return: float):
        """Update performance tracking."""
        self.scorer.update_performance(portfolio_return)

    def get_stats(self) -> Dict:
        """Get filtering statistics."""
        return self.scorer.get_stats()


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Create filter
    filter = ConvictionFilter(min_score=0.6)

    # Simulate raw model outputs (10 tickers)
    raw_actions = {
        'BTC/USD': 0.8,   # Strong buy
        'ETH/USD': 0.4,   # Moderate buy
        'SOL/USD': 0.1,   # Weak buy
        'AVAX/USD': -0.6, # Strong sell
        'MATIC/USD': -0.2, # Weak sell
        'LINK/USD': 0.05,  # Tiny buy (noise)
        'DOT/USD': 0.3,   # Moderate buy
    }

    # Simulate state
    state = {
        'positions': {
            'BTC/USD': 0.05,
            'ETH/USD': 0.02,
        },
        'prices': {
            'BTC/USD': 40000,
            'ETH/USD': 2500,
            'SOL/USD': 100,
            'AVAX/USD': 20,
            'MATIC/USD': 0.8,
            'LINK/USD': 15,
            'DOT/USD': 7,
        },
        'cash': 5000,
        'portfolio_value': 10000,
        'price_history': np.random.randn(50) * 0.01 + 1.0,  # Simulated prices
    }

    # Filter actions
    filtered_actions = filter.filter_actions(raw_actions, state)

    print(f"\nRaw actions: {len(raw_actions)}")
    print(f"Filtered actions: {len(filtered_actions)}")
    print(f"Pass rate: {len(filtered_actions)/len(raw_actions)*100:.1f}%")

    print("\nFiltered actions:")
    for ticker, action in filtered_actions.items():
        print(f"  {ticker}: {action:.3f}")
