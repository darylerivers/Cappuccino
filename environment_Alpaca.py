"""The CryptoEnvAlpaca class is a custom environment for trading multiple cryptocurrencies with respect to the Alpaca
trading platform. It is initialized with a configuration dictionary containing the price and technical indicator
arrays, and a dictionary of environment parameters such as the lookback period and normalization constants. The
environment also has several class variables such as the initial capital, buy and sell costs, and the discount factor.

The class has several methods such as reset(), step(), _generate_action_normalizer(),
and _get_state() for interacting with the environment. The reset() method resets the environment to the initial state,
the step() method takes in an action and returns the next state, reward, and done.
The _generate_action_normalizer() method generates the normalizer for the action,
and the _get_state() method returns the current state of the environment.

The environment also has several class variables such as the initial capital, buy and sell costs, and the discount
factor."""

import numpy as np
import math
from config_main import ALPACA_LIMITS
from constants import TRADING
from typing import Optional

# Step 2: Fee Tier and Timeframe Constraint imports
try:
    from fee_tier_manager import FeeTierManager
    from timeframe_constraint import TimeFrameConstraint
    FEE_TIER_AVAILABLE = True
except ImportError:
    FEE_TIER_AVAILABLE = False
    FeeTierManager = None
    TimeFrameConstraint = None


class CryptoEnvAlpaca:  # custom env
    def __init__(self, config, env_params, initial_capital=None,
                 buy_cost_pct=0.0025, sell_cost_pct=0.0025, gamma=0.99, if_log=False,
                 sentiment_service=None, use_sentiment=False, tickers=None,
                 # Step 2: Fee tier and timeframe constraint parameters
                 use_dynamic_fees=False, fee_mode='progressive', fee_interval='1h',
                 use_timeframe_constraint=False, timeframe=None, data_interval='1h'):

        self.if_log = if_log
        self.env_params = env_params
        self.lookback = env_params['lookback']
        # Use INITIAL_CAPITAL from constants if not explicitly provided
        if initial_capital is None:
            initial_capital = TRADING.INITIAL_CAPITAL
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct  # Base fee (used if dynamic fees disabled)
        self.sell_cost_pct = sell_cost_pct  # Base fee (used if dynamic fees disabled)
        self.gamma = gamma

        # Sentiment features
        self.sentiment_service = sentiment_service
        self.use_sentiment = use_sentiment and sentiment_service is not None
        self.tickers = tickers or []

        # Step 2: Fee Tier Manager (optional, dynamic fees based on volume)
        self.use_dynamic_fees = use_dynamic_fees and FEE_TIER_AVAILABLE
        if self.use_dynamic_fees:
            self.fee_tier_manager = FeeTierManager(
                interval=fee_interval,
                mode=fee_mode,
                initial_volume=0.0
            )
        else:
            self.fee_tier_manager = None

        # Step 2: Timeframe Constraint (optional, force liquidation at deadline)
        self.use_timeframe_constraint = use_timeframe_constraint and FEE_TIER_AVAILABLE
        if self.use_timeframe_constraint:
            if timeframe is None:
                raise ValueError("timeframe must be specified when use_timeframe_constraint=True")
            self.timeframe_constraint = TimeFrameConstraint(
                timeframe=timeframe,
                interval=data_interval,
                lookback=self.lookback
            )
        else:
            self.timeframe_constraint = None


        # Get initial price array to compute eqw
        self.price_array = config['price_array']
        self.tech_array = config['tech_array']

        # Validate array shapes
        if self.price_array.shape[0] <= self.lookback:
            raise ValueError(
                f"Price array length ({self.price_array.shape[0]}) must be > lookback ({self.lookback}). "
                f"Need at least {self.lookback + 1} samples for environment initialization."
            )
        if self.tech_array.shape[0] != self.price_array.shape[0]:
            raise ValueError(
                f"Tech array length ({self.tech_array.shape[0]}) must match price array length ({self.price_array.shape[0]})"
            )

        self.prices_initial = list(self.price_array[0, :])
        # Equal-weight portfolio - account for initial purchase fees
        # If buying $X worth of stock, we pay $X * (1 + fee), so we can buy fewer shares
        # Handle zero prices (can happen in extreme CGE stress scenarios)
        self.equal_weight_stock = np.array([
            (self.initial_cash / len(self.prices_initial)) / (self.prices_initial[i] * (1 + self.buy_cost_pct))
            if self.prices_initial[i] > 0 else 0.0
            for i in range(len(self.prices_initial))
        ])

        # read normalization of cash, stocks and tech
        self.norm_cash = env_params['norm_cash']
        self.norm_stocks = env_params['norm_stocks']
        self.norm_tech = env_params['norm_tech']
        self.norm_reward = env_params['norm_reward']
        self.norm_action = env_params['norm_action']
        self.time_decay_floor = float(env_params.get('time_decay_floor', 0.0))

        # Risk management parameters
        self.min_cash_reserve = float(env_params.get('min_cash_reserve', 0.0))  # 0.0 = no constraint
        self.concentration_penalty = float(env_params.get('concentration_penalty', 0.0))  # 0.0 = no penalty

        # Trailing stop loss parameters
        self.trailing_stop_pct = float(env_params.get('trailing_stop_pct', 0.0))  # 0.0 = disabled
        self.use_trailing_stop = self.trailing_stop_pct > 0.0

        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - self.lookback - 1
        self._timeline_offset = self.lookback - 1
        effective_horizon = self.max_step - self._timeline_offset
        self._trading_horizon = max(effective_horizon, 1)

        # reset
        self.time = self._timeline_offset
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.stocks_cooldown = None
        self.safety_factor_stock_buy = 1 - 0.01

        # Trailing stop loss tracking
        self.highest_price_since_buy = np.zeros(self.crypto_num, dtype=np.float32)
        self.trailing_stop_triggered = np.zeros(self.crypto_num, dtype=bool)

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.total_asset_eqw = np.sum(self.equal_weight_stock * self.price_array[self.time])

        # Calculate equal-weight portfolio's initial purchase fee (one-time cost)
        eqw_initial_investment = np.sum(self.equal_weight_stock * self.prices_initial)
        self.eqw_initial_fee = eqw_initial_investment * self.buy_cost_pct

        self.episode_return = 0.0
        self.gamma_return = 0.0

        # Fee tracking
        self.total_fees_paid = 0.0
        self.buy_fees_paid = 0.0
        self.sell_fees_paid = 0.0
        self.num_buy_trades = 0
        self.num_sell_trades = 0

        '''env information'''
        self.env_name = 'MulticryptoEnv'
        self.env_num = 1  # Single environment (not vectorized)

        # state_dim = cash[1,1] + stocks[1,n_crypto] + tech_array[1,n_features] * lookback
        # + sentiment_features[1, n_crypto * 4] (if use_sentiment)
        self.state_dim = 1 + self.price_array.shape[1] + self.tech_array.shape[1] * self.lookback
        if self.use_sentiment:
            # Add 4 sentiment features per ticker
            self.state_dim += self.crypto_num * 4
        self.action_dim = self.price_array.shape[1]
        if ALPACA_LIMITS.shape[0] < self.crypto_num:
            raise ValueError(
                "ALPACA_LIMITS length is shorter than number of tickers configured "
                f"({ALPACA_LIMITS.shape[0]} < {self.crypto_num})."
            )

        self.minimum_qty_alpaca = ALPACA_LIMITS[: self.crypto_num] * 1.1  # 10 % safety factor
        self.if_discrete = False
        self.target_return = 10**8

    def reset(self) -> np.ndarray:
        self.time = self._timeline_offset
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash  # reset()
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.stocks_cooldown = np.zeros_like(self.stocks)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()

        # Reset trailing stop loss tracking
        self.highest_price_since_buy = np.zeros(self.crypto_num, dtype=np.float32)
        self.trailing_stop_triggered = np.zeros(self.crypto_num, dtype=bool)

        # Reset fee tracking
        self.total_fees_paid = 0.0
        self.buy_fees_paid = 0.0
        self.sell_fees_paid = 0.0
        self.num_buy_trades = 0
        self.num_sell_trades = 0

        state = self.get_state()
        return state

    def _get_current_fees(self) -> tuple[float, float]:
        """
        Get current trading fees (either dynamic or static).

        Returns:
            Tuple of (buy_fee, sell_fee) as decimals
        """
        if self.use_dynamic_fees and self.fee_tier_manager is not None:
            maker_fee, taker_fee = self.fee_tier_manager.get_current_fees()
            # Assume buys are maker orders, sells are taker orders (conservative)
            return (maker_fee, taker_fee)
        else:
            # Use static fees
            return (self.buy_cost_pct, self.sell_cost_pct)

    def _track_trade_volume(self, trade_value: float) -> None:
        """
        Track trade volume for fee tier progression (Step 2).

        Args:
            trade_value: Dollar value of the trade (buy or sell)
        """
        if self.use_dynamic_fees and self.fee_tier_manager is not None:
            self.fee_tier_manager.update_volume(abs(trade_value), self.time)

    def step(self, actions) -> tuple[np.ndarray, float, bool, dict]:
        # Input validation
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions, dtype=np.float32)

        if actions.shape != (self.action_dim,):
            raise ValueError(
                f"Action shape mismatch: expected ({self.action_dim},), got {actions.shape}"
            )

        # Sanity check: clip extreme actions (prevent NaN propagation)
        actions = np.clip(actions, -1000, 1000)

        # Check for NaN/Inf
        if not np.isfinite(actions).all():
            raise ValueError(f"Actions contain NaN or Inf: {actions}")

        self.time += 1

        # Step 2: Check timeframe deadline
        if self.use_timeframe_constraint and self.timeframe_constraint is not None:
            if self.timeframe_constraint.is_deadline_reached(self.time):
                # Force liquidation at deadline
                price = self.price_array[self.time]
                final_value = self.timeframe_constraint.calculate_forced_liquidation_value(
                    self.cash, self.stocks, price
                )
                # Calculate benchmark (HODL) value at current time
                initial_price = self.price_array[self.lookback - 1]
                benchmark_value = self.initial_total_asset * np.mean(price / initial_price)
                final_reward = self.timeframe_constraint.calculate_final_reward(
                    final_value, self.initial_total_asset, benchmark_value
                )
                # Episode ends at deadline
                state = self.get_state()
                return state, final_reward, True, {'deadline_reached': True, 'forced_liquidation': final_value}

        # Step 2: Get current fees (dynamic or static)
        # Update instance variables so all fee calculations use current rates
        self.buy_cost_pct, self.sell_cost_pct = self._get_current_fees()

        # if a stock is held add to its cooldown
        for i in range(len(actions)):
            if self.stocks[i] > 0:
                self.stocks_cooldown[i] += 1

        price = self.price_array[self.time]

        # Update trailing stop loss tracking
        if self.use_trailing_stop:
            for i in range(self.crypto_num):
                if self.stocks[i] > 0:
                    # Update highest price if current price is higher
                    if price[i] > self.highest_price_since_buy[i]:
                        self.highest_price_since_buy[i] = price[i]

                    # Check if trailing stop loss is triggered
                    if self.highest_price_since_buy[i] > 0:
                        stop_loss_price = self.highest_price_since_buy[i] * (1.0 - self.trailing_stop_pct)
                        if price[i] < stop_loss_price and price[i] > 0:
                            # Trigger trailing stop loss - sell entire position
                            sell_num_shares = self.stocks[i]
                            self.stocks[i] = 0
                            self.stocks_cooldown[i] = 0
                            sell_value = price[i] * sell_num_shares
                            fee = sell_value * self.sell_cost_pct
                            self.cash += sell_value * (1 - self.sell_cost_pct)
                            self._track_trade_volume(sell_value)  # Step 2: Track volume
                            self.highest_price_since_buy[i] = 0
                            self.trailing_stop_triggered[i] = True
                            # Track fee
                            self.sell_fees_paid += fee
                            self.total_fees_paid += fee
                            self.num_sell_trades += 1

        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = actions[i] * norm_vector_i

        # ENFORCE CONCENTRATION LIMIT (on scaled actions)
        # This check happens AFTER action scaling to ensure actual trade quantities respect limits
        if hasattr(self, 'max_position_pct') and self.max_position_pct > 0:
            total_asset = self.cash + np.sum(self.stocks * price)

            for i in range(self.action_dim):
                if actions[i] > 0:  # Buy orders only
                    current_position_value = self.stocks[i] * price[i]
                    buy_value = actions[i] * price[i]
                    new_position_value = current_position_value + buy_value
                    new_position_pct = new_position_value / total_asset if total_asset > 0 else 0

                    if new_position_pct > self.max_position_pct:
                        # Cap the buy to stay within concentration limit
                        max_position_value = self.max_position_pct * total_asset
                        max_additional_value = max_position_value - current_position_value

                        # Get ticker name for logging
                        ticker_name = self.tickers[i] if self.tickers and i < len(self.tickers) else f"Asset {i}"

                        if max_additional_value > 0:
                            max_additional_shares = max_additional_value / price[i]
                            old_action = actions[i]
                            actions[i] = min(actions[i], max_additional_shares)

                            # Log if significantly reduced (>20% reduction)
                            if actions[i] < old_action * 0.8:
                                current_pct = current_position_value / total_asset * 100 if total_asset > 0 else 0
                                print(f"  🛡️  Concentration limit: {ticker_name} buy capped from {old_action:.2f} to {actions[i]:.2f} shares")
                                print(f"      Current: {current_pct:.1f}%, Would be: {new_position_pct*100:.1f}%, Limit: {self.max_position_pct*100:.0f}%")
                        else:
                            # Already at or over limit - block the buy
                            if actions[i] > 0.01:  # Only log if it was a significant buy attempt
                                current_pct = current_position_value / total_asset * 100 if total_asset > 0 else 0
                                print(f"  🛡️  Concentration limit: {ticker_name} buy blocked (already at {current_pct:.1f}%, limit {self.max_position_pct*100:.0f}%)")
                            actions[i] = 0

        # Compute actions in dollars
        actions_dollars = actions * price

        # Sell
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        for index in np.where(actions < -self.minimum_qty_alpaca)[0]:

            if self.stocks[index] > 0:

                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])

                    assert sell_num_shares >= 0, "Negative sell!"

                    self.stocks_cooldown[index] = 0
                    self.stocks[index] -= sell_num_shares
                    sell_value = price[index] * sell_num_shares
                    fee = sell_value * self.sell_cost_pct
                    self.cash += sell_value * (1 - self.sell_cost_pct)
                    self._track_trade_volume(sell_value)  # Step 2: Track volume

                    # Track fee
                    self.sell_fees_paid += fee
                    self.total_fees_paid += fee
                    self.num_sell_trades += 1

                    # Reset trailing stop loss tracking if position is fully closed
                    if self.stocks[index] == 0:
                        self.highest_price_since_buy[index] = 0
                        self.trailing_stop_triggered[index] = False

        # FORCE 5% SELL every half day (30 min timeframe -> (24 * 2 / 2) * 30)
        for index in np.where(self.stocks_cooldown >= 48)[0]:
            sell_num_shares = self.stocks[index] * 0.05
            self.stocks_cooldown[index] = 0
            self.stocks[index] -= sell_num_shares
            sell_value = price[index] * sell_num_shares
            fee = sell_value * self.sell_cost_pct
            self.cash += sell_value * (1 - self.sell_cost_pct)
            self._track_trade_volume(sell_value)  # Step 2: Track volume

            # Track fee
            self.sell_fees_paid += fee
            self.total_fees_paid += fee
            self.num_sell_trades += 1

            # Update highest price tracking (position reduced but not closed)
            # No need to reset here as position still exists

        # Buy
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################

        for index in np.where(actions > self.minimum_qty_alpaca)[0]:
            if price[index] > 0:  # Buy only if the price is > 0 (no missing data in this particular date)

                # Apply cash reserve constraint
                reserved_cash = self.initial_cash * self.min_cash_reserve
                available_cash = max(0, self.cash - reserved_cash)

                fee_corrected_asset = available_cash / (1 + self.buy_cost_pct)
                max_stocks_can_buy = (fee_corrected_asset / price[index]) * self.safety_factor_stock_buy
                buy_num_shares = min(max_stocks_can_buy, actions[index])
                buy_num_shares_old = buy_num_shares
                if buy_num_shares < self.minimum_qty_alpaca[index]:
                    buy_num_shares = 0

                # Initialize or update trailing stop loss tracking for new/added positions
                if buy_num_shares > 0:
                    if self.stocks[index] == 0:
                        # New position - set initial highest price
                        self.highest_price_since_buy[index] = price[index]
                        self.trailing_stop_triggered[index] = False
                    # If adding to existing position, keep tracking the existing highest price

                    # Calculate and track fee
                    buy_value = price[index] * buy_num_shares
                    fee = buy_value * self.buy_cost_pct
                    self.buy_fees_paid += fee
                    self.total_fees_paid += fee
                    self.num_buy_trades += 1

                self.stocks[index] += buy_num_shares
                buy_value = price[index] * buy_num_shares
                self.cash -= buy_value * (1 + self.buy_cost_pct)
                self._track_trade_volume(buy_value)  # Step 2: Track volume

        """update time"""
        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        next_total_asset_eqw = np.sum(self.equal_weight_stock * self.price_array[self.time])

        # Difference in portfolio value + cooldown management
        delta_bot = next_total_asset - self.total_asset
        delta_eqw = next_total_asset_eqw - self.total_asset_eqw

        # Reward function with time-decay so late outcomes are down-weighted
        steps_elapsed = max(self.time - self._timeline_offset, 0)
        progress = min(max(steps_elapsed / self._trading_horizon, 0.0), 1.0)
        decay_factor = max(1.0 - progress, self.time_decay_floor)

        # REVISED REWARD: Hybrid approach (alpha + absolute returns)
        # Component 1: Alpha - beating equal-weight benchmark (original)
        alpha_reward = (delta_bot - delta_eqw) * self.norm_reward * decay_factor

        # Component 2: Absolute returns - incentivize positive gains
        absolute_return = delta_bot / self.initial_total_asset  # Normalized by initial capital
        absolute_reward = absolute_return * self.norm_reward * decay_factor

        # Component 3: Cash management - reward holding cash during downtrends
        cash_ratio = self.cash / next_total_asset if next_total_asset > 0 else 0
        market_return = delta_eqw / self.total_asset_eqw if self.total_asset_eqw > 0 else 0
        cash_bonus = 0.0
        if market_return < 0 and cash_ratio > 0.1:  # Market down & holding cash
            cash_bonus = abs(market_return) * cash_ratio * self.norm_reward * 0.1

        # Weighted combination: 50% alpha, 30% absolute, 20% cash management
        reward = 0.5 * alpha_reward + 0.3 * absolute_reward + 0.2 * cash_bonus

        # Fix 4: Enforce minimum cash reserve (penalty for going below threshold)
        if self.min_cash_reserve > 0:
            required_cash = self.initial_cash * self.min_cash_reserve
            cash_shortfall = max(0, required_cash - self.cash)
            if cash_shortfall > 0:
                # Penalize violating cash reserve constraint
                cash_penalty = -(cash_shortfall / self.initial_cash) * self.norm_reward * 0.5
                reward += cash_penalty

        # Apply concentration penalty (REVISED - requires diversification)
        if self.concentration_penalty > 0 and next_total_asset > 0:
            # Calculate position concentrations
            position_values = self.stocks * price
            position_ratios = position_values / next_total_asset

            # Fix 1: Penalize single position concentration (lowered from 60% to 40%)
            max_concentration = position_ratios.max()
            if max_concentration > 0.40:
                concentration_excess = max_concentration - 0.40
                penalty = -self.concentration_penalty * concentration_excess
                reward += penalty

            # Fix 2: Require minimum number of positions (prevent 2-asset concentration exploit)
            active_positions = (position_ratios > 0.01).sum()  # Count positions > 1% of portfolio
            if active_positions < 3:
                diversification_penalty = -self.concentration_penalty * 0.5 * (3 - active_positions)
                reward += diversification_penalty

            # Fix 3: Bonus for balanced diversification
            # Gini coefficient: 0 = perfectly equal, 1 = one position holds all
            if active_positions >= 3:
                sorted_ratios = np.sort(position_ratios[position_ratios > 0.01])
                n = len(sorted_ratios)
                if n > 0:
                    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_ratios)) / (n * np.sum(sorted_ratios)) - (n + 1) / n
                    # Reward lower Gini (more equal distribution)
                    # Gini < 0.3 is well-diversified, > 0.6 is concentrated
                    if gini < 0.4:
                        diversification_bonus = self.concentration_penalty * 0.1 * (0.4 - gini)
                        reward += diversification_bonus

        self.total_asset = next_total_asset
        self.total_asset_eqw = next_total_asset_eqw

        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash

        # Build info dict with fee tracking
        info = {
            'total_fees_paid': float(self.total_fees_paid),
            'buy_fees_paid': float(self.buy_fees_paid),
            'sell_fees_paid': float(self.sell_fees_paid),
            'num_buy_trades': int(self.num_buy_trades),
            'num_sell_trades': int(self.num_sell_trades),
        }

        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash

            # Calculate gross return (what return would be without fees)
            self.gross_return = (self.total_asset + self.total_fees_paid) / self.initial_cash

            # Add episode-end metrics to info
            info['episode_return_net'] = float(self.episode_return)
            info['episode_return_gross'] = float(self.gross_return)
            info['fee_impact_pct'] = float((self.gross_return - self.episode_return) * 100)

            # Log fee statistics if logging enabled
            if self.if_log:
                fee_pct_of_initial = (self.total_fees_paid / self.initial_cash) * 100
                eqw_return = self.total_asset_eqw / self.initial_cash
                print(f"\n{'='*70}")
                print(f"EPISODE COMPLETE - Fee Report")
                print(f"{'='*70}")
                print(f"  Initial Capital:      ${self.initial_cash:,.2f}")
                print(f"  Final Portfolio:      ${self.total_asset:,.2f}")
                print(f"  Net Return:           {(self.episode_return - 1) * 100:+.2f}%")
                print(f"  Total Fees Paid:      ${self.total_fees_paid:,.2f} ({fee_pct_of_initial:.2f}% of initial)")
                print(f"  Gross Return:         {(self.gross_return - 1) * 100:+.2f}% (before fees)")
                print(f"  Fee Impact:           {(self.gross_return - self.episode_return) * 100:.2f}%")
                print(f"  Buy Fees:             ${self.buy_fees_paid:,.2f} ({self.num_buy_trades} trades)")
                print(f"  Sell Fees:            ${self.sell_fees_paid:,.2f} ({self.num_sell_trades} trades)")
                print(f"  Total Trades:         {self.num_buy_trades + self.num_sell_trades}")
                total_trades = self.num_buy_trades + self.num_sell_trades
                avg_fee = self.total_fees_paid / total_trades if total_trades > 0 else 0
                print(f"  Avg Fee per Trade:    ${avg_fee:.2f}")
                print(f"\n  BENCHMARK COMPARISON:")
                print(f"  Equal-Weight Return:  {(eqw_return - 1) * 100:+.2f}%")
                print(f"  Equal-Weight Fee:     ${self.eqw_initial_fee:,.2f} (one-time initial purchase)")
                print(f"  Alpha vs Benchmark:   {((self.episode_return - eqw_return)) * 100:+.2f}%")
                print(f"{'='*70}\n")

        return state, reward, done, info

    def get_state(self):
        state = np.hstack((self.cash * self.norm_cash, self.stocks * self.norm_stocks))
        for i in range(self.lookback):
            tech_i = self.tech_array[self.time - i]
            normalized_tech_i = tech_i * self.norm_tech
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)

        # Add sentiment features if enabled
        if self.use_sentiment:
            sentiment_features = self._get_sentiment_features()
            state = np.hstack((state, sentiment_features)).astype(np.float32)

        return state

    def _get_sentiment_features(self):
        """Get sentiment features for all tickers."""
        if not self.use_sentiment or self.sentiment_service is None:
            return np.zeros(self.crypto_num * 4, dtype=np.float32)

        try:
            sentiment_arrays = self.sentiment_service.get_all_sentiment_arrays()

            # Build feature array in ticker order
            features = []
            for ticker in self.tickers:
                if ticker in sentiment_arrays:
                    features.append(sentiment_arrays[ticker])
                else:
                    # No data for this ticker, use neutral
                    features.append(np.zeros(4, dtype=np.float32))

            return np.concatenate(features)

        except Exception as e:
            if self.if_log:
                print(f"Error getting sentiment features: {e}")
            # Return neutral features on error
            return np.zeros(self.crypto_num * 4, dtype=np.float32)

    def close(self):
        pass

    def __getstate__(self):
        """Support pickling/deepcopy by removing unpicklable sentiment_service."""
        state = self.__dict__.copy()
        # Remove the sentiment service (it has threading locks)
        state['sentiment_service'] = None
        return state

    def __setstate__(self, state):
        """Restore from pickle/deepcopy."""
        self.__dict__.update(state)
        # Sentiment service will remain None after unpickling
        # This is OK because sentiment features are only used during state observation,
        # and we'll return zeros if sentiment_service is None

    def _generate_action_normalizer(self):
        """
        Generate action normalizer using value-based approach.

        OLD BROKEN METHOD (price-order-of-magnitude):
        - Bitcoin ($93k): norm = 0.0001 × norm_action = tiny trades
        - LINK ($13): norm = 0.1 × norm_action = massive trades
        - Created 10,000x bias favoring cheap altcoins

        NEW VALUE-BASED METHOD:
        - Normalizes by: initial_portfolio / (price × norm_action)
        - Action of 1.0 represents ~(initial_portfolio / norm_action) dollars for ANY asset
        - With norm_action=2000: action of 1.0 = $1000/2000 = $0.50 of portfolio value
        - Equal treatment of all assets regardless of price
        """
        price_0 = self.price_array[0]

        # Value-based normalization: action represents fraction of portfolio
        # action_value = action × price × norm_vector
        # We want: norm_vector = initial_portfolio / (price × norm_action)
        # So that: action=1.0 trades (initial_portfolio / norm_action) dollars of any asset
        action_norm_vector = self.initial_total_asset / (price_0 * self.norm_action)

        self.action_norm_vector = np.asarray(action_norm_vector, dtype=np.float32)
