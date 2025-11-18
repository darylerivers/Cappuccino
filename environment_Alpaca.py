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


class CryptoEnvAlpaca:  # custom env
    def __init__(self, config, env_params, initial_capital=1000,
                 buy_cost_pct=0.0025, sell_cost_pct=0.0025, gamma=0.99, if_log=False,
                 sentiment_service=None, use_sentiment=False, tickers=None):

        self.if_log = if_log
        self.env_params = env_params
        self.lookback = env_params['lookback']
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma

        # Sentiment features
        self.sentiment_service = sentiment_service
        self.use_sentiment = use_sentiment and sentiment_service is not None
        self.tickers = tickers or []


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
        self.equal_weight_stock = np.array([self.initial_cash /
                                            len(self.prices_initial) /
                                            self.prices_initial[i] for i in
                                            range(len(self.prices_initial))])

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

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.total_asset_eqw = np.sum(self.equal_weight_stock * self.price_array[self.time])

        self.episode_return = 0.0
        self.gamma_return = 0.0

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

        state = self.get_state()
        return state

    def step(self, actions) -> (np.ndarray, float, bool, None):
        self.time += 1

        # if a stock is held add to its cooldown
        for i in range(len(actions)):
            if self.stocks[i] > 0:
                self.stocks_cooldown[i] += 1

        price = self.price_array[self.time]
        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = actions[i] * norm_vector_i

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
                    self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

        # FORCE 5% SELL every half day (30 min timeframe -> (24 * 2 / 2) * 30)
        for index in np.where(self.stocks_cooldown >= 48)[0]:
            sell_num_shares = self.stocks[index] * 0.05
            self.stocks_cooldown[index] = 0
            self.stocks[index] -= sell_num_shares
            self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

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
                self.stocks[index] += buy_num_shares
                self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

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
        reward = (delta_bot - delta_eqw) * self.norm_reward * decay_factor

        # Apply concentration penalty
        if self.concentration_penalty > 0 and next_total_asset > 0:
            # Calculate position concentrations
            position_values = self.stocks * price
            position_ratios = position_values / next_total_asset

            # Penalize if any single position exceeds 60%
            max_concentration = position_ratios.max()
            if max_concentration > 0.60:
                concentration_excess = max_concentration - 0.60
                penalty = -self.concentration_penalty * concentration_excess
                reward += penalty

        self.total_asset = next_total_asset
        self.total_asset_eqw = next_total_asset_eqw

        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash

        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash
        return state, reward, done, None

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
        action_norm_vector = []
        price_0 = self.price_array[0]
        for price in price_0:
            x = math.floor(math.log(price, 10))  # the order of magnitude
            action_norm_vector.append(1 / ((10) ** x))

        action_norm_vector = np.asarray(action_norm_vector) * self.norm_action
        self.action_norm_vector = np.asarray(action_norm_vector)
