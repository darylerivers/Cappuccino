"""
Fully vectorized batch environment for CryptoEnvAlpaca.
Processes N environments simultaneously using pure NumPy vectorized operations.
This eliminates the sequential Python loop bottleneck and dramatically
improves CPU utilization, allowing the GPU to stay saturated.
"""

import numpy as np
import torch
from typing import Tuple


class BatchVectorizedCryptoEnv:
    """
    Fully vectorized crypto trading environment.
    Processes N environments simultaneously using pure NumPy operations.

    This eliminates the sequential Python loop bottleneck and dramatically
    improves CPU utilization, allowing the GPU to stay saturated.
    """

    def __init__(self, config, env_params, n_envs=8, initial_capital=500.0,
                 buy_cost_pct=0.0025, sell_cost_pct=0.0025, gamma=0.99,
                 use_trailing_stop=False, trailing_stop_pct=0.15,
                 min_cash_reserve=0.0, tickers=None, **kwargs):
        """
        Args:
            config: dict with 'price_array' (T, n_assets) and 'tech_array' (T, n_tech)
            env_params: dict with 'lookback' and other params
            n_envs: number of parallel environments
        """
        self.n_envs = n_envs
        self.lookback = env_params['lookback']
        self.initial_capital = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.min_cash_reserve = min_cash_reserve
        self.tickers = tickers or []

        # Data arrays (shared across all envs)
        self.price_array = config['price_array'].astype(np.float32)  # (T, n_assets)
        self.tech_array = config['tech_array'].astype(np.float32)    # (T, n_tech)

        self.max_time = self.price_array.shape[0] - 1
        self.n_assets = self.price_array.shape[1]
        self.n_tech = self.tech_array.shape[1]

        # Environment dimensions
        self.action_dim = self.n_assets
        self.state_dim = (
            1 +  # cash
            self.n_assets +  # holdings
            self.n_assets +  # prices
            self.n_tech +    # tech indicators
            self.lookback * self.n_assets  # price history
        )

        # Min trade sizes (Alpaca minimums)
        self.minimum_qty_alpaca = np.ones(self.n_assets, dtype=np.float32) * 0.0001

        # Action normalization (shared)
        initial_price = self.price_array[self.lookback - 1]
        self.action_norm_vector = (self.initial_capital / (initial_price + 1e-8)).astype(np.float32)

        # Batch state (n_envs copies)
        self.time = np.full(n_envs, self.lookback, dtype=np.int32)
        self.cash = np.full(n_envs, initial_capital, dtype=np.float32)
        self.stocks = np.zeros((n_envs, self.n_assets), dtype=np.float32)
        self.stocks_cooldown = np.zeros((n_envs, self.n_assets), dtype=np.int32)

        # Trailing stop state
        self.highest_price_since_buy = np.zeros((n_envs, self.n_assets), dtype=np.float32)

        # Fee tracking
        self.buy_fees_paid = np.zeros(n_envs, dtype=np.float32)
        self.sell_fees_paid = np.zeros(n_envs, dtype=np.float32)

        # For ElegantRL compatibility
        self.env_num = n_envs
        self.env_name = f'BatchVectorizedCryptoEnv_x{n_envs}'
        self.if_discrete = False
        self.max_step = self.max_time - self.lookback
        self.target_return = 2.0

        # GPU device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def reset(self):
        """Reset all environments to initial state."""
        self.time[:] = self.lookback
        self.cash[:] = self.initial_capital
        self.stocks[:] = 0
        self.stocks_cooldown[:] = 0
        self.highest_price_since_buy[:] = 0
        self.buy_fees_paid[:] = 0
        self.sell_fees_paid[:] = 0

        # Return initial states as torch tensor
        states = self._get_states()

        if self.n_envs == 1:
            return states[0]
        return torch.from_numpy(states).to(dtype=torch.float32, device=self.device)

    def _get_states(self) -> np.ndarray:
        """
        Get current states for all environments.
        Returns: (n_envs, state_dim)
        """
        states = np.empty((self.n_envs, self.state_dim), dtype=np.float32)

        for i in range(self.n_envs):
            t = self.time[i]

            # Current prices and tech
            price = self.price_array[t]
            tech = self.tech_array[t]

            # Price history lookback
            price_history = self.price_array[t - self.lookback + 1 : t + 1].flatten()

            # Concatenate: [cash, stocks, price, tech, price_history]
            states[i] = np.concatenate([
                [self.cash[i]],
                self.stocks[i],
                price,
                tech,
                price_history
            ])

        return states

    def step(self, actions):
        """
        Step all environments with batched actions using vectorized NumPy operations.

        Args:
            actions: (n_envs, action_dim) numpy or torch tensor

        Returns:
            states: (n_envs, state_dim) torch tensor
            rewards: (n_envs,) torch tensor
            dones: (n_envs,) torch tensor (bool)
            infos: list of dicts
        """
        # Convert torch to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        # Ensure 2D
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        # Increment time
        self.time += 1

        # Get current prices (n_envs, n_assets)
        prices = self.price_array[self.time]

        # Cooldown increment for held stocks (vectorized)
        self.stocks_cooldown += (self.stocks > 0).astype(np.int32)

        # Trailing stop loss (vectorized)
        if self.use_trailing_stop:
            # Update highest prices
            holding_mask = self.stocks > 0
            self.highest_price_since_buy = np.where(
                holding_mask & (prices > self.highest_price_since_buy),
                prices,
                self.highest_price_since_buy
            )

            # Check stop loss triggers
            stop_price = self.highest_price_since_buy * (1.0 - self.trailing_stop_pct)
            trigger_mask = (
                holding_mask &
                (self.highest_price_since_buy > 0) &
                (prices < stop_price) &
                (prices > 0)
            )

            if trigger_mask.any():
                # Execute stop loss sells
                sell_qty = np.where(trigger_mask, self.stocks, 0)
                sell_value = prices * sell_qty
                fees = sell_value * self.sell_cost_pct

                self.stocks = np.where(trigger_mask, 0, self.stocks)
                self.cash += (sell_value * (1 - self.sell_cost_pct)).sum(axis=1)
                self.sell_fees_paid += fees.sum(axis=1)
                self.highest_price_since_buy = np.where(trigger_mask, 0, self.highest_price_since_buy)

        # Scale actions
        actions_scaled = actions * self.action_norm_vector

        # === EXECUTE SELLS (vectorized) ===
        sell_mask = actions_scaled < -self.minimum_qty_alpaca
        has_stock = self.stocks > 0
        valid_sell = sell_mask & has_stock & (prices > 0)

        if valid_sell.any():
            sell_qty = np.minimum(
                self.stocks,
                np.where(valid_sell, -actions_scaled, 0)
            )
            sell_value = prices * sell_qty
            fees = sell_value * self.sell_cost_pct

            self.stocks -= sell_qty
            self.cash += (sell_value * (1 - self.sell_cost_pct)).sum(axis=1)
            self.sell_fees_paid += fees.sum(axis=1)
            self.stocks_cooldown = np.where(valid_sell, 0, self.stocks_cooldown)

        # === FORCED COOLDOWN SELL (vectorized) ===
        cooldown_expired = self.stocks_cooldown >= 48
        if cooldown_expired.any():
            forced_sell_qty = self.stocks * 0.05 * cooldown_expired
            sell_value = (prices * forced_sell_qty).sum(axis=1)
            fees = sell_value * self.sell_cost_pct

            self.stocks -= forced_sell_qty
            self.cash += sell_value * (1 - self.sell_cost_pct)
            self.sell_fees_paid += fees
            self.stocks_cooldown = np.where(cooldown_expired, 0, self.stocks_cooldown)

        # === EXECUTE BUYS (vectorized) ===
        buy_mask = actions_scaled > self.minimum_qty_alpaca
        valid_price = prices > 0

        if (buy_mask & valid_price).any():
            # Cash reserve constraint
            reserved = self.initial_capital * self.min_cash_reserve
            available_cash = np.maximum(0, self.cash - reserved)

            # Max buyable per asset
            fee_corrected = available_cash[:, None] / (1 + self.buy_cost_pct)
            max_buy_per_asset = fee_corrected / (prices + 1e-8)

            # Actual buy quantities
            buy_qty = np.minimum(
                max_buy_per_asset,
                np.where(buy_mask & valid_price, actions_scaled, 0)
            )

            # Filter by minimum qty
            buy_qty = np.where(buy_qty >= self.minimum_qty_alpaca, buy_qty, 0)

            if buy_qty.sum() > 0:
                buy_value = prices * buy_qty
                fees = buy_value * self.buy_cost_pct
                total_cost = buy_value * (1 + self.buy_cost_pct)

                self.stocks += buy_qty
                self.cash -= total_cost.sum(axis=1)
                self.buy_fees_paid += fees.sum(axis=1)

                # Update highest price for new positions
                new_position = (buy_qty > 0) & (self.stocks == buy_qty)
                self.highest_price_since_buy = np.where(
                    new_position,
                    prices,
                    self.highest_price_since_buy
                )

        # === CALCULATE REWARDS ===
        total_assets = self.cash + (self.stocks * prices).sum(axis=1)
        rewards = (total_assets / self.initial_capital - 1.0).astype(np.float32)

        # === CHECK DONE ===
        dones = (self.time >= self.max_time)

        # Auto-reset done environments
        if dones.any():
            for i in np.where(dones)[0]:
                self.time[i] = self.lookback
                self.cash[i] = self.initial_capital
                self.stocks[i] = 0
                self.stocks_cooldown[i] = 0
                self.highest_price_since_buy[i] = 0

        # Get new states
        states = self._get_states()

        # Return format
        if self.n_envs == 1:
            return (
                states[0],
                rewards[0],
                dones[0],
                {}
            )

        # Vectorized return (torch tensors on GPU)
        states_t = torch.from_numpy(states).to(dtype=torch.float32, device=self.device)
        rewards_t = torch.from_numpy(rewards).to(dtype=torch.float32, device=self.device)
        dones_t = torch.from_numpy(dones).to(dtype=torch.bool, device=self.device)

        return states_t, rewards_t, dones_t, [{} for _ in range(self.n_envs)]
