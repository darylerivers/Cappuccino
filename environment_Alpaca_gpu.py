"""
GPU-accelerated batch crypto trading environment using PyTorch.
All trading logic runs on GPU using PyTorch operations.
"""

import torch
import numpy as np
from typing import Tuple, Dict


class GPUBatchCryptoEnv:
    """
    Fully GPU-accelerated crypto trading environment.

    All operations (price lookups, trading logic, fee calculations, etc.)
    run on GPU using PyTorch tensors. This eliminates CPU bottleneck and
    maximizes GPU utilization.
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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

        # Normalization constants (must match CryptoEnvAlpaca)
        self.norm_cash = env_params.get('norm_cash', 1.0)
        self.norm_stocks = env_params.get('norm_stocks', 1.0)
        self.norm_tech = env_params.get('norm_tech', 1.0)

        # Move data arrays to GPU
        price_np = config['price_array'].astype(np.float32)
        tech_np = config['tech_array'].astype(np.float32)

        self.price_array = torch.from_numpy(price_np).to(self.device)  # (T, n_assets) on GPU
        self.tech_array = torch.from_numpy(tech_np).to(self.device)    # (T, n_tech) on GPU

        # Delete NumPy arrays to free RAM (data now on GPU)
        del price_np, tech_np
        config['price_array'] = None  # Remove reference to save RAM
        config['tech_array'] = None

        self.max_time = self.price_array.shape[0] - 1
        self.n_assets = self.price_array.shape[1]
        self.n_tech = self.tech_array.shape[1]

        # Environment dimensions - MUST match CryptoEnvAlpaca.get_state()
        # state = [cash, stocks, tech[t], tech[t-1], ..., tech[t-lookback+1]]
        self.action_dim = self.n_assets
        self.state_dim = 1 + self.n_assets + self.n_tech * self.lookback

        # Min trade sizes on GPU
        self.minimum_qty = torch.full((self.n_assets,), 0.0001,
                                     dtype=torch.float32, device=self.device)

        # Action normalization on GPU
        initial_price = self.price_array[self.lookback - 1]
        self.action_norm_vector = (self.initial_capital / (initial_price + 1e-8)).to(self.device)

        # Batch state tensors (all on GPU)
        self.time = torch.full((n_envs,), self.lookback, dtype=torch.long, device=self.device)
        self.cash = torch.full((n_envs,), initial_capital, dtype=torch.float32, device=self.device)
        self.stocks = torch.zeros((n_envs, self.n_assets), dtype=torch.float32, device=self.device)
        self.stocks_cooldown = torch.zeros((n_envs, self.n_assets), dtype=torch.long, device=self.device)
        self.highest_price_since_buy = torch.zeros((n_envs, self.n_assets), dtype=torch.float32, device=self.device)

        # Fee tracking
        self.buy_fees_paid = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
        self.sell_fees_paid = torch.zeros(n_envs, dtype=torch.float32, device=self.device)

        # ElegantRL compatibility
        self.env_num = n_envs
        self.env_name = f'GPUBatchCryptoEnv_x{n_envs}'
        self.if_discrete = False
        self.max_step = self.max_time - self.lookback
        self.target_return = 2.0

        # Pre-computed lookback offsets for vectorized _get_states (avoids allocation per step)
        self._lookback_offsets = torch.arange(self.lookback, dtype=torch.long, device=self.device)

    def reset(self):
        """Reset all environments to initial state."""
        self.time[:] = self.lookback
        self.cash[:] = self.initial_capital
        self.stocks[:] = 0
        self.stocks_cooldown[:] = 0
        self.highest_price_since_buy[:] = 0
        self.buy_fees_paid[:] = 0
        self.sell_fees_paid[:] = 0

        states = self._get_states()  # Already on GPU

        if self.n_envs == 1:
            return states[0]
        return states

    def _get_states(self) -> torch.Tensor:
        """
        Get current states for all environments (fully vectorized GPU operations).
        MUST match CryptoEnvAlpaca.get_state() exactly:
          state = [cash*norm_cash, stocks*norm_stocks, tech[t]*norm_tech, tech[t-1]*norm_tech, ...]
        Returns: (n_envs, state_dim) tensor on GPU

        Vectorized implementation: no Python loops, no .item() GPU→CPU syncs.
        """
        # (n_envs, 1)
        cash_feat = self.cash.unsqueeze(1) * self.norm_cash

        # (n_envs, n_assets)
        stocks_feat = self.stocks * self.norm_stocks

        # Gather tech history for all envs at once:
        #   time_indices[i, j] = self.time[i] - j  →  (n_envs, lookback)
        time_indices = self.time.unsqueeze(1) - self._lookback_offsets

        # (n_envs, lookback, n_tech) → flatten to (n_envs, lookback * n_tech)
        tech_feat = (self.tech_array[time_indices] * self.norm_tech).reshape(self.n_envs, -1)

        # (n_envs, 1 + n_assets + lookback * n_tech)
        return torch.cat([cash_feat, stocks_feat, tech_feat], dim=1)

    def step(self, actions):
        """
        Step all environments with batched actions (all GPU operations).

        Args:
            actions: (n_envs, action_dim) tensor (CPU or GPU)

        Returns:
            states: (n_envs, state_dim) tensor on GPU
            rewards: (n_envs,) tensor on GPU
            dones: (n_envs,) tensor on GPU (bool)
            infos: list of dicts
        """
        # Ensure actions are on GPU
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(actions).to(dtype=torch.float32, device=self.device)
        elif actions.device != self.device:
            actions = actions.to(self.device)

        # Ensure 2D
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        # Increment time
        self.time += 1

        # Get current prices (n_envs, n_assets) - GPU indexing
        prices = self.price_array[self.time]

        # Cooldown increment for held stocks (GPU)
        self.stocks_cooldown += (self.stocks > 0).long()

        # === TRAILING STOP LOSS (GPU) ===
        if self.use_trailing_stop:
            holding_mask = self.stocks > 0

            # Update highest prices (GPU)
            price_higher = prices > self.highest_price_since_buy
            self.highest_price_since_buy = torch.where(
                holding_mask & price_higher,
                prices,
                self.highest_price_since_buy
            )

            # Check stop loss triggers (GPU)
            stop_price = self.highest_price_since_buy * (1.0 - self.trailing_stop_pct)
            trigger_mask = (
                holding_mask &
                (self.highest_price_since_buy > 0) &
                (prices < stop_price) &
                (prices > 0)
            )

            if trigger_mask.any():
                # Execute stop loss sells (GPU)
                sell_qty = torch.where(trigger_mask, self.stocks, torch.zeros_like(self.stocks))
                sell_value = prices * sell_qty
                fees = sell_value * self.sell_cost_pct

                self.stocks = torch.where(trigger_mask, torch.zeros_like(self.stocks), self.stocks)
                self.cash += (sell_value * (1 - self.sell_cost_pct)).sum(dim=1)
                self.sell_fees_paid += fees.sum(dim=1)
                self.highest_price_since_buy = torch.where(trigger_mask, torch.zeros_like(self.highest_price_since_buy), self.highest_price_since_buy)

        # Scale actions (GPU)
        actions_scaled = actions * self.action_norm_vector

        # === EXECUTE SELLS (GPU) ===
        sell_mask = actions_scaled < -self.minimum_qty
        has_stock = self.stocks > 0
        valid_sell = sell_mask & has_stock & (prices > 0)

        if valid_sell.any():
            sell_qty = torch.minimum(
                self.stocks,
                torch.where(valid_sell, -actions_scaled, torch.zeros_like(actions_scaled))
            )
            sell_value = prices * sell_qty
            fees = sell_value * self.sell_cost_pct

            self.stocks -= sell_qty
            self.cash += (sell_value * (1 - self.sell_cost_pct)).sum(dim=1)
            self.sell_fees_paid += fees.sum(dim=1)
            self.stocks_cooldown = torch.where(valid_sell, torch.zeros_like(self.stocks_cooldown), self.stocks_cooldown)

        # === FORCED COOLDOWN SELL (GPU) ===
        cooldown_expired = self.stocks_cooldown >= 48
        if cooldown_expired.any():
            forced_sell_qty = self.stocks * 0.05 * cooldown_expired.float()
            sell_value = (prices * forced_sell_qty).sum(dim=1)
            fees = sell_value * self.sell_cost_pct

            self.stocks -= forced_sell_qty
            self.cash += sell_value * (1 - self.sell_cost_pct)
            self.sell_fees_paid += fees
            self.stocks_cooldown = torch.where(cooldown_expired, torch.zeros_like(self.stocks_cooldown), self.stocks_cooldown)

        # === EXECUTE BUYS (GPU) ===
        buy_mask = actions_scaled > self.minimum_qty
        valid_price = prices > 0

        if (buy_mask & valid_price).any():
            # Cash reserve constraint (GPU)
            reserved = self.initial_capital * self.min_cash_reserve
            available_cash = torch.clamp(self.cash - reserved, min=0)

            # Max buyable per asset (GPU)
            fee_corrected = available_cash.unsqueeze(1) / (1 + self.buy_cost_pct)
            max_buy_per_asset = fee_corrected / (prices + 1e-8)

            # Actual buy quantities (GPU)
            buy_qty = torch.minimum(
                max_buy_per_asset,
                torch.where(buy_mask & valid_price, actions_scaled, torch.zeros_like(actions_scaled))
            )

            # Filter by minimum qty (GPU)
            buy_qty = torch.where(buy_qty >= self.minimum_qty, buy_qty, torch.zeros_like(buy_qty))

            if buy_qty.sum() > 0:
                buy_value = prices * buy_qty
                fees = buy_value * self.buy_cost_pct
                total_cost = buy_value * (1 + self.buy_cost_pct)

                self.stocks += buy_qty
                self.cash -= total_cost.sum(dim=1)
                self.buy_fees_paid += fees.sum(dim=1)

                # Update highest price for new positions (GPU)
                new_position = (buy_qty > 0) & (self.stocks == buy_qty)
                self.highest_price_since_buy = torch.where(
                    new_position,
                    prices,
                    self.highest_price_since_buy
                )

        # === CALCULATE REWARDS (GPU) ===
        total_assets = self.cash + (self.stocks * prices).sum(dim=1)
        rewards = (total_assets / self.initial_capital - 1.0)

        # === CHECK DONE (GPU) ===
        dones = (self.time >= self.max_time)

        # Auto-reset done environments (GPU)
        if dones.any():
            done_mask = dones.unsqueeze(1)  # (n_envs, 1)

            self.time = torch.where(dones, torch.full_like(self.time, self.lookback), self.time)
            self.cash = torch.where(dones, torch.full_like(self.cash, self.initial_capital), self.cash)
            self.stocks = torch.where(done_mask, torch.zeros_like(self.stocks), self.stocks)
            self.stocks_cooldown = torch.where(done_mask, torch.zeros_like(self.stocks_cooldown), self.stocks_cooldown)
            self.highest_price_since_buy = torch.where(done_mask, torch.zeros_like(self.highest_price_since_buy), self.highest_price_since_buy)

        # Get new states (on GPU)
        states = self._get_states()

        # Return format
        if self.n_envs == 1:
            return (
                states[0],
                rewards[0],
                dones[0],
                {}
            )

        # All tensors already on GPU
        return states, rewards, dones.bool(), [{} for _ in range(self.n_envs)]
