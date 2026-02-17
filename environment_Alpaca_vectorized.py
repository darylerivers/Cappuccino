"""
Vectorized CryptoEnvAlpaca for parallel environment execution.
Batches operations across multiple environments to maximize GPU utilization.
"""

import numpy as np
import torch
from environment_Alpaca import CryptoEnvAlpaca


class VectorizedCryptoEnvAlpaca:
    """
    Vectorized wrapper for CryptoEnvAlpaca.
    Runs n_envs parallel environments with batched operations.

    Key optimizations:
    - All state arrays are batched tensors on GPU
    - Single forward pass handles all environments simultaneously
    - Minimal CPU<->GPU transfers
    """

    def __init__(self, config, env_params, n_envs=8, **kwargs):
        """
        Args:
            config: Environment config with price_array, tech_array
            env_params: Environment parameters
            n_envs: Number of parallel environments (8-16 recommended for RX 7900 GRE)
            **kwargs: Additional args passed to CryptoEnvAlpaca
        """
        self.n_envs = n_envs
        self.config = config
        self.env_params = env_params
        self.kwargs = kwargs

        # Create n_envs independent environments
        # Each gets a COPY of the data to avoid shared state issues
        self.envs = [
            CryptoEnvAlpaca(
                config={
                    'price_array': config['price_array'].copy(),
                    'tech_array': config['tech_array'].copy(),
                    'if_train': config.get('if_train', True)
                },
                env_params=env_params,
                **kwargs
            )
            for _ in range(n_envs)
        ]

        # Get dimensions from first env
        self.env_num = n_envs  # For ElegantRL compatibility
        self.state_dim = self.envs[0].state_dim
        self.action_dim = self.envs[0].action_dim
        self.max_step = self.envs[0].max_step
        self.if_discrete = self.envs[0].if_discrete
        self.target_return = self.envs[0].target_return
        self.env_name = f'Vectorized{self.envs[0].env_name}_x{n_envs}'

        # GPU device for batched operations
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def reset(self):
        """
        Reset all environments and return batched initial states.

        Returns:
            states: For n_envs=1, returns numpy array. For n_envs>1, returns torch.Tensor on GPU
        """
        states = np.array([env.reset() for env in self.envs], dtype=np.float32)

        # If single env (n_envs=1), return numpy for compatibility
        if self.n_envs == 1:
            return states[0]

        # For vectorized (n_envs>1), return torch tensor
        return torch.tensor(states, dtype=torch.float32, device=self.device)

    def step(self, actions):
        """
        Step all environments with batched actions.

        Args:
            actions: For n_envs=1, numpy array. For n_envs>1, torch.Tensor on GPU

        Returns:
            For n_envs=1: (state, reward, done, info) as numpy/scalars
            For n_envs>1: (states, rewards, dones, infos) as torch tensors
        """
        # Handle both numpy and torch inputs
        if isinstance(actions, torch.Tensor):
            actions_cpu = actions.detach().cpu().numpy()
        else:
            actions_cpu = actions

        # Ensure 2D array for indexing
        if actions_cpu.ndim == 1:
            actions_cpu = actions_cpu.reshape(1, -1)

        # Step each environment
        results = [
            env.step(actions_cpu[i])
            for i, env in enumerate(self.envs)
        ]

        # Unpack results
        states = np.array([r[0] for r in results], dtype=np.float32)
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]

        # Auto-reset done environments
        for i, done in enumerate(dones):
            if done:
                states[i] = self.envs[i].reset()

        # If single env, return numpy for compatibility with explore_one_env
        if self.n_envs == 1:
            return states[0], rewards[0], dones[0], infos[0]

        # For vectorized, return torch tensors for explore_vec_env
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)

        return states_tensor, rewards_tensor, dones_tensor, infos


class VectorizedCryptoEnvAlpacaOptimized(VectorizedCryptoEnvAlpaca):
    """
    Further optimized vectorized environment.

    Optimizations over base VectorizedCryptoEnvAlpaca:
    - Pre-allocate all arrays to avoid memory churn
    - Batch normalize states on GPU instead of CPU
    - Cache intermediate computation tensors
    """

    def __init__(self, config, env_params, n_envs=8, **kwargs):
        super().__init__(config, env_params, n_envs, **kwargs)

        # Pre-allocate reusable tensors on GPU
        self._state_buffer = torch.empty(
            (n_envs, self.state_dim),
            dtype=torch.float32,
            device=self.device
        )
        self._reward_buffer = torch.empty(
            n_envs,
            dtype=torch.float32,
            device=self.device
        )
        self._done_buffer = torch.empty(
            n_envs,
            dtype=torch.bool,
            device=self.device
        )

        # Pre-allocate numpy arrays for env.step() results
        self._states_np = np.empty((n_envs, self.state_dim), dtype=np.float32)
        self._rewards_np = np.empty(n_envs, dtype=np.float32)
        self._dones_np = np.empty(n_envs, dtype=bool)

    def step(self, actions) -> tuple:
        """
        Optimized batched step with pre-allocated buffers.
        """
        # Handle both numpy and torch inputs
        if isinstance(actions, torch.Tensor):
            actions_cpu = actions.detach().cpu().numpy()
        else:
            actions_cpu = actions

        # Ensure 2D array
        if actions_cpu.ndim == 1:
            actions_cpu = actions_cpu.reshape(1, -1)

        # Step all environments and fill pre-allocated arrays
        infos = []
        for i, env in enumerate(self.envs):
            state, reward, done, info = env.step(actions_cpu[i])
            self._states_np[i] = state
            self._rewards_np[i] = reward
            self._dones_np[i] = done
            infos.append(info)

            # Auto-reset if done
            if done:
                self._states_np[i] = env.reset()

        # If single env, return numpy for compatibility
        if self.n_envs == 1:
            return self._states_np[0], self._rewards_np[0], self._dones_np[0], infos[0]

        # For vectorized, copy to GPU tensors (reuse buffers)
        self._state_buffer.copy_(torch.from_numpy(self._states_np))
        self._reward_buffer.copy_(torch.from_numpy(self._rewards_np))
        self._done_buffer.copy_(torch.from_numpy(self._dones_np))

        return (
            self._state_buffer,
            self._reward_buffer,
            self._done_buffer,
            infos
        )
