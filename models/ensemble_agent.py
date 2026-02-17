#!/usr/bin/env python3
"""
Ensemble Agent - Loads multiple trained models and averages predictions.

Usage:
    Use this as a drop-in replacement for a single agent in paper trading.
"""

import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch

from drl_agents.elegantrl_models import MODELS
from train.config import Arguments
from train.run import init_agent


class EnsembleAgent:
    """Ensemble of multiple trained agents that votes/averages predictions."""

    def __init__(self, manifest_path: str = "train_results/ensemble/ensemble_manifest.json", gpu_id: int = -1):
        """
        Load ensemble from manifest.

        Args:
            manifest_path: Path to ensemble manifest JSON
            gpu_id: GPU to use (-1 for CPU)
        """
        self.manifest_path = Path(manifest_path)
        self.gpu_id = gpu_id
        self.agents = []
        self.device = torch.device("cpu" if gpu_id < 0 else f"cuda:{gpu_id}")

        print(f"\n{'='*80}")
        print("LOADING ENSEMBLE")
        print(f"{'='*80}")

        # Load manifest
        with self.manifest_path.open("r") as f:
            self.manifest = json.load(f)

        print(f"Ensemble has {self.manifest['model_count']} models")
        print(f"Mean value: {self.manifest['mean_value']:.6f}")
        print(f"Best value: {self.manifest['best_value']:.6f}")
        print()

    def load_models(self, env):
        """Load all models from manifest into memory."""
        print("Loading models...")

        trial_paths = self.manifest['trial_paths']
        actor_paths = self.manifest['actor_paths']
        trial_numbers = self.manifest['trial_numbers']

        for i, (trial_num, trial_path, actor_path) in enumerate(zip(trial_numbers, trial_paths, actor_paths), 1):
            try:
                # Load trial hyperparameters
                trial_file = Path(trial_path) / "best_trial"

                if not trial_file.exists():
                    print(f"  ⚠️  Trial #{trial_num}: No best_trial file, skipping")
                    continue

                with trial_file.open("rb") as f:
                    trial = pickle.load(f)

                # Get model configuration
                model_name = trial.user_attrs.get("model_name", "ppo")
                params = trial.params
                net_dimension = int(params["net_dimension"])

                # Initialize agent
                args = Arguments(agent=MODELS[model_name], env=env)
                args.cwd = str(Path(trial_path))
                args.if_remove = False
                args.net_dim = net_dimension

                agent = init_agent(args, gpu_id=self.gpu_id, env=env)

                self.agents.append({
                    'agent': agent,
                    'trial_number': trial_num,
                    'value': self.manifest['trial_values'][i-1],
                })

                print(f"  ✓ Model {i}/{len(trial_numbers)}: Trial #{trial_num} (value={self.manifest['trial_values'][i-1]:.6f})")

            except Exception as e:
                print(f"  ✗ Model {i}/{len(trial_numbers)}: Trial #{trial_num} failed - {e}")
                continue

        print()
        print(f"✓ Loaded {len(self.agents)}/{self.manifest['model_count']} models successfully")
        print(f"{'='*80}\n")

        if len(self.agents) == 0:
            raise RuntimeError("Failed to load any models for ensemble")

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble prediction by averaging all model outputs.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Averaged action tensor
        """
        predictions = []

        with torch.no_grad():
            for model_info in self.agents:
                agent = model_info['agent']
                action = agent.act(state.to(agent.device))
                predictions.append(action.cpu())

        # Average all predictions
        ensemble_action = torch.stack(predictions).mean(dim=0)

        return ensemble_action.to(self.device)

    def select_actions(self, state: np.ndarray) -> np.ndarray:
        """
        Compatibility method for environments expecting select_actions.

        Args:
            state: State array

        Returns:
            Action array
        """
        state_tensor = torch.from_numpy(state.astype(np.float32))
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        action_tensor = self.act(state_tensor)
        return action_tensor.squeeze(0).numpy()

    def __repr__(self):
        return f"EnsembleAgent(models={len(self.agents)}, mean_value={self.manifest['mean_value']:.6f})"


def test_ensemble():
    """Test ensemble loading."""
    print("Testing ensemble agent...")

    # Create dummy environment
    from environment_Alpaca import CryptoEnvAlpaca
    from config_main import TICKER_LIST, TECHNICAL_INDICATORS_LIST

    n_tickers = len(TICKER_LIST)
    n_tech = len(TECHNICAL_INDICATORS_LIST)

    dummy_price = np.ones((100, n_tickers), dtype=np.float32)
    dummy_tech = np.ones((100, n_tickers * n_tech), dtype=np.float32)

    env_config = {
        "price_array": dummy_price,
        "tech_array": dummy_tech,
        "if_train": False,
    }
    env_params = {
        "lookback": 60,
        "norm_cash": 2**-11,
        "norm_stocks": 2**-8,
        "norm_tech": 2**-14,
        "norm_reward": 2**-9,
        "norm_action": 100,
        "time_decay_floor": 0.0,
        "min_cash_reserve": 0.1,
        "concentration_penalty": 0.05,
    }

    env = CryptoEnvAlpaca(env_config, env_params, if_log=False, use_sentiment=False)

    # Create ensemble
    ensemble = EnsembleAgent(gpu_id=-1)
    ensemble.load_models(env)

    # Test prediction
    state = env.reset()
    action = ensemble.select_actions(state)

    print(f"\nTest prediction:")
    print(f"  State shape: {state.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Action sample: {action[:3]}")
    print("\n✓ Ensemble working!")


if __name__ == "__main__":
    test_ensemble()
