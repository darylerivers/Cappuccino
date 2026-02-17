#!/usr/bin/env python3
"""
Simple Ensemble Agent - Just loads actor.pth files and averages them.

No need for trial metadata, just raw model files.
"""

import json
from pathlib import Path

import numpy as np
import torch


class SimpleEnsembleAgent:
    """Dead simple ensemble - just load actor.pth files and average."""

    def __init__(self, manifest_path: str = "train_results/ensemble/ensemble_manifest.json", gpu_id: int = -1):
        self.manifest_path = Path(manifest_path)
        self.gpu_id = gpu_id
        self.device = torch.device("cpu" if gpu_id < 0 else f"cuda:{gpu_id}")
        self.actors = []

        print(f"\n{'='*80}")
        print("LOADING SIMPLE ENSEMBLE")
        print(f"{'='*80}")

        # Load manifest
        with self.manifest_path.open("r") as f:
            self.manifest = json.load(f)

        print(f"Ensemble has {self.manifest['model_count']} models")
        print(f"Mean value: {self.manifest['mean_value']:.6f}")
        print(f"Best value: {self.manifest['best_value']:.6f}")
        print()

        self._load_actors()

    def _load_actors(self):
        """Load all actor models."""
        print("Loading actor models...")

        actor_paths = self.manifest['actor_paths']
        trial_numbers = self.manifest['trial_numbers']
        trial_values = self.manifest['trial_values']

        for i, (trial_num, actor_path, value) in enumerate(zip(trial_numbers, actor_paths, trial_values), 1):
            try:
                actor_path = Path(actor_path)

                if not actor_path.exists():
                    print(f"  ⚠️  Model {i}: {actor_path} not found")
                    continue

                # Load actor state dict
                state_dict = torch.load(actor_path, map_location=self.device, weights_only=True)

                # Create a simple actor wrapper
                actor = torch.nn.Module()
                actor.state_dict = lambda: state_dict
                actor.load_state_dict(state_dict)
                actor.to(self.device)
                actor.eval()

                self.actors.append({
                    'model': actor,
                    'trial_number': trial_num,
                    'value': value,
                })

                print(f"  ✓ Model {i}/{len(trial_numbers)}: Trial #{trial_num} (value={value:.6f})")

            except Exception as e:
                print(f"  ✗ Model {i}: Trial #{trial_num} failed - {e}")
                continue

        print()
        print(f"✓ Loaded {len(self.actors)}/{self.manifest['model_count']} models")
        print(f"{'='*80}\n")

        if len(self.actors) == 0:
            raise RuntimeError("Failed to load any models")

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble prediction by running state through each actor and averaging.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Averaged action tensor
        """
        predictions = []

        with torch.no_grad():
            state = state.to(self.device)

            for actor_info in self.actors:
                actor = actor_info['model']

                # Forward pass through actor network
                # Most PPO actors have structure: net.0, net.2, net.4 layers
                action = state
                for name, module in actor.named_children():
                    if 'net' in name:
                        action = module(action)

                predictions.append(action.cpu())

        # Average all predictions
        ensemble_action = torch.stack(predictions).mean(dim=0)

        return ensemble_action

    def select_actions(self, state: np.ndarray) -> np.ndarray:
        """
        Numpy interface for compatibility.

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
        return f"SimpleEnsembleAgent(models={len(self.actors)}, mean_value={self.manifest['mean_value']:.6f})"


# Create wrapper that looks like a regular agent
class EnsembleActWrapper:
    """Wrapper to make ensemble look like a regular agent with act() method."""

    def __init__(self, manifest_path: str = "train_results/ensemble/ensemble_manifest.json", gpu_id: int = -1):
        self.ensemble = SimpleEnsembleAgent(manifest_path, gpu_id)
        self.device = self.ensemble.device
        self.act = self.ensemble.act

    def __getattr__(self, name):
        """Delegate to ensemble."""
        return getattr(self.ensemble, name)


def test():
    """Test ensemble."""
    print("Testing simple ensemble...")

    ensemble = SimpleEnsembleAgent(gpu_id=-1)

    # Test with dummy state
    state = torch.randn(1, 100)  # Batch size 1, state dim 100
    action = ensemble.act(state)

    print(f"\nTest prediction:")
    print(f"  State shape: {state.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Action sample: {action[0, :5]}")
    print("\n✓ Ensemble working!")


if __name__ == "__main__":
    test()
