#!/usr/bin/env python3
"""
Multi-Timeframe Ensemble Agent

Combines strategic (1h) and tactical (5m/15m) models for better entry/exit timing.

Architecture:
- Strategic Layer (1h): Decides overall direction and position sizing
- Tactical Layer (5m/15m): Refines entry/exit timing
- Combiner: Merges signals based on confidence and alignment

Usage:
    from multi_timeframe_ensemble_agent import MultiTimeframeEnsemble

    ensemble = MultiTimeframeEnsemble(
        strategic_dir='train_results/ensemble_1h',
        tactical_dirs=['train_results/ensemble_5m', 'train_results/ensemble_15m'],
        gpu_id=-1
    )

    action = ensemble.act(state_1h, state_5m, state_15m)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class MultiTimeframeEnsemble:
    """
    Ensemble that combines multiple timeframes for strategic and tactical trading.
    """

    def __init__(
        self,
        strategic_dir: str = "train_results/ensemble_1h",
        tactical_dirs: List[str] = None,
        gpu_id: int = -1,
        strategy: str = "timing_refinement",
        strategic_weight: float = 0.7,
        tactical_weight: float = 0.3,
        confidence_threshold: float = 0.1,
    ):
        """
        Initialize multi-timeframe ensemble.

        Args:
            strategic_dir: Directory with 1h ensemble
            tactical_dirs: List of directories with short TF ensembles
            gpu_id: GPU device (-1 for CPU)
            strategy: Combination strategy ('gating', 'weighted', 'timing_refinement')
            strategic_weight: Weight for strategic signals (0-1)
            tactical_weight: Weight for tactical signals (0-1)
            confidence_threshold: Minimum confidence to act
        """
        self.device = torch.device("cpu" if gpu_id < 0 else f"cuda:{gpu_id}")
        self.strategy = strategy
        self.strategic_weight = strategic_weight
        self.tactical_weight = tactical_weight
        self.confidence_threshold = confidence_threshold

        print(f"\n{'='*80}")
        print("LOADING MULTI-TIMEFRAME ENSEMBLE")
        print(f"{'='*80}")
        print(f"Strategy: {strategy}")
        print(f"Weights: Strategic {strategic_weight:.1%} | Tactical {tactical_weight:.1%}")
        print()

        # Load strategic ensemble (1h)
        print("[1/3] Loading Strategic Layer (1h)...")
        self.strategic_ensemble = self._load_ensemble(strategic_dir, "Strategic")

        # Load tactical ensembles (5m, 15m, etc)
        self.tactical_ensembles = []
        if tactical_dirs:
            print(f"[2/3] Loading Tactical Layer ({len(tactical_dirs)} timeframes)...")
            for i, tac_dir in enumerate(tactical_dirs, 1):
                ensemble = self._load_ensemble(tac_dir, f"Tactical-{i}")
                self.tactical_ensembles.append(ensemble)
        else:
            print("[2/3] No tactical ensembles specified - using strategic only")

        print("[3/3] Multi-Timeframe Ensemble Ready!")
        print(f"{'='*80}\n")

    def _load_ensemble(self, ensemble_dir: str, label: str) -> Dict:
        """Load an ensemble from directory."""
        ensemble_path = Path(ensemble_dir)

        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")

        manifest_file = ensemble_path / "ensemble_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with manifest_file.open("r") as f:
            manifest = json.load(f)

        # Load actor state dicts
        models = []
        actor_paths = manifest['actor_paths']
        trial_numbers = manifest['trial_numbers']
        trial_values = manifest['trial_values']

        print(f"  {label}: Loading {len(actor_paths)} models...")

        for i, (actor_path, trial_num, value) in enumerate(zip(actor_paths, trial_numbers, trial_values), 1):
            try:
                state_dict = torch.load(actor_path, map_location=self.device, weights_only=True)
                models.append({
                    'state_dict': state_dict,
                    'trial_num': trial_num,
                    'value': value,
                })
                if i <= 3 or i == len(actor_paths):
                    print(f"    Model {i}: Trial #{trial_num} (value={value:.6f})")
            except Exception as e:
                print(f"    ✗ Model {i} failed: {e}")

        print(f"  ✓ Loaded {len(models)} models")
        print()

        return {
            'models': models,
            'manifest': manifest,
            'dir': ensemble_dir,
        }

    def _predict_single_ensemble(self, ensemble: Dict, state: torch.Tensor) -> np.ndarray:
        """Get prediction from a single ensemble."""
        predictions = []

        with torch.no_grad():
            state = state.to(self.device)

            for model in ensemble['models']:
                state_dict = model['state_dict']

                # Manual forward pass
                x = state

                # Layer 0: Linear + ReLU
                x = torch.nn.functional.linear(x, state_dict['net.0.weight'], state_dict['net.0.bias'])
                x = torch.relu(x)

                # Layer 2: Linear + ReLU
                x = torch.nn.functional.linear(x, state_dict['net.2.weight'], state_dict['net.2.bias'])
                x = torch.relu(x)

                # Layer 4: Linear (output)
                x = torch.nn.functional.linear(x, state_dict['net.4.weight'], state_dict['net.4.bias'])

                predictions.append(x.cpu().numpy())

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    def act(self, state_strategic: torch.Tensor, states_tactical: List[torch.Tensor] = None) -> np.ndarray:
        """
        Get action from multi-timeframe ensemble.

        Args:
            state_strategic: State from strategic timeframe (1h)
            states_tactical: List of states from tactical timeframes (5m, 15m, etc)

        Returns:
            Combined action array
        """
        # Get strategic prediction
        strategic_action = self._predict_single_ensemble(self.strategic_ensemble, state_strategic)

        # Get tactical predictions
        tactical_actions = []
        if states_tactical and self.tactical_ensembles:
            for state_tac, ensemble_tac in zip(states_tactical, self.tactical_ensembles):
                tac_action = self._predict_single_ensemble(ensemble_tac, state_tac)
                tactical_actions.append(tac_action)

        # Combine based on strategy
        if self.strategy == "gating":
            return self._combine_gating(strategic_action, tactical_actions)
        elif self.strategy == "weighted":
            return self._combine_weighted(strategic_action, tactical_actions)
        elif self.strategy == "timing_refinement":
            return self._combine_timing_refinement(strategic_action, tactical_actions)
        else:
            # Default: strategic only
            return strategic_action

    def _combine_gating(self, strategic_action: np.ndarray, tactical_actions: List[np.ndarray]) -> np.ndarray:
        """
        Gating strategy: Strategic must agree before considering tactical.

        Only execute if both strategic and tactical signals are strong and aligned.
        """
        if not tactical_actions:
            return strategic_action

        # Average tactical signals
        tactical_avg = np.mean(tactical_actions, axis=0)

        # Check if strategic signal is strong enough
        strategic_strong = np.abs(strategic_action) > self.confidence_threshold

        # Check if tactical agrees
        tactical_agrees = np.sign(strategic_action) == np.sign(tactical_avg)

        # Gate: only pass through if both agree
        combined = np.where(
            strategic_strong & tactical_agrees,
            strategic_action,  # Both agree - use strategic
            np.zeros_like(strategic_action)  # Disagree - hold
        )

        return combined

    def _combine_weighted(self, strategic_action: np.ndarray, tactical_actions: List[np.ndarray]) -> np.ndarray:
        """
        Weighted average: Combine signals with fixed weights.
        """
        if not tactical_actions:
            return strategic_action

        tactical_avg = np.mean(tactical_actions, axis=0)

        combined = (
            strategic_action * self.strategic_weight +
            tactical_avg * self.tactical_weight
        )

        return combined

    def _combine_timing_refinement(self, strategic_action: np.ndarray, tactical_actions: List[np.ndarray]) -> np.ndarray:
        """
        Timing refinement: Strategic decides IF, tactical decides WHEN.

        - Strategic determines direction and magnitude
        - Tactical provides timing filter (momentum check)
        - Only execute when both align and tactical shows momentum
        """
        if not tactical_actions:
            return strategic_action

        # Average tactical signals
        tactical_avg = np.mean(tactical_actions, axis=0)

        # Get directions
        strategic_direction = np.sign(strategic_action)
        tactical_direction = np.sign(tactical_avg)

        # Check alignment
        aligned = strategic_direction == tactical_direction

        # Check tactical momentum (must be reasonably strong)
        tactical_strong = np.abs(tactical_avg) > 0.05

        # Combine: Use strategic magnitude, but only where aligned and tactical is strong
        combined = np.where(
            aligned & tactical_strong,
            strategic_action,  # Good timing - execute
            strategic_action * 0.3  # Poor timing - reduce size or wait
        )

        return combined

    def get_signal_breakdown(self, state_strategic: torch.Tensor, states_tactical: List[torch.Tensor] = None) -> Dict:
        """
        Get detailed breakdown of signals from each timeframe.

        Returns:
            Dictionary with strategic, tactical, and combined signals plus metadata
        """
        # Get predictions
        strategic_action = self._predict_single_ensemble(self.strategic_ensemble, state_strategic)

        tactical_actions = []
        if states_tactical and self.tactical_ensembles:
            for state_tac, ensemble_tac in zip(states_tactical, self.tactical_ensembles):
                tac_action = self._predict_single_ensemble(ensemble_tac, state_tac)
                tactical_actions.append(tac_action)

        tactical_avg = np.mean(tactical_actions, axis=0) if tactical_actions else None

        # Calculate combined
        if tactical_actions:
            combined_action = self.act(state_strategic, states_tactical)
        else:
            combined_action = strategic_action

        # Calculate alignment
        if tactical_avg is not None:
            alignment = np.mean(np.sign(strategic_action) == np.sign(tactical_avg))
        else:
            alignment = 1.0

        return {
            'strategic': strategic_action,
            'tactical_individual': tactical_actions,
            'tactical_average': tactical_avg,
            'combined': combined_action,
            'alignment': alignment,
            'strategy_used': self.strategy,
        }


def test():
    """Test multi-timeframe ensemble."""
    print("Testing Multi-Timeframe Ensemble...")
    print()

    # Check if ensembles exist
    strategic_dir = "train_results/ensemble"

    if not Path(strategic_dir).exists():
        print(f"Error: Strategic ensemble not found at {strategic_dir}")
        print("Run create_simple_ensemble.py first")
        return

    # Load ensemble
    ensemble = MultiTimeframeEnsemble(
        strategic_dir=strategic_dir,
        tactical_dirs=None,  # No tactical yet
        strategy="timing_refinement"
    )

    # Test with dummy state
    state_dim = 87  # Typical state dimension
    dummy_state = torch.randn(1, state_dim)

    action = ensemble.act(dummy_state)

    print("Test Results:")
    print(f"  Input shape: {dummy_state.shape}")
    print(f"  Output shape: {action.shape}")
    print(f"  Sample actions: {action[0, :5]}")
    print()
    print("✓ Multi-Timeframe Ensemble working!")
    print()
    print("Next steps:")
    print("  1. Train tactical models: ./train_multi_timeframe.sh --timeframe 5m")
    print("  2. Create tactical ensemble: python create_simple_ensemble.py --study <5m_study>")
    print("  3. Use with paper trading: --use-multi-timeframe flag")


if __name__ == "__main__":
    test()
