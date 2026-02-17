#!/usr/bin/env python3
"""
Ultra Simple Ensemble - Just load state dicts and average raw predictions.
"""

import json
import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Load active study from environment
from dotenv import load_dotenv
load_dotenv('.env.training')
_DEFAULT_STUDY = os.getenv('ACTIVE_STUDY_NAME', 'cappuccino_week_20251206')


class UltraSimpleEnsemble:
    """Load actor state dicts and average their outputs."""

    def __init__(self, manifest_path: str = "train_results/ensemble/ensemble_manifest.json", gpu_id: int = -1, db_path: str = "databases/optuna_cappuccino.db"):
        self.device = torch.device("cpu" if gpu_id < 0 else f"cuda:{gpu_id}")
        self.state_dicts = []
        self.trial_params = []
        self.db_path = db_path

        print(f"\n{'='*80}")
        print("LOADING ULTRA SIMPLE ENSEMBLE")
        print(f"{'='*80}")

        # Load manifest
        with Path(manifest_path).open("r") as f:
            self.manifest = json.load(f)

        print(f"Ensemble: {self.manifest['model_count']} models")
        print(f"Mean value: {self.manifest['mean_value']:.6f}")
        print()

        # Load models with hyperparameters
        self._load_models()

        # Store device for compatibility
        self.act = self._act_method

    def _load_hyperparameters_from_db(self, trial_number: int, study_name: str = None) -> dict:
        """Load trial hyperparameters from Optuna database."""
        if study_name is None:
            study_name = _DEFAULT_STUDY

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get trial params
        cursor.execute("""
            SELECT tp.param_name, tp.param_value
            FROM trial_params tp
            JOIN trials t ON tp.trial_id = t.trial_id
            WHERE t.number = ? AND t.study_id = (
                SELECT study_id FROM studies WHERE study_name = ?
            )
        """, (trial_number, study_name))

        params = {}
        for param_name, param_value in cursor.fetchall():
            params[param_name] = param_value

        conn.close()
        return params

    def _load_models(self):
        """Load all models and their hyperparameters."""
        print("Loading models...")

        trial_paths = self.manifest['trial_paths']
        actor_paths = self.manifest['actor_paths']
        trial_numbers = self.manifest['trial_numbers']
        trial_values = self.manifest['trial_values']

        self.inferred_lookbacks = []  # Store actual lookback for each model

        for i, (trial_num, trial_path, actor_path, value) in enumerate(
            zip(trial_numbers, trial_paths, actor_paths, trial_values), 1
        ):
            try:
                # Load hyperparameters from database
                params = self._load_hyperparameters_from_db(trial_num)

                # Load actor weights
                state_dict = torch.load(actor_path, map_location=self.device, weights_only=True)

                # INFER actual lookback from model's input dimension
                # State format: 1 (cash) + 7 (stocks) + 77*lookback (tech indicators)
                if 'net.0.weight' in state_dict:
                    input_dim = state_dict['net.0.weight'].shape[1]
                    inferred_lookback = int(round((input_dim - 8) / 77))
                    self.inferred_lookbacks.append(inferred_lookback)
                else:
                    # Fallback to database value
                    inferred_lookback = int(params.get('lookback', 2))
                    self.inferred_lookbacks.append(inferred_lookback)

                self.state_dicts.append(state_dict)
                self.trial_params.append(params)

                db_lookback = int(params.get('lookback', 0))
                if db_lookback != inferred_lookback:
                    print(f"  ✓ Model {i}: Trial #{trial_num} (value={value:.6f}, lookback={inferred_lookback}, db says {db_lookback})")
                else:
                    print(f"  ✓ Model {i}: Trial #{trial_num} (value={value:.6f}, lookback={inferred_lookback})")
            except Exception as e:
                print(f"  ✗ Model {i}: Failed - {e}")

        print(f"\n✓ Loaded {len(self.state_dicts)} models")

        # Determine common hyperparameters using INFERRED lookbacks (not database values)
        if self.inferred_lookbacks:
            # Use MAXIMUM lookback across all models to ensure we have enough history
            # Each model will receive the full state and use what it needs
            self.lookback = max(self.inferred_lookbacks)
            self.model_lookbacks = self.inferred_lookbacks  # Use inferred values for extraction
            print(f"✓ Using lookback: {self.lookback} (max of {min(self.inferred_lookbacks)}-{max(self.inferred_lookbacks)} across models)")
        else:
            self.lookback = 60
            self.model_lookbacks = [60]
            print("⚠️  No models loaded, using default lookback: 60")

        print(f"{'='*80}\n")

    def get_required_hyperparameters(self):
        """Get hyperparameters needed for environment."""
        if not self.trial_params or not any(p for p in self.trial_params if p):
            return {
                'lookback': 3,
                'norm_cash': 2**-13,
                'norm_stocks': 2**-7,
                'norm_tech': 2**-15,
                'norm_reward': 2**-8,
                'norm_action': 25000,
                'time_decay_floor': 0.3,
                'min_cash_reserve': 0.1,
                'concentration_penalty': 0.1,
            }

        # Find first valid params
        params = next((p for p in self.trial_params if p), None)
        if not params:
            # Recursive call to get defaults
            temp_params = []
            self.trial_params = temp_params
            result = self.get_required_hyperparameters()
            self.trial_params = [params]
            return result

        return {
            'lookback': self.lookback,  # Use the max lookback we calculated in _load_models()
            'norm_cash': float(2 ** params.get('norm_cash_exp', -13)),
            'norm_stocks': float(2 ** params.get('norm_stocks_exp', -7)),
            'norm_tech': float(2 ** params.get('norm_tech_exp', -15)),
            'norm_reward': float(2 ** params.get('norm_reward_exp', -8)),
            'norm_action': float(params.get('norm_action', 25000)),
            'time_decay_floor': float(params.get('time_decay_floor', 0.3)),
            'min_cash_reserve': float(params.get('min_cash_reserve', 0.1)),
            'concentration_penalty': float(params.get('concentration_penalty', 0.1)),
        }

    def _act_method(self, state: torch.Tensor) -> torch.Tensor:
        """
        Run state through all models and average outputs.

        State is constructed with max lookback across all models.
        Extract appropriate sub-state for each model based on its individual lookback.

        Args:
            state: Input tensor (batch_size, state_dim) with max lookback

        Returns:
            Averaged action tensor
        """
        predictions = []

        with torch.no_grad():
            state = state.to(self.device)

            for i, state_dict in enumerate(self.state_dicts):
                # Extract sub-state for this model's lookback
                # State format: [cash, stocks[n_tickers], tech[n_tickers * n_indicators * lookback]]
                # We need to extract only the lookback steps this model needs

                n_tickers = 7  # Number of crypto assets
                model_lookback = self.model_lookbacks[i] if hasattr(self, 'model_lookbacks') else self.lookback

                # Calculate state dimensions
                base_features = 1 + n_tickers  # cash + stocks
                tech_features_per_step = (state.shape[1] - base_features) // self.lookback  # features per lookback step

                # Extract: base features + (tech features * model's lookback)
                if model_lookback == self.lookback:
                    # Model uses same lookback as environment - use full state
                    x = state
                else:
                    # Extract subset of state for this model's lookback
                    # Take base features + first (model_lookback) timesteps of tech features
                    model_state_dim = base_features + (tech_features_per_step * model_lookback)
                    x = state[:, :model_state_dim]

                # Manual forward pass through network layers
                # Layer 0: Linear + ReLU
                x = torch.nn.functional.linear(x, state_dict['net.0.weight'], state_dict['net.0.bias'])
                x = torch.relu(x)

                # Layer 2: Linear + ReLU
                x = torch.nn.functional.linear(x, state_dict['net.2.weight'], state_dict['net.2.bias'])
                x = torch.relu(x)

                # Layer 4: Linear (output)
                x = torch.nn.functional.linear(x, state_dict['net.4.weight'], state_dict['net.4.bias'])

                predictions.append(x.cpu())

                # Store individual prediction for logging
                if not hasattr(self, '_last_individual_predictions'):
                    self._last_individual_predictions = []
                if i == 0:
                    self._last_individual_predictions = []
                self._last_individual_predictions.append(x.cpu().numpy())

        # Average predictions
        ensemble_action = torch.stack(predictions).mean(dim=0).to(self.device)

        # Store ensemble decision
        self._last_ensemble_action = ensemble_action.cpu().numpy()

        return ensemble_action

    def get_voting_breakdown(self, ticker_names: list = None) -> dict:
        """
        Get the breakdown of how each model voted.

        Returns:
            Dictionary with individual predictions and ensemble average
        """
        if not hasattr(self, '_last_individual_predictions'):
            return {'error': 'No predictions yet'}

        breakdown = {
            'individual_predictions': self._last_individual_predictions,
            'ensemble_average': self._last_ensemble_action,
            'num_models': len(self._last_individual_predictions),
        }

        if ticker_names:
            breakdown['ticker_names'] = ticker_names

        return breakdown


def test():
    """Test."""
    print("Testing ultra simple ensemble...")

    ensemble = UltraSimpleEnsemble(gpu_id=-1)

    # Get required dimensions from first model
    first_model = ensemble.state_dicts[0]
    input_dim = first_model['net.0.weight'].shape[1]

    # Dummy state (correct dimensions)
    state = torch.randn(1, input_dim)
    action = ensemble.act(state)

    print(f"Test:")
    print(f"  State: {state.shape}")
    print(f"  Action: {action.shape}")
    print(f"  Sample: {action[0, :5]}")
    print("\n✓ Ensemble working!")
    print("\nUsage:")
    print("  Replace the single agent in paper_trader_alpaca_polling.py")
    print("  with ensemble.act() - it returns averaged predictions from 10 models")


if __name__ == "__main__":
    test()
