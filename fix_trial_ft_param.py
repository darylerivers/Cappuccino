#!/usr/bin/env python3
"""Add use_ft_encoder parameter to Trial #91's best_trial file."""
import pickle
from pathlib import Path

# Load the trial
trial_path = Path("train_results/cwd_tests/trial_91_1h/best_trial")
with open(trial_path, 'rb') as f:
    trial = pickle.load(f)

print(f"Original params:")
print(f"  use_ft_encoder: {trial.params.get('use_ft_encoder', 'NOT SET')}")

# Add the missing parameter
trial._params['use_ft_encoder'] = True

# Also add it to distributions (for completeness)
from optuna.distributions import CategoricalDistribution
trial._distributions['use_ft_encoder'] = CategoricalDistribution(choices=[True, False])

print(f"\nUpdated params:")
print(f"  use_ft_encoder: {trial.params['use_ft_encoder']}")

# Save back
with open(trial_path, 'wb') as f:
    pickle.dump(trial, f)

print(f"\n✓ Updated {trial_path}")
