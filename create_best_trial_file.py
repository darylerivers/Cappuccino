#!/usr/bin/env python3
"""Create best_trial file for a specific trial."""
import pickle
import sys
from pathlib import Path

import optuna

# Get trial info from command line
study_name = sys.argv[1]
trial_number = int(sys.argv[2])
output_dir = sys.argv[3]

# Load study
storage = "sqlite:///databases/ensemble_ft_campaign.db"
study = optuna.load_study(study_name=study_name, storage=storage)

# Get the specific trial
trial = None
for t in study.trials:
    if t.number == trial_number:
        trial = t
        break

if trial is None:
    print(f"Trial #{trial_number} not found in study {study_name}")
    sys.exit(1)

# Save to best_trial file
output_path = Path(output_dir) / "best_trial"
with open(output_path, 'wb') as f:
    pickle.dump(trial, f)

print(f"✓ Created {output_path}")
print(f"  Trial #{trial.number}")
print(f"  State: {trial.state}")
print(f"  Value: {trial.value}")
