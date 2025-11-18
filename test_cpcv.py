#!/usr/bin/env python3
"""Test CPCV setup to find bottleneck"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory
PARENT_DIR = Path(__file__).parent.parent / "ghost/FinRL_Crypto"
sys.path.insert(0, str(PARENT_DIR))

from function_CPCV import CombPurgedKFoldCV

print("Loading data...")
# Simple test data
n_samples = 1614
tech_array = np.random.randn(n_samples, 77)
time_array = pd.date_range('2023-07-25', periods=n_samples, freq='1h')

print(f"Creating dataframe with {n_samples} samples...")
data = pd.DataFrame(tech_array)
data = data.set_index(pd.DatetimeIndex(time_array))
data = data[:-10]  # Remove last 10

y = pd.Series([0] * len(data), index=data.index)
prediction_times = pd.Series(data.index, index=data.index)
evaluation_times = pd.Series(data.index, index=data.index)

print("Creating CPCV splitter...")
n_total_groups = 4
k_test_groups = 2
embargo_td = pd.Timedelta(hours=50)

cv = CombPurgedKFoldCV(
    n_splits=n_total_groups,
    n_test_splits=k_test_groups,
    embargo_td=embargo_td
)

print("Starting to generate splits...")
splits = list(cv.split(data, y, pred_times=prediction_times, eval_times=evaluation_times))
print(f"Generated {len(splits)} splits successfully!")

for i, (train_idx, test_idx) in enumerate(splits):
    print(f"  Split {i+1}: train={len(train_idx)}, test={len(test_idx)}")

print("CPCV test completed successfully!")
