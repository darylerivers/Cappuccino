#!/usr/bin/env python3
"""Force maximum VRAM usage for training."""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

import torch
# Reserve 95% of VRAM upfront
torch.cuda.set_per_process_memory_fraction(0.95, 0)
# Preallocate
dummy = torch.randn(1024, 1024, 1024, dtype=torch.float16, device='cuda:0')
del dummy
torch.cuda.empty_cache()

# Now run the training
import sys
sys.argv = ['1_optimize_unified.py', '--n-trials', '2000', '--gpu', '0', '--study-name', 'maxvram', '--storage', 'sqlite:///databases/optuna_maxvram.db', '--data-dir', 'data/2year_fresh_20260112']

# Import and run main
from importlib import import_module
spec = import_module('1_optimize_unified')
