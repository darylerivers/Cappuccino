#!/usr/bin/env python3
"""
Maximum GPU Utilization Training Script
- Forces large batch sizes (8192-16384)
- Forces large network dimensions (2048-4096)
- Optimized for RTX 3070 8GB VRAM
"""

import sys
import os

# Patch the original script's hyperparameter ranges
original_file = '/opt/user-data/experiment/cappuccino/1_optimize_unified.py'

# Read original
with open(original_file, 'r') as f:
    content = f.read()

# Modify batch size ranges to be much larger
# Original: batch_size = trial.suggest_int('batch_size', 512, 4096, log=True)
# New: Force 8192-16384
content = content.replace(
    "trial.suggest_int('batch_size', 512, 4096, log=True)",
    "trial.suggest_int('batch_size', 8192, 16384, log=True)"
)

# Modify network dimensions to be larger
# Original: net_dimension = trial.suggest_int('net_dimension', 512, 2048, step=64)
# New: Force 2048-4096
content = content.replace(
    "trial.suggest_int('net_dimension', 512, 2048, step=64)",
    "trial.suggest_int('net_dimension', 2048, 4096, step=128)"
)

# Also increase for best ranges mode
content = content.replace(
    "trial.suggest_int('batch_size', 2048, 4096, log=True)",
    "trial.suggest_int('batch_size', 8192, 16384, log=True)"
)
content = content.replace(
    "trial.suggest_int('net_dimension', 1024, 2048, step=64)",
    "trial.suggest_int('net_dimension', 2048, 4096, step=128)"
)

# Execute modified script
exec(content, {'__name__': '__main__'})
