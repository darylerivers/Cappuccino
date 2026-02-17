#!/bin/bash
cd /opt/user-data/experiment/cappuccino
source activate_rocm_env.sh
python scripts/training/1_optimize_unified.py --study-name test_run_huge --n-trials 10
