#!/usr/bin/env python3
"""
Launch Maximum Performance Training
- 1000 trials
- Parallel workers
- Maximum GPU/RAM utilization
- Aggressive batch sizes
"""

import subprocess
import time
import sys

# Configuration
N_PARALLEL_WORKERS = 3  # Run 3 training processes in parallel
TOTAL_TRIALS = 1000
STUDY_NAME = 'cappuccino_cge_1000trials'
GPU_ID = 0

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║         LAUNCHING MAXIMUM PERFORMANCE TRAINING - 1000 TRIALS          ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

Configuration:
  • Total trials: {TOTAL_TRIALS}
  • Parallel workers: {N_PARALLEL_WORKERS}
  • Study: {STUDY_NAME}
  • GPU: {GPU_ID}
  • Strategy: Each worker runs trials in parallel
  • Database: Shared SQLite with WAL mode

Each worker will:
  ✓ Use aggressive batch sizes (8192-16384)
  ✓ Use large network dimensions (2048-4096)
  ✓ Pull trials from shared queue
  ✓ Maximize GPU/RAM usage

Starting workers...
""")

processes = []

for worker_id in range(N_PARALLEL_WORKERS):
    cmd = [
        'python3',
        '1_optimize_unified.py',
        '--n-trials', str(TOTAL_TRIALS),
        '--gpu', str(GPU_ID),
        '--study-name', STUDY_NAME,
        '--use-best-ranges',  # Use tightened ranges for faster convergence
    ]

    log_file = f'training_worker_{worker_id}.log'

    print(f"\n🚀 Starting Worker {worker_id + 1}/{N_PARALLEL_WORKERS}")
    print(f"   Log: {log_file}")

    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd='/opt/user-data/experiment/cappuccino'
        )

    processes.append({
        'id': worker_id,
        'pid': proc.pid,
        'process': proc,
        'log': log_file
    })

    print(f"   PID: {proc.pid} ✓")

    # Stagger starts slightly
    time.sleep(2)

print(f"\n{'='*70}")
print(f"✅ ALL {N_PARALLEL_WORKERS} WORKERS LAUNCHED")
print(f"{'='*70}\n")

print("Worker Details:")
for p in processes:
    print(f"  Worker {p['id']}: PID {p['pid']}, Log: {p['log']}")

print(f"\nMonitoring Commands:")
print(f"  # Watch all workers")
print(f"  tail -f training_worker_*.log")
print(f"")
print(f"  # Check GPU usage")
print(f"  nvidia-smi")
print(f"")
print(f"  # Check progress")
print(f"  python3 -c \"import optuna; s = optuna.load_study(study_name='{STUDY_NAME}', storage='sqlite:///databases/optuna_cappuccino.db'); print(f'{{len([t for t in s.trials if t.state.name==\\\"COMPLETE\\\"])}}/{TOTAL_TRIALS} complete')\"")
print(f"")
print(f"  # Stop all workers")
print(f"  kill {' '.join([str(p['pid']) for p in processes])}")

print(f"\n{'='*70}")
print(f"Training running in background. Workers will complete ~{TOTAL_TRIALS} trials total.")
print(f"Expected completion: 12-24 hours")
print(f"{'='*70}\n")

# Save PIDs
with open('training_workers.pids', 'w') as f:
    for p in processes:
        f.write(f"{p['pid']}\n")

print("✓ PIDs saved to training_workers.pids")
print("✓ Full training launched at maximum capacity!\n")
