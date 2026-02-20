# Cappuccino

DRL crypto trading system using PPO + CPCV cross-validation, optimized with Optuna on an AMD RX 7900 GRE (ROCm).

## Quick Start

### Start training workers
```bash
bash start_safe_workers.sh
```

### Monitor training
```bash
python monitor_training_dashboard.py --watch
# or tail logs directly:
tail -f logs/worker_safe_1.log logs/worker_safe_2.log
```

### Pause / resume workers
```bash
# Pause
kill -STOP $(awk '{print $1}' logs/worker_pids.txt)
# Resume
kill -CONT $(awk '{print $1}' logs/worker_pids.txt)
```

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/training/1_optimize_unified.py` | Main Optuna training loop (PPO + CPCV) |
| `start_safe_workers.sh` | Launch 2 workers with OOM protection |
| `monitor_training_dashboard.py` | Live training dashboard (auto-detects study) |
| `monitor_progress.py` | Compact progress summary |
| `scripts/optimization/analyze_training_results.py` | Detailed Sharpe/param analysis |
| `scripts/data/0_dl_trainval_data.py` | Download/update training data |
| `scripts/deployment/paper_trader_alpaca_polling.py` | Paper trading via Alpaca |

## Architecture

```
environment_Alpaca_gpu.py     # Fully GPU-resident trading env (PyTorch)
train/run.py                  # ElegantRL PPO training loop
train/replay_buffer.py        # On-policy replay buffer
utils/function_train_test.py  # CPCV split train+test wrapper
utils/function_CPCV.py        # Combinatorially Purged K-Fold CV
utils/study_config.py         # Auto-detect active Optuna study
```

**State space:** `[cash, positions(7), tech_indicators(98) × lookback]`
**Action space:** Continuous position sizing per asset (7 crypto assets)
**Objective:** Maximize Sharpe ratio across CPCV folds
**GPU env:** 8 parallel envs, all tensor ops on GPU — no CPU sync during rollout

## Database

Optuna study: `databases/optuna_cappuccino.db`
Active study name stored in `.current_study` (written by `start_safe_workers.sh`).

Quick DB queries:
```bash
# Best trials
sqlite3 databases/optuna_cappuccino.db "
SELECT number, value FROM trials t
JOIN trial_values tv ON t.trial_id=tv.trial_id
WHERE study_id=(SELECT study_id FROM studies WHERE study_name='$(cat .current_study)')
  AND state='COMPLETE' ORDER BY value DESC LIMIT 10;"

# Current state counts
sqlite3 databases/optuna_cappuccino.db "
SELECT state, COUNT(*) FROM trials
WHERE study_id=(SELECT study_id FROM studies WHERE study_name='$(cat .current_study)')
GROUP BY state;"
```

## Hardware

- GPU: AMD RX 7900 GRE — 16 GB VRAM (ROCm / gfx1100)
- RAM: ~32 GB (workers use ~6–8 GB each at N_ENVS=8)
- Workers: 2 parallel Optuna workers, same SQLite DB

## Troubleshooting

**Stale RUNNING trials after crash:**
```bash
sqlite3 databases/optuna_cappuccino.db "
UPDATE trials SET state='FAIL'
WHERE study_id=(SELECT study_id FROM studies WHERE study_name='$(cat .current_study)')
  AND state='RUNNING';"
```

**Workers not starting / GPU idle:**
Check for memory cascades in logs — `grep 'Low RAM\|Emergency\|TrialPruned' logs/worker_safe_*.log | tail -20`

**4+ workers running (watchdog spawned extras):**
```bash
pkill -f 1_optimize_unified.py
# then restart cleanly:
truncate -s 0 logs/worker_pids.txt && bash start_safe_workers.sh
```
