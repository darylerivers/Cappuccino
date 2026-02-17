# Two-Phase Training - Automation Integration Guide

## Quick Start

The two-phase training system is now fully integrated with Cappuccino's automation system. It can run automatically on a schedule or be triggered manually.

---

## Configuration

Edit `.env.training` to configure two-phase training:

```bash
# Two-Phase Training Scheduler
TWO_PHASE_ENABLED=true              # Enable/disable scheduler
TWO_PHASE_SCHEDULE="weekly"         # weekly, monthly, or custom
TWO_PHASE_DAY="sunday"              # Day of week (for weekly)
TWO_PHASE_TIME="02:00"              # Start time (24h format)
TWO_PHASE_MODE="mini"               # mini (20 trials) or full (900 trials)
TWO_PHASE_AUTO_DEPLOY=true          # Auto-deploy winning models
TWO_PHASE_NOTIFICATION=true         # Log notifications on completion
```

### Configuration Options

| Setting | Values | Description |
|---------|--------|-------------|
| `TWO_PHASE_ENABLED` | `true`, `false` | Enable/disable automated training |
| `TWO_PHASE_SCHEDULE` | `weekly`, `monthly` | Training frequency |
| `TWO_PHASE_DAY` | `monday`, `tuesday`, ..., `sunday` | Day for weekly runs |
| `TWO_PHASE_TIME` | `HH:MM` (24h) | Time to start training |
| `TWO_PHASE_MODE` | `mini`, `full` | Mini test (20 trials) or full (900 trials) |
| `TWO_PHASE_AUTO_DEPLOY` | `true`, `false` | Automatically deploy winning models |
| `TWO_PHASE_NOTIFICATION` | `true`, `false` | Log notifications on completion |

---

## Using with Automation

### Enable and Start

1. **Enable in config**:
   ```bash
   # Edit .env.training
   nano .env.training

   # Set TWO_PHASE_ENABLED=true
   # Configure schedule (e.g., Sunday at 2 AM)
   ```

2. **Start automation** (includes two-phase scheduler):
   ```bash
   ./start_automation.sh
   ```

3. **Check status**:
   ```bash
   ./status_automation.sh
   ```

   You'll see:
   ```
   ✓ Two-Phase Scheduler
     Status: RUNNING
     PID:    12345
     ...

   Two-Phase Training Schedule
   ============================
     Next run:     2025-12-22 02:00:00
     Time until:   48.5 hours
     Last run:     2025-12-15T03:45:12
     Total runs:   3/3 successful
     Status:       IDLE
   ```

### Stop Automation

```bash
./stop_automation.sh
```

This stops all automation including the two-phase scheduler.

---

## Manual Control

### Run Immediate Training (One-Shot)

Run training immediately without waiting for schedule:

```bash
# Mini test (20 trials, ~30 minutes)
python two_phase_scheduler.py --run-now

# Or run directly
python run_two_phase_training.py --mini-test
```

### Check Schedule

View current schedule without starting daemon:

```bash
python two_phase_scheduler.py --show-schedule
```

Output:
```
============================================================
Two-Phase Training Schedule
============================================================
Enabled:      true
Schedule:     weekly
Day:          sunday
Time:         02:00
Mode:         mini
Auto-deploy:  true
Notification: true

Next run:     2025-12-22 02:00:00
Time until:   48.5 hours

Last run:     2025-12-15T03:45:12

Total runs:   3
Successful:   3/3

Recent runs:
  ✓ 2025-12-15T03:45:12 (0.52h) - mini
  ✓ 2025-12-08T02:00:45 (0.48h) - mini
  ✓ 2025-12-01T02:01:15 (0.51h) - mini
============================================================
```

### Run Standalone (No Automation)

Run two-phase training without the scheduler:

```bash
# Mini test
python run_two_phase_training.py --mini-test

# Full run
python run_two_phase_training.py

# Individual phases
python run_two_phase_training.py --phase1-only
python run_two_phase_training.py --phase2-only
```

---

## Integration Features

### 1. Auto-Deployment

When `TWO_PHASE_AUTO_DEPLOY=true`, winning models are automatically:
- Copied to `deployments/` directory
- Saved with timestamp: `two_phase_{algorithm}_{trial}_{timestamp}.pth`
- Logged in `deployments/two_phase_deployment.json`

Example deployment state:
```json
{
  "two_phase_deployment": {
    "algorithm": "ppo",
    "trial_number": 142,
    "sharpe_bot": 1.7234,
    "sharpe_hodl": 1.2145,
    "value": 0.4812,
    "model_path": "deployments/two_phase_ppo_142_20251216_035512.pth",
    "deployed_at": "2025-12-16T03:55:12"
  }
}
```

### 2. Notifications

When `TWO_PHASE_NOTIFICATION=true`, completion notifications are logged to:
- `logs/two_phase_notifications.jsonl`

Each notification includes:
- Timestamp
- Success/failure status
- Duration
- Phase 1 winner (time-frame/interval)
- Phase 2 winner (algorithm)
- Training mode (mini/full)

Example notification:
```json
{
  "timestamp": "2025-12-16T03:55:15",
  "type": "two_phase_training",
  "success": true,
  "duration": 1825.4,
  "mode": "mini",
  "message": "Two-Phase Training Completed Successfully!\nDuration: 0.51 hours\nPhase 1 Winner: 7d @ 1h\nPhase 2 Winner: PPO\n"
}
```

### 3. Progress Monitoring

Monitor training progress:

```bash
# Check automation status
./status_automation.sh

# Follow scheduler log
tail -f logs/two_phase_scheduler.log

# Follow training console output
tail -f logs/two_phase_scheduler_console.log
```

### 4. State Persistence

Scheduler state is saved in `deployments/two_phase_scheduler_state.json`:

```json
{
  "last_run": "2025-12-15T03:45:12",
  "next_run": "2025-12-22T02:00:00",
  "status": "idle",
  "runs": [
    {
      "timestamp": "2025-12-15T03:45:12",
      "duration": 1825.4,
      "mode": "mini",
      "success": true,
      "results": { ... }
    }
  ]
}
```

This allows resume after system restart.

---

## Typical Workflows

### Weekly Mini-Test Strategy

**Best for**: Regular model refresh without heavy compute

```bash
# 1. Configure weekly mini-tests
nano .env.training
```
```
TWO_PHASE_ENABLED=true
TWO_PHASE_SCHEDULE="weekly"
TWO_PHASE_DAY="sunday"
TWO_PHASE_TIME="02:00"
TWO_PHASE_MODE="mini"
TWO_PHASE_AUTO_DEPLOY=true
```
```bash
# 2. Start automation
./start_automation.sh

# 3. Monitor weekly
./status_automation.sh
```

**Result**: Every Sunday at 2 AM, runs 20-trial mini-test (~30 min), auto-deploys winner.

### Monthly Full Optimization

**Best for**: Comprehensive optimization on monthly basis

```bash
# Configure monthly full runs
nano .env.training
```
```
TWO_PHASE_ENABLED=true
TWO_PHASE_SCHEDULE="monthly"
TWO_PHASE_TIME="01:00"
TWO_PHASE_MODE="full"
TWO_PHASE_AUTO_DEPLOY=true
```
```bash
# Start automation
./start_automation.sh
```

**Result**: 1st of each month at 1 AM, runs full 900-trial optimization (~48-72 hours), auto-deploys winner.

### Manual Testing + Auto Production

**Best for**: Test manually, automate production runs

```bash
# 1. Run manual mini-test to validate
python run_two_phase_training.py --mini-test

# 2. Configure production schedule (disabled initially)
nano .env.training
```
```
TWO_PHASE_ENABLED=false  # Start disabled
TWO_PHASE_SCHEDULE="weekly"
TWO_PHASE_MODE="full"
```
```bash
# 3. When ready, enable
sed -i 's/TWO_PHASE_ENABLED=false/TWO_PHASE_ENABLED=true/' .env.training

# 4. Restart automation
./stop_automation.sh
./start_automation.sh
```

### Hybrid: Continuous Regular + Scheduled Two-Phase

**Best for**: Maximum performance

```bash
# 1. Run regular continuous optimization (existing system)
./start_training.sh  # 3 workers, continuous

# 2. Enable weekly two-phase mini-tests
nano .env.training
```
```
TWO_PHASE_ENABLED=true
TWO_PHASE_SCHEDULE="weekly"
TWO_PHASE_MODE="mini"
```
```bash
# 3. Start automation
./start_automation.sh
```

**Result**:
- Daily: Continuous regular optimization (11,160+ trials database)
- Weekly: Two-phase mini-test validates time-frames and features
- Best of both: Regular deep search + periodic strategic validation

---

## Troubleshooting

### Scheduler Not Starting

**Check 1**: Is it enabled?
```bash
grep TWO_PHASE_ENABLED .env.training
# Should show: TWO_PHASE_ENABLED=true
```

**Check 2**: Is it running?
```bash
ps aux | grep two_phase_scheduler
```

**Check 3**: Check logs
```bash
tail -n 50 logs/two_phase_scheduler.log
```

### Training Fails

**Check 1**: Prerequisites
```bash
python run_two_phase_training.py --mini-test
```

**Check 2**: Data files
```bash
ls -lh data/*.npy
# Should see price_array, tech_array, time_array
```

**Check 3**: Disk space
```bash
df -h .
```

**Check 4**: Check error logs
```bash
tail -n 100 logs/two_phase_scheduler.log
```

### Schedule Not Updating

**Issue**: Next run time stuck in past

**Solution**:
```bash
# Stop scheduler
./stop_automation.sh

# Remove state file
rm deployments/two_phase_scheduler_state.json

# Restart
./start_automation.sh

# Verify
./status_automation.sh
```

### Models Not Auto-Deploying

**Check 1**: Is auto-deploy enabled?
```bash
grep TWO_PHASE_AUTO_DEPLOY .env.training
# Should show: TWO_PHASE_AUTO_DEPLOY=true
```

**Check 2**: Check deployment directory
```bash
ls -lh deployments/two_phase_*.pth
```

**Check 3**: Check deployment state
```bash
cat deployments/two_phase_deployment.json
```

---

## Logs Reference

| Log File | Contents |
|----------|----------|
| `logs/two_phase_scheduler.log` | Scheduler daemon log (schedule checks, run triggers) |
| `logs/two_phase_scheduler_console.log` | Training process stdout/stderr |
| `logs/two_phase_notifications.jsonl` | Completion notifications (JSON lines) |
| `phase1_winner.json` | Phase 1 results (time-frame winner) |
| `phase2_comparison.json` | Phase 2 results (PPO vs DDQN) |
| `two_phase_training_report.json` | Final comprehensive report |
| `deployments/two_phase_scheduler_state.json` | Scheduler state (runs, schedule) |
| `deployments/two_phase_deployment.json` | Latest deployed model info |

---

## System Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Cappuccino Automation System                    │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┐    ┌──────────────────────┐
│  start_automation.sh │───>│  .env.training       │
│                      │    │  Configuration       │
└──────────────────────┘    └──────────────────────┘
            │
            ├──> Auto-Model Deployer    (monitors Optuna DB)
            ├──> System Watchdog        (monitors processes)
            ├──> Performance Monitor    (tracks trading)
            ├──> Ensemble Updater       (updates top models)
            │
            └──> Two-Phase Scheduler    (NEW!)
                         │
                         ├──> Checks schedule (every minute)
                         ├──> Triggers: run_two_phase_training.py
                         │         ├──> Phase 1 Optimizer
                         │         └──> Phase 2 Optimizer
                         │
                         ├──> Auto-deploys winner
                         └──> Logs notifications

┌──────────────────────┐
│ status_automation.sh │───> Shows status of all components
└──────────────────────┘    including two-phase scheduler

┌──────────────────────┐
│  stop_automation.sh  │───> Stops all components
└──────────────────────┘    including two-phase scheduler
```

---

## Summary

The two-phase training system is now a first-class citizen in Cappuccino's automation:

✓ **Scheduled Runs**: Weekly/monthly automatic optimization
✓ **Auto-Deployment**: Winning models deployed automatically
✓ **Full Integration**: Works with existing automation scripts
✓ **Monitoring**: Status checks via `./status_automation.sh`
✓ **Persistence**: State saved, survives restarts
✓ **Flexible**: Run manually or on schedule
✓ **Notifications**: Logged completion status

**Enable it**:
```bash
# 1. Edit .env.training
# 2. Set TWO_PHASE_ENABLED=true
# 3. ./start_automation.sh
# 4. ./status_automation.sh
```

**That's it!** The system will automatically run two-phase optimization on schedule and deploy winning models.
