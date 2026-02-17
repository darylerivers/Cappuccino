# Cappuccino System Diagnostics - Quick Start

## What I Created For You

I've created a complete diagnostic system to help you troubleshoot issues:

### 1. **Automated Diagnostic Script** (`./diagnose.sh`)
A one-command health check that tests:
- Configuration files
- Database connectivity
- GPU availability
- Disk space and memory
- All running components
- API connections
- Recent errors

**Usage:**
```bash
./diagnose.sh
```

**Exit codes:**
- `0` = All good
- `1` = Errors detected (some components not working)
- `2` = Critical issues (system won't function)

---

### 2. **Comprehensive Diagnostic Guide** (`DIAGNOSTIC_GUIDE.md`)
A complete troubleshooting manual with:
- Component-by-component diagnostics
- Common problems and solutions
- Configuration issue fixes
- Resource problem resolution
- Emergency recovery procedures
- Monitoring best practices

**When to use:** When `diagnose.sh` finds issues or you need detailed troubleshooting steps.

---

### 3. **Quick Reference Card** (`QUICK_DIAGNOSTICS.md`)
A one-page cheat sheet with:
- First response checklist
- Common problems & instant solutions
- Emergency commands
- Key file locations
- Monitoring commands

**When to use:** For quick fixes when you know what's wrong or need a fast reference.

---

## Quick Start Guide

### When Something Goes Wrong

**Step 1:** Run automated diagnostics
```bash
./diagnose.sh
```

**Step 2:** Review the output
- ✓ (green) = Working correctly
- ⚠ (yellow) = Warning - may need attention
- ✗ (red) = Error - needs fixing
- ✗✗ (red) = Critical - system won't work

**Step 3:** Fix issues
- For common problems, see `QUICK_DIAGNOSTICS.md`
- For detailed troubleshooting, see `DIAGNOSTIC_GUIDE.md`

**Step 4:** Verify the fix
```bash
./diagnose.sh
./verify_system.sh
```

---

## Common Scenarios

### Scenario 1: System Not Responding
```bash
# Quick check
./diagnose.sh

# If components are stopped, restart
./start_automation.sh
./start_training.sh
```

### Scenario 2: Components Using Wrong Study
```bash
# Check configuration
./verify_system.sh

# If mismatched, restart with correct config
./stop_automation.sh
pkill -f 1_optimize_unified.py
./start_automation.sh
./start_training.sh
```

### Scenario 3: Paper Trader Crashing
```bash
# Check logs for the error
tail -100 logs/paper_trading_live.log

# Common fixes are in QUICK_DIAGNOSTICS.md
```

### Scenario 4: Training Not Progressing
```bash
# Check workers
ps aux | grep 1_optimize

# Check GPU
nvidia-smi

# Check worker logs
tail -100 logs/worker_*.log
```

### Scenario 5: Complete System Reset Needed
```bash
# Stop everything
./stop_automation.sh
pkill -f 1_optimize_unified.py

# Clean stale PIDs
rm -f deployments/*.pid

# Restart
./start_automation.sh
./start_training.sh

# Verify
./diagnose.sh
```

---

## Daily Monitoring

Add to your routine:

```bash
# Morning check
./diagnose.sh

# During the day (optional)
./status_automation.sh

# Evening check
./verify_system.sh
```

---

## Current System Status

Based on the diagnostic run just now:

**Working:**
- ✓ Configuration files present
- ✓ Database accessible (686 trials)
- ✓ GPU working (99% utilization)
- ✓ 3 training workers on correct study
- ✓ System Watchdog running
- ✓ Performance Monitor running
- ✓ Ensemble Updater running
- ✓ Paper trader running

**Needs Attention:**
- ✗ Auto-Model Deployer has stale PID (fix: `rm deployments/auto_deployer.pid && ./start_automation.sh`)
- ⚠ Some errors in logs (review: `grep -i error logs/*.log | tail -20`)

**To fix the Auto-Model Deployer:**
```bash
rm deployments/auto_deployer.pid
./start_automation.sh
```

---

## Document Index

| Document | Purpose | When to Use |
|----------|---------|-------------|
| `diagnose.sh` | Automated health check | First thing when troubleshooting |
| `QUICK_DIAGNOSTICS.md` | Quick reference card | Fast fixes, common problems |
| `DIAGNOSTIC_GUIDE.md` | Complete troubleshooting manual | Detailed investigation |
| `verify_system.sh` | Configuration consistency check | After config changes |
| `status_automation.sh` | Component status check | Regular monitoring |

---

## Tips

1. **Run diagnostics regularly** - Don't wait for problems
   ```bash
   # Add to crontab for daily checks
   0 9 * * * cd /home/mrc/experiment/cappuccino && ./diagnose.sh
   ```

2. **Monitor logs** - Catch issues early
   ```bash
   tail -f logs/*.log
   ```

3. **Keep backups** - Before major changes
   ```bash
   cp databases/optuna_cappuccino.db backups/backup_$(date +%Y%m%d).db
   ```

4. **Check GPU health** - GPU is critical for training
   ```bash
   watch -n 2 nvidia-smi
   ```

5. **Monitor disk space** - Running out causes crashes
   ```bash
   df -h
   ```

---

## Support Resources

1. **Diagnostic script:** `./diagnose.sh`
2. **Quick fixes:** `QUICK_DIAGNOSTICS.md`
3. **Detailed guide:** `DIAGNOSTIC_GUIDE.md`
4. **Log files:** `logs/` directory
5. **State files:** `deployments/` directory

---

## Next Steps

1. Fix the current issues:
   ```bash
   rm deployments/auto_deployer.pid
   ./start_automation.sh
   ./diagnose.sh  # Verify fix
   ```

2. Set up daily monitoring:
   - Run `./diagnose.sh` every morning
   - Check `./status_automation.sh` periodically
   - Review logs weekly

3. Familiarize yourself with:
   - `QUICK_DIAGNOSTICS.md` for common fixes
   - `DIAGNOSTIC_GUIDE.md` for detailed troubleshooting

---

## Summary

You now have a complete diagnostic system:
- **Automated checks** via `./diagnose.sh`
- **Quick fixes** in `QUICK_DIAGNOSTICS.md`
- **Detailed troubleshooting** in `DIAGNOSTIC_GUIDE.md`

When something goes wrong:
1. Run `./diagnose.sh`
2. Check the relevant guide
3. Apply the fix
4. Verify with `./diagnose.sh`

Your system should be much easier to maintain and troubleshoot now!
