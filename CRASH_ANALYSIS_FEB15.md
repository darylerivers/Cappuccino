# Training Crash Analysis - February 15, 2026

## 🔴 Summary

**System crashed multiple times overnight despite memory safety system being active.**

- **3 crashes/reboots**: 23:30 (Feb 14), 00:05, 01:14 (Feb 15)
- **Root cause**: Emergency brake worked but memory stayed at 94.2% usage
- **Emergency brake activations**: 951 times (!)
- **Critical finding**: Worker watchdog was NOT running

---

## 📊 Timeline of Events

### Feb 14 - 23:30
- System crashed and rebooted

### Feb 15 - 00:05
- System crashed and rebooted again (35 minutes later)

### Feb 15 - 00:59:39
- Workers 11858 and 11944 died
- Watchdog detected and restarted as PIDs 77107 and 77180

### Feb 15 - 01:14
- **System crashed and rebooted** (hard crash)
- Watchdog logs stopped at 01:14:14
- This is the last crash before current uptime

### Feb 15 - 01:23:17
- Watchdog manually restarted after boot
- Found workers 77107/77180 dead
- Started new workers 17650 and 17734

### Feb 15 - 01:24:31
- Last watchdog log entry showing workers healthy
- **Watchdog then stopped running** (critical issue!)

### Feb 15 - 02:46:17
- Workers completed 500 trials each
- Emergency brake had fired 951 times
- Memory stuck at **29.41 GB / 31.23 GB (94.2%)**
- Only 1.82 GB available throughout

### Feb 15 - 11:53 (now)
- No workers running
- Watchdog not running
- System stable with 24GB free

---

## 🔍 Root Cause Analysis

### Critical Issue #1: Watchdog Not Running
```bash
# Expected: worker_watchdog.sh process
# Actual: No watchdog process found

ps aux | grep worker_watchdog | grep -v grep
# (no results)
```

**Impact**: Without watchdog:
- Workers ran for 1h+ without restart (memory leak accumulation)
- No memory-based emergency restarts
- No automatic recovery when workers died

### Critical Issue #2: Memory Stayed at 94.2%
```
Memory usage: 29.41 GB / 31.23 GB (94.2%)
Available: 1.82 GB (below 2.0 GB threshold)
Emergency brake triggered: 951 times
```

**What happened**:
1. Workers consumed memory over time (leak still present)
2. Memory reached 94.2% and stayed there
3. Emergency brake prevented NEW trials from starting
4. But didn't help REDUCE existing memory usage
5. System eventually crashed from sustained high memory

### Critical Issue #3: Emergency Brake Limitation

**What it did (correctly)**:
- ✅ Detected low memory (< 2GB available)
- ✅ Killed 951 trials before they could start
- ✅ Prevented memory from going to 100% briefly

**What it couldn't do**:
- ❌ Didn't reduce existing memory usage
- ❌ Couldn't force worker restart
- ❌ Couldn't prevent crash from sustained 94% usage

**Design limitation**: Emergency brake kills individual trials but doesn't have authority to:
- Restart the worker process (only watchdog can do that)
- Clear accumulated memory (only worker restart can do that)

---

## 📈 Memory Usage Pattern

```
Start:         ~4-5 GB (system baseline)
During trials: Growing slowly
Peak:          29.41 GB (94.2%) - STUCK HERE
Available:     1.82 GB (triggered emergency brake)
Crash point:   System couldn't sustain 94%+ for hours
```

**Emergency brake behavior**:
```
Trial starts → Check memory → 1.82GB < 2.0GB → Kill trial
Trial starts → Check memory → 1.82GB < 2.0GB → Kill trial
(repeated 951 times)
```

Workers kept trying to start new trials, emergency brake kept killing them, but memory never went down because:
1. Workers themselves were holding the memory
2. Only worker restart releases accumulated memory
3. Watchdog wasn't running to restart workers

---

## 🎯 Why Our Triple-Layer Defense Failed

### Layer 1: Memory Leak Fix ❓
**Status**: Partially working but leak still present
- Workers should stay at ~2-3GB
- Actually grew to consume ~15GB each (estimated)
- 2 workers × 15GB = 30GB ≈ 94% of 31GB

### Layer 2: Watchdog ❌ FAILED
**Status**: NOT RUNNING (critical failure)
- Should have restarted workers every 60min
- Should have detected memory > 8GB per worker
- Should have emergency restarted at system < 2GB
- **None of this happened because watchdog wasn't active**

### Layer 3: Emergency Brake ⚠️ WORKED BUT LIMITED
**Status**: Working as designed but insufficient
- Killed 951 trials successfully
- Prevented individual trial OOM
- **BUT** couldn't restart workers or reduce existing memory

---

## 🔧 What Needs to be Fixed

### Priority 1: Ensure Watchdog Stays Running

**Problem**: Watchdog stops after system boots
```bash
# Last log entry
[2026-02-15 01:24:31] ✅ Worker 2 PID 17734 healthy...
# Then nothing - watchdog died
```

**Solution**: Make watchdog persistent as systemd service

### Priority 2: Enhance Emergency Brake

**Current**: Kills trials when memory < 2GB
**Needed**: Also kill the worker process when sustained high memory

**Proposal**:
- If emergency brake fires 10+ times in a row → Kill worker process
- Let watchdog restart it with fresh memory
- This creates emergency worker restart without waiting 60min

### Priority 3: Fix Remaining Memory Leak

**Evidence**: Workers growing from 2GB → 15GB over 1+ hour
**Location**: Somewhere in training loop (replay_buffer fix wasn't enough)

**Investigation needed**:
- Check if multiple datasets/arrays accumulating
- Check if GPU memory transfers not being freed
- Check if model checkpoints accumulating in RAM

### Priority 4: Add Memory Release Between Trials

**Proposal**:
- After each trial completes: force garbage collection
- Clear PyTorch cache explicitly
- Maybe even restart worker every N trials (e.g., 50)

---

## 🚨 Immediate Actions Required

### 1. Start Watchdog (NOW)
```bash
cd /opt/user-data/experiment/cappuccino
nohup bash worker_watchdog.sh > /dev/null 2>&1 &
echo $! > logs/watchdog.pid
```

### 2. Verify It's Running
```bash
ps aux | grep worker_watchdog | grep -v grep
tail -f logs/watchdog.log
```

### 3. Restart Training with Watchdog Active
```bash
./start_safe_workers.sh
```

### 4. Monitor Closely
```bash
# Watch memory every 10 seconds
watch -n 10 'free -h; echo "---"; ps aux | grep "1_optimize_unified" | grep -v grep'
```

---

## 📊 Current System Status

```bash
Workers:     0 running (stopped after crash)
Watchdog:    NOT RUNNING (critical)
Memory:      24GB free (safe now, but was 1.8GB)
Last crash:  01:14 (10.5 hours ago)
Uptime:      10h 38min (stable since last reboot)
```

**Good news**:
- System is stable now
- Emergency brake did prevent complete RAM exhaustion
- We have detailed logs showing exactly what happened

**Bad news**:
- Watchdog not running (must fix immediately)
- Memory leak still present (workers grew too large)
- System crashed 3 times despite safety measures

---

## 🎓 Lessons Learned

### What Worked
1. ✅ Emergency brake prevented instant crashes (gave us time)
2. ✅ Detailed logging showed exactly what happened
3. ✅ Workers completed 500 trials despite high memory pressure
4. ✅ Emergency brake killed 951 trials gracefully

### What Failed
1. ❌ Watchdog wasn't running (single point of failure)
2. ❌ Memory leak still present (fix incomplete)
3. ❌ Emergency brake can't restart workers (design limitation)
4. ❌ No systemd service to keep watchdog alive

### What We Need
1. 🔧 Systemd service for watchdog (persistence)
2. 🔧 Enhanced emergency brake that can restart workers
3. 🔧 Better memory leak fix (investigate further)
4. 🔧 Aggressive garbage collection between trials
5. 🔧 Worker restart every N trials (failsafe)

---

## 📋 Next Steps

### Immediate (Next 5 minutes)
1. Start watchdog manually
2. Verify it's running and logging
3. Check worker status

### Short-term (Today)
1. Create systemd service for watchdog
2. Enhance emergency brake to restart workers after 10+ triggers
3. Add aggressive GC between trials
4. Restart training with monitoring

### Medium-term (This week)
1. Deep investigation of memory leak
2. Profile memory usage during training
3. Consider reducing to 1 worker temporarily
4. Add worker restart every 50 trials

### Long-term (Next week)
1. Implement proper memory profiling
2. Fix leak at source
3. Stress test with 2 workers for 12+ hours
4. Document all fixes

---

## 🆘 Emergency Contacts

**If system crashes again**:
1. Check `journalctl --since "1 hour ago" | grep -i oom`
2. Check `logs/worker_safe_*.log` for emergency brake count
3. Check `logs/watchdog.log` to see if watchdog was running
4. Restart watchdog immediately after reboot
5. Consider reducing to 1 worker

**Files to check**:
- `logs/watchdog.log` - Watchdog status
- `logs/worker_safe_1.log` - Worker 1 activity
- `logs/worker_safe_2.log` - Worker 2 activity
- `free -h` - Current memory
- `ps aux | grep python` - Worker memory usage

---

**Generated**: 2026-02-15 11:53
**Crash count**: 3 reboots overnight
**Emergency brakes**: 951 activations
**Status**: System stable but watchdog not running (CRITICAL FIX NEEDED)
