# Pipeline V1 Failure Analysis

## Executive Summary

The current pipeline orchestrator has **5 critical failures** that prevent any trials from reaching paper trading. After hours of debugging, **0 out of 28 completed trials** have been successfully deployed.

## Critical Issues

### 1. CGE Stress Test Subprocess Failures

**Symptom**: All CGE tests fail with exit code 1, empty stderr

**Root Cause**:
- CGE runner generates Python script and executes as subprocess
- Script crashes silently (no error output captured)
- Divide-by-zero was fixed but deeper issues remain
- Error: Process fails without traceback or error message

**Impact**: 100% failure rate on CGE gate

**Evidence**:
```
[ERROR] Stress test failed with code 1
[ERROR] stderr:
```

### 2. Pipeline State Tracking Broken

**Symptom**: All trials show "not_in_pipeline" in viewer

**Root Cause**:
- State file: `pipeline/state/pipeline_state.json`
- Gets created but viewer can't read it properly
- State cleared between runs, losing progress
- No persistence across orchestrator restarts

**Impact**: Cannot track trial progress through stages

**Evidence**:
```
Pipeline Stage Summary:
  Not in pipeline: 28
  Deployed: 0
```

### 3. Trial Detection Only Processes New Trials

**Symptom**: Restarting orchestrator doesn't retry failed trials

**Root Cause**:
- Orchestrator only processes trials not in state file
- Once marked "failed", they're never retried
- No manual retry mechanism
- Clearing state file doesn't help (trials already processed)

**Impact**: Can't recover from failures

### 4. Daemon Mode Instability

**Symptom**: Orchestrator receives SIGTERM and dies randomly

**Root Causes**:
- No proper signal handling
- Gets killed by system or user commands
- Sleeps for 30 minutes (unresponsive)
- No health monitoring or auto-restart

**Impact**: Pipeline stops processing without notification

**Evidence**:
```
[19:22:31] Received signal 15, shutting down...
[20:31:54] Received signal 15, shutting down...
```

### 5. Complex Dependencies on Optuna Objects

**Symptom**: Paper trader can't load trials (expects Optuna trial objects)

**Root Cause**:
- Paper trader expects `trial.user_attrs`, `trial.params` etc.
- We're passing dicts
- Tight coupling to Optuna internals
- No abstraction layer

**Impact**: Can't deploy even if trials pass all gates

**Evidence**:
```python
AttributeError: 'dict' object has no attribute 'user_attrs'
```

## Secondary Issues

### 6. No Incremental Processing
- Must process all stages in one run
- Can't pause/resume
- Can't skip stages

### 7. Poor Error Handling
- Subprocess errors not captured
- No retry with backoff
- Silent failures common

### 8. No Health Monitoring
- No status endpoint
- No heartbeat
- Can't tell if running or stuck

### 9. Overcomplicated Architecture
- Multiple state files
- JSON + pickle + SQLite
- Hard to debug
- No single source of truth

## Statistics

| Metric | Value |
|--------|-------|
| Trials completed | 28 |
| Trials in pipeline | 0 |
| Trials deployed | 0 |
| Success rate | 0% |
| Hours debugging | 6+ |

## Conclusion

The pipeline V1 is **architecturally flawed** and needs a complete rewrite. Quick fixes have failed. A simpler, more robust design is required.

## Requirements for V2

1. **Simple**: Single SQLite database for all state
2. **Robust**: Proper error handling, retries, logging
3. **Testable**: Each component can run standalone
4. **Monitored**: Health checks, status reporting
5. **Flexible**: Stages can be skipped/disabled
6. **Decoupled**: No tight Optuna coupling
