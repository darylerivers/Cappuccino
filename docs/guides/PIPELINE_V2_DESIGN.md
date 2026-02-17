# Pipeline V2 Architecture Design

## Design Principles

1. **Simple**: One database, clear data flow
2. **Robust**: Graceful failures, retries, comprehensive logging
3. **Testable**: Each stage runs independently
4. **Observable**: Real-time status, health checks
5. **Flexible**: Stages can be enabled/disabled

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE V2 MANAGER                      │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Discovery  │───>│  Execution   │───>│  Monitoring  │ │
│  │              │    │              │    │              │ │
│  │ • Scan DB    │    │ • Run stages │    │ • Track      │ │
│  │ • Find new   │    │ • Handle     │    │ • Report     │ │
│  │ • Queue      │    │   errors     │    │ • Alert      │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  STATE DATABASE │
                    │   (SQLite)      │
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   BACKTEST    │    │   OPTIONAL    │    │ PAPER TRADING │
│   VALIDATOR   │    │   CGE STRESS  │    │   DEPLOYER    │
│               │    │               │    │               │
│ • Load model  │    │ • Disabled by │    │ • Start proc  │
│ • Run test    │    │   default     │    │ • Monitor     │
│ • Store       │    │               │    │ • Restart     │
└───────────────┘    └───────────────┘    └───────────────┘
```

## Database Schema

### Single SQLite Database: `pipeline_v2.db`

```sql
-- Trial tracking
CREATE TABLE trials (
    trial_id INTEGER PRIMARY KEY,
    trial_number INTEGER UNIQUE NOT NULL,
    value REAL,
    status TEXT, -- 'pending', 'processing', 'deployed', 'failed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stage tracking
CREATE TABLE stages (
    id INTEGER PRIMARY KEY,
    trial_id INTEGER,
    stage_name TEXT, -- 'backtest', 'cge', 'deploy'
    status TEXT, -- 'pending', 'running', 'passed', 'failed', 'skipped'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    error_message TEXT,
    results JSON,
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);

-- Deployment tracking
CREATE TABLE deployments (
    id INTEGER PRIMARY KEY,
    trial_id INTEGER,
    process_id INTEGER,
    started_at TIMESTAMP,
    stopped_at TIMESTAMP,
    status TEXT, -- 'running', 'stopped', 'crashed'
    log_file TEXT,
    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
);

-- System health
CREATE TABLE health (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    component TEXT, -- 'manager', 'backtest', 'deploy'
    status TEXT, -- 'healthy', 'degraded', 'down'
    message TEXT
);
```

## Core Components

### 1. Pipeline Manager (`pipeline_v2.py`)

**Responsibilities**:
- Discover new trials from Optuna DB
- Queue trials for processing
- Execute stages in order
- Handle retries with exponential backoff
- Log all activity
- Update state database

**Key Methods**:
```python
class PipelineV2:
    def discover_trials(self) -> List[int]
    def process_trial(self, trial_num: int) -> bool
    def execute_stage(self, trial_id: int, stage: str) -> bool
    def retry_failed(self, trial_id: int, stage: str) -> bool
    def get_status(self) -> Dict
    def run_daemon(self, interval: int = 300)  # 5 min checks
```

### 2. Backtest Validator (`backtest_v2.py`)

**Responsibilities**:
- Load trial parameters from Optuna DB
- Load model from filesystem
- Create environment with correct params
- Run validation
- Return structured results

**Key Methods**:
```python
class BacktestValidator:
    def validate(self, trial_num: int) -> Dict[str, Any]
    def load_model(self, model_dir: Path) -> Agent
    def load_params(self, trial_num: int) -> Dict
    def run_test(self, agent, env) -> Dict
```

### 3. Paper Trading Deployer (`deploy_v2.py`)

**Responsibilities**:
- Start paper trading process
- Monitor process health
- Restart on crash
- Log trading activity
- Stop on command

**Key Methods**:
```python
class PaperTradingDeployer:
    def deploy(self, trial_num: int) -> int  # Returns PID
    def monitor(self, pid: int) -> str  # Returns status
    def restart(self, trial_id: int) -> int
    def stop(self, pid: int) -> bool
    def get_logs(self, trial_id: int) -> str
```

## Stage Configuration

### Enabled Stages (configurable)
```json
{
  "stages": {
    "backtest": {
      "enabled": true,
      "max_attempts": 3,
      "timeout_seconds": 60
    },
    "cge_stress": {
      "enabled": false,
      "max_attempts": 2,
      "timeout_seconds": 300
    },
    "deploy": {
      "enabled": true,
      "max_attempts": 1,
      "auto_restart": true
    }
  }
}
```

## Error Handling

### Retry Strategy
- Exponential backoff: 5s, 30s, 5min
- Max 3 attempts per stage
- Different strategies per stage
- Permanent failures logged

### Failure Recovery
- Failed trials can be manually retried
- Stages can be skipped
- Deployments auto-restart on crash

## Monitoring & Observability

### Status Endpoint
```python
GET /status
{
  "pipeline": "healthy",
  "trials_pending": 28,
  "trials_processing": 2,
  "trials_deployed": 3,
  "last_check": "2026-01-29T21:30:00"
}
```

### Health Checks
- Pipeline heartbeat every 60s
- Component status checks
- Process monitoring
- Disk space warnings

### Logging
- Structured JSON logs
- Per-trial log files
- Centralized logging
- Log rotation

## Migration Path

### Step 1: Build V2 alongside V1
- Don't break existing system
- Test thoroughly
- Parallel operation

### Step 2: Migrate trials
- Import V1 state if needed
- Process backlog
- Verify deployments

### Step 3: Deprecate V1
- Stop V1 orchestrator
- Archive V1 code
- Update docs

## Success Criteria

- ✅ Process all 28 pending trials
- ✅ Deploy top 3 trials to paper trading
- ✅ Run for 24 hours without crashes
- ✅ Clear status reporting
- ✅ Easy to debug failures

## Timeline

- Task 3: Core manager (30 min)
- Task 4: Backtest validator (20 min)
- Task 5: Deployer (20 min)
- Task 6: Testing (30 min)

**Total: ~2 hours for complete rebuild**
