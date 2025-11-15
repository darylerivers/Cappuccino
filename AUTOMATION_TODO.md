# Automation System - Known Gaps & TODO List

Branch: `automation-fixes`

## 🐛 Known Issues & Gaps

### High Priority

1. **Paper Trading Model Compatibility**
   - **Issue:** Not all trial models have `best_trial` file or `name_folder` attribute
   - **Impact:** Auto-deployer can't deploy newer models without manual intervention
   - **Status:** Partially fixed (auto_model_deployer.py creates missing files)
   - **TODO:**
     - Test with more trial models
     - Add validation for `stored_agent` directory existence
     - Handle models without proper directory structure

2. **Trial Training State Tracking**
   - **Issue:** Some trials may be running but not in database yet
   - **Impact:** Could deploy incomplete models
   - **TODO:**
     - Add check for trial completion time vs file modification time
     - Verify actor.pth and critic.pth are recent and complete
     - Add file size validation (corrupted files)

3. **Database Lock Handling**
   - **Issue:** SQLite database can be locked during concurrent access
   - **Impact:** Automation scripts may fail with "database is locked"
   - **TODO:**
     - Add retry logic with exponential backoff
     - Implement connection timeout handling
     - Consider read-only connections for monitoring

4. **Paper Trading Error Handling**
   - **Issue:** Paper trader crashes on missing dependencies or API errors
   - **Impact:** Trading stops until manual restart (watchdog will restart but root cause not fixed)
   - **TODO:**
     - Better error messages in paper_trader_alpaca_polling.py
     - Graceful degradation on API failures
     - Notification when paper trader restarts multiple times

### Medium Priority

5. **Notification System**
   - **Issue:** Desktop notifications only work on GUI systems
   - **Impact:** Notifications fail on headless servers
   - **TODO:**
     - Add email notification support
     - Add Slack/Discord webhook integration
     - Add SMS notifications (Twilio)
     - Make notification backends configurable

6. **Performance Monitor - Trade Detection**
   - **Issue:** Trade counting logic assumes $1000 initial balance
   - **Impact:** May miscount trades if balance changes
   - **TODO:**
     - Parse CSV for actual trade entries
     - Track position changes instead of balance
     - Add P&L calculation

7. **GPU Temperature Alerts**
   - **Issue:** Alert threshold is hardcoded (80°C)
   - **Impact:** May not be appropriate for all GPUs
   - **TODO:**
     - Make temperature threshold configurable
     - Add GPU-specific profiles
     - Add power draw alerts

8. **Deployment Validation**
   - **Issue:** No backtesting before deployment
   - **Impact:** Could deploy models that perform poorly in recent conditions
   - **TODO:**
     - Add quick backtest on last 7 days before deployment
     - Require positive Sharpe ratio on recent data
     - Add A/B testing mode (run old and new model in parallel)

### Low Priority

9. **Logging**
   - **Issue:** Log files grow unbounded
   - **Impact:** Disk space usage over time
   - **TODO:**
     - Implement log rotation (logrotate or Python logging.handlers.RotatingFileHandler)
     - Archive old logs
     - Add log compression

10. **State File Management**
    - **Issue:** State files (.json, .pid) accumulate
    - **Impact:** Minor disk usage, stale PID files
    - **TODO:**
      - Cleanup stale PID files on startup
      - Validate PID files point to actual processes
      - Add state file cleanup command

11. **Dashboard/Web UI**
    - **Issue:** No visual dashboard, command-line only
    - **Impact:** Harder to monitor at a glance
    - **TODO:**
      - Create simple web dashboard (Flask/FastAPI)
      - Real-time charts for training progress
      - System health visualization
      - Mobile-responsive design

12. **Deployment Rollback**
    - **Issue:** Rollback capability exists but not tested
    - **Impact:** May not work when needed
    - **TODO:**
      - Add `./rollback_deployment.sh` script
      - Test rollback procedure
      - Add "safe mode" to only deploy after X successful hours

13. **Multi-Model Ensemble**
    - **Issue:** Only deploys single best model
    - **Impact:** Missing potential for ensemble trading
    - **TODO:**
      - Support deploying multiple models
      - Ensemble voting system
      - Weighted average based on recent performance

## 🔧 Improvements & Enhancements

### Performance Optimizations

1. **Reduce Database Queries**
   - Cache trial data locally
   - Use database views for common queries
   - Batch queries when possible

2. **Concurrent Processing**
   - Run health checks in parallel
   - Use asyncio for I/O-bound operations
   - Thread pool for validation tasks

3. **Resource Usage**
   - Add memory profiling
   - Optimize state file sizes
   - Reduce redundant file reads

### Feature Additions

1. **Smart Deployment**
   - Deploy only during low-volatility periods
   - Wait for market close before deploying
   - Schedule deployments for specific times

2. **Performance Analytics**
   - Generate daily/weekly reports
   - Compare deployed model vs top 5 models
   - Track deployment success rate

3. **Risk Management**
   - Add drawdown monitoring
   - Auto-pause trading on large losses
   - Position size limits

4. **Training Optimization**
   - Auto-adjust worker count based on GPU availability
   - Pause training during high system load
   - Smart hyperparameter search space pruning

## 📋 Testing Checklist

- [ ] Test auto-deployment with new best trial
- [ ] Test watchdog restart on crashed training worker
- [ ] Test watchdog restart on crashed paper trader
- [ ] Test notifications on headless system (should fail gracefully)
- [ ] Test database lock handling
- [ ] Test deployment rollback
- [ ] Test with corrupted model files
- [ ] Test GPU temperature alert
- [ ] Test disk space alert
- [ ] Test with multiple concurrent deployments

## 🚀 Quick Wins (Easy Improvements)

1. **Add deployment confirmation**
   ```bash
   # Before deploying, send notification:
   notify-send "Deploying new model" "Trial #1234, improvement: 5.2%"
   ```

2. **Add status command to show all PIDs**
   ```bash
   ./status_automation.sh --all  # Include training, trading, etc.
   ```

3. **Add restart individual component**
   ```bash
   ./restart_automation.sh watchdog
   ./restart_automation.sh deployer
   ```

4. **Add health check endpoint**
   ```python
   # Simple HTTP endpoint for monitoring
   python -m http.server 8080  # Serve status.json
   ```

5. **Add backup script**
   ```bash
   ./backup_automation.sh  # Backup state files and deployment history
   ```

## 📝 Documentation Needs

- [ ] Add examples to AUTOMATION_GUIDE.md
- [ ] Add troubleshooting flowchart
- [ ] Document state file formats
- [ ] Add API documentation for extending automation
- [ ] Create video walkthrough

## 🎯 Next Sprint Goals

### Sprint 1: Reliability (1-2 days)
- Fix database lock handling
- Add proper error handling in paper trader
- Implement log rotation
- Test all restart scenarios

### Sprint 2: Observability (2-3 days)
- Add email notifications
- Create web dashboard
- Add performance analytics
- Better logging and metrics

### Sprint 3: Intelligence (3-5 days)
- Add pre-deployment backtesting
- Implement ensemble trading
- Smart deployment timing
- Auto-optimize hyperparameter search space

## 🔄 Maintenance Tasks

### Daily
- Check automation status: `./status_automation.sh`
- Review recent alerts: `cat deployments/watchdog_state.json`
- Monitor disk space: `df -h`

### Weekly
- Review deployment history
- Analyze training progress
- Check for new best models manually
- Backup state files

### Monthly
- Rotate logs
- Review and prune old trial models
- Update dependencies
- Performance review

---

## Branch Strategy

**Current Branch:** `automation-fixes`

**Purpose:** Iterative improvements and bug fixes for automation system

**Merge to Master:** After each sprint when features are tested and stable

**Commit Strategy:**
- Small, focused commits
- Clear commit messages
- Reference issue numbers from this TODO

**Testing:** Test each change with `./status_automation.sh` before committing

---

Last Updated: 2025-11-14
Branch: automation-fixes
Status: Active Development
