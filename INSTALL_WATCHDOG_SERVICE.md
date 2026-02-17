# Install Watchdog Systemd Service

## Quick Install (Run as sudo)

```bash
cd /opt/user-data/experiment/cappuccino
sudo bash systemd/install_services.sh
```

That's it! The watchdog will now:
- ✅ Start automatically on boot
- ✅ Restart if it crashes
- ✅ Run persistently in the background
- ✅ Monitor and restart workers every 60min

---

## Manual Installation (if script doesn't work)

### Step 1: Copy service file
```bash
sudo cp systemd/cappuccino-watchdog.service /etc/systemd/system/
sudo chmod 644 /etc/systemd/system/cappuccino-watchdog.service
```

### Step 2: Reload systemd
```bash
sudo systemctl daemon-reload
```

### Step 3: Enable auto-start on boot
```bash
sudo systemctl enable cappuccino-watchdog.service
```

### Step 4: Start the service
```bash
sudo systemctl start cappuccino-watchdog.service
```

### Step 5: Check status
```bash
sudo systemctl status cappuccino-watchdog.service
```

---

## Verify It's Working

```bash
# Check service is running
sudo systemctl status cappuccino-watchdog

# View live logs
sudo journalctl -u cappuccino-watchdog -f

# Check watchdog log file
tail -f logs/watchdog.log
```

---

## Useful Commands

```bash
# View status
sudo systemctl status cappuccino-watchdog

# View logs (systemd)
sudo journalctl -u cappuccino-watchdog -f

# View logs (file)
tail -f logs/watchdog.log

# Restart service
sudo systemctl restart cappuccino-watchdog

# Stop service
sudo systemctl stop cappuccino-watchdog

# Disable auto-start
sudo systemctl disable cappuccino-watchdog

# Re-enable auto-start
sudo systemctl enable cappuccino-watchdog
```

---

## What The Service Does

**Monitors workers**:
- Checks worker health every 60 seconds
- Monitors memory usage per worker
- Monitors system memory availability

**Auto-restarts workers when**:
- Worker age > 60 minutes (prevents leak accumulation)
- Worker memory > 8GB (leak detected)
- System memory < 2GB available (critical condition)
- Worker process dies unexpectedly

**Survives reboots**:
- Automatically starts on system boot
- Restarts if watchdog itself crashes
- Persistent logging to files

**Resource limits**:
- CPU: 10% max (low overhead)
- Memory: 500MB max (watchdog itself)
- Nice: 10 (lower priority than training)

---

## Troubleshooting

### Service won't start
```bash
# Check for errors
sudo journalctl -u cappuccino-watchdog -n 50

# Check service file syntax
sudo systemd-analyze verify /etc/systemd/system/cappuccino-watchdog.service

# Check file permissions
ls -l /etc/systemd/system/cappuccino-watchdog.service
```

### Service running but watchdog not working
```bash
# Check if worker_watchdog.sh exists
ls -l /opt/user-data/experiment/cappuccino/worker_watchdog.sh

# Check if it's executable
chmod +x /opt/user-data/experiment/cappuccino/worker_watchdog.sh

# Check logs
tail -100 logs/watchdog_service.log
tail -100 logs/watchdog_service_error.log
```

### Need to update service file
```bash
# Edit the source
nano systemd/cappuccino-watchdog.service

# Reinstall
sudo cp systemd/cappuccino-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart cappuccino-watchdog
```

---

## After Installation

Once installed, the watchdog runs automatically. You can:

1. **Start training normally**:
   ```bash
   ./start_safe_workers.sh
   ```

2. **Monitor everything**:
   ```bash
   # Watchdog status
   sudo systemctl status cappuccino-watchdog

   # Worker status
   tail -f logs/watchdog.log

   # Training progress
   python scripts/automation/dashboard_detailed.py
   ```

3. **Reboot anytime**:
   - Watchdog auto-starts on boot
   - Workers will be restarted automatically
   - Training resumes from last checkpoint

---

## Log Files

The watchdog creates multiple log files:

- `logs/watchdog.log` - Main watchdog activity log
- `logs/watchdog_service.log` - Systemd service stdout
- `logs/watchdog_service_error.log` - Systemd service stderr
- `logs/worker_safe_1.log` - Worker 1 training logs
- `logs/worker_safe_2.log` - Worker 2 training logs

All logs are automatically rotated and preserved.

---

**Status**: Service files created, ready to install
**Next**: Run `sudo bash systemd/install_services.sh`
