#!/usr/bin/env python3
"""
System Watchdog - Health Monitoring & Auto-Restart
Monitors all critical processes and restarts them if they crash.

Monitors:
- Training workers (3x)
- Paper trading
- Autonomous AI advisor
- GPU health
- Database integrity

Features:
- Auto-restart crashed processes
- Health checks
- Alert logging
- Resource monitoring
- Email/desktop notifications (optional)

Usage:
    python system_watchdog.py --check-interval 60
"""

import argparse
import json
import logging
import os
import psutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Load active study from environment
from dotenv import load_dotenv
load_dotenv('.env.training')
_DEFAULT_STUDY = os.getenv('ACTIVE_STUDY_NAME', 'cappuccino_week_20251206')

# Import redundancy manager for backups and atomic writes
try:
    from redundancy_manager import (
        backup_database,
        save_state,
        load_state,
        sync_ensemble_backups,
        get_system_health,
        DatabaseBackupManager,
        AtomicStateManager,
    )
    REDUNDANCY_AVAILABLE = True
except ImportError:
    REDUNDANCY_AVAILABLE = False


class SystemWatchdog:
    def __init__(
        self,
        check_interval: int = 60,
        auto_restart: bool = True,
        max_restarts: int = 3,
        restart_cooldown: int = 300,
        enable_alpha_monitoring: bool = True,
        alpha_decay_threshold: float = -3.0,
    ):
        self.check_interval = check_interval
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.restart_cooldown = restart_cooldown
        self.enable_alpha_monitoring = enable_alpha_monitoring
        self.alpha_decay_threshold = alpha_decay_threshold

        # State
        self.running = True
        self.state_file = Path("deployments/watchdog_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        self.state = self._load_state()

        # Setup logging
        self._setup_logging()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Process definitions
        self.processes = {
            "training_workers": {
                "pattern": "1_optimize_unified.py",
                "expected_count": 3,
                "restart_cmd": self._restart_training_workers,
                "critical": True,
            },
            "paper_trading": {
                "pattern": "paper_trader_alpaca_polling.py",
                "expected_count": 1,
                "restart_cmd": self._restart_paper_trading,
                "critical": True,
            },
            "ai_advisor": {
                "pattern": "ollama_autonomous_advisor.py",
                "expected_count": 1,
                "restart_cmd": self._restart_ai_advisor,
                "critical": False,
            },
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/watchdog.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _load_state(self) -> Dict:
        """Load watchdog state with automatic fallback to backup."""
        default_state = {
            "restart_counts": {},
            "last_restart_times": {},
            "alerts": [],
        }

        if REDUNDANCY_AVAILABLE:
            return load_state(str(self.state_file), default_state)

        # Fallback to basic loading
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return default_state

    def _save_state(self):
        """Save watchdog state atomically with backup."""
        if REDUNDANCY_AVAILABLE:
            save_state(str(self.state_file), self.state)
        else:
            # Fallback to basic save
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)

    def _log_alert(self, severity: str, process: str, message: str):
        """Log an alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "process": process,
            "message": message,
        }

        self.state["alerts"].append(alert)

        # Keep only last 100 alerts
        if len(self.state["alerts"]) > 100:
            self.state["alerts"] = self.state["alerts"][-100:]

        self._save_state()

        log_func = {
            "INFO": self.logger.info,
            "WARNING": self.logger.warning,
            "ERROR": self.logger.error,
            "CRITICAL": self.logger.critical,
        }.get(severity, self.logger.info)

        log_func(f"[{process}] {message}")

    def find_processes(self, pattern: str) -> List[psutil.Process]:
        """Find processes matching pattern."""
        matches = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if pattern in cmdline and 'grep' not in cmdline:
                    matches.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return matches

    def check_process_health(self, name: str, config: Dict) -> Tuple[bool, str]:
        """Check health of a process."""
        pattern = config["pattern"]
        expected_count = config["expected_count"]

        procs = self.find_processes(pattern)
        actual_count = len(procs)

        if actual_count == 0:
            return False, f"No processes running (expected {expected_count})"

        if actual_count < expected_count:
            return False, f"Only {actual_count}/{expected_count} processes running"

        # Check if processes are responsive (not zombies)
        for proc in procs:
            try:
                if proc.status() == psutil.STATUS_ZOMBIE:
                    return False, f"Zombie process detected: PID {proc.pid}"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return True, f"OK ({actual_count}/{expected_count} running)"

    def check_paper_trader_staleness(self) -> Tuple[bool, str]:
        """Check if paper trader is actively polling via heartbeat file.

        Returns:
            (is_healthy, status_message)
        """
        heartbeat_path = Path("paper_trades/.heartbeat")

        if not heartbeat_path.exists():
            # No heartbeat file yet - might be starting up
            return True, "No heartbeat yet (starting up?)"

        try:
            with open(heartbeat_path) as f:
                heartbeat = json.load(f)

            # Parse timestamp
            heartbeat_time = datetime.fromisoformat(heartbeat['timestamp'].replace('Z', '+00:00'))
            now = datetime.now(heartbeat_time.tzinfo)
            age_seconds = (now - heartbeat_time).total_seconds()
            age_minutes = age_seconds / 60

            # Get expected poll interval from heartbeat
            poll_interval = heartbeat.get('poll_interval', 60)

            # Calculate staleness threshold:
            # - At least 3x poll interval
            # - At least 10 minutes
            # - Maximum 2 hours (for 1h timeframe polling)
            staleness_threshold = max(poll_interval * 3, 600)  # At least 10 minutes
            staleness_threshold = min(staleness_threshold, 7200)  # At most 2 hours

            if age_seconds > staleness_threshold:
                return False, f"STALE: Last heartbeat {age_minutes:.1f}min ago (threshold: {staleness_threshold/60:.0f}min)"

            poll_count = heartbeat.get('poll_count', 0)
            return True, f"Active: heartbeat {age_minutes:.1f}min ago, poll #{poll_count}"

        except Exception as e:
            self.logger.warning(f"Error reading heartbeat: {e}")
            return True, f"Heartbeat check error: {e}"

    def _can_restart(self, process_name: str) -> Tuple[bool, str]:
        """Check if process can be restarted."""
        # Check restart count
        restart_count = self.state["restart_counts"].get(process_name, 0)
        if restart_count >= self.max_restarts:
            return False, f"Max restarts ({self.max_restarts}) exceeded"

        # Check cooldown
        last_restart = self.state["last_restart_times"].get(process_name)
        if last_restart:
            last_restart_dt = datetime.fromisoformat(last_restart)
            elapsed = (datetime.now() - last_restart_dt).total_seconds()
            if elapsed < self.restart_cooldown:
                return False, f"Cooldown period ({int(self.restart_cooldown - elapsed)}s remaining)"

        return True, "OK"

    def _record_restart(self, process_name: str):
        """Record a restart."""
        self.state["restart_counts"][process_name] = \
            self.state["restart_counts"].get(process_name, 0) + 1
        self.state["last_restart_times"][process_name] = datetime.now().isoformat()
        self._save_state()

    def _reset_restart_count(self, process_name: str):
        """Reset restart count after successful operation."""
        if process_name in self.state["restart_counts"]:
            self.state["restart_counts"][process_name] = 0
            self._save_state()

    def _restart_training_workers(self) -> bool:
        """Restart training workers."""
        self.logger.info("Restarting training workers...")

        # Kill existing workers
        subprocess.run(["pkill", "-f", "1_optimize_unified.py"], check=False)
        time.sleep(5)

        # Start new workers - read study name from .env.training
        cmd = f"""
        # Load configuration from .env.training
        if [ -f ".env.training" ]; then
            source .env.training
            STUDY_NAME="$ACTIVE_STUDY_NAME"
        else
            STUDY_NAME="{_DEFAULT_STUDY}"
        fi

        N_PARALLEL=3
        mkdir -p logs/parallel_training

        for i in $(seq 1 $N_PARALLEL); do
            echo "[$(date)] Launching worker $i..."
            python -u 1_optimize_unified.py \
                --n-trials 500 \
                --gpu 0 \
                --study-name $STUDY_NAME \
                2>&1 | sed "s/^/[W$i] /" > logs/parallel_training/worker_$i.log &
            sleep 5
        done
        """

        try:
            subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(15)

            # Verify workers started
            procs = self.find_processes("1_optimize_unified.py")
            if len(procs) >= 3:
                self.logger.info(f"Successfully restarted {len(procs)} training workers")
                return True
            else:
                self.logger.error(f"Only {len(procs)} workers started")
                return False

        except Exception as e:
            self.logger.error(f"Failed to restart training workers: {e}")
            return False

    def _ensure_stored_agent_directory(self, model_dir: Path):
        """
        Ensure stored_agent directory exists with model weights.
        Paper trader requires weights to be in model_dir/stored_agent/
        """
        import shutil

        stored_agent_dir = model_dir / "stored_agent"

        # If it already exists and has files, we're good
        if stored_agent_dir.exists():
            weight_files = list(stored_agent_dir.glob("*.pth"))
            if len(weight_files) >= 2:  # At least actor.pth and critic.pth
                return

        # Create directory
        stored_agent_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Creating stored_agent directory: {stored_agent_dir}")

        # Copy all .pth files from parent directory
        weight_files = list(model_dir.glob("*.pth"))
        if not weight_files:
            self.logger.warning(f"No .pth files found in {model_dir}")
            return

        copied_count = 0
        for weight_file in weight_files:
            try:
                dest = stored_agent_dir / weight_file.name
                shutil.copy2(weight_file, dest)
                copied_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to copy {weight_file}: {e}")

        self.logger.info(f"Copied {copied_count} weight files to stored_agent/")

    def _restart_paper_trading(self) -> bool:
        """Restart paper trading."""
        self.logger.info("Restarting paper trading...")

        # Kill existing
        subprocess.run(["pkill", "-f", "paper_trader_alpaca_polling.py"], check=False)
        time.sleep(2)

        # ALWAYS use ensemble directory (top 10 models with game theory voting)
        model_dir = "train_results/ensemble"
        self.logger.info(f"Using ensemble with top 10 models from training")

        # Start paper trading
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/paper_trading_watchdog_{timestamp}.log"
        csv_file = f"paper_trades/watchdog_session_{timestamp}.csv"

        cmd = [
            "nohup", "python", "-u", "paper_trader_alpaca_polling.py",
            "--model-dir", model_dir,
            "--tickers", "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "LINK/USD", "UNI/USD", "AAVE/USD",
            "--timeframe", "1h",
            "--history-hours", "120",
            "--poll-interval", "60",
            "--gpu", "-1",
            "--log-file", csv_file,
            # Risk management defaults
            "--max-position-pct", "0.30",  # Max 30% in single asset
            "--stop-loss-pct", "0.10",  # 10% stop-loss
        ]

        try:
            with open(log_file, 'w') as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

            time.sleep(5)

            # Verify started
            if proc.poll() is None:
                self.logger.info(f"Successfully restarted paper trading (PID {proc.pid})")
                return True
            else:
                self.logger.error("Paper trading process terminated immediately")
                return False

        except Exception as e:
            self.logger.error(f"Failed to restart paper trading: {e}")
            return False

    def _restart_ai_advisor(self) -> bool:
        """Restart AI advisor."""
        self.logger.info("Restarting AI advisor...")

        # Kill existing
        subprocess.run(["pkill", "-f", "ollama_autonomous_advisor.py"], check=False)
        time.sleep(2)

        # Start AI advisor
        try:
            subprocess.run(["./start_autonomous_advisor.sh"], check=True, capture_output=True)
            time.sleep(5)

            # Verify started
            procs = self.find_processes("ollama_autonomous_advisor.py")
            if len(procs) >= 1:
                self.logger.info("Successfully restarted AI advisor")
                return True
            else:
                self.logger.error("AI advisor did not start")
                return False

        except Exception as e:
            self.logger.error(f"Failed to restart AI advisor: {e}")
            return False

    def check_gpu_health(self) -> Tuple[bool, str]:
        """Check GPU health."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                return False, "nvidia-smi failed"

            temp, util = result.stdout.strip().split(',')
            temp = int(temp.strip())
            util = int(util.strip())

            # Check temperature
            if temp > 85:
                return False, f"GPU temperature critical: {temp}°C"

            if temp > 80:
                self._log_alert("WARNING", "GPU", f"High temperature: {temp}°C")

            return True, f"OK (Temp: {temp}°C, Util: {util}%)"

        except Exception as e:
            return False, f"GPU check failed: {e}"

    def check_disk_space(self) -> Tuple[bool, str]:
        """Check disk space."""
        try:
            usage = psutil.disk_usage('/')
            percent = usage.percent

            if percent > 95:
                return False, f"Disk space critical: {percent}% used"

            if percent > 90:
                self._log_alert("WARNING", "DISK", f"Low disk space: {percent}% used")

            return True, f"OK ({percent}% used)"

        except Exception as e:
            return False, f"Disk check failed: {e}"

    def check_database_integrity(self) -> Tuple[bool, str]:
        """Check database integrity."""
        db_path = "databases/optuna_cappuccino.db"

        if not Path(db_path).exists():
            return False, "Database file not found"

        try:
            result = subprocess.run(
                ["sqlite3", db_path, "PRAGMA integrity_check;"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                return False, "Database integrity check failed"

            if "ok" in result.stdout.lower():
                return True, "OK"
            else:
                return False, f"Database issues: {result.stdout}"

        except Exception as e:
            return False, f"Database check failed: {e}"

    def run_health_checks(self):
        """Run all health checks."""
        self.logger.info("=" * 80)
        self.logger.info("Running health checks...")

        all_healthy = True

        # Check processes
        for name, config in self.processes.items():
            healthy, status = self.check_process_health(name, config)

            # For paper trading, also check staleness (heartbeat freshness)
            if name == "paper_trading" and healthy:
                stale_healthy, stale_status = self.check_paper_trader_staleness()
                if not stale_healthy:
                    healthy = False
                    status = stale_status
                else:
                    status = f"{status} | {stale_status}"

            if healthy:
                self.logger.info(f"✓ {name}: {status}")
                self._reset_restart_count(name)
            else:
                severity = "CRITICAL" if config["critical"] else "WARNING"
                self._log_alert(severity, name, f"Health check failed: {status}")

                if self.auto_restart:
                    can_restart, reason = self._can_restart(name)

                    if can_restart:
                        self.logger.warning(f"Attempting to restart {name}...")
                        success = config["restart_cmd"]()

                        if success:
                            self._record_restart(name)
                            self._log_alert("INFO", name, "Successfully restarted")
                        else:
                            self._log_alert("ERROR", name, "Restart failed")
                            all_healthy = False
                    else:
                        self._log_alert("ERROR", name, f"Cannot restart: {reason}")
                        all_healthy = False
                else:
                    self.logger.warning(f"Auto-restart disabled for {name}")
                    all_healthy = False

        # Check GPU
        healthy, status = self.check_gpu_health()
        if healthy:
            self.logger.info(f"✓ GPU: {status}")
        else:
            self._log_alert("CRITICAL", "GPU", status)
            all_healthy = False

        # Check disk
        healthy, status = self.check_disk_space()
        if healthy:
            self.logger.info(f"✓ Disk: {status}")
        else:
            self._log_alert("CRITICAL", "DISK", status)
            all_healthy = False

        # Check database
        healthy, status = self.check_database_integrity()
        if healthy:
            self.logger.info(f"✓ Database: {status}")
        else:
            self._log_alert("ERROR", "DATABASE", status)
            all_healthy = False

        # Check alpha performance (model vs market)
        healthy, status, alpha = self.check_alpha_performance()
        if healthy:
            self.logger.info(f"✓ Alpha Performance: {status}")
        else:
            self._log_alert("WARNING", "ALPHA", status)
            self.logger.warning(f"⚠️  Model underperforming market by {abs(alpha):.2f}%")

            # Trigger retraining if alpha decay detected
            if self.auto_restart and alpha < self.alpha_decay_threshold:
                self.logger.warning("Triggering automatic retraining due to alpha decay...")
                if self.trigger_retraining():
                    self._log_alert("INFO", "ALPHA", "Retraining initiated to recover alpha")
                else:
                    self._log_alert("ERROR", "ALPHA", "Failed to initiate retraining")

        # Check for ensemble updates (new models from training)
        ensemble_updated, ensemble_msg = self.check_ensemble_updates()
        if ensemble_updated:
            self.logger.info(f"🔄 Ensemble Updated: {ensemble_msg}")
            self._log_alert("INFO", "ENSEMBLE", f"New models detected: {ensemble_msg}")

            # Check cooldown before restarting paper trader
            # Only restart daily if new best models are available
            last_restart_key = "last_paper_trader_restart"
            cooldown_minutes = 1440  # Cooldown: 24 hours (daily restart only)

            can_restart = True
            if last_restart_key in self.state:
                from datetime import datetime
                last_restart = datetime.fromisoformat(self.state[last_restart_key])
                minutes_since = (datetime.now() - last_restart).total_seconds() / 60

                if minutes_since < cooldown_minutes:
                    can_restart = False
                    self.logger.info(f"Paper trader restart cooldown active: {cooldown_minutes - minutes_since:.1f}min remaining")
                    self.logger.info("Ensemble will be used on next scheduled restart")

            # Restart paper trader if cooldown allows
            if can_restart and self.auto_restart:
                self.logger.info("Restarting paper trader with updated ensemble...")
                if self._restart_paper_trading():
                    self._log_alert("INFO", "PAPER_TRADING", "Restarted with new ensemble models")
                    # Record restart time
                    from datetime import datetime
                    self.state[last_restart_key] = datetime.now().isoformat()
                    self._save_state()
                    # Clear the reload flag
                    reload_flag = Path("train_results/ensemble/.reload_models")
                    if reload_flag.exists():
                        reload_flag.unlink()
                else:
                    self._log_alert("ERROR", "PAPER_TRADING", "Failed to restart with new ensemble")
            elif not can_restart:
                # Clear the flag anyway to prevent stale detection
                reload_flag = Path("train_results/ensemble/.reload_models")
                if reload_flag.exists():
                    reload_flag.unlink()

        if all_healthy:
            self.logger.info("All systems healthy ✓")
        else:
            self.logger.warning("Some systems unhealthy!")

        self.logger.info("=" * 80)

    def check_alpha_performance(self) -> Tuple[bool, str, float]:
        """Check if model is underperforming (alpha decay).

        Returns:
            (is_healthy, message, alpha_value)
        """
        if not self.enable_alpha_monitoring:
            return True, "Alpha monitoring disabled", 0.0

        try:
            # Get current portfolio performance
            positions_file = Path("paper_trades/positions_state.json")
            if not positions_file.exists():
                return True, "No positions file yet", 0.0

            with open(positions_file) as f:
                state = json.load(f)

            current_value = state['portfolio_value']
            initial_value = state['portfolio_protection']['initial_value']
            portfolio_return = (current_value / initial_value - 1) * 100

            # Get market benchmark (simplified - just BTC for now)
            try:
                import os
                from dotenv import load_dotenv
                load_dotenv()

                import alpaca_trade_api as tradeapi
                api = tradeapi.REST(
                    os.getenv("ALPACA_API_KEY"),
                    os.getenv("ALPACA_API_SECRET"),
                    "https://paper-api.alpaca.markets",
                    api_version='v2'
                )

                from datetime import timedelta
                end = datetime.now()
                start = end - timedelta(hours=24)

                # Get BTC as market proxy
                bars = api.get_crypto_bars(
                    'BTC/USD',
                    tradeapi.rest.TimeFrame.Hour,
                    start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    end.strftime('%Y-%m-%dT%H:%M:%SZ')
                ).df

                if not bars.empty:
                    market_return = (bars['close'].iloc[-1] / bars['close'].iloc[0] - 1) * 100
                else:
                    market_return = 0.0

            except Exception as e:
                self.logger.debug(f"Could not get market benchmark: {e}")
                market_return = 0.0

            # Calculate alpha
            alpha = portfolio_return - market_return

            self.logger.info(f"Performance: Portfolio {portfolio_return:+.2f}% | Market {market_return:+.2f}% | Alpha {alpha:+.2f}%")

            # Check if alpha decay threshold exceeded
            if alpha < self.alpha_decay_threshold:
                message = f"ALPHA DECAY: {alpha:.2f}% (threshold: {self.alpha_decay_threshold}%)"
                return False, message, alpha

            return True, f"Alpha: {alpha:+.2f}%", alpha

        except Exception as e:
            self.logger.debug(f"Error checking alpha: {e}")
            return True, f"Alpha check error: {e}", 0.0

    def check_ensemble_updates(self) -> Tuple[bool, str]:
        """Check if ensemble has been updated with new models.

        Returns:
            (needs_reload, message)
        """
        try:
            ensemble_dir = Path("train_results/ensemble")
            if not ensemble_dir.exists():
                return False, "Ensemble directory not found"

            # Check for reload flag (created by ensemble_auto_updater.py)
            reload_flag = ensemble_dir / ".reload_models"
            if reload_flag.exists():
                # Check when flag was created
                flag_age_seconds = time.time() - reload_flag.stat().st_mtime
                flag_age_minutes = flag_age_seconds / 60

                # Only reload if flag is recent (< 10 minutes old)
                # This prevents reloading old stale flags on watchdog restart
                if flag_age_minutes < 10:
                    manifest_path = ensemble_dir / "ensemble_manifest.json"
                    if manifest_path.exists():
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                        model_count = manifest.get("model_count", 0)
                        best_value = manifest.get("best_value", 0)
                        updated = manifest.get("updated", "unknown")
                        return True, f"{model_count} models (best: {best_value:.4f}, updated: {updated})"
                    else:
                        return True, "Reload flag found"
                else:
                    # Flag is old, probably from previous session - ignore it
                    self.logger.debug(f"Ignoring old reload flag ({flag_age_minutes:.1f} minutes old)")
                    return False, "Reload flag too old"

            return False, "No reload needed"

        except Exception as e:
            self.logger.debug(f"Error checking ensemble updates: {e}")
            return False, f"Check error: {e}"

    def trigger_retraining(self) -> bool:
        """Trigger model retraining on fresh data.

        Returns:
            True if retraining initiated successfully
        """
        self.logger.info("=" * 80)
        self.logger.info("TRIGGERING MODEL RETRAINING")
        self.logger.info("=" * 80)

        # Check cooldown - only retrain once per week
        last_retrain_key = "last_retrain_time"
        if last_retrain_key in self.state:
            last_retrain = datetime.fromisoformat(self.state[last_retrain_key])
            hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
            cooldown_hours = 168  # 1 week

            if hours_since < cooldown_hours:
                self.logger.warning(f"Retrain cooldown active: {cooldown_hours - hours_since:.1f}h remaining")
                return False

        try:
            # Record retrain timestamp
            self.state[last_retrain_key] = datetime.now().isoformat()
            self.state["retrain_count"] = self.state.get("retrain_count", 0) + 1
            self._save_state()

            # Stop current training workers
            self.logger.info("Stopping current training workers...")
            subprocess.run(["pkill", "-f", "1_optimize_unified.py"], check=False)
            time.sleep(5)

            # Start fresh training with new study name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            study_name = f"alpha_recovery_{timestamp}"

            self.logger.info(f"Starting fresh training: {study_name}")

            cmd = f"""
            STUDY_NAME="{study_name}"
            N_PARALLEL=3
            mkdir -p logs/parallel_training

            for i in $(seq 1 $N_PARALLEL); do
                echo "[$(date)] Launching retraining worker $i..."
                python -u 1_optimize_unified.py \
                    --n-trials 200 \
                    --gpu 0 \
                    --study-name $STUDY_NAME \
                    2>&1 | sed "s/^/[W$i] /" > logs/parallel_training/worker_$i.log &
                sleep 5
            done
            """

            subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(15)

            # Verify workers started
            procs = self.find_processes("1_optimize_unified.py")
            if len(procs) >= 3:
                self.logger.info(f"✓ Retraining started: {len(procs)} workers on study '{study_name}'")
                self.logger.info(f"  Models will auto-deploy to ensemble when training completes")
                self._log_alert("INFO", "Retraining", f"Started study '{study_name}' due to alpha decay")
                return True
            else:
                self.logger.error("Failed to start retraining workers")
                return False

        except Exception as e:
            self.logger.error(f"Error triggering retraining: {e}", exc_info=True)
            return False

    def run(self):
        """Main loop."""
        self.logger.info("=" * 80)
        self.logger.info("SYSTEM WATCHDOG STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Check interval: {self.check_interval}s")
        self.logger.info(f"Auto-restart: {self.auto_restart}")
        self.logger.info(f"Max restarts: {self.max_restarts}")
        self.logger.info(f"Restart cooldown: {self.restart_cooldown}s")
        self.logger.info(f"Redundancy manager: {'enabled' if REDUNDANCY_AVAILABLE else 'disabled'}")
        self.logger.info("=" * 80)

        # Track last backup time for hourly backups
        last_backup_time = None
        last_ensemble_sync = None

        # Initial check
        self.run_health_checks()

        # Initial backup if redundancy available
        if REDUNDANCY_AVAILABLE:
            self._run_redundancy_tasks(force=True)
            last_backup_time = datetime.now()
            last_ensemble_sync = datetime.now()

        # Main loop
        while self.running:
            try:
                self.logger.info(f"Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)

                if self.running:
                    self.run_health_checks()

                    # Hourly redundancy tasks
                    if REDUNDANCY_AVAILABLE:
                        now = datetime.now()

                        # Database backup every hour
                        if last_backup_time is None or (now - last_backup_time).total_seconds() >= 3600:
                            self._run_redundancy_tasks()
                            last_backup_time = now

                        # Ensemble sync every 30 minutes
                        if last_ensemble_sync is None or (now - last_ensemble_sync).total_seconds() >= 1800:
                            self._sync_ensemble()
                            last_ensemble_sync = now

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)

        self.logger.info("System watchdog stopped")

    def _run_redundancy_tasks(self, force: bool = False):
        """Run redundancy tasks (database backup, etc.)."""
        if not REDUNDANCY_AVAILABLE:
            return

        try:
            self.logger.info("Running redundancy tasks...")

            # Database backup
            backup_path = backup_database()
            if backup_path:
                self._log_alert("INFO", "BACKUP", f"Database backup created: {backup_path.name}")
            else:
                self._log_alert("WARNING", "BACKUP", "Database backup failed")

        except Exception as e:
            self.logger.error(f"Redundancy tasks failed: {e}")

    def _sync_ensemble(self):
        """Sync ensemble to backup locations."""
        if not REDUNDANCY_AVAILABLE:
            return

        try:
            if sync_ensemble_backups():
                self.logger.info("Ensemble synced to backup locations")
        except Exception as e:
            self.logger.warning(f"Ensemble sync failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="System watchdog for health monitoring + alpha decay detection")
    parser.add_argument("--check-interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--no-auto-restart", action="store_true", help="Disable auto-restart")
    parser.add_argument("--max-restarts", type=int, default=3, help="Max restarts per process")
    parser.add_argument("--restart-cooldown", type=int, default=300, help="Cooldown between restarts (seconds)")
    parser.add_argument("--no-alpha-monitoring", action="store_true", help="Disable alpha decay monitoring")
    parser.add_argument("--alpha-threshold", type=float, default=-3.0, help="Alpha decay threshold for retraining (default: -3.0%%)")

    args = parser.parse_args()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    watchdog = SystemWatchdog(
        check_interval=args.check_interval,
        auto_restart=not args.no_auto_restart,
        max_restarts=args.max_restarts,
        restart_cooldown=args.restart_cooldown,
        enable_alpha_monitoring=not args.no_alpha_monitoring,
        alpha_decay_threshold=args.alpha_threshold,
    )

    # Log configuration
    if watchdog.enable_alpha_monitoring:
        watchdog.logger.info(f"Alpha monitoring enabled (threshold: {args.alpha_threshold}%)")

    watchdog.run()


if __name__ == "__main__":
    main()
