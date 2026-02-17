#!/usr/bin/env python3
"""
Two-Phase Training Scheduler

Automated daemon that:
1. Runs two-phase training on a schedule (weekly/monthly)
2. Monitors training progress
3. Automatically deploys winning models to production
4. Logs results and sends alerts
5. Integrates with existing automation system

Usage:
    # Run as daemon
    python two_phase_scheduler.py --daemon

    # Run immediate one-shot (no scheduling)
    python two_phase_scheduler.py --run-now

    # Check schedule
    python two_phase_scheduler.py --show-schedule

Configuration (in .env.training):
    TWO_PHASE_ENABLED=true
    TWO_PHASE_SCHEDULE="weekly"         # weekly, monthly, or cron expression
    TWO_PHASE_DAY="sunday"              # Day of week for weekly schedule
    TWO_PHASE_TIME="02:00"              # Time to start (24h format)
    TWO_PHASE_MODE="mini"               # mini or full
    TWO_PHASE_AUTO_DEPLOY="true"        # Auto-deploy winning models
    TWO_PHASE_NOTIFICATION="true"       # Send notifications on completion
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import signal


class TwoPhaseScheduler:
    """Scheduler daemon for two-phase training."""

    def __init__(self, config_file='.env.training'):
        """Initialize scheduler."""
        self.config_file = config_file
        self.config = self.load_config()
        self.running = False
        self.state_file = 'deployments/two_phase_scheduler_state.json'
        self.log_file = 'logs/two_phase_scheduler.log'

        # Setup logging
        self.setup_logging()

        # Load state
        self.state = self.load_state()

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def setup_logging(self):
        """Setup logging configuration."""
        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from .env.training."""
        config = {
            'enabled': False,
            'schedule': 'weekly',
            'day': 'sunday',
            'time': '02:00',
            'mode': 'mini',
            'auto_deploy': True,
            'notification': True,
        }

        if not os.path.exists(self.config_file):
            self.logger.warning(f"Config file not found: {self.config_file}")
            return config

        # Parse .env file
        with open(self.config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line or '=' not in line:
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key == 'TWO_PHASE_ENABLED':
                    config['enabled'] = value.lower() in ('true', '1', 'yes')
                elif key == 'TWO_PHASE_SCHEDULE':
                    config['schedule'] = value
                elif key == 'TWO_PHASE_DAY':
                    config['day'] = value.lower()
                elif key == 'TWO_PHASE_TIME':
                    config['time'] = value
                elif key == 'TWO_PHASE_MODE':
                    config['mode'] = value.lower()
                elif key == 'TWO_PHASE_AUTO_DEPLOY':
                    config['auto_deploy'] = value.lower() in ('true', '1', 'yes')
                elif key == 'TWO_PHASE_NOTIFICATION':
                    config['notification'] = value.lower() in ('true', '1', 'yes')

        return config

    def load_state(self) -> Dict[str, Any]:
        """Load scheduler state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")

        return {
            'last_run': None,
            'next_run': None,
            'runs': [],
            'status': 'idle'
        }

    def save_state(self):
        """Save scheduler state."""
        os.makedirs('deployments', exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def calculate_next_run(self) -> datetime:
        """Calculate next scheduled run time."""
        now = datetime.now()
        schedule = self.config['schedule']

        if schedule == 'weekly':
            # Parse day and time
            target_day = self.config['day'].lower()
            target_time = self.config['time']

            days_of_week = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }

            target_weekday = days_of_week.get(target_day, 6)  # Default to Sunday
            target_hour, target_minute = map(int, target_time.split(':'))

            # Calculate days until next occurrence
            current_weekday = now.weekday()
            days_ahead = target_weekday - current_weekday

            if days_ahead <= 0:  # Target day already passed this week
                days_ahead += 7

            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

            # If we're past the time today and it's the target day, schedule for next week
            if next_run <= now:
                next_run += timedelta(days=7)

            return next_run

        elif schedule == 'monthly':
            # Run on the 1st of each month at specified time
            target_time = self.config['time']
            target_hour, target_minute = map(int, target_time.split(':'))

            if now.day == 1 and now.hour < target_hour:
                # Today is the 1st and we haven't passed the time yet
                next_run = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
            else:
                # Schedule for next month
                if now.month == 12:
                    next_run = datetime(now.year + 1, 1, 1, target_hour, target_minute)
                else:
                    next_run = datetime(now.year, now.month + 1, 1, target_hour, target_minute)

            return next_run

        else:
            # Default: 1 week from now
            return now + timedelta(days=7)

    def should_run_now(self) -> bool:
        """Check if training should run now."""
        if not self.config['enabled']:
            return False

        if self.state['status'] == 'running':
            return False  # Already running

        next_run = self.state.get('next_run')
        if next_run is None:
            return True  # First run

        next_run_dt = datetime.fromisoformat(next_run)
        return datetime.now() >= next_run_dt

    def run_training(self):
        """Execute two-phase training."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Two-Phase Training")
        self.logger.info("=" * 60)

        run_start = datetime.now()
        self.state['status'] = 'running'
        self.state['last_run'] = run_start.isoformat()
        self.save_state()

        # Build command
        cmd = [sys.executable, 'run_two_phase_training.py']

        if self.config['mode'] == 'mini':
            cmd.append('--mini-test')
            self.logger.info("Mode: Mini test (20 trials)")
        else:
            self.logger.info("Mode: Full run (900 trials)")

        # Add skip prerequisites (daemon environment is assumed validated)
        cmd.append('--skip-prerequisites')

        # Run training
        self.logger.info(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            run_duration = (datetime.now() - run_start).total_seconds()

            self.logger.info("=" * 60)
            self.logger.info("Two-Phase Training Completed Successfully")
            self.logger.info("=" * 60)
            self.logger.info(f"Duration: {run_duration/3600:.2f} hours")

            # Parse results
            results = self.parse_results()

            # Record run
            run_record = {
                'timestamp': run_start.isoformat(),
                'duration': run_duration,
                'mode': self.config['mode'],
                'success': True,
                'results': results
            }
            self.state['runs'].append(run_record)
            self.state['status'] = 'idle'
            self.state['next_run'] = self.calculate_next_run().isoformat()
            self.save_state()

            # Auto-deploy if enabled
            if self.config['auto_deploy'] and results:
                self.auto_deploy_winner(results)

            # Send notification if enabled
            if self.config['notification']:
                self.send_notification(run_record)

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error("Two-Phase Training Failed")
            self.logger.error(f"Exit code: {e.returncode}")
            self.logger.error(f"Error output: {e.stderr}")

            run_duration = (datetime.now() - run_start).total_seconds()

            run_record = {
                'timestamp': run_start.isoformat(),
                'duration': run_duration,
                'mode': self.config['mode'],
                'success': False,
                'error': str(e)
            }
            self.state['runs'].append(run_record)
            self.state['status'] = 'idle'
            self.state['next_run'] = self.calculate_next_run().isoformat()
            self.save_state()

            # Send error notification
            if self.config['notification']:
                self.send_notification(run_record)

            return False

    def parse_results(self) -> Optional[Dict[str, Any]]:
        """Parse training results from output files."""
        results = {}

        # Load Phase 1 winner
        if os.path.exists('phase1_winner.json'):
            with open('phase1_winner.json', 'r') as f:
                results['phase1'] = json.load(f)

        # Load Phase 2 comparison
        if os.path.exists('phase2_comparison.json'):
            with open('phase2_comparison.json', 'r') as f:
                results['phase2'] = json.load(f)

        # Load final report
        if os.path.exists('two_phase_training_report.json'):
            with open('two_phase_training_report.json', 'r') as f:
                results['report'] = json.load(f)

        return results if results else None

    def auto_deploy_winner(self, results: Dict[str, Any]):
        """Automatically deploy winning model to production."""
        self.logger.info("=" * 60)
        self.logger.info("Auto-deploying Winning Model")
        self.logger.info("=" * 60)

        try:
            phase2 = results.get('phase2', {})
            winner_alg = phase2.get('winner', 'ppo')
            algorithms = phase2.get('results', {})
            winner_data = algorithms.get(winner_alg, {})

            trial_number = winner_data.get('best_trial_number')
            if trial_number is None:
                self.logger.warning("No trial number found in results")
                return

            # Find model file
            model_dir = f"train_results/phase2_{winner_alg}/phase2_{winner_alg}_trial_{trial_number}"
            model_file = f"{model_dir}/actor.pth"

            if not os.path.exists(model_file):
                self.logger.warning(f"Model file not found: {model_file}")
                return

            # Copy to deployments
            os.makedirs('deployments', exist_ok=True)
            deployment_name = f"two_phase_{winner_alg}_{trial_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            deployment_path = f"deployments/{deployment_name}"

            subprocess.run(['cp', model_file, deployment_path], check=True)

            self.logger.info(f"✓ Model deployed: {deployment_path}")
            self.logger.info(f"  Algorithm: {winner_alg.upper()}")
            self.logger.info(f"  Trial: {trial_number}")
            self.logger.info(f"  Sharpe: {winner_data.get('best_sharpe_bot', 0):.4f}")

            # Update deployment state
            deployment_state = {
                'two_phase_deployment': {
                    'algorithm': winner_alg,
                    'trial_number': trial_number,
                    'sharpe_bot': winner_data.get('best_sharpe_bot'),
                    'sharpe_hodl': winner_data.get('best_sharpe_hodl'),
                    'value': winner_data.get('best_value'),
                    'model_path': deployment_path,
                    'deployed_at': datetime.now().isoformat()
                }
            }

            state_file = 'deployments/two_phase_deployment.json'
            with open(state_file, 'w') as f:
                json.dump(deployment_state, f, indent=2)

            self.logger.info(f"✓ Deployment state saved: {state_file}")

        except Exception as e:
            self.logger.error(f"Auto-deploy failed: {e}")

    def send_notification(self, run_record: Dict[str, Any]):
        """Send notification about training completion."""
        # This can be extended to send email, Slack, Discord, etc.
        # For now, just write to a notifications file

        notification = {
            'timestamp': datetime.now().isoformat(),
            'type': 'two_phase_training',
            'success': run_record['success'],
            'duration': run_record['duration'],
            'mode': run_record['mode']
        }

        if run_record['success']:
            results = run_record.get('results', {})
            phase1 = results.get('phase1', {})
            phase2 = results.get('phase2', {})

            notification['message'] = (
                f"Two-Phase Training Completed Successfully!\n"
                f"Duration: {run_record['duration']/3600:.2f} hours\n"
                f"Phase 1 Winner: {phase1.get('timeframe')} @ {phase1.get('interval')}\n"
                f"Phase 2 Winner: {phase2.get('winner', 'N/A').upper()}\n"
            )
        else:
            notification['message'] = (
                f"Two-Phase Training Failed!\n"
                f"Error: {run_record.get('error', 'Unknown error')}\n"
            )

        # Write to notifications file
        notifications_file = 'logs/two_phase_notifications.jsonl'
        with open(notifications_file, 'a') as f:
            f.write(json.dumps(notification) + '\n')

        self.logger.info(f"Notification logged: {notifications_file}")

    def run_daemon(self):
        """Run as daemon - check schedule and execute training."""
        self.logger.info("=" * 60)
        self.logger.info("Two-Phase Training Scheduler Started")
        self.logger.info("=" * 60)
        self.logger.info(f"Config: {self.config_file}")
        self.logger.info(f"Schedule: {self.config['schedule']}")
        self.logger.info(f"Mode: {self.config['mode']}")
        self.logger.info(f"Auto-deploy: {self.config['auto_deploy']}")
        self.logger.info(f"Enabled: {self.config['enabled']}")

        if not self.config['enabled']:
            self.logger.warning("Two-phase training is DISABLED in config")
            self.logger.info("Set TWO_PHASE_ENABLED=true in .env.training to enable")

        # Calculate next run
        if self.state.get('next_run') is None:
            self.state['next_run'] = self.calculate_next_run().isoformat()
            self.save_state()

        self.logger.info(f"Next scheduled run: {self.state['next_run']}")
        self.logger.info("=" * 60)

        self.running = True
        check_interval = 60  # Check every minute

        while self.running:
            try:
                if self.should_run_now():
                    self.logger.info("Scheduled run triggered")
                    self.run_training()
                else:
                    # Log next run time periodically
                    next_run = datetime.fromisoformat(self.state['next_run'])
                    time_until = (next_run - datetime.now()).total_seconds()

                    if time_until > 0:
                        self.logger.debug(f"Next run in {time_until/3600:.1f} hours")

                time.sleep(check_interval)

            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(check_interval)

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        self.logger.info("Shutdown signal received")
        self.running = False
        sys.exit(0)

    def show_schedule(self):
        """Display current schedule information."""
        print("=" * 60)
        print("Two-Phase Training Schedule")
        print("=" * 60)
        print(f"Enabled:      {self.config['enabled']}")
        print(f"Schedule:     {self.config['schedule']}")
        print(f"Day:          {self.config['day']}")
        print(f"Time:         {self.config['time']}")
        print(f"Mode:         {self.config['mode']}")
        print(f"Auto-deploy:  {self.config['auto_deploy']}")
        print(f"Notification: {self.config['notification']}")
        print()

        if self.state.get('next_run'):
            next_run = datetime.fromisoformat(self.state['next_run'])
            time_until = (next_run - datetime.now()).total_seconds()
            print(f"Next run:     {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Time until:   {time_until/3600:.1f} hours")
        else:
            print("Next run:     Not scheduled")

        print()

        if self.state.get('last_run'):
            print(f"Last run:     {self.state['last_run']}")

        print()

        if self.state.get('runs'):
            print(f"Total runs:   {len(self.state['runs'])}")
            successful = sum(1 for r in self.state['runs'] if r['success'])
            print(f"Successful:   {successful}/{len(self.state['runs'])}")

            # Show last 3 runs
            print("\nRecent runs:")
            for run in self.state['runs'][-3:]:
                status = "✓" if run['success'] else "✗"
                ts = run['timestamp'][:19]
                duration = run['duration'] / 3600
                print(f"  {status} {ts} ({duration:.2f}h) - {run['mode']}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Two-Phase Training Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--daemon', action='store_true',
                        help='Run as daemon (continuous scheduling)')
    parser.add_argument('--run-now', action='store_true',
                        help='Run training immediately (one-shot)')
    parser.add_argument('--show-schedule', action='store_true',
                        help='Show current schedule')
    parser.add_argument('--config', type=str, default='.env.training',
                        help='Config file (default: .env.training)')

    args = parser.parse_args()

    scheduler = TwoPhaseScheduler(config_file=args.config)

    if args.show_schedule:
        scheduler.show_schedule()
    elif args.run_now:
        scheduler.logger.info("Running immediate one-shot training")
        scheduler.run_training()
    elif args.daemon:
        scheduler.run_daemon()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
