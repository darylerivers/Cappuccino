#!/usr/bin/env python3
"""
Sequential Trial Scheduler

Runs Optuna trials one at a time to avoid memory issues.
Automatically restarts when trials are killed by OOM or crashes.

Usage:
    python sequential_trial_scheduler.py --study cappuccino_auto_20260214_2059 --n-trials 100
"""

import os
import sys
import time
import signal
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import psutil

# Add parent to path
PARENT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))


class SequentialTrialScheduler:
    """Run trials sequentially with automatic recovery from crashes."""

    def __init__(
        self,
        study_name: str,
        n_trials: int = 100,
        gpu_id: int = 0,
        n_envs: int = 12,
        data_dir: str = "data/1h_1680",
        timeframe: str = "1h",
        base_dir: Path = None
    ):
        self.study_name = study_name
        self.n_trials = n_trials
        self.gpu_id = gpu_id
        self.n_envs = n_envs
        self.data_dir = data_dir
        self.timeframe = timeframe
        self.base_dir = Path(base_dir) if base_dir else PARENT_DIR

        self.current_process = None
        self.trials_completed = 0
        self.trials_failed = 0
        self.start_time = datetime.now()
        self.stop_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n\n⚠️  Shutdown signal received, cleaning up...")
        self.stop_requested = True
        if self.current_process:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
        sys.exit(0)

    def _get_python_cmd(self) -> str:
        """Get the correct Python executable (prefer pyenv ROCm)."""
        pyenv_python = Path.home() / ".pyenv/versions/cappuccino-rocm/bin/python"
        if pyenv_python.exists():
            return str(pyenv_python)
        return "python"

    def _prepare_environment(self) -> dict:
        """Prepare environment variables for ROCm GPU."""
        env = os.environ.copy()
        env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
        env['PYTHONUNBUFFERED'] = '1'
        return env

    def _get_memory_usage(self) -> tuple:
        """Get current memory usage (used_gb, total_gb, percent)."""
        mem = psutil.virtual_memory()
        return (mem.used / 1024**3, mem.total / 1024**3, mem.percent)

    def _run_single_trial(self, trial_number: int) -> bool:
        """
        Run a single trial.

        Returns:
            True if trial completed successfully, False if crashed
        """
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_number}/{self.n_trials}")
        print(f"Study: {self.study_name}")
        print(f"{'='*80}")

        # Check memory before starting
        mem_used, mem_total, mem_pct = self._get_memory_usage()
        print(f"💾 Memory: {mem_used:.1f}GB / {mem_total:.1f}GB ({mem_pct:.1f}%)")

        if mem_pct > 85:
            print(f"⚠️  Warning: Memory usage high, waiting 30s for cleanup...")
            time.sleep(30)

        # Prepare command
        python_cmd = self._get_python_cmd()
        script = self.base_dir / "scripts/training/1_optimize_unified.py"

        cmd = [
            python_cmd, "-u", str(script),
            "--n-trials", "1",  # Only 1 trial per worker
            "--gpu", str(self.gpu_id),
            "--n-envs", str(self.n_envs),  # GPU-accelerated parallel environments
            "--study-name", self.study_name,
            "--timeframe", self.timeframe,
            "--data-dir", self.data_dir
        ]

        # Prepare environment
        env = self._prepare_environment()

        # Start process
        print(f"🚀 Starting trial...")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        try:
            self.current_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Monitor process output
            for line in self.current_process.stdout:
                print(line, end='')

                # Check for stop request
                if self.stop_requested:
                    self.current_process.terminate()
                    return False

            # Wait for completion
            exit_code = self.current_process.wait()

            if exit_code == 0:
                print(f"\n✅ Trial {trial_number} completed successfully")
                return True
            else:
                print(f"\n❌ Trial {trial_number} failed with exit code {exit_code}")
                return False

        except Exception as e:
            print(f"\n❌ Trial {trial_number} crashed: {e}")
            return False
        finally:
            self.current_process = None

    def run(self):
        """Run trials sequentially."""
        print(f"\n{'='*80}")
        print(f"SEQUENTIAL TRIAL SCHEDULER")
        print(f"{'='*80}")
        print(f"Study: {self.study_name}")
        print(f"Total trials to run: {self.n_trials}")
        print(f"GPU: {self.gpu_id}")
        print(f"Parallel envs: {self.n_envs} (GPU-accelerated)" if self.n_envs > 1 else "Parallel envs: 1 (CPU mode)")
        print(f"Data: {self.data_dir} ({self.timeframe})")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        for i in range(1, self.n_trials + 1):
            if self.stop_requested:
                break

            success = self._run_single_trial(i)

            if success:
                self.trials_completed += 1
            else:
                self.trials_failed += 1
                print(f"⚠️  Trial failed, will retry with next trial")
                print(f"   Waiting 10s before starting next trial...")
                time.sleep(10)

            # Print progress
            elapsed = datetime.now() - self.start_time
            print(f"\n📊 Progress: {self.trials_completed} completed, {self.trials_failed} failed")
            print(f"⏱️  Elapsed: {elapsed}")
            if self.trials_completed > 0:
                avg_time = elapsed.total_seconds() / (self.trials_completed + self.trials_failed)
                remaining = avg_time * (self.n_trials - i)
                print(f"⏳ Estimated remaining: {remaining/3600:.1f}h")
            print()

        # Final summary
        print(f"\n{'='*80}")
        print(f"SCHEDULER FINISHED")
        print(f"{'='*80}")
        print(f"Total trials attempted: {self.trials_completed + self.trials_failed}")
        print(f"Completed successfully: {self.trials_completed}")
        print(f"Failed/Crashed: {self.trials_failed}")
        print(f"Total time: {datetime.now() - self.start_time}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Trial Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 50 trials sequentially on current study
  python sequential_trial_scheduler.py --study cappuccino_auto_20260214_2059 --n-trials 50

  # Run with 5-minute data
  python sequential_trial_scheduler.py --study my_study --n-trials 100 \\
      --data-dir data/5m_crypto --timeframe 5m

  # Run on CPU
  python sequential_trial_scheduler.py --study my_study --n-trials 20 --gpu -1
        """
    )

    parser.add_argument('--study', type=str, required=True,
                       help='Optuna study name')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of trials to run (default: 100)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU, default: 0)')
    parser.add_argument('--n-envs', type=int, default=12,
                       help='Parallel GPU environments (8-16 recommended, 1=CPU mode)')
    parser.add_argument('--data-dir', type=str, default='data/1h_1680',
                       help='Training data directory')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe (1h, 5m, etc.)')

    args = parser.parse_args()

    scheduler = SequentialTrialScheduler(
        study_name=args.study,
        n_trials=args.n_trials,
        gpu_id=args.gpu,
        n_envs=args.n_envs,
        data_dir=args.data_dir,
        timeframe=args.timeframe
    )

    scheduler.run()


if __name__ == '__main__':
    main()
