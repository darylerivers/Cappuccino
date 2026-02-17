"""
Memory Safety Monitor for Cappuccino Training System

Prevents OOM crashes by monitoring system memory and providing
emergency brake functionality to kill trials before they crash the system.
"""

import psutil
import os
import sys
import time
from typing import Tuple, Optional
import optuna


class MemoryMonitor:
    """Monitor system memory and provide emergency brake for OOM prevention."""

    def __init__(self, safe_threshold_gb: float = 2.0, warning_threshold_gb: float = 4.0,
                 consecutive_brake_limit: int = 10):
        """
        Initialize memory monitor.

        Args:
            safe_threshold_gb: Minimum free RAM (GB) before emergency kill
            warning_threshold_gb: Free RAM (GB) to trigger warnings
            consecutive_brake_limit: Number of consecutive brakes before worker restart
        """
        self.safe_threshold = safe_threshold_gb * (1024 ** 3)  # Convert to bytes
        self.warning_threshold = warning_threshold_gb * (1024 ** 3)
        self.last_warning_time = 0
        self.warning_cooldown = 30  # seconds between warnings

        # Enhanced: Track consecutive emergency brakes
        self.consecutive_brakes = 0
        self.consecutive_brake_limit = consecutive_brake_limit
        self.last_brake_time = 0

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024 ** 3),
            'available_gb': mem.available / (1024 ** 3),
            'used_gb': mem.used / (1024 ** 3),
            'percent': mem.percent,
            'free_gb': mem.free / (1024 ** 3),
        }

    def check_memory_safety(self, context: str = "") -> Tuple[bool, dict]:
        """
        Check if memory is in safe range.

        Args:
            context: Description of where check is happening (for logging)

        Returns:
            (is_safe, memory_stats) tuple
        """
        stats = self.get_memory_stats()
        available_bytes = stats['available_gb'] * (1024 ** 3)

        is_safe = available_bytes >= self.safe_threshold
        is_warning = available_bytes < self.warning_threshold

        # Print warnings with cooldown to avoid spam
        current_time = time.time()
        if is_warning and (current_time - self.last_warning_time) > self.warning_cooldown:
            if not is_safe:
                print(f"\n{'='*80}")
                print(f"🚨 CRITICAL MEMORY WARNING {context}")
                print(f"{'='*80}")
                print(f"Available: {stats['available_gb']:.2f} GB")
                print(f"Threshold: {self.safe_threshold / (1024**3):.2f} GB")
                print(f"Used: {stats['used_gb']:.2f} GB / {stats['total_gb']:.2f} GB ({stats['percent']:.1f}%)")
                print(f"{'='*80}\n")
            else:
                print(f"\n⚠️  Memory warning {context}: {stats['available_gb']:.2f} GB available "
                      f"(threshold: {self.warning_threshold / (1024**3):.2f} GB)\n")
            self.last_warning_time = current_time
            sys.stdout.flush()

        return is_safe, stats

    def emergency_brake(self, trial: optuna.Trial, context: str = "") -> None:
        """
        Emergency brake - kill trial to prevent system OOM crash.

        This marks the trial with worst possible score and raises TrialPruned
        to gracefully terminate before system crashes.

        Args:
            trial: Optuna trial to kill
            context: Description of where emergency was triggered
        """
        stats = self.get_memory_stats()

        print(f"\n{'='*80}")
        print(f"🛑 EMERGENCY BRAKE ACTIVATED {context}")
        print(f"{'='*80}")
        print(f"Trial #{trial.number} killed to prevent OOM crash")
        print(f"Available memory: {stats['available_gb']:.2f} GB")
        print(f"Safety threshold: {self.safe_threshold / (1024**3):.2f} GB")
        print(f"Memory usage: {stats['used_gb']:.2f} GB / {stats['total_gb']:.2f} GB ({stats['percent']:.1f}%)")
        print(f"{'='*80}\n")
        sys.stdout.flush()

        # Mark trial as emergency killed
        trial.set_user_attr('emergency_killed', True)
        trial.set_user_attr('kill_reason', f'OOM prevention {context}')
        trial.set_user_attr('available_gb_at_kill', stats['available_gb'])

        # Raise TrialPruned with worst score to ensure this trial is marked as failed
        raise optuna.exceptions.TrialPruned(
            f"Emergency brake: OOM risk at {context} "
            f"({stats['available_gb']:.2f} GB available < {self.safe_threshold / (1024**3):.2f} GB threshold)"
        )

    def check_and_brake(self, trial: optuna.Trial, context: str = "") -> None:
        """
        Convenience method: check memory and trigger emergency brake if unsafe.

        Args:
            trial: Optuna trial object
            context: Description of where check is happening
        """
        is_safe, stats = self.check_memory_safety(context)
        if not is_safe:
            self.consecutive_brakes += 1
            self.last_brake_time = time.time()

            # Check if we've hit the consecutive brake limit
            if self.consecutive_brakes >= self.consecutive_brake_limit:
                self.trigger_worker_restart(trial, context, stats)
            else:
                self.emergency_brake(trial, context)
        else:
            # Reset counter if memory is safe
            if self.consecutive_brakes > 0 and (time.time() - self.last_brake_time) > 60:
                # Only reset if it's been > 60s since last brake
                self.consecutive_brakes = 0

    def trigger_worker_restart(self, trial: optuna.Trial, context: str, stats: dict) -> None:
        """
        Trigger worker restart by exiting the process.

        Called when emergency brake fires too many times consecutively,
        indicating sustained high memory that won't resolve without restart.

        Args:
            trial: Optuna trial being killed
            context: Description of where emergency was triggered
            stats: Memory statistics
        """
        print(f"\n{'='*80}")
        print(f"🚨🚨🚨 CRITICAL: WORKER RESTART REQUIRED 🚨🚨🚨")
        print(f"{'='*80}")
        print(f"Emergency brake fired {self.consecutive_brakes} times consecutively")
        print(f"Limit: {self.consecutive_brake_limit} consecutive brakes")
        print(f"Memory stuck at: {stats['used_gb']:.2f} GB / {stats['total_gb']:.2f} GB ({stats['percent']:.1f}%)")
        print(f"Available: {stats['available_gb']:.2f} GB (threshold: {self.safe_threshold / (1024**3):.2f} GB)")
        print(f"\nWORKER PROCESS WILL EXIT - Watchdog will restart it")
        print(f"{'='*80}\n")
        sys.stdout.flush()

        # Mark trial
        trial.set_user_attr('worker_restart_triggered', True)
        trial.set_user_attr('consecutive_brakes', self.consecutive_brakes)
        trial.set_user_attr('kill_reason', f'Worker restart: {self.consecutive_brakes} consecutive brakes')

        # Exit the worker process - watchdog will restart it
        # Use exit code 42 to indicate intentional restart (not a crash)
        print(f"Exiting worker process (PID {os.getpid()}) with exit code 42...")
        sys.stdout.flush()
        time.sleep(2)  # Give logs time to flush
        os._exit(42)  # Immediate exit, bypass Python cleanup

    def get_process_memory(self, pid: Optional[int] = None) -> float:
        """
        Get memory usage of specific process in GB.

        Args:
            pid: Process ID (defaults to current process)

        Returns:
            Memory usage in GB
        """
        if pid is None:
            pid = os.getpid()

        try:
            process = psutil.Process(pid)
            mem_info = process.memory_info()
            return mem_info.rss / (1024 ** 3)  # RSS in GB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    def log_memory_snapshot(self, label: str = "") -> None:
        """Log detailed memory snapshot for debugging."""
        stats = self.get_memory_stats()
        process_mem = self.get_process_memory()

        print(f"\n{'─'*60}")
        print(f"📊 Memory Snapshot {label}")
        print(f"{'─'*60}")
        print(f"System Total:     {stats['total_gb']:.2f} GB")
        print(f"System Used:      {stats['used_gb']:.2f} GB ({stats['percent']:.1f}%)")
        print(f"System Available: {stats['available_gb']:.2f} GB")
        print(f"System Free:      {stats['free_gb']:.2f} GB")
        print(f"This Process:     {process_mem:.2f} GB (PID {os.getpid()})")
        print(f"{'─'*60}\n")
        sys.stdout.flush()


# Global singleton instance
_global_monitor = None

def get_monitor(safe_threshold_gb: float = 2.0, warning_threshold_gb: float = 4.0,
                consecutive_brake_limit: int = 10) -> MemoryMonitor:
    """Get or create global memory monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor(safe_threshold_gb, warning_threshold_gb, consecutive_brake_limit)
    return _global_monitor


def check_memory(trial: optuna.Trial, context: str = "",
                 safe_threshold_gb: float = 2.0,
                 consecutive_brake_limit: int = 10) -> None:
    """
    Quick check function for use in training scripts.

    Usage:
        from utils.memory_monitor import check_memory
        check_memory(trial, "before training split 3")

    Args:
        trial: Optuna trial object
        context: Description of checkpoint
        safe_threshold_gb: Safety threshold in GB
        consecutive_brake_limit: Number of consecutive brakes before worker restart
    """
    monitor = get_monitor(safe_threshold_gb=safe_threshold_gb,
                         consecutive_brake_limit=consecutive_brake_limit)
    monitor.check_and_brake(trial, context)


if __name__ == '__main__':
    # Test the monitor
    print("Testing Memory Monitor\n")

    monitor = MemoryMonitor(safe_threshold_gb=2.0, warning_threshold_gb=4.0)
    monitor.log_memory_snapshot("Startup Test")

    is_safe, stats = monitor.check_memory_safety("Test Context")

    if is_safe:
        print("✅ Memory is in safe range")
    else:
        print("❌ Memory is below safety threshold!")

    print(f"\nCurrent available: {stats['available_gb']:.2f} GB")
    print(f"Safety threshold:  {monitor.safe_threshold / (1024**3):.2f} GB")
    print(f"Warning threshold: {monitor.warning_threshold / (1024**3):.2f} GB")
