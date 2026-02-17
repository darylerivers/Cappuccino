#!/usr/bin/env python3
"""
Automated Training Pipeline

Fully automated training system that:
1. Cleans old logs and trials
2. Starts fresh training with dynamic study naming
3. Archives top 10% of trials automatically
4. Deploys best models to paper trading
5. No manual study management required

Usage:
    # Run full automated cycle
    python automated_training_pipeline.py --mode full

    # Clean and start training only
    python automated_training_pipeline.py --mode training

    # Archive existing trials only
    python automated_training_pipeline.py --mode archive

    # Deploy best model to paper trading
    python automated_training_pipeline.py --mode deploy
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import json
import psutil

# Add parent to path
PARENT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from utils.trial_manager import TrialManager
from utils.trial_naming import generate_trial_vin


def get_recommended_workers() -> int:
    """
    Determine recommended number of workers based on available RAM.

    Returns:
        Recommended worker count (2-3)
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)

    # Conservative allocation: ~10-12GB per worker for safety
    if available_gb >= 24:
        return 3
    elif available_gb >= 12:
        return 2
    else:
        return 1


class AutomatedTrainingPipeline:
    """Automated end-to-end training pipeline."""

    def __init__(
        self,
        base_dir: Path = None,
        n_workers: int = None,
        n_trials_per_worker: int = 500,
        gpu_id: int = 0
    ):
        """
        Initialize automated pipeline.

        Args:
            base_dir: Base directory (defaults to cappuccino root)
            n_workers: Number of parallel training workers (auto-detect if None)
            n_trials_per_worker: Trials per worker
            gpu_id: GPU device ID
        """
        self.base_dir = Path(base_dir) if base_dir else PARENT_DIR

        # Auto-detect workers based on available RAM if not specified
        if n_workers is None:
            self.n_workers = get_recommended_workers()
            print(f"Auto-detected {self.n_workers} workers based on available RAM")
        else:
            self.n_workers = min(n_workers, 3)  # Cap at 3 for stability

        self.n_trials = n_trials_per_worker
        self.gpu_id = gpu_id

        self.trial_manager = TrialManager(base_dir=self.base_dir)
        self.worker_pids = []
        self.study_name = None

    def cleanup_old_data(self, keep_days: int = 7, dry_run: bool = False):
        """Clean up old logs and trial data."""
        print(f"\n{'='*70}")
        print(f"STEP 1: CLEANUP OLD DATA")
        print(f"{'='*70}\n")

        # Clean logs
        self.trial_manager.clean_old_logs(keep_days=keep_days, dry_run=dry_run)

        # Clean old trial directories (keep top 20 for safety)
        self.trial_manager.clean_old_trials(keep_top_n=20, dry_run=dry_run)

    def generate_study_name(self) -> str:
        """Generate a unique study name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"cappuccino_auto_{timestamp}"

    def start_training(
        self,
        data_dir: str = "data/1h_1680",
        timeframe: str = "1h"
    ) -> str:
        """
        Start training workers.

        Args:
            data_dir: Directory containing training data
            timeframe: Timeframe for training (1h, 5m, etc.)

        Returns:
            Study name
        """
        print(f"\n{'='*70}")
        print(f"STEP 2: START TRAINING")
        print(f"{'='*70}\n")

        self.study_name = self.generate_study_name()
        print(f"📊 Study Name: {self.study_name}")
        print(f"👥 Workers: {self.n_workers}")
        print(f"🎯 Trials per worker: {self.n_trials}")
        print(f"💾 Data: {data_dir}")
        print(f"⏱️  Timeframe: {timeframe}\n")

        # Save current study name
        current_study_file = self.base_dir / ".current_study"
        with open(current_study_file, 'w') as f:
            f.write(self.study_name)

        # Launch workers
        script_path = self.base_dir / "scripts/training/1_optimize_unified.py"
        logs_dir = self.base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        for worker_id in range(1, self.n_workers + 1):
            log_file = logs_dir / f"worker_auto_{worker_id}.log"

            # Use pyenv ROCm environment
            pyenv_python = Path.home() / ".pyenv/versions/cappuccino-rocm/bin/python"
            python_cmd = str(pyenv_python) if pyenv_python.exists() else "python"

            cmd = [
                python_cmd, "-u",
                str(script_path),
                "--n-trials", str(self.n_trials),
                "--gpu", str(self.gpu_id),
                "--study-name", self.study_name,
                "--timeframe", timeframe,
                "--data-dir", data_dir
            ]

            # Set ROCm environment variables
            env = os.environ.copy()
            env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'  # For RX 7900

            # Start worker in background
            with open(log_file, 'w') as log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.base_dir),
                    env=env
                )
                self.worker_pids.append(proc.pid)
                print(f"✅ Worker {worker_id} started (PID: {proc.pid}, Log: {log_file.name})")

            time.sleep(2)  # Stagger worker starts

        print(f"\n🚀 All {self.n_workers} workers launched!")
        print(f"📊 Monitor progress: tail -f logs/worker_auto_*.log")

        return self.study_name

    def wait_for_training(self, check_interval: int = 60):
        """
        Wait for training to complete.

        Args:
            check_interval: Seconds between checks
        """
        print(f"\n{'='*70}")
        print(f"WAITING FOR TRAINING TO COMPLETE")
        print(f"{'='*70}\n")

        print(f"⏳ Checking worker status every {check_interval}s...")
        print(f"   Press Ctrl+C to stop monitoring (workers will continue)\n")

        try:
            while True:
                # Check if any workers are still running
                running = []
                for pid in self.worker_pids:
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        running.append(pid)
                    except OSError:
                        pass  # Process finished

                if not running:
                    print("\n✅ All workers completed!")
                    break

                print(f"⏳ {len(running)}/{len(self.worker_pids)} workers still running... (PIDs: {running})")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n⚠️  Monitoring stopped (workers continue in background)")
            print(f"   Worker PIDs: {self.worker_pids}")

    def archive_best_trials(self, top_percent: int = 10) -> List[dict]:
        """
        Archive the best trials from the current study.

        Args:
            top_percent: Top percentage to archive

        Returns:
            List of archived trial metadata
        """
        print(f"\n{'='*70}")
        print(f"STEP 3: ARCHIVE BEST TRIALS")
        print(f"{'='*70}\n")

        if not self.study_name:
            # Try to load from .current_study
            current_study_file = self.base_dir / ".current_study"
            if current_study_file.exists():
                self.study_name = current_study_file.read_text().strip()
            else:
                print("⚠️  No study name available, using latest")

        archived = self.trial_manager.archive_top_trials(
            study_name=self.study_name,
            top_percentile=top_percent
        )

        return archived

    def deploy_best_model(self, deployment_slot: int = 0) -> Optional[dict]:
        """
        Deploy the best model to paper trading.

        Args:
            deployment_slot: Which deployment slot to use (0-9)

        Returns:
            Metadata of deployed model
        """
        print(f"\n{'='*70}")
        print(f"STEP 4: DEPLOY BEST MODEL")
        print(f"{'='*70}\n")

        best_trial = self.trial_manager.get_best_trial_for_deployment()
        if not best_trial:
            print("⚠️  No archived trials available for deployment")
            return None

        print(f"🎯 Deploying: {best_trial['vin']}")
        print(f"   Grade: {best_trial['grade']}")
        print(f"   Sharpe: {best_trial['sharpe']:.4f}")

        # Copy model to deployment directory
        deployment_dir = self.base_dir / "deployments" / f"model_{deployment_slot}"
        deployment_dir.mkdir(parents=True, exist_ok=True)

        trial_dir = self.trial_manager.archive_dir / best_trial['vin']
        model_file = trial_dir / best_trial.get('model_file', 'actor.pth')

        if model_file.exists():
            import shutil
            shutil.copy2(model_file, deployment_dir / 'actor.pth')
            shutil.copy2(trial_dir / 'metadata.json', deployment_dir / 'trial_metadata.json')
            shutil.copy2(trial_dir / 'hyperparams.json', deployment_dir / 'hyperparams.json')

            print(f"✅ Model deployed to {deployment_dir}")

            # Create deployment info
            deploy_info = {
                'deployed_at': datetime.now().isoformat(),
                'vin': best_trial['vin'],
                'grade': best_trial['grade'],
                'sharpe': best_trial['sharpe'],
                'trial_number': best_trial['trial_number'],
                'slot': deployment_slot
            }

            with open(deployment_dir / 'deployment_info.json', 'w') as f:
                json.dump(deploy_info, f, indent=2)

            return deploy_info
        else:
            print(f"❌ Model file not found: {model_file}")
            return None

    def run_full_pipeline(
        self,
        data_dir: str = "data/1h_1680",
        timeframe: str = "1h",
        wait_for_completion: bool = True,
        deploy_after: bool = True
    ):
        """
        Run the full automated pipeline.

        Args:
            data_dir: Training data directory
            timeframe: Timeframe for training
            wait_for_completion: Wait for training to finish before archiving
            deploy_after: Deploy best model after archiving
        """
        print(f"\n{'='*70}")
        print(f"AUTOMATED TRAINING PIPELINE")
        print(f"{'='*70}\n")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Step 1: Cleanup
        self.cleanup_old_data(keep_days=7, dry_run=False)

        # Step 2: Start training
        self.start_training(data_dir=data_dir, timeframe=timeframe)

        # Step 3: Wait for completion (optional)
        if wait_for_completion:
            self.wait_for_training()

            # Step 4: Archive best trials
            archived = self.archive_best_trials(top_percent=10)

            if archived:
                # Step 5: Deploy (optional)
                if deploy_after:
                    self.deploy_best_model(deployment_slot=0)

            # Print summary
            self.trial_manager.print_summary()
        else:
            print("\n⚠️  Training started in background")
            print(f"   Study: {self.study_name}")
            print(f"   Workers: {self.worker_pids}")
            print(f"\n   To archive results later, run:")
            print(f"   python automated_training_pipeline.py --mode archive\n")

    def stop_all_workers(self):
        """Stop all running workers."""
        print("\n🛑 Stopping all workers...")
        for pid in self.worker_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"   Stopped PID {pid}")
            except OSError:
                pass  # Already stopped


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full automated pipeline (clean, train, archive, deploy)
    python automated_training_pipeline.py --mode full

    # Start training in background (don't wait for completion)
    python automated_training_pipeline.py --mode training --background

    # Archive results from current/previous training
    python automated_training_pipeline.py --mode archive

    # Deploy best archived model
    python automated_training_pipeline.py --mode deploy

    # Clean old data only
    python automated_training_pipeline.py --mode clean
        """
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'training', 'archive', 'deploy', 'clean'],
        default='full',
        help='Pipeline mode (default: full)'
    )
    parser.add_argument('--workers', type=int, default=3, help='Number of training workers')
    parser.add_argument('--trials', type=int, default=500, help='Trials per worker')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--data-dir', type=str, default='data/1h_1680', help='Training data directory')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (1h, 5m, etc.)')
    parser.add_argument('--background', action='store_true', help='Run training in background')
    parser.add_argument('--top-percent', type=int, default=10, help='Top percentage to archive')
    parser.add_argument('--deployment-slot', type=int, default=0, help='Deployment slot (0-9)')

    args = parser.parse_args()

    pipeline = AutomatedTrainingPipeline(
        n_workers=args.workers,
        n_trials_per_worker=args.trials,
        gpu_id=args.gpu
    )

    try:
        if args.mode == 'full':
            pipeline.run_full_pipeline(
                data_dir=args.data_dir,
                timeframe=args.timeframe,
                wait_for_completion=not args.background,
                deploy_after=True
            )

        elif args.mode == 'training':
            pipeline.cleanup_old_data()
            pipeline.start_training(data_dir=args.data_dir, timeframe=args.timeframe)

            if not args.background:
                pipeline.wait_for_training()

        elif args.mode == 'archive':
            pipeline.archive_best_trials(top_percent=args.top_percent)
            pipeline.trial_manager.print_summary()

        elif args.mode == 'deploy':
            pipeline.deploy_best_model(deployment_slot=args.deployment_slot)

        elif args.mode == 'clean':
            pipeline.cleanup_old_data()

        print(f"\n✅ Pipeline completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        pipeline.stop_all_workers()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
