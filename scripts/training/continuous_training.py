#!/usr/bin/env python3
"""
Continuous Training Daemon
Runs Optuna training trials continuously and integrates with pipeline.

Features:
- Runs trials one at a time with cooldown
- Preserves model weights for top performing trials
- Triggers pipeline check after each trial
- Monitors resource usage
- Graceful shutdown

Usage:
    python continuous_training.py
    python continuous_training.py --trials-per-cycle 5 --cooldown 300
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv('.env.training')

class ContinuousTrainer:
    """Manages continuous training with pipeline integration."""
    
    def __init__(self, 
                 trials_per_cycle: int = 1,
                 cooldown_seconds: int = 300,
                 study_name: str = None,
                 gpu_id: int = 0,
                 trigger_pipeline: bool = True):
        self.trials_per_cycle = trials_per_cycle
        self.cooldown_seconds = cooldown_seconds
        self.study_name = study_name or os.getenv('ACTIVE_STUDY_NAME')
        self.gpu_id = gpu_id
        self.trigger_pipeline = trigger_pipeline
        self.running = True
        
        # Setup logging
        self._setup_logging()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Continuous Trainer initialized")
        self.logger.info(f"Study: {self.study_name}")
        self.logger.info(f"Trials per cycle: {self.trials_per_cycle}")
        self.logger.info(f"Cooldown: {self.cooldown_seconds}s")
        self.logger.info(f"GPU: {self.gpu_id}")
        self.logger.info(f"Pipeline integration: {self.trigger_pipeline}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/continuous_training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run_training_cycle(self) -> bool:
        """
        Run one training cycle (N trials).
        
        Returns:
            True if successful, False if error
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Starting training cycle: {self.trials_per_cycle} trials")
        self.logger.info("=" * 70)
        
        try:
            # Build training command
            cmd = [
                sys.executable,
                "1_optimize_unified.py",
                "--n-trials", str(self.trials_per_cycle),
                "--gpu", str(self.gpu_id),
                "--study", self.study_name,
            ]
            
            self.logger.info(f"Command: {' '.join(cmd)}")
            
            # Run training
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=False,  # Let output go to terminal
                text=True,
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.info(f"✓ Training cycle completed in {duration:.1f}s")
                return True
            else:
                self.logger.error(f"✗ Training cycle failed with code {result.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Training error: {e}", exc_info=True)
            return False
    
    def trigger_pipeline_check(self):
        """Trigger pipeline to check for new trials."""
        if not self.trigger_pipeline:
            return
        
        self.logger.info("Triggering pipeline check...")
        
        try:
            result = subprocess.run(
                [sys.executable, "pipeline_orchestrator.py", "--once"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                self.logger.info("✓ Pipeline check completed")
            else:
                self.logger.warning(f"Pipeline check returned code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Pipeline check timed out")
        except Exception as e:
            self.logger.error(f"Pipeline trigger error: {e}")
    
    def check_resources(self) -> bool:
        """
        Check if system has resources available for training.
        
        Returns:
            True if resources available, False otherwise
        """
        try:
            # Check disk space
            import shutil
            stat = shutil.disk_usage('.')
            free_gb = stat.free / (1024**3)
            
            if free_gb < 5:  # Less than 5GB free
                self.logger.warning(f"Low disk space: {free_gb:.1f}GB free")
                return False
            
            # Check GPU availability (if using GPU)
            if self.gpu_id >= 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_mem = torch.cuda.mem_get_info(self.gpu_id)
                        free_mem_gb = gpu_mem[0] / (1024**3)
                        
                        if free_mem_gb < 1:  # Less than 1GB GPU memory
                            self.logger.warning(f"Low GPU memory: {free_mem_gb:.1f}GB free")
                            return False
                except:
                    pass
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Resource check error: {e}")
            return True  # Assume OK if check fails
    
    def run(self):
        """Main continuous training loop."""
        self.logger.info("=" * 70)
        self.logger.info("CONTINUOUS TRAINING STARTED")
        self.logger.info("=" * 70)
        
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"CYCLE {cycle_count}")
                self.logger.info(f"{'='*70}")
                
                # Check resources
                if not self.check_resources():
                    self.logger.warning("Insufficient resources, waiting...")
                    time.sleep(600)  # Wait 10 minutes
                    continue
                
                # Run training cycle
                success = self.run_training_cycle()
                
                if success:
                    # Trigger pipeline check
                    self.trigger_pipeline_check()
                else:
                    self.logger.warning("Training cycle failed, extending cooldown")
                    time.sleep(self.cooldown_seconds * 2)  # Double cooldown on failure
                    continue
                
                # Cooldown before next cycle
                if self.running:
                    self.logger.info(f"Cooldown for {self.cooldown_seconds}s...")
                    time.sleep(self.cooldown_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 minute on error
        
        self.logger.info("=" * 70)
        self.logger.info("CONTINUOUS TRAINING STOPPED")
        self.logger.info(f"Total cycles completed: {cycle_count}")
        self.logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Continuous training daemon")
    parser.add_argument("--trials-per-cycle", type=int, default=1,
                        help="Number of trials to run per cycle")
    parser.add_argument("--cooldown", type=int, default=300,
                        help="Cooldown seconds between cycles")
    parser.add_argument("--study", default=None,
                        help="Study name (default from .env.training)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="Don't trigger pipeline after training")
    
    args = parser.parse_args()
    
    trainer = ContinuousTrainer(
        trials_per_cycle=args.trials_per_cycle,
        cooldown_seconds=args.cooldown,
        study_name=args.study,
        gpu_id=args.gpu,
        trigger_pipeline=not args.no_pipeline
    )
    
    trainer.run()


if __name__ == "__main__":
    main()
