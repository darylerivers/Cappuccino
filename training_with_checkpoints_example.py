#!/usr/bin/env python3
"""
Example: Training Script with Full Checkpoint Support

This is a complete example showing how to add checkpoints to your training pipeline.
Copy this pattern to your own training scripts.

Features:
- Auto-save every N trials
- Save on improvement
- Resume from last checkpoint
- Handle interruptions gracefully
- Cleanup old checkpoints
"""

import optuna
import sys
from pathlib import Path

# Add parent directory for imports (if needed)
sys.path.insert(0, str(Path(__file__).parent))

from checkpoint_manager import (
    CheckpointManager,
    TrainingCheckpointCallback,
    resume_training
)


def objective(trial):
    """
    Your optimization objective.

    This is just an example - replace with your actual training code.
    """
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    gamma = trial.suggest_float('gamma', 0.90, 0.99)

    # Simulate training (replace with actual training)
    import time
    import random

    print(f"  Training with lr={lr:.6f}, batch_size={batch_size}, gamma={gamma:.3f}")
    time.sleep(0.5)  # Simulate training time

    # Return metric (replace with actual evaluation)
    sharpe_ratio = random.uniform(0.5, 2.0)
    return sharpe_ratio


def main():
    """Main training function with checkpoint support."""

    # ========================================================================
    # STEP 1: Create Checkpoint Manager
    # ========================================================================
    print("=" * 80)
    print("TRAINING WITH CHECKPOINT SUPPORT")
    print("=" * 80)

    checkpoint_manager = CheckpointManager(
        process_type='training',
        process_name='example_experiment',  # Change to your experiment name
        checkpoint_dir='checkpoints',  # Where to save checkpoints
        keep_last_n=5,  # Keep last 5 checkpoints (saves disk space)
        compress=True,  # Use gzip compression
        verbose=True   # Print checkpoint operations
    )

    # ========================================================================
    # STEP 2: Check for Existing Checkpoints
    # ========================================================================
    recovery_info = checkpoint_manager.get_recovery_info()

    if recovery_info['has_checkpoint']:
        print(f"\n{'='*80}")
        print("CHECKPOINT FOUND")
        print(f"{'='*80}")
        print(f"  Checkpoint ID: {recovery_info['checkpoint_id']}")
        print(f"  Step: {recovery_info['step']}")
        print(f"  Progress: {recovery_info.get('progress_pct', 'Unknown')}%")
        print(f"  Timestamp: {recovery_info['timestamp']}")
        print(f"  Status: {recovery_info['status']}")
        print(f"{'='*80}")

        # Ask user if they want to resume
        print("\nResume from checkpoint? (y/n): ", end='', flush=True)
        response = input().strip().lower()

        if response == 'y':
            resumed, start_step, state = resume_training(checkpoint_manager)

            # Extract previous state
            n_completed_trials = state.get('n_trials', 0)
            best_value = state.get('best_value', None)
            best_params = state.get('best_params', None)

            print(f"\nResuming optimization:")
            print(f"  Completed trials: {n_completed_trials}")
            if best_value is not None:
                print(f"  Best value so far: {best_value:.4f}")
            if best_params is not None:
                print(f"  Best params: {best_params}")

            resume_from = True
        else:
            print("\nStarting fresh optimization (ignoring checkpoint)")
            resume_from = False
            n_completed_trials = 0
    else:
        print("\nNo checkpoint found. Starting fresh optimization.")
        resume_from = False
        n_completed_trials = 0

    # ========================================================================
    # STEP 3: Create Optuna Study
    # ========================================================================
    total_trials = 50  # Total trials to run

    study = optuna.create_study(
        study_name='example_study',
        storage='sqlite:///databases/example_optuna.db',  # Persist study
        direction='maximize',
        load_if_exists=True  # Load if study exists
    )

    # If resuming, we've already done some trials
    if resume_from:
        remaining_trials = max(0, total_trials - n_completed_trials)
        print(f"\nRemaining trials: {remaining_trials}/{total_trials}")
    else:
        remaining_trials = total_trials

    # ========================================================================
    # STEP 4: Create Checkpoint Callback
    # ========================================================================
    checkpoint_callback = TrainingCheckpointCallback(
        checkpoint_manager=checkpoint_manager,
        save_every_n_trials=10,  # Save every 10 trials
        save_on_improvement=True  # Save when best value improves
    )

    # ========================================================================
    # STEP 5: Run Optimization with Checkpointing
    # ========================================================================
    print(f"\n{'='*80}")
    print("STARTING OPTIMIZATION")
    print(f"{'='*80}")

    try:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[checkpoint_callback],
            show_progress_bar=True
        )

        # Save final checkpoint
        final_state = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number
        }

        checkpoint_manager.save_checkpoint(
            final_state,
            step=len(study.trials),
            total_steps=total_trials,
            status='completed'
        )

        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"  Total trials: {len(study.trials)}")
        print(f"  Best value: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")
        print(f"  Best trial: {study.best_trial.number}")
        print(f"{'='*80}")

    except KeyboardInterrupt:
        print(f"\n{'='*80}")
        print("OPTIMIZATION INTERRUPTED")
        print(f"{'='*80}")
        print("Checkpoint saved. Run again to resume.")
        print(f"Completed trials: {len(study.trials)}/{total_trials}")

        # Save interrupted state
        interrupted_state = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'best_value': study.best_value if study.trials else None,
            'best_params': study.best_params if study.trials else None,
            'interrupted': True
        }

        checkpoint_manager.save_checkpoint(
            interrupted_state,
            step=len(study.trials),
            total_steps=total_trials,
            status='interrupted'
        )

    except Exception as e:
        print(f"\n{'='*80}")
        print("OPTIMIZATION FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")

        # Save failed state for debugging
        failed_state = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'error': str(e),
            'error_type': type(e).__name__
        }

        checkpoint_manager.save_checkpoint(
            failed_state,
            step=len(study.trials),
            status='failed'
        )

        raise  # Re-raise exception


if __name__ == "__main__":
    # Create databases directory
    Path('databases').mkdir(exist_ok=True)

    # Run training
    main()
