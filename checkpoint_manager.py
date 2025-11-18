#!/usr/bin/env python3
"""
Checkpoint Manager for Cappuccino Trading System

Provides robust checkpointing for long-running processes:
- Training runs (save every N trials/episodes)
- Data downloads (resume interrupted downloads)
- Trading sessions (save portfolio state)
- Backtests (save intermediate results)

Features:
- Automatic versioning
- Atomic writes (no corruption)
- Compression support
- Recovery helpers
- Progress tracking
"""

import os
import json
import pickle
import shutil
import hashlib
import gzip
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
import warnings


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files."""
    checkpoint_id: str
    timestamp: str
    process_type: str  # 'training', 'trading', 'download', 'backtest'
    process_name: str
    step: int
    total_steps: Optional[int]
    progress_pct: Optional[float]
    status: str  # 'in_progress', 'completed', 'failed'
    version: str = "1.0"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class CheckpointManager:
    """
    Manages checkpoints for long-running processes.

    Features:
    - Automatic save/load with versioning
    - Atomic writes (temp file -> rename)
    - Compression support
    - Multiple checkpoint retention
    - Recovery helpers

    Example:
        >>> manager = CheckpointManager('training', 'my_experiment')
        >>>
        >>> # Save checkpoint
        >>> manager.save_checkpoint({
        ...     'trial': 42,
        ...     'best_sharpe': 1.5,
        ...     'model_state': model.state_dict()
        ... }, step=42, total_steps=100)
        >>>
        >>> # Load latest checkpoint
        >>> checkpoint = manager.load_latest_checkpoint()
        >>> print(checkpoint['trial'])  # 42
    """

    def __init__(
        self,
        process_type: str,
        process_name: str,
        checkpoint_dir: str = "checkpoints",
        keep_last_n: int = 5,
        compress: bool = True,
        verbose: bool = True
    ):
        """
        Initialize CheckpointManager.

        Args:
            process_type: Type of process ('training', 'trading', 'download', 'backtest')
            process_name: Name of specific process (e.g., 'experiment_1', 'btc_eth_sol')
            checkpoint_dir: Base directory for checkpoints
            keep_last_n: Keep last N checkpoints (0 = keep all)
            compress: Use gzip compression
            verbose: Print checkpoint operations
        """
        self.process_type = process_type
        self.process_name = process_name
        self.checkpoint_dir = Path(checkpoint_dir) / process_type / process_name
        self.keep_last_n = keep_last_n
        self.compress = compress
        self.verbose = verbose

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        step: int,
        total_steps: Optional[int] = None,
        status: str = "in_progress",
        checkpoint_id: Optional[str] = None
    ) -> str:
        """
        Save checkpoint with metadata.

        Args:
            state: Dictionary containing state to save
            step: Current step/trial/episode number
            total_steps: Total steps (if known)
            status: Status ('in_progress', 'completed', 'failed')
            checkpoint_id: Custom checkpoint ID (default: auto-generated)

        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint ID if not provided
        if checkpoint_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_id = f"checkpoint_step_{step}_{timestamp}"

        # Calculate progress
        progress_pct = None
        if total_steps is not None and total_steps > 0:
            progress_pct = (step / total_steps) * 100

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            process_type=self.process_type,
            process_name=self.process_name,
            step=step,
            total_steps=total_steps,
            progress_pct=progress_pct,
            status=status
        )

        # Prepare checkpoint data
        checkpoint_data = {
            'metadata': metadata.to_dict(),
            'state': state
        }

        # Determine file extension
        ext = '.pkl.gz' if self.compress else '.pkl'
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}{ext}"
        temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')

        try:
            # Write to temp file first (atomic write)
            if self.compress:
                with gzip.open(temp_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(temp_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic rename
            shutil.move(str(temp_path), str(checkpoint_path))

            if self.verbose:
                progress_str = f" ({progress_pct:.1f}%)" if progress_pct else ""
                print(f"✓ Checkpoint saved: {checkpoint_id} (step {step}/{total_steps or '?'}){progress_str}")

            # Cleanup old checkpoints
            if self.keep_last_n > 0:
                self._cleanup_old_checkpoints()

            return str(checkpoint_path)

        except Exception as e:
            # Cleanup temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load specific checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID to load

        Returns:
            Dictionary with 'metadata' and 'state'
        """
        # Try both compressed and uncompressed
        for ext in ['.pkl.gz', '.pkl']:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}{ext}"
            if checkpoint_path.exists():
                try:
                    if ext == '.pkl.gz':
                        with gzip.open(checkpoint_path, 'rb') as f:
                            checkpoint_data = pickle.load(f)
                    else:
                        with open(checkpoint_path, 'rb') as f:
                            checkpoint_data = pickle.load(f)

                    if self.verbose:
                        step = checkpoint_data['metadata']['step']
                        print(f"✓ Checkpoint loaded: {checkpoint_id} (step {step})")

                    return checkpoint_data

                except Exception as e:
                    warnings.warn(f"Failed to load checkpoint {checkpoint_id}: {e}")
                    continue

        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Returns:
            Checkpoint data or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            if self.verbose:
                print("No checkpoints found")
            return None

        # Sort by step (latest first)
        latest = max(checkpoints, key=lambda x: x['step'])
        return self.load_checkpoint(latest['checkpoint_id'])

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = []

        for ext in ['.pkl.gz', '.pkl']:
            for ckpt_file in self.checkpoint_dir.glob(f"*{ext}"):
                if ckpt_file.name.endswith('.tmp'):
                    continue

                try:
                    if ext == '.pkl.gz':
                        with gzip.open(ckpt_file, 'rb') as f:
                            data = pickle.load(f)
                    else:
                        with open(ckpt_file, 'rb') as f:
                            data = pickle.load(f)

                    checkpoints.append(data['metadata'])

                except Exception as e:
                    warnings.warn(f"Failed to read checkpoint {ckpt_file}: {e}")

        # Sort by step
        checkpoints.sort(key=lambda x: x['step'])
        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str):
        """Delete specific checkpoint."""
        for ext in ['.pkl.gz', '.pkl']:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}{ext}"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                if self.verbose:
                    print(f"✓ Deleted checkpoint: {checkpoint_id}")
                return

        warnings.warn(f"Checkpoint not found: {checkpoint_id}")

    def _cleanup_old_checkpoints(self):
        """Keep only last N checkpoints."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= self.keep_last_n:
            return

        # Sort by step (oldest first)
        checkpoints.sort(key=lambda x: x['step'])

        # Delete oldest checkpoints
        to_delete = checkpoints[:-self.keep_last_n]
        for ckpt in to_delete:
            try:
                self.delete_checkpoint(ckpt['checkpoint_id'])
            except Exception as e:
                warnings.warn(f"Failed to delete old checkpoint: {e}")

    def get_recovery_info(self) -> Dict[str, Any]:
        """
        Get information for recovery.

        Returns:
            Dictionary with recovery information
        """
        latest = self.load_latest_checkpoint()
        if latest is None:
            return {
                'has_checkpoint': False,
                'message': 'No checkpoints available'
            }

        metadata = latest['metadata']

        return {
            'has_checkpoint': True,
            'checkpoint_id': metadata['checkpoint_id'],
            'step': metadata['step'],
            'total_steps': metadata['total_steps'],
            'progress_pct': metadata['progress_pct'],
            'timestamp': metadata['timestamp'],
            'status': metadata['status'],
            'can_resume': metadata['status'] == 'in_progress'
        }


class TrainingCheckpointCallback:
    """
    Callback for saving checkpoints during training.

    Use with Optuna or standalone training loops.

    Example with Optuna:
        >>> callback = TrainingCheckpointCallback(
        ...     checkpoint_manager=manager,
        ...     save_every_n_trials=10
        ... )
        >>>
        >>> study.optimize(objective, n_trials=100, callbacks=[callback])

    Example standalone:
        >>> callback = TrainingCheckpointCallback(manager, save_every_n_episodes=100)
        >>>
        >>> for episode in range(1000):
        ...     # Training code
        ...     callback.on_episode_end(episode, episode_reward, agent_state)
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        save_every_n_trials: Optional[int] = None,
        save_every_n_episodes: Optional[int] = None,
        save_on_improvement: bool = True
    ):
        """
        Initialize training checkpoint callback.

        Args:
            checkpoint_manager: CheckpointManager instance
            save_every_n_trials: Save every N Optuna trials
            save_every_n_episodes: Save every N training episodes
            save_on_improvement: Save when metric improves
        """
        self.manager = checkpoint_manager
        self.save_every_n_trials = save_every_n_trials
        self.save_every_n_episodes = save_every_n_episodes
        self.save_on_improvement = save_on_improvement
        self.best_value = float('-inf')

    def __call__(self, study, trial):
        """
        Optuna callback function.

        Args:
            study: Optuna study
            trial: Completed trial
        """
        if self.save_every_n_trials and trial.number % self.save_every_n_trials == 0:
            state = {
                'study_name': study.study_name,
                'trial_number': trial.number,
                'best_value': study.best_value,
                'best_params': study.best_params,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials)
            }

            self.manager.save_checkpoint(
                state,
                step=trial.number,
                total_steps=None,  # Optuna doesn't have fixed total
                status='in_progress'
            )

        # Save on improvement
        if self.save_on_improvement and trial.value > self.best_value:
            self.best_value = trial.value
            state = {
                'study_name': study.study_name,
                'trial_number': trial.number,
                'best_value': trial.value,
                'best_params': trial.params,
                'improvement': 'best_so_far'
            }

            self.manager.save_checkpoint(
                state,
                step=trial.number,
                checkpoint_id=f"best_trial_{trial.number}",
                status='in_progress'
            )

    def on_episode_end(
        self,
        episode: int,
        episode_reward: float,
        agent_state: Dict[str, Any],
        total_episodes: Optional[int] = None
    ):
        """
        Save checkpoint after episode.

        Args:
            episode: Episode number
            episode_reward: Reward for this episode
            agent_state: Agent state dictionary (e.g., model weights)
            total_episodes: Total episodes (if known)
        """
        if self.save_every_n_episodes and episode % self.save_every_n_episodes == 0:
            state = {
                'episode': episode,
                'episode_reward': episode_reward,
                'agent_state': agent_state,
                'best_value': self.best_value
            }

            self.manager.save_checkpoint(
                state,
                step=episode,
                total_steps=total_episodes,
                status='in_progress'
            )

        # Save on improvement
        if self.save_on_improvement and episode_reward > self.best_value:
            self.best_value = episode_reward
            state = {
                'episode': episode,
                'episode_reward': episode_reward,
                'agent_state': agent_state,
                'improvement': 'best_so_far'
            }

            self.manager.save_checkpoint(
                state,
                step=episode,
                checkpoint_id=f"best_episode_{episode}",
                total_steps=total_episodes,
                status='in_progress'
            )


def resume_training(
    checkpoint_manager: CheckpointManager,
    start_from_scratch_if_none: bool = True
) -> tuple[bool, int, Dict[str, Any]]:
    """
    Helper to resume training from checkpoint.

    Args:
        checkpoint_manager: CheckpointManager instance
        start_from_scratch_if_none: If True, start from 0 if no checkpoint

    Returns:
        (resumed, start_step, state)
        - resumed: True if resuming from checkpoint
        - start_step: Step to start from (0 if new)
        - state: Checkpoint state (empty dict if new)

    Example:
        >>> manager = CheckpointManager('training', 'my_exp')
        >>> resumed, start_step, state = resume_training(manager)
        >>>
        >>> if resumed:
        ...     print(f"Resuming from step {start_step}")
        ...     model.load_state_dict(state['agent_state'])
        ... else:
        ...     print("Starting from scratch")
    """
    latest = checkpoint_manager.load_latest_checkpoint()

    if latest is None:
        if start_from_scratch_if_none:
            print("No checkpoint found. Starting from scratch.")
            return False, 0, {}
        else:
            raise RuntimeError("No checkpoint found and start_from_scratch_if_none=False")

    metadata = latest['metadata']
    state = latest['state']

    print(f"Found checkpoint: {metadata['checkpoint_id']}")
    print(f"  Step: {metadata['step']}")
    print(f"  Progress: {metadata.get('progress_pct', 'Unknown')}%")
    print(f"  Timestamp: {metadata['timestamp']}")

    return True, metadata['step'], state


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("CHECKPOINT MANAGER - EXAMPLES")
    print("=" * 80)

    # Example 1: Training checkpoints
    print("\n1. Training Checkpoint Example")
    print("-" * 80)

    manager = CheckpointManager(
        process_type='training',
        process_name='example_experiment',
        checkpoint_dir='example_checkpoints',
        keep_last_n=3,
        compress=True
    )

    # Simulate training
    for trial in range(1, 11):
        state = {
            'trial': trial,
            'best_sharpe': 0.5 + trial * 0.1,
            'hyperparams': {'learning_rate': 0.001}
        }

        manager.save_checkpoint(
            state,
            step=trial,
            total_steps=10,
            status='in_progress'
        )

    # Mark complete
    manager.save_checkpoint(
        state,
        step=10,
        total_steps=10,
        status='completed'
    )

    # List checkpoints
    print("\nAvailable checkpoints:")
    for ckpt in manager.list_checkpoints():
        print(f"  - Step {ckpt['step']}: {ckpt['checkpoint_id']} ({ckpt['status']})")

    # Load latest
    print("\nLoading latest checkpoint...")
    latest = manager.load_latest_checkpoint()
    print(f"  Trial: {latest['state']['trial']}")
    print(f"  Best Sharpe: {latest['state']['best_sharpe']}")

    # Recovery info
    print("\nRecovery Info:")
    info = manager.get_recovery_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Cleanup
    shutil.rmtree('example_checkpoints')
    print("\n✓ Example complete (cleaned up)")
