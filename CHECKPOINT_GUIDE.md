# Checkpoint System Guide

## Overview

The checkpoint system provides automatic save/resume functionality for long-running processes. This prevents data loss and allows resuming after interruptions.

---

## Features

✅ **Automatic Checkpointing**
- Save every N trials/episodes/steps
- Save on metric improvement
- Configurable frequency

✅ **Robust Storage**
- Atomic writes (no corruption)
- Optional compression
- Automatic versioning
- Keep last N checkpoints

✅ **Easy Recovery**
- One-line resume
- Progress tracking
- Recovery helpers

✅ **Multiple Process Types**
- Training (Optuna trials, RL episodes)
- Data downloads (resume partial downloads)
- Trading (portfolio state)
- Backtests (intermediate results)

---

## Quick Start

### 1. Training with Optuna

```python
from checkpoint_manager import CheckpointManager, TrainingCheckpointCallback

# Create checkpoint manager
manager = CheckpointManager(
    process_type='training',
    process_name='btc_eth_sol_experiment',
    keep_last_n=5  # Keep last 5 checkpoints
)

# Create callback
callback = TrainingCheckpointCallback(
    checkpoint_manager=manager,
    save_every_n_trials=10,  # Save every 10 trials
    save_on_improvement=True  # Save when best trial improves
)

# Run optimization with checkpointing
study.optimize(objective, n_trials=100, callbacks=[callback])
```

### 2. Resume Training

```python
from checkpoint_manager import CheckpointManager, resume_training

# Create manager (same config as before)
manager = CheckpointManager('training', 'btc_eth_sol_experiment')

# Try to resume
resumed, start_trial, state = resume_training(manager)

if resumed:
    print(f"Resuming from trial {start_trial}")
    best_value = state['best_value']
    best_params = state['best_params']
else:
    print("Starting from scratch")
    start_trial = 0
```

### 3. RL Training (Standalone)

```python
from checkpoint_manager import CheckpointManager, TrainingCheckpointCallback

manager = CheckpointManager('training', 'ppo_agent')
callback = TrainingCheckpointCallback(
    checkpoint_manager=manager,
    save_every_n_episodes=100,
    save_on_improvement=True
)

# Training loop
for episode in range(1000):
    # Train episode
    episode_reward = train_episode(agent, env)

    # Save checkpoint
    callback.on_episode_end(
        episode=episode,
        episode_reward=episode_reward,
        agent_state={'weights': agent.get_weights()},
        total_episodes=1000
    )
```

---

## Integration Examples

### Training Script with Checkpointing

```python
#!/usr/bin/env python3
"""
Training script with checkpoint support.
"""

import optuna
from checkpoint_manager import (
    CheckpointManager,
    TrainingCheckpointCallback,
    resume_training
)


def objective(trial):
    # Your optimization objective
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])

    # Train and evaluate
    sharpe = train_and_evaluate(lr, batch_size)
    return sharpe


def main():
    # Create checkpoint manager
    manager = CheckpointManager(
        process_type='training',
        process_name='my_experiment',
        checkpoint_dir='checkpoints',
        keep_last_n=5,
        compress=True
    )

    # Check for existing checkpoints
    recovery_info = manager.get_recovery_info()

    if recovery_info['has_checkpoint']:
        print(f"Found checkpoint at step {recovery_info['step']}")
        print(f"Progress: {recovery_info['progress_pct']:.1f}%")
        print("Resume? (y/n): ", end='')

        if input().lower() == 'y':
            resumed, start_step, state = resume_training(manager)
            # Continue from where we left off
            n_remaining_trials = 100 - start_step
        else:
            print("Starting from scratch")
            n_remaining_trials = 100
    else:
        n_remaining_trials = 100

    # Create study
    study = optuna.create_study(
        study_name='my_study',
        storage='sqlite:///optuna.db',
        direction='maximize',
        load_if_exists=True
    )

    # Create checkpoint callback
    callback = TrainingCheckpointCallback(
        checkpoint_manager=manager,
        save_every_n_trials=10,
        save_on_improvement=True
    )

    # Run optimization with checkpointing
    try:
        study.optimize(
            objective,
            n_trials=n_remaining_trials,
            callbacks=[callback]
        )

        # Save final checkpoint
        final_state = {
            'study_name': study.study_name,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }
        manager.save_checkpoint(
            final_state,
            step=len(study.trials),
            status='completed'
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted. Checkpoint saved.")
        print("Run again to resume.")


if __name__ == "__main__":
    main()
```

---

### Data Download with Checkpointing

```python
#!/usr/bin/env python3
"""
FRED data download with checkpoint support.
Resume partial downloads after interruptions.
"""

from checkpoint_manager import CheckpointManager
from processor_FRED import FREDProcessor


def download_with_checkpointing():
    # Create checkpoint manager
    manager = CheckpointManager(
        process_type='download',
        process_name='fred_data',
        checkpoint_dir='checkpoints'
    )

    # Check for existing checkpoint
    resumed, last_step, state = resume_training(
        manager,
        start_from_scratch_if_none=True
    )

    # List of series to download
    all_series = [
        'DFF', 'EFFR', 'WALCL', 'RRPONTSYD',
        'CPIAUCSL', 'CPILFESL', 'PCEPI',
        'T10Y2Y', 'BAMLH0A0HYM2', 'DTWEXBGS', 'VIXCLS',
        'NFCI', 'STLFSI4'
    ]

    # Resume from where we left off
    if resumed:
        downloaded_series = state.get('downloaded_series', [])
        print(f"Resuming: {len(downloaded_series)} series already downloaded")
    else:
        downloaded_series = []

    # Download remaining series
    processor = FREDProcessor()

    for i, series_id in enumerate(all_series):
        if series_id in downloaded_series:
            continue  # Skip already downloaded

        try:
            print(f"Downloading {series_id}...")
            data = processor.fred.get_series(series_id)

            # Save to checkpoint
            downloaded_series.append(series_id)

            manager.save_checkpoint(
                state={'downloaded_series': downloaded_series},
                step=len(downloaded_series),
                total_steps=len(all_series),
                status='in_progress'
            )

        except KeyboardInterrupt:
            print("\nDownload interrupted. Progress saved.")
            print(f"Downloaded {len(downloaded_series)}/{len(all_series)} series")
            return

    # Mark complete
    manager.save_checkpoint(
        state={'downloaded_series': downloaded_series},
        step=len(all_series),
        total_steps=len(all_series),
        status='completed'
    )

    print(f"✓ All {len(all_series)} series downloaded")
```

---

### Trading with State Checkpointing

```python
#!/usr/bin/env python3
"""
Paper trading with portfolio state checkpointing.
Recover from crashes without losing positions.
"""

from checkpoint_manager import CheckpointManager


class PaperTrader:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.positions = {}
        self.trades = []

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            process_type='trading',
            process_name='paper_trading',
            checkpoint_dir='checkpoints',
            keep_last_n=10  # Keep last 10 states
        )

        # Try to resume
        self._resume_state()

    def _resume_state(self):
        """Resume from last checkpoint."""
        resumed, _, state = resume_training(
            self.checkpoint_manager,
            start_from_scratch_if_none=True
        )

        if resumed:
            self.capital = state['capital']
            self.positions = state['positions']
            self.trades = state['trades']
            print(f"Resumed trading session")
            print(f"  Capital: ${self.capital:,.2f}")
            print(f"  Positions: {len(self.positions)}")
            print(f"  Trades: {len(self.trades)}")

    def save_state(self):
        """Save current portfolio state."""
        state = {
            'capital': self.capital,
            'positions': self.positions,
            'trades': self.trades,
            'total_value': self._calculate_portfolio_value()
        }

        self.checkpoint_manager.save_checkpoint(
            state,
            step=len(self.trades),
            status='in_progress'
        )

    def execute_trade(self, ticker, action, quantity, price):
        """Execute trade and save state."""
        # Execute trade logic
        if action == 'buy':
            cost = quantity * price
            self.capital -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
        elif action == 'sell':
            proceeds = quantity * price
            self.capital += proceeds
            self.positions[ticker] = self.positions.get(ticker, 0) - quantity

        self.trades.append({
            'ticker': ticker,
            'action': action,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().isoformat()
        })

        # Save state after each trade
        self.save_state()

    def run_trading_loop(self):
        """Main trading loop with automatic checkpointing."""
        try:
            while True:
                # Get trading signals
                signals = self.get_signals()

                # Execute trades
                for signal in signals:
                    self.execute_trade(**signal)

                # Save state every 10 trades
                if len(self.trades) % 10 == 0:
                    self.save_state()

                time.sleep(60)  # Wait 1 minute

        except KeyboardInterrupt:
            print("\nTrading stopped. State saved.")
            self.save_state()

        except Exception as e:
            print(f"Error: {e}")
            self.save_state()  # Save before crash
            raise
```

---

## CLI Usage

### List Checkpoints

```bash
python -c "
from checkpoint_manager import CheckpointManager

manager = CheckpointManager('training', 'my_experiment')

print('Available Checkpoints:')
for ckpt in manager.list_checkpoints():
    print(f\"  Step {ckpt['step']}: {ckpt['checkpoint_id']}\")
    print(f\"    Progress: {ckpt.get('progress_pct', 'N/A')}%\")
    print(f\"    Status: {ckpt['status']}\")
"
```

### Check Recovery Info

```bash
python -c "
from checkpoint_manager import CheckpointManager

manager = CheckpointManager('training', 'my_experiment')
info = manager.get_recovery_info()

print('Recovery Info:')
for key, value in info.items():
    print(f'  {key}: {value}')
"
```

### Delete Old Checkpoints

```bash
python -c "
from checkpoint_manager import CheckpointManager

manager = CheckpointManager('training', 'my_experiment')

# Delete specific checkpoint
manager.delete_checkpoint('checkpoint_step_42_20250127_120000')

# Or cleanup old ones (keeps last N)
manager._cleanup_old_checkpoints()
"
```

---

## Best Practices

### 1. **Save Frequency**
- Training: Every 10-50 trials (depending on trial length)
- RL: Every 100-1000 episodes
- Trading: Every trade or every N minutes
- Downloads: After each successful fetch

### 2. **What to Save**
**Training**:
- Trial/episode number
- Best metrics (Sharpe, returns, etc.)
- Hyperparameters
- Model weights (if needed)

**Trading**:
- Portfolio positions
- Cash balance
- Trade history
- Last update timestamp

**Downloads**:
- List of completed items
- Partial data (if applicable)
- Progress indicators

### 3. **Storage Management**
- Use `keep_last_n=5-10` to limit disk usage
- Enable compression for large checkpoints
- Save "best" checkpoints separately

### 4. **Error Handling**
```python
try:
    # Long-running process
    run_training()
except KeyboardInterrupt:
    manager.save_checkpoint(state, status='interrupted')
    print("Interrupted. Progress saved.")
except Exception as e:
    manager.save_checkpoint(state, status='failed')
    print(f"Failed: {e}. State saved for debugging.")
    raise
finally:
    # Always save final state
    manager.save_checkpoint(state, status='completed')
```

---

## Advanced Usage

### Custom Checkpoint Logic

```python
class CustomCheckpointManager(CheckpointManager):
    def save_model_weights(self, model, step):
        """Save model weights separately."""
        state = {
            'model_weights': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
        self.save_checkpoint(state, step)

    def load_model_weights(self, model, optimizer):
        """Load model weights from checkpoint."""
        latest = self.load_latest_checkpoint()
        if latest:
            model.load_state_dict(latest['state']['model_weights'])
            optimizer.load_state_dict(latest['state']['optimizer_state'])
            return True
        return False
```

### Distributed Training

```python
# Save on rank 0 only
if rank == 0:
    manager.save_checkpoint(state, step)

# All ranks can read
checkpoint = manager.load_latest_checkpoint()
```

---

## File Structure

Checkpoints are stored in:
```
checkpoints/
├── training/
│   ├── experiment_1/
│   │   ├── checkpoint_step_10_20250127_120000.pkl.gz
│   │   ├── checkpoint_step_20_20250127_120500.pkl.gz
│   │   └── best_trial_42.pkl.gz
│   └── experiment_2/
│       └── ...
├── trading/
│   └── paper_trading/
│       ├── checkpoint_step_100_20250127_120000.pkl.gz
│       └── ...
└── download/
    └── fred_data/
        └── ...
```

---

## Troubleshooting

### Issue: "Checkpoint not found"
**Solution**: Check process_type and process_name match

### Issue: "Corrupted checkpoint"
**Solution**: Load previous checkpoint:
```python
checkpoints = manager.list_checkpoints()
prev_ckpt = checkpoints[-2]  # Get second-to-last
data = manager.load_checkpoint(prev_ckpt['checkpoint_id'])
```

### Issue: "Out of disk space"
**Solution**: Reduce `keep_last_n` or enable compression

---

## Summary

✅ **Easy to use**: One-line integration
✅ **Robust**: Atomic writes, compression, versioning
✅ **Flexible**: Works with any process type
✅ **Production-ready**: Error handling, recovery helpers

Start checkpointing your long-running processes today!
