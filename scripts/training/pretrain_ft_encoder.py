#!/usr/bin/env python3
"""
Pre-train FT-Transformer Feature Encoder

Uses self-supervised learning on historical crypto data to learn feature representations
before RL training. This warm-starts the encoder with knowledge of feature patterns.

Pre-training Task: Masked Feature Modeling
- Randomly mask a subset of input features
- Train encoder + reconstruction head to predict masked values
- Similar to BERT for language, but adapted for numerical tabular data

Benefits:
1. Learn feature correlations and interactions
2. Improve sample efficiency during RL training
3. Better generalization to unseen market conditions

Usage:
    # Pre-train on 1h data with default settings
    python scripts/training/pretrain_ft_encoder.py --data-dir data/1h_1680

    # Custom hyperparameters
    python scripts/training/pretrain_ft_encoder.py \
        --data-dir data/1h_1680 \
        --d-token 96 \
        --n-blocks 3 \
        --epochs 50 \
        --batch-size 128
"""

import argparse
import sys
from pathlib import Path
import pickle
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
PARENT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from models.ft_transformer_encoder import FTTransformerEncoder


class CryptoStateDataset(Dataset):
    """
    Dataset of crypto trading states for pre-training.

    Each sample is a state vector: [cash, stocks, tech_features]
    We'll use the tech_features across time for pre-training.
    """
    def __init__(self, price_array, tech_array, n_crypto=7, lookback=60):
        """
        Args:
            price_array: [timesteps, n_crypto] - price data
            tech_array: [timesteps, n_features] - technical indicators
            n_crypto: Number of cryptocurrencies
            lookback: Lookback window for state construction
        """
        self.price_array = price_array
        self.tech_array = tech_array
        self.n_crypto = n_crypto
        self.lookback = lookback

        # Calculate state dimension
        # 1 (cash) + n_crypto (stocks) + tech_features * lookback
        self.n_tech_features = tech_array.shape[1]
        self.state_dim = 1 + n_crypto + (self.n_tech_features * lookback)

        # Valid indices (need lookback history)
        self.valid_indices = list(range(lookback, len(price_array)))

        print(f"Dataset created:")
        print(f"  Total timesteps: {len(price_array)}")
        print(f"  Valid samples: {len(self.valid_indices)}")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Tech features per timestep: {self.n_tech_features}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Build a state vector similar to environment.get_state()

        We'll use random cash/stocks values for pre-training since
        we're focusing on learning feature representations.
        """
        time_idx = self.valid_indices[idx]

        # Random cash and stocks (normalized to [0, 1] range)
        cash = np.random.rand(1).astype(np.float32)
        stocks = np.random.rand(self.n_crypto).astype(np.float32)

        # Build state: [cash, stocks, tech_features_lookback]
        state = np.concatenate([cash, stocks])

        # Add lookback tech features
        for i in range(self.lookback):
            tech_i = self.tech_array[time_idx - i]
            state = np.concatenate([state, tech_i])

        return torch.from_numpy(state.astype(np.float32))


class MaskedFeaturePredictor(nn.Module):
    """
    FT-Transformer encoder + reconstruction head for masked feature modeling.

    Architecture:
    1. Mask random features in input
    2. Encode with FT-Transformer
    3. Reconstruct masked features with MLP head
    """
    def __init__(self, encoder: FTTransformerEncoder, state_dim: int):
        super().__init__()
        self.encoder = encoder
        self.state_dim = state_dim

        # Reconstruction head: encoding_dim -> state_dim
        # Predicts all features (we'll only use loss on masked ones)
        encoding_dim = encoder.output_dim
        self.reconstruction_head = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(encoding_dim // 2, state_dim)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, state_dim] - input state
            mask: [batch, state_dim] - binary mask (1 = keep, 0 = mask)

        Returns:
            reconstructed: [batch, state_dim] - predicted features
        """
        # Apply mask to input (set masked features to 0)
        if mask is not None:
            x_masked = x * mask
        else:
            x_masked = x

        # Encode
        encoded = self.encoder(x_masked)

        # Reconstruct
        reconstructed = self.reconstruction_head(encoded)

        return reconstructed


def create_random_mask(batch_size, state_dim, mask_ratio=0.15, device='cpu'):
    """
    Create random binary mask for masked feature modeling.

    Args:
        batch_size: Number of samples in batch
        state_dim: Dimension of state space
        mask_ratio: Fraction of features to mask (default: 0.15, similar to BERT)
        device: torch device

    Returns:
        mask: [batch, state_dim] binary tensor (1 = keep, 0 = mask)
    """
    # Generate random mask
    mask = torch.rand(batch_size, state_dim, device=device)
    mask = (mask > mask_ratio).float()

    return mask


def train_epoch(model, dataloader, optimizer, mask_ratio, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, states in enumerate(dataloader):
        states = states.to(device)
        batch_size = states.shape[0]

        # Create random mask
        mask = create_random_mask(batch_size, states.shape[1], mask_ratio, device)

        # Forward pass
        reconstructed = model(states, mask)

        # Loss only on masked features
        # MSE between original and reconstructed for masked positions
        loss = torch.mean(((reconstructed - states) ** 2) * (1 - mask))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_samples
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {avg_loss:.6f}")

    avg_loss = total_loss / total_samples
    return avg_loss


def validate_epoch(model, dataloader, mask_ratio, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for states in dataloader:
            states = states.to(device)
            batch_size = states.shape[0]

            # Create random mask
            mask = create_random_mask(batch_size, states.shape[1], mask_ratio, device)

            # Forward pass
            reconstructed = model(states, mask)

            # Loss only on masked features
            loss = torch.mean(((reconstructed - states) ** 2) * (1 - mask))

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Pre-train FT-Transformer encoder')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Data directory (e.g., data/1h_1680)')
    parser.add_argument('--n-crypto', type=int, default=7,
                        help='Number of cryptocurrencies (default: 7)')
    parser.add_argument('--lookback', type=int, default=60,
                        help='Lookback window (default: 60)')

    # Encoder hyperparameters
    parser.add_argument('--d-token', type=int, default=64,
                        help='Token embedding dimension (default: 64)')
    parser.add_argument('--n-blocks', type=int, default=2,
                        help='Number of transformer blocks (default: 2)')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads (default: 4)')
    parser.add_argument('--encoding-dim', type=int, default=512,
                        help='Output encoding dimension (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--mask-ratio', type=float, default=0.15,
                        help='Ratio of features to mask (default: 0.15)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')

    # Output settings
    parser.add_argument('--output-dir', type=str, default='train_results/pretrained_encoders',
                        help='Output directory for saved models')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')

    # Device
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (default: 0, -1 for CPU)')

    args = parser.parse_args()

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load data
    print(f"\n{'='*70}")
    print(f"Loading data from: {args.data_dir}")
    print(f"{'='*70}\n")

    data_path = Path(args.data_dir)

    # Custom unpickler for numpy compatibility
    class NumpyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'numpy._core.numeric':
                module = 'numpy.core.numeric'
            elif module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            return super().find_class(module, name)

    with open(data_path / 'price_array', 'rb') as f:
        price_array = NumpyUnpickler(f).load()
    with open(data_path / 'tech_array', 'rb') as f:
        tech_array = NumpyUnpickler(f).load()

    print(f"Data loaded:")
    print(f"  Price shape: {price_array.shape}")
    print(f"  Tech shape: {tech_array.shape}")

    # Create dataset
    dataset = CryptoStateDataset(
        price_array=price_array,
        tech_array=tech_array,
        n_crypto=args.n_crypto,
        lookback=args.lookback
    )

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    print(f"\n{'='*70}")
    print(f"Creating FT-Transformer encoder")
    print(f"{'='*70}\n")

    state_dim = dataset.state_dim

    encoder = FTTransformerEncoder(
        n_features=state_dim,
        d_token=args.d_token,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        d_ff_factor=4,
        dropout=args.dropout,
        attention_dropout=0.0,
        activation='relu',
        prenorm=True,
        output_dim=args.encoding_dim
    )

    model = MaskedFeaturePredictor(encoder, state_dim)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        'data_dir': args.data_dir,
        'n_crypto': args.n_crypto,
        'lookback': args.lookback,
        'state_dim': state_dim,
        'd_token': args.d_token,
        'n_blocks': args.n_blocks,
        'n_heads': args.n_heads,
        'encoding_dim': args.encoding_dim,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'mask_ratio': args.mask_ratio,
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"ft_encoder_{timestamp}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaving to: {run_dir}")

    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting pre-training")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    training_history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, args.mask_ratio, device
        )

        # Validate
        val_loss = validate_epoch(
            model, val_loader, args.mask_ratio, device
        )

        # Update learning rate
        scheduler.step()

        # Log
        lr = optimizer.param_groups[0]['lr']
        print(f"\n  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {lr:.6e}")

        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr
        })

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  ✓ New best validation loss!")

        if epoch % args.save_every == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'training_history': training_history
            }

            if is_best:
                torch.save(checkpoint, run_dir / 'best_encoder.pth')
                print(f"  Saved best checkpoint")

            if epoch % args.save_every == 0:
                torch.save(checkpoint, run_dir / f'encoder_epoch_{epoch}.pth')
                print(f"  Saved checkpoint for epoch {epoch}")

    # Save final checkpoint
    checkpoint = {
        'epoch': args.epochs,
        'encoder_state_dict': model.encoder.state_dict(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
        'training_history': training_history
    }

    torch.save(checkpoint, run_dir / 'final_encoder.pth')

    # Save training history
    with open(run_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Pre-training complete!")
    print(f"{'='*70}")
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Final validation loss: {val_loss:.6f}")
    print(f"\nCheckpoints saved to: {run_dir}")
    print(f"  - best_encoder.pth (best validation loss)")
    print(f"  - final_encoder.pth (final epoch)")
    print(f"  - encoder_epoch_N.pth (every {args.save_every} epochs)")


if __name__ == '__main__':
    main()
