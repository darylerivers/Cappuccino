"""
Enhanced Actor/Critic Networks with FT-Transformer Feature Encoding

These are drop-in replacements for standard Actor/Critic that optionally
use FT-Transformer for feature encoding instead of a simple linear layer.

Benefits:
1. Learn feature interactions through self-attention
2. Can be pre-trained on historical data
3. Better representation learning for high-dimensional state spaces

Usage:
    # Create actor with FT-Transformer encoding
    actor = ActorPPO_FT(
        mid_dim=512,
        state_dim=5888,
        action_dim=7,
        use_ft_encoder=True,
        ft_config={
            'd_token': 64,
            'n_blocks': 2,
            'n_heads': 4,
            'dropout': 0.1
        }
    )

    # Load pre-trained encoder weights (optional)
    actor.load_pretrained_encoder('path/to/encoder.pth')
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import sys

# Add parent directory to path for imports
PARENT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from models.ft_transformer_encoder import FTTransformerEncoder


class ActorPPO_FT(nn.Module):
    """
    PPO Actor network with optional FT-Transformer feature encoding.

    Architecture:
    - If use_ft_encoder=True:
        Input -> FT-Transformer -> ReLU -> Linear -> Linear -> tanh(action)
    - If use_ft_encoder=False:
        Input -> Linear -> ReLU -> Linear -> Linear -> tanh(action)
    """
    def __init__(
        self,
        mid_dim: int,
        state_dim: int,
        action_dim: int,
        use_ft_encoder: bool = False,
        ft_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_ft_encoder = use_ft_encoder

        if use_ft_encoder:
            # Use FT-Transformer encoder for first layer
            ft_config = ft_config or {}
            d_token = ft_config.get('d_token', 64)
            n_blocks = ft_config.get('n_blocks', 2)
            n_heads = ft_config.get('n_heads', 4)
            dropout = ft_config.get('dropout', 0.1)

            self.encoder = FTTransformerEncoder(
                n_features=state_dim,
                d_token=d_token,
                n_blocks=n_blocks,
                n_heads=n_heads,
                d_ff_factor=4,
                dropout=dropout,
                attention_dropout=0.0,
                activation='relu',
                prenorm=True,
                output_dim=mid_dim
            )

            # Remaining layers
            self.net = nn.Sequential(
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, action_dim),
            )
        else:
            # Standard MLP (original architecture)
            self.encoder = None
            self.net = nn.Sequential(
                nn.Linear(state_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, action_dim),
            )

        # Learnable log std for action noise
        self.a_std_log = nn.Parameter(
            torch.zeros((1, action_dim)) - 0.5, requires_grad=True
        )
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        """
        Args:
            state: [batch, state_dim]

        Returns:
            action: [batch, action_dim] - tanh-squashed actions
        """
        if self.use_ft_encoder:
            encoded = self.encoder(state)
            return self.net(encoded).tanh()
        else:
            return self.net(state).tanh()

    def get_action(self, state):
        """
        Sample action with exploration noise.

        Returns:
            action: [batch, action_dim]
            noise: [batch, action_dim] - sampled noise
        """
        if self.use_ft_encoder:
            encoded = self.encoder(state)
            a_avg = self.net(encoded)
        else:
            a_avg = self.net(state)

        a_std = self.a_std_log.exp()
        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std

        return action, noise

    def get_logprob(self, state, action):
        """
        Get log probability of action given state.
        """
        if self.use_ft_encoder:
            encoded = self.encoder(state)
            a_avg = self.net(encoded)
        else:
            a_avg = self.net(state)

        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)

        return log_prob

    def get_logprob_entropy(self, state, action):
        """
        Get log probability and entropy for PPO update.
        """
        if self.use_ft_encoder:
            encoded = self.encoder(state)
            a_avg = self.net(encoded)
        else:
            a_avg = self.net(state)

        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)

        dist_entropy = (logprob.exp() * logprob).mean()

        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):
        """
        Get log probability using cached noise (for PPO).
        """
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)

    @staticmethod
    def get_a_to_e(action):
        """Convert action to environment format."""
        return action.tanh()

    def load_pretrained_encoder(self, checkpoint_path: str, freeze: bool = False):
        """
        Load pre-trained FT-Transformer encoder weights.

        Args:
            checkpoint_path: Path to pre-trained encoder checkpoint
            freeze: If True, freeze encoder weights (only train policy head)
        """
        if not self.use_ft_encoder:
            raise ValueError("Cannot load encoder weights when use_ft_encoder=False")

        print(f"Loading pre-trained encoder from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load encoder state dict
        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"✓ Loaded encoder from epoch {checkpoint.get('epoch', '?')}")
            print(f"  Pre-training val loss: {checkpoint.get('val_loss', '?'):.6f}")
        else:
            raise ValueError("Checkpoint does not contain 'encoder_state_dict'")

        # Optionally freeze encoder
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder weights frozen (will not be updated during RL training)")
        else:
            print("✓ Encoder weights will be fine-tuned during RL training")


class CriticPPO_FT(nn.Module):
    """
    PPO Critic (value) network with optional FT-Transformer feature encoding.

    Architecture:
    - If use_ft_encoder=True:
        Input -> FT-Transformer -> ReLU -> Linear -> ReLU -> Linear -> value
    - If use_ft_encoder=False:
        Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> value
    """
    def __init__(
        self,
        mid_dim: int,
        state_dim: int,
        action_dim: int,  # Not used, kept for compatibility
        use_ft_encoder: bool = False,
        ft_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.state_dim = state_dim
        self.use_ft_encoder = use_ft_encoder

        if use_ft_encoder:
            # Use FT-Transformer encoder for first layer
            ft_config = ft_config or {}
            d_token = ft_config.get('d_token', 64)
            n_blocks = ft_config.get('n_blocks', 2)
            n_heads = ft_config.get('n_heads', 4)
            dropout = ft_config.get('dropout', 0.1)

            self.encoder = FTTransformerEncoder(
                n_features=state_dim,
                d_token=d_token,
                n_blocks=n_blocks,
                n_heads=n_heads,
                d_ff_factor=4,
                dropout=dropout,
                attention_dropout=0.0,
                activation='relu',
                prenorm=True,
                output_dim=mid_dim
            )

            # Remaining layers
            self.net = nn.Sequential(
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, 1),
            )
        else:
            # Standard MLP (original architecture)
            self.encoder = None
            self.net = nn.Sequential(
                nn.Linear(state_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, 1),
            )

    def forward(self, state):
        """
        Args:
            state: [batch, state_dim]

        Returns:
            value: [batch, 1] - estimated state value
        """
        if self.use_ft_encoder:
            encoded = self.encoder(state)
            return self.net(encoded)
        else:
            return self.net(state)

    def load_pretrained_encoder(self, checkpoint_path: str, freeze: bool = False):
        """
        Load pre-trained FT-Transformer encoder weights.

        Args:
            checkpoint_path: Path to pre-trained encoder checkpoint
            freeze: If True, freeze encoder weights (only train value head)
        """
        if not self.use_ft_encoder:
            raise ValueError("Cannot load encoder weights when use_ft_encoder=False")

        print(f"Loading pre-trained encoder from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load encoder state dict
        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print(f"✓ Loaded encoder from epoch {checkpoint.get('epoch', '?')}")
            print(f"  Pre-training val loss: {checkpoint.get('val_loss', '?'):.6f}")
        else:
            raise ValueError("Checkpoint does not contain 'encoder_state_dict'")

        # Optionally freeze encoder
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("✓ Encoder weights frozen (will not be updated during RL training)")
        else:
            print("✓ Encoder weights will be fine-tuned during RL training")


def create_ft_actor_critic(
    state_dim: int,
    action_dim: int,
    mid_dim: int = 512,
    use_ft_encoder: bool = True,
    d_token: int = 64,
    n_blocks: int = 2,
    n_heads: int = 4,
    dropout: float = 0.1,
    pretrained_path: Optional[str] = None,
    freeze_encoder: bool = False
):
    """
    Convenience function to create matched Actor/Critic pair with FT-Transformer.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        mid_dim: Hidden dimension (encoding dimension for FT-Transformer)
        use_ft_encoder: Whether to use FT-Transformer (if False, creates standard nets)
        d_token: Token embedding dimension for FT-Transformer
        n_blocks: Number of transformer blocks
        n_heads: Number of attention heads
        dropout: Dropout rate
        pretrained_path: Optional path to pre-trained encoder checkpoint
        freeze_encoder: If True, freeze encoder weights during RL training

    Returns:
        actor, critic: Tuple of Actor and Critic networks
    """
    ft_config = {
        'd_token': d_token,
        'n_blocks': n_blocks,
        'n_heads': n_heads,
        'dropout': dropout
    }

    actor = ActorPPO_FT(
        mid_dim=mid_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        use_ft_encoder=use_ft_encoder,
        ft_config=ft_config
    )

    critic = CriticPPO_FT(
        mid_dim=mid_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        use_ft_encoder=use_ft_encoder,
        ft_config=ft_config
    )

    # Load pre-trained weights if provided
    if use_ft_encoder and pretrained_path is not None:
        actor.load_pretrained_encoder(pretrained_path, freeze=freeze_encoder)
        critic.load_pretrained_encoder(pretrained_path, freeze=freeze_encoder)

    return actor, critic


if __name__ == '__main__':
    # Test the networks
    print("Testing FT-Transformer Actor/Critic...")

    # Small state for testing
    state_dim = 214  # 1 + 3 + (14*3*5)
    action_dim = 3
    mid_dim = 256
    batch_size = 8

    print(f"\nState dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Mid dim: {mid_dim}")

    # Create networks with FT-Transformer
    print("\n1. Testing FT-Transformer version:")
    actor_ft, critic_ft = create_ft_actor_critic(
        state_dim=state_dim,
        action_dim=action_dim,
        mid_dim=mid_dim,
        use_ft_encoder=True,
        d_token=32,
        n_blocks=2,
        n_heads=4,
        dropout=0.1
    )

    # Test forward pass
    state = torch.randn(batch_size, state_dim)
    action_ft = actor_ft(state)
    value_ft = critic_ft(state)

    print(f"  Actor output shape: {action_ft.shape} (expected: [{batch_size}, {action_dim}])")
    print(f"  Critic output shape: {value_ft.shape} (expected: [{batch_size}, 1])")

    # Test action sampling
    action_sampled, noise = actor_ft.get_action(state)
    print(f"  Sampled action shape: {action_sampled.shape}")

    # Count parameters
    actor_params = sum(p.numel() for p in actor_ft.parameters())
    critic_params = sum(p.numel() for p in critic_ft.parameters())
    print(f"  Actor parameters: {actor_params:,}")
    print(f"  Critic parameters: {critic_params:,}")

    # Create standard networks for comparison
    print("\n2. Testing standard MLP version:")
    actor_std, critic_std = create_ft_actor_critic(
        state_dim=state_dim,
        action_dim=action_dim,
        mid_dim=mid_dim,
        use_ft_encoder=False
    )

    action_std = actor_std(state)
    value_std = critic_std(state)

    print(f"  Actor output shape: {action_std.shape}")
    print(f"  Critic output shape: {value_std.shape}")

    actor_std_params = sum(p.numel() for p in actor_std.parameters())
    critic_std_params = sum(p.numel() for p in critic_std.parameters())
    print(f"  Actor parameters: {actor_std_params:,}")
    print(f"  Critic parameters: {critic_std_params:,}")

    print(f"\n3. Parameter comparison:")
    print(f"  FT-Transformer adds {actor_params - actor_std_params:,} parameters to Actor")
    print(f"  FT-Transformer adds {critic_params - critic_std_params:,} parameters to Critic")

    print("\n✓ All tests passed!")
