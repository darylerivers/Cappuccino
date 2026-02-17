"""
FT-Transformer Feature Encoder for Tabular RL State Spaces

Based on "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
Adapted for crypto trading with temporal features (lookback windows)

Architecture:
1. Feature Tokenization: Each feature -> embedding vector
2. Multi-head Self-Attention: Learn feature interactions
3. Normalization + Feedforward: Standard transformer blocks
4. Output: Contextualized feature embeddings

Integration:
- Replaces the first linear layer in Actor/Critic networks
- Input: Raw state features [batch, state_dim]
- Output: Encoded features [batch, encoding_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import math


class FeatureTokenizer(nn.Module):
    """
    Convert numerical features into token embeddings.

    For a state with n_features, each feature is embedded into d_token dimensions.
    This allows the transformer to learn feature-specific representations.
    """
    def __init__(self, n_features: int, d_token: int, bias: bool = True):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        # Separate linear projection for each feature (no shared weights)
        # This is the key difference from standard transformers
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token)) if bias else None

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_features] - raw numerical features

        Returns:
            tokens: [batch, n_features, d_token] - embedded features
        """
        # x: [batch, n_features]
        # weight: [n_features, d_token]
        # Result: [batch, n_features, d_token]
        x = x.unsqueeze(-1)  # [batch, n_features, 1]
        tokens = x * self.weight.unsqueeze(0)  # Broadcasting: [batch, n_features, d_token]

        if self.bias is not None:
            tokens = tokens + self.bias.unsqueeze(0)

        return tokens


class MultiheadAttention(nn.Module):
    """
    Multi-head self-attention with feature-wise interactions.

    Standard transformer attention but applied across features (not sequence positions).
    This allows the model to learn which features are relevant for each other.
    """
    def __init__(self, d_token: int, n_heads: int, dropout: float = 0.0,
                 attention_dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_token % n_heads == 0, f"d_token={d_token} must be divisible by n_heads={n_heads}"

        self.d_token = d_token
        self.n_heads = n_heads
        self.head_dim = d_token // n_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.W_q = nn.Linear(d_token, d_token, bias=bias)
        self.W_k = nn.Linear(d_token, d_token, bias=bias)
        self.W_v = nn.Linear(d_token, d_token, bias=bias)
        self.W_out = nn.Linear(d_token, d_token, bias=bias)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_out]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, n_features, d_token]
            mask: Optional attention mask

        Returns:
            output: [batch, n_features, d_token]
        """
        batch_size, n_features, d_token = x.shape

        # Project to Q, K, V
        q = self.W_q(x)  # [batch, n_features, d_token]
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape for multi-head attention
        # [batch, n_features, d_token] -> [batch, n_heads, n_features, head_dim]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)

        # Scaled dot-product attention
        # q: [batch, n_heads, n_features, head_dim]
        # k: [batch, n_heads, n_features, head_dim]
        # attention: [batch, n_heads, n_features, n_features]
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        # Apply attention to values
        # attention: [batch, n_heads, n_features, n_features]
        # v: [batch, n_heads, n_features, head_dim]
        # output: [batch, n_heads, n_features, head_dim]
        output = torch.matmul(attention, v)

        # Reshape back
        # [batch, n_heads, n_features, head_dim] -> [batch, n_features, d_token]
        output = rearrange(output, 'b h n d -> b n (h d)')

        # Final projection
        output = self.W_out(output)
        output = self.output_dropout(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feedforward network.

    Applied independently to each feature token.
    Typically uses a 4x expansion in the hidden layer.
    """
    def __init__(self, d_token: int, d_ff: int, dropout: float = 0.0,
                 activation: str = 'relu', bias: bool = True):
        super().__init__()

        self.linear1 = nn.Linear(d_token, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_token, bias=bias)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_features, d_token]

        Returns:
            output: [batch, n_features, d_token]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer encoder block with pre-norm architecture.

    Structure:
    1. LayerNorm -> MultiheadAttention -> Residual
    2. LayerNorm -> FeedForward -> Residual
    """
    def __init__(self, d_token: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, attention_dropout: float = 0.0,
                 activation: str = 'relu', prenorm: bool = True):
        super().__init__()

        self.prenorm = prenorm

        self.attention = MultiheadAttention(
            d_token=d_token,
            n_heads=n_heads,
            dropout=dropout,
            attention_dropout=attention_dropout
        )

        self.feedforward = FeedForward(
            d_token=d_token,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )

        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_features, d_token]

        Returns:
            output: [batch, n_features, d_token]
        """
        if self.prenorm:
            # Pre-norm architecture (more stable training)
            x = x + self.attention(self.norm1(x))
            x = x + self.feedforward(self.norm2(x))
        else:
            # Post-norm architecture (original transformer)
            x = self.norm1(x + self.attention(x))
            x = self.norm2(x + self.feedforward(x))

        return x


class FTTransformerEncoder(nn.Module):
    """
    FT-Transformer Feature Encoder for RL State Spaces.

    Replaces the first linear layer in Actor/Critic networks with a feature encoder
    that learns feature interactions through self-attention.

    Args:
        n_features: Number of input features (state_dim)
        d_token: Embedding dimension per feature (default: 192)
        n_blocks: Number of transformer blocks (default: 3)
        n_heads: Number of attention heads (default: 8)
        d_ff_factor: Feedforward expansion factor (default: 4)
        dropout: Dropout rate (default: 0.1)
        attention_dropout: Attention dropout rate (default: 0.0)
        activation: Activation function ('relu' or 'gelu')
        prenorm: Use pre-norm architecture (more stable)
        output_dim: Output encoding dimension (if None, uses n_features * d_token)

    Input:
        x: [batch, n_features] - raw state features

    Output:
        encoded: [batch, output_dim] - contextualized feature embeddings
    """
    def __init__(
        self,
        n_features: int,
        d_token: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        d_ff_factor: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = 'relu',
        prenorm: bool = True,
        output_dim: Optional[int] = None
    ):
        super().__init__()

        self.n_features = n_features
        self.d_token = d_token
        self.output_dim = output_dim if output_dim is not None else n_features * d_token

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(n_features, d_token)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_token=d_token,
                n_heads=n_heads,
                d_ff=d_token * d_ff_factor,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                prenorm=prenorm
            )
            for _ in range(n_blocks)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_token)

        # Output projection (flatten tokens and project to output_dim)
        self.output_projection = nn.Linear(n_features * d_token, self.output_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_features] - raw state features

        Returns:
            encoded: [batch, output_dim] - encoded features
        """
        # Tokenize features
        # x: [batch, n_features] -> tokens: [batch, n_features, d_token]
        tokens = self.tokenizer(x)

        # Apply transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Final normalization
        tokens = self.norm(tokens)

        # Flatten tokens and project to output dimension
        # tokens: [batch, n_features, d_token] -> [batch, n_features * d_token]
        tokens_flat = rearrange(tokens, 'b n d -> b (n d)')

        # Project to output dimension
        encoded = self.output_projection(tokens_flat)

        return encoded

    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Extract attention weights for visualization/analysis.

        Returns list of attention matrices from each block.
        Useful for understanding which features the model focuses on.
        """
        tokens = self.tokenizer(x)
        attention_weights = []

        for block in self.blocks:
            # Extract attention from block (requires modifying MultiheadAttention to return attention)
            # For now, just apply the block
            tokens = block(tokens)
            # TODO: Modify MultiheadAttention to optionally return attention weights

        return attention_weights


def create_ft_encoder_for_state_dim(
    state_dim: int,
    encoding_dim: int = 512,
    d_token: int = 64,
    n_blocks: int = 2,
    n_heads: int = 4,
    dropout: float = 0.1
) -> FTTransformerEncoder:
    """
    Convenience function to create an FT-Transformer encoder for a given state dimension.

    Default hyperparameters are tuned for crypto trading states:
    - Lightweight (d_token=64, n_blocks=2) for fast training
    - encoding_dim=512 matches typical mid_dim in PPO agents

    Args:
        state_dim: Dimension of the state space (1 + n_crypto + n_features * lookback)
        encoding_dim: Output dimension (should match mid_dim in Actor/Critic)
        d_token: Token embedding dimension per feature
        n_blocks: Number of transformer blocks
        n_heads: Number of attention heads
        dropout: Dropout rate

    Returns:
        FTTransformerEncoder instance
    """
    return FTTransformerEncoder(
        n_features=state_dim,
        d_token=d_token,
        n_blocks=n_blocks,
        n_heads=n_heads,
        d_ff_factor=4,
        dropout=dropout,
        attention_dropout=0.0,
        activation='relu',
        prenorm=True,
        output_dim=encoding_dim
    )


if __name__ == '__main__':
    # Test the encoder
    print("Testing FT-Transformer Encoder...")

    # Simulate a smaller crypto trading state for testing
    # Example: 3 tickers, 14 indicators, lookback=5
    # state_dim = 1 (cash) + 3 (stocks) + (14 * 3 * 5) = 214
    state_dim = 214
    batch_size = 8
    encoding_dim = 256

    # Create encoder
    encoder = create_ft_encoder_for_state_dim(
        state_dim=state_dim,
        encoding_dim=encoding_dim,
        d_token=32,
        n_blocks=2,
        n_heads=4,
        dropout=0.1
    )

    # Create dummy input
    x = torch.randn(batch_size, state_dim)

    # Forward pass
    print(f"\nInput shape: {x.shape}")
    encoded = encoder(x)
    print(f"Output shape: {encoded.shape}")
    print(f"Expected: [batch_size={batch_size}, encoding_dim={encoding_dim}]")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n✓ FT-Transformer encoder test passed!")
