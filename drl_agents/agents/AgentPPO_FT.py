"""
PPO Agent with FT-Transformer Feature Encoding

Drop-in replacement for AgentPPO that optionally uses FT-Transformer
for feature encoding instead of standard MLPs.

Usage:
    # In training script, replace:
    # agent = AgentPPO
    # with:
    # agent = AgentPPO_FT

    # Then pass FT-Transformer config via args:
    # args.use_ft_encoder = True
    # args.ft_config = {...}
    # args.pretrained_encoder_path = 'path/to/encoder.pth'
"""

import torch
from drl_agents.agents.AgentPPO import AgentPPO
from drl_agents.agents.net_ft import ActorPPO_FT, CriticPPO_FT
from typing import Optional


class AgentPPO_FT(AgentPPO):
    """
    PPO agent with optional FT-Transformer feature encoding.

    Inherits all behavior from AgentPPO but uses enhanced Actor/Critic networks.
    """
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        # Check if FT-Transformer should be used
        self.use_ft_encoder = getattr(args, 'use_ft_encoder', False)
        self.ft_config = getattr(args, 'ft_config', None)
        self.pretrained_encoder_path = getattr(args, 'pretrained_encoder_path', None)
        self.freeze_encoder = getattr(args, 'freeze_encoder', False)

        # Override actor/critic classes
        if self.use_ft_encoder:
            self.act_class = ActorPPO_FT
            self.cri_class = CriticPPO_FT
            print(f"\n{'='*70}")
            print(f"Using FT-Transformer Feature Encoding")
            print(f"{'='*70}")
            if self.ft_config:
                print(f"Config: {self.ft_config}")
            if self.pretrained_encoder_path:
                print(f"Pre-trained encoder: {self.pretrained_encoder_path}")
                print(f"Freeze encoder: {self.freeze_encoder}")
            print(f"{'='*70}\n")

        # Initialize parent class (will create actor/critic with appropriate classes)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

        # Load pre-trained encoder if specified
        if self.use_ft_encoder and self.pretrained_encoder_path:
            print("Loading pre-trained encoder weights...")
            self.act.load_pretrained_encoder(
                self.pretrained_encoder_path,
                freeze=self.freeze_encoder
            )
            self.cri.load_pretrained_encoder(
                self.pretrained_encoder_path,
                freeze=self.freeze_encoder
            )
            if self.cri_target:
                self.cri_target.load_pretrained_encoder(
                    self.pretrained_encoder_path,
                    freeze=self.freeze_encoder
                )
            print("✓ Pre-trained encoder loaded successfully\n")

    def init_actor_critic(self, net_dim, state_dim, action_dim, gpu_id, args):
        """
        Initialize actor and critic networks.

        Overrides parent method to pass FT-Transformer config.
        """
        if self.use_ft_encoder:
            # Create FT-Transformer networks
            ft_config = self.ft_config or {
                'd_token': 64,
                'n_blocks': 2,
                'n_heads': 4,
                'dropout': 0.1
            }

            self.act = self.act_class(
                mid_dim=net_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                use_ft_encoder=True,
                ft_config=ft_config
            ).to(self.device)

            self.cri = self.cri_class(
                mid_dim=net_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                use_ft_encoder=True,
                ft_config=ft_config
            ).to(self.device)
        else:
            # Use standard networks from parent class
            self.act = self.act_class(net_dim, state_dim, action_dim).to(self.device)
            self.cri = self.cri_class(net_dim, state_dim, action_dim).to(self.device)

        # Create target critic if needed
        if self.if_cri_target:
            if self.use_ft_encoder:
                self.cri_target = self.cri_class(
                    mid_dim=net_dim,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    use_ft_encoder=True,
                    ft_config=ft_config
                ).to(self.device)
            else:
                self.cri_target = self.cri_class(net_dim, state_dim, action_dim).to(
                    self.device
                )
            self.cri_target.load_state_dict(self.cri.state_dict())
        else:
            self.cri_target = self.cri

        # Create optimizers
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate)


# Convenience function to check if FT-Transformer should be used
def should_use_ft_transformer(trial_params: dict) -> tuple:
    """
    Check if FT-Transformer should be used based on trial parameters.

    Args:
        trial_params: Optuna trial parameters dict

    Returns:
        (use_ft_encoder, ft_config, pretrained_path, freeze)
    """
    use_ft_encoder = trial_params.get('use_ft_encoder', False)

    if not use_ft_encoder:
        return False, None, None, False

    # Extract FT-Transformer config
    ft_config = {
        'd_token': trial_params.get('ft_d_token', 64),
        'n_blocks': trial_params.get('ft_n_blocks', 2),
        'n_heads': trial_params.get('ft_n_heads', 4),
        'dropout': trial_params.get('ft_dropout', 0.1)
    }

    pretrained_path = trial_params.get('pretrained_encoder_path', None)
    freeze = trial_params.get('freeze_encoder', False)

    return use_ft_encoder, ft_config, pretrained_path, freeze
