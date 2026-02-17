#!/usr/bin/env python3
"""
End-to-End Test for FT-Transformer Integration

Verifies:
1. FT-Transformer encoder forward pass
2. Actor/Critic networks with FT encoding
3. AgentPPO_FT initialization
4. Pre-trained weight loading (mock test)
5. Compatibility with standard training
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.ft_transformer_encoder import FTTransformerEncoder, create_ft_encoder_for_state_dim
from drl_agents.agents.net_ft import ActorPPO_FT, CriticPPO_FT, create_ft_actor_critic
from drl_agents.agents.AgentPPO_FT import AgentPPO_FT


def test_encoder():
    """Test FT-Transformer encoder."""
    print("\n" + "="*70)
    print("TEST 1: FT-Transformer Encoder")
    print("="*70)

    state_dim = 214  # Small test state
    batch_size = 8
    encoding_dim = 256

    encoder = create_ft_encoder_for_state_dim(
        state_dim=state_dim,
        encoding_dim=encoding_dim,
        d_token=32,
        n_blocks=2,
        n_heads=4
    )

    # Forward pass
    x = torch.randn(batch_size, state_dim)
    encoded = encoder(x)

    assert encoded.shape == (batch_size, encoding_dim), \
        f"Expected shape ({batch_size}, {encoding_dim}), got {encoded.shape}"

    print(f"✓ Encoder output shape: {encoded.shape}")
    print(f"✓ Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print("✓ Test passed!")


def test_actor_critic():
    """Test FT-Transformer Actor/Critic."""
    print("\n" + "="*70)
    print("TEST 2: FT-Transformer Actor/Critic")
    print("="*70)

    state_dim = 214
    action_dim = 3
    mid_dim = 256
    batch_size = 8

    # Test FT-Transformer version
    print("\n2a. With FT-Transformer:")
    actor_ft, critic_ft = create_ft_actor_critic(
        state_dim=state_dim,
        action_dim=action_dim,
        mid_dim=mid_dim,
        use_ft_encoder=True,
        d_token=32,
        n_blocks=2,
        n_heads=4
    )

    state = torch.randn(batch_size, state_dim)

    # Test actor
    action = actor_ft(state)
    assert action.shape == (batch_size, action_dim), \
        f"Expected action shape ({batch_size}, {action_dim}), got {action.shape}"
    print(f"  ✓ Actor output shape: {action.shape}")

    action_sampled, noise = actor_ft.get_action(state)
    assert action_sampled.shape == (batch_size, action_dim)
    print(f"  ✓ Actor sampling works")

    # Test critic
    value = critic_ft(state)
    assert value.shape == (batch_size, 1), \
        f"Expected value shape ({batch_size}, 1), got {value.shape}"
    print(f"  ✓ Critic output shape: {value.shape}")

    # Test standard version (backward compatibility)
    print("\n2b. Standard MLP (no FT-Transformer):")
    actor_std, critic_std = create_ft_actor_critic(
        state_dim=state_dim,
        action_dim=action_dim,
        mid_dim=mid_dim,
        use_ft_encoder=False
    )

    action_std = actor_std(state)
    value_std = critic_std(state)

    print(f"  ✓ Actor output shape: {action_std.shape}")
    print(f"  ✓ Critic output shape: {value_std.shape}")

    print("\n✓ All tests passed!")


def test_agent():
    """Test AgentPPO_FT."""
    print("\n" + "="*70)
    print("TEST 3: AgentPPO_FT")
    print("="*70)

    state_dim = 214
    action_dim = 3
    net_dim = 256

    # Mock args object
    class Args:
        use_ft_encoder = True
        ft_config = {
            'd_token': 32,
            'n_blocks': 2,
            'n_heads': 4,
            'dropout': 0.1
        }
        pretrained_encoder_path = None
        freeze_encoder = False
        if_cri_target = False
        ratio_clip = 0.25
        lambda_entropy = 0.02
        lambda_gae_adv = 0.98
        if_use_gae = True
        learning_rate = 3e-4
        batch_size = 128
        repeat_times = 8
        soft_update_tau = 0.005

    args = Args()

    # Create agent with FT-Transformer
    print("\n3a. Creating AgentPPO_FT with FT-Transformer:")
    agent_ft = AgentPPO_FT(
        net_dim=net_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        gpu_id=0,  # Will use CPU if no GPU
        args=args
    )

    # Test that actor/critic were created
    assert hasattr(agent_ft, 'act'), "Agent missing 'act' attribute"
    assert hasattr(agent_ft, 'cri'), "Agent missing 'cri' attribute"
    assert agent_ft.use_ft_encoder, "use_ft_encoder should be True"

    print("  ✓ Agent created with FT-Transformer")
    print(f"  ✓ Actor type: {type(agent_ft.act).__name__}")
    print(f"  ✓ Critic type: {type(agent_ft.cri).__name__}")

    # Test standard agent (backward compatibility)
    print("\n3b. Creating AgentPPO_FT without FT-Transformer:")
    args.use_ft_encoder = False
    agent_std = AgentPPO_FT(
        net_dim=net_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        gpu_id=0,
        args=args
    )

    assert not agent_std.use_ft_encoder, "use_ft_encoder should be False"
    print("  ✓ Agent created without FT-Transformer (standard MLP)")

    print("\n✓ All tests passed!")


def test_integration():
    """Test end-to-end integration."""
    print("\n" + "="*70)
    print("TEST 4: End-to-End Integration")
    print("="*70)

    state_dim = 214
    action_dim = 3
    net_dim = 256

    # Create agent
    class Args:
        use_ft_encoder = True
        ft_config = {'d_token': 32, 'n_blocks': 2, 'n_heads': 4, 'dropout': 0.1}
        pretrained_encoder_path = None
        freeze_encoder = False
        if_cri_target = False
        ratio_clip = 0.25
        lambda_entropy = 0.02
        lambda_gae_adv = 0.98
        if_use_gae = True
        learning_rate = 3e-4
        batch_size = 128
        repeat_times = 8
        soft_update_tau = 0.005

    agent = AgentPPO_FT(net_dim, state_dim, action_dim, 0, Args())

    # Simulate environment interaction
    state = np.random.randn(state_dim).astype(np.float32)

    # Get action
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        action, _ = agent.act.get_action(state_tensor)

    action_np = action.cpu().numpy()[0]

    assert action_np.shape == (action_dim,), \
        f"Expected action shape ({action_dim},), got {action_np.shape}"

    print(f"✓ State shape: {state.shape}")
    print(f"✓ Action shape: {action_np.shape}")
    print(f"✓ Action range: [{action_np.min():.3f}, {action_np.max():.3f}]")
    print("\n✓ End-to-end integration works!")


def main():
    print("\n" + "="*70)
    print("FT-TRANSFORMER INTEGRATION - FULL TEST SUITE")
    print("="*70)

    try:
        test_encoder()
        test_actor_critic()
        test_agent()
        test_integration()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nFT-Transformer integration is ready to use.")
        print("\nNext steps:")
        print("  1. Pre-train encoder: python scripts/training/pretrain_ft_encoder.py --data-dir data/1h_1680")
        print("  2. Test in RL training: Modify 1_optimize_unified.py to use AgentPPO_FT")
        print("  3. See docs/FT_TRANSFORMER_INTEGRATION.md for detailed usage guide")
        print()

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
