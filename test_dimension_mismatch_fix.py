#!/usr/bin/env python3
"""
Test State Dimension Mismatch Fix

Verifies that the training code can handle models with different lookback
values (and thus different state dimensions) without crashing.
"""

import sys
import torch
import tempfile
import os
from pathlib import Path

# Add parent directory to path
PARENT_DIR = Path(__file__).parent
sys.path.insert(0, str(PARENT_DIR))

from drl_agents.agents.AgentPPO import AgentPPO


def test_dimension_mismatch_handling():
    """Test that dimension mismatch is handled gracefully."""

    print("=" * 80)
    print("Testing State Dimension Mismatch Handling")
    print("=" * 80)
    print()

    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp(prefix="test_dim_mismatch_")
    print(f"Test directory: {temp_dir}")
    print()

    try:
        # Step 1: Create and save a model with state_dim=100
        print("Step 1: Creating model with state_dim=100...")

        class Args1:
            gamma = 0.99
            env_num = 1
            batch_size = 128
            repeat_times = 8
            reward_scale = 1.0
            lambda_gae_adv = 0.98
            if_use_old_traj = False
            soft_update_tau = 0.005
            if_act_target = False
            if_cri_target = False
            if_off_policy = False
            learning_rate = 3e-4
            ratio_clip = 0.2
            lambda_entropy = 0.02
            if_use_gae = True

        agent1 = AgentPPO(
            net_dim=256,
            state_dim=100,  # Lookback = 1
            action_dim=7,
            gpu_id=-1,
            args=Args1()
        )

        # Save the model
        agent1.save_or_load_agent(temp_dir, if_save=True)
        print(f"✓ Model saved to {temp_dir}")
        print(f"  Actor input dim: {agent1.act.net[0].in_features}")
        print()

        # Step 2: Try to load with DIFFERENT state_dim
        print("Step 2: Attempting to load into model with state_dim=988...")
        print("(This should fail gracefully with dimension mismatch)")
        print()

        class Args2:
            gamma = 0.99
            env_num = 1
            batch_size = 128
            repeat_times = 8
            reward_scale = 1.0
            lambda_gae_adv = 0.98
            if_use_old_traj = False
            soft_update_tau = 0.005
            if_act_target = False
            if_cri_target = False
            if_off_policy = False
            learning_rate = 3e-4
            ratio_clip = 0.2
            lambda_entropy = 0.02
            if_use_gae = True

        agent2 = AgentPPO(
            net_dim=256,
            state_dim=988,  # Different! Lookback = 10
            action_dim=7,
            gpu_id=-1,
            args=Args2()
        )

        print(f"  New agent actor input dim: {agent2.act.net[0].in_features}")
        print()

        # Try to load - this should raise ValueError
        try:
            agent2.save_or_load_agent(temp_dir, if_save=False)
            print("❌ FAILED: Loading should have raised ValueError!")
            return False
        except ValueError as e:
            if "State dimension mismatch" in str(e):
                print("✓ Correctly caught dimension mismatch!")
                print(f"  Error message: {str(e)[:100]}...")
                print()
            else:
                print(f"❌ FAILED: Wrong ValueError: {e}")
                return False

        # Step 3: Verify loading with SAME state_dim still works
        print("Step 3: Verifying loading with matching state_dim=100...")

        agent3 = AgentPPO(
            net_dim=256,
            state_dim=100,  # Same as saved model
            action_dim=7,
            gpu_id=-1,
            args=Args1()
        )

        # This should succeed
        try:
            agent3.save_or_load_agent(temp_dir, if_save=False)
            print("✓ Successfully loaded checkpoint with matching dimensions!")
            print()
        except Exception as e:
            print(f"❌ FAILED: Should have loaded successfully: {e}")
            return False

        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("Summary:")
        print("  ✓ Models with different state_dim raise ValueError")
        print("  ✓ Error message correctly identifies dimension mismatch")
        print("  ✓ Models with matching state_dim load successfully")
        print()
        print("The fix is working correctly! Training can now handle mixed lookback values.")
        print()

        return True

    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up test directory: {temp_dir}")


if __name__ == "__main__":
    success = test_dimension_mismatch_handling()
    sys.exit(0 if success else 1)
