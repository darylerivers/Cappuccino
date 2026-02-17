"""
Optimized config for faster Optuna trials - eliminates inter-trial delays

Key optimizations:
1. Skip directory cleanup between trials (saves 10-25s per trial)
2. Only clean once at study start
3. Reuse working directory across trials

Drop-in replacement for train.config.Arguments
"""

import os
import torch
import numpy as np
from copy import deepcopy
from pprint import pprint

"""config for agent"""


class Arguments:
    def __init__(self, agent, env=None, env_func=None, env_args=None):
        self.env = env  # the environment for training
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.update_attr(
            "env_num"
        )  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.update_attr("max_step")  # the max step of an episode
        self.env_name = self.update_attr(
            "env_name"
        )  # the env name. Be used to set 'cwd'.
        self.state_dim = self.update_attr(
            "state_dim"
        )  # vector dimension (feature number) of state
        self.action_dim = self.update_attr(
            "action_dim"
        )  # vector dimension (feature number) of action
        self.if_discrete = self.update_attr(
            "if_discrete"
        )  # discrete or continuous action space
        self.target_return = self.update_attr(
            "target_return"
        )  # target average episode return

        self.agent = agent  # DRL algorithm
        self.net_dim = 2**9  # wider default network for RTX 3070 class GPU
        self.layer_num = (
            3  # layer number of MLP (Multi-layer perception, `assert layer_num>=2`)
        )
        self.if_off_policy = (
            self.get_if_off_policy()
        )  # agent is on-policy or off-policy
        self.if_use_old_traj = False  # save old data to splice and get a complete trajectory (for vector env)
        if self.if_off_policy:  # off-policy
            self.max_memo = 2**21  # capacity of replay buffer
            self.target_step = (
                2**10
            )  # repeatedly update network to keep critic's loss small
            self.batch_size = (
                self.net_dim
            )  # num of transitions sampled from replay buffer.
            self.repeat_times = 2**0  # collect target_step, then update network
            self.if_use_per = (
                False  # use PER (Prioritized Experience Replay) for sparse reward
            )
        else:  # on-policy
            self.max_memo = 2**12  # capacity of replay buffer
            self.target_step = (
                self.max_memo
            )  # repeatedly update network to keep critic's loss small
            self.batch_size = (
                self.net_dim * 2
            )  # num of transitions sampled from replay buffer.
            self.repeat_times = 2**4  # collect target_step, then update network
            self.if_use_gae = False  # use PER: GAE (Generalized Advantage Estimation) for sparse reward

        """Arguments for training"""
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = (
            2**0
        )  # an approximate target reward usually be closed to 256
        self.learning_rate = 2**-12  # 2 ** -15 ~= 3e-5
        self.soft_update_tau = 2**-8  # 2 ** -8 ~= 5e-3

        """Arguments for device"""
        cpu_total = os.cpu_count() or 16
        self.thread_num = max(4, min(cpu_total, cpu_total - 2))
        self.worker_num = 2  # Limited to 2-3 for RAM stability
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        """Arguments for evaluate"""
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.if_over_write = False  # over write the best policy network (actor.pth)
        self.if_allow_break = (
            True  # allow break training when reach goal (early termination)
        )

        """Arguments for evaluate"""
        self.eval_gap = 2**7  # evaluate the agent per eval_gap seconds
        self.eval_times = 2**4  # number of times that get episode return

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(4, self.thread_num // 2))
            except RuntimeError:
                pass
        torch.set_default_dtype(torch.float32)

        """auto set"""
        if self.cwd is None:
            self.cwd = (
                f"./{self.env_name}_{self.agent.__name__[5:]}_{self.learner_gpus}"
            )

        """remove history - OPTIMIZED VERSION"""
        if self.if_remove is None:
            self.if_remove = bool(
                input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == "y"
            )

        if self.if_remove:
            import shutil
            import glob
            # OPTIMIZATION: Only remove directory once, but clean checkpoints every trial
            # This avoids architecture mismatch errors while keeping speed benefits
            cleanup_marker = os.path.join(self.cwd, '.cleanup_done')

            if not os.path.exists(cleanup_marker):
                # First time - full cleanup
                shutil.rmtree(self.cwd, ignore_errors=True)
                os.makedirs(self.cwd, exist_ok=True)

                # Create marker file
                with open(cleanup_marker, 'w') as f:
                    f.write('1')

                print(f"| Arguments Initial cleanup: {self.cwd}")
            else:
                # Already cleaned directory - just remove checkpoint files to avoid architecture conflicts
                # This is very fast (< 0.1s) compared to full directory removal (10-30s)
                os.makedirs(self.cwd, exist_ok=True)
                for checkpoint_pattern in ['*.pth', 'actor.pth', 'critic.pth', 'actor_critic.pth']:
                    for f in glob.glob(os.path.join(self.cwd, checkpoint_pattern)):
                        try:
                            os.remove(f)
                        except OSError:
                            pass
                print(f"| Arguments Reusing cwd (fast mode): {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")

        os.makedirs(self.cwd, exist_ok=True)

    def update_attr(self, attr: str):
        return getattr(self.env, attr) if self.env_args is None else self.env_args[attr]

    def get_if_off_policy(self):
        name = self.agent.__name__
        return all((name.find("PPO") == -1, name.find("A2C") == -1))  # if_off_policy

    def print(self):
        # prints out args in a neat, readable format
        pprint(vars(self))
